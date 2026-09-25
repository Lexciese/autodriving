import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

# projects into NWU system
def imu_to_yaw(q, offset=0.0):
    r = R.from_quat(q)
    # Project the sensor's X-axis (+X Forward) into world horizontal frame
    x_world = r.apply([1, 0, 0])
    
    # ENU Frame Compass Bearing: arctan2(East, North) -> arctan2(x_world[0], x_world[1])
    # North = 0, East = +pi/2 (+90 deg), West = -pi/2 (-90 deg)
    yaw_rad = np.arctan2(-x_world[0], x_world[1])
    
    return (yaw_rad + offset + np.pi) % (2.0 * np.pi) - np.pi

# Projects into NWU system
def latlon_to_yaw(lat, lon, lat0, lon0, offset=0.0):
    lat, lon, lat0, lon0 = map(np.radians, [lat, lon, lat0, lon0])
    dlon = lon - lon0
    x = np.sin(dlon) * np.cos(lat)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    # Forward azimuth formula: arctan2(x, y) where North = 0, East = +pi/2
    yaw = np.arctan2(-x, y)
    return ((yaw + offset) + np.pi) % (2.0 * np.pi) - np.pi

# Projects calibrated magnetometer readings into NWU system matching latlon_to_yaw frame
def mag_to_yaw(mx, my, offset=0.0):
    """
    mx: Forward (+X), my: Left (+Y) in body frame.
    Returns yaw in radians within range [-pi, pi].
    North = 0, West = +pi/2 (+90 deg), East = -pi/2 (-90 deg).
    """
    yaw = np.arctan2(-my, mx)
    return ((yaw + offset) + np.pi) % (2.0 * np.pi) - np.pi

def hampel_filter(data, window_size=3, n_sigmas=3.0):
    s = pd.Series(data)
    rolling_median = (
        s.rolling(window=window_size, min_periods=1, center=True).median()
    )
    rolling_mad = (
        (s - rolling_median)
        .abs()
        .rolling(window=window_size, min_periods=1, center=True)
        .median()
    )
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = (s - rolling_median).abs()
    outliers = difference > threshold
    s_filtered = s.copy()
    s_filtered[outliers] = rolling_median[outliers]
    return s_filtered.tolist()

# 2D Ellipse Calibration Functions (Halir & Flusser)
# Fits ax^2 + bxy + cy^2 + dx + ey + f = 0 using direct linear least squares.
def fit_ellipse(x, y):
    D1 = np.vstack([x**2, x * y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


# Converts Cartesian conic coefficients (a, b, c, d, e, f) to ellipse parameters:
# (x0, y0, ap, bp, phi) -> center (x0, y0), semi-major/minor axes (ap, bp), rotation (phi).
def cart_to_pol(coeffs):
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a * c
    if den > 0:
        raise ValueError("Coefficients do not represent an ellipse!")

    x0 = (c * d - b * f) / den
    y0 = (a * f - b * d) / den

    num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b**2)
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2.0 * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2

    return x0, y0, ap, bp, phi


# Calculates scaling matrix Q = R^T * S * R
def compute_calibration_matrix(ap, bp, phi):
    sx = bp / ap
    sy = 1.0
    cp = np.cos(phi)
    sp = np.sin(phi)

    R_mat = np.array([[cp, sp], [-sp, cp]])
    S_mat = np.array([[sx, 0.0], [0.0, sy]])

    Q = R_mat.T @ S_mat @ R_mat
    return Q


def main():
    dataset_path = Path("/media/mf/SATA4TB/autodriving/datasetx/ugm_baru_with_magnetometer")
    meta_path = dataset_path / "meta"
    file_list = sorted(os.listdir(meta_path))

    meta = [yaml.safe_load(open(meta_path / f, "r"))
            for f in tqdm(file_list, desc="Loading YAMLs")]

    file_names = np.array(file_list)
    raw_lats = [m["global_position_latlon"][0] for m in tqdm(meta)]
    raw_lons = [m["global_position_latlon"][1] for m in tqdm(meta)]
    raw_mags = np.array([m["magnetic_field"] for m in meta], dtype=float)
    filtered_lats = hampel_filter(raw_lats, window_size=5)
    filtered_lons = hampel_filter(raw_lons, window_size=5)

    prev_lat = filtered_lats[0]
    prev_lon = filtered_lons[0]
    ref_bearing = []
    imu_bearing = []
    valid_indices = []

    for i in tqdm(range(len(file_list)), desc="Extracting Bearings"):
        curr_meta = meta[i]
        curr_imu = curr_meta["global_orientation_xyzw"]
        velocity = np.abs(curr_meta["velocity"])
        curr_lat = filtered_lats[i]
        curr_lon = filtered_lons[i]

        dLat_m = (curr_lat - prev_lat) * 40008000 / 360
        dLon_m = (curr_lon - prev_lon) * 40075000 * np.cos(np.radians(curr_lat)) / 360

        # Extract bearing only when vehicle moves >= 1.0m and velocity > 2.0 m/s
        if np.sqrt(dLat_m**2 + dLon_m**2) >= 1.0 and velocity > 2.0:
            latlon_yaw = latlon_to_yaw(curr_lat, curr_lon, prev_lat, prev_lon)
            imu_yaw = imu_to_yaw(curr_imu)

            ref_bearing.append(latlon_yaw)
            imu_bearing.append(imu_yaw)
            valid_indices.append(i)

            prev_lat = curr_lat
            prev_lon = curr_lon

    imu_bearing = np.array(imu_bearing)
    ref_bearing = np.array(ref_bearing)

    # Scale to microteslas (µT)
    scale = 1e6 if np.max(np.abs(raw_mags)) < 1.0 else 1.0
    mags_uT = raw_mags * scale

    # Reorder channels according to ROS FLU standard (x-Forward, y-Left, z-Up)
    x_body = mags_uT[:, 0]  # Forward (+X)
    y_body = mags_uT[:, 2]  # Left (+Y)
    z_body = mags_uT[:, 1]  # Up (+Z)

    # 2D Ellipse Calibration on XY Plane
    coeffs = fit_ellipse(x_body, y_body)
    x0, y0, ap, bp, phi = cart_to_pol(coeffs)
    Q = compute_calibration_matrix(ap, bp, phi)
    print(f"\n--- Calibration Results ---")
    print(f"Hard-Iron Offset (X0, Y0): ({x0:.2f}, {y0:.2f}) µT")
    print(f"Semi-major (a): {ap:.2f}, Semi-minor (b): {bp:.2f}")
    print(f"Ellipse Angle (phi): {np.degrees(phi):.2f}°")
    print(f"Correction Matrix Q:\n{Q}\n")
    calibration_file = dict()
    calibration_file["offset"] = [float(x0), float(y0)]
    calibration_file["Q"] = Q.tolist()
    calibration_file["semi_major_minor"] = [float(ap), float(bp)]
    calibration_file["ellipse_angle_rad"] = float(phi)
    with open('magnetometer_calib.yaml', 'w') as file:
        yaml.safe_dump(calibration_file, file)

    # Apply Offset and Soft-Iron Q Matrix
    mags_offset = np.column_stack([x_body - x0, y_body - y0])
    mags_calibrated_xy = mags_offset @ Q.T
    x_cal = mags_calibrated_xy[:, 0]
    y_cal = mags_calibrated_xy[:, 1]
    z_cal = z_body - np.mean(z_body)  # Center Z at 0 for side-by-side level comparison

    # Compute Calibrated Magnetometer Yaw & Yaw Error relative to latlon_to_yaw
    valid_indices = np.array(valid_indices)
    mag_bearing = mag_to_yaw(x_cal[valid_indices], y_cal[valid_indices])

    # Calculate angular error wrapped to [-180, 180] degrees
    yaw_error_rad = np.arctan2(np.sin(mag_bearing - ref_bearing), np.cos(mag_bearing - ref_bearing))
    yaw_error_deg = np.degrees(yaw_error_rad)
    ref_bearing_deg = np.degrees(ref_bearing)

    # Plot and Save Yaw Error Graph
    plt.figure(figsize=(10, 6))
    plt.scatter(ref_bearing_deg, yaw_error_deg, color='blue', alpha=0.6, edgecolors='none', s=18)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
    plt.xlim(-180, 180)
    plt.xlabel("Reference Yaw / GNSS LatLon Yaw (degrees)", fontsize=11)
    plt.ylabel("Yaw Error [Mag - Ref] (degrees)", fontsize=11)
    plt.title("Calibrated Magnetometer Yaw Error vs. Reference Yaw", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("magnetometer_yaw_error.png", dpi=300)
    plt.close()

    # Parametric Overlay Curves
    theta = np.linspace(0, 2 * np.pi, 200)
    # Parametric Raw Fitted Ellipse
    ellipse_x = x0 + ap * np.cos(theta) * np.cos(phi) - bp * np.sin(theta) * np.sin(phi)
    ellipse_y = y0 + ap * np.cos(theta) * np.sin(phi) + bp * np.sin(theta) * np.cos(phi)
    ellipse_z = np.full_like(theta, np.mean(z_body))
    # Parametric Calibrated Ideal Circle
    circle_x = bp * np.cos(theta)
    circle_y = bp * np.sin(theta)
    circle_z = np.zeros_like(theta)

    fig = go.Figure()
    # Uncalibrated 3D Data Scatter
    fig.add_trace(go.Scatter3d(
        x=x_body, y=y_body, z=z_body,
        mode='markers',
        marker=dict(size=2, color='royalblue', opacity=0.4),
        name='Uncalibrated Raw Points',
        customdata=file_names,
        hovertemplate="<b>[Uncalibrated]</b><br>File: %{customdata}<br>X: %{x:.2f} µT<br>Y: %{y:.2f} µT<br>Z: %{z:.2f} µT<extra></extra>"
    ))

    # Fitted Raw Ellipse Surface Ring
    fig.add_trace(go.Scatter3d(
        x=ellipse_x, y=ellipse_y, z=ellipse_z,
        mode='lines',
        line=dict(color='red', width=5),
        name='Fitted Raw Ellipse'
    ))
    # Hard-Iron Center Point
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[np.mean(z_body)],
        mode='markers+text',
        marker=dict(size=6, color='darkred'),
        text=[f"Hard-Iron Center<br>({x0:.1f}, {y0:.1f})"],
        textposition="top center",
        name='Hard-Iron Center'
    ))
    # Calibrated 3D Data Scatter
    fig.add_trace(go.Scatter3d(
        x=x_cal, y=y_cal, z=z_cal,
        mode='markers',
        marker=dict(size=2, color='limegreen', opacity=0.4),
        name='Calibrated Points',
        customdata=file_names,
        hovertemplate="<b>[Calibrated]</b><br>File: %{customdata}<br>X: %{x:.2f} µT<br>Y: %{y:.2f} µT<br>Z: %{z:.2f} µT<extra></extra>"
    ))
    # Ideal Calibrated Circle Ring
    fig.add_trace(go.Scatter3d(
        x=circle_x, y=circle_y, z=circle_z,
        mode='lines',
        line=dict(color='cyan', width=5),
        name='Ideal Calibrated Circle'
    ))
    # Calibrated Center Point (0,0,0)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=6, color='darkgreen'),
        text=["Origin (0,0,0)"],
        textposition="top center",
        name='Calibrated Center'
    ))
    # Layout Configuration
    fig.update_layout(
        title="3D Overlay: Uncalibrated vs. Calibrated Magnetometer Data",
        scene=dict(
            xaxis_title="Vehicle X / Forward (µT)",
            yaxis_title="Vehicle Y / Lateral (µT)",
            zaxis_title="Vehicle Z / Vertical (µT)",
            aspectmode='data'
        ),
        legend=dict(x=0.02, y=0.98),
        width=1000,
        height=800
    )

    fig.write_html("magnetometer_overlay_3d.html")

if __name__ == "__main__":
    main()