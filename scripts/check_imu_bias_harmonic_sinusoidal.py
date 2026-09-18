import os
import yaml
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import numpy as np
from pathlib import Path
import plotly.graph_objects as go


def imu_to_yaw(q, offset=0.0):
    r = R.from_quat(q)
    # Project the sensor's X-axis (+X Forward) into world horizontal frame
    x_world = r.apply([1, 0, 0])
    
    # ENU Frame Compass Bearing: arctan2(East, North) -> arctan2(x_world[0], x_world[1])
    # North = 0, East = +pi/2 (+90 deg), West = -pi/2 (-90 deg)
    yaw_rad = np.arctan2(x_world[0], x_world[1])
    
    return (yaw_rad + offset + np.pi) % (2.0 * np.pi) - np.pi


def latlon_to_yaw(lat, lon, lat0, lon0, offset=0.0):
    lat, lon, lat0, lon0 = map(np.radians, [lat, lon, lat0, lon0])
    dlon = lon - lon0
    x = np.sin(dlon) * np.cos(lat)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    # Forward azimuth formula: arctan2(x, y) where North = 0, East = +pi/2
    yaw = np.arctan2(x, y)
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


def build_harmonic_correction_function(imu_bearings_rad, ref_bearings_rad, n_harmonics=2):
    # Fits a harmonic sinusoidal model (Fourier series) to the orientation error:
    # err(theta) = A_0 + sum_{k=1}^N [ A_k * cos(k * theta) + B_k * sin(k * theta) ]

    # Calculate shortest angular error: ref - imu wrapped to [-pi, pi]
    errors_rad = np.arctan2(
        np.sin(ref_bearings_rad - imu_bearings_rad),
        np.cos(ref_bearings_rad - imu_bearings_rad)
    )

    # Construct linear design matrix (basis expansion)
    # [1, cos(theta), sin(theta), cos(2*theta), sin(2*theta), ...]
    A = [np.ones_like(imu_bearings_rad)]
    for k in range(1, n_harmonics + 1):
        A.append(np.cos(k * imu_bearings_rad))
        A.append(np.sin(k * imu_bearings_rad))
    
    A = np.column_stack(A)

    # Solve linear least squares: A * coeffs = errors_rad
    coeffs, _, _, _ = np.linalg.lstsq(A, errors_rad, rcond=None)

    def correct_imu(imu_rad):
        imu_arr = np.atleast_1d(imu_rad)
        
        # Build design matrix for test input
        A_test = [np.ones_like(imu_arr)]
        for k in range(1, n_harmonics + 1):
            A_test.append(np.cos(k * imu_arr))
            A_test.append(np.sin(k * imu_arr))
        
        A_test = np.column_stack(A_test)
        
        # Predict angular error and apply correction
        predicted_error_rad = A_test @ coeffs
        corrected_rad = imu_arr + predicted_error_rad
        
        # Wrap output back to [-pi, pi]
        wrapped_rad = (corrected_rad + np.pi) % (2.0 * np.pi) - np.pi
        return wrapped_rad if np.ndim(imu_rad) > 0 else wrapped_rad[0]

    return correct_imu, coeffs


def plot_bearing_comparison(folder_name, ref_deg, raw_imu_deg, corrected_imu_deg):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=ref_deg,
            mode="lines",
            name="GNSS Reference Bearing",
            line=dict(color="green", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            y=raw_imu_deg,
            mode="lines",
            name="Raw IMU Bearing",
            line=dict(color="red", width=1, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            y=corrected_imu_deg,
            mode="lines",
            name="Corrected IMU Bearing (Harmonic)",
            line=dict(color="blue", width=1.5),
        )
    )

    fig.update_layout(
        title=f"Bearing Comparison (GNSS vs Raw IMU vs Harmonic Corrected) - {folder_name}",
        xaxis_title="Sample Index",
        yaxis_title="Bearing (degrees)",
        hovermode="x unified",
        width=1100,
        height=600,
    )

    output_file = f"bearing_comparison_{folder_name}.html"
    fig.write_html(output_file)
    print(f"Saved bearing plot to {output_file}")

def plot_harmonic_correction_curve(
    folder_name, imu_bearings_rad, ref_bearings_rad, correct_imu_fn
):
    raw_errors_deg = np.degrees(
        np.arctan2(
            np.sin(ref_bearings_rad - imu_bearings_rad),
            np.cos(ref_bearings_rad - imu_bearings_rad),
        )
    )
    imu_deg = np.degrees(imu_bearings_rad)

    theta_eval_deg = np.linspace(-180, 180, 1000)
    theta_eval_rad = np.radians(theta_eval_deg)

    corrected_eval_rad = correct_imu_fn(theta_eval_rad)
    predicted_error_deg = np.degrees(
        np.arctan2(
            np.sin(corrected_eval_rad - theta_eval_rad),
            np.cos(corrected_eval_rad - theta_eval_rad),
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=imu_deg,
            y=raw_errors_deg,
            mode="markers",
            name="Raw IMU Error Samples",
            marker=dict(color="rgba(200, 50, 50, 0.3)", size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=theta_eval_deg,
            y=predicted_error_deg,
            mode="lines",
            name="Harmonic Correction Model (Fourier Fit)",
            line=dict(color="blue", width=3),
        )
    )

    fig.update_layout(
        title=f"IMU Yaw Harmonic Correction Function - {folder_name}",
        xaxis_title="Raw IMU Bearing Angle (degrees)",
        yaxis_title="Bearing Error Correction Offset (degrees)",
        hovermode="x unified",
        width=1000,
        height=500,
    )

    output_file = f"harmonic_correction_curve_{folder_name}.html"
    fig.write_html(output_file)
    print(f"Saved correction curve plot to {output_file}")

def main():
    dataset_path = Path("/media/mf/SATA4TB/autodriving/datasetx/ugm_baru")
    meta_path = dataset_path / "meta"
    file_list = os.listdir(meta_path)
    file_list.sort()

    meta = [
        yaml.safe_load(open(meta_path / meta_file, "r"))
        for meta_file in tqdm(file_list)
    ]
    raw_lats = [m["global_position_latlon"][0] for m in tqdm(meta)]
    raw_lons = [m["global_position_latlon"][1] for m in tqdm(meta)]
    filtered_lats = hampel_filter(raw_lats, window_size=5)
    filtered_lons = hampel_filter(raw_lons, window_size=5)

    prev_lat = filtered_lats[0]
    prev_lon = filtered_lons[0]
    ref_bearing = []
    imu_bearing = []

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

            prev_lat = curr_lat
            prev_lon = curr_lon

    imu_bearing = np.array(imu_bearing)
    ref_bearing = np.array(ref_bearing)

    correct_imu_fn, coeff = build_harmonic_correction_function(imu_bearing, ref_bearing, n_harmonics=2)
    print(f"harmonic sinusoidal coeffs: {coeff}")
    with open("imu_harmonic_coeffs.yml", "w") as c:
        yaml.dump(coeff.astype(float).tolist(), c)

    corrected_imu_bearing = correct_imu_fn(imu_bearing)

    ref_deg = np.degrees(ref_bearing)
    raw_imu_deg = np.degrees(imu_bearing)
    corrected_imu_deg = np.degrees(corrected_imu_bearing)

    plot_bearing_comparison(
        dataset_path.name,
        ref_deg,
        raw_imu_deg,
        corrected_imu_deg
    )

    plot_harmonic_correction_curve(
        dataset_path.name, imu_bearing, ref_bearing, correct_imu_fn
    )


if __name__ == "__main__":
    main()