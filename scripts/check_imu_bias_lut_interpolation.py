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
    yaw_rad = np.arctan2(-x_world[0], x_world[1])
    
    return (yaw_rad + offset + np.pi) % (2.0 * np.pi) - np.pi


def latlon_to_yaw(lat, lon, lat0, lon0, offset=0.0):
    lat, lon, lat0, lon0 = map(np.radians, [lat, lon, lat0, lon0])
    dlon = lon - lon0
    x = np.sin(dlon) * np.cos(lat)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    # Forward azimuth formula: arctan2(x, y) where North = 0, East = +pi/2
    yaw = np.arctan2(-x, y)
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


def build_lut_correction_function(imu_bearings_rad, ref_bearings_rad, n_bins=360):
    # Builds a continuous 1D Lookup Table (LUT) mapping function using binned median values.
    # Handles angle unwrapping to prevent boundary artifacts across -pi/pi.
    # Convert to degrees for easier binning
    imu_deg = np.degrees(imu_bearings_rad)
    ref_deg = np.degrees(ref_bearings_rad)

    # Calculate shortest angular error: ref - imu
    errors_deg = np.degrees(np.arctan2(np.sin(ref_bearings_rad - imu_bearings_rad), 
                                       np.cos(ref_bearings_rad - imu_bearings_rad)))

    # Create uniform bins across -180 to 180 degrees
    bin_edges = np.linspace(-180, 180, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    binned_errors = []
    for i in range(n_bins):
        mask = (imu_deg >= bin_edges[i]) & (imu_deg < bin_edges[i+1])
        if np.any(mask):
            binned_errors.append(np.median(errors_deg[mask]))
        else:
            binned_errors.append(np.nan)

    # Fill empty bins using linear interpolation across missing segments
    s_err = pd.Series(binned_errors)
    s_err = s_err.interpolate(method='linear').bfill().ffill()
    lut_errors_deg = s_err.to_numpy()

    def correct_imu(imu_rad):
        imu_d = np.degrees(imu_rad)
        # Interpolate error based on raw IMU bearing
        corr_error_d = np.interp(imu_d, bin_centers, lut_errors_deg, period=360.0)
        corrected_d = imu_d + corr_error_d
        # Wrap to [-180, 180]
        corrected_rad = np.radians(corrected_d)
        return (corrected_rad + np.pi) % (2.0 * np.pi) - np.pi

    return correct_imu


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
            name="Corrected IMU Bearing (LUT)",
            line=dict(color="blue", width=1.5),
        )
    )

    fig.update_layout(
        title=f"Bearing Comparison (GNSS vs Raw IMU vs LUT Corrected) - {folder_name}",
        xaxis_title="Sample Index",
        yaxis_title="Bearing (degrees)",
        hovermode="x unified",
        width=1100,
        height=600,
    )

    output_file = f"bearing_comparison_{folder_name}.html"
    fig.write_html(output_file)
    print(f"Saved bearing plot to {output_file}")


def main():
    dataset_path = Path("/media/mf/SATA4TB/autodriving/datasetx/ugm_baru")
    meta_path = dataset_path / "meta"
    file_list = os.listdir(meta_path)
    file_list.sort()

    meta = [
        yaml.safe_load(open(meta_path / meta_file, "r"))
        for meta_file in file_list
    ]
    raw_lats = [m["global_position_latlon"][0] for m in meta]
    raw_lons = [m["global_position_latlon"][1] for m in meta]
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

    # 1. Build the LUT correction function from thousands of points
    correct_imu_fn = build_lut_correction_function(imu_bearing, ref_bearing, n_bins=360)

    # 2. Apply LUT correction to raw IMU bearing data
    corrected_imu_bearing = np.array([correct_imu_fn(b) for b in imu_bearing])

    # Convert radians to degrees for visualization
    ref_deg = np.degrees(ref_bearing)
    raw_imu_deg = np.degrees(imu_bearing)
    corrected_imu_deg = np.degrees(corrected_imu_bearing)

    # 3. Plot bearings using Plotly
    plot_bearing_comparison(
        dataset_path.name,
        ref_deg,
        raw_imu_deg,
        corrected_imu_deg
    )


if __name__ == "__main__":
    main()