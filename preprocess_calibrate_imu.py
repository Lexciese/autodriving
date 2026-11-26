from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from config import GlobalConfig

config = GlobalConfig()
MIN_SPEED_KMH = 1.0
MAX_YAW_RATE_DEG_S = 3.0
STRAIGHT_WINDOW = 5
MAX_WINDOW_SPAN_DEG = 4.0
DEFAULT_DT_S = getattr(config, "gnss_dt", 1.0)
SECTOR_NAMES = [
    "0 to 60",
    "60 to 120",
    "120 to 180",
    "-180 to -120",
    "-120 to -60",
    "-60 to 0",
]

def wrap_rad(rad):
    """Wrap angle in radians to [-pi, pi)."""
    return (rad + np.pi) % (2 * np.pi) - np.pi

def wrap_deg(deg):
    """Wrap angle in degrees to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0

def circular_mean_deg(angles_deg):
    """Compute circular mean of angles in degrees."""
    rad = np.radians(angles_deg)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)))

def compute_rmse(errors):
    """Compute Root Mean Square Error (RMSE) of an array."""
    if len(errors) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(errors))))

def quat_to_yaw(q):
    """Convert quaternion to heading yaw in radians."""
    x, y, z, w = q
    numerator = -(1.0 - 2.0 * (y * y + z * z))
    denominator = 2.0 * (w * z + x * y)
    return wrap_rad(np.arctan2(numerator, denominator))

def load_route_data(subfolder, route):
    """
    Load GNSS CSV and match rows with IMU metadata YAML files by timestamp.
    Returns (lats, lons, imu_yaws_rad, timestamps) or None if insufficient data.
    """
    gnss_file = subfolder / f"{route}_gnss.csv"
    meta_dir = subfolder / "meta"
    if not meta_dir.exists() or not gnss_file.exists():
        return None
    gnss_df = pd.read_csv(gnss_file)
    gnss_df["ts_key"] = gnss_df["timestamp"].astype(str).str.replace(".yml", "", regex=False)
    time_col = next((c for c in ["time", "timestamp_s", "sec", "epoch"] if c in gnss_df.columns), None)
    gnss_lookup = gnss_df.set_index("ts_key").to_dict("index")
    matched_rows = []
    for idx, yml_file in enumerate(sorted(meta_dir.glob("*.yml"))):
        gnss_row = gnss_lookup.get(yml_file.stem)
        if not gnss_row:
            continue
        meta_data = yaml.safe_load(yml_file.read_text())
        if not meta_data or meta_data.get("global_orientation_xyzw") is None:
            continue
        if time_col and pd.notna(gnss_row[time_col]):
            timestamp = gnss_row[time_col]
        elif isinstance(meta_data.get("timestamp"), (int, float)):
            timestamp = meta_data["timestamp"]
        else:
            timestamp = idx * DEFAULT_DT_S
        yaw_rad = quat_to_yaw(meta_data["global_orientation_xyzw"])
        matched_rows.append((gnss_row["latitude"], gnss_row["longitude"], yaw_rad, float(timestamp)))
    if len(matched_rows) < 2:
        print(f"[WARNING] Insufficient matched frames for {route}.")
        return None
    lats, lons, imu_yaws, ts = map(np.array, zip(*matched_rows))
    return lats, lons, imu_yaws, ts


def compute_gnss_kinematics(lat, lon):
    earth_radius_m = 6371000.0
    r_lat, r_lon = np.radians(lat), np.radians(lon)
    dlat, dlon = np.diff(r_lat), np.diff(r_lon)
    # Haversine distance between consecutive points
    a = np.sin(dlat / 2) ** 2 + np.cos(r_lat[:-1]) * np.cos(r_lat[1:]) * np.sin(dlon / 2) ** 2
    dist_m = 2.0 * earth_radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    x = np.sin(dlon) * np.cos(r_lat[1:])
    y = np.cos(r_lat[:-1]) * np.sin(r_lat[1:]) - np.sin(r_lat[:-1]) * np.cos(r_lat[1:]) * np.cos(dlon)
    valid_yaw = np.where(dist_m >= 0.05, wrap_rad(np.arctan2(-x, y)), np.nan)
    gnss_yaw_rad = np.append(valid_yaw, np.nan)
    return dist_m, gnss_yaw_rad


def detect_straight_segments(dist_m, imu_yaws, gnss_yaws, ts):
    n_frames = len(imu_yaws)
    dt = np.diff(ts)
    valid_dt = dt > 0
    safe_dt = np.where(valid_dt, dt, 1.0)
    speed_kmh = np.where(valid_dt, (dist_m / safe_dt) * 3.6, 0.0)
    imu_rate = np.where(valid_dt, np.abs(np.degrees(wrap_rad(np.diff(imu_yaws)))) / safe_dt, np.inf)
    gnss_diff_deg = np.degrees(wrap_rad(np.diff(gnss_yaws)))
    gnss_rate = np.where(valid_dt & ~np.isnan(gnss_diff_deg), np.abs(gnss_diff_deg) / safe_dt, 0.0)
    step_is_straight = np.append(
        valid_dt
        & (speed_kmh > MIN_SPEED_KMH)
        & ~np.isnan(gnss_yaws[:-1])
        & (imu_rate <= MAX_YAW_RATE_DEG_S)
        & (gnss_rate <= MAX_YAW_RATE_DEG_S),
        False,
    )
    straight_mask = np.zeros(n_frames, dtype=bool)
    half_win = STRAIGHT_WINDOW // 2
    for i in range(half_win, n_frames - half_win):
        win = slice(i - half_win, i + half_win + 1)
        if np.all(step_is_straight[win]):
            imu_span = np.max(np.abs(np.degrees(wrap_rad(imu_yaws[win] - imu_yaws[i]))))
            gnss_span = np.max(np.abs(np.degrees(wrap_rad(gnss_yaws[win] - gnss_yaws[i]))))
            if imu_span <= MAX_WINDOW_SPAN_DEG and gnss_span <= MAX_WINDOW_SPAN_DEG:
                straight_mask[i] = True
    return straight_mask

def assign_sectors(angles_deg):
    """Map angles in [-180, 180] into 6 sectors (0 to 5, each 60 deg wide)."""
    sector_ids = np.floor(angles_deg / 60.0).astype(int) % 6
    sector_ids[angles_deg == 180.0] = 2
    return sector_ids

def calibrate_sector_bias(imu_yaws, gnss_yaws, straight_mask):
    raw_deg = np.degrees(imu_yaws)
    sector_ids = assign_sectors(raw_deg)
    error_samples = np.degrees(wrap_rad(gnss_yaws[straight_mask] - imu_yaws[straight_mask]))
    global_bias = circular_mean_deg(error_samples)
    masked_sectors = sector_ids[straight_mask]
    sector_biases = []
    sample_counts = []
    for s in range(6):
        in_sector = masked_sectors == s
        count = int(np.sum(in_sector))
        sample_counts.append(count)
        bias_val = circular_mean_deg(error_samples[in_sector]) if count > 0 else global_bias
        sector_biases.append(round(bias_val, 4))

    return sector_biases, sample_counts, sector_ids

def apply_bearing_bias(angles_deg, bias_list):
    boundary_mid = (bias_list[2] + bias_list[3]) / 2.0
    xp = [-180, -170, -130, -110, -70, -50, 0, 50, 70, 110, 130, 170, 180]
    fp = [
        boundary_mid,
        bias_list[3], bias_list[3],
        bias_list[4], bias_list[4],
        bias_list[5],
        bias_list[0], bias_list[0],
        bias_list[1], bias_list[1],
        bias_list[2], bias_list[2],
        boundary_mid,
    ]
    corrected = wrap_deg(np.asarray(angles_deg) + np.interp(angles_deg, xp, fp))
    return float(corrected) if np.isscalar(angles_deg) else corrected

def print_and_save_summary(subfolder, route, total_frames, straight_mask, bias, counts, raw_err, corr_err):
    """Print calibration metrics to console and write sector bias YAML file."""
    print(f"\n[{route}] Total frames: {total_frames} | Straight > {MIN_SPEED_KMH} km/h: {straight_mask.sum()}")
    print(f"  bearing_bias (deg): {bias}")

    for name, b_val, cnt in zip(SECTOR_NAMES, bias, counts):
        fallback_note = " (fallback: global mean)" if cnt == 0 else ""
        print(f"    Sector [{name:>11} deg]: {b_val:+7.2f}°  (n={cnt}){fallback_note}")

    print(
        f"  Before -> Bias: {circular_mean_deg(raw_err):+.2f}° | "
        f"MAE: {np.mean(np.abs(raw_err)):.2f}° | RMSE: {compute_rmse(raw_err):.2f}°"
    )
    print(
        f"  After  -> Bias: {circular_mean_deg(corr_err):+.2f}° | "
        f"MAE: {np.mean(np.abs(corr_err)):.2f}° | RMSE: {compute_rmse(corr_err):.2f}°"
    )

    output_yaml = subfolder / f"{route}_imu_sector_bias.yml"
    payload = {"bearing_bias_deg": dict(zip(SECTOR_NAMES, bias))}
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False))


def plot_error_analysis(route, frames, straight_mask, sector_ids, bias, counts, raw_err, corr_err):
    """Save Plot 1: Frame-by-frame heading residuals and per-sector RMSE bars."""
    fig, (ax_res, ax_bar) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1.3, 1]})

    ax_res.scatter(
        frames[straight_mask], raw_err, c="darkorange", s=10, alpha=0.55,
        label=f"Raw IMU Error (Bias={circular_mean_deg(raw_err):+.2f}°, MAE={np.mean(np.abs(raw_err)):.2f}°, RMSE={compute_rmse(raw_err):.2f}°)"
    )
    ax_res.scatter(
        frames[straight_mask], corr_err, c="teal", s=10, alpha=0.8,
        label=f"Corrected IMU Error (Bias={circular_mean_deg(corr_err):+.2f}°, MAE={np.mean(np.abs(corr_err)):.2f}°, RMSE={compute_rmse(corr_err):.2f}°)"
    )
    ax_res.axhline(0, c="k", ls=":", lw=1.1)
    ax_res.set(
        title=f"Heading Residual & RMSE Comparison (Straight > {MIN_SPEED_KMH} km/h): {route}",
        xlabel="Frame Index",
        ylabel="Wrapped Error (IMU - GNSS, deg)",
        ylim=(-185, 185),
    )
    ax_res.grid(True, ls="--", alpha=0.5)
    ax_res.legend()

    x = np.arange(6)
    bar_width = 0.28
    masked_sectors = sector_ids[straight_mask]
    raw_sector_rmse = [compute_rmse(raw_err[masked_sectors == s]) for s in range(6)]
    corr_sector_rmse = [compute_rmse(corr_err[masked_sectors == s]) for s in range(6)]

    ax_bar.bar(x - bar_width, raw_sector_rmse, bar_width, color="darkorange", alpha=0.8, label="Raw Sector RMSE (deg)")
    ax_bar.bar(x, corr_sector_rmse, bar_width, color="teal", alpha=0.85, label="Corrected Sector RMSE (deg)")
    ax_bar.bar(x + bar_width, bias, bar_width, color="slateblue", alpha=0.8, label="Applied Sector Bias (deg)")
    ax_bar.axhline(0, c="k", ls=":", lw=1.0)
    ax_bar.set(
        xticks=x,
        xticklabels=[f"{name}\n(n={cnt})" for name, cnt in zip(SECTOR_NAMES, counts)],
        ylabel="Degrees (deg)",
        title="Per-Sector RMSE Reduction & Applied Bias Offset",
    )
    ax_bar.grid(True, axis="y", ls="--", alpha=0.5)
    ax_bar.legend()

    fig.tight_layout()
    fig.savefig(f"imu_bias_error_{route}.png", dpi=300)
    plt.close(fig)


def plot_trajectory_headings(route, frames, lats, lons, imu_yaws, raw_deg, corr_deg):
    fig, (ax_map, ax_ts) = plt.subplots(2, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [1.8, 1]})

    ax_map.plot(lons, lats, c="lightgray", lw=1.5, alpha=0.8, label="Trajectory Path")
    quiver_configs = [
        (imu_yaws, "darkorange", 0.75, 0.0022, 3.2, 4.2, "Old (Raw) IMU Yaw"),
        (np.radians(corr_deg), "teal", 0.90, 0.0025, 3.5, 4.5, "Corrected IMU Yaw"),
    ]
    for y_rad, color, alpha, width, hw, hl, label in quiver_configs:
        ax_map.quiver(
            lons[::20], lats[::20],
            -np.sin(y_rad[::20]), np.cos(y_rad[::20]),
            color=color, alpha=alpha, scale=45, width=width,
            headwidth=hw, headlength=hl, pivot="middle", label=label,
        )
    ax_map.set(
        title=f"Old IMU vs. Corrected IMU Spatial Heading: {route}",
        xlabel="Longitude (East)",
        ylabel="Latitude (North)",
    )
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.grid(True, ls="--", alpha=0.5)
    ax_map.legend()

    ax_ts.plot(frames, raw_deg, c="darkorange", alpha=0.75, lw=1.2, label="Old (Raw) IMU Yaw (deg)")
    ax_ts.plot(frames, corr_deg, c="teal", alpha=0.90, lw=1.3, label="Corrected IMU Yaw (deg)")
    ax_ts.set(
        title="Old (Raw) IMU Yaw vs. Sector-Corrected IMU Yaw",
        xlabel="Frame Index",
        ylabel="Yaw (deg, [-180, 180])",
        ylim=(-190, 190),
    )
    ax_ts.grid(True, ls="--", alpha=0.5)
    ax_ts.legend()

    fig.tight_layout()
    fig.savefig(f"imu_raw_vs_corrected_{route}.png", dpi=300)
    plt.close(fig)

def process_route(route):
    subfolder = config.datadir / route

    loaded = load_route_data(subfolder, route)
    if loaded is None:
        return
    lats, lons, imu_yaws, ts = loaded

    dist_m, gnss_yaws = compute_gnss_kinematics(lats, lons)

    straight_mask = detect_straight_segments(dist_m, imu_yaws, gnss_yaws, ts)
    if not np.any(straight_mask):
        print(f"[WARNING] No straight moving samples > {MIN_SPEED_KMH} km/h for {route}.")
        return

    bias, counts, sector_ids = calibrate_sector_bias(imu_yaws, gnss_yaws, straight_mask)
    raw_deg = np.degrees(imu_yaws)
    corr_deg = apply_bearing_bias(raw_deg, bias)

    raw_err = np.degrees(wrap_rad(imu_yaws[straight_mask] - gnss_yaws[straight_mask]))
    corr_err = np.degrees(wrap_rad(np.radians(corr_deg[straight_mask]) - gnss_yaws[straight_mask]))

    frames = np.arange(len(lats))
    print_and_save_summary(subfolder, route, len(lats), straight_mask, bias, counts, raw_err, corr_err)
    plot_error_analysis(route, frames, straight_mask, sector_ids, bias, counts, raw_err, corr_err)
    plot_trajectory_headings(route, frames, lats, lons, imu_yaws, raw_deg, corr_deg)


def main():
    print(f"Selected routes: {config.select_route}")
    for route in config.select_route:
        process_route(route)


if __name__ == "__main__":
    main()