from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from config import GlobalConfig

config = GlobalConfig()
MIN_SPEED_KMH, MAX_YAW_RATE_DEG_S = 1.0, 3.0
STRAIGHT_WINDOW, MAX_WINDOW_SPAN_DEG = 5, 4.0
DEFAULT_DT_S = getattr(config, "gnss_dt", 1.0)
SECTOR_NAMES = ["0 to 60", "60 to 120", "120 to 180", "-180 to -120", "-120 to -60", "-60 to 0"]

wrap_rad = lambda r: (r + np.pi) % (2 * np.pi) - np.pi
wrap_deg = lambda d: (d + 180.0) % 360.0 - 180.0
circ_mean = lambda d: float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(d))), np.mean(np.cos(np.radians(d))))))
rmse = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def bearing_biasing(angles_deg, b):
    """Vectorized 6-sector piecewise-linear bias correction matching original breakpoints."""
    mid = (b[2] + b[3]) / 2.0
    xp = [-180, -170, -130, -110, -70, -50, 0, 50, 70, 110, 130, 170, 180]
    fp = [mid, b[3], b[3], b[4], b[4], b[5], b[0], b[0], b[1], b[1], b[2], b[2], mid]
    out = wrap_deg(np.asarray(angles_deg) + np.interp(angles_deg, xp, fp))
    return float(out) if np.isscalar(angles_deg) else out


def quat_to_yaw(q):
    x, y, z, w = q
    return wrap_rad(np.arctan2(-(1.0 - 2.0 * (y * y + z * z)), 2.0 * (w * z + x * y)))


def haversine_and_yaw(lat, lon):
    """Vectorized distance (m) and forward bearing (rad) between consecutive points."""
    r_lat, r_lon = np.radians(lat), np.radians(lon)
    dlat, dlon = np.diff(r_lat), np.diff(r_lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(r_lat[:-1]) * np.cos(r_lat[1:]) * np.sin(dlon / 2) ** 2
    dist = 2.0 * 6371000.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    x = np.sin(dlon) * np.cos(r_lat[1:])
    y = np.cos(r_lat[:-1]) * np.sin(r_lat[1:]) - np.sin(r_lat[:-1]) * np.cos(r_lat[1:]) * np.cos(dlon)
    yaw = np.append(np.where(dist >= 0.05, wrap_rad(np.arctan2(-x, y)), np.nan), np.nan)
    return dist, yaw


def get_straight_mask(dist_m, imu_yaws, gnss_yaws, ts):
    n, dt = len(imu_yaws), np.diff(ts)
    valid_dt = dt > 0
    speed_kmh = np.where(valid_dt, (dist_m / np.where(valid_dt, dt, 1.0)) * 3.6, 0.0)
    imu_rate = np.where(valid_dt, np.abs(np.degrees(wrap_rad(np.diff(imu_yaws)))) / np.where(valid_dt, dt, 1.0), np.inf)
    gnss_diff = np.degrees(wrap_rad(np.diff(gnss_yaws)))
    gnss_rate = np.where(valid_dt & ~np.isnan(gnss_diff), np.abs(gnss_diff) / np.where(valid_dt, dt, 1.0), 0.0)

    step_ok = np.append(
        valid_dt & (speed_kmh > MIN_SPEED_KMH) & ~np.isnan(gnss_yaws[:-1])
        & (imu_rate <= MAX_YAW_RATE_DEG_S) & (gnss_rate <= MAX_YAW_RATE_DEG_S),
        False,
    )
    mask = np.zeros(n, dtype=bool)
    hw = STRAIGHT_WINDOW // 2
    for i in range(hw, n - hw):
        s = slice(i - hw, i + hw + 1)
        if np.all(step_ok[s]):
            imu_span = np.max(np.abs(np.degrees(wrap_rad(imu_yaws[s] - imu_yaws[i]))))
            gnss_span = np.max(np.abs(np.degrees(wrap_rad(gnss_yaws[s] - gnss_yaws[i]))))
            mask[i] = (imu_span <= MAX_WINDOW_SPAN_DEG) and (gnss_span <= MAX_WINDOW_SPAN_DEG)
    return mask


print(f"Selected routes: {config.select_route}")
for route in config.select_route:
    subfolder = config.datadir / route
    gnss_file = subfolder / f"{route}_gnss.csv"
    if not (subfolder / "meta").exists() or not gnss_file.exists():
        continue

    gnss_df = pd.read_csv(gnss_file)
    gnss_df["ts_key"] = gnss_df["timestamp"].astype(str).str.replace(".yml", "", regex=False)
    t_col = next((c for c in ["time", "timestamp_s", "sec", "epoch"] if c in gnss_df.columns), None)
    gnss_map = gnss_df.set_index("ts_key").to_dict("index")

    rows = []
    for idx, yml_file in enumerate(sorted((subfolder / "meta").glob("*.yml"))):
        row = gnss_map.get(yml_file.stem)
        data = yaml.safe_load(yml_file.read_text()) if row else None
        if row and data and data.get("global_orientation_xyzw") is not None:
            t = row[t_col] if (t_col and pd.notna(row[t_col])) else (
                data["timestamp"] if isinstance(data.get("timestamp"), (int, float)) else idx * DEFAULT_DT_S
            )
            rows.append((row["latitude"], row["longitude"], quat_to_yaw(data["global_orientation_xyzw"]), float(t)))

    if len(rows) < 2:
        print(f"[WARNING] Insufficient matched frames for {route}.")
        continue

    lats, lons, imu_yaws, ts = map(np.array, zip(*rows))
    dist_m, gnss_yaws = haversine_and_yaw(lats, lons)
    mask = get_straight_mask(dist_m, imu_yaws, gnss_yaws, ts)
    if not np.any(mask):
        print(f"[WARNING] No straight moving samples > {MIN_SPEED_KMH} km/h for {route}.")
        continue

    # Sector assignment (0..5) and bias calculation
    raw_deg = np.degrees(imu_yaws)
    sec_ids = np.floor(raw_deg / 60.0).astype(int) % 6
    sec_ids[raw_deg == 180.0] = 2

    corr_samples = np.degrees(wrap_rad(gnss_yaws[mask] - imu_yaws[mask]))
    global_bias = circ_mean(corr_samples)
    counts = [int(np.sum(sec_ids[mask] == s)) for s in range(6)]
    bias = [round(circ_mean(corr_samples[sec_ids[mask] == s]) if counts[s] else global_bias, 4) for s in range(6)]

    corr_deg = bearing_biasing(raw_deg, bias)
    raw_err = np.degrees(wrap_rad(imu_yaws[mask] - gnss_yaws[mask]))
    corr_err = np.degrees(wrap_rad(np.radians(corr_deg[mask]) - gnss_yaws[mask]))

    print(f"\n[{route}] Total frames: {len(lats)} | Straight > {MIN_SPEED_KMH} km/h: {mask.sum()}")
    print(f"  bearing_bias (deg): {bias}")
    for name, b_val, cnt in zip(SECTOR_NAMES, bias, counts):
        print(f"    Sector [{name:>11} deg]: {b_val:+7.2f}°  (n={cnt}){' (fallback: global mean)' if not cnt else ''}")
    print(f"  Before -> Bias: {circ_mean(raw_err):+.2f}° | MAE: {np.mean(np.abs(raw_err)):.2f}° | RMSE: {rmse(raw_err):.2f}°")
    print(f"  After  -> Bias: {circ_mean(corr_err):+.2f}° | MAE: {np.mean(np.abs(corr_err)):.2f}° | RMSE: {rmse(corr_err):.2f}°")

    bias_path = subfolder / f"{route}_imu_sector_bias.yml"
    bias_path.write_text(yaml.safe_dump({"bearing_bias_deg": dict(zip(SECTOR_NAMES, bias))}, sort_keys=False))

    # Plot 1: Residuals & Per-Sector RMSE
    frames = np.arange(len(lats))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1.3, 1]})
    ax1.scatter(frames[mask], raw_err, c="darkorange", s=10, alpha=0.55,
                label=f"Raw IMU Error (Bias={circ_mean(raw_err):+.2f}°, MAE={np.mean(np.abs(raw_err)):.2f}°, RMSE={rmse(raw_err):.2f}°)")
    ax1.scatter(frames[mask], corr_err, c="teal", s=10, alpha=0.8,
                label=f"Corrected IMU Error (Bias={circ_mean(corr_err):+.2f}°, MAE={np.mean(np.abs(corr_err)):.2f}°, RMSE={rmse(corr_err):.2f}°)")
    ax1.axhline(0, c="k", ls=":", lw=1.1)
    ax1.set(title=f"Heading Residual & RMSE Comparison (Straight > {MIN_SPEED_KMH} km/h): {route}",
            xlabel="Frame Index", ylabel="Wrapped Error (IMU - GNSS, deg)", ylim=(-185, 185))
    ax1.grid(True, ls="--", alpha=0.5); ax1.legend()

    x, w = np.arange(6), 0.28
    ax2.bar(x - w, [rmse(raw_err[sec_ids[mask] == s]) for s in range(6)], w, color="darkorange", alpha=0.8, label="Raw Sector RMSE (deg)")
    ax2.bar(x, [rmse(corr_err[sec_ids[mask] == s]) for s in range(6)], w, color="teal", alpha=0.85, label="Corrected Sector RMSE (deg)")
    ax2.bar(x + w, bias, w, color="slateblue", alpha=0.8, label="Applied Sector Bias (deg)")
    ax2.axhline(0, c="k", ls=":", lw=1.0)
    ax2.set(xticks=x, xticklabels=[f"{n}\n(n={c})" for n, c in zip(SECTOR_NAMES, counts)],
            ylabel="Degrees (deg)", title="Per-Sector RMSE Reduction & Applied Bias Offset")
    ax2.grid(True, axis="y", ls="--", alpha=0.5); ax2.legend()
    fig.tight_layout(); fig.savefig(f"imu_bias_error_{route}.png", dpi=300); plt.close(fig)

    # Plot 2: Spatial Quiver & Time-Series Yaw
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [1.8, 1]})
    ax1.plot(lons, lats, c="lightgray", lw=1.5, alpha=0.8, label="Trajectory Path")
    for y_rad, col, alp, wid, hw, hl, lbl in [
        (imu_yaws, "darkorange", 0.75, 0.0022, 3.2, 4.2, "Old (Raw) IMU Yaw"),
        (np.radians(corr_deg), "teal", 0.9, 0.0025, 3.5, 4.5, "Corrected IMU Yaw"),
    ]:
        ax1.quiver(lons[::20], lats[::20], -np.sin(y_rad[::20]), np.cos(y_rad[::20]), color=col,
                   alpha=alp, scale=45, width=wid, headwidth=hw, headlength=hl, pivot="middle", label=lbl)
    ax1.set(title=f"Old IMU vs. Corrected IMU Spatial Heading: {route}", xlabel="Longitude (East)", ylabel="Latitude (North)")
    ax1.set_aspect("equal", adjustable="datalim"); ax1.grid(True, ls="--", alpha=0.5); ax1.legend()

    ax2.plot(frames, raw_deg, c="darkorange", alpha=0.75, lw=1.2, label="Old (Raw) IMU Yaw (deg)")
    ax2.plot(frames, corr_deg, c="teal", alpha=0.9, lw=1.3, label="Corrected IMU Yaw (deg)")
    ax2.set(title="Old (Raw) IMU Yaw vs. Sector-Corrected IMU Yaw", xlabel="Frame Index", ylabel="Yaw (deg, [-180, 180])", ylim=(-190, 190))
    ax2.grid(True, ls="--", alpha=0.5); ax2.legend()
    fig.tight_layout(); fig.savefig(f"imu_raw_vs_corrected_{route}.png", dpi=300); plt.close(fig)