from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GlobalConfig

config = GlobalConfig()


def mag_to_yaw(mx, my, offset=0.0):
    """
    mx: Forward (+X), my: Left (+Y) in body frame.
    Returns yaw in radians within range [-pi, pi].
    North = 0, West = +pi/2 (+90 deg), East = -pi/2 (-90 deg).
    """
    yaw = np.arctan2(-my, mx)
    return ((yaw + offset) + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_to_yaw(q_raw):
    from scipy.spatial.transform import Rotation as R

    r_align = R.from_euler("x", 90, degrees=True)
    r_raw = R.from_quat(q_raw)
    r_correct = r_align * r_raw
    euler_angles = r_correct.as_euler("xyz", degrees=True)
    return euler_angles[2]


subfolders = sorted(
    p for p in config.datadir.iterdir() if p.is_dir() and (p / "meta").exists()
)
if config.select_route != "all":
    subfolders = [config.datadir / config.select_route]
    print(f"Only route selected: {config.select_route}")

x0, y0 = config.magnetometer_calib["offset"]
Q = np.array(config.magnetometer_calib["Q"])

for subfolder in subfolders:
    lats, lons = [], []
    quat_bearings, cal_mag_bearings, raw_mag_bearings = [], [], []
    yml_files = sorted((subfolder / "meta").glob("*.yml"))

    for idx, yml_file in tqdm(enumerate(yml_files, start=1)):
        with open(yml_file, "r") as f:
            data = yaml.safe_load(f)

        latlon = data.get("global_position_latlon")
        lats.append(latlon[0])
        lons.append(latlon[1])

        q_raw = data.get("global_orientation_xyzw")
        q_yaw = quaternion_to_yaw(q_raw)
        quat_bearings.append(q_yaw)

        raw_mag = np.array(data["magnetic_field"], dtype=float)
        scale = 1e6 if np.max(np.abs(raw_mag)) < 1.0 else 1.0
        mags_uT = raw_mag * scale
        x_body = mags_uT[0]  # Forward (+X)
        y_body = mags_uT[2]  # Left (+Y)

        # Raw Mag Bearing
        raw_yaw = np.degrees(mag_to_yaw(x_body, y_body))
        raw_mag_bearings.append(raw_yaw)

        # Calibrated Mag Bearing
        mags_offset = np.array([x_body - x0, y_body - y0])
        mags_calibrated_xy = mags_offset @ Q.T
        cal_yaw = np.degrees(mag_to_yaw(mags_calibrated_xy[0], mags_calibrated_xy[1]))
        cal_mag_bearings.append(cal_yaw)

    lats = np.array(lats)
    lons = np.array(lons)
    quat_bearings = np.array(quat_bearings)
    cal_mag_bearings = np.array(cal_mag_bearings)
    raw_mag_bearings = np.array(raw_mag_bearings)

    step = 30
    sub_lats = lats[::step]
    sub_lons = lons[::step]

    u_q = np.sin(np.radians(quat_bearings[::step]))
    v_q = np.cos(np.radians(quat_bearings[::step]))

    u_cal = -np.sin(np.radians(cal_mag_bearings[::step]))
    v_cal = np.cos(np.radians(cal_mag_bearings[::step]))

    u_raw = -np.sin(np.radians(raw_mag_bearings[::step]))
    v_raw = np.cos(np.radians(raw_mag_bearings[::step]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    sources = [
        ("Quaternion Bearing", u_q, v_q, "blue"),
        ("Raw Mag Bearing", u_raw, v_raw, "red"),
        ("Calibrated Mag Bearing", u_cal, v_cal, "green"),
    ]

    for ax, (title, u, v, color) in zip(axes, sources):
        
        ax.quiver(
            sub_lons,
            sub_lats,
            u,
            v,
            color=color,
            scale=50,
            width=0.003,
            headwidth=5,
            headlength=5,
            pivot="middle",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (East)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("Latitude (North)")

    fig.suptitle(
        f"Route Trajectory & Bearings Comparison: {subfolder.name}",
        fontsize=14,
        fontweight="bold",
    )

    output_png_path = f"bearings_map_{subfolder.name}.png"
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)
    print(f"Saved bearing map plot to {output_png_path}")