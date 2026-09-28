from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import yaml

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

def quaternion_to_yaw(q, offset=0):
    r = R.from_quat(q)
    euler_angles = r.as_euler('zxy', degrees=True)
    return euler_angles[0]

subfolders = sorted(
    p for p in config.datadir.iterdir() if p.is_dir() and (p / "meta").exists()
)
if config.select_route != "all":
    subfolders = [config.datadir / config.select_route]

subfolder = subfolders[0]
yml_files = sorted((subfolder / "meta").glob("*.yml"))

if not yml_files:
    raise FileNotFoundError(f"No .yml files found in {subfolder / 'meta'}")

sample_yml_idx = {"east": 0, "west": 9918, "north": 8967, "south": 12900}

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes_flat = axes.flatten()

for ax, (direction, idx) in zip(axes_flat, sample_yml_idx.items()):
    target_idx = min(idx, len(yml_files) - 1)
    sample_yml = yml_files[target_idx]

    with open(sample_yml, "r") as f:
        data = yaml.safe_load(f)

    raw_mag =  np.array(data['magnetic_field'], dtype=float)
    scale = 1e6 if np.max(np.abs(raw_mag)) < 1.0 else 1.0
    x0, y0 = config.magnetometer_calib["offset"]
    Q = np.array(config.magnetometer_calib["Q"])
    mags_uT = raw_mag * scale
    x_body = mags_uT[0]  # Forward (+X)
    y_body = mags_uT[2]  # Left (+Y)
    z_body = mags_uT[1]  # Up (+Z)

    mags_offset = np.array([x_body - x0, y_body - y0])

    mags_calibrated_xy = mags_offset @ Q.T
    x_cal = mags_calibrated_xy[0]
    y_cal = mags_calibrated_xy[1]
    z_cal = z_body - np.mean(z_body)
    mag_yaw_calibrated = mag_to_yaw(x_cal, y_cal)
    u_mag_calibrated = -np.sin(mag_yaw_calibrated)
    v_mag_calibrated = np.cos(mag_yaw_calibrated)

    mag_yaw_raw = mag_to_yaw(x_body, y_body)
    u_mag_raw = -np.sin(mag_yaw_raw)
    v_mag_raw = np.cos(mag_yaw_raw)

    q_raw = data.get("global_orientation_xyzw")

    quat_yaw = quaternion_to_yaw(q_raw)
    print("quat_yaw: ", quat_yaw)
    u_q = np.sin(np.radians(quaternion_to_yaw(q_raw)))
    v_q = np.cos(np.radians(quaternion_to_yaw(q_raw)))

    ax.quiver(
        0,
        0,
        u_q,
        v_q,
        angles='xy',
        scale_units='xy',
        scale=1,
        color='blue',
        width=0.015,
        label='Quaternion Yaw'
    )

    ax.quiver(
        0,
        0,
        u_mag_calibrated,
        v_mag_calibrated,
        angles='xy',
        scale_units='xy',
        scale=1,
        color='red',
        width=0.015,
        label='Calibrated Mag'
    )

    ax.quiver(
        0,
        0,
        u_mag_raw,
        v_mag_raw,
        angles='xy',
        scale_units='xy',
        scale=1,
        color='orange',
        width=0.015,
        label='Raw Mag'
    )

    ax.text(0, 1.6, "N", fontsize=12, fontweight="bold", ha="center", va="bottom", color="darkred")
    ax.text(0, -1.6, "S", fontsize=12, fontweight="bold", ha="center", va="top", color="darkred")
    ax.text(1.6, 0, "E", fontsize=12, fontweight="bold", ha="left", va="center", color="darkred")
    ax.text(-1.6, 0, "W", fontsize=12, fontweight="bold", ha="right", va="center", color="darkred")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f"Direction: {direction.capitalize()}\nQuat Yaw: {quat_yaw:.2f}°\nCalibrated Mag Raw: {np.degrees(mag_yaw_raw):.2f}\nCalibrated Mag Yaw: {np.degrees(mag_yaw_calibrated):.2f}", fontsize=12, fontweight="bold")

plt.tight_layout()
fig.savefig("quaternion.png", dpi=300)
plt.close(fig)