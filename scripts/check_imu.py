from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GlobalConfig

config = GlobalConfig()


def quaternion_to_yaw(q_raw):
    """
    Returns yaw angle in degrees.
    Assumes standard orientation output from scipy (counter-clockwise from East).
    """
    r_align = R.from_euler("x", 90, degrees=True)
    r_raw = R.from_quat(q_raw)
    r_correct = r_align * r_raw
    # Get yaw around Z axis
    euler_angles = r_correct.as_euler("zyx", degrees=True)
    yaw = euler_angles[0]
    return yaw


subfolders = sorted(
    p for p in config.datadir.iterdir() if p.is_dir() and (p / "meta").exists()
)
if config.select_route != "all":
    subfolders = [config.datadir / config.select_route]
    print(f"only route: {config.select_route} is selected")

for subfolder in subfolders:
    bearings, labels, lats, lons = [], [], [], []
    yml_files = sorted((subfolder / "meta").glob("*.yml"))

    for idx, yml_file in enumerate(yml_files, start=1):
        with open(yml_file, "r") as f:
            data = yaml.safe_load(f)
        latlon = data.get("global_position_latlon")
        lats.append(latlon[0])
        lons.append(latlon[1])
        q_raw = data.get("global_orientation_xyzw")
        bearing = quaternion_to_yaw(q_raw)
        bearings.append(bearing)
        labels.append(f"#{idx} {yml_file.name}")

    lats = np.array(lats)
    lons = np.array(lons)
    bearings = np.array(bearings)

    # Convert compass bearing (0° = North, 90° = East) to Cartesian directions (u = East, v = North)
    # If bearing is compass angle (0° North, clockwise):
    rad = np.radians(bearings)
    u = np.sin(rad)  # East (+X)
    v = np.cos(rad)  # North (+Y)

    step = 20
    sub_lats = lats[::step]
    sub_lons = lons[::step]
    sub_u = u[::step]
    sub_v = v[::step]
    sub_bearings = bearings[::step]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot full route trajectory line (Longitude = X, Latitude = Y)
    ax.plot(
        lons, lats, color="blue", linewidth=1.5, alpha=0.7, label="Route Trajectory"
    )

    # Plot heading arrows
    quiver = ax.quiver(
        sub_lons,
        sub_lats,
        sub_u,
        sub_v,
        sub_bearings,
        cmap="viridis",
        scale=50,  # Adjusted scale for better arrow visibility
        width=0.003,
        headwidth=3,
        headlength=4,
        pivot="middle",
    )

    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label("Bearing (°)")

    ax.set_title(
        f"Route Trajectory & Bearings (1 in every {step} frames): {subfolder.name}",
        fontsize=14,
    )
    ax.set_xlabel("Longitude (East)")
    ax.set_ylabel("Latitude (North)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    # Ensures 1 degree Latitude equals 1 degree Longitude spatially
    ax.set_aspect("equal", adjustable="datalim")

    output_png_path = f"bearings_map_{subfolder.name}.png"
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    print(f"Saved bearing plot to {output_png_path}")