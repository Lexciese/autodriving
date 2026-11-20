"""Estimate the per-route rigid alignment between GNSS ENU and ZED odometry.

The alignment maps local ENU metres (derived from ``route_gnss_base.csv``) to
the ZED odom world used by waypoints:

    [x_odom, y_odom] = scale * R(theta) @ [east, north] + [tx, ty]

Outputs
-------
<route>/align_gnss_odom.npz
<route>/align_gnss_odom.png   (unless --no-plot)

Run with the training environment, for example:

    conda run -n dl python scripts/estimate_route_alignment.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GlobalConfig
from transforms import fit_rigid_2d, latlon_to_enu, save_alignment

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_gnss(route: str, route_path: Path, config: GlobalConfig):
    base_path = route_path / config.route_gnss_base_filename
    raw_path = route_path / f"{route}_gnss.csv"
    if base_path.exists():
        return pd.read_csv(base_path, dtype={'timestamp': str}), base_path
    if raw_path.exists():
        print(f"[warn] {base_path.name} not found, falling back to {raw_path.name}")
        return pd.read_csv(raw_path, dtype={'timestamp': str}), raw_path
    raise FileNotFoundError(
        f"No GNSS CSV found for {route_path}. Run preprocess_route.py first."
    )


def load_odom(route_path: Path):
    meta_dir = route_path / "meta"
    odom = {}
    for meta_path in sorted(meta_dir.glob("*.yml")):
        with open(meta_path, "r") as handle:
            meta = yaml.safe_load(handle)
        position = meta.get("local_position_xyz")
        if position is None:
            continue
        odom[meta_path.stem] = (float(position[0]), float(position[1]))
    return odom


def align_route(route: str, config: GlobalConfig, args):
    route_path = config.datadir / route
    gnss_df, gnss_source = load_gnss(route, route_path, config)
    odom = load_odom(route_path)

    timestamps = gnss_df["timestamp"].astype(str).tolist()
    lats = gnss_df["latitude"].to_numpy(dtype=float)
    lons = gnss_df["longitude"].to_numpy(dtype=float)

    src = []
    dst = []
    used_timestamps = []
    for timestamp, lat, lon in zip(timestamps, lats, lons):
        if timestamp not in odom:
            continue
        east, north = latlon_to_enu(lat, lon, lats[0], lons[0])
        src.append((east, north))
        dst.append(odom[timestamp])
        used_timestamps.append(timestamp)

    if len(src) < 3:
        raise RuntimeError(f"Route {route}: only {len(src)} GNSS/odom pairs, need >= 3")

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    fit = fit_rigid_2d(
        src,
        dst,
        with_scale=args.with_scale,
        trim_frac=args.trim_frac,
        iters=args.iters,
    )

    out_path = route_path / config.align_gnss_odom_filename
    save_alignment(
        out_path,
        fit["R"],
        fit["t"],
        fit["scale"],
        anchor_latlon=np.array([lats[0], lons[0]]),
        route=route,
        source=str(gnss_source),
        mode="similarity" if args.with_scale else "rigid",
        n_points=len(src),
        theta_rad=fit["theta"],
        theta_deg=np.degrees(fit["theta"]),
        residual_rms=fit["residual_rms"],
        residual_max=fit["residual_max"],
        inlier_mask=fit["inlier_mask"],
        timestamps=np.asarray(used_timestamps),
        created=datetime.now().isoformat(timespec="seconds"),
    )

    print(
        f"[{route}] n={len(src)} scale={fit['scale']:.6f} "
        f"theta={np.degrees(fit['theta']):.3f} deg "
        f"t=({fit['t'][0]:.3f}, {fit['t'][1]:.3f}) "
        f"rms={fit['residual_rms']:.3f} m max={fit['residual_max']:.3f} m"
    )
    print(f"[{route}] saved {out_path}")

    if not args.no_plot:
        plot_alignment(route_path, src, dst, fit, route, config)
    return fit


def plot_alignment(route_path: Path, src, dst, fit, route, config):
    aligned = fit["scale"] * (src @ fit["R"].T) + fit["t"]
    residual = np.linalg.norm(aligned - dst, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(src[:, 0], src[:, 1], color="tab:blue", label="GNSS ENU")
    ax.plot(dst[:, 0], dst[:, 1], color="tab:orange", label="ZED odom")
    ax.plot(aligned[:, 0], aligned[:, 1], "--", color="tab:green", label="aligned GNSS")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / East (m)")
    ax.set_ylabel("y / North (m)")
    ax.set_title(f"{route}: trajectories")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(residual, color="tab:red", linewidth=1.0)
    ax.axhline(fit["residual_rms"], color="black", linestyle="--", label="RMS")
    ax.set_xlabel("sample")
    ax.set_ylabel("residual (m)")
    ax.set_title("alignment residual")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path = route_path / config.align_gnss_odom_filename.replace(".npz", ".png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[{route}] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", nargs="*", default=None, help="route names (default: config.select_route)")
    parser.add_argument("--with-scale", action="store_true", help="estimate scale (default: rigid, scale=1)")
    parser.add_argument("--trim-frac", type=float, default=0.9, help="inlier fraction per trimming iteration")
    parser.add_argument("--iters", type=int, default=6, help="trimming iterations")
    parser.add_argument("--no-plot", action="store_true", help="skip diagnostic plots")
    args = parser.parse_args()

    config = GlobalConfig()
    routes = args.routes if args.routes else list(config.select_route)
    for route in routes:
        align_route(route, config, args)


if __name__ == "__main__":
    main()
