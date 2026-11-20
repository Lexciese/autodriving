"""Overlay rp1/rp2 and waypoints on the LiDAR BEV/front projections.

The script uses the exact same ``plot_lidbev_rpwp`` / ``plot_lidfront_rpwp``
helpers as the rest of the repo.  Because rp1/rp2 and waypoints are now both
expressed in the ZED odom frame, and the LiDAR projections are generated in the
same base frame, the markers should land on the road consistently.

Examples
--------
    conda run -n dl python scripts/overlay_rp_wp.py --split val --start 0 --end 200 --step 5
    conda run -n dl python scripts/overlay_rp_wp.py --split test --video overlay_test.avi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a_dataloader import KarrDataset, SplitDataset
from config import GlobalConfig
from preprocess_util import plot_lidbev_rpwp, plot_lidfront_rpwp
from preview_dataloader import colorize_logdepth, colorize_seg


def draw_overlay(sample, config, index):
    colmap = config.SEG_CLASSES['colors']
    bev_seg = sample['bev_segs'][-1]
    bev_dep = sample['bev_deps'][-1]
    front_seg = sample['front_segs'][-1]
    front_dep = sample['front_deps'][-1]

    bev = colorize_seg(bev_seg, colmap)
    front = colorize_seg(front_seg, colmap)
    bev_dep = colorize_logdepth(bev_dep)
    front_dep = colorize_logdepth(front_dep)

    if bev.ndim == 2:
        bev = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    if front.ndim == 2:
        front = cv2.cvtColor(front, cv2.COLOR_GRAY2BGR)
    bev = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
    front = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
    front_dep = cv2.cvtColor(front_dep, cv2.COLOR_GRAY2BGR) if front_dep.ndim == 2 else front_dep

    rp_colors = [(255, 255, 255), (255, 255, 0)]
    for k, key in enumerate(('rp1', 'rp2')):
        rp = sample[key]
        bx, by = plot_lidbev_rpwp(config, rp[0], rp[1])
        fx, fy = plot_lidfront_rpwp(config, rp[0], rp[1])
        cv2.circle(bev, (int(bx), int(by)), 5, rp_colors[k], 2)
        cv2.circle(front, (int(fx), int(fy)), 5, rp_colors[k], 2)

    for wp in sample['waypoints']:
        bx, by = plot_lidbev_rpwp(config, wp[0], wp[1])
        fx, fy = plot_lidfront_rpwp(config, wp[0], wp[1])
        cv2.circle(bev, (int(bx), int(by)), 3, (255, 255, 255), -1)
        cv2.circle(front, (int(fx), int(fy)), 3, (255, 255, 255), -1)

    # Ego origin marker (base origin).
    ex, ey = plot_lidbev_rpwp(config, 0.0, 0.0)
    cv2.drawMarker(bev, (int(ex), int(ey)), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)

    header = np.zeros((30, bev.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"idx={index} file={sample['filename']} zed_bearing={sample.get('bearing_zed', 0.0):.2f} deg",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    bev_h = int(bev.shape[0] * (front.shape[1] / bev.shape[1])) if bev.shape[1] else bev.shape[0]
    bev = cv2.resize(bev, (front.shape[1], bev_h))
    return np.concatenate((header, front, np.zeros((10, front.shape[1], 3), np.uint8), bev), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("overlay_rp_wp"))
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    config = GlobalConfig()
    dataset = KarrDataset(config)
    splits = SplitDataset(dataset)
    split_set = getattr(splits, f"{args.split}_set")

    total = len(split_set)
    end = total if args.end is None else min(args.end, total)
    indices = list(range(args.start, end, args.step))
    if args.max_frames is not None:
        indices = indices[: args.max_frames]

    args.out.mkdir(parents=True, exist_ok=True)
    writer = None
    print(f"Rendering {len(indices)} frames from split '{args.split}' (total={total})")

    for index in indices:
        sample = split_set[index]
        frame = draw_overlay(sample, config, index)
        out_path = args.out / f"{args.split}_{index:06d}.png"
        cv2.imwrite(str(out_path), frame)
        if args.video is not None:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.video),
                    cv2.VideoWriter_fourcc(*"XVID"),
                    config.fps,
                    (w, h),
                )
            writer.write(frame)
        print(f"saved {out_path}")

    if writer is not None:
        writer.release()
        print(f"saved {args.video}")


if __name__ == "__main__":
    main()
