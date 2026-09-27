import yaml
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
from config import GlobalConfig

config = GlobalConfig()

BAG = Path("/media/mf/AUTODRIVING-4TB1/UGM Baru/rosbag2_2025_11_05-11_00_19/rosbag2_2025_11_05-11_00_19_0.mcap")
MAG_TOPIC = '/magnetometer'
SLOP_NS = 150_000_000  # 0.15 s, matches preprocess_rosbag_extraction.py
OVERWRITE = False  # recompute files that already have magnetic_field


def load_magnetometer(bag_path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    stamps = []
    fields = []
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == MAG_TOPIC]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            stamps.append(stamp)
            fields.append((
                float(msg.magnetic_field.x),
                float(msg.magnetic_field.y),
                float(msg.magnetic_field.z),
                [float(x) for x in msg.magnetic_field_covariance],
            ))
    return np.asarray(stamps, dtype=np.int64), fields


def main():
    if config.select_route == "all":
        raise ValueError("append_magnetic_field.py requires a single select_route, not 'all'")
    meta_path = config.datadir / config.select_route / "meta"
    file_list = sorted(meta_path.glob("*.yml"))
    if not file_list:
        raise FileNotFoundError(f"No .yml meta files found under {meta_path}")

    stamps, fields = load_magnetometer(BAG)
    if stamps.size == 0:
        raise RuntimeError(f"No '{MAG_TOPIC}' messages found in {BAG}")

    updated = skipped = unmatched = 0
    for f in tqdm(file_list, desc="Backfilling"):
        meta = yaml.safe_load(f.read_text())
        if 'magnetic_field' in meta and not OVERWRITE:
            skipped += 1
            continue

        target_ns = int(meta['sec']) * 1_000_000_000 + int(meta['nanosec'])
        idx = int(np.argmin(np.abs(stamps - target_ns)))
        if abs(int(stamps[idx]) - target_ns) > SLOP_NS:
            unmatched += 1
            print(f"[warn] no magnetometer match within slop for {f.name}")
            continue

        x, y, z, cov = fields[idx]
        meta['magnetic_field'] = [x, y, z]
        meta['magnetic_field_covariance'] = cov
        with open(f, 'w') as fh:
            yaml.dump(meta, fh)
        updated += 1

    print(f"updated={updated} skipped={skipped} unmatched={unmatched} total={len(file_list)}")


if __name__ == "__main__":
    main()
