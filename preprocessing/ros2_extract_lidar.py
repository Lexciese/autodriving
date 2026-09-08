from pathlib import Path
import numpy as np
from datetime import date
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
# from pypcd import pypcd
from pypcd4 import PointCloud, Encoding
from preprocessing.config import GlobalConfig

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Initialize Configuration
configx = GlobalConfig()
BAG = Path("/media/mf/AUTODRIVING-4TB1/UGM Baru/rosbag2_2025_11_05-11_00_19/rosbag2_2025_11_05-11_00_19_0.mcap")
DATADIR = configx.datadir
PREFIX = str(date.today()) + "_route00"
TOPIC = '/rslidar_points'

# Set up output directory
lidar_dir = Path(DATADIR) / PREFIX / "lidar" / "cld"
lidar_dir.mkdir(parents=True, exist_ok=True)

def save_lidar(msg):
    ts = msg.header.stamp
    sec = str(ts.sec).zfill(10)
    nsec = str(ts.nanosec).zfill(10)
    fname = f"{sec}_{nsec}"
    lidar_pc = PointCloud.from_msg(msg)
    lidar_pc = lidar_pc.numpy(["x", "y", "z", "intensity"])
    xyz_mask = ~np.isnan(lidar_pc[:, :3]).any(axis=1)
    lidar_pc = lidar_pc[xyz_mask]
    lidar_pc = PointCloud.from_xyzi_points(lidar_pc)
    out_path = lidar_dir / f"{fname}.pcd" 
    lidar_pc.save(str(out_path), encoding=Encoding.BINARY_COMPRESSED)

def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    with AnyReader([BAG], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        if not conns:
            print(f"Topic {TOPIC} not found in bag file.")
            return

        total = sum(c.msgcount for c in conns if hasattr(c, 'msgcount')) or None

        msg_iter = reader.messages(connections=conns)
        if tqdm is not None:
            msg_iter = tqdm(msg_iter, total=total, desc="Extracting LiDAR", unit="msg")

        for conn, ts, raw in msg_iter:
            msg = reader.deserialize(raw, conn.msgtype)
            print(type(msg))
            save_lidar(msg)

if __name__ == "__main__":
    main()