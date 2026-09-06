from pathlib import Path
import numpy as np
from datetime import date
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pypcd import pypcd
from config import GlobalConfig

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Initialize Configuration
configx = GlobalConfig()
BAG = Path("/home/lexciese/Dev/autodriving/ai23/lidar_new_college/lidar_new_college.mcap")
DATADIR = configx.datadir
PREFIX = str(date.today()) + "_route00"
TOPIC = '/velodyne_points'

# Set up output directory
lidar_dir = Path(DATADIR) / PREFIX / "lidar" / "cld"
lidar_dir.mkdir(parents=True, exist_ok=True)

def save_lidar(msg):
    ts = msg.header.stamp
    sec = str(ts.sec).zfill(10)
    nsec = str(ts.nanosec).zfill(10)
    fname = f"{sec}_{nsec}"
    lidar_pc = pypcd.PointCloud.from_msg(msg)
    print(f"datatype lidar msg: {type(msg)}")
    print(f"datatype lidar_pc: {type(lidar_pc)}")

    # Filter out invalid NaN points (x, y, z)
    mask = ~(
        np.isnan(lidar_pc.pc_data['x']) | 
        np.isnan(lidar_pc.pc_data['y']) | 
        np.isnan(lidar_pc.pc_data['z'])
    )
    lidar_pc.pc_data = lidar_pc.pc_data[mask]

    out_path = lidar_dir / f"{fname}.pcd"
    lidar_pc.save_pcd(str(out_path), compression='binary_compressed')

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
            save_lidar(msg)

if __name__ == "__main__":
    main()