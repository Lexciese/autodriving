from pathlib import Path
from collections import deque
import yaml
import numpy as np
import cv2
from datetime import date
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pypcd import pypcd
from cv_bridge import CvBridge
from config import GlobalConfig

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

configx = GlobalConfig()
BAG = Path("/media/mf/AUTODRIVING-4TB1/UGM Baru/rosbag2_2025_11_05-11_00_19/rosbag2_2025_11_05-11_00_19_0.mcap")
DATADIR = configx.datadir
PREFIX = str(date.today()) + "_route00"
SLOP_NS = 150_000_000
TOPICS = [
    '/gnss/fix',
    '/gnss/fix_velocity',
    '/zed/zed_node/odom',
    '/imu',
    '/zed/zed_node/rgb/image_rect_color',
    '/zed/zed_node/point_cloud/cloud_registered',
    '/zed/zed_node/depth/depth_registered',
    '/rslidar_points'
]

dirs = {
    'meta': DATADIR + PREFIX + "/meta/",
    'rgb': DATADIR + PREFIX + "/camera/rgb/",
    'depth_cld': DATADIR + PREFIX + "/camera/depth/cld/",
    'depth_cld2': DATADIR + PREFIX + "/camera/depth/cld2/",
    'depth_map': DATADIR + PREFIX + "/camera/depth/map/",
    'lidar': DATADIR + PREFIX + "/lidar/cld/"
}
for d in dirs.values():
    Path(d).mkdir(parents=True, exist_ok=True)

bridge = CvBridge()

def closest_in_dq(dq, target, slop):
    best, best_diff = None, slop + 1
    for t, msg in dq:
        diff = abs(t - target)
        if diff < best_diff:
            best_diff = diff
            best = (t, msg)
    return best if best_diff <= slop else None

def trim_dq(dq, limit):
    while dq and dq[0][0] < limit:
        dq.popleft()

def save(gnss_t, gnss_msg, sync_data):
    sec = str(gnss_msg.header.stamp.sec).zfill(10)
    nsec = str(gnss_msg.header.stamp.nanosec).zfill(10)
    fname = sec + "_" + nsec

    meta = {
        'sec': sec,
        'nanosec': nsec,
        'global_position_latlon': [gnss_msg.latitude, gnss_msg.longitude],
        'velocity': sync_data['/gnss/fix_velocity'][1].twist.linear.x,
        'local_position_xyz': [sync_data['/zed/zed_node/odom'][1].pose.pose.position.x,
                               sync_data['/zed/zed_node/odom'][1].pose.pose.position.y,
                               sync_data['/zed/zed_node/odom'][1].pose.pose.position.z],
        'local_orientation_xyzw': [sync_data['/zed/zed_node/odom'][1].pose.pose.orientation.x,
                                   sync_data['/zed/zed_node/odom'][1].pose.pose.orientation.y,
                                   sync_data['/zed/zed_node/odom'][1].pose.pose.orientation.z,
                                   sync_data['/zed/zed_node/odom'][1].pose.pose.orientation.w],
        'global_orientation_xyzw': [sync_data['/imu'][1].orientation.x,
                                    sync_data['/imu'][1].orientation.y,
                                    sync_data['/imu'][1].orientation.z,
                                    sync_data['/imu'][1].orientation.w],
        'angular_speed_xyz': [sync_data['/imu'][1].angular_velocity.x,
                              sync_data['/imu'][1].angular_velocity.y,
                              sync_data['/imu'][1].angular_velocity.z],
        'acceleration_xyz': [sync_data['/imu'][1].linear_acceleration.x,
                             sync_data['/imu'][1].linear_acceleration.y,
                             sync_data['/imu'][1].linear_acceleration.z]
    }
    with open(dirs['meta'] + fname + ".yml", 'w') as f:
        yaml.dump(meta, f)

    rgb = sync_data['/zed/zed_node/rgb/image_rect_color'][1]
    cv_img = bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
    cv2.imwrite(dirs['rgb'] + fname + ".png", cv_img)

    dep_pc_msg = sync_data['/zed/zed_node/point_cloud/cloud_registered'][1]
    dep_pc = pypcd.PointCloud.from_msg(dep_pc_msg)
    dep_pc.save_pcd(dirs['depth_cld'] + fname + ".pcd", compression='binary_compressed')
    points3 = dep_pc.pc_data[['x', 'y', 'z']]
    np.save(dirs['depth_cld2'] + fname + ".npy", points3)

    dep_img = sync_data['/zed/zed_node/depth/depth_registered'][1]
    depth = bridge.imgmsg_to_cv2(dep_img, desired_encoding='passthrough')
    np.save(dirs['depth_map'] + fname + ".npy", depth)

    lidar_msg = sync_data['/rslidar_points'][1]
    lidar_pc = pypcd.PointCloud.from_msg(lidar_msg)
    lidar_pc.save_pcd(dirs['lidar'] + fname + ".pcd", compression='binary_compressed')

def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([BAG], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        deques = {t: deque() for t in TOPICS if t != '/gnss/fix'}
        pending_gnss = deque()

        msg_iter = reader.messages(connections=conns)
        if tqdm is not None:
            msg_iter = tqdm(msg_iter, desc="Processing", unit="msg")

        for conn, ts, raw in msg_iter:
            msg = reader.deserialize(raw, conn.msgtype)
            if not hasattr(msg, 'header'):
                continue
            t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            topic = conn.topic

            if topic == '/gnss/fix':
                pending_gnss.append((t_ns, msg))
            else:
                deques[topic].append((t_ns, msg))
                trim_dq(deques[topic], t_ns - SLOP_NS)

            while pending_gnss and pending_gnss[0][0] + SLOP_NS <= t_ns:
                gnss_t, gnss_msg = pending_gnss.popleft()
                sync_data = {}
                ok = True
                for t in deques:
                    closest = closest_in_dq(deques[t], gnss_t, SLOP_NS)
                    if closest is None:
                        ok = False
                        break
                    sync_data[t] = closest
                if ok:
                    save(gnss_t, gnss_msg, sync_data)

        while pending_gnss:
            gnss_t, gnss_msg = pending_gnss.popleft()
            sync_data = {}
            ok = True
            for t in deques:
                closest = closest_in_dq(deques[t], gnss_t, SLOP_NS)
                if closest is None:
                    ok = False
                    break
                sync_data[t] = closest
            if ok:
                save(gnss_t, gnss_msg, sync_data)

if __name__ == "__main__":
    main()