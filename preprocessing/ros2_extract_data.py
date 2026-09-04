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
SLOP_NS = 150_000_000  # 0.15 s
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

class ApproximateTimeSynchronizer:
    def __init__(self, topics, slop_ns, callback):
        self.topics = topics
        self.slop_ns = slop_ns
        self.callback = callback
        self.queues = {t: deque() for t in topics}

    def add(self, topic, msg):
        if topic not in self.queues:
            return
        self.queues[topic].append(msg)
        self._process()

    def _process(self):
        while all(self.queues[t] for t in self.topics):
            t0 = {t: self._get_timestamp(self.queues[t][0]) for t in self.topics}
            min_t = min(t0.values())
            max_t = max(t0.values())
            if max_t - min_t <= self.slop_ns:
                sync_msgs = {t: self.queues[t].popleft() for t in self.topics}
                self.callback(sync_msgs)
            else:
                min_topic = min(t0, key=t0.get)
                self.queues[min_topic].popleft()

    def _get_timestamp(self, msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def flush(self):
        self._process()

def save(sync_data):
    first_topic = next(iter(sync_data))
    ts = sync_data[first_topic].header.stamp
    sec = str(ts.sec).zfill(10)
    nsec = str(ts.nanosec).zfill(10)
    fname = sec + "_" + nsec

    meta = {
        'sec': sec,
        'nanosec': nsec,
        'global_position_latlon': [sync_data['/gnss/fix'].latitude, sync_data['/gnss/fix'].longitude],
        'velocity': sync_data['/gnss/fix_velocity'].twist.linear.x,
        'local_position_xyz': [sync_data['/zed/zed_node/odom'].pose.pose.position.x,
                               sync_data['/zed/zed_node/odom'].pose.pose.position.y,
                               sync_data['/zed/zed_node/odom'].pose.pose.position.z],
        'local_orientation_xyzw': [sync_data['/zed/zed_node/odom'].pose.pose.orientation.x,
                                   sync_data['/zed/zed_node/odom'].pose.pose.orientation.y,
                                   sync_data['/zed/zed_node/odom'].pose.pose.orientation.z,
                                   sync_data['/zed/zed_node/odom'].pose.pose.orientation.w],
        'global_orientation_xyzw': [sync_data['/imu'].orientation.x,
                                    sync_data['/imu'].orientation.y,
                                    sync_data['/imu'].orientation.z,
                                    sync_data['/imu'].orientation.w],
        'angular_speed_xyz': [sync_data['/imu'].angular_velocity.x,
                              sync_data['/imu'].angular_velocity.y,
                              sync_data['/imu'].angular_velocity.z],
        'acceleration_xyz': [sync_data['/imu'].linear_acceleration.x,
                             sync_data['/imu'].linear_acceleration.y,
                             sync_data['/imu'].linear_acceleration.z]
    }
    with open(dirs['meta'] + fname + ".yml", 'w') as f:
        yaml.dump(meta, f)

    rgb_msg = sync_data['/zed/zed_node/rgb/image_rect_color']
    cv_img = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
    cv2.imwrite(dirs['rgb'] + fname + ".png", cv_img)

    dep_pc_msg = sync_data['/zed/zed_node/point_cloud/cloud_registered']
    dep_pc = pypcd.PointCloud.from_msg(dep_pc_msg)
    dep_pc.save_pcd(dirs['depth_cld'] + fname + ".pcd", compression='binary_compressed')
    points3 = dep_pc.pc_data[['x', 'y', 'z']]
    np.save(dirs['depth_cld2'] + fname + ".npy", points3)

    dep_img_msg = sync_data['/zed/zed_node/depth/depth_registered']
    depth = bridge.imgmsg_to_cv2(dep_img_msg, desired_encoding='passthrough')
    np.save(dirs['depth_map'] + fname + ".npy", depth)

    lidar_msg = sync_data['/rslidar_points']
    lidar_pc = pypcd.PointCloud.from_msg(lidar_msg)
    lidar_pc.save_pcd(dirs['lidar'] + fname + ".pcd", compression='binary_compressed')

def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([BAG], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        synchronizer = ApproximateTimeSynchronizer(TOPICS, SLOP_NS, save)

        msg_iter = reader.messages(connections=conns)
        if tqdm is not None:
            msg_iter = tqdm(msg_iter, desc="Processing", unit="msg")

        for conn, ts, raw in msg_iter:
            msg = reader.deserialize(raw, conn.msgtype)
            if not hasattr(msg, 'header'):
                continue
            synchronizer.add(conn.topic, msg)

        synchronizer.flush()

if __name__ == "__main__":
    main()