from pathlib import Path
import yaml
import numpy as np
import cv2
import itertools
from datetime import date
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pypcd4 import PointCloud, Encoding
from cv_bridge import CvBridge
from preprocessing.config import GlobalConfig

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

configx = GlobalConfig()
BAG = Path("/media/mf/AUTODRIVING-4TB1/UGM Baru/rosbag2_2025_11_05-11_00_19/rosbag2_2025_11_05-11_00_19_0.mcap")
DATADIR = configx.datadir
PREFIX = str(date.today()) + "_route00"
SLOP_NS = 150_000_000 # 0.15 s
QUEUE_SIZE = 50

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
    def __init__(self, topics, slop_ns, callback, queue_size=100):
        self.topics = topics
        self.slop_ns = slop_ns
        self.callback = callback
        self.queue_size = queue_size
        self.queues = {t: {} for t in topics}

    def _get_timestamp(self, msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def add(self, topic, msg):
        if topic not in self.queues:
            return

        stamp = self._get_timestamp(msg)
        my_queue = self.queues[topic]

        my_queue[stamp] = msg

        while len(my_queue) > self.queue_size:
            del my_queue[min(my_queue.keys())]

        self._process(stamp)

    def _process(self, latest_stamp):
        # Sort and leave only reasonable stamps for synchronization
        stamps = []
        topic_list = list(self.topics)

        for t in topic_list:
            queue = self.queues[t]
            topic_stamps = []

            for s in queue.keys():
                stamp_delta = abs(s - latest_stamp)
                if stamp_delta > self.slop_ns:
                    continue  # far over the slop
                topic_stamps.append((s, stamp_delta))

            if not topic_stamps:
                return
            topic_stamps.sort(key=lambda x: x[1])
            stamps.append(topic_stamps)

        for vv in itertools.product(*[[s[0] for s in ts] for ts in stamps]):
            vv = list(vv)
            if (max(vv) - min(vv)) < self.slop_ns:
                valid = True
                for t, s in zip(topic_list, vv):
                    if s not in self.queues[t]:
                        valid = False
                        break

                if not valid:
                    continue
                sync_msgs = {t: self.queues[t][s] for t, s in zip(topic_list, vv)}

                self.callback(sync_msgs)

                for t, s in zip(topic_list, vv):
                    del self.queues[t][s]

                break

    def flush(self):
        pass

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
    dep_pc = PointCloud.from_msg(dep_pc_msg)
    dep_pc.save(dirs['depth_cld'] + fname + ".pcd", encoding=Encoding.BINARY_COMPRESSED)
    points3 = dep_pc.pc_data[['x', 'y', 'z']]
    np.save(dirs['depth_cld2'] + fname + ".npy", points3)

    dep_img_msg = sync_data['/zed/zed_node/depth/depth_registered']
    depth = bridge.imgmsg_to_cv2(dep_img_msg, desired_encoding='passthrough')
    np.save(dirs['depth_map'] + fname + ".npy", depth)

    lidar_msg = sync_data['/rslidar_points']
    lidar_pc = PointCloud.from_msg(lidar_msg)
    lidar_pc = lidar_pc.numpy(["x", "y", "z", "intensity"])
    xyz_mask = ~np.isnan(lidar_pc[:, :3]).any(axis=1)
    lidar_pc = lidar_pc[xyz_mask]
    lidar_pc = PointCloud.from_xyzi_points(lidar_pc)
    lidar_pc.save(dirs['lidar'] + fname + ".pcd", encoding=Encoding.BINARY_COMPRESSED)

def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([BAG], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        synchronizer = ApproximateTimeSynchronizer(TOPICS, SLOP_NS, save, QUEUE_SIZE)

        total = 0
        for c in conns:
            try:
                total += c.msgcount
            except AttributeError:
                total = None
                break

        msg_iter = reader.messages(connections=conns)
        if tqdm is not None:
            msg_iter = tqdm(msg_iter, total=total, desc="Processing", unit="msg")

        for conn, ts, raw in msg_iter:
            msg = reader.deserialize(raw, conn.msgtype)
            if not hasattr(msg, 'header'):
                continue
            synchronizer.add(conn.topic, msg)

if __name__ == "__main__":
    main()
