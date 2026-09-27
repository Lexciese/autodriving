import sys
import yaml
import itertools
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GlobalConfig

config = GlobalConfig()

BAG = Path("/media/mf/AUTODRIVING-4TB1/7 NOV RINGROAD/rosbag2_2025_11_07-15_20_20/rosbag2_2025_11_07-15_20_20_0.mcap")
SLOP_NS = 150_000_000  # 0.15 s
QUEUE_SIZE = 50
OVERWRITE = True

TOPICS = [
    '/gnss/fix',
    '/gnss/fix_velocity',
    '/zed/zed_node/odom',
    '/imu',
    '/magnetometer',
    '/zed/zed_node/rgb/image_rect_color',
    '/zed/zed_node/point_cloud/cloud_registered',
    '/zed/zed_node/depth/depth_registered',
    '/rslidar_points'
]


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
        stamps = []
        topic_list = list(self.topics)

        for t in topic_list:
            queue = self.queues[t]
            topic_stamps = []

            for s in queue.keys():
                stamp_delta = abs(s - latest_stamp)
                if stamp_delta > self.slop_ns:
                    continue
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


def main():
    if config.select_route == "all":
        raise ValueError("append_magnetic_field.py requires a single select_route, not 'all'")

    meta_path = config.datadir / config.select_route / "meta"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta directory not found: {meta_path}")

    stats = {'updated': 0, 'skipped': 0, 'not_found': 0}

    def sync_callback(sync_data):
        first_topic = next(iter(sync_data))
        ts = sync_data[first_topic].header.stamp
        sec = str(ts.sec).zfill(10)
        nsec = str(ts.nanosec).zfill(10)
        fname = f"{sec}_{nsec}.yml"
        meta_file = meta_path / fname

        if not meta_file.exists():
            stats['not_found'] += 1
            return

        meta = yaml.safe_load(meta_file.read_text())
        if 'magnetic_field' in meta and not OVERWRITE:
            stats['skipped'] += 1
            return

        mag_msg = sync_data['/magnetometer']
        meta['magnetic_field'] = [
            float(mag_msg.magnetic_field.x),
            float(mag_msg.magnetic_field.y),
            float(mag_msg.magnetic_field.z),
        ]
        meta['magnetic_field_covariance'] = [
            float(x) for x in mag_msg.magnetic_field_covariance
        ]

        with open(meta_file, 'w') as f:
            yaml.dump(meta, f, sort_keys=False)

        stats['updated'] += 1

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([BAG], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        synchronizer = ApproximateTimeSynchronizer(TOPICS, SLOP_NS, sync_callback, QUEUE_SIZE)

        total = sum(c.msgcount for c in conns if hasattr(c, 'msgcount')) or None
        msg_iter = reader.messages(connections=conns)
        
        if tqdm is not None:
            msg_iter = tqdm(msg_iter, total=total, desc="Backfilling Meta Files", unit="msg")

        for conn, ts, raw in msg_iter:
            msg = reader.deserialize(raw, conn.msgtype)
            if not hasattr(msg, 'header'):
                continue
            synchronizer.add(conn.topic, msg)

    print(f"\nFinished: updated={stats['updated']} skipped={stats['skipped']} missing_file={stats['not_found']}")


if __name__ == "__main__":
    main()