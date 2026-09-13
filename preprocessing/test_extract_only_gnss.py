from pathlib import Path
import yaml
from datetime import date
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from preprocessing.config import GlobalConfig

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

configx = GlobalConfig()
BAG = Path("/home/lexciese/Dev/ikaz-labs/gnss_serial/rosbag2_2026_09_13-15_18_47/rosbag2_2026_09_13-15_18_47_0.mcap")
DATADIR = configx.datadir
PREFIX = str(date.today()) + "_route00"
SLOP_NS = 150_000_000  # 0.15 s
QUEUE_SIZE = 50

# Only GNSS topics retained
TOPICS = [
    '/gnss/fix',
    '/gnss/fix_velocity'
]

meta_dir = Path(DATADIR + PREFIX + "/meta/")
meta_dir.mkdir(parents=True, exist_ok=True)


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
        import itertools
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


def save_gnss(sync_data):
    first_topic = next(iter(sync_data))
    ts = sync_data[first_topic].header.stamp
    sec = str(ts.sec).zfill(10)
    nsec = str(ts.nanosec).zfill(10)
    fname = sec + "_" + nsec

    gnss_meta = {
        'sec': sec,
        'nanosec': nsec,
        'global_position_latlon': [
            sync_data['/gnss/fix'].latitude,
            sync_data['/gnss/fix'].longitude
        ],
        'velocity': sync_data['/gnss/fix_velocity'].twist.linear.x
    }

    with open(meta_dir / f"{fname}.yml", 'w') as f:
        yaml.dump(gnss_meta, f)


def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([BAG], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        synchronizer = ApproximateTimeSynchronizer(TOPICS, SLOP_NS, save_gnss, QUEUE_SIZE)

        total = 0
        for c in conns:
            try:
                total += c.msgcount
            except AttributeError:
                total = None
                break

        msg_iter = reader.messages(connections=conns)
        if tqdm is not None:
            msg_iter = tqdm(msg_iter, total=total, desc="Processing GNSS", unit="msg")

        for conn, ts, raw in msg_iter:
            msg = reader.deserialize(raw, conn.msgtype)
            if not hasattr(msg, 'header'):
                continue
            synchronizer.add(conn.topic, msg)


if __name__ == "__main__":
    main()
