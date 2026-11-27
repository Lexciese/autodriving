from typing import cast
import cv2
import yaml
import numpy as np
import copy
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pypcd4.pypcd4 import PointCloud
import h5py
import hdf5plugin
from collections import deque
from torch.utils.data import Dataset, Subset

from a_utility import (
    new_yaw_from_quaternion,
    euler_from_quaternion,
    transform_2d_points,
    resize_img,
    lidar_to_histogram_features,
    bearing_filter,
    bearing_biasing,
    hampel_filter,
)
from config import GlobalConfig


LIDAR_HDF5_NAME = "transfuser_lidar_hist_v2.hdf5"
PRELOAD_VERSION = "imu_v2"


def _read_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _load_raw_xyz(pcd_path):
    """Load a PCD and return finite (forward, left, up) points."""
    pcd = PointCloud.from_path(str(pcd_path))
    xyz = np.column_stack([
        pcd.pc_data['x'],
        pcd.pc_data['y'],
        pcd.pc_data['z'],
    ]).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid]


def _register_lidar_to_ego(raw_xyz, past_pose, ego_pose):
    """Transform raw (forward, left, up) points into current (right, forward, up)."""
    past_x, past_y, past_heading = past_pose
    ego_x, ego_y, ego_heading = ego_pose

    forward = raw_xyz[:, 0]
    left = raw_xyz[:, 1]
    up = raw_xyz[:, 2]

    # past local (forward, left) -> global (x, y)
    cos_past = np.cos(past_heading)
    sin_past = np.sin(past_heading)
    global_x = past_x + forward * cos_past - left * sin_past
    global_y = past_y + forward * sin_past + left * cos_past

    # global -> current local (forward, left)
    dx = global_x - ego_x
    dy = global_y - ego_y
    cos_ego = np.cos(ego_heading)
    sin_ego = np.sin(ego_heading)
    ego_forward = dx * cos_ego + dy * sin_ego
    ego_left = -dx * sin_ego + dy * cos_ego

    # histogram helper expects (right, forward, up)
    return np.column_stack([-ego_left, ego_forward, up]).astype(np.float32)


def _bev_hist_from_raw(raw_xyz, past_pose, ego_pose, config):
    """Register a raw point cloud into the current ego frame and build a BEV."""
    points = _register_lidar_to_ego(raw_xyz, past_pose, ego_pose)
    return lidar_to_histogram_features(points, config)


def _cache_key(current_frame, sequence_frame):
    # The histogram of a past frame depends on the current ego pose, so the
    # cache key must include both frames.
    return f"{current_frame}__{sequence_frame}"


def _preload_is_compatible(preload_data):
    required = (
        "filename", "rgb", "raw_pcd", "lat", "lon",
        "local_x", "local_y", "local_heading",
        "past_local_x", "past_local_y", "past_local_heading",
        "rp1_lat", "rp1_lon", "rp2_lat", "rp2_lon",
        "bearing", "velocity",
    )
    return all(key in preload_data for key in required)


class KarrDataset(Dataset):
    def __init__(self, config: GlobalConfig, phase="train"):
        self.config: GlobalConfig = config
        self.phase = phase
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.data_rate = self.config.hz
        self.rp1_close = self.config.rp1_close

        self.filename = []
        self.rgb = []
        self.raw_pcd = []
        self.lat = []
        self.lon = []
        self.local_x = []
        self.local_y = []
        self.rp1_lon = []
        self.rp1_lat = []
        self.rp2_lon = []
        self.rp2_lat = []
        self.bearing = []
        self.local_heading = []
        # Poses of the input sequence, used to register past LiDAR frames.
        self.past_local_x = []
        self.past_local_y = []
        self.past_local_heading = []
        self.velocity = []
        self.route_sample_counts = {}

        for route in self.config.select_route:
            path = self.config.datadir / route
            self.dir_meta = path / "meta"             # .yml
            self.dir_rgb = path / "camera" / "rgb"    # .png
            self.dir_raw_pcd = path / "lidar" / "cld" # .pcd

            self.preload_data = {
                "filename": [],
                "rgb": [],
                "raw_pcd": [],
                "lat": [],
                "lon": [],
                "local_x": [],
                "local_y": [],
                "rp1_lat": [],
                "rp1_lon": [],
                "rp2_lat": [],
                "rp2_lon": [],
                "bearing": [],
                "local_heading": [],
                "past_local_x": [],
                "past_local_y": [],
                "past_local_heading": [],
                "velocity": [],
            }

            lidar_hdf5_path = path / "lidar" / LIDAR_HDF5_NAME
            preload_path = path / (
                f"preload_seq{self.seq_len}_pred{self.pred_len}"
                f"_rp{self.rp1_close}_{PRELOAD_VERSION}.npy"
            )

            if preload_path.exists() and lidar_hdf5_path.exists():
                cached = np.load(preload_path, allow_pickle=True)
                if isinstance(cached, np.ndarray) and cached.shape == ():
                    cached = cached.item()
                if isinstance(cached, dict) and _preload_is_compatible(cached):
                    self.preload_data = cached
                    self._load_preload(cached)
                    self.route_sample_counts[route] = len(cached["filename"])
                    continue

            bearing_buffer = {
                'sin': deque(maxlen=5),
                'cos': deque(maxlen=5)
            }

            gnss_path = path / f"{route}_gnss.csv"
            if gnss_path.exists():
                gnss_df = pd.read_csv(
                    gnss_path, dtype={'timestamp': str}
                )
                gnss_df['timestamp'] = gnss_df['timestamp'].astype(str)
                gnss_lookup = gnss_df.set_index('timestamp')[
                    ['latitude', 'longitude']
                ].to_dict('index')
            else:
                gnss_lookup = None

            latlon_buffer = {
                'lat_buf': deque(maxlen=3),
                'lon_buf': deque(maxlen=3),
                'window_size': 3,
            }

            try:
                imu_bias = _read_yaml(path / f"{route}_imu_sector_bias.yml")
                bearing_bias = imu_bias["bearing_bias_deg"]
            except FileNotFoundError:
                bearing_bias = getattr(self.config, "bearing_bias", [0.0] * 6)

            self.files = sorted(p.stem for p in self.dir_meta.glob("*.yml"))
            self.len_files = len(self.files)

            with open(path / f"{route}_routepoint_list.yml", "r") as rp_listx:
                rp_list = yaml.safe_load(rp_listx)
                # assign end point as the last route point
                rp_list['route_point']['latitude'].append(
                    rp_list['last_point']['latitude']
                )
                rp_list['route_point']['longitude'].append(
                    rp_list['last_point']['longitude']
                )

            lidar_hdf5 = h5py.File(lidar_hdf5_path, "a", locking=False)
            frames_grp = lidar_hdf5.require_group("frames")

            # Past: [current_idx - seq_len + 1, current_idx] -> rgb, raw_pcd
            # Current: self.files[current_idx]                -> local pose, latlon, velocity
            # Future: current_idx + data_rate ...              -> local pose for waypoints
            for current_idx in tqdm(
                range((self.seq_len - 1),
                      (self.len_files - self.pred_len * self.data_rate)),
                desc="Preload Process",
                dynamic_ncols=True,
            ):
                current_frame = self.files[current_idx]
                meta_current = _read_yaml(self.dir_meta / f"{current_frame}.yml")
                curr_x = meta_current["local_position_xyz"][0]
                curr_y = meta_current["local_position_xyz"][1]
                local_quaternion = meta_current["local_orientation_xyzw"]
                curr_heading = euler_from_quaternion(
                    local_quaternion[3],
                    local_quaternion[0],
                    local_quaternion[1],
                    local_quaternion[2],
                    rad=True,
                )[2]
                ego_pose = (curr_x, curr_y, curr_heading)

                seq_rgb = []
                seq_raw_pcd = []
                seq_past_local_x = []
                seq_past_local_y = []
                seq_past_local_heading = []

                # past frames (including the current frame)
                for past_idx in range(
                    current_idx - (self.seq_len - 1), current_idx + 1
                ):
                    sequence_frame = self.files[past_idx]
                    seq_rgb.append(self.dir_rgb / f"{sequence_frame}.png")
                    seq_raw_pcd.append(self.dir_raw_pcd / f"{sequence_frame}.pcd")

                    if past_idx == current_idx:
                        meta_past = meta_current
                    else:
                        meta_past = _read_yaml(
                            self.dir_meta / f"{sequence_frame}.yml"
                        )

                    past_x = meta_past["local_position_xyz"][0]
                    past_y = meta_past["local_position_xyz"][1]
                    past_quaternion = meta_past["local_orientation_xyzw"]
                    past_heading = euler_from_quaternion(
                        past_quaternion[3],
                        past_quaternion[0],
                        past_quaternion[1],
                        past_quaternion[2],
                        rad=True,
                    )[2]
                    seq_past_local_x.append(past_x)
                    seq_past_local_y.append(past_y)
                    seq_past_local_heading.append(past_heading)

                    key = _cache_key(current_frame, sequence_frame)
                    if key not in frames_grp:
                        raw_xyz = _load_raw_xyz(
                            self.dir_raw_pcd / f"{sequence_frame}.pcd"
                        )
                        lidar_processed = _bev_hist_from_raw(
                            raw_xyz,
                            (past_x, past_y, past_heading),
                            ego_pose,
                            self.config,
                        )
                        frame_node = frames_grp.create_group(key)
                        frame_node.create_dataset(
                            "lidar_processed",
                            data=lidar_processed,
                            chunks=True,
                            **hdf5plugin.Zstd(clevel=3),
                        )

                self.preload_data["filename"].append(current_frame)
                self.preload_data["rgb"].append(seq_rgb)
                self.preload_data["raw_pcd"].append(seq_raw_pcd)
                self.preload_data["past_local_x"].append(seq_past_local_x)
                self.preload_data["past_local_y"].append(seq_past_local_y)
                self.preload_data["past_local_heading"].append(
                    seq_past_local_heading
                )

                velocity = np.abs(meta_current["velocity"])

                gnss_entry = (
                    gnss_lookup.get(str(current_frame))
                    if gnss_lookup is not None
                    else None
                )
                if gnss_entry is not None:
                    curr_lat = gnss_entry["latitude"]
                    curr_lon = gnss_entry["longitude"]
                else:
                    # Fall back to the raw meta GNSS when the cleaned CSV or
                    # the timestamp is unavailable.
                    raw_lat = meta_current["global_position_latlon"][0]
                    raw_lon = meta_current["global_position_latlon"][1]
                    curr_lat, curr_lon, _ = hampel_filter(
                        raw_lat, raw_lon, latlon_buffer, n_sigmas=3.0
                    )

                global_quaternion = meta_current["global_orientation_xyzw"]
                global_yaw = new_yaw_from_quaternion(
                    global_quaternion[3],
                    global_quaternion[0],
                    global_quaternion[1],
                    global_quaternion[2],
                )
                corrected_bearing = np.deg2rad(
                    bearing_biasing(np.rad2deg(global_yaw), bearing_bias)
                )
                bearing = bearing_filter(corrected_bearing, bearing_buffer)

                self.preload_data["bearing"].append(bearing)
                self.preload_data["lat"].append(curr_lat)
                self.preload_data["lon"].append(curr_lon)
                self.preload_data["velocity"].append(velocity)

                about_to_finish = False
                for j in range(2):
                    next_lat_rp = rp_list["route_point"]["latitude"][j]
                    next_lon_rp = rp_list["route_point"]["longitude"][j]
                    dLat_m = (next_lat_rp - curr_lat) * 40008000 / 360
                    dLon_m = (
                        (next_lon_rp - curr_lon)
                        * 40075000
                        * np.cos(np.radians(curr_lat))
                        / 360
                    )
                    if (
                        j == 0
                        and np.sqrt(dLat_m**2 + dLon_m**2) <= self.rp1_close
                        and not about_to_finish
                    ):
                        if len(rp_list['route_point']['latitude']) > 2:
                            rp_list['route_point']['latitude'].pop(0)
                            rp_list['route_point']['longitude'].pop(0)
                        else:
                            about_to_finish = True
                            rp_list['route_point']['latitude'][0] = (
                                rp_list['route_point']['latitude'][-1]
                            )
                            rp_list['route_point']['longitude'][0] = (
                                rp_list['route_point']['longitude'][-1]
                            )
                        next_lat_rp = rp_list['route_point']['latitude'][j]
                        next_lon_rp = rp_list['route_point']['longitude'][j]
                    if j == 0:
                        self.preload_data["rp1_lat"].append(next_lat_rp)
                        self.preload_data["rp1_lon"].append(next_lon_rp)
                    else:
                        self.preload_data["rp2_lon"].append(next_lon_rp)
                        self.preload_data["rp2_lat"].append(next_lat_rp)

                # current + future local poses, used for waypoint labels
                seq_local_x = [curr_x]
                seq_local_y = [curr_y]
                seq_local_heading = [curr_heading]
                for future_idx in range(
                    (current_idx + self.data_rate),
                    (current_idx + (self.pred_len + 1) * self.data_rate),
                    self.data_rate,
                ):
                    future_frame = self.files[future_idx]
                    meta_future = _read_yaml(
                        self.dir_meta / f"{future_frame}.yml"
                    )
                    seq_local_x.append(meta_future["local_position_xyz"][0])
                    seq_local_y.append(meta_future["local_position_xyz"][1])
                    future_quaternion = meta_future["local_orientation_xyzw"]
                    seq_local_heading.append(
                        euler_from_quaternion(
                            future_quaternion[3],
                            future_quaternion[0],
                            future_quaternion[1],
                            future_quaternion[2],
                            rad=True,
                        )[2]
                    )
                self.preload_data["local_x"].append(seq_local_x)
                self.preload_data["local_y"].append(seq_local_y)
                self.preload_data["local_heading"].append(seq_local_heading)

            lidar_hdf5.close()
            np.save(
                preload_path,
                np.array(self.preload_data, dtype=object),
                allow_pickle=True,
            )
            self.preload_data = np.load(preload_path, allow_pickle=True)
            cached = self.preload_data
            if isinstance(cached, np.ndarray) and cached.shape == ():
                cached = cached.item()
            self.route_sample_counts[route] = len(cached["filename"])
            self._load_preload(self.preload_data)

    def __len__(self):
        return len(self.raw_pcd)

    def _get_hdf5(self, pcd_path):
        route_dir = Path(str(pcd_path)).parents[2]
        route_hdf5_path = route_dir / "lidar" / LIDAR_HDF5_NAME
        if not route_hdf5_path.exists():
            return None
        if not hasattr(self, "hdf5_handles"):
            self.hdf5_handles = {}
        if route_hdf5_path not in self.hdf5_handles:
            self.hdf5_handles[route_hdf5_path] = h5py.File(
                route_hdf5_path, "r", locking=False
            )
        return self.hdf5_handles[route_hdf5_path]

    def __getitem__(self, index):
        data = dict()
        data['filename'] = self.filename[index]
        data['rgb'] = []
        data['pcd'] = []
        seq_rgb = self.rgb[index]
        seq_raw_pcd = self.raw_pcd[index]
        seq_local_x = self.local_x[index]
        seq_local_y = self.local_y[index]
        seq_local_heading = self.local_heading[index]
        seq_past_local_x = self.past_local_x[index]
        seq_past_local_y = self.past_local_y[index]
        seq_past_local_heading = self.past_local_heading[index]

        ego_x = seq_local_x[0]
        ego_y = seq_local_y[0]
        ego_heading = seq_local_heading[0]
        ego_pose = (ego_x, ego_y, ego_heading)
        current_frame = self.filename[index]

        for i in range(0, self.seq_len):
            pcd_path = Path(str(seq_raw_pcd[i]))
            sequence_frame = pcd_path.stem
            key = _cache_key(current_frame, sequence_frame)

            lidar_processed = None
            if self.phase != "test":
                h5_file = self._get_hdf5(pcd_path)
                if h5_file is not None:
                    dataset_path = f"frames/{key}/lidar_processed"
                    dataset = h5_file.get(dataset_path)
                    if dataset is not None:
                        lidar_processed = cast(h5py.Dataset, dataset)[:]

            if lidar_processed is None:
                raw_xyz = _load_raw_xyz(pcd_path)
                lidar_processed = _bev_hist_from_raw(
                    raw_xyz,
                    (
                        seq_past_local_x[i],
                        seq_past_local_y[i],
                        seq_past_local_heading[i],
                    ),
                    ego_pose,
                    self.config,
                )

            data['pcd'].append(lidar_processed)

            image = cv2.imread(str(seq_rgb[i]))
            if image is None:
                raise FileNotFoundError(f"Could not read image {seq_rgb[i]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_img(
                image,
                resize_w=self.config.front_w,
                resize_h=self.config.front_h,
            )
            data['rgb'].append(np.array(image).transpose(2, 0, 1))

        # current ego position and heading is index 0
        data["waypoints"] = []
        for j in range(1, self.pred_len + 1):
            local_waypoint = transform_2d_points(
                np.zeros((1, 3)),
                np.pi / 2 - seq_local_heading[j],
                seq_local_x[j],
                seq_local_y[j],
                np.pi / 2 - ego_heading,
                ego_x,
                ego_y,
            )
            data["waypoints"].append(tuple(local_waypoint[0, :2]))

        # convert rp1, rp2 from global to local coordinates
        bearing_robot = self.bearing[index]
        lat_robot = self.lat[index]
        lon_robot = self.lon[index]
        R_matrix = np.array([
            [np.cos(bearing_robot), -np.sin(bearing_robot)],
            [np.sin(bearing_robot),  np.cos(bearing_robot)],
        ])
        dLat1_m = (self.rp1_lat[index] - lat_robot) * 40008000 / 360
        dLon1_m = (
            (self.rp1_lon[index] - lon_robot)
            * 40075000
            * np.cos(np.radians(lat_robot))
            / 360
        )
        dLat2_m = (self.rp2_lat[index] - lat_robot) * 40008000 / 360
        dLon2_m = (
            (self.rp2_lon[index] - lon_robot)
            * 40075000
            * np.cos(np.radians(lat_robot))
            / 360
        )
        data['rp1'] = tuple(R_matrix.T.dot(np.array([dLon1_m, dLat1_m])))
        data['rp2'] = tuple(R_matrix.T.dot(np.array([dLon2_m, dLat2_m])))
        data['rp1_lat'] = self.rp1_lat[index]
        data['rp2_lat'] = self.rp2_lat[index]
        data['rp1_lon'] = self.rp1_lon[index]
        data['rp2_lon'] = self.rp2_lon[index]

        data["velocity"] = self.velocity[index]
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        # horizontal flip augmentation: image, BEV, waypoints, route points
        if self.phase == "train" and random.random() > 0.5:
            for i in range(self.seq_len):
                data['pcd'][i] = np.flip(data['pcd'][i], axis=-1).copy()
                data['rgb'][i] = np.flip(data['rgb'][i], axis=-1).copy()

            data["waypoints"] = [(-wp[0], wp[1]) for wp in data["waypoints"]]
            data['rp1'] = (-data['rp1'][0], data['rp1'][1])
            data['rp2'] = (-data['rp2'][0], data['rp2'][1])

        return data

    def _load_preload(self, preload_data):
        if isinstance(preload_data, np.ndarray):
            preload_data = preload_data.item()
        self.filename += preload_data["filename"]
        self.rgb += preload_data["rgb"]
        self.raw_pcd += preload_data["raw_pcd"]
        self.lat += preload_data["lat"]
        self.lon += preload_data["lon"]
        self.local_x += preload_data["local_x"]
        self.local_y += preload_data["local_y"]
        self.past_local_x += preload_data["past_local_x"]
        self.past_local_y += preload_data["past_local_y"]
        self.past_local_heading += preload_data["past_local_heading"]
        self.rp1_lat += preload_data["rp1_lat"]
        self.rp1_lon += preload_data["rp1_lon"]
        self.rp2_lat += preload_data["rp2_lat"]
        self.rp2_lon += preload_data["rp2_lon"]
        self.bearing += preload_data["bearing"]
        self.local_heading += preload_data["local_heading"]
        self.velocity += preload_data["velocity"]


class SplitDataset:
    def __init__(self, dataset: KarrDataset):
        self.dataset = dataset
        self.config = dataset.config

        seq_offset = self.config.seq_len - 1
        future_offset = self.config.pred_len * self.config.hz

        train_indices = []
        val_indices = []
        test_indices = []
        route_offset = 0

        for route in self.config.select_route:
            route_path = self.config.datadir / route

            with open(route_path / "split.yml", "r") as handle:
                split_config = yaml.safe_load(handle)

            total_frames = len(list((route_path / "meta").glob("*.yml")))
            max_valid_samples = self.dataset.route_sample_counts.get(
                route,
                max(0, total_frames - seq_offset - future_offset),
            )

            splits = [
                ("train", train_indices),
                ("val", val_indices),
                ("test", test_indices),
            ]

            for split_key, target_list in splits:
                for region in (split_config.get(split_key) or []):
                    for start_frame, stop_frame in region:
                        # split.yml ranges are inclusive 0-based frame indices
                        sample_start = max(0, start_frame - seq_offset)
                        sample_stop = min(
                            max_valid_samples,
                            stop_frame + 1 - seq_offset,
                        )
                        if sample_start < sample_stop:
                            target_list.extend(
                                range(
                                    sample_start + route_offset,
                                    sample_stop + route_offset,
                                )
                            )

            route_offset += max_valid_samples

        val_dataset = copy.copy(dataset)
        val_dataset.phase = "val"
        if hasattr(val_dataset, "hdf5_handles"):
            val_dataset.hdf5_handles = {}

        test_dataset = copy.copy(dataset)
        test_dataset.phase = "test"
        if hasattr(test_dataset, "hdf5_handles"):
            test_dataset.hdf5_handles = {}

        self.train_set = Subset(dataset, train_indices)
        self.val_set = Subset(val_dataset, val_indices)
        self.test_set = Subset(test_dataset, test_indices)
        self.total_len = (
            len(self.train_set) + len(self.val_set) + len(self.test_set)
        )


if __name__ == "__main__":
    config = GlobalConfig()
    split_dataset = SplitDataset(KarrDataset(config))
