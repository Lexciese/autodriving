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
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from a_utility import compute_imu_yaw, magneto_to_yaw, latlon_to_yaw, euler_from_quaternion, transform_2d_points, resizecrop_matrix, crop_matrix, resize_img, cls2one_hot, colorize_depth
from a_utility import hampel_filter, bearing_filter
from config import GlobalConfig
from preprocess_lidar import gen_bev_front_rear_seg_dep_numpy
from a_dataloader import KarrDataset

def _identity_collate(batch):
    return batch[0]

class SplitDataset:
    def __init__(self, dataset: KarrDataset):
        self.dataset = dataset
        self.config = dataset.config

        seq_offset = self.config.seq_len - 1
        future_offset = self.config.pred_len * self.config.hz

        train_indices = []
        val_indices = []
        test_indices = []
        self.test_indices_by_route = {}
        route_offset = 0

        for route in self.config.select_route:
            route_path = self.config.datadir / route

            with open(route_path / "split.yml", "r") as f:
                split_config = yaml.safe_load(f)

            total_frames = len(list((route_path / "meta").glob("*.yml")))
            max_valid_samples = max(0, total_frames - seq_offset - future_offset)

            route_test_indices = []
            splits = [
                ("train", train_indices),
                ("val", val_indices),
                ("test", route_test_indices),
            ]

            for split_key, target_list in splits:
                for region in split_config[split_key]:
                    for start_frame, stop_frame in region:
                        sample_start = max(0, start_frame - seq_offset)
                        sample_stop = min(max_valid_samples, stop_frame - seq_offset)

                        if sample_start < sample_stop:
                            index_range = range(sample_start + route_offset, sample_stop + route_offset)
                            target_list.extend(index_range)

            self.test_indices_by_route[route] = route_test_indices
            test_indices.extend(route_test_indices)
            route_offset += max_valid_samples

        val_dataset = copy.deepcopy(dataset)
        val_dataset.phase = "val"

        self.test_dataset = copy.deepcopy(dataset)
        self.test_dataset.phase = "test"

        self.train_set = Subset(dataset, train_indices)
        self.val_set = Subset(val_dataset, val_indices)
        self.test_set = Subset(self.test_dataset, test_indices)
        self.total_len = len(self.train_set) + len(self.val_set) + len(self.test_set)

    def preload_split(self, num_workers: int = 4):
        zstd = hdf5plugin.Zstd(clevel=3)

        for route in self.config.select_route:
            route_path = self.config.datadir / route
            preload_test_hdf5_path = route_path / f"preload_test_split_seq{self.config.seq_len}_pred{self.config.pred_len}.hdf5"

            if preload_test_hdf5_path.exists():
                continue

            route_test_set = Subset(self.test_dataset, self.test_indices_by_route[route])
            test_loader = DataLoader(
                route_test_set,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_identity_collate,
            )

            tmp_hdf5_path = preload_test_hdf5_path.with_suffix(".hdf5.tmp")
            with h5py.File(tmp_hdf5_path, "w", locking=False) as preload_test_hdf5:
                for idx, data in enumerate(tqdm(test_loader, desc=f"Preloading Test Split ({route})")):
                    idx_node = preload_test_hdf5.create_group(str(idx))

                    idx_node.create_dataset("rgb", data=np.stack(data["rgb"]), chunks=True, **zstd)
                    idx_node.create_dataset("bev_segs", data=np.stack(data["bev_segs"]), chunks=True, **zstd)
                    idx_node.create_dataset("bev_deps", data=np.stack(data["bev_deps"]), chunks=True, **zstd)
                    idx_node.create_dataset("front_segs", data=np.stack(data["front_segs"]), chunks=True, **zstd)
                    idx_node.create_dataset("front_deps", data=np.stack(data["front_deps"]), chunks=True, **zstd)
                    idx_node.create_dataset("waypoints", data=np.asarray(data["waypoints"], dtype=np.float32), chunks=True, **zstd)
                    idx_node.create_dataset("rp1", data=np.asarray(data["rp1"], dtype=np.float64), chunks=True, **zstd)
                    idx_node.create_dataset("rp2", data=np.asarray(data["rp2"], dtype=np.float64), chunks=True, **zstd)

                    idx_node.create_dataset("filename", data=data["filename"])
                    idx_node.create_dataset("rp1_lat", data=data["rp1_lat"])
                    idx_node.create_dataset("rp2_lat", data=data["rp2_lat"])
                    idx_node.create_dataset("rp1_lon", data=data["rp1_lon"])
                    idx_node.create_dataset("rp2_lon", data=data["rp2_lon"])
                    idx_node.create_dataset("velocity", data=data["velocity"])
                    idx_node.create_dataset("bearing_robot", data=data["bearing_robot"])
                    idx_node.create_dataset("lat_robot", data=data["lat_robot"])
                    idx_node.create_dataset("lon_robot", data=data["lon_robot"])
            tmp_hdf5_path.replace(preload_test_hdf5_path)

        return self

if __name__ == "__main__":
    config = GlobalConfig()
    split_dataset = SplitDataset(KarrDataset(config)).preload_split()

