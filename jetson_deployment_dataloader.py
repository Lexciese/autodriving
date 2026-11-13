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

class KarrDataset(Dataset):
    def __init__(self, config: GlobalConfig, phase="train"):
        self.config: GlobalConfig = config
        self.phase = phase
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.data_rate = self.config.hz
        self.rp1_close = self.config.rp1_close
        self.magnetometer_calib = self.config.magnetometer_calib

        self.filename = []
        self.rgb = []
        self.raw_pcd = []
        self.seg_pcd = []
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
        self.gnss_heading = []
        self.imu_heading = []
        self.velocity = []

        for route in self.config.select_route:
            route_path = self.config.datadir / route
            self.preload_test_hdf5_path = route_path / f"preload_test_split_seq{self.config.seq_len}_pred{self.config.pred_len}.hdf5"

            self.preload_data = {
                "filename": [],
                "rgb": [],
                "raw_pcd": [],
                "seg_pcd": [],
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
                "velocity": []
            }

    def __len__(self):
        return len(self.raw_pcd)

    def __getitem__(self, index):
        data = dict()

        if not hasattr(self, "hdf5_handles"):
            self.hdf5_handles = {}


        return data


if __name__ == "__main__":
    config = GlobalConfig()

