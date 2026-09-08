import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from pypcd4 import PointCloud
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from ai23.utility.utility import latlon_to_yaw, euler_from_quaternion, transform_2d_points, resizecrop_matrix, crop_matrix, cls2one_hot, colorize_depth
from ai23.config import GlobalConfig

config = GlobalConfig()

class KarrDataset(Dataset):
    def __init__(self):
        self.config = config
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.data_rate = self.config.data_rate
        self.rp1_close = self.config.rp1_close

        self.rgb = []
        self.seg = []
        self.pcd = []
        self.lon = []
        self.lat = []
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

        self.root_path = ["/media/mf/AUTODRIVING-4TB1/UGM Baru/autoriving-oskarnatan/datasetx/2026-04-15_route00"]

        for path in self.root_path:
            path = Path(path)
            self.dir_meta      = path / "meta"             # .yml
            self.dir_rgb       = path / "camera" / "rgb"   # .png
            self.dir_raw_pcd   = path / "lidar" / "cld"    # .pcd
            self.dir_seg_pcd   = path / "lidar" / "seg"    # .npy

            self.files = os.listdir(self.dir_meta)
            self.files.sort()
            self.files = [os.path.splitext(filename)[0] for filename in self.files] # remove extension string
            self.len_files = len(self.files)

            with open(path / "routepoint_list.yml", "r") as rp_listx:
                rp_list = yaml.safe_load(rp_listx)
                #assign end point sebagai route terakhir
                rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
                rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])
            
            # Past: [current_idx - seq_len + 1, current_idx]
            # Current: self.files[current_idx]
            # Future: current_idx + data_rate, current_idx + 2*data_rate,
            for current_idx in range((self.seq_len - 1), (self.len_files - self.pred_len * self.data_rate)):
                # past frames
                for past_idx in range(current_idx - (self.seq_len - 1), current_idx + 1):
                    pass

                # current frames
                temp = self.files[current_idx]

                # future frames
                for future_idx in range((current_idx + self.data_rate), (current_idx + (self.pred_len + 1) * self.data_rate), step=self.data_rate):
                    pass


        
        

    def __len__(self):
        return len(self.rgb)
    
    def __getitem__(self, index):
        data = dict()
        data["rgbs"] = []
        data['segs'] = []
        data['pcd_xs'] = []
        data['pcd_zs'] = []
        seq_rgbs = self.rgb[index]
        seq_segs = self.seg[index]
        seq_pcds = self.pcd[index]
        seq_local_xs = self.local_x[index]
        seq_local_ys = self.local_y[index]
        seq_local_headings = self.local_heading[index]

        for i in range(0, self.seq_len):
            rgb_img = cv2.imread(seq_rgbs[i])
            rgb_crop = crop_matrix(rgb_img, resize=self.config.scale, crop=self.config.crop_roi)
            rgb_transpose = rgb_crop.transpose(2, 0, 1)
            data["rgbs"].append(torch.from_numpy(np.array(rgb_transpose)))

            seg_img = cv2.imread(seq_segs[i])
            seg_crop = crop_matrix(seg_img, resize=self.config.scale, crop=self.config.crop_roi)
            seg_onehot = cls2one_hot(seg_crop, n_class=self.config.n_class)
            data["segs"].append(torch.from_numpy(np.array(seg_onehot)))

            pcd_raw = np.load(seq_pcds[i], allow_pickle=True)
            pcd_clean = np.nan_to_num(pcd_raw, nan=40.0, posinf=40.0, neginf=0.3)
            pcd_clean = colorize_depth(pcd_clean)
            pcd_crop = crop_matrix(pcd_clean, resize=config.scale, D3=False, crop=self.config.crop_roi)
            pcd_transpose = pcd_crop.transpose(2, 0, 1)
            data["pcd_xs"].append(torch.from_numpy(np.array(pcd_transpose[0:1, :, :])))
            data["pcd_zs"].append(torch.from_numpy(np.array(pcd_transpose[2:3, :, :])))

        ego_local_x = seq_local_xs[0]
        ego_local_y = seq_local_ys[0]
        ego_local_heading = seq_local_headings[0]
        # convert waypoint to local coordinate
        data["waypoints"] = []
        for j in range(1, self.pred_len + 1):
            local_waypoint = transform_2d_points(
                np.zeros((1, 3)),
                np.pi/2 - seq_local_headings[j], seq_local_xs[j], seq_local_ys[j],
                np.pi/2 - ego_local_heading, ego_local_x, ego_local_y
            )
            data["waypoints"].append(tuple(local_waypoint[0, :2]))
        # convert rp1, rp2 from gnss to local coordinates
        bearing_robot = self.bearing[index]
        lat_robot = self.lat[index]
        lon_robot = self.lon[index]
        R_matrix = np.array([
            [np.cos(bearing_robot), -np.sin(bearing_robot)],
            [np.sin(bearing_robot),  np.cos(bearing_robot)]
        ])
        dLat1_m = (self.rp1_lat[index] - lat_robot) * 40008000 / 360
        dLon1_m = (self.rp1_lon[index] - lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360
        dLat2_m = (self.rp2_lat[index] - lat_robot) * 40008000 / 360
        dLon2_m = (self.rp2_lon[index] - lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360
        data['rp1'] = tuple(R_matrix.T.dot(np.array([dLon1_m, dLat1_m])))
        data['rp2'] = tuple(R_matrix.T.dot(np.array([dLon2_m, dLat2_m])))

        data["velocity"] = self.velocity[index]

        return data
        

if __name__ == "__main__":
    dataset = KarrDataset()
    print(len(dataset))
    subset_dataset = Subset(dataset, list(range(100)))
    dataloader = DataLoader(subset_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)
    print(len(dataloader))
    # for batch_idx, batch in enumerate(dataloader):
    #     print(batch_idx)
    # iterator = iter(dataloader)
    # first_step = next(iter(iterator))
    # print(first_step)

    # for batch in dataloader:
    #     print(batch)