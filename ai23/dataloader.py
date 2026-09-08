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
from preprocessing.preprocessing_lidar import PreprocessingLidar

config = GlobalConfig()

class KarrDataset(Dataset):
    def __init__(self):
        self.preproc_lidar = PreprocessingLidar()
        self.config = config
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.data_rate = self.config.data_rate
        self.rp1_close = self.config.rp1_close

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

        self.root_path = ["/media/mf/AUTODRIVING-4TB1/UGM Baru/autoriving-oskarnatan/datasetx/2026-04-15_route00"]

        for path in self.root_path:
            path = Path(path)
            preload_path = f"{path}/seq{str(self.seq_len)}_pred{self.pred_len}.npy"
            if os.path.exist(preload_path):
                preload_data = np.load(preload_path, allow_pickle=True)
                self._load_preload(preload_data)
                return

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
                filename = ""
                seq_rgb = []
                seq_raw_pcd = []
                seq_seg_pcd = []
                seq_local_x = []
                seq_local_y = []
                seq_local_heading = []

                # past frames
                for past_idx in range(current_idx - (self.seq_len - 1), current_idx + 1):
                    filename = self.files[past_idx]
                    seq_rgb.append(f"{self.dir_rgb_front}/{filename}.png")
                    seq_raw_pcd.append(f"{self.dir_raw_pcd}/{filename}.pcd")
                    seq_seg_pcd.append(f"{self.dir_seg_pcd}/{filename}.npy")
                preload_data["filename"].append(filename)
                preload_data["rgb"].append(seq_rgb)
                preload_data["raw_pcd"].append(seq_raw_pcd)
                preload_data["seg_pcd"].append(seq_seg_pcd)

                # current frames
                filename = self.files[current_idx]
                with open(f"{self.dir_meta}/{filename}.yml", "r") as read_meta_current:
                    meta_current = yaml.safe.load(read_meta_current)
                seq_local_x.append(meta_current["local_position_xyz"][0])
                seq_local_y.append(meta_current["local_position_xyz"][1])
                local_quaternion = meta_current["local_orientation_xyzw"]
                seq_local_heading.append(euler_from_quaternion(local_quaternion[3], local_quaternion[0], local_quaternion[1], local_quaternion[2], rad=True)[2])
                preload_data["lat"].append(meta_current["global_position_latlon"][0])
                preload_data["lon"].append(meta_current["global_position_latlon"][1])
                velocity = np.abs(meta_current["velocity"])
                preload_data["velocity"].append(velocity)
                preload_data["bearing"].append(69) # TODO: implement bearing switching between latlon_to_yaw and IMU

                about_to_finish = False
                for j in range(2):
                    next_lat_rp = rp_list["route_point"]["latitude"][j]
                    next_lon_rp = rp_list["route_point"]["longitude"][j]
                    dLat_m = (next_lat_rp - meta_current["global_position_latlon"][0]) * 40008000 / 360
                    dLon_m = (next_lon_rp - meta_current['global_position_latlon'][1]) * 40075000 * np.cos(np.radians(meta_current['global_position_latlon'][0])) / 360
                    if j == 0 and np.sqrt(dLat_m**2 + dLon_m**2) <= self.rp1_close and not about_to_finish:
                        if len(rp_list['route_point']['latitude']) > 2:
                            rp_list['route_point']['latitude'].pop(0)
                            rp_list['route_point']['longitude'].pop(0)
                        else:
                            about_to_finish = True
                            rp_list['route_point']['latitude'][0] = rp_list['route_point']['latitude'][-1]
                            rp_list['route_point']['longitude'][0] = rp_list['route_point']['longitude'][-1]
                        next_lat_rp = rp_list['route_point']['latitude'][j]
                        next_lon_rp = rp_list['route_point']['longitude'][j]
                    if j == 0:
                        preload_data["rp1_lat"].append(next_lat_rp)
                        preload_data["rp1_lon"].append(next_lon_rp)
                    else:
                        preload_data["rp2_lon"].append(next_lon_rp)
                        preload_data["rp2_lat"].append(next_lat_rp)

                # future frames
                for future_idx in range((current_idx + self.data_rate), (current_idx + (self.pred_len + 1) * self.data_rate), step=self.data_rate):
                    filename = self.files[future_idx]
                    with open(f"{self.dir_meta}/{filename}.yml", "r") as read_meta_future:
                        meta_future = yaml.safe_load(read_meta_future)
                    seq_local_x.append(meta_current["local_position_xyz"][0])
                    seq_local_y.append(meta_current["local_position_xyz"][1])
                    local_quaternion = meta_current["local_orientation_xyzw"]
                    seq_local_heading.append(euler_from_quaternion(local_quaternion[3], local_quaternion[0], local_quaternion[1], local_quaternion[2], rad=True)[2])
                preload_data["local_x"].append(seq_local_x)
                preload_data["local_y"].append(seq_local_y)
                preload_data["local_heading"].append(seq_local_heading)
            np.save(preload_path, preload_data)
            self._load_preload(preload_data)

    def __len__(self):
        return len(self.raw_pcd)

    def __getitem__(self, index):
        data = dict()
        seq_rgb = []
        seq_raw_pcd = []
        seq_seg_pcd = []
        seq_local_x = []
        seq_local_y = []
        seq_local_heading = []
        seq_rgb = self.rgb[index]
        seq_raw_pcd = self.raw_pcd[index]
        seq_seg_pcd = self.seg_pcd[index]
        seq_local_x = self.local_x[index]
        seq_local_y = self.local_y[index]
        seq_local_heading = self.local_heading[index]

        for i in range(0, self.seq_len):
            raw_pcd = PointCloud.from_path(seq_raw_pcd[i])
            seg_pcd = np.load(seq_seg_pcd[i])
            bev_seg, bev_dep, front_seg, front_dep, _, _ = self.preproc_lidar.gen_bev_front_rear_seg_dep(ptx, pty, ptz, ptseg, gpu=False, bev_multiplier=9, front_multiplier=9, rear_multiplier=9, bs=1, config=self.config)
            data['bev_segs'].append(bev_seg[0])
            data['bev_deps'].append(bev_dep[0])
            data['front_segs'].append(front_seg[0])
            data['front_deps'].append(front_dep[0])

        # current ego robot position dan heading di index 0
        ego_local_x = seq_local_x[0]
        ego_local_y = seq_local_y[0]
        ego_local_heading = seq_local_heading[0]
        # convert waypoint to local coordinate
        data["waypoints"] = []
        for j in range(1, self.pred_len + 1):
            local_waypoint = transform_2d_points(
                np.zeros((1, 3)),
                np.pi/2 - seq_local_heading[j], seq_local_x[j], seq_local_y[j],
                np.pi/2 - ego_local_heading, ego_local_x, ego_local_y
            )
            data["waypoints"].append(tuple(local_waypoint[0, :2]))
        # convert rp1, rp2 from gnss to local coordinates
        # komputasi dari global ke local
        # ref: https://gamedev.stackexchange.com/questions/79765/how-do-i-convert-from-the-global-coordinate-space-to-a-local-space
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
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        return data

    def _load_preload(self, preload_data):
        self.filename += preload_data.item()["filename"]
        self.rgb += preload_data.item()["rgb"]
        self.raw_pcd += preload_data.item()["raw_pcd"]
        self.seg_pcd += preload_data.item()["seg_pcd"]
        self.lat += preload_data.item()["lat"]
        self.lon += preload_data.item()["lon"]
        self.local_x += preload_data.item()["local_x"]
        self.local_y += preload_data.item()["local_y"]
        self.rp1_lat += preload_data.item()["rp1_lat"]
        self.rp1_lon += preload_data.item()["rp1_lon"]
        self.rp2_lat += preload_data.item()["rp2_lat"]
        self.rp2_lon += preload_data.item()["rp2_lon"]
        self.bearing += preload_data.item()["bearing"]
        self.local_heading += preload_data.item()["local_heading"]
        self.velocity += preload_data.item()["velocity"]


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
