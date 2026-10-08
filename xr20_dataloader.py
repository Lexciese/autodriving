from typing import cast
import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pypcd4.pypcd4 import PointCloud
import h5py
import hdf5plugin
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from xr20_utility import compute_imu_yaw, magneto_to_yaw, latlon_to_yaw, euler_from_quaternion, transform_2d_points, resizecrop_matrix, crop_matrix, resize_img, cls2one_hot, colorize_depth
from xr20_utility import hampel_filter, bearing_filter
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

        self.route_list = sorted(p.name for p in config.datadir.iterdir() if p.is_dir())

        if self.config.select_route != "all":
            self.route_list = [self.config.select_route]
            print(f"only route: {self.config.select_route} is selected")

        sample_idx = 0
        for route in self.route_list:
            path = self.config.datadir / route
            self.dir_meta      = path / "meta"             # .yml
            self.dir_rgb       = path / "camera" / "rgb"   # .png
            self.dir_raw_pcd   = path / "lidar" / "cld"    # .pcd
            self.dir_seg_pcd   = path / "lidar" / "seg"    # .npy

            # 3 for ringroad, 3 or 7 for ugm_baru
            history_buff = max(3, self.seq_len, self.config.gap_bearing + 1)
            latlon_buffer = {
                'lat_buf': deque(maxlen=3),
                'lon_buf': deque(maxlen=3),
                'window_size': 3
            }
            bearing_buffer = {
                'sin': deque(maxlen=5),
                'cos': deque(maxlen=5)
            }

            self.lidar_hdf5_path = path / "lidar" / "lidar_bev_front_seg_dep.hdf5"
            self.file_hdf5 = None
            lidar_hdf5 = h5py.File(self.lidar_hdf5_path, "a")
            frames_grp = lidar_hdf5.require_group("frames")

            preload_path = path / f"seq{str(self.seq_len)}_pred{self.pred_len}_w{latlon_buffer['window_size']}.npy"
            if preload_path.exists():
                self.preload_data = np.load(preload_path, allow_pickle=True)
                self._load_preload(self.preload_data)
                return

            self.files = sorted(p.stem for p in self.dir_meta.glob("*.yml"))
            self.len_files = len(self.files)

            with open(path / f"{route}_routepoint_list.yml", "r") as rp_listx:
                rp_list = yaml.safe_load(rp_listx)
                #assign end point sebagai route terakhir
                rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
                rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])

            # Past: [current_idx - seq_len + 1, current_idx]                -> rgb, raw_pcd, seg_pcd
            # Current: self.files[current_idx]                              -> local position & heading, latlon, velocity
            # Future: current_idx + data_rate, current_idx + 2*data_rate    -> local position & heading
            for current_idx in tqdm(range((self.seq_len - 1), (self.len_files - self.pred_len * self.data_rate)), desc="Preload Process", dynamic_ncols=True):
                filename = ""
                seq_rgb = []
                seq_raw_pcd = []
                seq_seg_pcd = []
                seq_local_x = []
                seq_local_y = []
                seq_local_heading = []

                # past frames
                seq_len_idx = 0
                for past_idx in range(current_idx - (self.seq_len - 1), current_idx + 1):
                    filename = self.files[past_idx]
                    seq_rgb.append(self.dir_rgb / f"{filename}.png")
                    seq_raw_pcd.append(self.dir_raw_pcd / f"{filename}.pcd")
                    seq_seg_pcd.append(self.dir_seg_pcd / f"{filename}.npy")
                    frame_key = f"{sample_idx:06d}:{seq_len_idx:06d}"
                    if filename not in frames_grp:
                        raw_pcd = PointCloud.from_path(str(self.dir_raw_pcd / f"{filename}.pcd"))
                        seg_pcd = np.load(self.dir_seg_pcd / f"{filename}.npy")
                        ptx = np.array(raw_pcd.pc_data['y']) * -1
                        pty = np.array(raw_pcd.pc_data['z'])
                        ptz = np.array(raw_pcd.pc_data['x'])
                        ptseg = np.array(seg_pcd[:, 0])
                        
                        bev_seg, bev_dep, front_seg, front_dep, _, _ = gen_bev_front_rear_seg_dep_numpy(
                            ptx, pty, ptz, ptseg, cfg=self.config
                        )
                        
                        frame_node = frames_grp.create_group(filename)
                        frame_node.create_dataset("bev_seg", data=bev_seg, chunks=True, **hdf5plugin.Zstd(clevel=3))
                        frame_node.create_dataset("bev_dep", data=bev_dep, chunks=True, **hdf5plugin.Zstd(clevel=3))
                        frame_node.create_dataset("front_seg", data=front_seg, chunks=True, **hdf5plugin.Zstd(clevel=3))
                        frame_node.create_dataset("front_dep", data=front_dep, chunks=True, **hdf5plugin.Zstd(clevel=3))
                    seq_len_idx += 1
                sample_idx += 1

                self.preload_data["filename"].append(filename)
                self.preload_data["rgb"].append(seq_rgb)
                self.preload_data["raw_pcd"].append(seq_raw_pcd)
                self.preload_data["seg_pcd"].append(seq_seg_pcd)

                # current frames
                filename = self.files[current_idx]
                with open(self.dir_meta / f"{filename}.yml", "r") as read_meta_current:
                    meta_current = yaml.safe_load(read_meta_current)
                seq_local_x.append(meta_current["local_position_xyz"][0])
                seq_local_y.append(meta_current["local_position_xyz"][1])
                local_quaternion = meta_current["local_orientation_xyzw"]
                seq_local_heading.append(euler_from_quaternion(local_quaternion[3], local_quaternion[0], local_quaternion[1], local_quaternion[2], rad=True)[2])
                raw_curr_lat = meta_current["global_position_latlon"][0]
                raw_curr_lon = meta_current["global_position_latlon"][1]
                velocity = np.abs(meta_current["velocity"])

                curr_lat, curr_lon, is_outlier = hampel_filter(raw_curr_lat, raw_curr_lon, latlon_buffer, n_sigmas=3.0)

                raw_mag = meta_current['magnetic_field']
                magneto_x0, magneto_y0 = self.magnetometer_calib["offset"]
                Q = np.array(self.magnetometer_calib["Q"])
                # Apply Offset and Soft-Iron Q Matrix
                scale = 1e6
                x_body = raw_mag[0] * scale # Forward (+X)
                y_body = raw_mag[2] * scale # Left (+Y)
                z_body = raw_mag[1] * scale # Up (+Z)
                mags_offset = np.column_stack([x_body - magneto_x0, y_body - magneto_y0])
                mags_calibrated_xy = mags_offset @ Q.T
                x_cal = mags_calibrated_xy[:, 0]
                y_cal = mags_calibrated_xy[:, 1]
                z_cal = z_body - np.mean(z_body)
                corrected_magneto_bearing = magneto_to_yaw(x_cal, y_cal)[0]
        
                bearing = bearing_filter(corrected_magneto_bearing, bearing_buffer)

                self.preload_data["bearing"].append(bearing)
                self.preload_data["lat"].append(curr_lat)
                self.preload_data["lon"].append(curr_lon)
                self.preload_data["velocity"].append(velocity)

                about_to_finish = False
                for j in range(2):
                    next_lat_rp = rp_list["route_point"]["latitude"][j]
                    next_lon_rp = rp_list["route_point"]["longitude"][j]
                    dLat_m = (next_lat_rp - curr_lat) * 40008000 / 360
                    dLon_m = (next_lon_rp - curr_lon) * 40075000 * np.cos(np.radians(curr_lat)) / 360
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
                        self.preload_data["rp1_lat"].append(next_lat_rp)
                        self.preload_data["rp1_lon"].append(next_lon_rp)
                    else:
                        self.preload_data["rp2_lon"].append(next_lon_rp)
                        self.preload_data["rp2_lat"].append(next_lat_rp)

                # future frames
                for future_idx in range((current_idx + self.data_rate), (current_idx + (self.pred_len + 1) * self.data_rate), self.data_rate):
                    filename = self.files[future_idx]
                    with open(self.dir_meta / f"{filename}.yml", "r") as read_meta_future:
                        meta_future = yaml.safe_load(read_meta_future)
                    seq_local_x.append(meta_future["local_position_xyz"][0])
                    seq_local_y.append(meta_future["local_position_xyz"][1])
                    local_quaternion = meta_future["local_orientation_xyzw"]
                    seq_local_heading.append(euler_from_quaternion(local_quaternion[3], local_quaternion[0], local_quaternion[1], local_quaternion[2], rad=True)[2])
                self.preload_data["local_x"].append(seq_local_x)
                self.preload_data["local_y"].append(seq_local_y)
                self.preload_data["local_heading"].append(seq_local_heading)
            lidar_hdf5.close()
            np.save(preload_path, np.array(self.preload_data, dtype=object), allow_pickle=True)
            self.preload_data = np.load(preload_path, allow_pickle=True)
            self._load_preload(self.preload_data)

    def __len__(self):
        return len(self.raw_pcd)

    def __getitem__(self, index):
        data = dict()
        data['filename'] = self.filename[index]
        data['rgb'] = []
        data['bev_deps'] = []
        data['bev_segs'] = []
        data['front_deps'] = []
        data['front_segs'] = []
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

        if self.file_hdf5 is None:
            self.file_hdf5 = h5py.File(self.lidar_hdf5_path, "r")

        for i in range(0, self.seq_len):
            if self.phase == "test":
                raw_pcd = PointCloud.from_path(str(seq_raw_pcd[i]))
                seg_pcd = np.load(seq_seg_pcd[i])
                ptx = np.array(raw_pcd.pc_data['y']) * -1
                pty = np.array(raw_pcd.pc_data['z'])
                ptz = np.array(raw_pcd.pc_data['x'])
                ptseg = np.array(seg_pcd[:,0])
                bev_seg, bev_dep, front_seg, front_dep, _, _ = gen_bev_front_rear_seg_dep_numpy(ptx, pty, ptz, ptseg, cfg=self.config)
            else: # train or validation
                frame_name = Path(seq_raw_pcd[i]).stem
                bev_seg   = cast(h5py.Dataset, self.file_hdf5[f"frames/{frame_name}/bev_seg"])[:]
                bev_dep   = cast(h5py.Dataset, self.file_hdf5[f"frames/{frame_name}/bev_dep"])[:]
                front_seg = cast(h5py.Dataset, self.file_hdf5[f"frames/{frame_name}/front_seg"])[:]
                front_dep = cast(h5py.Dataset, self.file_hdf5[f"frames/{frame_name}/front_dep"])[:]
                # bev_seg[:], bev_dep[:], front_seg[:], front_dep[:] = 0, 0, 0, 0 # for testing null
            data['bev_segs'].append(bev_seg[0])
            data['bev_deps'].append(bev_dep[0])
            data['front_segs'].append(front_seg[0])
            data['front_deps'].append(front_dep[0])
            data['rgb'].append(np.array(resize_img(cv2.imread(seq_rgb[i]), resize_w=self.config.front_w , resize_h=self.config.front_h)).transpose(2,0,1))

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
        data['rp1_lat'] = self.rp1_lat[index]
        data['rp2_lat'] = self.rp2_lat[index]
        data['rp1_lon'] = self.rp1_lon[index]
        data['rp2_lon'] = self.rp2_lon[index]

        data["velocity"] = self.velocity[index]
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        return data

    def _load_preload(self, preload_data):
        if isinstance(preload_data, np.ndarray):
            preload_data = preload_data.item()
        self.filename += preload_data["filename"]
        self.rgb += preload_data["rgb"]
        self.raw_pcd += preload_data["raw_pcd"]
        self.seg_pcd += preload_data["seg_pcd"]
        self.lat += preload_data["lat"]
        self.lon += preload_data["lon"]
        self.local_x += preload_data["local_x"]
        self.local_y += preload_data["local_y"]
        self.rp1_lat += preload_data["rp1_lat"]
        self.rp1_lon += preload_data["rp1_lon"]
        self.rp2_lat += preload_data["rp2_lat"]
        self.rp2_lon += preload_data["rp2_lon"]
        self.bearing += preload_data["bearing"]
        self.local_heading += preload_data["local_heading"]
        self.velocity += preload_data["velocity"]

class SplitDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.config  = dataset.config

        self.route_list = sorted(p.name for p in self.config.datadir.iterdir() if p.is_dir())
        if self.config.select_route != "all":
            self.route_list = [self.config.select_route]

        train_range = []
        val_range = []
        test_range = []
        for route in self.route_list:
            path = self.config.datadir / route / "split.yml"
            split_yml = yaml.safe_load(open(path, "r"))
            n_regions = split_yml["n_regions"]
            train_split = split_yml["train"]
            val_split = split_yml["val"]
            test_split = split_yml["test"]

            for i in range(n_regions):
                for start, stop in train_split[i]:
                    train_range.extend(range(start, stop))
                for start, stop in val_split[i]:
                    val_range.extend(range(start, stop))
                for start, stop in test_split[i]:
                    test_range.extend(range(start, stop))

        self.train_set = Subset(self.dataset, train_range)
        self.val_set = Subset(self.dataset, val_range)
        self.test_set = Subset(self.dataset, test_range)
        self.total_len = len(self.train_set) + len(self.val_set) + len(self.test_set)


if __name__ == "__main__":
    config = GlobalConfig()
    # dataset = KarrDataset(config)
    # print(len(dataset))
    # subset_dataset = Subset(dataset, list(range(100)))
    # dataloader = DataLoader(subset_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)
    # print(len(dataloader))
    # for batch_idx, batch in enumerate(dataloader):
    #     print(f"Batch {batch_idx} Waypoints Shape/Structure:")
    # iterator = iter(dataloader)
    # first_step = next(iter(iterator))

    split_dataset = SplitDataset(KarrDataset(config))

