from typing import cast
import time
import h5py
import hdf5plugin
import numpy as np
from torch.utils.data import Dataset

from config import GlobalConfig
from preprocess_lidar import GenerateLidSegDep, gen_bev_front_rear_seg_dep_numpy


class KarrDataset(Dataset):
    def __init__(self, config: GlobalConfig, phase="test", test_lidar_from_scratch=True):
        self.config: GlobalConfig = config
        self.phase = phase
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.test_lidar_from_scratch = test_lidar_from_scratch

        self.index_map = []
        for route in self.config.select_route:
            route_path = self.config.datadir / route
            preload_test_hdf5_path = route_path / f"preload_test_split_seq{self.config.seq_len}_pred{self.config.pred_len}.hdf5"

            with h5py.File(preload_test_hdf5_path, "r", locking=False) as preload_test_hdf5:
                keys = sorted(preload_test_hdf5.keys(), key=int)
                if self.test_lidar_from_scratch and len(keys) > 0 and "raw_pcd" not in preload_test_hdf5[keys[0]]:
                    raise KeyError(f"'raw_pcd' not found in {preload_test_hdf5_path}. Regenerate it with preprocess_preload_test_split.py.")

            self.index_map.extend((preload_test_hdf5_path, key) for key in keys)

    def __len__(self):
        return len(self.index_map)

    def _get_lidar_pipeline(self):
        if not hasattr(self, "_lidar_pipeline"):
            self._lidar_pipeline = GenerateLidSegDep(self.config)
        return self._lidar_pipeline

    def __getitem__(self, index):
        data = dict()

        if not hasattr(self, "hdf5_handles"):
            self.hdf5_handles = {}

        preload_test_hdf5_path, key = self.index_map[index]
        if preload_test_hdf5_path not in self.hdf5_handles:
            self.hdf5_handles[preload_test_hdf5_path] = h5py.File(preload_test_hdf5_path, "r", locking=False)
        h5_file = self.hdf5_handles[preload_test_hdf5_path]
        idx_node = h5_file[key]

        filename = idx_node["filename"][()]
        if isinstance(filename, bytes):
            filename = filename.decode()
        data['filename'] = filename
        rgb = cast(h5py.Dataset, idx_node["rgb"])[:]
        data['rgb'] = [rgb[i] for i in range(self.seq_len)]
        data['bev_segs'] = []
        data['bev_deps'] = []
        data['front_segs'] = []
        data['front_deps'] = []

        lidar_elapsed_time = 0.0
        if self.test_lidar_from_scratch:
            lidar_pipeline = self._get_lidar_pipeline()
            raw_pcd_group = idx_node["raw_pcd"]
            for i in range(0, self.seq_len):
                raw_pcd = cast(h5py.Dataset, raw_pcd_group[str(i)])[:]

                lidar_start_time = time.time()
                ptseg = lidar_pipeline.segment(raw_pcd)
                ptx = raw_pcd[:, 1] * -1
                pty = raw_pcd[:, 2]
                ptz = raw_pcd[:, 0]
                bev_seg, bev_dep, front_seg, front_dep, _, _ = gen_bev_front_rear_seg_dep_numpy(ptx, pty, ptz, ptseg, cfg=self.config)
                lidar_elapsed_time += time.time() - lidar_start_time

                data['bev_segs'].append(bev_seg[0])
                data['bev_deps'].append(bev_dep[0])
                data['front_segs'].append(front_seg[0])
                data['front_deps'].append(front_dep[0])
        else:
            bev_segs = cast(h5py.Dataset, idx_node["bev_segs"])[:]
            bev_deps = cast(h5py.Dataset, idx_node["bev_deps"])[:]
            front_segs = cast(h5py.Dataset, idx_node["front_segs"])[:]
            front_deps = cast(h5py.Dataset, idx_node["front_deps"])[:]
            for i in range(0, self.seq_len):
                data['bev_segs'].append(bev_segs[i])
                data['bev_deps'].append(bev_deps[i])
                data['front_segs'].append(front_segs[i])
                data['front_deps'].append(front_deps[i])

        data['lidar_elapsed_time'] = lidar_elapsed_time

        waypoints = cast(h5py.Dataset, idx_node["waypoints"])[:]
        data["waypoints"] = [tuple(waypoint) for waypoint in waypoints]
        data['rp1'] = tuple(cast(h5py.Dataset, idx_node["rp1"])[:])
        data['rp2'] = tuple(cast(h5py.Dataset, idx_node["rp2"])[:])
        data['rp1_lat'] = idx_node["rp1_lat"][()]
        data['rp2_lat'] = idx_node["rp2_lat"][()]
        data['rp1_lon'] = idx_node["rp1_lon"][()]
        data['rp2_lon'] = idx_node["rp2_lon"][()]
        data['velocity'] = idx_node["velocity"][()]
        data['bearing_robot'] = idx_node["bearing_robot"][()]
        data['lat_robot'] = idx_node["lat_robot"][()]
        data['lon_robot'] = idx_node["lon_robot"][()]

        return data


if __name__ == "__main__":
    config = GlobalConfig()
    dataset = KarrDataset(config)
    print(len(dataset))
