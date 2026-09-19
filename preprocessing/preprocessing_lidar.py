import numpy as np
import yaml
import cv2
import torch
import os
from pathlib import Path
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from pypcd4 import PointCloud
from pypcd4.pointcloud2 import sensor_msgs__msg__PointCloud2
from preprocessing.polarseg.network.BEV_Unet import BEV_Unet
from preprocessing.polarseg.network.ptBEV import ptBEVnet
from preprocessing.data_util import preproc_spherical, colorize_seg, colorize_logdepth

from preprocessing.preprocessing import Preprocessing
from preprocessing.config import GlobalConfig

class PreprocessingLidar(Preprocessing):
    def __init__(self, config: GlobalConfig, use_tensor=False):
        super().__init__("lidar")
        self.config: GlobalConfig = config
        self.use_tensor = use_tensor
        self.pcd = None

        if self.use_tensor == False:
            self.grid_size = np.asarray(self.config.grid_size)
            self.max_volume_space = np.asarray(self.config.max_volume_space)
            self.min_volume_space = np.asarray(self.config.min_volume_space)
        else:
            self.grid_size = torch.from_numpy(np.asarray(self.config.grid_size)).to(self.config.gpu_device, dtype=self.config.dtype)
            self.max_volume_space = torch.from_numpy(np.asarray(self.config.max_volume_space)).to(self.config.gpu_device, dtype=self.config.dtype)
            self.min_volume_space = torch.from_numpy(np.asarray(self.config.min_volume_space)).to(self.config.gpu_device, dtype=self.config.dtype)
        self.intervals = (self.max_volume_space - self.min_volume_space) / (self.grid_size-1)

        self.BEV_model=BEV_Unet(n_class=self.config.n_class_kitti-1, n_height=self.config.grid_size[2], input_batch_norm=True, dropout=0.5, circular_padding=True)
        self.polarseg = ptBEVnet(self.BEV_model, pt_model='pointnet', grid_size=self.config.grid_size, fea_dim=9, max_pt_per_encode=256, out_pt_fea_dim=512, kernal_size=1, pt_selection='random', fea_compre=self.config.grid_size[2])
        self.polarseg.load_state_dict(torch.load(self.config.polarseg_weight_path))
        self.polarseg.to(self.config.gpu_device)
        self.polarseg.eval()
        self.predict_labels = None

    def set_input(self, input):
        if isinstance(input, (str, os.PathLike)) and os.path.isfile(input):
            self.pcd = PointCloud.from_path(input)
            return
        if isinstance(input, sensor_msgs__msg__PointCloud2):
            self.pcd = PointCloud.from_msg(input)
            return

    def process(self):
        with torch.no_grad():
            pcd_x = self.pcd.pc_data['x']
            pcd_y = self.pcd.pc_data['y']
            pcd_z = self.pcd.pc_data['z']
            if 'intensity' in self.pcd.pc_data.dtype.names:
                pcd_i = self.pcd.pc_data['intensity']
            else:
                pcd_i = np.ones(pcd_x.shape[0], dtype=np.float32)
            in_pcd = np.column_stack((pcd_x, pcd_y, pcd_z, pcd_i)).astype('float32')
            valid_mask = np.isfinite(in_pcd).all(axis=1)
            in_pcd = in_pcd[valid_mask]
            if self.use_tensor:
                in_pcd = torch.from_numpy(in_pcd).to(self.config.gpu_device, dtype=self.config.dtype)

            # preprocess
            grid_ind, pt_fea = self._preproc_spherical(in_pcd, use_tensor=self.use_tensor)
            if self.use_tensor == False:
                pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.config.gpu_device) for i in pt_fea]
                grid_ind_ten = [torch.from_numpy(i[:, :2]).to(self.config.gpu_device) for i in grid_ind]
                predict_labels = self.polarseg(pt_fea_ten, grid_ind_ten)
                predict_labels = torch.argmax(predict_labels,1).type(torch.uint8)
                predict_labels = predict_labels.cpu().detach().numpy()
                predict_labels = predict_labels[0, grid_ind[0][:,0], grid_ind[0][:,1], grid_ind[0][:,2]] + 1
                # Convert to tensor for BEV generation
                ptseg_ten = torch.tensor(predict_labels).to(self.config.gpu_device, dtype=self.config.dtype)
                predict_labels = np.expand_dims(predict_labels, axis=1)
                self.predict_labels = predict_labels
            else:
                predict_labels = self.polarseg([pt_fea],[grid_ind[:,:2]])
                predict_labels = torch.argmax(predict_labels, 1)
                ptseg_ten = predict_labels[0, grid_ind[:,0], grid_ind[:,1], grid_ind[:, 2]] + 1
                predict_labels_np = np.expand_dims(ptseg_ten.cpu().detach().numpy(), axis=1)
                self.predict_labels = predict_labels_np

            if self.use_tensor == False:
                pcd_coords = torch.tensor(in_pcd[:, :3]).to(self.config.gpu_device, dtype=self.config.dtype)
            else:
                pcd_coords = in_pcd[:, :3]  # tensor

            if self.config.lidar_sensor == "rs32":
                ptx_ten = pcd_coords[:, 1] * -1
                pty_ten = pcd_coords[:, 2]
                ptz_ten = pcd_coords[:, 0]
            else:
                ptx_ten = pcd_coords[:, 0]
                pty_ten = pcd_coords[:, 1]
                ptz_ten = pcd_coords[:, 2]

            self.bev_seg, self.bev_dep, self.front_seg, self.front_dep, self.rear_seg, self.rear_dep = self.gen_bev_front_rear_seg_dep(ptx_ten, pty_ten, ptz_ten, ptseg_ten, use_tensor=self.use_tensor)

            self.bev_segcol = colorize_seg(self.bev_seg.cpu().detach().numpy(), self.config.SEG_CLASSES['colors'])
            self.bev_depcol = colorize_logdepth(self.bev_dep.cpu().detach().numpy())
            self.front_segcol = colorize_seg(self.front_seg.cpu().detach().numpy(), self.config.SEG_CLASSES['colors'])
            self.front_depcol = colorize_logdepth(self.front_dep.cpu().detach().numpy())
            self.rear_segcol = colorize_seg(self.rear_seg.cpu().detach().numpy(), self.config.SEG_CLASSES['colors'])
            self.rear_depcol = colorize_logdepth(self.rear_dep.cpu().detach().numpy())

    def get_output(self, as_image=True, to_file=True):
        # Save generated images if paths are provided
        if to_file and hasattr(self, 'paths'):
            np.save(str(self.paths['seg']), self.predict_labels)
            cv2.imwrite(str(self.paths['bev_seg']), self.bev_segcol)
            cv2.imwrite(str(self.paths['bev_dep']), self.bev_depcol)
            cv2.imwrite(str(self.paths['front_seg']), self.front_segcol)
            cv2.imwrite(str(self.paths['front_dep']), self.front_depcol)
            cv2.imwrite(str(self.paths['rear_seg']), self.rear_segcol)
            cv2.imwrite(str(self.paths['rear_dep']), self.rear_depcol)
        elif as_image == False and to_file == False:
            return self.bev_seg, self.bev_dep, self.self.front_seg, self.front_dep, self.rear_seg, self.rear_dep
        elif as_image == True and to_file == False:
            return self.bev_segcol, self.bev_depcol, self.front_segcol, self.front_depcol, self.rear_segcol, self.rear_depcol

    def _cart2polar(self, input, use_tensor=False):
        if use_tensor == False:
            rho = np.sqrt(input[:,0]**2 + input[:,1]**2)
            phi = np.arctan2(input[:,1], input[:,0])
            return np.stack((rho, phi, input[:,2]), axis=1)
        else:
            rho = torch.sqrt(input[:,0]**2 + input[:,1]**2)
            phi = torch.atan2(input[:,1], input[:,0])
            return torch.stack((rho, phi, input[:,2]), dim=1)

    def _preproc_spherical(self, pcd, use_tensor=False):
        if use_tensor == False:
            xyz_polar = self._cart2polar(pcd[:,:3], False)
            # normalize intensity
            sig = np.clip(np.squeeze(pcd[:,3])/self.config.max_intensity, 0.0, 1.0)
            # get grid index
            crop_range  = self.max_volume_space - self.min_volume_space
            intervals   = crop_range / (self.grid_size - 1)
            xyz_clip = np.clip(xyz_polar, self.min_volume_space, self.max_volume_space)
            grid_ind    = (np.floor((xyz_clip - self.min_volume_space) / intervals)).astype(np.int32)

            # center data on each vocel for PTnet
            voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + self.min_volume_space
            return_xyz = xyz_polar - voxel_centers
            return_xyz = np.concatenate((return_xyz, xyz_polar, pcd[:,:2]), axis=1)
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
            return [grid_ind], [return_fea]
        else:
            xyz_polar = self._cart2polar(pcd[:,:3], True)
            sig = torch.clip(pcd[:,3]/self.config.max_intensity, 0.0, 1.0)
            # get grid index
            xyz_clip = torch.clip(xyz_polar, self.min_volume_space, self.max_volume_space)
            grid_ind = torch.floor((xyz_clip-self.min_volume_space)/self.intervals)
            # center data on each voxel for PTnet
            voxel_centers = (grid_ind + 0.5)*self.intervals + self.min_volume_space
            return_xyz = xyz_polar - voxel_centers
            return_xyz = torch.cat((return_xyz, xyz_polar, pcd[:,:2]), dim=1)
            return_fea = torch.cat((return_xyz, sig[:,None]), dim=1)
            return grid_ind.long(), return_fea

    def gen_bev_front_rear_seg_dep(self, ptx, pty, ptz, ptseg, use_tensor=False, config: GlobalConfig | None = None, bs=None):
        cfg: GlobalConfig = config if config is not None else self.config
        if bs == None:
            bs = cfg.bs
        total_pts = len(ptx)

        if use_tensor == False:
            ptx = np.asarray(ptx, dtype=np.float32).ravel()
            pty = np.asarray(pty, dtype=np.float32).ravel()
            ptz = np.asarray(ptz, dtype=np.float32).ravel()
            ptseg = np.asarray(ptseg, dtype=np.int32).ravel()

            # Batch index for each point (correctly sized)
            ptn = np.repeat(np.arange(bs, dtype=np.int32), total_pts // bs)
            # BEV projection
            # BEV uses X (forward) and Z (height) axes; coordinate normalization.
            # X: map from [-lid_cover_area_lr, lid_cover_area_lr] to [0, lidbev_w-1]
            # Z: map from [lid_cover_area_rf[0], lid_cover_area_rf[1]] to [0, lidbev_h-1] (with offset so that lowest height becomes 0)

            z_offset = cfg.lid_cover_area_rf[0]
            z_range = cfg.lid_cover_area_rf[1] - cfg.lid_cover_area_rf[0]
            ptz_bev = ptz - z_offset

            bev_x = np.floor((ptx + cfg.lid_cover_area_lr) * (cfg.lidbev_w - 1) / (2 * cfg.lid_cover_area_lr)).astype(np.int32)
            bev_z = np.floor((ptz_bev * (1 - cfg.lidbev_h) / z_range) + (cfg.lidbev_h - 1)).astype(np.int32)

            # BEV segmentation
            valid_bev = (bev_x >= 0) & (bev_x <= cfg.lidbev_w - 1) & (bev_z >= 0) & (bev_z <= cfg.lidbev_h - 1)
            
            # Filter point indices FIRST before calculating depth
            v_ptn = ptn[valid_bev]
            v_seg = ptseg[valid_bev]
            v_bz  = bev_z[valid_bev]
            v_bx  = bev_x[valid_bev]

            bev_seg = np.zeros((bs, cfg.n_class_kitti, cfg.lidbev_h, cfg.lidbev_w), dtype=np.float32)
            bev_seg[v_ptn, v_seg, v_bz, v_bx] = 1.0

            # BEV depth (logarithmic encoding)
            # Depth is mapped linearly from [dep_min, dep_max] to [1, 10],
            # then transformed logarithmically: log_d = -log(linear_d) + 1.
            # This gives higher resolution for near objects.

            # Radial distance calculated ONLY for valid BEV points
            v_d_lidar = np.sqrt(ptx[valid_bev]**2 + pty[valid_bev]**2 + ptz[valid_bev]**2)
            linear_d = np.clip(cfg.bev_multiplier * (v_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, a_min=1.0, a_max=10.0)
            log_d = -np.log(linear_d) + 1.0   # ranges from 1 (near) to ~ -1.3 (far)

            bev_dep = np.zeros((bs, 1, cfg.lidbev_h, cfg.lidbev_w), dtype=np.float32)
            bev_dep[v_ptn, 0, v_bz, v_bx] = log_d

            # Front and rear projections (to image‑like coordinates)
            # Front: horizontal angle = atan2(-z, x)  (points looking forward)
            # Rear:  horizontal angle = atan2( z, x)  (points looking backward)
            # Vertical angle = atan2(y, sqrt(x^2+z^2))
            # Angles are then scaled by angular resolutions and shifted to pixel coordinates.

            h_res_rad = cfg.h_res_rad
            v_res_rad = cfg.v_res_rad

            # Horizontal angles (radians)
            front_ang_h = np.arctan2(-ptz, ptx)          # front
            rear_ang_h  = np.arctan2( ptz, ptx)          # rear

            # Vertical angle
            ang_v = np.arctan2(pty, np.sqrt(ptx**2 + ptz**2))

            # Convert to pixel coordinates
            # Theoretical minimum x (horizontal) based on ±360° FOV
            x_min = -360.0 / cfg.h_res / 2
            front_x = np.floor(front_ang_h / h_res_rad - x_min).astype(np.int32)
            rear_x  = np.floor(rear_ang_h  / h_res_rad - x_min).astype(np.int32)

            y_min = cfg.v_fov[0] / cfg.v_res
            y_max = int(cfg.v_fov_total / cfg.v_res)
            y = np.floor(-1.0 * ((ang_v / v_res_rad - y_min) - y_max)).astype(np.int32)

            # Front segmentation
            valid_front = (front_x >= 0) & (front_x <= cfg.lidfront_w - 1) & (y >= 0) & (y <= cfg.lidfront_h - 1)
            f_ptn = ptn[valid_front]
            f_seg = ptseg[valid_front]
            f_y   = y[valid_front]
            f_x   = front_x[valid_front]

            front_seg = np.zeros((bs, cfg.n_class_kitti, cfg.lidfront_h, cfg.lidfront_w), dtype=np.float32)
            front_seg[f_ptn, f_seg, f_y, f_x] = 1.0

            # Front depth
            f_d_lidar = np.sqrt(ptx[valid_front]**2 + pty[valid_front]**2 + ptz[valid_front]**2)
            linear_d_front = np.clip(cfg.front_multiplier * (f_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, a_min=1.0, a_max=10.0)
            log_d_front = -np.log(linear_d_front) + 1.0

            front_dep = np.zeros((bs, 1, cfg.lidfront_h, cfg.lidfront_w), dtype=np.float32)
            front_dep[f_ptn, 0, f_y, f_x] = log_d_front

            # Rear segmentation
            valid_rear = (rear_x >= 0) & (rear_x <= cfg.lidfront_w - 1) & (y >= 0) & (y <= cfg.lidfront_h - 1)
            r_ptn = ptn[valid_rear]
            r_seg = ptseg[valid_rear]
            r_y   = y[valid_rear]
            r_x   = rear_x[valid_rear]

            rear_seg = np.zeros((bs, cfg.n_class_kitti, cfg.lidfront_h, cfg.lidfront_w), dtype=np.float32)
            rear_seg[r_ptn, r_seg, r_y, r_x] = 1.0

            # Rear depth
            r_d_lidar = np.sqrt(ptx[valid_rear]**2 + pty[valid_rear]**2 + ptz[valid_rear]**2)
            linear_d_rear = np.clip(cfg.rear_multiplier * (r_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, a_min=1.0, a_max=10.0)
            log_d_rear = -np.log(linear_d_rear) + 1.0

            rear_dep = np.zeros((bs, 1, cfg.lidfront_h, cfg.lidfront_w), dtype=np.float32)
            rear_dep[r_ptn, 0, r_y, r_x] = log_d_rear

        else:
            ptx = ptx.ravel().float()
            pty = pty.ravel().float()
            ptz = ptz.ravel().float()
            ptseg = ptseg.ravel().long()

            # Batch index for each point (correctly sized)
            ptn = torch.arange(bs, device=cfg.gpu_device, dtype=torch.long).repeat_interleave(total_pts // bs)
            # BEV projection
            # BEV uses X (forward) and Z (height) axes; coordinate normalization.
            # X: map from [-lid_cover_area_lr, lid_cover_area_lr] to [0, lidbev_w-1]
            # Z: map from [lid_cover_area_rf[0], lid_cover_area_rf[1]] to [0, lidbev_h-1] (with offset so that lowest height becomes 0)

            z_offset = cfg.lid_cover_area_rf[0]
            z_range = cfg.lid_cover_area_rf[1] - cfg.lid_cover_area_rf[0]
            ptz_bev = ptz - z_offset

            bev_x = torch.floor((ptx + cfg.lid_cover_area_lr) * (cfg.lidbev_w - 1) / (2 * cfg.lid_cover_area_lr)).long()
            bev_z = torch.floor((ptz_bev * (1 - cfg.lidbev_h) / z_range) + (cfg.lidbev_h - 1)).long()

            # BEV segmentation
            valid_bev = (bev_x >= 0) & (bev_x <= cfg.lidbev_w - 1) & (bev_z >= 0) & (bev_z <= cfg.lidbev_h - 1)
            
            v_ptn = ptn[valid_bev]
            v_seg = ptseg[valid_bev]
            v_bz  = bev_z[valid_bev]
            v_bx  = bev_x[valid_bev]

            bev_seg = torch.zeros((bs, cfg.n_class_kitti, cfg.lidbev_h, cfg.lidbev_w), dtype=cfg.dtype, device=cfg.gpu_device)
            bev_seg[v_ptn, v_seg, v_bz, v_bx] = 1.0

            # BEV depth (logarithmic encoding)
            # Depth is mapped linearly from [dep_min, dep_max] to [1, 10],
            # then transformed logarithmically: log_d = -log(linear_d) + 1.
            # This gives higher resolution for near objects.

            v_d_lidar = torch.sqrt(ptx[valid_bev]**2 + pty[valid_bev]**2 + ptz[valid_bev]**2)
            linear_d = torch.clamp(cfg.bev_multiplier * (v_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, min=1.0, max=10.0)
            log_d = -torch.log(linear_d) + 1.0   # ranges from 1 (near) to ~ -1.3 (far)

            bev_dep = torch.zeros((bs, 1, cfg.lidbev_h, cfg.lidbev_w), dtype=cfg.dtype, device=cfg.gpu_device)
            bev_dep[v_ptn, 0, v_bz, v_bx] = log_d

            # Front and rear projections (to image‑like coordinates)
            # Front: horizontal angle = atan2(-z, x)  (points looking forward)
            # Rear:  horizontal angle = atan2( z, x)  (points looking backward)
            # Vertical angle = atan2(y, sqrt(x^2+z^2))
            # Angles are then scaled by angular resolutions and shifted to pixel coordinates.

            h_res_rad = cfg.h_res_rad
            v_res_rad = cfg.v_res_rad

            # Horizontal angles (radians)
            front_ang_h = torch.atan2(-ptz, ptx)          # front
            rear_ang_h  = torch.atan2( ptz, ptx)          # rear

            # Vertical angle
            ang_v = torch.atan2(pty, torch.sqrt(ptx**2 + ptz**2))

            # Convert to pixel coordinates
            # Theoretical minimum x (horizontal) based on ±360° FOV
            x_min = -360.0 / cfg.h_res / 2
            front_x = torch.floor(front_ang_h / h_res_rad - x_min).long()
            rear_x  = torch.floor(rear_ang_h  / h_res_rad - x_min).long()

            y_min = cfg.v_fov[0] / cfg.v_res
            y_max = int(cfg.v_fov_total / cfg.v_res)
            y = torch.floor(-1.0 * ((ang_v / v_res_rad - y_min) - y_max)).long()

            # Front segmentation
            valid_front = (front_x >= 0) & (front_x <= cfg.lidfront_w - 1) & (y >= 0) & (y <= cfg.lidfront_h - 1)
            f_ptn = ptn[valid_front]
            f_seg = ptseg[valid_front]
            f_y   = y[valid_front]
            f_x   = front_x[valid_front]

            front_seg = torch.zeros((bs, cfg.n_class_kitti, cfg.lidfront_h, cfg.lidfront_w), dtype=cfg.dtype, device=cfg.gpu_device)
            front_seg[f_ptn, f_seg, f_y, f_x] = 1.0

            # Front depth
            f_d_lidar = torch.sqrt(ptx[valid_front]**2 + pty[valid_front]**2 + ptz[valid_front]**2)
            linear_d_front = torch.clamp(cfg.front_multiplier * (f_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, min=1.0, max=10.0)
            log_d_front = -torch.log(linear_d_front) + 1.0

            front_dep = torch.zeros((bs, 1, cfg.lidfront_h, cfg.lidfront_w), dtype=cfg.dtype, device=cfg.gpu_device)
            front_dep[f_ptn, 0, f_y, f_x] = log_d_front

            # Rear segmentation
            valid_rear = (rear_x >= 0) & (rear_x <= cfg.lidfront_w - 1) & (y >= 0) & (y <= cfg.lidfront_h - 1)
            r_ptn = ptn[valid_rear]
            r_seg = ptseg[valid_rear]
            r_y   = y[valid_rear]
            r_x   = rear_x[valid_rear]

            rear_seg = torch.zeros((bs, cfg.n_class_kitti, cfg.lidfront_h, cfg.lidfront_w), dtype=cfg.dtype, device=cfg.gpu_device)
            rear_seg[r_ptn, r_seg, r_y, r_x] = 1.0

            # Rear depth
            r_d_lidar = torch.sqrt(ptx[valid_rear]**2 + pty[valid_rear]**2 + ptz[valid_rear]**2)
            linear_d_rear = torch.clamp(cfg.rear_multiplier * (r_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, min=1.0, max=10.0)
            log_d_rear = -torch.log(linear_d_rear) + 1.0

            rear_dep = torch.zeros((bs, 1, cfg.lidfront_h, cfg.lidfront_w), dtype=cfg.dtype, device=cfg.gpu_device)
            rear_dep[r_ptn, 0, r_y, r_x] = log_d_rear

        return bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep



def main():
    config = GlobalConfig()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    preproc_lidar = PreprocessingLidar(config, config.use_tensor)

    route_list = os.listdir(config.datadir)
    route_list.sort()
    for route in route_list:
        route = Path(config.datadir) / route
        if os.path.isfile(route):
            continue
        print(route)
        dir_meta = route / "meta"
        dir_lidar = route / "lidar" / "cld"
        dir_lidseg = route / "lidar" / "seg"
        # visualisasi
        dir_lidseg_bev = route / "lidar" / "img" / "bev_seg"
        dir_lidseg_fro = route / "lidar" / "img" / "front_seg"
        dir_lidseg_rea = route / "lidar" / "img" / "rear_seg"
        dir_liddep_bev = route / "lidar" / "img" / "bev_dep"
        dir_liddep_fro = route / "lidar" / "img" / "front_dep"
        dir_liddep_rea = route / "lidar" / "img" / "rear_dep"
        dir_lidseg.mkdir(parents=True, exist_ok=True)
        dir_lidseg_bev.mkdir(parents=True, exist_ok=True)
        dir_lidseg_fro.mkdir(parents=True, exist_ok=True)
        dir_lidseg_rea.mkdir(parents=True, exist_ok=True)
        dir_liddep_bev.mkdir(parents=True, exist_ok=True)
        dir_liddep_fro.mkdir(parents=True, exist_ok=True)
        dir_liddep_rea.mkdir(parents=True, exist_ok=True)
        file_list = os.listdir(dir_meta)
        file_list.sort()

        for file in tqdm(file_list, desc="Processing LiDAR PCD", unit="files"):
            file = file[:-4]
            preproc_lidar.paths = {
                'seg': dir_lidseg / f"{file}.npy",
                'bev_seg': dir_lidseg_bev / f"{file}.png",
                'bev_dep': dir_liddep_bev / f"{file}.png",
                'front_seg': dir_lidseg_fro / f"{file}.png",
                'front_dep': dir_liddep_fro / f"{file}.png",
                'rear_seg': dir_lidseg_rea / f"{file}.png",
                'rear_dep': dir_liddep_rea / f"{file}.png"
            }
            for key, value in preproc_lidar.paths.items():
                if value.exists():
                    preproc_lidar.paths = None
                    break
            if preproc_lidar.paths == None:
                continue

            preproc_lidar.set_input(f"{dir_lidar}/{file}.pcd")
            preproc_lidar.process()
            preproc_lidar.get_output(as_image=True, to_file=True)


if __name__ == "__main__":
    main()
