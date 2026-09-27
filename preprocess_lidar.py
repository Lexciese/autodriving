from dataclasses import dataclass
import numpy as np
from typing import Union
import yaml
import cv2
import torch
import os
from pathlib import Path
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from pypcd4.pypcd4 import PointCloud, PathLike
from pypcd4.pointcloud2 import sensor_msgs__msg__PointCloud2
from common.polarseg.network.BEV_Unet import BEV_Unet
from common.polarseg.network.ptBEV import ptBEVnet
from preprocess_util import colorize_seg, colorize_logdepth

from config import GlobalConfig

@dataclass(frozen=True)
class ProjectionOutputs:
    bev_seg: torch.Tensor | np.ndarray
    bev_dep: torch.Tensor | np.ndarray
    front_seg: torch.Tensor | np.ndarray
    front_dep: torch.Tensor | np.ndarray
    rear_seg: torch.Tensor | np.ndarray
    rear_dep: torch.Tensor | np.ndarray

@dataclass(frozen=True)
class ColorizedOutputs:
    bev_segcol: np.ndarray
    bev_depcol: np.ndarray
    front_segcol: np.ndarray
    front_depcol: np.ndarray
    rear_segcol: np.ndarray
    rear_depcol: np.ndarray

def get_pcd(path: PathLike):
    pcd = PointCloud.from_path(path)
    pcd_x = pcd.pc_data['x']
    pcd_y = pcd.pc_data['y']
    pcd_z = pcd.pc_data['z']
    names = pcd.pc_data.dtype.names
    pcd_i = pcd.pc_data['intensity'] if names is not None and 'intensity' in names else np.ones(pcd_x.shape[0], dtype=np.float32)
    in_pcd = np.column_stack((pcd_x, pcd_y, pcd_z, pcd_i)).astype('float32')
    valid_mask = np.isfinite(in_pcd).all(axis=1)
    in_pcd = in_pcd[valid_mask]
    return in_pcd

def compute_cart2polar_numpy(pcd: np.ndarray):
    rho = np.sqrt(pcd[:,0]**2 + pcd[:,1]**2)
    phi = np.arctan2(pcd[:,1], pcd[:,0])
    return np.stack((rho, phi, pcd[:,2]), axis=1)

def compute_cart2polar_tensor(pcd: torch.Tensor):
    rho = torch.sqrt(pcd[:,0]**2 + pcd[:,1]**2)
    phi = torch.atan2(pcd[:,1], pcd[:,0])
    return torch.stack((rho, phi, pcd[:,2]), dim=1)

def preproc_spherical_numpy(
        pcd: np.ndarray, 
        min_volume_space: np.ndarray, 
        max_volume_space: np.ndarray, 
        grid_size: np.ndarray, 
        max_intensity: float
    ):
    xyz_polar = compute_cart2polar_numpy(pcd[:,:3])
    # normalize intensity
    sig = np.clip(np.squeeze(pcd[:,3]) / max_intensity, 0.0, 1.0)
    # get grid index
    crop_range  = max_volume_space - min_volume_space
    intervals   = crop_range / (grid_size - 1)
    xyz_clip = np.clip(xyz_polar, min_volume_space, max_volume_space)
    grid_ind    = (np.floor((xyz_clip - min_volume_space) / intervals)).astype(np.int32)
    # center data on each vocel for PTnet
    voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_volume_space
    return_xyz = xyz_polar - voxel_centers
    return_xyz = np.concatenate((return_xyz, xyz_polar, pcd[:,:2]), axis=1)
    return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
    return [grid_ind], [return_fea]

def preproc_spherical_tensor(
    pcd: torch.Tensor, 
    min_volume_space: torch.Tensor, 
    max_volume_space: torch.Tensor, 
    grid_size: torch.Tensor, 
    max_intensity: float
):
    xyz_polar = compute_cart2polar_tensor(pcd[:,:3])
    # normalize intensity
    sig = torch.clip(pcd[:,3] / max_intensity, 0.0, 1.0)
    # get grid index
    crop_range  = max_volume_space - min_volume_space
    intervals   = crop_range / (grid_size - 1)
    xyz_clip = torch.clip(xyz_polar, min_volume_space, max_volume_space)
    grid_ind = torch.floor((xyz_clip-min_volume_space) / intervals)
    # center data on each voxel for PTnet
    voxel_centers = (grid_ind + 0.5)*intervals + min_volume_space
    return_xyz = xyz_polar - voxel_centers
    return_xyz = torch.cat((return_xyz, xyz_polar, pcd[:,:2]), dim=1)
    return_fea = torch.cat((return_xyz, sig[:,None]), dim=1)
    return grid_ind.long(), return_fea

def gen_bev_front_rear_seg_dep_numpy(
    ptx: np.ndarray,
    pty: np.ndarray,
    ptz: np.ndarray,
    ptseg: np.ndarray,
    cfg: GlobalConfig
):
    ptx = np.asarray(ptx, dtype=np.float32).ravel()
    pty = np.asarray(pty, dtype=np.float32).ravel()
    ptz = np.asarray(ptz, dtype=np.float32).ravel()
    ptseg = np.asarray(ptseg, dtype=np.int32).ravel()
    total_pts = len(ptx)

    # Batch index for each point (correctly sized)
    ptn = np.repeat(np.arange(cfg.bs, dtype=np.int32), total_pts // cfg.bs)
    # BEV projection
    # BEV uses X (forward) and Z (height) axes; coordinate normalization.
    # X: map from [-cover_area_lr, cover_area_lr] to [0, bev_w-1]
    # Z: map from [lid_cover_area_rf[0], lid_cover_area_rf[1]] to [0, bev_h-1] (with offset so that lowest height becomes 0)

    z_offset = cfg.lid_cover_area_rf[0]
    z_range = cfg.lid_cover_area_rf[1] - cfg.lid_cover_area_rf[0]
    ptz_bev = ptz - z_offset

    bev_x = np.floor((ptx + cfg.cover_area_lr) * (cfg.bev_w - 1) / (2 * cfg.cover_area_lr)).astype(np.int32)
    bev_z = np.floor((ptz_bev * (1 - cfg.bev_h) / z_range) + (cfg.bev_h - 1)).astype(np.int32)

    # BEV segmentation
    valid_bev = (bev_x >= 0) & (bev_x <= cfg.bev_w - 1) & (bev_z >= 0) & (bev_z <= cfg.bev_h - 1)
    
    # Filter point indices FIRST before calculating depth
    v_ptn = ptn[valid_bev]
    v_seg = ptseg[valid_bev]
    v_bz  = bev_z[valid_bev]
    v_bx  = bev_x[valid_bev]

    bev_seg = np.zeros((cfg.bs, cfg.n_class_kitti, cfg.bev_h, cfg.bev_w), dtype=np.float32)
    bev_seg[v_ptn, v_seg, v_bz, v_bx] = 1.0

    # BEV depth (logarithmic encoding)
    # Depth is mapped linearly from [dep_min, dep_max] to [1, 10],
    # then transformed logarithmically: log_d = -log(linear_d) + 1.
    # This gives higher resolution for near objects.

    # Radial distance calculated ONLY for valid BEV points
    v_d_lidar = np.sqrt(ptx[valid_bev]**2 + pty[valid_bev]**2 + ptz[valid_bev]**2)
    linear_d = np.clip(cfg.bev_multiplier * (v_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, a_min=1.0, a_max=10.0)
    log_d = -np.log(linear_d) + 1.0   # ranges from 1 (near) to ~ -1.3 (far)

    bev_dep = np.zeros((cfg.bs, 1, cfg.bev_h, cfg.bev_w), dtype=np.float32)
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
    valid_front = (front_x >= 0) & (front_x <= cfg.front_w - 1) & (y >= 0) & (y <= cfg.front_h - 1)
    f_ptn = ptn[valid_front]
    f_seg = ptseg[valid_front]
    f_y   = y[valid_front]
    f_x   = front_x[valid_front]

    front_seg = np.zeros((cfg.bs, cfg.n_class_kitti, cfg.front_h, cfg.front_w), dtype=np.float32)
    front_seg[f_ptn, f_seg, f_y, f_x] = 1.0

    # Front depth
    f_d_lidar = np.sqrt(ptx[valid_front]**2 + pty[valid_front]**2 + ptz[valid_front]**2)
    linear_d_front = np.clip(cfg.front_multiplier * (f_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, a_min=1.0, a_max=10.0)
    log_d_front = -np.log(linear_d_front) + 1.0

    front_dep = np.zeros((cfg.bs, 1, cfg.front_h, cfg.front_w), dtype=np.float32)
    front_dep[f_ptn, 0, f_y, f_x] = log_d_front

    # Rear segmentation
    valid_rear = (rear_x >= 0) & (rear_x <= cfg.front_w - 1) & (y >= 0) & (y <= cfg.front_h - 1)
    r_ptn = ptn[valid_rear]
    r_seg = ptseg[valid_rear]
    r_y   = y[valid_rear]
    r_x   = rear_x[valid_rear]

    rear_seg = np.zeros((cfg.bs, cfg.n_class_kitti, cfg.front_h, cfg.front_w), dtype=np.float32)
    rear_seg[r_ptn, r_seg, r_y, r_x] = 1.0

    # Rear depth
    r_d_lidar = np.sqrt(ptx[valid_rear]**2 + pty[valid_rear]**2 + ptz[valid_rear]**2)
    linear_d_rear = np.clip(cfg.rear_multiplier * (r_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, a_min=1.0, a_max=10.0)
    log_d_rear = -np.log(linear_d_rear) + 1.0

    rear_dep = np.zeros((cfg.bs, 1, cfg.front_h, cfg.front_w), dtype=np.float32)
    rear_dep[r_ptn, 0, r_y, r_x] = log_d_rear
    return bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep

def gen_bev_front_rear_seg_dep_tensor(
    ptx: torch.Tensor,
    pty: torch.Tensor,
    ptz: torch.Tensor,
    ptseg: torch.Tensor,
    cfg: GlobalConfig
):
    ptx = ptx.ravel().float()
    pty = pty.ravel().float()
    ptz = ptz.ravel().float()
    ptseg = ptseg.ravel().long()
    total_pts = len(ptx)
    
    # Batch index for each point (correctly sized)
    ptn = torch.arange(cfg.bs, device=cfg.gpu_device, dtype=torch.long).repeat_interleave(total_pts // cfg.bs)
    # BEV projection
    # BEV uses X (forward) and Z (height) axes; coordinate normalization.
    # X: map from [-cover_area_lr, cover_area_lr] to [0, bev_w-1]
    # Z: map from [lid_cover_area_rf[0], lid_cover_area_rf[1]] to [0, bev_h-1] (with offset so that lowest height becomes 0)
    
    z_offset = cfg.lid_cover_area_rf[0]
    z_range = cfg.lid_cover_area_rf[1] - cfg.lid_cover_area_rf[0]
    ptz_bev = ptz - z_offset
    
    bev_x = torch.floor((ptx + cfg.cover_area_lr) * (cfg.bev_w - 1) / (2 * cfg.cover_area_lr)).long()
    bev_z = torch.floor((ptz_bev * (1 - cfg.bev_h) / z_range) + (cfg.bev_h - 1)).long()
    
    # BEV segmentation
    valid_bev = (bev_x >= 0) & (bev_x <= cfg.bev_w - 1) & (bev_z >= 0) & (bev_z <= cfg.bev_h - 1)
    
    v_ptn = ptn[valid_bev]
    v_seg = ptseg[valid_bev]
    v_bz  = bev_z[valid_bev]
    v_bx  = bev_x[valid_bev]
    
    bev_seg = torch.zeros((cfg.bs, cfg.n_class_kitti, cfg.bev_h, cfg.bev_w), dtype=torch.float32, device=cfg.gpu_device)
    bev_seg[v_ptn, v_seg, v_bz, v_bx] = 1.0
    
    # BEV depth (logarithmic encoding)
    # Depth is mapped linearly from [dep_min, dep_max] to [1, 10],
    # then transformed logarithmically: log_d = -log(linear_d) + 1.
    # This gives higher resolution for near objects.
    
    v_d_lidar = torch.sqrt(ptx[valid_bev]**2 + pty[valid_bev]**2 + ptz[valid_bev]**2)
    linear_d = torch.clamp(cfg.bev_multiplier * (v_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, min=1.0, max=10.0)
    log_d = -torch.log(linear_d) + 1.0   # ranges from 1 (near) to ~ -1.3 (far)
    
    bev_dep = torch.zeros((cfg.bs, 1, cfg.bev_h, cfg.bev_w), dtype=torch.float32, device=cfg.gpu_device)
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
    valid_front = (front_x >= 0) & (front_x <= cfg.front_w - 1) & (y >= 0) & (y <= cfg.front_h - 1)
    f_ptn = ptn[valid_front]
    f_seg = ptseg[valid_front]
    f_y   = y[valid_front]
    f_x   = front_x[valid_front]
    
    front_seg = torch.zeros((cfg.bs, cfg.n_class_kitti, cfg.front_h, cfg.front_w), dtype=torch.float32, device=cfg.gpu_device)
    front_seg[f_ptn, f_seg, f_y, f_x] = 1.0
    
    # Front depth
    f_d_lidar = torch.sqrt(ptx[valid_front]**2 + pty[valid_front]**2 + ptz[valid_front]**2)
    linear_d_front = torch.clamp(cfg.front_multiplier * (f_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, min=1.0, max=10.0)
    log_d_front = -torch.log(linear_d_front) + 1.0
    
    front_dep = torch.zeros((cfg.bs, 1, cfg.front_h, cfg.front_w), dtype=torch.float32, device=cfg.gpu_device)
    front_dep[f_ptn, 0, f_y, f_x] = log_d_front
    
    # Rear segmentation
    valid_rear = (rear_x >= 0) & (rear_x <= cfg.front_w - 1) & (y >= 0) & (y <= cfg.front_h - 1)
    r_ptn = ptn[valid_rear]
    r_seg = ptseg[valid_rear]
    r_y   = y[valid_rear]
    r_x   = rear_x[valid_rear]
    
    rear_seg = torch.zeros((cfg.bs, cfg.n_class_kitti, cfg.front_h, cfg.front_w), dtype=torch.float32, device=cfg.gpu_device)
    rear_seg[r_ptn, r_seg, r_y, r_x] = 1.0
    
    # Rear depth
    r_d_lidar = torch.sqrt(ptx[valid_rear]**2 + pty[valid_rear]**2 + ptz[valid_rear]**2)
    linear_d_rear = torch.clamp(cfg.rear_multiplier * (r_d_lidar - cfg.dep_min) / (cfg.dep_max - cfg.dep_min) + 1, min=1.0, max=10.0)
    log_d_rear = -torch.log(linear_d_rear) + 1.0
    
    rear_dep = torch.zeros((cfg.bs, 1, cfg.front_h, cfg.front_w), dtype=torch.float32, device=cfg.gpu_device)
    rear_dep[r_ptn, 0, r_y, r_x] = log_d_rear
    return bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep

def save_outputs(output_dir_paths: dict[str, Path], labels: np.ndarray, colorized: ColorizedOutputs):
    np.save(str(output_dir_paths['seg']), labels)
    cv2.imwrite(str(output_dir_paths['bev_seg']), colorized.bev_segcol)
    cv2.imwrite(str(output_dir_paths['bev_dep']), colorized.bev_depcol)
    cv2.imwrite(str(output_dir_paths['front_seg']), colorized.front_segcol)
    cv2.imwrite(str(output_dir_paths['front_dep']), colorized.front_depcol)
    cv2.imwrite(str(output_dir_paths['rear_seg']), colorized.rear_segcol)
    cv2.imwrite(str(output_dir_paths['rear_dep']), colorized.rear_depcol)

def colorize_projections(bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep, colors: list):
    bev_seg = bev_seg.cpu().numpy() if isinstance(bev_seg, torch.Tensor) else bev_seg
    bev_dep = bev_dep.cpu().numpy() if isinstance(bev_dep, torch.Tensor) else bev_dep
    front_seg = front_seg.cpu().numpy() if isinstance(front_seg, torch.Tensor) else front_seg
    front_dep = front_dep.cpu().numpy() if isinstance(front_dep, torch.Tensor) else front_dep
    rear_seg = rear_seg.cpu().numpy() if isinstance(rear_seg, torch.Tensor) else rear_seg
    rear_dep = rear_dep.cpu().numpy() if isinstance(rear_dep, torch.Tensor) else rear_dep

    return ColorizedOutputs(
        bev_segcol=colorize_seg(bev_seg, colors),
        bev_depcol=colorize_logdepth(bev_dep),
        front_segcol=colorize_seg(front_seg, colors),
        front_depcol=colorize_logdepth(front_dep),
        rear_segcol=colorize_seg(rear_seg, colors),
        rear_depcol=colorize_logdepth(rear_dep),
    )

class LidarSegmentationPipeline:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.grid_size = torch.from_numpy(np.asarray(self.config.grid_size)).to(self.config.gpu_device, dtype=self.config.dtype)
        self.max_volume_space = torch.from_numpy(np.asarray(self.config.max_volume_space)).to(self.config.gpu_device, dtype=self.config.dtype)
        self.min_volume_space = torch.from_numpy(np.asarray(self.config.min_volume_space)).to(self.config.gpu_device, dtype=self.config.dtype)
        self.model = self._init_model()

    def _init_model(self) -> ptBEVnet:
        bev_model = BEV_Unet(
            n_class=self.config.n_class_kitti - 1,
            n_height=self.config.grid_size[2],
            input_batch_norm=True,
            dropout=0.5,
            circular_padding=True,
        )
        model = ptBEVnet(
            bev_model,
            pt_model='pointnet',
            grid_size=self.config.grid_size,
            fea_dim=9,
            max_pt_per_encode=256,
            out_pt_fea_dim=512,
            kernal_size=1,
            pt_selection='random',
            fea_compre=self.config.grid_size[2],
        )
        model.load_state_dict(torch.load(self.config.polarseg_weight_path))
        model.to(self.config.gpu_device)
        model.eval()
        return model

    def run(self, input_source: PathLike):
        raw_pcd = get_pcd(input_source)
        pcd_tensor = torch.from_numpy(raw_pcd).to(self.config.gpu_device, dtype=self.config.dtype)

        with torch.no_grad():
            grid_ind, pt_fea = preproc_spherical_tensor(
                pcd_tensor,
                self.min_volume_space,
                self.max_volume_space,
                self.grid_size,
                self.config.max_intensity
            )            
            # Forward pass through model
            logits = self.model([pt_fea], [grid_ind[:, :2]])
            ptseg = torch.argmax(logits, 1)[0, grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]] + 1
            
            # Coordinate re-mapping based on sensor orientation
            coords = pcd_tensor[:, :3]
            if self.config.lidar_sensor == "rs32":
                coords = torch.stack([-coords[:, 1], coords[:, 2], coords[:, 0]], dim=1)

            bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep = gen_bev_front_rear_seg_dep_tensor(coords[:, 0], coords[:, 1], coords[:, 2], ptseg, self.config)
            colorized = colorize_projections(bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep, self.config.SEG_CLASSES['colors'])
            
            predict_labels = np.expand_dims(ptseg.cpu().numpy(), axis=1)
            
        return predict_labels, colorized


def main():
    config = GlobalConfig()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    torch.backends.cudnn.benchmark = True

    pipeline = LidarSegmentationPipeline(config)
    routes = sorted([p for p in Path(config.datadir).iterdir() if p.is_dir()])
    if config.select_route != "all":
        routes = sorted([Path(p) for p in config.select_route])
        print(f"only route: {config.select_route} is selected")

    for route in routes:
        print(f"Processing route: {route}")
        dir_meta = route / "meta"
        dir_lidar = route / "lidar" / "cld"

        target_dirs = {
            'seg': route / "lidar" / "seg",
            'bev_seg': route / "lidar" / "img" / "bev_seg",
            'bev_dep': route / "lidar" / "img" / "bev_dep",
            'front_seg': route / "lidar" / "img" / "front_seg",
            'front_dep': route / "lidar" / "img" / "front_dep",
            'rear_seg': route / "lidar" / "img" / "rear_seg",
            'rear_dep': route / "lidar" / "img" / "rear_dep",
        }
        
        for d in target_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        meta_files = sorted(dir_meta.glob("*.yml"))
        
        for meta_file in tqdm(meta_files, desc="Processing PCD Files", unit="file"):
            stem = meta_file.stem
            file_paths = {k: v / f"{stem}.png" if k != 'seg' else v / f"{stem}.npy" for k, v in target_dirs.items()}

            if all(p.exists() for p in file_paths.values()):
                continue

            pcd_file = dir_lidar / f"{stem}.pcd"
            predict_labels, colorized_imgs = pipeline.run(pcd_file)
            save_outputs(file_paths, predict_labels, colorized_imgs)

if __name__ == "__main__":
    main()