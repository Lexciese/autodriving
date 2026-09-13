import os
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
import torch

from ai23.config import GlobalConfig
from preprocessing.data_util import plot_sdc_rpwp, plot_lidbev_rpwp, plot_lidfront_rpwp
from ai23.dataloader import KarrDataset


def colorize_seg(sem_map, colmap):
    """
    Colorize multi-channel semantic segmentation map (B, C, H, W) or (C, H, W).
    Returns RGB image (H, W, 3).
    """
    if sem_map.ndim == 3:
        sem_map = np.expand_dims(sem_map, axis=0)  # Convert to (1, C, H, W)

    sem_img = np.zeros((sem_map.shape[2], sem_map.shape[3], 3), dtype=np.uint8)
    idx = np.argmax(sem_map[0], axis=0)

    for cmap in colmap:
        cmap_id = colmap.index(cmap)
        sem_img[np.where(idx == cmap_id)] = cmap

    return sem_img


def colorize_logdepth(depth_map):
    """
    Colorize normalized single-channel depth map (B, C, H, W) or (C, H, W).
    Returns RGB image (H, W, 3).
    """
    if depth_map.ndim == 2:
        depth_map = np.expand_dims(depth_map, axis=(0, 1))
    elif depth_map.ndim == 3:
        depth_map = np.expand_dims(depth_map, axis=0)

    norm_dep = depth_map[0][0]
    logdepth = np.repeat(norm_dep[:, :, np.newaxis], 3, axis=2) * 255.0
    return np.uint8(np.clip(logdepth, 0, 255))


def visualize_dataset_sample(dataset: KarrDataset, index: int, output_path: str = "sample_output.jpg"):
    configx = dataset.config
    sample = dataset[index]

    filenum = sample['filename']
    velocity = sample['velocity'] * 3.6  # m/s to km/h
    bearing_deg = sample['bearing_robot']
    veh_curr_lat = sample['lat_robot']
    veh_curr_lon = sample['lon_robot']

    rp1_local = sample['rp1']
    rp2_local = sample['rp2']
    rp1_lat = sample['rp1_lat']
    rp1_lon = sample['rp1_lon']
    rp2_lat = sample['rp2_lat']
    rp2_lon = sample['rp2_lon']
    waypoints = sample['waypoints']

    bev_seg = sample['bev_segs'][-1]
    bev_dep = sample['bev_deps'][-1]
    front_seg = sample['front_segs'][-1]
    front_dep = sample['front_deps'][-1]

    if isinstance(bev_seg, torch.Tensor):
        bev_seg = bev_seg.cpu().numpy()
        bev_dep = bev_dep.cpu().numpy()
        front_seg = front_seg.cpu().numpy()
        front_dep = front_dep.cpu().numpy()


    colmap = config.SEG_CLASSES['colors']
    # Colorize segmentation and depth maps using custom functions
    lidar_bev_segcol = colorize_seg(bev_seg, colmap)
    lidar_bev_depcol = colorize_logdepth(bev_dep)
    lidar_front_segcol = colorize_seg(front_seg, colmap)
    lidar_front_depcol = colorize_logdepth(front_dep)

    # Convert RGB colorized images to OpenCV BGR format
    lidar_bev_segcol = cv2.cvtColor(lidar_bev_segcol, cv2.COLOR_RGB2BGR)
    lidar_bev_depcol = cv2.cvtColor(lidar_bev_depcol, cv2.COLOR_RGB2BGR)
    lidar_front_segcol = cv2.cvtColor(lidar_front_segcol, cv2.COLOR_RGB2BGR)
    lidar_front_depcol = cv2.cvtColor(lidar_front_depcol, cv2.COLOR_RGB2BGR)

    rgb_front = cv2.imread(dataset.rgb[index][-1])

    rp_lidbev_frame = []
    rp_lidfront_frame = []

    for rp in [rp1_local, rp2_local]:
        bx, by = plot_lidbev_rpwp(configx, rp[0], rp[1])
        fx, fy = plot_lidfront_rpwp(configx, rp[0], rp[1])
        rp_lidbev_frame.append((bx, by))
        rp_lidfront_frame.append((fx, fy))

    # Project Future Waypoints
    wp_lidbev_frame = []
    wp_lidfront_frame = []
    for wp in waypoints:
        bx, by = plot_lidbev_rpwp(configx, wp[0], wp[1])
        fx, fy = plot_lidfront_rpwp(configx, wp[0], wp[1])
        wp_lidbev_frame.append((bx, by))
        wp_lidfront_frame.append((fx, fy))

    lidar_bev_segcol_wprp = lidar_bev_segcol.copy()
    lidar_front_segcol_wprp = lidar_front_segcol.copy()

    for k in range(2):
        bev_color = (255, 255, 255) if k == 0 else (255, 255, 0)
        cv2.circle(lidar_bev_segcol_wprp, (int(rp_lidbev_frame[k][0]), int(rp_lidbev_frame[k][1])), radius=3, color=bev_color, thickness=2)
        cv2.circle(lidar_front_segcol_wprp, (int(rp_lidfront_frame[k][0]), int(rp_lidfront_frame[k][1])), radius=3, color=(255, 255, 255), thickness=2)

    for k in range(len(waypoints)):
        cv2.circle(lidar_bev_segcol_wprp, (int(wp_lidbev_frame[k][0]), int(wp_lidbev_frame[k][1])), radius=2, color=(255, 255, 255), thickness=-1)
        cv2.circle(lidar_front_segcol_wprp, (int(wp_lidfront_frame[k][0]), int(wp_lidfront_frame[k][1])), radius=2, color=(255, 255, 255), thickness=-1)

    left_column_w = 1024

    rgb_h = int(rgb_front.shape[0] * (left_column_w / rgb_front.shape[1]))
    rgb_front_scaled = cv2.resize(rgb_front, (left_column_w, rgb_h), interpolation=cv2.INTER_LINEAR)

    front_lidar_h = int(lidar_front_depcol.shape[0] * (left_column_w / lidar_front_depcol.shape[1]))
    front_dep_scaled = cv2.resize(lidar_front_depcol, (left_column_w, front_lidar_h), interpolation=cv2.INTER_LINEAR)
    front_seg_scaled = cv2.resize(lidar_front_segcol_wprp, (left_column_w, front_lidar_h), interpolation=cv2.INTER_LINEAR)

    left_column = np.concatenate((rgb_front_scaled, front_dep_scaled, front_seg_scaled), axis=0)
    total_h = left_column.shape[0]

    telemetry_lines = [
        ("INPUT", ""),
        (f"Sample Index: {index}", ""),
        (f"File Name: {filenum}", ""),
        (f"Speed: {format(np.round(velocity, 3), '.3f')} km/h", ""),
        (f"Bearing: {format(np.round(bearing_deg, 3), '.3f')} deg", ""),
        (f"Robot Lat: {format(np.round(veh_curr_lat, 6), '.6f')}", ""),
        (f"Robot Lon: {format(np.round(veh_curr_lon, 6), '.6f')}", ""),
        (f"Rp1 Lat: {format(np.round(rp1_lat, 6), '.6f')}", ""),
        (f"Rp1 Lon: {format(np.round(rp1_lon, 6), '.6f')}", ""),
        (f"Rp2 Lat: {format(np.round(rp2_lat, 6), '.6f')}", ""),
        (f"Rp2 Lon: {format(np.round(rp2_lon, 6), '.6f')}", ""),
        ("", ""),
        ("OUTPUT WAYPOINTS", ""),
    ]

    for wp_idx, wp in enumerate(waypoints[:3]):
        txt_wp = f"Wp{wp_idx+1} Loc: x: {format(np.round(wp[0], 3), '.3f')} | y: {format(np.round(wp[1], 3), '.3f')}"
        telemetry_lines.append((txt_wp, ""))

    line_gap = min(22, int(rgb_h / (len(telemetry_lines) + 2)))
    overlay_w = 420
    overlay_h = (len(telemetry_lines) + 1) * line_gap

    overlay_pil = Image.new('RGBA', (overlay_w, overlay_h), (0, 0, 0, 140))
    draw = ImageDraw.Draw(overlay_pil)

    for idx, (line_text, _) in enumerate(telemetry_lines):
        if line_text:
            draw.text((12, 8 + idx * line_gap), line_text, fill=(255, 255, 255, 255))

    left_column_pil = Image.fromarray(cv2.cvtColor(left_column, cv2.COLOR_BGR2RGB)).convert('RGBA')
    left_column_pil.paste(overlay_pil, (10, 10), overlay_pil)
    left_column = cv2.cvtColor(np.array(left_column_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

    bev_stack_raw = np.concatenate((lidar_bev_depcol, lidar_bev_segcol_wprp), axis=0)
    bev_target_w = int(bev_stack_raw.shape[1] * (total_h / bev_stack_raw.shape[0]))
    bev_column = cv2.resize(bev_stack_raw, (bev_target_w, total_h), interpolation=cv2.INTER_LINEAR)

    final_img = np.concatenate((left_column, bev_column), axis=1)

    # Save to disk
    cv2.imwrite(output_path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Sample image successfully generated and saved to {output_path}")


if __name__ == "__main__":
    config = GlobalConfig()
    dataset = KarrDataset(config)

    TARGET_INDEX = 1005
    visualize_dataset_sample(dataset, index=TARGET_INDEX, output_path=f"sample_shot_idx_{TARGET_INDEX}.jpg")