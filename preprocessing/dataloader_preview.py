import os
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
import torch

from ai23.config import GlobalConfig
from preprocessing.data_util import plot_sdc_rpwp, plot_lidbev_rpwp, plot_lidfront_rpwp
from ai23.dataloader import KarrDataset


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

    if bev_seg.ndim == 3:
        if bev_seg.shape[0] < bev_seg.shape[1] and bev_seg.shape[0] < bev_seg.shape[2]:
            bev_seg = np.argmax(bev_seg, axis=0) if bev_seg.shape[0] > 1 else bev_seg[0]
        elif bev_seg.shape[2] > 4:
            bev_seg = np.argmax(bev_seg, axis=2)

    if bev_dep.ndim == 3:
        if bev_dep.shape[0] in [1, 3]:
            bev_dep = bev_dep[0] if bev_dep.shape[0] == 1 else np.transpose(bev_dep, (1, 2, 0))

    if front_seg.ndim == 3:
        if front_seg.shape[0] < front_seg.shape[1] and front_seg.shape[0] < front_seg.shape[2]:
            front_seg = np.argmax(front_seg, axis=0) if front_seg.shape[0] > 1 else front_seg[0]
        elif front_seg.shape[2] > 4:
            front_seg = np.argmax(front_seg, axis=2)

    if front_dep.ndim == 3:
        if front_dep.shape[0] in [1, 3]:
            front_dep = front_dep[0] if front_dep.shape[0] == 1 else np.transpose(front_dep, (1, 2, 0))

    bev_seg_u8 = np.uint8(bev_seg / bev_seg.max() * 255.0) if bev_seg.max() > 1 else np.uint8(bev_seg * 255)
    bev_dep_u8 = np.uint8(bev_dep)
    front_seg_u8 = np.uint8(front_seg / front_seg.max() * 255.0) if front_seg.max() > 1 else np.uint8(front_seg * 255)
    front_dep_u8 = np.uint8(front_dep)

    lidar_bev_segcol = cv2.cvtColor(bev_seg_u8, cv2.COLOR_GRAY2BGR) if bev_seg_u8.ndim == 2 else bev_seg_u8[:, :, :3]
    lidar_bev_depcol = cv2.cvtColor(bev_dep_u8, cv2.COLOR_GRAY2BGR) if bev_dep_u8.ndim == 2 else bev_dep_u8[:, :, :3]
    lidar_front_segcol = cv2.cvtColor(front_seg_u8, cv2.COLOR_GRAY2BGR) if front_seg_u8.ndim == 2 else front_seg_u8[:, :, :3]
    lidar_front_depcol = cv2.cvtColor(front_dep_u8, cv2.COLOR_GRAY2BGR) if front_dep_u8.ndim == 2 else front_dep_u8[:, :, :3]

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