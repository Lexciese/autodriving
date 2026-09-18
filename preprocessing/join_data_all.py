import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from preprocessing.data_util import hampel_filter, bearing_filter, resizecrop_matrix, transform_2d_points, plot_lidbev_rpwp, plot_lidfront_rpwp, plot_sdc_rpwp, latlon_to_yaw, euler_from_quaternion
from preprocessing.data_util import PIDController, pid_control

from collections import deque

def normalize_angle_deg(angle):
    return (angle + 180) % 360 - 180

def compute_imu_yaw(q, offset=0.0):
    r = R.from_quat(q)
    # Project the sensor's X-axis (+X Forward) into world horizontal frame
    x_world = r.apply([1, 0, 0])
    # Compute Compass Yaw: arctan2(East, North)
    yaw_rad = np.arctan2(x_world[0], x_world[1])
    return (yaw_rad + offset + np.pi) % (2.0 * np.pi) - np.pi


def build_lut_correction_function(meta_dir, file_list, filtered_lats, filtered_lons, n_bins=360):
    imu_bearings = []
    ref_bearings = []

    prev_lat = filtered_lats[0]
    prev_lon = filtered_lons[0]

    for i in range(len(file_list)):
        with open(os.path.join(meta_dir, file_list[i]), 'r') as f:
            curr_meta = yaml.safe_load(f)

        velocity = np.abs(curr_meta.get("velocity", 0.0))
        curr_lat = filtered_lats[i]
        curr_lon = filtered_lons[i]

        dLat_m = (curr_lat - prev_lat) * 40008000 / 360
        dLon_m = (curr_lon - prev_lon) * 40075000 * np.cos(np.radians(curr_lat)) / 360

        # Sample bearing only when moving to ensure clean reference data
        if np.sqrt(dLat_m**2 + dLon_m**2) >= 1.0 and velocity > 1.5:
            ref_yaw = latlon_to_yaw(curr_lat, curr_lon, prev_lat, prev_lon)
            raw_imu_yaw = compute_imu_yaw(curr_meta['global_orientation_xyzw'])

            ref_bearings.append(ref_yaw)
            imu_bearings.append(raw_imu_yaw)

            prev_lat = curr_lat
            prev_lon = curr_lon

    if len(imu_bearings) < 10:
        return lambda raw_rad: raw_rad

    imu_deg = np.degrees(np.array(imu_bearings))
    ref_deg = np.degrees(np.array(ref_bearings))

    # Calculate shortest angular error: ref - imu
    errors_deg = np.degrees(np.arctan2(np.sin(np.radians(ref_deg - imu_deg)), 
                                       np.cos(np.radians(ref_deg - imu_deg))))

    # Bin error across -180 to 180 degrees
    bin_edges = np.linspace(-180, 180, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    binned_errors = []
    for i in range(n_bins):
        mask = (imu_deg >= bin_edges[i]) & (imu_deg < bin_edges[i+1])
        if np.any(mask):
            binned_errors.append(np.median(errors_deg[mask]))
        else:
            binned_errors.append(np.nan)

    # Fill unmeasured bins via linear interpolation and boundary padding
    s_err = pd.Series(binned_errors).interpolate(method='linear').bfill().ffill()
    lut_errors_deg = s_err.to_numpy()

    def correct_imu_bearing_lut(raw_imu_rad):
        imu_d = np.degrees(raw_imu_rad)
        # Interpolate error dynamically with 360 degree periodicity
        corr_error_d = np.interp(imu_d, bin_centers, lut_errors_deg, period=360.0)
        corrected_d = imu_d + corr_error_d
        corrected_rad = np.radians(corrected_d)
        return (corrected_rad + np.pi) % (2.0 * np.pi) - np.pi

    return correct_imu_bearing_lut


# PID Controller
turn_controller = PIDController(K_P=0.5, K_I=0.25, K_D=0.15, n=15)
speed_controller = PIDController(K_P=1.5, K_I=0.25, K_D=0.5, n=15)

from preprocessing.config import GlobalConfig
configx = GlobalConfig()

# Loop pada semua route
route_list = os.listdir(configx.datadir)
route_list.sort()
for route in route_list:
    if os.path.isfile(configx.datadir + route):  # skip file
        continue
    print(route)

    latlon_buffer = {
        'lat_buf': deque(maxlen=3),
        'lon_buf': deque(maxlen=3),
        'window_size': 3
    }
    bearing_buffer = {
        'sin': deque(maxlen=5),
        'cos': deque(maxlen=5)
    }
    # Paths
    ddir_meta = configx.datadir + route + "/meta/"
    ddir_rgb_front = configx.datadir + route + "/camera/rgb/"

    ddir_lidar = configx.datadir + route + "/lidar/img/"
    ddir_lidseg_bev = ddir_lidar + "bev_seg/"
    ddir_lidseg_fro = ddir_lidar + "front_seg/"
    ddir_liddep_bev = ddir_lidar + "bev_dep/"
    ddir_liddep_fro = ddir_lidar + "front_dep/"

    join_img_folder = configx.datadir + route + "/join_img/all_img/"
    os.makedirs(join_img_folder, exist_ok=True)
    if Path(configx.datadir + route + "/join_img/" + route + ".avi").exists():
        print(f"{join_img_folder} is already generated")
        continue

    # Load route points
    with open(configx.datadir + route + "/" + route + "_routepoint_list.yml", 'r') as rp_listx:
        rp_list = yaml.safe_load(rp_listx)
        rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
        rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])

    file_list = os.listdir(ddir_meta)
    file_list.sort()

    # Pre-extract raw coordinates across entire route
    all_raw_lats = []
    all_raw_lons = []
    for f_name in file_list:
        with open(ddir_meta + f_name, 'r') as f_meta:
            m_data = yaml.safe_load(f_meta)
            all_raw_lats.append(m_data['global_position_latlon'][0])
            all_raw_lons.append(m_data['global_position_latlon'][1])

    # Run Hampel Filter across the full sequence first
    s_lats = pd.Series(all_raw_lats)
    s_lons = pd.Series(all_raw_lons)
    
    # Apply Hampel outlier removal to series
    med_lats = s_lats.rolling(window=5, min_periods=1, center=True).median()
    mad_lats = (s_lats - med_lats).abs().rolling(window=5, min_periods=1, center=True).median()
    outliers_lats = (s_lats - med_lats).abs() > (3.0 * 1.4826 * mad_lats)
    filtered_lats = s_lats.copy()
    filtered_lats[outliers_lats] = med_lats[outliers_lats]

    med_lons = s_lons.rolling(window=5, min_periods=1, center=True).median()
    mad_lons = (s_lons - med_lons).abs().rolling(window=5, min_periods=1, center=True).median()
    outliers_lons = (s_lons - med_lons).abs() > (3.0 * 1.4826 * mad_lons)
    filtered_lons = s_lons.copy()
    filtered_lons[outliers_lons] = med_lons[outliers_lons]

    filtered_lats = filtered_lats.tolist()
    filtered_lons = filtered_lons.tolist()

    # Construct LUT for IMU Bearing Correction using Hampel-filtered coordinates
    correct_imu_lut = build_lut_correction_function(
        ddir_meta, file_list, filtered_lats, filtered_lons, n_bins=360
    )

    out_video = None

    seq_len = configx.seq_len
    pred_len = configx.pred_len
    data_rate = configx.hz
    len_files = len(file_list)
    # Loop frames
    for current_idx in tqdm(range((seq_len - 1), (len_files - pred_len * data_rate)), "Generating Frames", len(file_list)):
        filenum = file_list[current_idx][:-4]

        # Global coordinate to local coordinate for next route
        with open(ddir_meta + filenum + ".yml", 'r') as curr_metafile:
            curr_meta = yaml.safe_load(curr_metafile)
        velocity = np.abs(curr_meta["velocity"])

        raw_curr_lat = curr_meta['global_position_latlon'][0]
        raw_curr_lon = curr_meta['global_position_latlon'][1]

        veh_curr_lat, veh_curr_lon, is_outlier = hampel_filter(
            raw_curr_lat, raw_curr_lon, latlon_buffer, n_sigmas=3.0
        )

        prev_offset = -(1 + configx.gap_bearing)
        if len(latlon_buffer['lat_buf']) >= abs(prev_offset):
            veh_prev_lat = latlon_buffer['lat_buf'][prev_offset]
            veh_prev_lon = latlon_buffer['lon_buf'][prev_offset]
        else:
            prev_idx = max(0, current_idx - configx.gap_bearing)
            with open(ddir_meta + file_list[prev_idx], 'r') as prev_metafile:
                prev_meta = yaml.safe_load(prev_metafile)
            veh_prev_lat = prev_meta['global_position_latlon'][0]
            veh_prev_lon = prev_meta['global_position_latlon'][1]

        dLat_m = (veh_curr_lat - veh_prev_lat) * 40008000 / 360
        dLon_m = (veh_curr_lon - veh_prev_lon) * 40075000 * np.cos(np.radians(veh_curr_lat)) / 360

        # Calculate latlon based bearing
        latlon_bearing = latlon_to_yaw(
                veh_curr_lat, veh_curr_lon,
                veh_prev_lat, veh_prev_lon
        )
        
        # Calculate raw IMU bearing
        q = curr_meta['global_orientation_xyzw']
        raw_imu_bearing = compute_imu_yaw(q)
        
        # Apply LUT Correction to raw IMU bearing
        corrected_imu_bearing = correct_imu_lut(raw_imu_bearing)

        velocity_ms = curr_meta['velocity']
        velocity_kmh = velocity_ms * 3.6

        bearing_est = "IMU (LUT)"
        raw_bearing_veh = corrected_imu_bearing

        bearing_veh = bearing_filter(raw_bearing_veh, bearing_buffer)
        bearing_veh_deg = np.degrees(bearing_veh)

        # Compute global to local transformation
        rp_sdc_frame = []
        rp_lidbev_frame = []
        rp_lidfront_frame = []
        R_matrix = np.array([[np.cos(bearing_veh), -np.sin(bearing_veh)],
                            [np.sin(bearing_veh),  np.cos(bearing_veh)]])
        about_to_finish = False

        for j in range(2):
            next_lat = rp_list['route_point']['latitude'][j]
            next_lon = rp_list['route_point']['longitude'][j]
            dLat_m = (next_lat - veh_curr_lat) * 40008000 / 360
            dLon_m = (next_lon - veh_curr_lon) * 40075000 * np.cos(np.radians(veh_curr_lat)) / 360
            dist = np.sqrt(dLat_m**2 + dLon_m**2)
            if j == 0 and dist <= configx.rp1_close and not about_to_finish:
                if len(rp_list['route_point']['latitude']) > 2:
                    rp_list['route_point']['latitude'].pop(0)
                    rp_list['route_point']['longitude'].pop(0)
                else:
                    about_to_finish = True
                    rp_list['route_point']['latitude'][0] = rp_list['route_point']['latitude'][-1]
                    rp_list['route_point']['longitude'][0] = rp_list['route_point']['longitude'][-1]

                next_lat = rp_list['route_point']['latitude'][j]
                next_lon = rp_list['route_point']['longitude'][j]
                dLat_m = (next_lat - veh_curr_lat) * 40008000 / 360
                dLon_m = (next_lon - veh_curr_lon) * 40075000 * np.cos(np.radians(veh_curr_lat)) / 360

            nextr_local_point = R_matrix.T.dot(np.array([dLon_m, dLat_m]))

            nextr_x_frame, nextr_y_frame = plot_sdc_rpwp(configx, nextr_local_point[0], nextr_local_point[1])
            rp_sdc_frame.append(np.array([nextr_x_frame, nextr_y_frame]))

            nextr_x_frame, nextr_y_frame = plot_lidbev_rpwp(configx, nextr_local_point[0], nextr_local_point[1])
            rp_lidbev_frame.append(np.array([nextr_x_frame, nextr_y_frame]))

            nextr_x_frame, nextr_y_frame = plot_lidfront_rpwp(configx, nextr_local_point[0], nextr_local_point[1])
            rp_lidfront_frame.append(np.array([nextr_x_frame, nextr_y_frame]))

        # Waypoint computation based on future_idx sampling logic from dataloader
        _, _, local_veh_heading = euler_from_quaternion(
            w=curr_meta['local_orientation_xyzw'][3],
            x=curr_meta['local_orientation_xyzw'][0],
            y=curr_meta['local_orientation_xyzw'][1],
            z=curr_meta['local_orientation_xyzw'][2],
            rad=True
        )

        wp_local = []
        wp_sdc_frame = []
        wp_lidbev_frame = []
        wp_lidfront_frame = []
        
        future_indices = range(
            current_idx + data_rate,
            current_idx + (pred_len + 1) * data_rate,
            data_rate
        )
        
        for future_idx in future_indices:
            file_name_next = file_list[future_idx]
            with open(ddir_meta + file_name_next[:-4] + ".yml", 'r') as next_metafile:
                next_meta = yaml.safe_load(next_metafile)
                _, _, seq_theta = euler_from_quaternion(
                    w=next_meta['local_orientation_xyzw'][3],
                    x=next_meta['local_orientation_xyzw'][0],
                    y=next_meta['local_orientation_xyzw'][1],
                    z=next_meta['local_orientation_xyzw'][2],
                    rad=True
                )

            local_point = transform_2d_points(
                np.zeros((1, 3)),
                np.pi/2 - seq_theta,
                next_meta['local_position_xyz'][0],
                next_meta['local_position_xyz'][1],
                np.pi/2 - local_veh_heading,
                curr_meta['local_position_xyz'][0],
                curr_meta['local_position_xyz'][1]
            )
            local_point = local_point[0]
            wp_local.append(local_point)

            x_frame, y_frame = plot_sdc_rpwp(configx, local_point[0], local_point[1])
            wp_sdc_frame.append(np.array([x_frame, y_frame]))

            x_frame, y_frame = plot_lidbev_rpwp(configx, local_point[0], local_point[1])
            wp_lidbev_frame.append(np.array([x_frame, y_frame]))

            x_frame, y_frame = plot_lidfront_rpwp(configx, local_point[0], local_point[1])
            wp_lidfront_frame.append(np.array([x_frame, y_frame]))

        # Controller outputs
        steering, throttle, brake = pid_control(wp_local, velocity_ms, turn_controller, speed_controller)

        # Load raw sensor frames
        lidar_bev_segcol = cv2.imread(ddir_lidseg_bev + filenum + ".png")       # 256x256
        lidar_bev_depcol = cv2.imread(ddir_liddep_bev + filenum + ".png")       # 256x256
        lidar_front_segcol = cv2.imread(ddir_lidseg_fro + filenum + ".png")     # 512x64
        lidar_front_depcol = cv2.imread(ddir_liddep_fro + filenum + ".png")     # 512x64
        rgb_front = cv2.imread(ddir_rgb_front + filenum + ".png")               # 1280x720

        # Plot route points and waypoints on segmentation images
        lidar_bev_segcol_wprp = lidar_bev_segcol.copy()
        lidar_front_segcol_wprp = lidar_front_segcol.copy()

        for k in range(2):
            bev_color = (255, 255, 255) if k == 0 else (255, 255, 0)

            lidar_bev_segcol_wprp = cv2.circle(lidar_bev_segcol_wprp, (rp_lidbev_frame[k][0], rp_lidbev_frame[k][1]), radius=3, color=bev_color, thickness=2)
            lidar_front_segcol_wprp = cv2.circle(lidar_front_segcol_wprp, (rp_lidfront_frame[k][0], rp_lidfront_frame[k][1]), radius=3, color=(255, 255, 255), thickness=2)

        for k in range(len(wp_local)):
            lidar_bev_segcol_wprp = cv2.circle(lidar_bev_segcol_wprp, (wp_lidbev_frame[k][0], wp_lidbev_frame[k][1]), radius=2, color=(255, 255, 255), thickness=-1)
            lidar_front_segcol_wprp = cv2.circle(lidar_front_segcol_wprp, (wp_lidfront_frame[k][0], wp_lidfront_frame[k][1]), radius=2, color=(255, 255, 255), thickness=-1)

        # ---------------- LAYOUT ASSEMBLY ----------------

        left_column_w = 1024

        rgb_h = int(rgb_front.shape[0] * (left_column_w / rgb_front.shape[1]))
        rgb_front_scaled = cv2.resize(rgb_front, (left_column_w, rgb_h), interpolation=cv2.INTER_LINEAR)

        front_lidar_h = int(lidar_front_depcol.shape[0] * (left_column_w / lidar_front_depcol.shape[1]))
        front_dep_scaled = cv2.resize(lidar_front_depcol, (left_column_w, front_lidar_h), interpolation=cv2.INTER_LINEAR)
        front_seg_scaled = cv2.resize(lidar_front_segcol_wprp, (left_column_w, front_lidar_h), interpolation=cv2.INTER_LINEAR)

        left_column = np.concatenate((rgb_front_scaled, front_dep_scaled, front_seg_scaled), axis=0)
        total_h = left_column.shape[0]

        # Telemetry Overlay
        telemetry_lines = [
            ("INPUT", ""),
            (f"File Name: {filenum}.yml", ""),
            (f"Speed: {format(np.round(velocity_kmh, 3), '.3f')} km/h", ""),
            (f"Bearing: {format(np.round(bearing_veh_deg, 3), '.3f')} ({bearing_est})", ""),
            (f"Latlon Bearing: {format(np.round(np.degrees(latlon_bearing), 3), '.3f')}", ""),
            (f"Raw IMU Bearing: {format(np.round(np.degrees(raw_imu_bearing), 3), '.3f')}", ""),
            (f"LUT IMU Bearing: {format(np.round(np.degrees(corrected_imu_bearing), 3), '.3f')}", ""),
            (f"Robot Lat: {format(np.round(veh_curr_lat, 6), '.6f')}", ""),
            (f"Robot Lon: {format(np.round(veh_curr_lon, 6), '.6f')}", ""),
            (f"Rp1 Lat: {format(np.round(rp_list['route_point']['latitude'][0], 6), '.6f')}", ""),
            (f"Rp1 Lon: {format(np.round(rp_list['route_point']['longitude'][0], 6), '.6f')}", ""),
            (f"Rp2 Lat (neon): {format(np.round(rp_list['route_point']['latitude'][1], 6), '.6f')}", ""),
            (f"Rp2 Lon (neon): {format(np.round(rp_list['route_point']['longitude'][1], 6), '.6f')}", ""),
            ("", ""),
            ("OUTPUT", ""),
        ]

        for wp_idx in range(min(3, len(wp_local))):
            txt_wp = f"Wp{wp_idx+1} Loc: x: {format(np.round(wp_local[wp_idx][0], 3), '.3f')} | y: {format(np.round(wp_local[wp_idx][1], 3), '.3f')}"
            telemetry_lines.append((txt_wp, ""))

        telemetry_lines.extend([
            ("", ""),
            ("CONTROL", ""),
            (f"Steering: {format(np.round(steering, 4), '.4f')}", ""),
            (f"Throttle: {format(np.round(throttle, 4), '.4f')}", ""),
            (f"Brake: {format(np.round(brake, 4), '.4f')}", "")
        ])

        line_gap = min(22, int(rgb_h / (len(telemetry_lines) + 2)))
        overlay_w = 420
        overlay_h = (len(telemetry_lines) + 1) * line_gap

        overlay_pil = Image.new('RGBA', (overlay_w, overlay_h), (0, 0, 0, 128))
        draw = ImageDraw.Draw(overlay_pil)

        x_offset = 12
        for idx, (line_text, _) in enumerate(telemetry_lines):
            if line_text:
                draw.text((x_offset, 8 + idx * line_gap), line_text, font=configx.fontx, fill=(255, 255, 255, 255))

        left_column_pil = Image.fromarray(cv2.cvtColor(left_column, cv2.COLOR_BGR2RGB)).convert('RGBA')
        left_column_pil.paste(overlay_pil, (10, 10), overlay_pil)
        left_column = cv2.cvtColor(np.array(left_column_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

        bev_stack_raw = np.concatenate((lidar_bev_depcol, lidar_bev_segcol_wprp), axis=0)
        bev_target_w = int(bev_stack_raw.shape[1] * (total_h / bev_stack_raw.shape[0]))
        bev_column = cv2.resize(bev_stack_raw, (bev_target_w, total_h), interpolation=cv2.INTER_LINEAR)

        final_img = np.concatenate((left_column, bev_column), axis=1)

        if out_video is None:
            out_video = cv2.VideoWriter(
                configx.datadir + route + '/join_img/' + route + '.avi',
                cv2.VideoWriter_fourcc(*'DIVX'),
                configx.fps,
                (final_img.shape[1], final_img.shape[0])
            )

        cv2.imwrite(join_img_folder+filenum+".jpg", final_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out_video.write(np.uint8(final_img))

    if out_video is not None:
        out_video.release()