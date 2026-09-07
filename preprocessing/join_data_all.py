import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import yaml

from data_util import resizecrop_matrix, transform_2d_points, plot_lidbev_rpwp, plot_lidfront_rpwp, plot_sdc_rpwp, latlon_to_yaw, euler_from_quaternion
from data_util import PIDController, pid_control

# PID Controller
turn_controller = PIDController(K_P=0.5, K_I=0.25, K_D=0.15, n=15)
speed_controller = PIDController(K_P=1.5, K_I=0.25, K_D=0.5, n=15)

from config import GlobalConfig
configx = GlobalConfig()

# Loop pada semua route
route_list = os.listdir(configx.datadir)
route_list.sort()
for route in route_list:
    if os.path.isfile(configx.datadir + route):  # skip file
        continue
    print(route)
    
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
    
    # Load route points
    with open(configx.datadir + route + "/" + route + "_routepoint_list.yml", 'r') as rp_listx:
        rp_list = yaml.safe_load(rp_listx)
        rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
        rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])
    
    file_list = os.listdir(ddir_meta)
    file_list.sort()
    
    out_video = None
    
    # Loop frames
    for i in range(configx.gap_bearing, int(len(file_list) - (configx.n_wp * configx.wp_gap))):
        filenum = file_list[i][:-4]     
        print(join_img_folder + filenum)

        # Global coordinate to local coordinate for next route
        with open(ddir_meta + filenum + ".yml", 'r') as curr_metafile:
            curr_meta = yaml.safe_load(curr_metafile)
        veh_curr_lat = curr_meta['global_position_latlon'][0]
        veh_curr_lon = curr_meta['global_position_latlon'][1]
        
        with open(ddir_meta + file_list[i - configx.gap_bearing], 'r') as prev_metafile:
            prev_meta = yaml.safe_load(prev_metafile)
        veh_prev_lat = prev_meta['global_position_latlon'][0]
        veh_prev_lon = prev_meta['global_position_latlon'][1]

        dLat_m = (veh_curr_lat - veh_prev_lat) * 40008000 / 360
        dLon_m = (veh_curr_lon - veh_prev_lon) * 40075000 * np.cos(np.radians(veh_curr_lat)) / 360
        
        if np.sqrt(dLat_m**2 + dLon_m**2) > 1.0:
            bearing_est = "GNSS"
            bearing_veh = latlon_to_yaw(veh_curr_lat, veh_curr_lon, veh_prev_lat, veh_prev_lon, offset=np.pi / 2.0)
            bearing_veh_deg = np.degrees(bearing_veh) - 90
            bearing_veh = np.radians(bearing_veh_deg)
        else:
            bearing_est = "IMU"
            q = curr_meta['global_orientation_xyzw']
            w, x, y, z = q[3], q[0], q[1], q[2]
            bearing_veh = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)) - 1.5708
            bearing_veh_deg = np.degrees(bearing_veh)
        
        velocity_ms = curr_meta['velocity']
        velocity = velocity_ms * 3600 / 1000  # Convert to km/h

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
            
            if j == 0 and np.sqrt(dLat_m**2 + dLon_m**2) <= configx.rp1_close and not about_to_finish:
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

        # Waypoint computation
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
        for j in range(1, configx.n_wp + 1):
            file_name_next = file_list[int(i + j * configx.wp_gap)]
            with open(ddir_meta + file_name_next[:-3] + "yml", 'r') as next_metafile:
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
            lidar_bev_segcol_wprp = cv2.circle(lidar_bev_segcol_wprp, (rp_lidbev_frame[k][0], rp_lidbev_frame[k][1]), radius=3, color=(255, 255, 255), thickness=2)
            lidar_front_segcol_wprp = cv2.circle(lidar_front_segcol_wprp, (rp_lidfront_frame[k][0], rp_lidfront_frame[k][1]), radius=3, color=(255, 255, 255), thickness=2)

        for k in range(configx.n_wp):
            lidar_bev_segcol_wprp = cv2.circle(lidar_bev_segcol_wprp, (wp_lidbev_frame[k][0], wp_lidbev_frame[k][1]), radius=2, color=(255, 255, 255), thickness=-1)
            lidar_front_segcol_wprp = cv2.circle(lidar_front_segcol_wprp, (wp_lidfront_frame[k][0], wp_lidfront_frame[k][1]), radius=2, color=(255, 255, 255), thickness=-1)

        # ---------------- LAYOUT ASSEMBLY ----------------
        
        # 1. Left column target width (1024px)
        left_column_w = 1024

        # 2. Scale RGB Front (1280x720 -> 1024x576)
        rgb_h = int(rgb_front.shape[0] * (left_column_w / rgb_front.shape[1]))
        rgb_front_scaled = cv2.resize(rgb_front, (left_column_w, rgb_h), interpolation=cv2.INTER_LINEAR)

        # 3. Scale Front LiDAR streams proportionally (512x64 -> 1024x128)
        front_lidar_h = int(lidar_front_depcol.shape[0] * (left_column_w / lidar_front_depcol.shape[1]))
        front_dep_scaled = cv2.resize(lidar_front_depcol, (left_column_w, front_lidar_h), interpolation=cv2.INTER_LINEAR)
        front_seg_scaled = cv2.resize(lidar_front_segcol_wprp, (left_column_w, front_lidar_h), interpolation=cv2.INTER_LINEAR)

        # 4. Construct Left Column: [RGB Front | Front Depth | Front Seg]
        left_column = np.concatenate((rgb_front_scaled, front_dep_scaled, front_seg_scaled), axis=0)
        total_h = left_column.shape[0]  # Total height = 832px

        # 5. DYNAMICALLY GENERATE NON-TRUNCATED OVERLAY
        telemetry_lines = [
            ("INPUT", ""),
            (f"File Name: {filenum}.yml", ""),
            (f"Speed: {format(np.round(velocity, 3), '.3f')} km/h", ""),
            (f"Bearing: {format(np.round(bearing_veh_deg, 3), '.3f')} ({bearing_est})", ""),
            (f"Robot Lat: {format(np.round(veh_curr_lat, 6), '.6f')}", ""),
            (f"Robot Lon: {format(np.round(veh_curr_lon, 6), '.6f')}", ""),
            (f"Rp1 Lat: {format(np.round(rp_list['route_point']['latitude'][0], 6), '.6f')}", ""),
            (f"Rp1 Lon: {format(np.round(rp_list['route_point']['longitude'][0], 6), '.6f')}", ""),
            (f"Rp2 Lat: {format(np.round(rp_list['route_point']['latitude'][1], 6), '.6f')}", ""),
            (f"Rp2 Lon: {format(np.round(rp_list['route_point']['longitude'][1], 6), '.6f')}", ""),
            ("", ""),
            ("OUTPUT", ""),
        ]
        
        for wp_idx in range(min(3, configx.n_wp)):
            txt_wp = f"Wp{wp_idx+1} Loc: x: {format(np.round(wp_local[wp_idx][0], 3), '.3f')} | y: {format(np.round(wp_local[wp_idx][1], 3), '.3f')}"
            telemetry_lines.append((txt_wp, ""))

        telemetry_lines.extend([
            ("", ""),
            ("CONTROL", ""),
            (f"Steering: {format(np.round(steering, 4), '.4f')}", ""),
            (f"Throttle: {format(np.round(throttle, 4), '.4f')}", ""),
            (f"Brake: {format(np.round(brake, 4), '.4f')}", "")
        ])

        # Compact line spacing calculation to fit overlay entirely inside RGB View
        line_gap = min(22, int(rgb_h / (len(telemetry_lines) + 2)))
        overlay_w = 420
        overlay_h = (len(telemetry_lines) + 1) * line_gap

        # Create translucent RGBA overlay panel (50% transparency = 128 Alpha)
        overlay_pil = Image.new('RGBA', (overlay_w, overlay_h), (0, 0, 0, 128))
        draw = ImageDraw.Draw(overlay_pil)

        x_offset = 12
        for idx, (line_text, _) in enumerate(telemetry_lines):
            if line_text:
                draw.text((x_offset, 8 + idx * line_gap), line_text, font=configx.fontx, fill=(255, 255, 255, 255))

        # Alpha blend overlay onto top-left corner of left_column frame
        left_column_pil = Image.fromarray(cv2.cvtColor(left_column, cv2.COLOR_BGR2RGB)).convert('RGBA')
        left_column_pil.paste(overlay_pil, (10, 10), overlay_pil)
        left_column = cv2.cvtColor(np.array(left_column_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

        # 6. ENLARGE BEV STACK TO FILL THE ENTIRE RIGHT COLUMN HEIGHT
        bev_stack_raw = np.concatenate((lidar_bev_depcol, lidar_bev_segcol_wprp), axis=0)  # Native 256x512
        bev_target_w = int(bev_stack_raw.shape[1] * (total_h / bev_stack_raw.shape[0]))    # Proportional width
        bev_column = cv2.resize(bev_stack_raw, (bev_target_w, total_h), interpolation=cv2.INTER_LINEAR)

        # 7. Final Concatenation: [ Left Column (with Overlay) | Scaled Right BEV Column ]
        final_img = np.concatenate((left_column, bev_column), axis=1)

        # Initialize VideoWriter
        if out_video is None:
            out_video = cv2.VideoWriter(
                configx.datadir + route + '/join_img/' + route + '.avi', 
                cv2.VideoWriter_fourcc(*'DIVX'), 
                configx.fps, 
                (final_img.shape[1], final_img.shape[0])
            )

        out_video.write(np.uint8(final_img))

    if out_video is not None:
        out_video.release()