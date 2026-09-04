import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import yaml

from data_util import resizecrop_matrix, transform_2d_points, plot_lidbev_rpwp, plot_lidfront_rpwp, plot_sdc_rpwp, latlon_to_yaw, euler_from_quaternion #, bearing_biasing
from data_util import PIDController, pid_control
#PID Controller
turn_controller = PIDController(K_P=0.5, K_I=0.25, K_D=0.15, n=15)
speed_controller = PIDController(K_P=1.5, K_I=0.25, K_D=0.5, n=15)

from config import GlobalConfig
configx = GlobalConfig()


#loop pada semua route
route_list = os.listdir(configx.datadir)
route_list.sort()
for route in route_list:
    # if route in route_listx: #kalau termasuk route yang tidak diproses, skip
    #     continue
    if os.path.isfile(configx.datadir+route):  #kalau dia file, maka skip
        continue
    print(route)
    #meta
    ddir_meta = configx.datadir+route+"/meta/"
    #rgbd
    ddir_rgb_front = configx.datadir+route+"/camera/rgb/"
    ddir_depth_front = configx.datadir+route+"/camera/depth/img/"
    ddir_rgbseg_front_col = configx.datadir+route+"/camera/seg/img/"
    ddir_histo = configx.datadir+route+"/camera/histogram/"
    ddir_optflow = configx.datadir+route+"/camera/optical_flow/"
    
    ddir_lidar = configx.datadir+route+"/lidar/img/"
    ddir_lidseg_bev = ddir_lidar+"bev_seg/"
    ddir_lidseg_fro = ddir_lidar+"front_seg/"
    ddir_lidseg_rea = ddir_lidar+"rear_seg/"
    ddir_liddep_bev = ddir_lidar+"bev_dep/"
    ddir_liddep_fro = ddir_lidar+"front_dep/"
    ddir_liddep_rea = ddir_lidar+"rear_dep/"
    
    #all
    join_img_folder = configx.datadir+route+"/join_img/all_img/"
    os.makedirs(join_img_folder, exist_ok=True)
    
    
    #load routenya
    with open(configx.datadir+route+"/"+route+"_routepoint_list.yml", 'r') as rp_listx:
    # with open(datadir+route+"/gmaps"+route[-2:]+"_routepoint_list.yml", 'r') as rp_listx:
        rp_list = yaml.safe_load(rp_listx)
        #assign end point sebagai route terakhir
        rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
        rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])
    
    #list file
    file_list = os.listdir(ddir_meta)
    file_list.sort()
    
    #load initial location dari mobil
    # with open(ddir_meta+file_list[0], 'r') as first_metafile:
    #     first_meta = yaml.safe_load(first_metafile)
    
    

    #buat object video
    out_video = cv2.VideoWriter(configx.datadir+route+'/join_img/'+route+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), configx.fps, (configx.vid_size[0], configx.vid_size[1])) 
    
    #loop semua
    for i in range(configx.gap_bearing, int(len(file_list)-(configx.n_wp*configx.wp_gap))): #n_buffer*hz
        filenum = file_list[i][:-4]     
        print(join_img_folder+filenum)
        # for filex in file_list:
        # filenum = filex[:-4]
        # print(filenum)

        #GLOBAL COORDINATE TO LOCAL COORDINATE  UNTUK NEXT ROUTE            
        with open(ddir_meta+filenum+".yml", 'r') as curr_metafile:
            curr_meta = yaml.safe_load(curr_metafile)
        veh_curr_lat = curr_meta['global_position_latlon'][0]
        veh_curr_lon = curr_meta['global_position_latlon'][1]
        with open(ddir_meta+file_list[i-configx.gap_bearing], 'r') as prev_metafile:
            prev_meta = yaml.safe_load(prev_metafile)
        veh_prev_lat = prev_meta['global_position_latlon'][0]
        veh_prev_lon = prev_meta['global_position_latlon'][1]

        
        #SUDAH GA DIPAKE, IMU NOISY
        dLat_m = (veh_curr_lat-veh_prev_lat) * 40008000 / 360 #111320 #Y
        dLon_m = (veh_curr_lon-veh_prev_lon) * 40075000 * np.cos(np.radians(veh_curr_lat)) / 360 #X
        
        if np.sqrt(dLat_m**2 + dLon_m**2) > 1.0: #bergerak lebih dari 50cm
            bearing_est = "GNSS"
            bearing_veh = latlon_to_yaw(veh_curr_lat, veh_curr_lon, veh_prev_lat, veh_prev_lon, offset=np.pi / 2.0)
            bearing_veh_deg = np.degrees(bearing_veh) - 90
            bearing_veh = np.radians(bearing_veh_deg)
        else: #dianggap diam, ambil dari IMU
            bearing_est = "IMU"
            q = curr_meta['global_orientation_xyzw']
            w, x, y, z = q[3], q[0], q[1], q[2]
            bearing_veh = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*2 + z*2)) - 1.5708 #1.5708 offset, diputer 90 degree
            bearing_veh_deg = np.degrees(bearing_veh)
        
        #hitung velocity
        """
        veh_dLat_m = (veh_prev_lat-veh_curr_lat) * 40008000 / 360 #111320 #Y
        veh_dLon_m = (veh_prev_lon-veh_curr_lon) * 40075000 * np.cos(np.radians(veh_curr_lat)) / 360 #X
        movement = np.sqrt(veh_dLat_m**2 + veh_dLon_m**2)
        velocity_ms = movement/(1/(configx.hz*configx.gap_bearing))
        """
        velocity_ms = curr_meta['velocity']
        velocity = velocity_ms*3600/1000 #convert ke kmh
        # veh_prev_lat = curr_meta['global_position_latlon'][0]
        # veh_prev_lon = curr_meta['global_position_latlon'][1]


        #komputasi dari global ke local
        #https://gamedev.stackexchange.com/questions/79765/how-do-i-convert-from-the-global-coordinate-space-to-a-local-space
        # rp_local = [] #rp dalam local coordinate
        rp_sdc_frame = [] #rp dalam sdc frame coordinate
        rp_lidbev_frame = [] #rp dalam lidbev frame coordinate
        rp_lidfront_frame = [] #rp dalam lidfront frame coordinate
        R_matrix = np.array([[np.cos(bearing_veh), -np.sin(bearing_veh)],
                            [np.sin(bearing_veh),  np.cos(bearing_veh)]])
        about_to_finish = False
        for j in range(2): #ada 2 route point
            next_lat = rp_list['route_point']['latitude'][j]
            next_lon = rp_list['route_point']['longitude'][j]
            dLat_m = (next_lat-veh_curr_lat) * 40008000 / 360 #111320 #Y
            dLon_m = (next_lon-veh_curr_lon) * 40075000 * np.cos(np.radians(veh_curr_lat)) / 360 #X
            
            if j==0 and np.sqrt(dLat_m**2 + dLon_m**2) <= configx.rp1_close and not about_to_finish: #jika jarak euclidian rp1 <= jarak min, hapus route dan loncat ke next route
                if len(rp_list['route_point']['latitude']) > 2: #jika jumlah route list masih > 2
                    rp_list['route_point']['latitude'].pop(0)
                    rp_list['route_point']['longitude'].pop(0)
                else: #berarti mendekati finish
                    about_to_finish = True
                    rp_list['route_point']['latitude'][0] = rp_list['route_point']['latitude'][-1]
                    rp_list['route_point']['longitude'][0] = rp_list['route_point']['longitude'][-1]

                next_lat = rp_list['route_point']['latitude'][j]
                next_lon = rp_list['route_point']['longitude'][j]
                dLat_m = (next_lat-veh_curr_lat) * 40008000 / 360 #111320 #Y
                dLon_m = (next_lon-veh_curr_lon) * 40075000 * np.cos(np.radians(veh_curr_lat)) / 360 #X


            nextr_local_point = R_matrix.T.dot(np.array([dLon_m, dLat_m]))
            # rp_local.append(nextr_local_point)

            #plot untuk cam sdc frame
            nextr_x_frame, nextr_y_frame = plot_sdc_rpwp(configx, nextr_local_point[0], nextr_local_point[1])
            rp_sdc_frame.append(np.array([nextr_x_frame, nextr_y_frame]))

            #plot untuk lidbev frame
            nextr_x_frame, nextr_y_frame = plot_lidbev_rpwp(configx, nextr_local_point[0], nextr_local_point[1])
            rp_lidbev_frame.append(np.array([nextr_x_frame, nextr_y_frame]))

            #plot untuk lidfront frame
            nextr_x_frame, nextr_y_frame = plot_lidfront_rpwp(configx, nextr_local_point[0], nextr_local_point[1])
            rp_lidfront_frame.append(np.array([nextr_x_frame, nextr_y_frame]))
       

        #PLOT WAYPOINT DARI LOCAL ROBOT ORIENTATION
        # local_veh_heading_deg = curr_meta['local_orientation_rpy'][2] #ambil yaw nya
        # local_veh_heading = np.radians(local_veh_heading_deg)
        _, _, local_veh_heading = euler_from_quaternion(w=curr_meta['local_orientation_xyzw'][3], x=curr_meta['local_orientation_xyzw'][0], y=curr_meta['local_orientation_xyzw'][1], z=curr_meta['local_orientation_xyzw'][2], rad=True)

        #https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
        wp_local = [] #wp dalam local coordinate
        wp_sdc_frame = [] #wp dalam sdc frame coordinate
        wp_lidbev_frame = [] #wp dalam lidbev frame coordinate
        wp_lidfront_frame = [] #wp dalam lidfront frame coordinate
        for j in range(1,configx.n_wp+1):
            file_name_next = file_list[int(i+j*configx.wp_gap)]
            with open(ddir_meta+file_name_next[:-3]+"yml", 'r') as next_metafile:
                next_meta = yaml.safe_load(next_metafile)
                _, _, seq_theta = euler_from_quaternion(w=next_meta['local_orientation_xyzw'][3], x=next_meta['local_orientation_xyzw'][0], y=next_meta['local_orientation_xyzw'][1], z=next_meta['local_orientation_xyzw'][2], rad=True)
            
            # seq_theta = np.radians(next_meta['local_orientation_rpy'][2])  #ambil yaw nya,
            local_point = transform_2d_points(np.zeros((1,3)), np.pi/2-seq_theta, next_meta['local_position_xyz'][0], next_meta['local_position_xyz'][1], np.pi/2-local_veh_heading, curr_meta['local_position_xyz'][0], curr_meta['local_position_xyz'][1])
            local_point = local_point[0]
            wp_local.append(local_point)

            #plot untuk cam sdc frame
            x_frame, y_frame = plot_sdc_rpwp(configx, local_point[0], local_point[1])
            wp_sdc_frame.append(np.array([x_frame, y_frame]))

            #plot untuk lidBEV frame
            x_frame, y_frame = plot_lidbev_rpwp(configx, local_point[0], local_point[1])
            wp_lidbev_frame.append(np.array([x_frame, y_frame]))

            #plot untuk lidfront frame
            x_frame, y_frame = plot_lidfront_rpwp(configx, local_point[0], local_point[1])
            wp_lidfront_frame.append(np.array([x_frame, y_frame]))


        #hitung control
        steering, throttle, brake = pid_control(wp_local, velocity_ms, turn_controller, speed_controller)


        #load image-image
        lidar_bev_segcol = cv2.imread(ddir_lidseg_bev+filenum+".png")
        lidar_bev_depcol = cv2.imread(ddir_liddep_bev+filenum+".png")
        lidar_front_segcol = cv2.imread(ddir_lidseg_fro+filenum+".png")
        lidar_front_depcol = cv2.imread(ddir_liddep_fro+filenum+".png")
        lidar_rear_segcol = cv2.imread(ddir_lidseg_rea+filenum+".png")
        lidar_rear_depcol = cv2.imread(ddir_liddep_rea+filenum+".png")

        rgb_front = resizecrop_matrix(cv2.imread(ddir_rgb_front+filenum+".png"), WH_resized=[configx.cam_w, configx.cam_h], crop_HW=[configx.cam_h, configx.cam_w])
        depth_front = resizecrop_matrix(cv2.imread(ddir_depth_front+filenum+".png"), WH_resized=[configx.cam_w, configx.cam_h], crop_HW=[configx.cam_h, configx.cam_w])
        rgbseg_front_col = resizecrop_matrix(cv2.imread(ddir_rgbseg_front_col+filenum+".png"), WH_resized=[configx.cam_w, configx.cam_h], crop_HW=[configx.cam_h, configx.cam_w])
        #optflow = resizecrop_matrix(cv2.imread(ddir_optflow+filenum+".png"), WH_resized=[configx.cam_w, configx.cam_h], crop_HW=[configx.cam_h, configx.cam_w])
        histox = resizecrop_matrix(cv2.imread(ddir_histo+filenum+".png"), WH_resized=[configx.cam_w, configx.cam_h], crop_HW=[configx.cam_h, configx.cam_w])
        

        #PLOT PADA GAMBAR
        # print(wp_bev_frame)
        # print(rp_bev_frame)
        # print(rp_front_frame)
        lidar_bev_segcol_wprp = lidar_bev_segcol.copy()
        lidar_front_segcol_wprp = lidar_front_segcol.copy()
        #"""
        for k in range(2): #plot rp
            lidar_bev_segcol_wprp = cv2.circle(lidar_bev_segcol_wprp, (rp_lidbev_frame[k][0], rp_lidbev_frame[k][1]), radius=3, color=(255, 255, 255), thickness=2)
            lidar_front_segcol_wprp = cv2.circle(lidar_front_segcol_wprp, (rp_lidfront_frame[k][0], rp_lidfront_frame[k][1]), radius=3, color=(255, 255, 255), thickness=2)
            #"""
        for k in range(configx.n_wp): #plot wp
            lidar_bev_segcol_wprp = cv2.circle(lidar_bev_segcol_wprp, (wp_lidbev_frame[k][0], wp_lidbev_frame[k][1]), radius=2, color=(255, 255, 255), thickness=-1)
            lidar_front_segcol_wprp = cv2.circle(lidar_front_segcol_wprp, (wp_lidfront_frame[k][0], wp_lidfront_frame[k][1]), radius=2, color=(255, 255, 255), thickness=-1)
            
        #tambahkan tulisan-tulisan metadata
        text_io_img = Image.new('RGB', (configx.cam_w, 4*configx.cam_h)) #2*configx.cam_h
        x_offset = 0
        y_offset = 0
        #ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+0*configx.text_gap),"Navigation", font=configx.fontx, fill=(255,255,255), stroke_width=1, stroke_fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+0*configx.text_gap),"X Translation Acc.", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+0*configx.text_gap),format(np.round(curr_meta['acceleration_xyz'][0], 3), ".3f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+1*configx.text_gap),"Y Translation Acc.", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+1*configx.text_gap),format(np.round(curr_meta['acceleration_xyz'][2], 3), ".3f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+2*configx.text_gap),"Z Rotation Speed", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+2*configx.text_gap),format(np.round(curr_meta['angular_speed_xyz'][2], 3), ".3f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+3*configx.text_gap),"Velocity", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+3*configx.text_gap),format(np.round(velocity, 3), ".3f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+4*configx.text_gap),"Bearing", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+4*configx.text_gap),format(np.round(bearing_veh_deg, 3), ".3f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+5*configx.text_gap),"Car Latitude", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+5*configx.text_gap),format(np.round(veh_curr_lat, 6), ".6f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+6*configx.text_gap),"Car Longitude", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+6*configx.text_gap),format(np.round(veh_curr_lon, 6), ".6f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+7*configx.text_gap),"Rp1 Latitude", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+7*configx.text_gap),format(np.round(rp_list['route_point']['latitude'][0], 6), ".6f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+8*configx.text_gap),"Rp1 Longitude", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+8*configx.text_gap),format(np.round(rp_list['route_point']['longitude'][0], 6), ".6f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+9*configx.text_gap),"Rp2 Latitude", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+9*configx.text_gap),format(np.round(rp_list['route_point']['latitude'][1], 6), ".6f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+10*configx.text_gap),"Rp2 Longitude", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+10*configx.text_gap),format(np.round(rp_list['route_point']['longitude'][1], 6), ".6f"), font=configx.fontx, fill=(255,255,255))
        # text_io_img = np.asarray(text_io_img)

        # text_output_img = Image.new('RGB', (configx.cam_w, configx.cam_h))
        # x_offset = 0
        # y_offset = 0
        #ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+12*configx.text_gap),"Waypoints", font=configx.fontx, fill=(255,255,255), stroke_width=1, stroke_fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+11*configx.text_gap),"Wp1", font=configx.fontx, fill=(255,255,255))
        txtx = "x: " + format(np.round(wp_local[0][0], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap, y_offset+11*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        txtx = "| y: "+format(np.round(wp_local[0][1], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+11*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+12*configx.text_gap),"Wp2", font=configx.fontx, fill=(255,255,255))
        txtx = "x: " + format(np.round(wp_local[1][0], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap, y_offset+12*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        txtx = "| y: "+format(np.round(wp_local[1][1], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+12*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+13*configx.text_gap),"Wp3", font=configx.fontx, fill=(255,255,255))
        txtx = "x: " + format(np.round(wp_local[2][0], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap, y_offset+13*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        txtx = "| y: "+format(np.round(wp_local[2][1], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+13*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+14*configx.text_gap),"Wp4", font=configx.fontx, fill=(255,255,255))
        txtx = "x: " + format(np.round(wp_local[3][0], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap, y_offset+14*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        txtx = "| y: "+format(np.round(wp_local[3][1], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+14*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+15*configx.text_gap),"Wp5", font=configx.fontx, fill=(255,255,255))
        txtx = "x: " + format(np.round(wp_local[4][0], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap, y_offset+15*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        txtx = "| y: "+format(np.round(wp_local[4][1], 3), ".3f")
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+15*configx.text_gap), txtx, font=configx.fontx, fill=(255,255,255))
        

        
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+16*configx.text_gap),"Steering", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+16*configx.text_gap),format(np.round(steering, 4), ".4f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+17*configx.text_gap),"Throttle", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+17*configx.text_gap),format(np.round(throttle, 4), ".4f"), font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+18*configx.text_gap),"Brake", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+18*configx.text_gap),format(np.round(brake, 4), ".4f"), font=configx.fontx, fill=(255,255,255))
        
        
        ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+19*configx.text_gap),"Bearing Est.", font=configx.fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img).text((x_offset+configx.metadata_gap+70, y_offset+19*configx.text_gap), bearing_est, font=configx.fontx, fill=(255,255,255))
        """"""

        # text_io_img = np.asarray(text_io_img)

        #ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+19*configx.text_gap),"DRIVING RECORD", font=configx.fontx, fill=(255,255,255), stroke_width=1, stroke_fill=(255,255,255))
        #ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+19*configx.text_gap),"Record rate: "+str(configx.hz)+"Hz | Speed up: "+str(configx.speedup_factor)+"x", font=configx.fontx, fill=(255,255,255))
        #ImageDraw.Draw(text_io_img).text((x_offset+5, y_offset+19*configx.text_gap),"AISL-TUT Autonomous Driving", font=configx.fontx, fill=(255,255,255), stroke_width=1, stroke_fill=(255,255,255))

        text_io_img = np.asarray(text_io_img)

        """
        #throttle steering brake
        text_io_img2 = Image.new('RGB', (configx.cam_w, 2*configx.cam_h))
        x_offset = 5
        y_offset = 5
        #other, buat join_img dll
        fontsize = 15
        font_mul = 1
        fontx = ImageFont.truetype(font="arial.ttf", size=font_mul*fontsize) #arialbold arial
        #jika error font tidak ketemu, donlod dulu di https://www.freefontspro.com/14454/arial.ttf lalu copas foldernya ke /usr/share/fonts/truetype/
        text_gap = (fontsize+5)*font_mul
        metadata_gap = 110*font_mul
        ImageDraw.Draw(text_io_img2).text((x_offset+5, y_offset+0*text_gap),"Steering", font=fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img2).text((x_offset+metadata_gap+75, y_offset+0*text_gap),format(np.round(steering, 4), ".4f"), font=fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img2).text((x_offset+5, y_offset+1*text_gap),"Throttle", font=fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img2).text((x_offset+metadata_gap+75, y_offset+1*text_gap),format(np.round(throttle, 4), ".4f"), font=fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img2).text((x_offset+5, y_offset+2*text_gap),"Brake", font=fontx, fill=(255,255,255))
        ImageDraw.Draw(text_io_img2).text((x_offset+metadata_gap+75, y_offset+2*text_gap),format(np.round(brake, 4), ".4f"), font=fontx, fill=(255,255,255))
        
        #ImageDraw.Draw(text_io_img2).text((x_offset+5, y_offset+5*text_gap),"Brake", font=fontx, fill=(255,255,255))
        #ImageDraw.Draw(text_io_img2).text((x_offset+metadata_gap+75, y_offset+2*text_gap),format(np.round(brake, 4), ".4f"), font=fontx, fill=(255,255,255))

        text_io_img2 = np.asarray(text_io_img2)
        #blank
        # blank = np.asarray(Image.new('RGB', (configx.cam_w, configx.cam_h)))
        """
 
        #gabung semua file
        camera_front1 = np.concatenate((rgb_front, depth_front), axis=1)
        camera_front2 = np.concatenate((rgbseg_front_col, histox), axis=1)
        camera_front = np.concatenate((camera_front1, camera_front2), axis=0)

        lidar_front_rear = np.concatenate((lidar_front_depcol, lidar_front_segcol_wprp, lidar_rear_depcol, lidar_rear_segcol), axis=0)
        
        cam_lidar_front_rear = np.concatenate((camera_front, lidar_front_rear), axis=0)

        lidar_bev = np.concatenate((lidar_bev_depcol, lidar_bev_segcol_wprp), axis=0)

        # perception = np.concatenate((camera_front, camera_rear, lidar), axis=0)
        #metaimg = np.concatenate((text_io_img, histox), axis=0)
        # final_img = np.concatenate((perception, dvs_meta), axis=1)

        # sesuaikan config, mau vertikal atau horizontal
        # final_img = np.concatenate((cam_lidar_front_rear, lidar_bev, dvs_meta), axis=1)
        final_img = np.concatenate((cam_lidar_front_rear, lidar_bev, text_io_img), axis=1)
        

        cv2.imwrite(join_img_folder+filenum+".png", final_img)
        #write ke video
        # final_img = Image.open(join_img_folder+filenum+".png")
        # pix = np.array(final_img) #jadikan ke array supaya bisa di save
        # #GANTI ORDER BGR KE RGB, SWAP!, PERLU KARENAA FORMAT PILLOW DAN OPENCV BEDA
        # img = swap_RGB2BGR(pix)
        out_video.write(np.uint8(final_img))

    out_video.release()
