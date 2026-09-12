import yaml
import os
import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from preprocessing.config import GlobalConfig
configx = GlobalConfig()

from preprocessing.data_util import euler_from_quaternion, latlon_to_yaw, quaternion_to_yaw

#persoalan QT plugins baca: https://github.com/NVlabs/instant-ngp/discussions/300

#loop pada semua route
route_list = os.listdir(configx.datadir)
route_list.sort()
for route in route_list:
    # if route in route_listx: #kalau termasuk route yang tidak diproses, skip
    #     continue
    if os.path.isfile(configx.datadir+route):  #kalau dia file, maka skip
        continue

    ddir_meta = configx.datadir+route+"/meta/"
    file_list = os.listdir(ddir_meta)
    file_list.sort()

    #ngecek
    #file_list = file_list[2001:2500]

    #EKSEKUSI PLOT BEARING GPS vs IMU

    meta = [yaml.safe_load(open(ddir_meta+meta_file, "r")) for meta_file in file_list]
    #print(meta)

    # Moving Average Filter to remove GNSS outliers from metadata list
    gnss_window_size = 3
    raw_lats = [m['global_position_latlon'][0] for m in meta]
    raw_lons = [m['global_position_latlon'][1] for m in meta]
    smooth_lats = pd.Series(raw_lats).rolling(window=gnss_window_size, min_periods=1, center=True).mean().tolist()
    smooth_lons = pd.Series(raw_lons).rolling(window=gnss_window_size, min_periods=1, center=True).mean().tolist()

    for idx, m in enumerate(meta):
        m['global_position_latlon'] = [smooth_lats[idx], smooth_lons[idx]]

    jarak_frame = 25
    lat = [m['global_position_latlon'][0] for m in meta][::jarak_frame]
    lon = [m['global_position_latlon'][1] for m in meta][::jarak_frame]
    gnss_yaw = latlon_to_yaw(lat[1:], lon[1:], lat[:-1], lon[:-1], offset=np.pi / 2.0)
    gnss_yaw = np.append(gnss_yaw, gnss_yaw[-1])

    quaternion = [m['global_orientation_xyzw'] for m in meta][::jarak_frame]
    imu_yaw = quaternion_to_yaw(quaternion, offset=np.pi * 1.5)

    plt.figure(figsize=(10, 10))
    scale = 0.003
    plt.quiver(lon, lat, np.cos(gnss_yaw)*scale, np.sin(gnss_yaw)*scale, angles='xy', scale_units='xy', scale=60, color='orange', label='GNSS-based Bearing')
    plt.quiver(lon, lat, np.cos(imu_yaw)*scale, np.sin(imu_yaw)*scale, angles='xy', scale_units='xy', scale=60, color='purple', label='IMU-based Bearing')
    plt.title("GNSS vs IMU Bearing")
    plt.axis("equal")
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.grid()
    plt.legend()
    plt.savefig(configx.datadir+route+"/"+route+"_bearing_viz.png", bbox_inches='tight', dpi=300)
    plt.close()


    #AHRS record, cek pergantian heading
    euler_log = OrderedDict([
        ('ekf_r', []),
        ('pose_r', []),
        ('ekf_p', []),
        ('pose_p', []),
        ('ekf_y', []),
        ('ekf_y_maf', []),
        ('pose_y', []),
        ('acc_x', []),
        ('acc_y', []),
        ('acc_z', []),
        ('ang_spd_x', []),
        ('ang_spd_y', []),
        ('ang_spd_z', []),
    ])

    #cek MAF dan renormalize dari -180 - 180 ke 0 - 360
    with open(ddir_meta+file_list[0], 'r') as first_filexx:
        first_filex = yaml.safe_load(first_filexx)
    with open(ddir_meta+file_list[-1], 'r') as last_filexx:
        last_filex = yaml.safe_load(last_filexx)
    sin_angle_buff = deque()

    # Apply MAF for initial route position markers
    first_filex['global_position_latlon'] = [smooth_lats[0], smooth_lons[0]]
    last_filex['global_position_latlon'] = [smooth_lats[-1], smooth_lons[-1]]

    global_orientation_r, global_orientation_p, global_orientation_y = euler_from_quaternion(w=first_filex['global_orientation_xyzw'][3], x=first_filex['global_orientation_xyzw'][0], y=first_filex['global_orientation_xyzw'][1], z=first_filex['global_orientation_xyzw'][2], rad=False)
    global_orientation_rpy = [global_orientation_r, global_orientation_p, global_orientation_y]

    if configx.n_buffer!=0:
        for a in range(0, configx.n_buffer*configx.hz-1): #-1 karena nanti akan diappend dulu dengan data baru
            sin_angle_buff.append(np.sin(np.radians(global_orientation_rpy[2])))

    #register start dan end position
    routes = { #simpan routepoints
        'first_point':{
            'latitude': None,
            'longitude': None,
        },
        'last_point':{
            'latitude': None,
            'longitude': None,
        },
        'route_point':{
            'latitude': [],
            'longitude': [],
        },
    }
    routes['first_point']['latitude'] = first_filex['global_position_latlon'][0]
    routes['first_point']['longitude'] = first_filex['global_position_latlon'][1]
    routes['last_point']['latitude'] = last_filex['global_position_latlon'][0]
    routes['last_point']['longitude'] = last_filex['global_position_latlon'][1]

    #ROUTE BERDASARKAN JARAK SESUNGGUHNYA, DIHITUNG DARI LATITUDE LONGITUDE
    prev_lat = routes['first_point']['latitude']
    prev_lon = routes['first_point']['longitude']

    #loop routepoints
    for i in range(0, len(file_list)):
        file_name = file_list[i]
        print(ddir_meta+file_name)

        with open(ddir_meta+file_name, 'r') as curr_metafile:
            curr_meta = yaml.safe_load(curr_metafile)

        # Apply smoothed GNSS coordinates
        curr_meta['global_position_latlon'] = [smooth_lats[i], smooth_lons[i]]

        #ROUTE BERDASARKAN JARAK SESUNGGUHNYA, DIHITUNG DARI LATITUDE LONGITUDE
        #hitung jarak dalam local coordinate, relatif ke prev_latlon
        dLat_m = (curr_meta['global_position_latlon'][0]-prev_lat) * 40008000 / 360 #111320 #Y
        dLon_m = (curr_meta['global_position_latlon'][1]-prev_lon) * 40075000 * np.cos(np.radians(curr_meta['global_position_latlon'][0])) / 360 #X
        if np.sqrt(dLat_m**2 + dLon_m**2) >= configx.route_gap_distance: #jika jarak euclidian >= route_gap_distance, maka dijadikan route point
            routes['route_point']['latitude'].append(curr_meta['global_position_latlon'][0])
            routes['route_point']['longitude'].append(curr_meta['global_position_latlon'][1])
            #update prev_latlon
            prev_lat = curr_meta['global_position_latlon'][0]
            prev_lon = curr_meta['global_position_latlon'][1]

        local_orientation_r, local_orientation_p, local_orientation_y = euler_from_quaternion(w=curr_meta['local_orientation_xyzw'][3], x=curr_meta['local_orientation_xyzw'][0], y=curr_meta['local_orientation_xyzw'][1], z=curr_meta['local_orientation_xyzw'][2], rad=False)

        #save data ke CSV
        global_orientation_r, global_orientation_p, global_orientation_y = euler_from_quaternion(w=curr_meta['global_orientation_xyzw'][3], x=curr_meta['global_orientation_xyzw'][0], y=curr_meta['global_orientation_xyzw'][1], z=curr_meta['global_orientation_xyzw'][2], rad=False)
        global_orientation_rpy = [global_orientation_r, global_orientation_p, global_orientation_y]

        euler_log['ekf_r'].append(global_orientation_rpy[0])
        euler_log['ekf_p'].append(global_orientation_rpy[1])
        euler_log['ekf_y'].append(global_orientation_rpy[2])
        euler_log['pose_r'].append(local_orientation_r)
        euler_log['pose_p'].append(local_orientation_p)
        euler_log['pose_y'].append(local_orientation_y)
        #IMU
        euler_log['acc_x'].append(curr_meta['acceleration_xyz'][0])
        euler_log['acc_y'].append(curr_meta['acceleration_xyz'][1])
        euler_log['acc_z'].append(curr_meta['acceleration_xyz'][2])
        euler_log['ang_spd_x'].append(curr_meta['angular_speed_xyz'][0])
        euler_log['ang_spd_y'].append(curr_meta['angular_speed_xyz'][1])
        euler_log['ang_spd_z'].append(curr_meta['angular_speed_xyz'][2])

        if configx.n_buffer!=0:
            angle_deg = global_orientation_rpy[2]
            sin_angle_buff.append(np.sin(np.radians(angle_deg)))
            sin_angle_buff_mean = np.array(sin_angle_buff).mean()
            #cek kuadran
            if 0 < angle_deg <= 90: #Q1
                angle_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
            elif 90 < angle_deg <= 180: #Q2
                angle_deg_maf = 180 - np.degrees(np.arcsin(sin_angle_buff_mean))
            elif -180 < angle_deg <= -90: #Q3 180 - 270
                angle_deg_maf = -180 - np.degrees(np.arcsin(sin_angle_buff_mean))
            elif -90 < angle_deg <= 0: #Q4 270 - 360
                angle_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
            sin_angle_buff.popleft() #hilangkan 1 untuk diisilagi dengan next data nantinya
            euler_log['ekf_y_maf'].append(angle_deg_maf) #kembalikan ke nilai asli
        else:
            euler_log['ekf_y_maf'].append(0)

        pd.DataFrame(euler_log).to_csv(configx.datadir+route+"/"+route+"_ahrs_rec_maf.csv", index=False)

    #save routepoints ke yaml
    with open(configx.datadir+route+"/"+route+"_routepoint_list_maf.yml", 'w') as c:
        yaml.dump(routes, c)

    gnss_lat = [m['global_position_latlon'][0] for m in meta]
    gnss_lon = [m['global_position_latlon'][1] for m in meta]
    gnss_df = pd.DataFrame({'latitude': gnss_lat, 'longitude': gnss_lon})
    gnss_df.to_csv(configx.datadir+route+"/"+route+"_gnss_maf.csv", index=False)

    plt.grid(linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')
    x = np.array(routes['route_point']['longitude'])# - routes['first_point']['longitude']
    y = np.array(routes['route_point']['latitude'])# - routes['first_point']['latitude']
    plt.scatter(x, y, c='green', marker='.', label='route points') #route points
    x = routes['first_point']['longitude']# - routes['first_point']['longitude']
    y = routes['first_point']['latitude']# - routes['first_point']['latitude']
    plt.scatter(x, y, c='blue', marker='X', label='start') #start
    x = routes['last_point']['longitude']# - routes['first_point']['longitude']
    y = routes['last_point']['latitude']# - routes['first_point']['latitude']
    plt.scatter(x, y, c='red', marker='X', label='finish') #last
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title("Route Points")
    plt.legend(loc='lower right')
    plt.savefig(configx.datadir+route+"/"+route+"_routepoint_viz_maf.png", bbox_inches='tight', dpi=300)
    plt.close()

    #plot ahrs, orientasi terhadap utara, angle yaw
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle='--')
    startnum=0 #int(first_filex['seq'])
    x = np.arange(start=startnum, stop=startnum+len(euler_log['ekf_y']), step=1)
    y = np.array(euler_log['ekf_y'])
    plt.plot(x, y, c='blue', label='Raw')
    if configx.n_buffer!=0:
        y = np.array(euler_log['ekf_y_maf'])
        plt.plot(x, y, c='orange', label='MAF-'+str(configx.n_buffer))
    plt.xticks(ticks=np.arange(start=startnum, stop=startnum+len(euler_log['ekf_y']), step=100))
    plt.title('Bearing')
    plt.xlabel('Time Step (@'+str(configx.hz)+' Hz)')
    plt.ylabel('Euler Angle (deg)')
    plt.legend(loc='lower right')
    plt.savefig(configx.datadir+route+"/"+route+"_ahrs_viz.png", bbox_inches='tight', dpi=300)
    plt.close()
