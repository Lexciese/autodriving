import yaml
import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from pathlib import Path
from config import GlobalConfig

configx = GlobalConfig()

from preprocess_util import euler_from_quaternion, latlon_to_yaw, quaternion_to_yaw

def clean_gnss_coordinates(lats, lons):
    df = pd.DataFrame({'latitude': lats, 'longitude': lons})

    lat_col = 'latitude'
    lon_col = 'longitude'

    df['Dist_prev'] = np.sqrt((df[lat_col] - df[lat_col].shift(1)) ** 2 + (df[lon_col] - df[lon_col].shift(1)) ** 2)
    df['Dist_next'] = np.sqrt((df[lat_col] - df[lat_col].shift(-1)) ** 2 + (df[lon_col] - df[lon_col].shift(-1)) ** 2)
    df['Dist_prev_next'] = np.sqrt((df[lat_col].shift(1) - df[lat_col].shift(-1)) ** 2 + (df[lon_col].shift(1) - df[lon_col].shift(-1)) ** 2)

    rolling_median = df['Dist_prev'].rolling(window=20, center=True, min_periods=1).median()
    dynamic_threshold = rolling_median * 2.0

    anomaly_mask = (df['Dist_prev'] > dynamic_threshold) & \
                   (df['Dist_next'] > dynamic_threshold) & \
                   (df['Dist_prev_next'] < (df['Dist_prev'] * 0.5))

    print(f"Total dynamic anomalies detected: {anomaly_mask.sum()}")

    df_fixed = df.copy()
    df_fixed.loc[anomaly_mask, [lat_col, lon_col]] = np.nan

    df_fixed[lat_col] = df_fixed[lat_col].interpolate(method='polynomial', order=2).interpolate(method='linear').bfill().ffill()
    df_fixed[lon_col] = df_fixed[lon_col].interpolate(method='polynomial', order=2).interpolate(method='linear').bfill().ffill()

    return df_fixed[lat_col].tolist(), df_fixed[lon_col].tolist()


route_list = sorted(p.name for p in configx.datadir.iterdir() if p.is_dir())
if configx.select_route != "all":
    route_list = [configx.select_route]
    print(f"only route: {configx.select_route} is selected")

for route in route_list:
    route_path = configx.datadir / route
    ddir_meta = route_path / "meta"
    file_list = sorted(p.name for p in ddir_meta.iterdir())

    meta = [yaml.safe_load(open(ddir_meta / meta_file, "r")) for meta_file in file_list]

    raw_lats = [m['global_position_latlon'][0] for m in meta]
    raw_lons = [m['global_position_latlon'][1] for m in meta]
    filtered_lats, filtered_lons = clean_gnss_coordinates(raw_lats, raw_lons)

    # Update metadata with cleaned coordinates
    for idx, m in enumerate(meta):
        m['global_position_latlon'] = [filtered_lats[idx], filtered_lons[idx]]

    # Plot Bearing GPS vs IMU
    jarak_frame = 25
    lat = [m['global_position_latlon'][0] for m in meta][::jarak_frame]
    lon = [m['global_position_latlon'][1] for m in meta][::jarak_frame]
    gnss_yaw = latlon_to_yaw(lat[1:], lon[1:], lat[:-1], lon[:-1], offset=np.pi / 2.0)
    gnss_yaw = np.append(gnss_yaw, gnss_yaw[-1])

    quaternion = [m['global_orientation_xyzw'] for m in meta][::jarak_frame]
    imu_yaw = quaternion_to_yaw(quaternion, offset=np.pi * 1.5)

    plt.figure(figsize=(10, 10))
    scale = 0.003
    plt.quiver(lon, lat, np.cos(gnss_yaw) * scale, np.sin(gnss_yaw) * scale, angles='xy', scale_units='xy', scale=60, color='orange', label='GNSS-based Bearing')
    plt.quiver(lon, lat, np.cos(imu_yaw) * scale, np.sin(imu_yaw) * scale, angles='xy', scale_units='xy', scale=60, color='purple', label='IMU-based Bearing')
    plt.title("GNSS vs IMU Bearing")
    plt.axis("equal")
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.grid()
    plt.legend()
    plt.savefig(route_path / f"{route}_bearing_viz.png", bbox_inches='tight', dpi=300)
    plt.close()

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

    with open(ddir_meta / file_list[0], 'r') as first_filexx:
        first_filex = yaml.safe_load(first_filexx)
    with open(ddir_meta / file_list[-1], 'r') as last_filexx:
        last_filex = yaml.safe_load(last_filexx)
    sin_angle_buff = deque()

    first_filex['global_position_latlon'] = [filtered_lats[0], filtered_lons[0]]
    last_filex['global_position_latlon'] = [filtered_lats[-1], filtered_lons[-1]]

    global_orientation_r, global_orientation_p, global_orientation_y = euler_from_quaternion(
        w=first_filex['global_orientation_xyzw'][3],
        x=first_filex['global_orientation_xyzw'][0],
        y=first_filex['global_orientation_xyzw'][1],
        z=first_filex['global_orientation_xyzw'][2],
        rad=False
    )
    global_orientation_rpy = [global_orientation_r, global_orientation_p, global_orientation_y]

    if configx.n_buffer != 0:
        for a in range(0, configx.n_buffer * configx.hz - 1):
            sin_angle_buff.append(np.sin(np.radians(global_orientation_rpy[2])))

    routes = {
        'first_point': {
            'latitude': first_filex['global_position_latlon'][0],
            'longitude': first_filex['global_position_latlon'][1],
        },
        'last_point': {
            'latitude': last_filex['global_position_latlon'][0],
            'longitude': last_filex['global_position_latlon'][1],
        },
        'route_point': {
            'latitude': [],
            'longitude': [],
        },
    }

    prev_lat = routes['first_point']['latitude']
    prev_lon = routes['first_point']['longitude']

    for i in tqdm(range(0, len(file_list))):
        file_name = file_list[i]

        with open(ddir_meta / file_name, 'r') as curr_metafile:
            curr_meta = yaml.safe_load(curr_metafile)

        curr_meta['global_position_latlon'] = [filtered_lats[i], filtered_lons[i]]

        # Calculate distance in local meters relative to prev_latlon
        dLat_m = (curr_meta['global_position_latlon'][0] - prev_lat) * 40008000 / 360
        dLon_m = (curr_meta['global_position_latlon'][1] - prev_lon) * 40075000 * np.cos(np.radians(curr_meta['global_position_latlon'][0])) / 360
        if np.sqrt(dLat_m ** 2 + dLon_m ** 2) >= configx.route_gap_distance:
            routes['route_point']['latitude'].append(curr_meta['global_position_latlon'][0])
            routes['route_point']['longitude'].append(curr_meta['global_position_latlon'][1])
            prev_lat = curr_meta['global_position_latlon'][0]
            prev_lon = curr_meta['global_position_latlon'][1]

        local_orientation_r, local_orientation_p, local_orientation_y = euler_from_quaternion(
            w=curr_meta['local_orientation_xyzw'][3],
            x=curr_meta['local_orientation_xyzw'][0],
            y=curr_meta['local_orientation_xyzw'][1],
            z=curr_meta['local_orientation_xyzw'][2],
            rad=False
        )

        global_orientation_r, global_orientation_p, global_orientation_y = euler_from_quaternion(
            w=curr_meta['global_orientation_xyzw'][3],
            x=curr_meta['global_orientation_xyzw'][0],
            y=curr_meta['global_orientation_xyzw'][1],
            z=curr_meta['global_orientation_xyzw'][2],
            rad=False
        )
        global_orientation_rpy = [global_orientation_r, global_orientation_p, global_orientation_y]

        euler_log['ekf_r'].append(global_orientation_rpy[0])
        euler_log['ekf_p'].append(global_orientation_rpy[1])
        euler_log['ekf_y'].append(global_orientation_rpy[2])
        euler_log['pose_r'].append(local_orientation_r)
        euler_log['pose_p'].append(local_orientation_p)
        euler_log['pose_y'].append(local_orientation_y)

        euler_log['acc_x'].append(curr_meta['acceleration_xyz'][0])
        euler_log['acc_y'].append(curr_meta['acceleration_xyz'][1])
        euler_log['acc_z'].append(curr_meta['acceleration_xyz'][2])
        euler_log['ang_spd_x'].append(curr_meta['angular_speed_xyz'][0])
        euler_log['ang_spd_y'].append(curr_meta['angular_speed_xyz'][1])
        euler_log['ang_spd_z'].append(curr_meta['angular_speed_xyz'][2])

        if configx.n_buffer != 0:
            angle_deg = global_orientation_rpy[2]
            sin_angle_buff.append(np.sin(np.radians(angle_deg)))
            sin_angle_buff_mean = np.array(sin_angle_buff).mean()

            if 0 < angle_deg <= 90:
                angle_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
            elif 90 < angle_deg <= 180:
                angle_deg_maf = 180 - np.degrees(np.arcsin(sin_angle_buff_mean))
            elif -180 < angle_deg <= -90:
                angle_deg_maf = -180 - np.degrees(np.arcsin(sin_angle_buff_mean))
            elif -90 < angle_deg <= 0:
                angle_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))

            sin_angle_buff.popleft()
            euler_log['ekf_y_maf'].append(angle_deg_maf)
        else:
            euler_log['ekf_y_maf'].append(0)

    pd.DataFrame(euler_log).to_csv(route_path / f"{route}_ahrs_rec.csv", index=False)

    with open(route_path / f"{route}_routepoint_list.yml", 'w') as c:
        yaml.dump(routes, c)

    timestamps = [Path(meta_file).stem for meta_file in file_list]
    gnss_path = route_path / f"{route}_gnss.csv"
    gnss_df = pd.DataFrame({'timestamp': timestamps, 'latitude': filtered_lats, 'longitude': filtered_lons})
    gnss_df.to_csv(gnss_path, index=False)
    print(f"saved to {gnss_path}")

    plt.grid(linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')
    x = np.array(routes['route_point']['longitude'])
    y = np.array(routes['route_point']['latitude'])
    plt.scatter(x, y, c='green', marker='.', label='route points')
    x = routes['first_point']['longitude']
    y = routes['first_point']['latitude']
    plt.scatter(x, y, c='blue', marker='X', label='start')
    x = routes['last_point']['longitude']
    y = routes['last_point']['latitude']
    plt.scatter(x, y, c='red', marker='X', label='finish')
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title("Route Points")
    plt.legend(loc='lower right')
    plt.savefig(route_path / f"{route}_routepoint_viz.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Plot AHRS Bearing
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle='--')
    startnum = 0
    x = np.arange(start=startnum, stop=startnum + len(euler_log['ekf_y']), step=1)
    y = np.array(euler_log['ekf_y'])
    plt.plot(x, y, c='blue', label='Raw')
    if configx.n_buffer != 0:
        y = np.array(euler_log['ekf_y_maf'])
        plt.plot(x, y, c='orange', label='MAF-' + str(configx.n_buffer))
    plt.xticks(ticks=np.arange(start=startnum, stop=startnum + len(euler_log['ekf_y']), step=100))
    plt.title('Bearing')
    plt.xlabel('Time Step (@' + str(configx.hz) + ' Hz)')
    plt.ylabel('Euler Angle (deg)')
    plt.legend(loc='lower right')
    plt.savefig(route_path / f"{route}_ahrs_viz.png", bbox_inches='tight', dpi=300)
    plt.close()