import os
import yaml
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import numpy as np
from pathlib import Path

def compute_imu_bearing_bias(imu, ref):
    # Shortest angular difference (wrapped to -pi..pi)
    diff_rad = np.arctan2(np.sin(imu - ref), np.cos(imu - ref))
    diff_deg = np.degrees(diff_rad)
    ref_deg = np.degrees(ref)
    
    bins = [-180, -120, -60, 0, 60, 120, 180]
    sector_labels = ['[-180, -120)', '[-120, -60)', '[-60, 0)', '[0, 60)', '[60, 120)', '[120, 180]']
    
    sector_biases = {}
    
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        
        # Include upper boundary for the last sector
        if i == len(bins) - 2:
            mask = (ref_deg >= low) & (ref_deg <= high)
        else:
            mask = (ref_deg >= low) & (ref_deg < high)
            
        sector_diffs = diff_deg[mask]
        
        if len(sector_diffs) > 0:
            sector_biases[sector_labels[i]] = {
                'mean_bias_deg': float(np.mean(sector_diffs)),
                'std_deg': float(np.std(sector_diffs)),
                'count': int(len(sector_diffs))
            }
        else:
            sector_biases[sector_labels[i]] = {'mean_bias_deg': None, 'std_deg': None, 'count': 0}
            
    return sector_biases

def imu_to_yaw(q, offset=0.0):
    r = R.from_quat(q)
    # Project the sensor's X-axis (+X Forward) into world horizontal frame
    x_world = r.apply([1, 0, 0])
    # Compute NWU Yaw: North = 0, West = +90, East = -90
    yaw_nwu_rad = np.arctan2(-x_world[0], x_world[1])
    # Wrap to [-pi, +pi]
    imu_bearing = (yaw_nwu_rad + offset + np.pi) % (2.0 * np.pi) - np.pi
    return imu_bearing

def latlon_to_yaw(lat, lon, lat0, lon0, offset=0.0):
    lat, lon, lat0, lon0 = map(np.radians, [lat, lon, lat0, lon0])
    dlon = lon - lon0
    x = np.sin(dlon) * np.cos(lat)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    yaw = np.arctan2(-x, y)
    return ((yaw + offset) + np.pi) % (2 * np.pi) - np.pi

def hampel_filter(data, window_size=3, n_sigmas=3.0):
    s = pd.Series(data)
    rolling_median = s.rolling(window=window_size, min_periods=1, center=True).median()
    rolling_mad = (s - rolling_median).abs().rolling(window=window_size, min_periods=1, center=True).median()
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = (s - rolling_median).abs()
    outliers = difference > threshold
    s_filtered = s.copy()
    s_filtered[outliers] = rolling_median[outliers]
    return s_filtered.tolist()

def main():
    dataset_path = Path("/media/mf/SATA4TB/autodriving/datasetx/ugm_baru")
    meta_path = dataset_path / "meta"
    file_list = os.listdir(meta_path)
    file_list.sort()
    with open(meta_path / file_list[0], 'r') as first_filexx:
        first_filex = yaml.safe_load(first_filexx)
    with open(meta_path / file_list[-1], 'r') as last_filexx:
        last_filex = yaml.safe_load(last_filexx)

    meta = [yaml.safe_load(open(meta_path / meta_file, "r")) for meta_file in file_list]
    raw_lats = [m['global_position_latlon'][0] for m in meta]
    raw_lons = [m['global_position_latlon'][1] for m in meta]
    filtered_lats = hampel_filter(raw_lats)
    filtered_lons = hampel_filter(raw_lons)

    for idx, m in enumerate(meta):
        m['global_position_latlon'] = [filtered_lats[idx], filtered_lons[idx]]
    first_filex['global_position_latlon'] = [filtered_lats[0], filtered_lons[0]]
    last_filex['global_position_latlon'] = [filtered_lats[-1], filtered_lons[-1]]

    prev_lat = first_filex['global_position_latlon'][0]
    prev_lon = first_filex['global_position_latlon'][1]
    ref_bearing = []
    imu_bearing = []
    for i in tqdm(range(len(file_list)), desc="load latlon & imu with distance minima"):
        file_name = file_list[i]
        with open(meta_path / file_name, 'r') as curr_metafile:
            curr_meta = yaml.safe_load(curr_metafile)

        curr_imu = curr_meta['global_orientation_xyzw']
        curr_meta['global_position_latlon'] = [filtered_lats[i], filtered_lons[i]]
        velocity = np.abs(curr_meta["velocity"])
        curr_lat = curr_meta['global_position_latlon'][0]
        curr_lon = curr_meta['global_position_latlon'][1]
        dLat_m = (curr_lat - prev_lat) * 40008000 / 360 #111320 #Y
        dLon_m = (curr_lon - prev_lon) * 40075000 * np.cos(np.radians(curr_lat)) / 360 #X
        if np.sqrt(dLat_m**2 + dLon_m**2) >= 1.0 and velocity > 2.0: # kalau jarak antar dua latlon lebih dari 1 meter, jadikan patokan bearing
            latlon_yaw  = latlon_to_yaw(curr_lat, curr_lon, prev_lat, prev_lon)
            imu_yaw     = imu_to_yaw(curr_imu)
            ref_bearing.append(latlon_yaw)
            imu_bearing.append(imu_yaw)

            prev_lat = curr_meta['global_position_latlon'][0]
            prev_lon = curr_meta['global_position_latlon'][1]

    imu_bearing = np.array(imu_bearing)
    ref_bearing = np.array(ref_bearing)
    sector_bias = compute_imu_bearing_bias(imu_bearing, ref_bearing)
    print(sector_bias)
    with open("imu_bias.yml", 'w') as c:
        yaml.dump(sector_bias, c)
    

if __name__ == "__main__":
    main()