import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GlobalConfig

configx = GlobalConfig()

route_list = sorted(p.name for p in configx.datadir.iterdir() if p.is_dir())
if configx.select_route != "all":
    route_list = [configx.select_route]
    print(f"only route: {configx.select_route} is selected")

for route in route_list:
    route_path = configx.datadir / route
    ddir_meta = route_path / "meta"
    file_list = sorted(p.name for p in ddir_meta.iterdir())
    meta = [yaml.safe_load(open(ddir_meta / meta_file, "r")) for meta_file in file_list]
    timestamps = [Path(meta_file).stem for meta_file in file_list]
    lats = [m['global_position_latlon'][0] for m in meta]
    lons = [m['global_position_latlon'][1] for m in meta]

    gnss_path = route_path / "ugm_baru_gnss.csv"
    gnss_df = pd.DataFrame({'timestamp': timestamps, 'latitude': lats, 'longitude': lons})
    gnss_df.to_csv(gnss_path, index=False)

    df = pd.read_csv(gnss_path)

    lat_col = 'latitude'
    lon_col = 'longitude'

    df['Dist_prev'] = np.sqrt((df[lat_col] - df[lat_col].shift(1))**2 + (df[lon_col] - df[lon_col].shift(1))**2)
    df['Dist_next'] = np.sqrt((df[lat_col] - df[lat_col].shift(-1))**2 + (df[lon_col] - df[lon_col].shift(-1))**2)
    df['Dist_prev_next'] = np.sqrt((df[lat_col].shift(1) - df[lat_col].shift(-1))**2 + (df[lon_col].shift(1) - df[lon_col].shift(-1))**2)

    rolling_median = df['Dist_prev'].rolling(window=20, center=True, min_periods=1).median()
    dynamic_threshold = rolling_median * 2.0  

    anomaly_mask = (df['Dist_prev'] > dynamic_threshold) & \
                (df['Dist_next'] > dynamic_threshold) & \
                (df['Dist_prev_next'] < (df['Dist_prev'] * 0.5))

    print(f"Total dynamic anomalies detected: {anomaly_mask.sum()}")

    df_fixed = df.copy()
    df_fixed.loc[anomaly_mask, [lat_col, lon_col]] = np.nan

    df_fixed[lat_col] = df_fixed[lat_col].interpolate(method='polynomial', order=2)
    df_fixed[lon_col] = df_fixed[lon_col].interpolate(method='polynomial', order=2)

    df_fixed[lat_col] = df_fixed[lat_col].interpolate(method='linear')
    df_fixed[lon_col] = df_fixed[lon_col].interpolate(method='linear')

    columns_to_keep = [col for col in df.columns if col not in ['Dist_prev', 'Dist_next', 'Dist_prev_next']]
    df_fixed[columns_to_keep].to_csv(gnss_path, index=False)
    print(f"saved to {gnss_path}")