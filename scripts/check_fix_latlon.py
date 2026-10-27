import sys
from pathlib import Path
import pandas as pd
import yaml
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GlobalConfig

config = GlobalConfig()

subfolders = sorted(p for p in config.datadir.iterdir() if p.is_dir() and (p / 'meta').exists())
if config.select_route != "all":
    subfolders = [config.datadir / config.select_route]
    print(f"Only route: {config.select_route} is selected")

for subfolder in subfolders:
    # 1. Read Raw Metadata Points
    lats, lons, labels = [], [], []
    yml_files = sorted((subfolder / 'meta').glob('*.yml'))

    for idx, yml_file in enumerate(yml_files, start=1):
        with open(yml_file, 'r') as f:
            data = yaml.safe_load(f)
        latlon = data.get('global_position_latlon')
        if latlon and len(latlon) == 2:
            lats.append(latlon[0])
            lons.append(latlon[1])
            labels.append(f"#{idx} {yml_file.name}")

    if not lats:
        print(f"Skipping {subfolder.name}: no valid points found.")
        continue

    # 2. Read Repaired GPS CSV
    fixed_csv_path = Path("ugm_baru_gnss_2_fixed_backward.csv")
    has_fixed_data = fixed_csv_path.exists()
    if has_fixed_data:
        fixed_df = pd.read_csv(fixed_csv_path)

    # 3. Create Interactive Plotly Figure
    fig = go.Figure()

    # Trace 1: Raw Meta Points (Red markers with Blue outline)
    fig.add_trace(go.Scatter(
        x=lons, y=lats,
        mode='markers',
        name='Raw Meta Points',
        marker=dict(size=8, color='red', symbol='circle-open', line=dict(width=2, color='red')),
        text=[f"Raw {label}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
              for label, lat, lon in zip(labels, lats, lons)],
        hoverinfo='text'
    ))

    # Trace 2: Repaired Trajectory Line (Solid Blue Line)
    if has_fixed_data:
        fig.add_trace(go.Scatter(
            x=fixed_df['longitude'], 
            y=fixed_df['latitude'],
            mode='lines+markers',
            name='Repaired Trajectory',
            line=dict(color='blue', width=2),
            marker=dict(size=4, color='blue'),
            text=[f"Timestamp: {ts}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}" 
                  for ts, lat, lon in zip(fixed_df['timestamp'], fixed_df['latitude'], fixed_df['longitude'])],
            hoverinfo='text'
        ))

    # 4. Layout & Equal Aspect Ratio
    fig.update_layout(
        title=f"GPS Trajectory Comparison: {subfolder.name}",
        xaxis_title='Longitude (deg)',
        yaxis_title='Latitude (deg)',
        hovermode='closest',
        width=1000, 
        height=700,
        legend=dict(x=0.01, y=0.99)
    )
    
    # Ensures 1 degree latitude = 1 degree longitude visually
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    output_file = f"comparison_trajectory_{subfolder.name}.html"
    fig.write_html(output_file)
    print(f"Saved {output_file} with raw and repaired points.")