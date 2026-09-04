import yaml
import plotly.graph_objects as go
from pathlib import Path

meta_folder = Path('../datasetx/meta')

lats, lons, labels = [], [], []

for yml_file in meta_folder.glob('*.yml'):
    with open(yml_file, 'r') as f:
        data = yaml.safe_load(f)
    latlon = data.get('global_position_latlon')
    if latlon and len(latlon) == 2:
        lats.append(latlon[0])
        lons.append(latlon[1])
        labels.append(yml_file.name)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=lons, y=lats,
    mode='lines+markers',
    name='Meta points',
    line=dict(color='blue', width=1),
    marker=dict(size=8, color='red'),
    text=[f"File: {label}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
          for label, lat, lon in zip(labels, lats, lons)],
    hoverinfo='text'
))
fig.update_layout(
    title='Points from YAML files (global_position_latlon)',
    xaxis_title='Longitude (deg)',
    yaxis_title='Latitude (deg)',
    hovermode='closest',
    width=1000, height=700,
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.write_html('check_latlon.html')