import plotly.graph_objects as go
from tqdm import tqdm
from ai23.config import GlobalConfig
from ai23.dataloader import KarrDataset

def plot_dataset_trajectory(
    dataset: KarrDataset, 
    start_idx: int = 0, 
    end_idx: int = None, 
    output_file: str = "robot_trajectory.html"
):
    lats, lons, labels = [], [], []
    
    total_samples = len(dataset)
    if end_idx is None or end_idx > total_samples:
        end_idx = total_samples
    if start_idx < 0:
        start_idx = 0
        
    if start_idx >= end_idx:
        print(f"Invalid range: start_idx ({start_idx}) must be less than end_idx ({end_idx}).")
        return

    print(f"Extracting telemetry data from index {start_idx} to {end_idx - 1}...")
    for idx in tqdm(range(start_idx, end_idx), desc="Processing samples"):
        sample = dataset[idx]
        
        lat = sample['lat_robot']
        lon = sample['lon_robot']
        filename = sample['filename']
        
        lats.append(lat)
        lons.append(lon)
        labels.append(f"#{idx} {filename}")
        
    if not lats:
        print("No points extracted. Skipping plot.")
        return

    print("Generating Plotly figure...")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lons, 
        y=lats,
        mode='lines+markers',
        name='Robot Trajectory',
        line=dict(color='blue', width=1),
        marker=dict(size=6, color='red'),
        text=[f"{label}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}" 
              for label, lat, lon in zip(labels, lats, lons)],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"Robot Trajectory (Frames {start_idx} to {end_idx - 1})",
        xaxis_title='Longitude (deg)',
        yaxis_title='Latitude (deg)',
        hovermode='closest',
        width=1000, 
        height=700,
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.write_html(output_file)
    print(f"Saved interactive trajectory to {output_file} with {len(lats)} points.")


if __name__ == "__main__":
    # Initialize your dataset
    config = GlobalConfig()
    dataset = KarrDataset(config)
    
    plot_dataset_trajectory(
        dataset=dataset, 
        start_idx=0, 
        end_idx=700, 
        output_file="trajectory_dataloader.html"
    )