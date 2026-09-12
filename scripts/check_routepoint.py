from pathlib import Path
import plotly.graph_objects as go
import yaml


def plot_yaml_route(file_path: str | Path, output_html: str = "route_plot.html"):
    file_path = Path(file_path)

    # 1. Read YAML file from disk
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 2. Extract start and end points
    first_pt = data.get("first_point", {})
    last_pt = data.get("last_point", {})

    # 3. Handle structure (route_point lats + root/nested longitudes)
    route_dict = data.get("route_point", {})
    route_lats = (
        route_dict.get("latitude", []) if isinstance(route_dict, dict) else []
    )
    route_lons = route_dict.get("longitude") or data.get("longitude", [])

    # Pair available lat/lon values
    route_coords = list(zip(route_lats, route_lons))

    fig = go.Figure()

    # Trace 1: Waypoints line & markers
    if route_coords:
        r_lats, r_lons = zip(*route_coords)
        fig.add_trace(
            go.Scatter(
                x=r_lons,
                y=r_lats,
                mode="lines+markers",
                name="Route Points",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=7, color="#1f77b4"),
                text=[
                    f"Point #{idx+1}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
                    for idx, (lat, lon) in enumerate(route_coords)
                ],
                hoverinfo="text",
            )
        )

    # Trace 2: First Point Marker
    if first_pt and "latitude" in first_pt:
        fig.add_trace(
            go.Scatter(
                x=[first_pt["longitude"]],
                y=[first_pt["latitude"]],
                mode="markers",
                name="First Point",
                marker=dict(size=14, color="green", symbol="circle"),
                text=[
                    f"First Point<br>Lat: {first_pt['latitude']:.6f}<br>Lon: {first_pt['longitude']:.6f}"
                ],
                hoverinfo="text",
            )
        )

    # Trace 3: Last Point Marker
    if last_pt and "latitude" in last_pt:
        fig.add_trace(
            go.Scatter(
                x=[last_pt["longitude"]],
                y=[last_pt["latitude"]],
                mode="markers",
                name="Last Point",
                marker=dict(size=14, color="red", symbol="x"),
                text=[
                    f"Last Point<br>Lat: {last_pt['latitude']:.6f}<br>Lon: {last_pt['longitude']:.6f}"
                ],
                hoverinfo="text",
            )
        )

    # Layout Setup
    fig.update_layout(
        title=f"Route Points from: {file_path.name}",
        xaxis_title="Longitude (deg)",
        yaxis_title="Latitude (deg)",
        hovermode="closest",
        width=1000,
        height=700,
        template="plotly_white",
    )

    # Lock 1:1 aspect ratio for geospatial coordinates
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Save output to HTML file
    fig.write_html(output_html)
    print(
        f"Saved plot to {output_html} ({len(route_coords)} route points plotted)."
    )
    return fig


# Usage example:
plot_yaml_route("/media/mf/SATA4TB/autodriving/datasetx/2026-09-12_route00/2026-09-12_route00_routepoint_list_old_math.yml", output_html="2026-09-12_route00_routepoint_list_old_math")