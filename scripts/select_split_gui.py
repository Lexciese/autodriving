import json
import re
from pathlib import Path
import sys

import numpy as np
import yaml
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update
from matplotlib.path import Path as MplPath
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GlobalConfig

config = GlobalConfig()
DATADIR = Path(config.datadir)

def available_routes():
    selected = config.select_route
    if selected != "all" and (DATADIR / selected / "meta").is_dir():
        return [selected]
    if DATADIR.is_dir():
        return sorted(p.name for p in DATADIR.iterdir() if (p / "meta").is_dir())
    return []


def load_meta_points(meta_dir):
    meta_dir = Path(meta_dir)
    points = []
    for index, yml_file in enumerate(sorted(meta_dir.glob("*.yml")), start=1):
        try:
            with open(yml_file, "r") as handle:
                data = yaml.safe_load(handle)
        except yaml.YAMLError:
            continue
        latlon = data.get("global_position_latlon") if isinstance(data, dict) else None
        if not latlon or len(latlon) != 2:
            continue
        lat, lon = float(latlon[0]), float(latlon[1])
        points.append(
            {"index": index, "filename": yml_file.name, "lat": lat, "lon": lon}
        )
    return points

def indices_in_shapes(points, shapes, claimed=None):
    if not points or not shapes:
        return []
    lon = np.array([p["lon"] for p in points])
    lat = np.array([p["lat"] for p in points])
    coords = np.column_stack([lon, lat])
    mask = np.zeros(len(points), dtype=bool)
    for shape in shapes:
        if shape["kind"] == "rect":
            x0, y0, x1, y1 = shape["x0"], shape["y0"], shape["x1"], shape["y1"]
            mask |= (
                (lon >= min(x0, x1))
                & (lon <= max(x0, x1))
                & (lat >= min(y0, y1))
                & (lat <= max(y0, y1))
            )
        else:
            mask |= MplPath(np.array(shape["points"], dtype=float)).contains_points(
                coords, radius=1e-9
            )
    claimed = claimed or set()
    return [points[i]["index"] for i in np.where(mask)[0] if points[i]["index"] not in claimed]


def split_regions(points, regions, ratios=(0.8, 0.1, 0.1)):
    by_index = {p["index"]: p for p in points}
    splits = {"train": [], "val": [], "test": []}
    for region in regions:
        indices = sorted(i for i in region["indices"] if i in by_index)
        n_total = len(indices)
        n_train = int(ratios[0] * n_total)
        n_val = int(ratios[1] * n_total)
        for position, index in enumerate(indices):
            entry = dict(by_index[index])
            entry["region"] = region["id"]
            if position < n_train:
                splits["train"].append(entry)
            elif position < n_train + n_val:
                splits["val"].append(entry)
            else:
                splits["test"].append(entry)
    return splits


def index_ranges(indices):
    ranges = []
    for index in sorted(indices):
        zero_based = index - 1
        if ranges and zero_based == ranges[-1][1] + 1:
            ranges[-1][1] = zero_based
        else:
            ranges.append([zero_based, zero_based])
    return ranges


def save_split(route, split, regions, ratios):
    grouped = {
        region["id"]: {name: [] for name in ("train", "val", "test")}
        for region in regions
    }
    for name in ("train", "val", "test"):
        for entry in split[name]:
            grouped[entry["region"]][name].append(entry["index"])

    def ranges_for(name):
        return [index_ranges(grouped[region["id"]][name]) for region in regions]

    lines = [
        f"route: {json.dumps(route)}",
        f"ratios: {json.dumps([float(r) for r in ratios])}",
        f"n_regions: {len(regions)}",
    ]
    if regions:
        lines.append("regions:")
        for region in regions:
            indices = region["indices"]
            lines.append(
                "- {id: %d, n_points: %d, index_min: %s, index_max: %s}"
                % (region["id"], len(indices),
                   min(indices) - 1 if indices else None,
                   max(indices) - 1 if indices else None)
            )
    for name in ("train", "val", "test"):
        lines.append(f"{name}: {json.dumps(ranges_for(name))}")

    out_path = DATADIR / route / "split.yml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return out_path


PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
SPLIT_COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}


def _scatter(points, name, color, size=8):
    return go.Scatter(
        x=[p["lon"] for p in points],
        y=[p["lat"] for p in points],
        mode="markers",
        name=name,
        marker=dict(size=size, color=color),
        text=[f"#{p['index']} {p['filename']}<br>lat={p['lat']:.6f} lon={p['lon']:.6f}"
              for p in points],
        hoverinfo="text",
    )


def build_figure(points, regions, split=None, color_by_split=False, shapes=None):
    fig = go.Figure()
    by_index = {p["index"]: p for p in points}

    if color_by_split and split is not None:
        for name in ("train", "val", "test"):
            group = [by_index[e["index"]] for e in split[name] if e["index"] in by_index]
            fig.add_trace(_scatter(group, f"{name} ({len(group)})", SPLIT_COLORS[name]))
    else:
        claimed = set()
        for region in regions:
            claimed |= set(region["indices"])
        unselected = [p for p in points if p["index"] not in claimed]
        fig.add_trace(_scatter(unselected, f"unselected ({len(unselected)})", "#d9d9d9", size=6))
        for region in regions:
            members = [by_index[i] for i in region["indices"] if i in by_index]
            color = PALETTE[(region["id"] - 1) % len(PALETTE)]
            fig.add_trace(_scatter(members, f"Region {region['id']} ({len(members)})", color))

    fig.update_layout(
        dragmode="drawrect",
        clickmode="event+select",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=20, t=30, b=45),
        height=720,
        uirevision="keep",
        shapes=shapes or [],
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig

routes = available_routes()
route_options = [{"label": r, "value": r} for r in routes]
default_route = routes[0] if routes else None

app = Dash(__name__)
app.title = "Region dataset splitter"

app.layout = html.Div(
    [
        html.H2("Region-based dataset splitter", style={"margin": "8px 0"}),
        dcc.Store(id="points-store"),
        dcc.Store(id="regions-store", data=[]),
        dcc.Store(id="split-store"),
        dcc.Store(id="shapes-store", data={}),
        html.Div(
            [
                html.Label("Route:", style={"fontWeight": "bold", "marginRight": "6px"}),
                dcc.Dropdown(
                    id="route-dropdown",
                    options=route_options,
                    value=default_route,
                    clearable=False,
                    style={"width": "340px", "display": "inline-block", "verticalAlign": "middle"},
                ),
                dcc.Checklist(
                    id="color-mode",
                    options=[{"label": "  Color by split", "value": "split"}],
                    value=[],
                    style={"display": "inline-block", "marginLeft": "16px", "verticalAlign": "middle"},
                ),
            ],
            style={"margin": "6px 0"},
        ),
        dcc.Graph(
            id="map",
            figure=build_figure([], []),
            config={
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawrect", "drawclosedpath", "eraseshape"],
            },
        ),
        html.Div(
            [
                html.Button("Commit Region", id="commit-btn", n_clicks=0),
                html.Button("Undo Last Region", id="undo-btn", n_clicks=0),
                html.Button("Clear All Regions", id="clear-btn", n_clicks=0),
                html.Button("Compute Split", id="split-btn", n_clicks=0),
                html.Button("Save split.yml", id="save-btn", n_clicks=0),
            ],
            style={"margin": "8px 0", "display": "flex", "gap": "8px"},
        ),
        html.Pre(id="status", style={"background": "#f5f5f5", "padding": "8px", "whiteSpace": "pre-wrap"}),
        html.Div(
            "Draw one or more shapes for a region using the rectangle tool or the "
            "'Draw closed path' lasso tool, click 'Commit Region', repeat for other "
            "regions, then 'Compute Split' and 'Save split.yml'.",
            style={"color": "#666", "fontSize": "12px"},
        ),
    ],
    style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "0 auto", "padding": "0 12px"},
)


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_path_points(path):
    """Extract ``(x, y)`` vertices from a plotly SVG-style path string."""
    numbers = [float(n) for n in _NUMBER.findall(path or "")]
    return [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers) - 1, 2)]


def _shape_key(shape):
    """Hashable key for a rect or path geometry dict."""
    if shape["kind"] == "rect":
        values = (shape["x0"], shape["y0"], shape["x1"], shape["y1"])
    else:
        values = tuple(coord for point in shape["points"] for coord in point)
    return (shape["kind"],) + tuple(round(float(v), 9) for v in values)


_SHAPE_KEY = re.compile(r"^shapes\[(\d+)\]\.(.+)$")


def _normalize_shape(shape):
    """Normalize one shape dict into rect/path geometry, or ``None``."""
    if not isinstance(shape, dict):
        return None
    if shape.get("path"):
        points = _parse_path_points(shape["path"])
        return {"kind": "path", "points": points} if len(points) >= 3 else None
    if all(k in shape for k in ("x0", "y0", "x1", "y1")):
        return {"kind": "rect", "x0": shape["x0"], "y0": shape["y0"],
                "x1": shape["x1"], "y1": shape["y1"]}
    return None


def _normalize_shapes(shape_store):
    """Normalize an index-keyed shape store into a geometry list."""
    if not shape_store:
        return []
    shapes = []
    for key in sorted(shape_store, key=lambda k: int(k)):
        geometry = _normalize_shape(shape_store[key])
        if geometry is not None:
            shapes.append(geometry)
    return shapes


def _merge_relayout(store, relayout):
    """Accumulate plotly relayout shape updates into an index-keyed store.

    A full ``shapes`` array replaces the store; incremental ``shapes[i].prop``
    keys (emitted while drawing freeform paths) are merged per index. Unrelated
    relayout keys (zoom/pan) leave the store untouched.
    """
    store = {str(k): dict(v) for k, v in (store or {}).items()}
    if not relayout:
        return store
    if "shapes" in relayout:
        return {str(i): dict(shape) for i, shape in enumerate(relayout["shapes"] or [])}
    for key, value in relayout.items():
        match = _SHAPE_KEY.match(key)
        if match:
            store.setdefault(match.group(1), {})[match.group(2)] = value
    return store


def _uncommitted_shapes(regions, shapes):
    """Drawn shapes that have not already been committed to a region."""
    existing = {_shape_key(s) for region in regions for s in region["shapes"]}
    return [s for s in shapes if _shape_key(s) not in existing]


@callback(
    Output("points-store", "data"),
    Output("regions-store", "data"),
    Output("split-store", "data"),
    Output("map", "figure"),
    Output("status", "children"),
    Output("shapes-store", "data"),
    Input("route-dropdown", "value"),
    Input("commit-btn", "n_clicks"),
    Input("undo-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    Input("split-btn", "n_clicks"),
    Input("save-btn", "n_clicks"),
    Input("color-mode", "value"),
    Input("map", "relayoutData"),
    State("points-store", "data"),
    State("regions-store", "data"),
    State("split-store", "data"),
    State("shapes-store", "data"),
)
def handle(route, n_commit, n_undo, n_clear, n_split, n_save, color_mode,
           relayout, points, regions, split, shapes_store):
    triggered = ctx.triggered_id
    color_by_split = bool(color_mode) and "split" in (color_mode or [])
    regions = regions or []

    # Load / reload points when the route changes (or on initial call).
    if triggered in (None, "route-dropdown") or points is None:
        points = load_meta_points(DATADIR / route / "meta") if route else []
        regions = []
        split = None
        fig = build_figure(points, regions)
        return (points, regions, split, fig,
                f"Loaded {len(points)} points from route '{route}'.", {})

    # Accumulate drawn shapes from plotly relayout events (rectangles emit a full
    # ``shapes`` array, freeform lassos emit incremental ``shapes[i].*`` keys).
    if triggered == "map":
        merged = _merge_relayout(shapes_store, relayout)
        return (no_update, no_update, no_update, no_update, no_update, merged)

    points = points or []
    routes_status = f"route '{route}'"

    if triggered == "commit-btn":
        drawn = _normalize_shapes(shapes_store)
        if not drawn:
            return (no_update, no_update, no_update, no_update,
                    "Draw a rectangle or lasso first.", no_update)
        new_shapes = _uncommitted_shapes(regions, drawn)
        if not new_shapes:
            return (no_update, no_update, no_update, no_update,
                    "Those shapes are already committed; draw a new one.", no_update)
        claimed = set()
        for region in regions:
            claimed |= set(region["indices"])
        indices = indices_in_shapes(points, new_shapes, claimed)
        if not indices:
            return (no_update, no_update, no_update, no_update,
                    "No unclaimed points inside the drawn shape(s).", no_update)
        region = {"id": len(regions) + 1, "shapes": new_shapes,
                  "indices": sorted(indices)}
        regions = regions + [region]
        split = None
        fig = build_figure(points, regions, split, color_by_split)
        return (no_update, regions, split, fig,
                f"Committed Region {region['id']} ({len(indices)} points).", {})

    if triggered == "undo-btn":
        if not regions:
            return (no_update, no_update, no_update, no_update, "No regions to undo.", no_update)
        removed = regions[-1]
        regions = regions[:-1]
        split = None
        fig = build_figure(points, regions, split, color_by_split)
        return (no_update, regions, split, fig, f"Removed Region {removed['id']}.", {})

    if triggered == "clear-btn":
        regions = []
        split = None
        fig = build_figure(points, regions)
        return (no_update, regions, split, fig, "Cleared all regions.", {})

    if triggered == "split-btn":
        if not regions:
            return (no_update, no_update, no_update, no_update,
                    "Commit at least one region first.", no_update)
        split = split_regions(points, regions)
        fig = build_figure(points, regions, split, True)
        summary = " | ".join(f"{k}: {len(v)}" for k, v in split.items())
        return (no_update, no_update, split, fig, f"Split {len(regions)} region(s) -> {summary}", no_update)

    if triggered == "save-btn":
        if split is None:
            return (no_update, no_update, no_update, no_update,
                    "Compute a split before saving.", no_update)
        out_path = save_split(route, split, regions, (0.8, 0.1, 0.1))
        return (no_update, no_update, no_update, no_update, f"Saved split to {out_path}", no_update)

    if triggered == "color-mode":
        fig = build_figure(points, regions, split, color_by_split)
        return (no_update, no_update, no_update, fig, f"View for {routes_status}.", no_update)

    # Fallback: refresh figure.
    fig = build_figure(points, regions, split, color_by_split)
    return (no_update, no_update, no_update, fig, no_update, no_update)


if __name__ == "__main__":
    if not routes:
        print(f"No routes with a 'meta' directory found under {DATADIR}")
    app.run(debug=False)
