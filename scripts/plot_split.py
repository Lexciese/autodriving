"""Plot the regions and train/val/test split stored in a ``split.yml``.

The split file is written by ``scripts/select_split_gui.py`` and stores 0-based,
inclusive index ranges per region (indices follow the sorted ``meta/*.yml`` file
order). This script maps those indices back to latitude/longitude, colours the
points by split (or by region), and writes a static interactive plotly HTML.

Run with the ``dl`` conda environment::

    conda run -n dl python scripts/plot_split.py
    conda run -n dl python scripts/plot_split.py --split datasetx/<route>/split.yml --open
"""

import argparse
from pathlib import Path
import sys
import webbrowser

import plotly.graph_objects as go
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPLIT_ORDER = ("train", "val", "test")
SPLIT_COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
SPLIT_SYMBOLS = {"train": "circle", "val": "diamond", "test": "square"}
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_config():
    """Import the repo config lazily so explicit --split works without it."""
    from config import GlobalConfig

    return GlobalConfig()


def _default_split_path(cfg):
    """``<datadir>/<select_route>/split.yml`` when a concrete route is selected."""
    selected = cfg.select_route
    if selected and selected != "all":
        return Path(cfg.datadir) / selected / "split.yml"
    return None


def _available_splits(cfg):
    datadir = Path(cfg.datadir)
    if not datadir.is_dir():
        return []
    return sorted(datadir.glob("*/split.yml"))


def load_meta_points(meta_dir):
    """Return ``[{index, filename, lat, lon}]`` from sorted ``*.yml`` files.

    ``index`` is 1-based over *all* sorted files (invalid entries still consume an
    index), so the 0-based index in ``split.yml`` is ``index - 1``.
    """
    points = []
    for index, yml_file in enumerate(sorted(Path(meta_dir).glob("*.yml")), start=1):
        try:
            with open(yml_file, "r") as handle:
                data = yaml.safe_load(handle)
        except yaml.YAMLError:
            continue
        latlon = data.get("global_position_latlon") if isinstance(data, dict) else None
        if not latlon or len(latlon) != 2:
            continue
        points.append({
            "index": index,
            "filename": yml_file.name,
            "lat": float(latlon[0]),
            "lon": float(latlon[1]),
        })
    return points


def load_split(split_path):
    with open(split_path, "r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{split_path} is not a valid split file")
    return data


def expand_ranges(ranges):
    """Expand inclusive 0-based ``[[start, end], ...]`` into a set of indices."""
    indices = set()
    for bounds in ranges or []:
        start, end = int(bounds[0]), int(bounds[1])
        indices.update(range(start, end + 1))
    return indices


def collect_split(split_data):
    """Build per ``(region, split)`` index sets and validate the file.

    Returns ``(members, region_ids, region_meta, warnings)``.
    """
    regions = split_data.get("regions") or []
    n_regions = len(regions)
    if n_regions == 0:
        n_regions = max((len(split_data.get(name) or []) for name in SPLIT_ORDER), default=0)

    region_ids, region_meta = [], {}
    for position in range(n_regions):
        summary = regions[position] if position < len(regions) and isinstance(regions[position], dict) else {}
        region_id = summary.get("id", position + 1)
        region_ids.append(region_id)
        region_meta[region_id] = summary

    members = {}
    warnings = []
    owner = {}
    for position, region_id in enumerate(region_ids):
        union = set()
        for name in SPLIT_ORDER:
            per_region = split_data.get(name) or []
            ranges = per_region[position] if position < len(per_region) else []
            indices = expand_ranges(ranges)
            members[(region_id, name)] = indices
            union |= indices
        for index in union:
            if index in owner and owner[index] != region_id:
                warnings.append(f"index {index} appears in regions {owner[index]} and {region_id}")
            owner[index] = region_id
        summary = region_meta[region_id]
        expected = summary.get("n_points")
        if expected is not None and int(expected) != len(union):
            warnings.append(
                f"region {region_id}: n_points={expected} but ranges cover {len(union)} indices"
            )
    return members, region_ids, region_meta, warnings


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _hover_text(points, index_region, index_split):
    return [
        f"index0={p['index'] - 1} index1={p['index']}<br>"
        f"{p['filename']}<br>"
        f"lat={p['lat']:.6f} lon={p['lon']:.6f}<br>"
        f"region={index_region.get(p['index'] - 1, '-')} "
        f"split={index_split.get(p['index'] - 1, '-')}"
        for p in points
    ]


def _scatter(points, name, color, index_region, index_split, symbol="circle", size=8):
    return go.Scatter(
        x=[p["lon"] for p in points],
        y=[p["lat"] for p in points],
        mode="markers",
        name=name,
        marker=dict(size=size, color=color, symbol=symbol),
        text=_hover_text(points, index_region, index_split),
        hoverinfo="text",
    )


def build_figure(points, members, region_ids, region_meta, split_data, color_by="split"):
    by_index = {p["index"] - 1: p for p in points}
    index_region, index_split = {}, {}
    for (region_id, name), indices in members.items():
        for index in indices:
            index_region[index] = region_id
            index_split[index] = name

    selected = set(index_region)
    fig = go.Figure()

    unselected = [p for p in points if p["index"] - 1 not in selected]
    if unselected:
        fig.add_trace(
            _scatter(unselected, f"unselected ({len(unselected)})", "#cfcfcf",
                     index_region, index_split, size=6)
        )

    if color_by == "region":
        for region_id in region_ids:
            color = PALETTE[(region_id - 1) % len(PALETTE)]
            for name in SPLIT_ORDER:
                indices = sorted(i for i in members.get((region_id, name), set()) if i in by_index)
                if not indices:
                    continue
                members_points = [by_index[i] for i in indices]
                fig.add_trace(_scatter(
                    members_points, f"Region {region_id} · {name} ({len(indices)})",
                    color, index_region, index_split, symbol=SPLIT_SYMBOLS[name],
                ))
    else:
        for name in SPLIT_ORDER:
            indices = sorted(
                i for region_id in region_ids
                for i in members.get((region_id, name), set())
                if i in by_index
            )
            if not indices:
                continue
            fig.add_trace(_scatter(
                [by_index[i] for i in indices], f"{name} ({len(indices)})",
                SPLIT_COLORS[name], index_region, index_split,
            ))

    route = split_data.get("route", "?")
    ratios = split_data.get("ratios", [])
    fig.update_layout(
        title=f"Split for route '{route}' (ratios={ratios}, regions={len(region_ids)})",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=70, b=45),
        height=760,
    )
    return fig


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot a split.yml with plotly.")
    parser.add_argument(
        "--split", type=Path, default=None,
        help="Path to split.yml (default: <datadir>/<select_route>/split.yml).",
    )
    parser.add_argument(
        "--meta", type=Path, default=None,
        help="Directory of meta *.yml files (default: <split parent>/meta).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output HTML path (default: <split parent>/split_plot.html).",
    )
    parser.add_argument(
        "--color-by", choices=("split", "region"), default="split",
        help="Colour points by split (default) or by region.",
    )
    parser.add_argument("--open", action="store_true", help="Open the HTML in a browser.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.split is not None:
        split_path = Path(args.split)
        cfg = None
    else:
        try:
            cfg = _load_config()
        except Exception as exc:  # noqa: BLE001 - surface a friendly message
            print(f"Could not load config: {exc}")
            print("Pass --split <path/to/split.yml> explicitly.")
            return 1
        split_path = _default_split_path(cfg)

    if split_path is None or not split_path.is_file():
        print(f"split.yml not found: {split_path}")
        if cfg is None:
            try:
                cfg = _load_config()
            except Exception:  # noqa: BLE001
                cfg = None
        available = _available_splits(cfg) if cfg is not None else []
        if available:
            print("Available split files:")
            for path in available:
                print(f"  {path}")
        else:
            print("No split.yml files found in the configured datadir")
        return 1

    meta_dir = args.meta or split_path.parent / "meta"
    out_path = args.out or split_path.parent / "split_plot.html"

    points = load_meta_points(meta_dir)
    if not points:
        print(f"No valid lat/lon points found in {meta_dir}")
        return 1

    split_data = load_split(split_path)
    members, region_ids, region_meta, warnings = collect_split(split_data)

    for warning in warnings:
        print(f"WARNING: {warning}")

    known = {p["index"] - 1 for p in points}
    missing = sorted({i for indices in members.values() for i in indices} - known)
    if missing:
        preview = ", ".join(str(i) for i in missing[:10])
        more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        print(f"WARNING: {len(missing)} split indices have no meta point: {preview}{more}")

    fig = build_figure(points, members, region_ids, region_meta, split_data, args.color_by)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)

    print(f"Loaded {len(points)} meta points from {meta_dir}")
    for name in SPLIT_ORDER:
        count = sum(len(members.get((region_id, name), set())) for region_id in region_ids)
        print(f"  {name}: {count} points")
    print(f"Regions: {len(region_ids)}")
    print(f"Saved plot to {out_path}")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
