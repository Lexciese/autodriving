from __future__ import annotations

import base64
import copy
import csv
import math
import os
import shutil
import tempfile
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml

ROUTE_PRECISION = 7
LAT_METERS_PER_DEGREE = 40008000.0 / 360.0
LON_METERS_PER_DEGREE = 40075000.0 / 360.0
OSM_TILE_URL = os.environ.get(
    "ROUTE_EDITOR_OSM_URL", "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
)
OSM_USER_AGENT = "routepoint-editor/1.0 (local route audit tool)"
OSM_ATTRIBUTION = "OpenStreetMap contributors"


class RoutePointError(ValueError):
    pass


class OSMError(RuntimeError):
    pass


def _as_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RoutePointError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise RoutePointError(f"{label} must be finite")
    return result


def _parse_point(value, label):
    if not isinstance(value, dict):
        raise RoutePointError(f"{label} must be a mapping with latitude/longitude")
    latitude = _as_float(value.get("latitude"), f"{label}.latitude")
    longitude = _as_float(value.get("longitude"), f"{label}.longitude")
    return {"latitude": latitude, "longitude": longitude}


def _round_coord(value):
    return round(float(value), ROUTE_PRECISION)


def validate_route_document(doc):
    if not isinstance(doc, dict):
        raise RoutePointError("Route document must be a mapping")
    _parse_point(doc.get("first_point"), "first_point")
    _parse_point(doc.get("last_point"), "last_point")
    route = doc.get("route_point")
    if not isinstance(route, dict):
        raise RoutePointError("route_point must be a mapping")
    lats = route.get("latitude")
    lons = route.get("longitude")
    if not isinstance(lats, list) or not isinstance(lons, list):
        raise RoutePointError("route_point.latitude and longitude must be lists")
    if len(lats) != len(lons):
        raise RoutePointError(
            f"latitude/longitude length mismatch: {len(lats)} != {len(lons)}"
        )
    if not lats:
        raise RoutePointError("route_point must contain at least one point")
    for index, (lat, lon) in enumerate(zip(lats, lons)):
        _as_float(lat, f"route_point.latitude[{index}]")
        _as_float(lon, f"route_point.longitude[{index}]")


def load_route_document(path):
    path = Path(path).expanduser()
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise RoutePointError("YAML root must be a mapping")
    first_point = _parse_point(raw.get("first_point"), "first_point")
    last_point = _parse_point(raw.get("last_point"), "last_point")
    route_raw = raw.get("route_point")
    if not isinstance(route_raw, dict):
        raise RoutePointError("YAML must contain a route_point mapping")
    lats = route_raw.get("latitude")
    lons = route_raw.get("longitude")
    if not isinstance(lats, list) or not isinstance(lons, list):
        raise RoutePointError("route_point.latitude and longitude must be lists")
    if len(lats) != len(lons):
        raise RoutePointError(
            f"latitude/longitude length mismatch: {len(lats)} != {len(lons)}"
        )
    if not lats:
        raise RoutePointError("route_point must contain at least one point")
    lats = [_as_float(value, f"route_point.latitude[{i}]") for i, value in enumerate(lats)]
    lons = [_as_float(value, f"route_point.longitude[{i}]") for i, value in enumerate(lons)]
    extra = {
        key: value
        for key, value in raw.items()
        if key not in {"first_point", "last_point", "route_point"}
    }
    route_extra = {
        key: value
        for key, value in route_raw.items()
        if key not in {"latitude", "longitude"}
    }
    doc = {
        "first_point": first_point,
        "last_point": last_point,
        "route_point": {"latitude": lats, "longitude": lons},
        "extra": extra,
        "route_extra": route_extra,
        "origin": {
            "latitude": first_point["latitude"],
            "longitude": first_point["longitude"],
        },
        "path": str(path),
    }
    validate_route_document(doc)
    return doc


def copy_route_document(doc):
    return copy.deepcopy(doc)


def origin_of(doc):
    origin = doc.get("origin")
    if isinstance(origin, dict):
        return (
            _as_float(origin.get("latitude"), "origin.latitude"),
            _as_float(origin.get("longitude"), "origin.longitude"),
        )
    first = doc["first_point"]
    return float(first["latitude"]), float(first["longitude"])


def latlon_to_local(lat, lon, lat0, lon0):
    lat_values = np.asarray(lat, dtype=float)
    lon_values = np.asarray(lon, dtype=float)
    x = (lon_values - float(lon0)) * LON_METERS_PER_DEGREE * math.cos(
        math.radians(float(lat0))
    )
    y = (lat_values - float(lat0)) * LAT_METERS_PER_DEGREE
    if x.ndim == 0:
        return float(x), float(y)
    return x, y


def local_to_latlon(x, y, lat0, lon0):
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    lat = float(lat0) + y_values / LAT_METERS_PER_DEGREE
    meters_per_lon = LON_METERS_PER_DEGREE * np.cos(np.radians(lat))
    lon = float(lon0) + x_values / meters_per_lon
    lat = float(lat0) + y_values / LAT_METERS_PER_DEGREE
    meters_per_lon = LON_METERS_PER_DEGREE * np.cos(np.radians(lat))
    lon = float(lon0) + x_values / meters_per_lon
    if lat.ndim == 0:
        return float(lat), float(lon)
    return lat, lon


def route_local_xy(doc):
    lat0, lon0 = origin_of(doc)
    lats = np.asarray(doc["route_point"]["latitude"], dtype=float)
    lons = np.asarray(doc["route_point"]["longitude"], dtype=float)
    return latlon_to_local(lats, lons, lat0, lon0)


def point_local_xy(doc, lat, lon):
    lat0, lon0 = origin_of(doc)
    return latlon_to_local(lat, lon, lat0, lon0)


def selection_local_xy(doc, selected):
    if not selected:
        return None
    kind = selected.get("kind", "route")
    if kind == "route":
        index = int(selected.get("index", 0))
        x, y = route_local_xy(doc)
        x_values = np.asarray(x, dtype=float)
        y_values = np.asarray(y, dtype=float)
        if 0 <= index < len(x_values):
            return float(x_values[index]), float(y_values[index])
        return None
    point = doc.get("first_point" if kind == "first" else "last_point")
    if not isinstance(point, dict):
        return None
    return point_local_xy(doc, point["latitude"], point["longitude"])


def set_route_point(doc, index, lat, lon):
    new_doc = copy_route_document(doc)
    lats = new_doc["route_point"]["latitude"]
    lons = new_doc["route_point"]["longitude"]
    if not 0 <= int(index) < len(lats):
        raise RoutePointError("Route point index out of range")
    lats[int(index)] = _round_coord(_as_float(lat, "latitude"))
    lons[int(index)] = _round_coord(_as_float(lon, "longitude"))
    return new_doc


def set_endpoint(doc, kind, lat, lon):
    if kind not in {"first", "last"}:
        raise RoutePointError("Endpoint must be 'first' or 'last'")
    new_doc = copy_route_document(doc)
    key = "first_point" if kind == "first" else "last_point"
    new_doc[key] = {
        "latitude": _round_coord(_as_float(lat, "latitude")),
        "longitude": _round_coord(_as_float(lon, "longitude")),
    }
    return new_doc


def insert_route_point(doc, index, lat, lon):
    new_doc = copy_route_document(doc)
    lats = new_doc["route_point"]["latitude"]
    lons = new_doc["route_point"]["longitude"]
    index = int(index)
    if not 0 <= index <= len(lats):
        raise RoutePointError("Insert index out of range")
    lats.insert(index, _round_coord(_as_float(lat, "latitude")))
    lons.insert(index, _round_coord(_as_float(lon, "longitude")))
    return new_doc


def insert_midpoint(doc, index):
    lats = doc["route_point"]["latitude"]
    lons = doc["route_point"]["longitude"]
    count = len(lats)
    if count < 2:
        raise RoutePointError("Need at least two route points to insert a midpoint")
    index = max(0, min(int(index), count - 1))
    if index < count - 1:
        neighbor = index + 1
        insert_at = index + 1
    else:
        neighbor = index - 1
        insert_at = index
    lat = (lats[index] + lats[neighbor]) / 2.0
    lon = (lons[index] + lons[neighbor]) / 2.0
    return insert_route_point(doc, insert_at, lat, lon), insert_at


def delete_route_point(doc, index):
    lats = doc["route_point"]["latitude"]
    count = len(lats)
    if count <= 1:
        raise RoutePointError("Cannot delete the only route point")
    index = max(0, min(int(index), count - 1))
    new_doc = copy_route_document(doc)
    new_doc["route_point"]["latitude"].pop(index)
    new_doc["route_point"]["longitude"].pop(index)
    return new_doc, max(0, min(index, count - 2))


def smooth_route_point(doc, index, radius=1):
    lats = doc["route_point"]["latitude"]
    lons = doc["route_point"]["longitude"]
    count = len(lats)
    if count < 2:
        raise RoutePointError("Need at least two route points to smooth")
    index = max(0, min(int(index), count - 1))
    radius = max(1, int(radius))
    start = max(0, index - radius)
    stop = min(count, index + radius + 1)
    lat = float(np.mean(lats[start:stop]))
    lon = float(np.mean(lons[start:stop]))
    return set_route_point(doc, index, lat, lon)


def normalize_selection(selected, route_doc):
    count = len(route_doc["route_point"]["latitude"]) if route_doc else 0
    if not count:
        return None
    if not selected:
        return {"kind": "route", "index": 0}
    kind = selected.get("kind", "route")
    if kind in {"first", "last"}:
        return {"kind": kind}
    index = int(selected.get("index", 0))
    return {"kind": "route", "index": max(0, min(index, count - 1))}


def _flag_counts(flags):
    counts = {"gap": 0, "tight": 0, "duplicate": 0, "turn": 0, "deviation": 0}
    for flag in flags:
        flag_type = flag.get("type")
        if flag_type in counts:
            counts[flag_type] += 1
    return counts


def compute_audit(
    doc,
    min_gap=2.0,
    max_gap=12.0,
    turn_threshold_deg=60.0,
    duplicate_gap=0.05,
):
    x, y = route_local_xy(doc)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    count = len(x)
    dx = np.diff(x)
    dy = np.diff(y)
    segment_distance = np.hypot(dx, dy)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_distance)))
    turn_angle = np.zeros(count, dtype=float)
    if count >= 3:
        v1x, v1y = dx[:-1], dy[:-1]
        v2x, v2y = dx[1:], dy[1:]
        denom = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
        dot = v1x * v2x + v1y * v2y
        cosine = np.divide(
            dot,
            denom,
            out=np.ones_like(denom),
            where=denom > 0,
        )
        turn_angle[1:-1] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    flags = []
    for index, distance in enumerate(segment_distance):
        distance = float(distance)
        if distance < duplicate_gap:
            flags.append(
                {
                    "index": index,
                    "end_index": index + 1,
                    "type": "duplicate",
                    "value": round(distance, 3),
                    "detail": f"{distance:.3f} m between consecutive points",
                }
            )
        elif distance > max_gap:
            flags.append(
                {
                    "index": index,
                    "end_index": index + 1,
                    "type": "gap",
                    "value": round(distance, 3),
                    "detail": f"{distance:.2f} m gap exceeds {max_gap:.2f} m",
                }
            )
        elif distance < min_gap:
            flags.append(
                {
                    "index": index,
                    "end_index": index + 1,
                    "type": "tight",
                    "value": round(distance, 3),
                    "detail": f"{distance:.2f} m spacing is below {min_gap:.2f} m",
                }
            )
    for index in range(1, count - 1):
        angle = float(turn_angle[index])
        if angle > turn_threshold_deg:
            flags.append(
                {
                    "index": index,
                    "end_index": None,
                    "type": "turn",
                    "value": round(angle, 2),
                    "detail": f"{angle:.1f} deg turn exceeds {turn_threshold_deg:.1f} deg",
                }
            )
    flags.sort(key=lambda item: (int(item["index"]), str(item["type"])))
    counts = _flag_counts(flags)
    summary = {
        "point_count": int(count),
        "segment_count": int(max(0, count - 1)),
        "total_length_m": float(cumulative[-1]) if count else 0.0,
        "min_spacing_m": float(np.min(segment_distance)) if count > 1 else 0.0,
        "mean_spacing_m": float(np.mean(segment_distance)) if count > 1 else 0.0,
        "median_spacing_m": float(np.median(segment_distance)) if count > 1 else 0.0,
        "max_spacing_m": float(np.max(segment_distance)) if count > 1 else 0.0,
        "flag_count": int(len(flags)),
        "flag_counts": counts,
    }
    return {
        "segment_distance": segment_distance.tolist(),
        "cumulative_distance": cumulative.tolist(),
        "turn_angle_deg": turn_angle.tolist(),
        "flags": flags,
        "summary": summary,
    }


def detect_reference_path(yaml_path):
    path = Path(yaml_path).expanduser()
    candidates = []
    suffixes = ("_routepoint_list.yml", "_routepoint_list.yaml")
    for suffix in suffixes:
        if path.name.endswith(suffix):
            base = path.name[: -len(suffix)]
            candidates.append(path.with_name(f"{base}_gnss.csv"))
    candidates.append(path.with_name(f"{path.stem}_gnss.csv"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    fallback = sorted(path.parent.glob("*_gnss.csv"))
    if len(fallback) == 1:
        return fallback[0]
    return None


def load_reference_csv(path):
    path = Path(path).expanduser()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RoutePointError("Reference CSV has no header")
        lookup = {name.strip().lower(): name for name in reader.fieldnames}
        lat_key = next(
            (lookup[key] for key in ("latitude", "lat") if key in lookup),
            None,
        )
        lon_key = next(
            (lookup[key] for key in ("longitude", "lon", "lng") if key in lookup),
            None,
        )
        if lat_key is None or lon_key is None:
            raise RoutePointError(
                "Reference CSV needs latitude/longitude or lat/lon columns"
            )
        lats = []
        lons = []
        for row in reader:
            try:
                lat = float(row[lat_key])
                lon = float(row[lon_key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(lat) or not math.isfinite(lon):
                continue
            lats.append(lat)
            lons.append(lon)
    if not lats:
        raise RoutePointError("Reference CSV contains no valid coordinates")
    return {
        "path": str(path),
        "latitude": lats,
        "longitude": lons,
    }


def reference_deviation(doc, reference, max_points=20000):
    if not reference:
        return None
    ref_lats = reference.get("latitude") or []
    ref_lons = reference.get("longitude") or []
    if not ref_lats or len(ref_lats) != len(ref_lons):
        return None
    lat0, lon0 = origin_of(doc)
    ref_x, ref_y = latlon_to_local(ref_lats, ref_lons, lat0, lon0)
    ref_points = np.column_stack([np.asarray(ref_x, dtype=float), np.asarray(ref_y, dtype=float)])
    if len(ref_points) > max_points:
        indices = np.linspace(0, len(ref_points) - 1, max_points).astype(int)
        ref_points = ref_points[indices]
    x, y = route_local_xy(doc)
    route_points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    try:
        import importlib

        spatial = importlib.import_module("scipy.spatial")
        tree = spatial.cKDTree(ref_points)
        distances, _ = tree.query(route_points, k=1)
        distances = np.asarray(distances, dtype=float)
    except (ImportError, AttributeError):
        distances = np.empty(len(route_points), dtype=float)
        for start in range(0, len(route_points), 128):
            chunk = route_points[start : start + 128]
            delta = chunk[:, None, :] - ref_points[None, :, :]
            distances[start : start + len(chunk)] = np.sqrt(
                np.sum(delta * delta, axis=2)
            ).min(axis=1)
    return {
        "distance_m": distances.tolist(),
        "max_m": float(np.max(distances)) if len(distances) else 0.0,
        "mean_m": float(np.mean(distances)) if len(distances) else 0.0,
    }


def add_reference_audit(audit, doc, reference, threshold=8.0):
    deviation = reference_deviation(doc, reference)
    if deviation is None:
        return audit
    updated = dict(audit)
    flags = list(updated.get("flags") or [])
    for index, distance in enumerate(deviation["distance_m"]):
        if distance > threshold:
            flags.append(
                {
                    "index": index,
                    "end_index": None,
                    "type": "deviation",
                    "value": round(float(distance), 3),
                    "detail": f"{distance:.2f} m from reference exceeds {threshold:.2f} m",
                }
            )
    flags.sort(key=lambda item: (int(item["index"]), str(item["type"])))
    summary = dict(updated.get("summary") or {})
    counts = _flag_counts(flags)
    summary["flag_count"] = int(len(flags))
    summary["flag_counts"] = counts
    summary["reference_mean_m"] = deviation["mean_m"]
    summary["reference_max_m"] = deviation["max_m"]
    updated["flags"] = flags
    updated["summary"] = summary
    updated["reference_distance_m"] = deviation["distance_m"]
    return updated


def write_audit_csv(audit, path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "point", "type", "value", "detail"])
        for flag in audit.get("flags") or []:
            writer.writerow(
                [
                    int(flag.get("index", 0)),
                    int(flag.get("index", 0)) + 1,
                    flag.get("type", ""),
                    flag.get("value", ""),
                    flag.get("detail", ""),
                ]
            )
    return path


def document_to_yaml_data(doc):
    validate_route_document(doc)
    route = {}
    for key, value in (doc.get("route_extra") or {}).items():
        route[key] = value
    route["latitude"] = [
        _round_coord(_as_float(value, "latitude"))
        for value in doc["route_point"]["latitude"]
    ]
    route["longitude"] = [
        _round_coord(_as_float(value, "longitude"))
        for value in doc["route_point"]["longitude"]
    ]
    data = {
        "first_point": {
            "latitude": _round_coord(doc["first_point"]["latitude"]),
            "longitude": _round_coord(doc["first_point"]["longitude"]),
        },
        "last_point": {
            "latitude": _round_coord(doc["last_point"]["latitude"]),
            "longitude": _round_coord(doc["last_point"]["longitude"]),
        },
        "route_point": route,
    }
    for key, value in (doc.get("extra") or {}).items():
        data[key] = value
    return data


def _atomic_write(path, text):
    path = Path(path)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temp_name, path.stat().st_mode)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def save_route_document(doc, path, create_backup=True):
    validate_route_document(doc)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = document_to_yaml_data(doc)
    text = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    backup_path = None
    if path.exists() and create_backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_path = path.with_name(f"{path.name}.bak-{stamp}")
        shutil.copy2(path, backup_path)
    _atomic_write(path, text)
    return backup_path


def _lon_to_tile_x(lon, zoom):
    return (float(lon) + 180.0) / 360.0 * (2 ** zoom)


def _lat_to_tile_y(lat, zoom):
    lat = max(-85.05112878, min(85.05112878, float(lat)))
    lat_rad = math.radians(lat)
    return (
        1.0
        - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi
    ) / 2.0 * (2 ** zoom)


def _tile_corner_latlon(x, y, zoom):
    n = 2 ** zoom
    lon = float(x) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * float(y) / n))))
    return lat, lon


def choose_osm_zoom(lat_south, lat_north, lon_west, lon_east, max_tiles=16):
    lon_span = max(abs(float(lon_east) - float(lon_west)), 1e-9)
    zoom = int(math.floor(math.log2(max(1.0, 1200.0 / lon_span))))
    zoom = max(3, min(18, zoom))
    while zoom > 3:
        x0 = int(math.floor(_lon_to_tile_x(lon_west, zoom)))
        x1 = int(math.floor(_lon_to_tile_x(lon_east, zoom)))
        y0 = int(math.floor(_lat_to_tile_y(lat_north, zoom)))
        y1 = int(math.floor(_lat_to_tile_y(lat_south, zoom)))
        tile_count = (x1 - x0 + 1) * (y1 - y0 + 1)
        if tile_count <= max_tiles:
            return zoom
        zoom -= 1
    return zoom


def _tile_cache_dir():
    return Path(tempfile.gettempdir()) / "routepoint_editor_osm"


def _read_or_fetch_tile(session, zoom, x, y, cache_dir, timeout):
    cache_path = cache_dir / str(zoom) / str(x) / f"{y}.png"
    if cache_path.is_file():
        return cache_path.read_bytes(), True, cache_path
    url = OSM_TILE_URL.format(z=zoom, x=x, y=y)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content, False, cache_path


def fetch_osm_overlay(
    x_range,
    y_range,
    lat0,
    lon0,
    max_tiles=16,
    cache_dir=None,
    timeout=8.0,
):
    try:
        import requests
        from PIL import Image
    except ImportError as exc:
        raise OSMError("requests and Pillow are required for OSM tiles") from exc

    x_min, x_max = sorted(float(value) for value in x_range)
    y_min, y_max = sorted(float(value) for value in y_range)
    if x_max - x_min <= 0 or y_max - y_min <= 0:
        raise OSMError("Invalid map extent")
    lat_south, lon_west = local_to_latlon(x_min, y_min, lat0, lon0)
    lat_north, lon_east = local_to_latlon(x_max, y_max, lat0, lon0)
    if lon_east <= lon_west:
        raise OSMError("Invalid longitude extent")
    zoom = choose_osm_zoom(lat_south, lat_north, lon_west, lon_east, max_tiles)
    n = 2 ** zoom
    tile_x0 = max(0, min(n - 1, int(math.floor(_lon_to_tile_x(lon_west, zoom)))))
    tile_x1 = max(0, min(n - 1, int(math.floor(_lon_to_tile_x(lon_east, zoom)))))
    tile_y0 = max(0, min(n - 1, int(math.floor(_lat_to_tile_y(lat_north, zoom)))))
    tile_y1 = max(0, min(n - 1, int(math.floor(_lat_to_tile_y(lat_south, zoom)))))
    tile_count = (tile_x1 - tile_x0 + 1) * (tile_y1 - tile_y0 + 1)
    if tile_count > max_tiles:
        raise OSMError(f"Map view needs {tile_count} tiles, limit is {max_tiles}")

    cache_dir = Path(cache_dir) if cache_dir else _tile_cache_dir()
    session = requests.Session()
    session.headers.update({"User-Agent": OSM_USER_AGENT})
    tile_size = 256
    width = (tile_x1 - tile_x0 + 1) * tile_size
    height = (tile_y1 - tile_y0 + 1) * tile_size
    mosaic = Image.new("RGB", (width, height), (245, 245, 245))
    fetched = 0
    cached = 0
    try:
        for tile_x in range(tile_x0, tile_x1 + 1):
            for tile_y in range(tile_y0, tile_y1 + 1):
                content, was_cached, cache_path = _read_or_fetch_tile(
                    session, zoom, tile_x, tile_y, cache_dir, timeout
                )
                tile = Image.open(BytesIO(content)).convert("RGB")
                if was_cached:
                    cached += 1
                else:
                    fetched += 1
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(content)
                if tile.size != (tile_size, tile_size):
                    tile = tile.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
                mosaic.paste(
                    tile,
                    ((tile_x - tile_x0) * tile_size, (tile_y - tile_y0) * tile_size),
                )
    except Exception as exc:
        raise OSMError(f"Could not load OSM tiles: {exc}") from exc
    finally:
        session.close()

    north, west = _tile_corner_latlon(tile_x0, tile_y0, zoom)
    south, east = _tile_corner_latlon(tile_x1 + 1, tile_y1 + 1, zoom)
    x_left, y_top = latlon_to_local(north, west, lat0, lon0)
    x_right, y_bottom = latlon_to_local(south, east, lat0, lon0)
    buffer = BytesIO()
    mosaic.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "image": f"data:image/png;base64,{encoded}",
        "x": [float(x_left), float(x_right)],
        "y": [float(y_bottom), float(y_top)],
        "zoom": int(zoom),
        "tile_count": int(tile_count),
        "fetched": int(fetched),
        "cached": int(cached),
        "attribution": OSM_ATTRIBUTION,
        "created": time.time(),
    }
