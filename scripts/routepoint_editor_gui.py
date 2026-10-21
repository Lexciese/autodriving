from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    State,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import routepoint_editor_core as core

DEFAULT_FILE = Path(__file__).resolve().parent / "utara_routepoint_list.yml"
HISTORY_LIMIT = 30
ROUTE_COLOR = "#2563eb"
SELECTED_COLOR = "#ea580c"
FLAG_COLOR = "#dc2626"
REFERENCE_COLOR = "#0891b2"
ORIGINAL_COLOR = "#94a3b8"


def _out(**values):
    keys = (
        "route",
        "file",
        "original",
        "selected",
        "history",
        "future",
        "dirty",
        "reference",
        "reset",
        "pending",
        "confirm",
        "status",
        "focus",
        "file_input",
        "save_as",
        "dirty_ind",
    )
    return tuple(values.get(key, no_update) for key in keys)


def _dirty_out(**values):
    values.setdefault("dirty", True)
    values.setdefault("dirty_ind", "Unsaved changes")
    return _out(**values)


def _record_edit(history, operation):
    items = list(history or [])
    items.append(operation)
    return items[-HISTORY_LIMIT:]


def _apply_history_operation(route, operation, reverse=False):
    kind = operation.get("kind")
    if kind == "point":
        index = int(operation.get("index", 0))
        latitude, longitude = operation.get("old" if reverse else "new")
        updated = core.set_route_point(route, index, latitude, longitude)
        return updated, {"kind": "route", "index": index}
    if kind == "endpoint":
        which = operation.get("which")
        latitude, longitude = operation.get("old" if reverse else "new")
        updated = core.set_endpoint(route, which, latitude, longitude)
        return updated, {"kind": which}
    if kind == "insert":
        index = int(operation.get("index", 0))
        if reverse:
            updated, selected = core.delete_route_point(route, index)
            return updated, {"kind": "route", "index": selected}
        updated = core.insert_route_point(
            route,
            index,
            operation.get("latitude"),
            operation.get("longitude"),
        )
        return updated, {"kind": "route", "index": index}
    if kind == "delete":
        index = int(operation.get("index", 0))
        if reverse:
            updated = core.insert_route_point(
                route,
                index,
                operation.get("latitude"),
                operation.get("longitude"),
            )
            return updated, {"kind": "route", "index": index}
        updated, selected = core.delete_route_point(route, index)
        return updated, {"kind": "route", "index": selected}
    raise core.RoutePointError("Unknown undo operation")


def _selected_point(route, selected):
    if not route or not selected:
        return None
    kind = selected.get("kind", "route")
    if kind == "route":
        index = int(selected.get("index", 0))
        lats = route["route_point"]["latitude"]
        lons = route["route_point"]["longitude"]
        if not 0 <= index < len(lats):
            return None
        return {
            "kind": "route",
            "index": index,
            "latitude": float(lats[index]),
            "longitude": float(lons[index]),
        }
    key = "first_point" if kind == "first" else "last_point"
    point = route.get(key)
    if not isinstance(point, dict):
        return None
    return {
        "kind": kind,
        "index": 0,
        "latitude": float(point["latitude"]),
        "longitude": float(point["longitude"]),
    }


def _focus_for(route, selected, span):
    xy = core.selection_local_xy(route, selected)
    if xy is None:
        return None
    return {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "span": max(10.0, float(span or 250.0)),
        "seq": time.time_ns(),
    }


def _route_extent(route, pad_fraction=0.06):
    x, y = core.route_local_xy(route)
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    first = route.get("first_point")
    last = route.get("last_point")
    if isinstance(first, dict):
        fx, fy = core.point_local_xy(
            route, first["latitude"], first["longitude"]
        )
        x_values = np.append(x_values, fx)
        y_values = np.append(y_values, fy)
    if isinstance(last, dict):
        lx, ly = core.point_local_xy(
            route, last["latitude"], last["longitude"]
        )
        x_values = np.append(x_values, lx)
        y_values = np.append(y_values, ly)
    if not len(x_values):
        return None
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_pad = max(20.0, (x_max - x_min) * pad_fraction)
    y_pad = max(20.0, (y_max - y_min) * pad_fraction)
    return [x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]


def _initial_load(path, reset_value):
    file_path = Path(path).expanduser()
    try:
        route = core.load_route_document(file_path)
    except Exception as exc:
        return _out(
            status=f"Load failed: {exc}",
            confirm=False,
            pending=None,
        )
    original = {
        "latitude": list(route["route_point"]["latitude"]),
        "longitude": list(route["route_point"]["longitude"]),
    }
    reference = None
    reference_message = ""
    try:
        reference_path = core.detect_reference_path(file_path)
        if reference_path is not None:
            reference = core.load_reference_csv(reference_path)
            reference_message = (
                f" Reference overlay: {reference_path.name} "
                f"({len(reference['latitude'])} points)."
            )
    except Exception as exc:
        reference_message = f" Reference overlay not loaded: {exc}."
    count = len(route["route_point"]["latitude"])
    return _out(
        route=route,
        file=str(file_path),
        original=original,
        selected={"kind": "route", "index": 0},
        history=[],
        future=[],
        dirty=False,
        reference=reference,
        reset=int(reset_value or 0) + 1,
        pending=None,
        confirm=False,
        status=f"Loaded {count} route points from {file_path}.{reference_message}",
        focus=None,
        file_input=str(file_path),
        save_as=str(file_path),
        dirty_ind="",
    )


def _apply_selected_edit(route, selected, latitude, longitude):
    point = _selected_point(route, selected)
    if point is None:
        raise core.RoutePointError("Select a point first")
    if point["kind"] == "route":
        return core.set_route_point(route, point["index"], latitude, longitude)
    return core.set_endpoint(route, point["kind"], latitude, longitude)


def state_callback(
    n_load,
    n_reload,
    n_confirm_submit,
    n_confirm_cancel,
    click_data,
    drag_data,
    n_apply,
    n_nudge_apply,
    n_nudge_up,
    n_nudge_down,
    n_nudge_left,
    n_nudge_right,
    n_insert,
    n_delete,
    n_smooth,
    n_undo,
    n_redo,
    n_go,
    n_fit,
    active_cell,
    n_save,
    n_save_as,
    n_export,
    n_reference,
    route_store,
    file_store,
    original_store,
    selected_store,
    history_store,
    future_store,
    dirty_store,
    audit_store,
    reference_store,
    reset_store,
    pending_store,
    file_input,
    save_as_input,
    lat_input,
    lon_input,
    nudge_dx,
    nudge_dy,
    nudge_step,
    smooth_radius,
    go_index,
    focus_span,
    reference_input,
):
    triggered = ctx.triggered_id
    triggered_props = set((ctx.triggered_prop_ids or {}).keys())
    route = route_store
    selected = selected_store
    history = list(history_store or [])
    future = list(future_store or [])
    dirty = bool(dirty_store)

    if triggered is None:
        if route is not None:
            return _out()
        path = file_input or save_as_input
        if not path:
            return _out(status="Choose a routepoint YAML file.")
        return _initial_load(path, reset_store)

    if triggered == "confirm-load":
        if (
            "confirm-load.cancel_n_clicks" in triggered_props
            and "confirm-load.submit_n_clicks" not in triggered_props
        ):
            return _out(confirm=False, pending=None, status="Load cancelled.")
        path = pending_store or file_input or file_store
        if not path:
            return _out(confirm=False, pending=None, status="No file to load.")
        return _initial_load(path, reset_store)

    if triggered in {"load-btn", "reload-btn"}:
        path = file_input if triggered == "load-btn" else file_store
        if not path:
            return _out(status="Enter a YAML file path first.")
        if route is not None and dirty:
            return _out(
                confirm=True,
                pending=str(path),
                status="Unsaved changes. Confirm load to discard them.",
            )
        return _initial_load(path, reset_store)

    if triggered == "route-graph":
        if not click_data:
            return _out()
        points = click_data.get("points") or []
        if not points:
            return _out()
        point = points[0]
        custom = point.get("customdata")
        if not custom:
            return _out()
        if len(custom) >= 3 and isinstance(custom[2], str):
            kind = custom[2]
            if kind not in {"first", "last"}:
                return _out()
            return _out(selected={"kind": kind})
        if len(custom) == 2 and isinstance(custom[1], str):
            try:
                index = int(custom[0])
            except (TypeError, ValueError):
                return _out()
            return _out(
                selected=core.normalize_selection(
                    {"kind": "route", "index": index}, route
                )
            )
        try:
            index = int(custom[2])
        except (IndexError, TypeError, ValueError):
            return _out()
        return _out(
            selected=core.normalize_selection(
                {"kind": "route", "index": index}, route
            )
        )

    if triggered == "drag-store":
        if not route or not drag_data:
            return _out()
        kind = drag_data.get("kind", "route")
        try:
            index = int(drag_data.get("index", 0))
            x = float(drag_data.get("x"))
            y = float(drag_data.get("y"))
        except (TypeError, ValueError):
            return _out(status="Invalid drag coordinates.")
        if not np.isfinite(x) or not np.isfinite(y):
            return _out(status="Invalid drag coordinates.")
        moved = bool(drag_data.get("moved"))
        if kind not in {"route", "first", "last"}:
            return _out(status="Invalid drag target.")
        if not moved:
            selection = (
                {"kind": kind}
                if kind in {"first", "last"}
                else {"kind": "route", "index": index}
            )
            return _out(
                selected=core.normalize_selection(selection, route)
            )
        latitude, longitude = core.local_to_latlon(
            x, y, *core.origin_of(route)
        )
        try:
            if kind == "route":
                old_latitude = route["route_point"]["latitude"][index]
                old_longitude = route["route_point"]["longitude"][index]
                updated = core.set_route_point(
                    route, index, latitude, longitude
                )
                selection = {"kind": "route", "index": index}
                operation = {
                    "kind": "point",
                    "index": index,
                    "old": [old_latitude, old_longitude],
                    "new": [latitude, longitude],
                }
            else:
                point_key = "first_point" if kind == "first" else "last_point"
                old_point = route[point_key]
                updated = core.set_endpoint(
                    route, kind, latitude, longitude
                )
                selection = {"kind": kind}
                operation = {
                    "kind": "endpoint",
                    "which": kind,
                    "old": [old_point["latitude"], old_point["longitude"]],
                    "new": [latitude, longitude],
                }
        except (core.RoutePointError, IndexError) as exc:
            return _out(status=f"Drag failed: {exc}")
        return _dirty_out(
            route=updated,
            selected=core.normalize_selection(selection, updated),
            history=_record_edit(history, operation),
            future=[],
            status=(
                f"Moved {kind} point to "
                f"{latitude:.6f}, {longitude:.6f}."
            ),
        )

    if triggered == "apply-btn":
        if not route or not selected:
            return _out(status="Select a point first.")
        try:
            latitude = float(lat_input)
            longitude = float(lon_input)
        except (TypeError, ValueError):
            return _out(status="Latitude and longitude must be numeric.")
        if not np.isfinite(latitude) or not np.isfinite(longitude):
            return _out(status="Latitude and longitude must be finite.")
        point = _selected_point(route, selected)
        if point is None:
            return _out(status="Select a point first.")
        try:
            updated = _apply_selected_edit(
                route, selected, latitude, longitude
            )
        except core.RoutePointError as exc:
            return _out(status=f"Apply failed: {exc}")
        if point["kind"] == "route":
            operation = {
                "kind": "point",
                "index": point["index"],
                "old": [point["latitude"], point["longitude"]],
                "new": [latitude, longitude],
            }
        else:
            operation = {
                "kind": "endpoint",
                "which": point["kind"],
                "old": [point["latitude"], point["longitude"]],
                "new": [latitude, longitude],
            }
        return _dirty_out(
            route=updated,
            history=_record_edit(history, operation),
            future=[],
            status=f"Updated point to {latitude:.6f}, {longitude:.6f}.",
        )

    if triggered in {
        "nudge-apply",
        "nudge-up",
        "nudge-down",
        "nudge-left",
        "nudge-right",
    }:
        if not route or not selected:
            return _out(status="Select a point first.")
        point = _selected_point(route, selected)
        if point is None:
            return _out(status="Select a point first.")
        try:
            if triggered == "nudge-apply":
                dx = float(nudge_dx or 0.0)
                dy = float(nudge_dy or 0.0)
            else:
                step = abs(float(nudge_step or 1.0))
                dx = step if triggered == "nudge-right" else 0.0
                dx = -step if triggered == "nudge-left" else dx
                dy = step if triggered == "nudge-up" else 0.0
                dy = -step if triggered == "nudge-down" else dy
        except (TypeError, ValueError):
            return _out(status="Nudge amounts must be numeric.")
        if not np.isfinite(dx) or not np.isfinite(dy):
            return _out(status="Nudge amounts must be finite.")
        local_x, local_y = core.point_local_xy(
            route, point["latitude"], point["longitude"]
        )
        latitude, longitude = core.local_to_latlon(
            local_x + dx, local_y + dy, *core.origin_of(route)
        )
        try:
            if point["kind"] == "route":
                updated = core.set_route_point(
                    route, point["index"], latitude, longitude
                )
                operation = {
                    "kind": "point",
                    "index": point["index"],
                    "old": [point["latitude"], point["longitude"]],
                    "new": [latitude, longitude],
                }
            else:
                updated = core.set_endpoint(
                    route, point["kind"], latitude, longitude
                )
                operation = {
                    "kind": "endpoint",
                    "which": point["kind"],
                    "old": [point["latitude"], point["longitude"]],
                    "new": [latitude, longitude],
                }
        except core.RoutePointError as exc:
            return _out(status=f"Nudge failed: {exc}")
        return _dirty_out(
            route=updated,
            history=_record_edit(history, operation),
            future=[],
            status=f"Nudged point by {dx:g} m east, {dy:g} m north.",
        )

    if triggered == "insert-btn":
        if not route or not selected or selected.get("kind") != "route":
            return _out(status="Select a route point first.")
        try:
            updated, index = core.insert_midpoint(
                route, int(selected.get("index", 0))
            )
        except core.RoutePointError as exc:
            return _out(status=f"Insert failed: {exc}")
        operation = {
            "kind": "insert",
            "index": index,
            "latitude": updated["route_point"]["latitude"][index],
            "longitude": updated["route_point"]["longitude"][index],
        }
        return _dirty_out(
            route=updated,
            selected={"kind": "route", "index": index},
            history=_record_edit(history, operation),
            future=[],
            status=f"Inserted midpoint at point {index + 1}.",
        )

    if triggered == "delete-btn":
        if not route or not selected or selected.get("kind") != "route":
            return _out(status="Select a route point first.")
        point = _selected_point(route, selected)
        if point is None:
            return _out(status="Select a route point first.")
        delete_index = point["index"]
        deleted_latitude = point["latitude"]
        deleted_longitude = point["longitude"]
        try:
            updated, index = core.delete_route_point(route, delete_index)
        except core.RoutePointError as exc:
            return _out(status=f"Delete failed: {exc}")
        operation = {
            "kind": "delete",
            "index": delete_index,
            "latitude": deleted_latitude,
            "longitude": deleted_longitude,
        }
        return _dirty_out(
            route=updated,
            selected={"kind": "route", "index": index},
            history=_record_edit(history, operation),
            future=[],
            status=f"Deleted one point; now {len(updated['route_point']['latitude'])} points.",
        )

    if triggered == "smooth-btn":
        if not route or not selected or selected.get("kind") != "route":
            return _out(status="Select a route point first.")
        point = _selected_point(route, selected)
        if point is None:
            return _out(status="Select a route point first.")
        smooth_index = point["index"]
        old_latitude = point["latitude"]
        old_longitude = point["longitude"]
        try:
            radius = max(1, int(smooth_radius or 1))
            updated = core.smooth_route_point(route, smooth_index, radius)
        except (TypeError, ValueError, core.RoutePointError) as exc:
            return _out(status=f"Smooth failed: {exc}")
        operation = {
            "kind": "point",
            "index": smooth_index,
            "old": [old_latitude, old_longitude],
            "new": [
                updated["route_point"]["latitude"][smooth_index],
                updated["route_point"]["longitude"][smooth_index],
            ],
        }
        return _dirty_out(
            route=updated,
            history=_record_edit(history, operation),
            future=[],
            status=f"Smoothed point with radius {radius}.",
        )

    if triggered == "undo-btn":
        if not history:
            return _out(status="Nothing to undo.")
        operation = history[-1]
        try:
            updated, selection = _apply_history_operation(
                route, operation, reverse=True
            )
        except (core.RoutePointError, KeyError, TypeError, ValueError) as exc:
            return _out(status=f"Undo failed: {exc}")
        new_future = list(future or []) + [operation]
        new_future = new_future[-HISTORY_LIMIT:]
        return _dirty_out(
            route=updated,
            selected=core.normalize_selection(selection, updated),
            history=history[:-1],
            future=new_future,
            status="Undid last edit.",
        )

    if triggered == "redo-btn":
        if not future:
            return _out(status="Nothing to redo.")
        operation = future[-1]
        try:
            updated, selection = _apply_history_operation(
                route, operation, reverse=False
            )
        except (core.RoutePointError, KeyError, TypeError, ValueError) as exc:
            return _out(status=f"Redo failed: {exc}")
        return _dirty_out(
            route=updated,
            selected=core.normalize_selection(selection, updated),
            history=_record_edit(history, operation),
            future=future[:-1],
            status="Redid last edit.",
        )

    if triggered == "go-btn":
        if not route:
            return _out(status="Load a route first.")
        try:
            index = int(float(go_index)) - 1
        except (TypeError, ValueError):
            return _out(status="Enter a numeric 1-based point index.")
        selection = core.normalize_selection(
            {"kind": "route", "index": index}, route
        )
        if selection is None:
            return _out(status="No route points available.")
        focus = _focus_for(route, selection, focus_span)
        return _out(
            selected=selection,
            focus=focus,
            status=f"Selected point {selection['index'] + 1}.",
        )

    if triggered == "fit-btn":
        return _out(
            reset=int(reset_store or 0) + 1,
            status="Fit route to the view.",
        )

    if triggered == "audit-table":
        if not active_cell or not audit_store or not route:
            return _out()
        flags = audit_store.get("flags") or []
        row = active_cell.get("row")
        if row is None or not 0 <= int(row) < len(flags):
            return _out()
        flag = flags[int(row)]
        index = int(flag.get("index", 0))
        selection = core.normalize_selection(
            {"kind": "route", "index": index}, route
        )
        if selection is None:
            return _out(status="No route points available.")
        focus = _focus_for(route, selection, focus_span)
        return _out(
            selected=selection,
            focus=focus,
            status=(
                f"Selected {flag.get('type', 'flag')} at point "
                f"{selection['index'] + 1}."
            ),
        )

    if triggered in {"save-btn", "save-as-btn"}:
        if not route:
            return _out(status="Load a route first.")
        path = file_store if triggered == "save-btn" else save_as_input
        if not path:
            return _out(status="Enter a save path.")
        try:
            backup = core.save_route_document(route, path, create_backup=True)
        except Exception as exc:
            return _out(status=f"Save failed: {exc}")
        message = (
            f"Saved {len(route['route_point']['latitude'])} points to {path}."
        )
        if backup is not None:
            message += f" Backup: {backup.name}."
        values = {
            "dirty": False,
            "status": message,
            "file_input": str(path),
            "dirty_ind": "Saved",
        }
        if triggered == "save-as-btn":
            values["file"] = str(path)
            values["save_as"] = str(path)
        return _out(**values)

    if triggered == "export-audit-btn":
        if not audit_store:
            return _out(status="Nothing to export.")
        base = Path(file_store or file_input or "routepoint_list.yml")
        out_path = base.with_name(f"{base.stem}_audit.csv")
        try:
            core.write_audit_csv(audit_store, out_path)
        except Exception as exc:
            return _out(status=f"Audit export failed: {exc}")
        return _out(status=f"Exported audit to {out_path}.")

    if triggered == "reference-btn":
        if not reference_input:
            return _out(status="Enter a reference CSV path.")
        try:
            reference = core.load_reference_csv(reference_input)
        except Exception as exc:
            return _out(status=f"Reference load failed: {exc}")
        return _out(
            reference=reference,
            status=(
                f"Loaded reference {reference['path']} "
                f"({len(reference['latitude'])} points)."
            ),
        )

    return _out()


def audit_callback(
    route_store,
    min_gap,
    max_gap,
    turn_threshold,
    deviation_threshold,
    deviation_toggle,
    reference_store,
):
    if not route_store:
        return None, "Load a route to audit.", []
    try:
        min_gap_value = max(0.0, float(min_gap if min_gap is not None else 2.0))
        max_gap_value = max(min_gap_value, float(max_gap if max_gap is not None else 12.0))
        turn_value = max(0.0, float(turn_threshold if turn_threshold is not None else 60.0))
        deviation_value = max(
            0.0, float(deviation_threshold if deviation_threshold is not None else 8.0)
        )
    except (TypeError, ValueError):
        return None, "Audit thresholds must be numeric.", []
    audit = core.compute_audit(
        route_store,
        min_gap=min_gap_value,
        max_gap=max_gap_value,
        turn_threshold_deg=turn_value,
    )
    if reference_store and "deviation" in (deviation_toggle or []):
        try:
            audit = core.add_reference_audit(
                audit, route_store, reference_store, deviation_value
            )
        except Exception as exc:
            audit["reference_error"] = str(exc)

    summary = audit["summary"]
    counts = summary.get("flag_counts") or {}
    stats = [
        ("Points", f"{summary['point_count']:,}"),
        ("Length", f"{summary['total_length_m'] / 1000.0:.3f} km"),
        (
            "Spacing",
            f"{summary['min_spacing_m']:.2f} / "
            f"{summary['median_spacing_m']:.2f} / "
            f"{summary['max_spacing_m']:.2f} m",
        ),
        (
            "Flags",
            f"{summary['flag_count']} ({counts.get('gap', 0)} gap, "
            f"{counts.get('tight', 0)} tight, "
            f"{counts.get('duplicate', 0)} duplicate, "
            f"{counts.get('turn', 0)} turn, "
            f"{counts.get('deviation', 0)} deviation)",
        ),
    ]
    if "reference_mean_m" in summary:
        stats.append(
            (
                "Reference",
                f"mean {summary['reference_mean_m']:.2f} m, "
                f"max {summary['reference_max_m']:.2f} m",
            )
        )
    if audit.get("reference_error"):
        stats.append(("Reference error", audit["reference_error"]))
    summary_children = html.Div(
        [_kv(label, value) for label, value in stats],
        className="summary-grid",
    )

    rows = []
    for flag in audit.get("flags") or []:
        flag_type = flag.get("type", "")
        value = flag.get("value", 0)
        try:
            if flag_type == "turn":
                value_text = f"{float(value):.1f} deg"
            elif flag_type == "deviation":
                value_text = f"{float(value):.2f} m"
            else:
                value_text = f"{float(value):.2f} m"
        except (TypeError, ValueError):
            value_text = str(value)
        rows.append(
            {
                "point": int(flag.get("index", 0)) + 1,
                "type": flag_type,
                "value": value_text,
                "detail": flag.get("detail", ""),
                "index": int(flag.get("index", 0)),
            }
        )
    return audit, summary_children, rows


def selection_callback(selected_store, route_store, audit_store):
    point = _selected_point(route_store, selected_store)
    if point is None:
        return (
            None,
            None,
            "No point selected",
            "Click a route point on the map or choose a flagged row.",
            True,
            True,
            True,
            True,
            True,
        )
    kind = point["kind"]
    index = int(point.get("index", 0))
    if kind == "route":
        title = (
            f"Route point #{index + 1} of "
            f"{len(route_store['route_point']['latitude'])}"
        )
    else:
        title = "First point" if kind == "first" else "Last point"
    local_x, local_y = core.point_local_xy(
        route_store, point["latitude"], point["longitude"]
    )
    details = [
        ("Latitude", f"{point['latitude']:.9f}"),
        ("Longitude", f"{point['longitude']:.9f}"),
        ("Local", f"{local_x:.2f} E, {local_y:.2f} N"),
    ]
    if kind == "route" and audit_store:
        cumulative = audit_store.get("cumulative_distance") or []
        segments = audit_store.get("segment_distance") or []
        turns = audit_store.get("turn_angle_deg") or []
        if 0 <= index < len(cumulative):
            details.append(("Cumulative", f"{float(cumulative[index]):.1f} m"))
        if 0 <= index < len(segments):
            details.append(("Next segment", f"{float(segments[index]):.2f} m"))
        if index > 0 and index - 1 < len(segments):
            details.append(
                ("Previous segment", f"{float(segments[index - 1]):.2f} m")
            )
        if 0 <= index < len(turns):
            details.append(("Turn angle", f"{float(turns[index]):.1f} deg"))
        reference_distance = audit_store.get("reference_distance_m") or []
        if 0 <= index < len(reference_distance):
            details.append(
                ("Reference", f"{float(reference_distance[index]):.2f} m")
            )
        flag_types = [
            flag.get("type", "")
            for flag in audit_store.get("flags") or []
            if int(flag.get("index", -1)) == index
        ]
        if flag_types:
            details.append(("Flags", ", ".join(sorted(set(flag_types)))))
    info = html.Div(
        [_kv(label, value) for label, value in details],
        className="summary-grid",
    )
    is_route = kind == "route"
    return (
        point["latitude"],
        point["longitude"],
        title,
        info,
        False,
        False,
        not is_route or len(route_store["route_point"]["latitude"]) < 2,
        not is_route or len(route_store["route_point"]["latitude"]) <= 1,
        not is_route or len(route_store["route_point"]["latitude"]) < 2,
    )


def render_callback(
    route_store,
    selected_store,
    audit_store,
    original_store,
    reference_store,
    show_original,
    show_reference,
    show_markers,
    osm_store,
    osm_toggle,
    reset_store,
    focus_store,
):
    if not route_store:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            uirevision="empty",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Load a routepoint YAML file to begin.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 18, "color": "#64748b"},
                }
            ],
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
        return fig

    x, y = core.route_local_xy(route_store)
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    lats = np.asarray(route_store["route_point"]["latitude"], dtype=float)
    lons = np.asarray(route_store["route_point"]["longitude"], dtype=float)
    count = len(x_values)
    traces = []
    marker_enabled = bool(show_markers)
    images = []
    annotations = []

    if show_original and original_store:
        original_lats = np.asarray(original_store.get("latitude") or [], dtype=float)
        original_lons = np.asarray(original_store.get("longitude") or [], dtype=float)
        if len(original_lats) == len(original_lons) and len(original_lats):
            ox, oy = core.latlon_to_local(
                original_lats, original_lons, *core.origin_of(route_store)
            )
            traces.append(
                go.Scatter(
                    x=ox,
                    y=oy,
                    mode="lines",
                    name="original",
                    line={"color": ORIGINAL_COLOR, "width": 1.5, "dash": "dot"},
                    hoverinfo="skip",
                )
            )

    if show_reference and reference_store:
        ref_lats = reference_store.get("latitude") or []
        ref_lons = reference_store.get("longitude") or []
        if ref_lats and len(ref_lats) == len(ref_lons):
            rx, ry = core.latlon_to_local(
                ref_lats, ref_lons, *core.origin_of(route_store)
            )
            traces.append(
                go.Scatter(
                    x=rx,
                    y=ry,
                    mode="lines",
                    name=f"reference ({len(ref_lats)})",
                    line={"color": REFERENCE_COLOR, "width": 1.5},
                    opacity=0.7,
                    hoverinfo="skip",
                )
            )

    custom = np.column_stack([lats, lons, np.arange(count)])
    traces.append(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers" if marker_enabled else "lines",
            name=f"route ({count})",
            meta="route",
            customdata=custom,
            line={"color": ROUTE_COLOR, "width": 2},
            marker={"size": 4, "color": ROUTE_COLOR, "opacity": 0.75},
            hovertemplate=(
                "#%{customdata[2]}<br>"
                "lat %{customdata[0]:.6f}<br>"
                "lon %{customdata[1]:.6f}<br>"
                "E %{x:.1f} m / N %{y:.1f} m<extra></extra>"
            ),
        )
    )

    if audit_store:
        segment_x = []
        segment_y = []
        point_x = []
        point_y = []
        point_text = []
        point_custom = []
        for flag in audit_store.get("flags") or []:
            index = int(flag.get("index", 0))
            end_index = flag.get("end_index")
            if (
                end_index is not None
                and 0 <= index < count
                and 0 <= int(end_index) < count
                and flag.get("type") in {"gap", "duplicate", "tight"}
            ):
                end_index = int(end_index)
                segment_x.extend([x_values[index], x_values[end_index], None])
                segment_y.extend([y_values[index], y_values[end_index], None])
            if 0 <= index < count:
                point_x.append(x_values[index])
                point_y.append(y_values[index])
                point_text.append(
                    f"point #{index + 1}<br>{flag.get('type')}: "
                    f"{flag.get('detail')}"
                )
                point_custom.append([index, str(flag.get("type", ""))])
        if segment_x:
            traces.append(
                go.Scatter(
                    x=segment_x,
                    y=segment_y,
                    mode="lines",
                    name="flagged segments",
                    line={"color": FLAG_COLOR, "width": 4},
                    hoverinfo="skip",
                )
            )
        if point_x:
            traces.append(
                go.Scatter(
                    x=point_x,
                    y=point_y,
                    mode="markers",
                    name="flagged points",
                    text=point_text,
                    customdata=point_custom,
                    marker={
                        "size": 9,
                        "color": FLAG_COLOR,
                        "symbol": "x",
                        "line": {"width": 1, "color": "#7f1d1d"},
                    },
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    first = route_store["first_point"]
    last = route_store["last_point"]
    first_xy = core.point_local_xy(
        route_store, first["latitude"], first["longitude"]
    )
    last_xy = core.point_local_xy(
        route_store, last["latitude"], last["longitude"]
    )
    traces.append(
        go.Scatter(
            x=[first_xy[0]],
            y=[first_xy[1]],
            mode="markers",
            name="first_point",
            meta="first",
            customdata=[[first["latitude"], first["longitude"], "first"]],
            marker={
                "size": 13,
                "color": "#16a34a",
                "symbol": "circle",
                "line": {"width": 2, "color": "#14532d"},
            },
            hovertemplate=(
                "first_point<br>lat %{customdata[0]:.6f}<br>"
                "lon %{customdata[1]:.6f}<extra></extra>"
            ),
        )
    )
    traces.append(
        go.Scatter(
            x=[last_xy[0]],
            y=[last_xy[1]],
            mode="markers",
            name="last_point",
            meta="last",
            customdata=[[last["latitude"], last["longitude"], "last"]],
            marker={
                "size": 13,
                "color": "#dc2626",
                "symbol": "circle",
                "line": {"width": 2, "color": "#7f1d1d"},
            },
            hovertemplate=(
                "last_point<br>lat %{customdata[0]:.6f}<br>"
                "lon %{customdata[1]:.6f}<extra></extra>"
            ),
        )
    )

    selected_xy = core.selection_local_xy(route_store, selected_store)
    if selected_xy is not None:
        selected_kind = selected_store.get("kind", "route")
        selected_index = int(selected_store.get("index", 0))
        traces.append(
            go.Scatter(
                x=[selected_xy[0]],
                y=[selected_xy[1]],
                mode="markers",
                name="selected",
                meta="selected",
                customdata=[[selected_kind, selected_index]],
                marker={
                    "size": 17,
                    "color": SELECTED_COLOR,
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "#7c2d12"},
                },
                hoverinfo="skip",
            )
        )

    if osm_toggle and osm_store and osm_store.get("image"):
        osm_x = osm_store.get("x") or [0, 1]
        osm_y = osm_store.get("y") or [0, 1]
        images.append(
            {
                "source": osm_store["image"],
                "xref": "x",
                "yref": "y",
                "x": float(osm_x[0]),
                "y": float(osm_y[1]),
                "sizex": float(osm_x[1]) - float(osm_x[0]),
                "sizey": float(osm_y[1]) - float(osm_y[0]),
                "xanchor": "left",
                "yanchor": "top",
                "sizing": "stretch",
                "opacity": 0.8,
                "layer": "below",
            }
        )
        annotations.append(
            {
                "text": "© OpenStreetMap contributors",
                "xref": "paper",
                "yref": "paper",
                "x": 0.995,
                "y": 0.01,
                "xanchor": "right",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 10, "color": "#475569"},
                "bgcolor": "rgba(255,255,255,0.7)",
            }
        )

    fig = go.Figure(data=traces)
    layout = {
        "template": "plotly_white",
        "dragmode": "pan",
        "hovermode": "closest",
        "uirevision": f"routepoint-editor-{reset_store or 0}",
        "margin": {"l": 60, "r": 20, "t": 45, "b": 50},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
        },
        "xaxis": {
            "title": "East (m)",
            "zeroline": False,
            "gridcolor": "#e2e8f0",
        },
        "yaxis": {
            "title": "North (m)",
            "zeroline": False,
            "gridcolor": "#e2e8f0",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        "images": images,
        "annotations": annotations,
    }
    triggered_props = set((ctx.triggered_prop_ids or {}).keys())
    if "focus-store.data" in triggered_props and focus_store:
        span = max(10.0, float(focus_store.get("span", 250.0)))
        center_x = float(focus_store.get("x", 0.0))
        center_y = float(focus_store.get("y", 0.0))
        layout["xaxis"]["range"] = [center_x - span / 2, center_x + span / 2]
        layout["yaxis"]["range"] = [center_y - span / 2, center_y + span / 2]
    elif "reset-store.data" in triggered_props:
        extent = _route_extent(route_store)
        if extent is not None:
            layout["xaxis"]["range"] = extent[0]
            layout["yaxis"]["range"] = extent[1]
    fig.update_layout(**layout)
    return fig


def view_callback(relayout, current_view, route_store):
    if not relayout:
        return no_update
    if route_store is None:
        return no_update
    if relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
        return None
    view = dict(current_view or {})
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        view["x"] = [
            float(relayout["xaxis.range[0]"]),
            float(relayout["xaxis.range[1]"]),
        ]
    if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
        view["y"] = [
            float(relayout["yaxis.range[0]"]),
            float(relayout["yaxis.range[1]"]),
        ]
    if "x" not in view or "y" not in view:
        return no_update
    return view


def osm_callback(osm_toggle, n_refresh, view_store, route_store):
    if not osm_toggle:
        return None, "OSM tiles off."
    if not route_store:
        return None, "Load a route first."
    if view_store and view_store.get("x") and view_store.get("y"):
        x_range = view_store["x"]
        y_range = view_store["y"]
    else:
        extent = _route_extent(route_store, pad_fraction=0.02)
        if extent is None:
            return None, "No route extent available."
        x_range, y_range = extent
    lat0, lon0 = core.origin_of(route_store)
    try:
        overlay = core.fetch_osm_overlay(
            x_range,
            y_range,
            lat0,
            lon0,
            max_tiles=16,
        )
    except core.OSMError as exc:
        return None, f"OSM unavailable: {exc}"
    except Exception as exc:
        return None, f"OSM unavailable: {exc}"
    return (
        overlay,
        f"OSM zoom {overlay['zoom']}, {overlay['tile_count']} tiles "
        f"({overlay['fetched']} fetched, {overlay['cached']} cached).",
    )


def _kv(label, value):
    return html.Div(
        [
            html.Span(label, className="k"),
            html.Span(value, className="v"),
        ],
        className="kv",
    )


def _section(title, children):
    return html.Div(
        [html.H3(title), html.Div(children, className="section-body")],
        className="section",
    )


def build_layout(initial_file):
    initial = str(initial_file or "")
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Route point file", className="brand"),
                            dcc.Input(
                                id="file-input",
                                type="text",
                                value=initial,
                                className="path-input",
                                debounce=True,
                            ),
                        ],
                        className="file-block",
                    ),
                    html.Button("Load", id="load-btn", className="button"),
                    html.Button("Reload", id="reload-btn", className="button"),
                    html.Button("Save", id="save-btn", className="button primary"),
                    dcc.Input(
                        id="save-as-input",
                        type="text",
                        value=initial,
                        className="path-input save-as",
                        debounce=True,
                    ),
                    html.Button("Save As", id="save-as-btn", className="button"),
                    html.Span(id="dirty-indicator", className="dirty-indicator"),
                ],
                className="topbar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="route-graph",
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "responsive": True,
                                    "modeBarButtonsToRemove": [
                                        "lasso2d",
                                        "select2d",
                                    ],
                                },
                                style={"height": "78vh"},
                            ),
                        ],
                        className="graph-panel",
                    ),
                    html.Div(
                        [
                            _section(
                                "Selected point",
                                [
                                    html.Div(
                                        id="selected-title",
                                        className="selected-title",
                                        children="No point selected",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Latitude"),
                                            dcc.Input(
                                                id="lat-input",
                                                type="number",
                                                step="any",
                                                className="number-input",
                                            ),
                                            html.Label("Longitude"),
                                            dcc.Input(
                                                id="lon-input",
                                                type="number",
                                                step="any",
                                                className="number-input",
                                            ),
                                        ],
                                        className="field-grid",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Apply lat/lon",
                                                id="apply-btn",
                                                className="button primary",
                                            ),
                                        ],
                                        className="button-row",
                                    ),
                                    html.Div(id="selected-info"),
                                ],
                            ),
                            _section(
                                "Nudge in meters",
                                [
                                    html.Div(
                                        [
                                            html.Label("Step"),
                                            dcc.Input(
                                                id="nudge-step",
                                                type="number",
                                                value=1,
                                                step=0.1,
                                                min=0.01,
                                                className="number-input",
                                            ),
                                            html.Button(
                                                "Up",
                                                id="nudge-up",
                                                className="button small",
                                            ),
                                            html.Button(
                                                "Down",
                                                id="nudge-down",
                                                className="button small",
                                            ),
                                            html.Button(
                                                "Left",
                                                id="nudge-left",
                                                className="button small",
                                            ),
                                            html.Button(
                                                "Right",
                                                id="nudge-right",
                                                className="button small",
                                            ),
                                        ],
                                        className="nudge-row",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("East"),
                                            dcc.Input(
                                                id="nudge-dx",
                                                type="number",
                                                value=0,
                                                step=0.1,
                                                className="number-input",
                                            ),
                                            html.Label("North"),
                                            dcc.Input(
                                                id="nudge-dy",
                                                type="number",
                                                value=0,
                                                step=0.1,
                                                className="number-input",
                                            ),
                                            html.Button(
                                                "Move by dx/dy",
                                                id="nudge-apply",
                                                className="button",
                                            ),
                                        ],
                                        className="nudge-row",
                                    ),
                                ],
                            ),
                            _section(
                                "Edit",
                                [
                                    html.Div(
                                        [
                                            html.Button(
                                                "Insert midpoint",
                                                id="insert-btn",
                                                className="button",
                                            ),
                                            html.Button(
                                                "Delete",
                                                id="delete-btn",
                                                className="button danger",
                                            ),
                                            html.Button(
                                                "Undo",
                                                id="undo-btn",
                                                className="button",
                                            ),
                                            html.Button(
                                                "Redo",
                                                id="redo-btn",
                                                className="button",
                                            ),
                                        ],
                                        className="button-row",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Smooth radius"),
                                            dcc.Input(
                                                id="smooth-radius",
                                                type="number",
                                                value=1,
                                                min=1,
                                                max=20,
                                                step=1,
                                                className="number-input",
                                            ),
                                            html.Button(
                                                "Smooth point",
                                                id="smooth-btn",
                                                className="button",
                                            ),
                                        ],
                                        className="nudge-row",
                                    ),
                                ],
                            ),
                            _section(
                                "Audit",
                                [
                                    html.Div(
                                        [
                                            html.Label("Min gap (m)"),
                                            dcc.Input(
                                                id="min-gap",
                                                type="number",
                                                value=2,
                                                step=0.5,
                                                min=0,
                                                className="number-input",
                                            ),
                                            html.Label("Max gap (m)"),
                                            dcc.Input(
                                                id="max-gap",
                                                type="number",
                                                value=12,
                                                step=0.5,
                                                min=0,
                                                className="number-input",
                                            ),
                                            html.Label("Turn (deg)"),
                                            dcc.Input(
                                                id="turn-threshold",
                                                type="number",
                                                value=60,
                                                step=1,
                                                min=0,
                                                className="number-input",
                                            ),
                                            html.Label("Deviation (m)"),
                                            dcc.Input(
                                                id="deviation-threshold",
                                                type="number",
                                                value=8,
                                                step=0.5,
                                                min=0,
                                                className="number-input",
                                            ),
                                        ],
                                        className="field-grid",
                                    ),
                                    dcc.Checklist(
                                        id="deviation-toggle",
                                        options=[
                                            {
                                                "label": "Flag deviations from reference",
                                                "value": "deviation",
                                            }
                                        ],
                                        value=["deviation"],
                                        className="checklist",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="reference-input",
                                                type="text",
                                                placeholder="Optional reference GNSS CSV path",
                                                className="path-input",
                                                debounce=True,
                                            ),
                                            html.Button(
                                                "Load reference",
                                                id="reference-btn",
                                                className="button",
                                            ),
                                        ],
                                        className="reference-row",
                                    ),
                                    html.Div(
                                        id="reference-status",
                                        className="hint",
                                        children="Reference auto-detected from *_gnss.csv.",
                                    ),
                                    html.Div(id="audit-summary"),
                                    dash_table.DataTable(
                                        id="audit-table",
                                        columns=[
                                            {"name": "Point", "id": "point"},
                                            {"name": "Type", "id": "type"},
                                            {"name": "Value", "id": "value"},
                                            {"name": "Detail", "id": "detail"},
                                        ],
                                        data=[],
                                        page_size=10,
                                        style_table={
                                            "overflowX": "auto",
                                            "maxHeight": "260px",
                                            "overflowY": "auto",
                                        },
                                        style_cell={
                                            "fontSize": 11,
                                            "padding": "4px 6px",
                                            "textAlign": "left",
                                        },
                                        style_header={
                                            "fontWeight": "600",
                                            "backgroundColor": "#f1f5f9",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Export audit CSV",
                                                id="export-audit-btn",
                                                className="button",
                                            )
                                        ],
                                        className="button-row",
                                    ),
                                ],
                            ),
                            _section(
                                "View",
                                [
                                    html.Div(
                                        [
                                            html.Button(
                                                "Fit route",
                                                id="fit-btn",
                                                className="button",
                                            ),
                                            html.Label("Go to index"),
                                            dcc.Input(
                                                id="go-index",
                                                type="number",
                                                min=1,
                                                step=1,
                                                className="number-input",
                                            ),
                                            html.Button(
                                                "Go",
                                                id="go-btn",
                                                className="button",
                                            ),
                                            html.Label("Focus span (m)"),
                                            dcc.Input(
                                                id="focus-span",
                                                type="number",
                                                value=250,
                                                min=10,
                                                step=10,
                                                className="number-input",
                                            ),
                                        ],
                                        className="nudge-row",
                                    ),
                                    dcc.Checklist(
                                        id="show-markers",
                                        options=[
                                            {
                                                "label": "Show point markers",
                                                "value": "markers",
                                            }
                                        ],
                                        value=["markers"],
                                        className="checklist",
                                    ),
                                    dcc.Checklist(
                                        id="show-original",
                                        options=[
                                            {
                                                "label": "Show original route",
                                                "value": "original",
                                            }
                                        ],
                                        value=[],
                                        className="checklist",
                                    ),
                                    dcc.Checklist(
                                        id="show-reference",
                                        options=[
                                            {
                                                "label": "Show GNSS reference",
                                                "value": "reference",
                                            }
                                        ],
                                        value=["reference"],
                                        className="checklist",
                                    ),
                                    dcc.Checklist(
                                        id="osm-toggle",
                                        options=[
                                            {
                                                "label": "Show OSM tiles",
                                                "value": "osm",
                                            }
                                        ],
                                        value=[],
                                        className="checklist",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Refresh OSM tiles",
                                                id="osm-refresh",
                                                className="button",
                                            )
                                        ],
                                        className="button-row",
                                    ),
                                    html.Div(
                                        id="osm-status",
                                        className="hint",
                                        children="OSM tiles off.",
                                    ),
                                ],
                            ),
                        ],
                        className="sidebar",
                    ),
                ],
                className="main",
            ),
            html.Div(
                id="status",
                className="status",
                children="Loading routepoint editor...",
            ),
            dcc.ConfirmDialog(
                id="confirm-load",
                message="Unsaved changes will be discarded. Continue?",
            ),
            dcc.Store(id="route-store"),
            dcc.Store(id="file-store"),
            dcc.Store(id="original-store"),
            dcc.Store(id="selected-store"),
            dcc.Store(id="history-store", data=[]),
            dcc.Store(id="future-store", data=[]),
            dcc.Store(id="dirty-store", data=False),
            dcc.Store(id="reference-store"),
            dcc.Store(id="audit-store"),
            dcc.Store(id="reset-store", data=0),
            dcc.Store(id="pending-load-store"),
            dcc.Store(id="focus-store"),
            dcc.Store(id="drag-store"),
            dcc.Store(id="view-store"),
            dcc.Store(id="osm-store"),
        ],
        className="app",
    )


def create_app(initial_file):
    app = Dash(
        __name__,
        title="Routepoint Editor",
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(initial_file)

    app.callback(
        Output("route-store", "data"),
        Output("file-store", "data"),
        Output("original-store", "data"),
        Output("selected-store", "data"),
        Output("history-store", "data"),
        Output("future-store", "data"),
        Output("dirty-store", "data"),
        Output("reference-store", "data"),
        Output("reset-store", "data"),
        Output("pending-load-store", "data"),
        Output("confirm-load", "display"),
        Output("status", "children"),
        Output("focus-store", "data"),
        Output("file-input", "value"),
        Output("save-as-input", "value"),
        Output("dirty-indicator", "children"),
        Input("load-btn", "n_clicks"),
        Input("reload-btn", "n_clicks"),
        Input("confirm-load", "submit_n_clicks"),
        Input("confirm-load", "cancel_n_clicks"),
        Input("route-graph", "clickData"),
        Input("drag-store", "data"),
        Input("apply-btn", "n_clicks"),
        Input("nudge-apply", "n_clicks"),
        Input("nudge-up", "n_clicks"),
        Input("nudge-down", "n_clicks"),
        Input("nudge-left", "n_clicks"),
        Input("nudge-right", "n_clicks"),
        Input("insert-btn", "n_clicks"),
        Input("delete-btn", "n_clicks"),
        Input("smooth-btn", "n_clicks"),
        Input("undo-btn", "n_clicks"),
        Input("redo-btn", "n_clicks"),
        Input("go-btn", "n_clicks"),
        Input("fit-btn", "n_clicks"),
        Input("audit-table", "active_cell"),
        Input("save-btn", "n_clicks"),
        Input("save-as-btn", "n_clicks"),
        Input("export-audit-btn", "n_clicks"),
        Input("reference-btn", "n_clicks"),
        State("route-store", "data"),
        State("file-store", "data"),
        State("original-store", "data"),
        State("selected-store", "data"),
        State("history-store", "data"),
        State("future-store", "data"),
        State("dirty-store", "data"),
        State("audit-store", "data"),
        State("reference-store", "data"),
        State("reset-store", "data"),
        State("pending-load-store", "data"),
        State("file-input", "value"),
        State("save-as-input", "value"),
        State("lat-input", "value"),
        State("lon-input", "value"),
        State("nudge-dx", "value"),
        State("nudge-dy", "value"),
        State("nudge-step", "value"),
        State("smooth-radius", "value"),
        State("go-index", "value"),
        State("focus-span", "value"),
        State("reference-input", "value"),
    )(state_callback)

    app.callback(
        Output("audit-store", "data"),
        Output("audit-summary", "children"),
        Output("audit-table", "data"),
        Input("route-store", "data"),
        Input("min-gap", "value"),
        Input("max-gap", "value"),
        Input("turn-threshold", "value"),
        Input("deviation-threshold", "value"),
        Input("deviation-toggle", "value"),
        Input("reference-store", "data"),
    )(audit_callback)

    app.callback(
        Output("lat-input", "value"),
        Output("lon-input", "value"),
        Output("selected-title", "children"),
        Output("selected-info", "children"),
        Output("apply-btn", "disabled"),
        Output("nudge-apply", "disabled"),
        Output("insert-btn", "disabled"),
        Output("delete-btn", "disabled"),
        Output("smooth-btn", "disabled"),
        Input("selected-store", "data"),
        Input("route-store", "data"),
        Input("audit-store", "data"),
    )(selection_callback)

    app.callback(
        Output("route-graph", "figure"),
        Input("route-store", "data"),
        Input("selected-store", "data"),
        Input("audit-store", "data"),
        Input("original-store", "data"),
        Input("reference-store", "data"),
        Input("show-original", "value"),
        Input("show-reference", "value"),
        Input("show-markers", "value"),
        Input("osm-store", "data"),
        Input("osm-toggle", "value"),
        Input("reset-store", "data"),
        Input("focus-store", "data"),
    )(render_callback)

    app.callback(
        Output("view-store", "data"),
        Input("route-graph", "relayoutData"),
        State("view-store", "data"),
        State("route-store", "data"),
        prevent_initial_call=True,
    )(view_callback)

    app.callback(
        Output("osm-store", "data"),
        Output("osm-status", "children"),
        Input("osm-toggle", "value"),
        Input("osm-refresh", "n_clicks"),
        State("view-store", "data"),
        State("route-store", "data"),
        prevent_initial_call=True,
    )(osm_callback)

    return app


def main():
    parser = argparse.ArgumentParser(description="Routepoint editor GUI")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE if DEFAULT_FILE.is_file() else None,
        help="Initial routepoint YAML file",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app = create_app(args.file)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
