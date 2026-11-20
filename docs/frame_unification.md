# Frame unification: rp1/rp2, waypoints and LiDAR projections

This document explains the frame changes, the calculations, and the new
artifacts introduced to make `rp1/rp2`, waypoints and the projected LiDAR
share one origin and one axis convention.

## 1. Frames and conventions

Three frames are involved:

| Frame | Definition | Used by |
|---|---|---|
| `base` | ZED odom origin. `x` forward, `y` left, `z` up. | waypoints, ego pose, rp1/rp2 |
| `projection` | `(x_forward, y_up, z_left)`. This is the legacy `(-pcd_y, pcd_z, pcd_x)` order expected by `gen_bev_front_rear_seg_dep_*`. | BEV / front / rear LiDAR images |
| `ENU` | Local East/North metres around a route anchor. `x = East`, `y = North`. | GNSS route points and alignment |

Waypoints already live in `base` because they are built from
`local_position_xyz` with `transform_2d_points`. That is why the waypoint
overlay already looks correct; the change keeps that behaviour unchanged.

The 2D convention of `base` is the one defined by `transform_2d_points`:

```
r2 = pi/2 - heading
R2 = [[cos(r2), sin(r2)], [-sin(r2), cos(r2)]]
world = R2 @ local + t
```

`rp1/rp2` are now passed through exactly the same function, so they inherit
the same convention as waypoints.

## 2. What changed

### New module `transforms.py`

Contains all shared math:

- `lidar_raw_to_proj_numpy` / `lidar_raw_to_proj_tensor`
- `latlon_to_enu` / `enu_to_latlon`
- `yaw_from_quaternion_xyzw`
- `antenna_to_base_latlon` (lever arm)
- `fit_rigid_2d` (Umeyama fit with trimming)
- `save_alignment` / `load_alignment`
- `apply_alignment` / `invert_alignment`
- `rp_odom_to_local` (identical to the waypoint `transform_2d_points` call)

### `config.py`

New fields:

```python
frame_version = 3
frame_cache_tag = "frame_v3"
lidar_hdf5_name = "preload_lidar_bev_front_seg_dep_frame_v3.hdf5"
preload_name_template = "preload_seq{seq}_pred{pred}_latlon_fix_frame_v3.npy"
test_preload_name_template = "preload_test_split_seq{seq}_pred{pred}_frame_v3.hdf5"
transfuser_lidar_hdf5_name = "transfuser_lidar_bev_front_seg_dep_frame_v3.hdf5"
transfuser_preload_name_template = "seq{seq}_pred{pred}_w{w}_frame_v3.npy"
align_gnss_odom_filename = "align_gnss_odom.npz"
route_gnss_base_filename = "route_gnss_base.csv"
routepoint_base_template = "{route}_routepoint_list_base.yml"

# Raw LiDAR -> projection frame (x_forward, y_up, z_left), origin at base.
T_lidar_proj = np.array([
    [0.0, -1.0, 0.0, -0.6465],
    [0.0,  0.0, 1.0,  0.2020],
    [1.0,  0.0, 0.0, -0.0345],
    [0.0,  0.0, 0.0,  1.0],
])

# GNSS antenna relative to the ZED camera base, [forward, left, up] metres.
gnss_ant_offset_base = [-0.0490, 0.1346, -0.0350]
enable_gnss_lever_arm = True
ground_z_base = None   # set to -camera_height_above_ground
```

The new cache names are versioned so that old caches are never silently
reused after a frame change.

### `preprocess_route.py`

- Keeps the raw `meta/*.yml`, `<route>_gnss.csv` and
  `<route>_routepoint_list.yml` unchanged.
- Adds a base-frame track: `global_position_latlon -> antenna_to_base_latlon`
  using the ZED yaw and `gnss_ant_offset_base`, then cleans it with the same
  outlier filter.
- Writes:
  - `<route>_gnss_base.csv`
  - `<route>_routepoint_list_base.yml`

### `scripts/estimate_route_alignment.py`

- Reads `<route>_gnss_base.csv` and the ZED poses in `meta/`.
- Fits a rigid/similarity 2D transform from ENU to ZED odom.
- Writes `<route>/align_gnss_odom.npz` and a diagnostic PNG.

### `a_dataloader.py`

- Loads the per-route alignment.
- During preload, converts each base route point from ENU to ZED odom, then to
  the local ego frame with `rp_odom_to_local` (same math as waypoints).
- Stores the local rp values in the preload cache and reads them back in
  `__getitem__`. The magnetometer is no longer used for the rp transform; it
  remains available as a diagnostic (`bearing_robot`).
- Adds `bearing_zed` for logging/visualisation.
- Applies `T_lidar_proj` before every `gen_bev_front_rear_seg_dep_*` call.
- Uses the versioned LiDAR HDF5 and preload file names.

### `preprocess_lidar.py`

- Applies `T_lidar_proj` to the point cloud before projection.
- Semantic segmentation (`PolarSeg`) still runs on the raw point cloud.
- Writes `lidar/img/frame_version.txt`; if it does not match
  `config.frame_version`, precomputed BEV/front images are regenerated
  instead of skipped.

### `transfuser_dataloader.py`, `jetson_deployment_dataloader.py`

- Same `T_lidar_proj` transform.
- Same rp precomputation/loading scheme.
- Versioned cache names.

### `preprocess_preload_test_split.py`

- Uses the versioned test preload HDF5 name.
- Stores the new local rp values (they come from `KarrDataset`).

### `preprocess_util.py`

- `plot_lidfront_rpwp` uses `config.ground_z_base` instead of the hardcoded
  `lid_cover_area_bt[0]`. This is required because the base origin may no
  longer be the LiDAR origin.

### `preview_join_data_all.py`

- Loads the base route points and base GNSS track.
- Loads `align_gnss_odom.npz`.
- Computes rp with `latlon_to_enu -> apply_alignment -> rp_odom_to_local`
  using the ZED heading.

### `scripts/overlay_rp_wp.py`

- New diagnostic renderer. It overlays rp1/rp2 and all waypoints on the BEV
  and front images using `plot_lidbev_rpwp` / `plot_lidfront_rpwp`.

## 3. LiDAR transform

`T_lidar_proj` is a full 4x4 that maps a raw LiDAR point to the projection
frame with origin at the base:

```
p_proj = T_lidar_proj @ [pcd_x, pcd_y, pcd_z, 1]
```

The configured value is the CAD-extracted transform: the legacy remap plus a
translation that moves the origin to the ZED camera base.

```
T_lidar_proj = [[ 0, -1,  0, -0.6465],
                [ 0,  0,  1,  0.2020],
                [ 1,  0,  0, -0.0345],
                [ 0,  0,  0,  1.0000]]
```

The legacy remap alone reproduces `(ptx, pty, ptz) = (-pcd_y, pcd_z, pcd_x)`.
The translation is the LiDAR geometry centre relative to the base in
projection order `[forward, up, left]`.

To change the origin, compose a base offset in projection axes. If the LiDAR
origin is at `t_lidar_base = [t_forward, t_left, t_up]` in base coordinates,
the projection-frame offset is `[t_forward, t_up, t_left]`:

```
T_base_offset = [[1, 0, 0, t_forward],
                 [0, 1, 0, t_up],
                 [0, 0, 1, t_left],
                 [0, 0, 0, 1]]
T_lidar_proj = T_base_offset @ T_LEGACY_REMAP
```

If there is also a rotational misalignment `R_res` expressed in the
projection frame:

```
T_lidar_proj = T_base_offset @ R_res @ T_LEGACY_REMAP
```

The CAD placement axes for the LiDAR are effectively identity, so no
rotational term is included. The base proxy is the ZED Camera geometry bbox
centre, because the ZED internal IMU is not a separate CAD component.

## 4. GNSS antenna to base

`gnss_ant_offset_base = [r_forward, r_left, r_up]` is the antenna position in
the base frame. With standard ROS yaw `yaw` from the ZED quaternion:

```
E, N = latlon_to_enu(lat, lon, lat0, lon0)
dE = cos(yaw) * r_forward - sin(yaw) * r_left
dN = sin(yaw) * r_forward + cos(yaw) * r_left
E_base = E - dE
N_base = N - dN
lat_base, lon_base = enu_to_latlon(E_base, N_base, lat0, lon0)
```

The configured CAD value is:

```
gnss_ant_offset_base = [-0.0490, 0.1346, -0.0350]  # [forward, left, up]
```

This is the `Module GNSS Antenna` geometry bbox centre relative to the ZED
Camera base, using CAD axes `+y forward`, `+x right`, `+z up`. After changing
it, re-run `preprocess_route.py` and `scripts/estimate_route_alignment.py`.

## 5. Per-route rigid alignment

Given paired ENU points `p_enu` and ZED odom points `p_odom`, the tool fits:

```
p_odom = scale * R(theta) @ p_enu + t
```

using a trimmed Umeyama (SVD) fit. The result is stored as:

- `R` (2x2)
- `t` (2)
- `scale` (expected near 1.0)
- `anchor_latlon` (the ENU origin)
- `theta`, `residual_rms`, `residual_max`, `inlier_mask`

`a_dataloader` applies:

```
E, N = latlon_to_enu(rp_lat, rp_lon, anchor_lat, anchor_lon)
x_odom, y_odom = apply_alignment(E, N, align)
rp_local = rp_odom_to_local(x_odom, y_odom, ego_x, ego_y, ego_heading)
```

`rp_odom_to_local` is byte-for-byte equivalent to the waypoint call:

```
transform_2d_points(
    np.zeros((1, 3)),
    0.0, x_odom, y_odom,
    np.pi/2 - ego_heading, ego_x, ego_y,
)
```

That is why rp1/rp2 and waypoints now share one origin and one heading.

## 6. Ground height in the front overlay

`plot_lidfront_rpwp` projects rp onto the ground plane. After moving the
origin from the LiDAR to the base, the ground height relative to the base is
`ground_z_base`. If the base is `h` metres above the ground, set:

```
ground_z_base = -h
```

The default is `lid_cover_area_bt[0]`, which matches the legacy LiDAR-origin
behaviour.

## 7. Cache versioning and regeneration

Versioned names mean old caches are ignored, but you must regenerate the new
ones:

1. `preprocess_route.py` -> `route_gnss_base.csv`,
   `routepoint_list_base.yml`.
2. `scripts/estimate_route_alignment.py` -> `align_gnss_odom.npz`.
3. `preprocess_lidar.py` (or let `KarrDataset` rebuild the HDF5) ->
   `*-frame_v3.hdf5` and new BEV/front images.
4. Delete or ignore old caches; the new names are:
   - `preload_lidar_bev_front_seg_dep_frame_v3.hdf5`
   - `preload_seq{seq}_pred{pred}_latlon_fix_frame_v3.npy`
   - `preload_test_split_seq{seq}_pred{pred}_frame_v3.hdf5`
   - `transfuser_lidar_bev_front_seg_dep_frame_v3.hdf5`
   - `seq{seq}_pred{pred}_w{w}_frame_v3.npy`
5. Retrain all models: both the BEV/front inputs and rp targets changed.

## 8. Validation

- Run `scripts/overlay_rp_wp.py` and check that rp1/rp2 and waypoints lie on
  the road in both BEV and front images.
- Inspect `align_gnss_odom.png`: the aligned GNSS trace should overlap the
  ZED odom trace, and the residual should stay small over the whole route.
- If the residual grows with distance, use a windowed alignment or fuse with
  GNSS course over ground.
- Run `python -m unittest discover -s tests` to check the transform math.

## 9. Known caveats

- `T_lidar_proj` and `gnss_ant_offset_base` now contain values extracted from
  `Assembly Cleanup.step`. The base proxy is the ZED Camera geometry bbox
  centre because the ZED internal IMU is not a separate CAD component.
- `ground_z_base` is still `None` and falls back to `lid_cover_area_bt[0]`.
  Set it to minus the ZED camera height above the ground.
- ZED odom yaw can drift on long routes. A single per-route alignment is
  enough only while the residual stays small; otherwise use a sliding-window
  offset.
- The camera is intentionally not aligned because `ai23` does not consume the
  RGB image.
