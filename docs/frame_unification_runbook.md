# Frame unification runbook

All commands assume the repository root:

```bash
cd /home/lexciese/Dev/autodriving/autodriving
```

The training environment is `dl`:

```bash
conda run -n dl python <script>
```

## 0. Fill the measured extrinsics

Before running anything, set these in `config.py`:

```python
# raw LiDAR -> projection frame (x_forward, y_up, z_left), origin at base.
# Default is the legacy (-y, z, x) remap with origin at the LiDAR.
T_lidar_proj = np.array([...], dtype=np.float64)

# Antenna position in the base frame, [forward, left, up] metres.
gnss_ant_offset_base = [x, y, z]

# Ground height relative to the base origin (negative below the base).
ground_z_base = -base_height_above_ground
```

If the LiDAR is mounted at `t_lidar_base = [t_forward, t_left, t_up]` in base
coordinates, compose:

```python
T_base_offset = np.array([
    [1, 0, 0, t_forward],
    [0, 1, 0, t_up],
    [0, 0, 1, t_left],
    [0, 0, 0, 1],
])
T_lidar_proj = T_base_offset @ T_LEGACY_REMAP
```

`T_LEGACY_REMAP` is exported by `transforms.py`.

## 1. Generate base GNSS and base route points

```bash
conda run -n dl python preprocess_route.py
```

Expected new files per route:

- `<route>/route_gnss_base.csv`
- `<route>/<route>_routepoint_list_base.yml`

Raw files (`<route>_gnss.csv`, `<route>_routepoint_list.yml`, `meta/*.yml`)
are left untouched.

## 2. Estimate the per-route alignment

```bash
conda run -n dl python scripts/estimate_route_alignment.py
```

Optional:

```bash
conda run -n dl python scripts/estimate_route_alignment.py \
    --routes ringroad-711 ugm_baru \
    --trim-frac 0.9 --iters 6
```

Expected files per route:

- `<route>/align_gnss_odom.npz`
- `<route>/align_gnss_odom.png`

Check the printed values:

- `scale` should be close to `1.0`.
- `rms` should be small (for example below `0.3 m`).
- `max` should not be dominated by a few outliers.
- The residual plot should not grow steadily; steady growth indicates ZED yaw
  drift and suggests a windowed alignment.

## 3. Regenerate LiDAR projections

```bash
conda run -n dl python preprocess_lidar.py
```

The script writes `lidar/img/frame_version.txt`. When it does not match
`config.frame_version`, existing images are regenerated instead of skipped.

Alternatively, delete the old HDF5 and let `KarrDataset` rebuild it on first
use. The new name is:

```
preload_lidar_bev_front_seg_dep_frame_v2.hdf5
```

## 4. Rebuild preload caches

The new versioned names mean old caches are ignored automatically. The new
files are created on first dataset access:

```
preload_seq{seq}_pred{pred}_latlon_fix_frame_v2.npy
preload_test_split_seq{seq}_pred{pred}_frame_v2.hdf5
transfuser_lidar_bev_front_seg_dep_frame_v2.hdf5
seq{seq}_pred{pred}_w{w}_frame_v2.npy
```

To force a rebuild, delete those files.

## 5. Validate the overlay

Single frames or a video:

```bash
conda run -n dl python scripts/overlay_rp_wp.py \
    --split val --start 0 --end 300 --step 10 --out overlay_rp_wp
```

```bash
conda run -n dl python scripts/overlay_rp_wp.py \
    --split test --video overlay_test.avi --max-frames 300
```

Check:

- rp1 (white) and rp2 (yellow) lie on the road in BEV and front.
- Waypoints (white dots) follow the road and are smooth relative to rp.
- The ego cross marker is at the expected base origin.
- No systematic lateral shift between rp/waypoints and the LiDAR image.

You can also use the existing preview tool:

```bash
conda run -n dl python preview_dataloader.py
```

## 6. Retrain

The BEV/front inputs and the rp targets changed, so retrain every model that
consumes them (`ai23`, `ai33`, `xr20`, transfuser). Deployment engines and
preloads must be rebuilt from the new caches.

## 7. Tests

```bash
conda run -n dl python -m unittest discover -s tests -v
conda run -n dl python -m py_compile transforms.py config.py \
    preprocess_lidar.py preprocess_route.py a_dataloader.py \
    transfuser_dataloader.py jetson_deployment_dataloader.py \
    preprocess_preload_test_split.py preview_join_data_all.py \
    scripts/estimate_route_alignment.py scripts/overlay_rp_wp.py
```

## 8. Troubleshooting

### rp is rotated by a constant angle

- The alignment was estimated before `gnss_ant_offset_base` was set, or the
  routes changed after the alignment was created. Re-run steps 1 and 2.
- Check `align_gnss_odom.png`; a constant yaw error shows as a rigidly
  rotated trajectory.

### rp looks shifted by a constant distance

- `T_lidar_proj` still has the LiDAR origin, so the LiDAR projections are
  offset from the odom origin. Set the LiDAR translation in `T_lidar_proj`.
- `gnss_ant_offset_base` is still zero, so the GNSS route points are still
  antenna-centred.

### Front rp markers are too high or too low

- `ground_z_base` does not match the base height above the ground. Set it to
  `-base_height_above_ground`.

### Alignment residual grows along the route

- ZED odom yaw drift. Use a windowed alignment or fuse ZED yaw with GNSS
  course over ground. The current `align_mode` config field is reserved for
  that extension.

### Dataset refuses to load / missing alignment

- `align_gnss_odom.npz` is required. Run step 2.
- If `route_gnss_base.csv` is missing, the loader falls back to the raw GNSS
  with a warning; regenerate step 1 to remove the warning and get the base
  frame.
