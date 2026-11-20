"""Shared frame transforms for GNSS, ZED odom and LiDAR projections.

All public helpers are deliberately dependency-light (NumPy only, optional
PyTorch) so they can be used both by preprocessing scripts and by the dataset
loader without importing heavy model code.

Frame contract
--------------
projection frame
    The coordinate order expected by ``gen_bev_front_rear_seg_dep_*``:
    ``(x_forward, y_up, z_left)``.  This is the legacy
    ``(ptx, pty, ptz) = (-pcd_y, pcd_z, pcd_x)`` remap and must not change.

base frame
    ZED odom origin, ``x`` forward, ``y`` left, ``z`` up.  Waypoints already
    live in this frame; ``transform_2d_points`` defines its 2D convention.

route/local frame
    The 2D ego frame produced by ``transform_2d_points``.  rp1/rp2 are
    converted into the ZED odom world and then passed through the exact same
    function as waypoints, so both share one convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # torch is optional for pure-NumPy tooling
    import torch
except Exception:  # pragma: no cover - torch is present in the training env
    torch = None


LAT_M_PER_DEG = 40008000.0 / 360.0
LON_M_PER_DEG = 40075000.0 / 360.0


# Legacy remap: raw LiDAR -> projection frame (x_forward, y_up, z_left).
# Keeping it as a 4x4 makes it composable with a measured base offset:
#   T_lidar_proj = T_base_offset @ T_LEGACY_REMAP
T_LEGACY_REMAP = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


FRAME_DEFAULTS = {
    "frame_version": 3,
    "lidar_hdf5_name": "preload_lidar_bev_front_seg_dep_frame_v3.hdf5",
    "preload_name_template": "preload_seq{seq}_pred{pred}_latlon_fix_frame_v3.npy",
    "test_preload_name_template": "preload_test_split_seq{seq}_pred{pred}_frame_v3.hdf5",
    "transfuser_lidar_hdf5_name": "transfuser_lidar_bev_front_seg_dep_frame_v3.hdf5",
    "transfuser_preload_name_template": "seq{seq}_pred{pred}_w{w}_frame_v3.npy",
    "align_gnss_odom_filename": "align_gnss_odom.npz",
    "route_gnss_base_filename": "route_gnss_base.csv",
    "routepoint_base_template": "{route}_routepoint_list_base.yml",
    "T_lidar_proj": T_LEGACY_REMAP,
    "ground_z_base": -1.5,
}


def cfg(config, name):
    """Read a frame config value with a safe fallback for old saved configs."""
    if name not in FRAME_DEFAULTS:
        raise KeyError(name)
    return getattr(config, name, FRAME_DEFAULTS[name])


def as_4x4(matrix) -> np.ndarray:
    """Return ``matrix`` as a float64 4x4 homogeneous transform."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got shape {matrix.shape}")
    return matrix


def lidar_raw_to_proj_numpy(xyz: np.ndarray, T_lidar_proj=T_LEGACY_REMAP) -> np.ndarray:
    """Map raw LiDAR ``xyz`` (N, >=3) into the projection frame.

    ``T_lidar_proj`` must already include the legacy axis remap.  The default
    reproduces the previous ``(-y, z, x)`` behaviour exactly.
    """
    xyz = np.asarray(xyz)
    T = as_4x4(T_lidar_proj)
    points = xyz[..., :3].reshape(-1, 3)
    rotated = points @ T[:3, :3].T + T[:3, 3]
    return rotated.reshape(xyz.shape[:-1] + (3,))


def lidar_raw_to_proj_tensor(xyz, T_lidar_proj=T_LEGACY_REMAP):
    """Torch variant of :func:`lidar_raw_to_proj_numpy`."""
    if torch is None:
        raise RuntimeError("PyTorch is required for lidar_raw_to_proj_tensor")
    T = torch.as_tensor(np.asarray(T_lidar_proj, dtype=np.float32), dtype=xyz.dtype, device=xyz.device)
    points = xyz[..., :3].reshape(-1, 3)
    return (points @ T[:3, :3].T + T[:3, 3]).reshape(xyz.shape[:-1] + (3,))


def latlon_to_enu(lat, lon, lat0, lon0):
    """Convert WGS84 degrees to local East/North metres around ``lat0/lon0``."""
    east = (np.asarray(lon) - lon0) * LON_M_PER_DEG * np.cos(np.radians(lat0))
    north = (np.asarray(lat) - lat0) * LAT_M_PER_DEG
    return east, north


def enu_to_latlon(east, north, lat0, lon0):
    """Inverse of :func:`latlon_to_enu`."""
    lon = lon0 + np.asarray(east) / (LON_M_PER_DEG * np.cos(np.radians(lat0)))
    lat = lat0 + np.asarray(north) / LAT_M_PER_DEG
    return lat, lon


def yaw_from_quaternion_xyzw(quat) -> float:
    """Extract standard ROS yaw from ``[x, y, z, w]``.

    This matches ``euler_from_quaternion(...)[2]`` used throughout the repo.
    """
    x, y, z, w = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def antenna_to_base_latlon(lat, lon, yaw, r_ant_base, lat0=None, lon0=None):
    """Shift a GNSS antenna fix to the ZED/odom base origin.

    ``r_ant_base`` is ``[forward, left]`` (metres) of the antenna expressed in
    the base frame.  ``yaw`` is the standard ROS yaw from the ZED odometry.
    """
    if lat0 is None:
        lat0 = lat
    if lon0 is None:
        lon0 = lon
    east, north = latlon_to_enu(lat, lon, lat0, lon0)
    rf, rl = float(r_ant_base[0]), float(r_ant_base[1])
    c, s = np.cos(yaw), np.sin(yaw)
    d_east = c * rf - s * rl
    d_north = s * rf + c * rl
    return enu_to_latlon(east - d_east, north - d_north, lat0, lon0)


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def apply_alignment(east, north, align):
    """Map local ENU metres to ZED odom metres with a fitted alignment."""
    R = np.asarray(align["R"], dtype=np.float64)
    t = np.asarray(align["t"], dtype=np.float64)
    scale = float(align.get("scale", 1.0))
    point = scale * R @ np.array([east, north], dtype=np.float64) + t
    return point[0], point[1]


def invert_alignment(x, y, align):
    """Map ZED odom metres back to local ENU metres."""
    R = np.asarray(align["R"], dtype=np.float64)
    t = np.asarray(align["t"], dtype=np.float64)
    scale = float(align.get("scale", 1.0))
    point = R.T @ (np.array([x, y], dtype=np.float64) - t) / scale
    return point[0], point[1]


def rp_odom_to_local(rp_x, rp_y, ego_x, ego_y, ego_heading):
    """Replicate ``transform_2d_points`` for an already world-frame target.

    This is mathematically identical to::

        transform_2d_points(
            np.zeros((1, 3)),
            0.0, rp_x, rp_y,
            np.pi / 2 - ego_heading, ego_x, ego_y,
        )

    Keeping it in one place guarantees rp and waypoints share one convention.
    """
    r = np.pi / 2.0 - ego_heading
    c, s = np.cos(r), np.sin(r)
    # R2 = [[c, s], [-s, c]] is the world-from-local rotation used by
    # transform_2d_points.  Its inverse is its transpose.
    d = np.array([rp_x - ego_x, rp_y - ego_y], dtype=np.float64)
    out = np.array([[c, s], [-s, c]], dtype=np.float64).T @ d
    return out[0], out[1]


def fit_rigid_2d(src, dst, with_scale=True, trim_frac=1.0, iters=6):
    """Fit ``dst ~= scale * R @ src + t`` robustly by trimmed Umeyama.

    Parameters
    ----------
    src, dst : (N, 2) arrays
        Source (ENU) and destination (ZED odom) points.
    with_scale : bool
        Estimate a scale factor.  ZED odom is metric so scale is expected to
        be close to 1; it is useful as a diagnostic.
    trim_frac : float
        Fraction of best-fitting points kept on each trimming iteration.
    iters : int
        Number of trimming iterations.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src and dst must both be (N, 2)")
    if len(src) < 3:
        raise ValueError("At least 3 point pairs are required")

    mask = np.ones(len(src), dtype=bool)
    R = np.eye(2)
    t = np.zeros(2)
    scale = 1.0
    for _ in range(max(1, iters)):
        x = src[mask]
        y = dst[mask]
        mu_x = x.mean(axis=0)
        mu_y = y.mean(axis=0)
        xc = x - mu_x
        yc = y - mu_y
        sigma = (yc.T @ xc) / len(x)
        U, D, Vt = np.linalg.svd(sigma)
        S = np.eye(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[1, 1] = -1.0
        R = U @ S @ Vt
        if with_scale:
            var_x = (xc ** 2).sum() / len(x)
            scale = float(np.trace(np.diag(D) @ S) / var_x) if var_x > 0 else 1.0
        else:
            scale = 1.0
        t = mu_y - scale * R @ mu_x

        residual = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)
        if trim_frac >= 1.0 or trim_frac <= 0.0:
            break
        keep = max(3, int(len(src) * trim_frac))
        order = np.argsort(residual)
        new_mask = np.zeros(len(src), dtype=bool)
        new_mask[order[:keep]] = True
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    residual = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    return {
        "R": R,
        "t": t,
        "scale": scale,
        "theta": theta,
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "residual_max": float(np.max(residual)),
        "inlier_mask": mask,
    }


def save_alignment(path, R, t, scale, anchor_latlon, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "R": np.asarray(R, dtype=np.float64),
        "t": np.asarray(t, dtype=np.float64),
        "scale": np.float64(scale),
        "anchor_latlon": np.asarray(anchor_latlon, dtype=np.float64),
    }
    for key, value in metadata.items():
        payload[key] = value
    np.savez(path, **payload)


def load_alignment(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing route alignment: {path}. "
            "Run scripts/estimate_route_alignment.py first."
        )
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}
