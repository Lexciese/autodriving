import unittest

import numpy as np

from transforms import (
    T_LEGACY_REMAP,
    antenna_to_base_latlon,
    apply_alignment,
    cfg,
    enu_to_latlon,
    fit_rigid_2d,
    invert_alignment,
    latlon_to_enu,
    lidar_raw_to_proj_numpy,
    rp_odom_to_local,
)


class FrameTransformTests(unittest.TestCase):
    def test_legacy_lidar_remap(self):
        xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        expected = np.column_stack((-xyz[:, 1], xyz[:, 2], xyz[:, 0]))
        np.testing.assert_allclose(
            lidar_raw_to_proj_numpy(xyz, T_LEGACY_REMAP), expected
        )

    def test_latlon_roundtrip(self):
        lat0, lon0 = -7.77, 110.38
        lat, lon = -7.7705, 110.3812
        east, north = latlon_to_enu(lat, lon, lat0, lon0)
        lat2, lon2 = enu_to_latlon(east, north, lat0, lon0)
        self.assertAlmostEqual(lat, float(lat2), places=12)
        self.assertAlmostEqual(lon, float(lon2), places=12)

    def test_alignment_roundtrip(self):
        theta = np.deg2rad(35.0)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        align = {"R": R, "t": np.array([12.0, -3.0]), "scale": 1.0}
        x, y = apply_alignment(4.0, -2.5, align)
        e, n = invert_alignment(x, y, align)
        self.assertAlmostEqual(e, 4.0, places=10)
        self.assertAlmostEqual(n, -2.5, places=10)

    def test_fit_rigid_recovers_known_transform(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(200, 2))
        theta = np.deg2rad(17.0)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = np.array([5.0, -8.0])
        dst = src @ R.T + t
        fit = fit_rigid_2d(src, dst, with_scale=False)
        np.testing.assert_allclose(fit["R"], R, atol=1e-10)
        np.testing.assert_allclose(fit["t"], t, atol=1e-10)
        self.assertLess(fit["residual_rms"], 1e-10)

    def test_rp_matches_transform_2d_convention(self):
        # transform_2d_points(zeros, 0, rp, pi/2 - heading, ego) reduces to
        # R2.T @ (rp - ego) with R2 = [[c, s], [-s, c]], r = pi/2 - heading.
        rp = (1.0, 3.0)
        ego = (1.0, 2.0)
        heading = 0.0
        out = rp_odom_to_local(rp[0], rp[1], ego[0], ego[1], heading)
        r = np.pi / 2.0 - heading
        c, s = np.cos(r), np.sin(r)
        expected = np.array([[c, s], [-s, c]]).T @ np.array([rp[0] - ego[0], rp[1] - ego[1]])
        np.testing.assert_allclose(out, expected)

    def test_antenna_lever_arm_zero_is_identity(self):
        lat, lon = -7.77, 110.38
        lat2, lon2 = antenna_to_base_latlon(lat, lon, 0.7, [0.0, 0.0], lat0=lat, lon0=lon)
        self.assertAlmostEqual(lat, float(lat2), places=12)
        self.assertAlmostEqual(lon, float(lon2), places=12)

    def test_cfg_fallback_for_old_configs(self):
        class LegacyConfig:
            pass

        self.assertEqual(cfg(LegacyConfig(), "frame_version"), 3)
        np.testing.assert_allclose(cfg(LegacyConfig(), "T_lidar_proj"), T_LEGACY_REMAP)


if __name__ == "__main__":
    unittest.main()
