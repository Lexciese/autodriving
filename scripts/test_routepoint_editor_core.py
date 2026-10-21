import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import routepoint_editor_core as core

SAMPLE = Path(__file__).resolve().parent / "utara_routepoint_list.yml"


class RoutepointEditorCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = core.load_route_document(SAMPLE)

    def test_load_sample(self):
        doc = self.doc
        self.assertEqual(
            set(doc),
            {
                "first_point",
                "last_point",
                "route_point",
                "extra",
                "route_extra",
                "origin",
                "path",
            },
        )
        self.assertEqual(len(doc["route_point"]["latitude"]), 8103)
        self.assertEqual(len(doc["route_point"]["longitude"]), 8103)
        self.assertAlmostEqual(doc["first_point"]["latitude"], -7.753441)
        self.assertAlmostEqual(doc["last_point"]["longitude"], 110.38358)

    def test_round_trip_and_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "routepoint_list.yml"
            core.save_route_document(self.doc, target, create_backup=True)
            self.assertTrue(target.is_file())
            reloaded = core.load_route_document(target)
            self.assertEqual(
                reloaded["route_point"]["latitude"],
                [core._round_coord(value) for value in self.doc["route_point"]["latitude"]],
            )
            self.assertEqual(
                reloaded["route_point"]["longitude"],
                [core._round_coord(value) for value in self.doc["route_point"]["longitude"]],
            )
            backup = core.save_route_document(self.doc, target, create_backup=True)
            self.assertIsNotNone(backup)
            if backup is None:
                self.fail("backup was not created")
            self.assertTrue(backup.is_file())
            self.assertIn(".bak-", backup.name)
            lines = target.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "first_point:")
            self.assertEqual(lines[3], "last_point:")
            self.assertEqual(lines[6], "route_point:")
            self.assertEqual(lines[7], "  latitude:")
            self.assertTrue(lines[8].startswith("  - "))

    def test_projection_inverse(self):
        x, y = core.route_local_xy(self.doc)
        lat0, lon0 = core.origin_of(self.doc)
        lat, lon = core.local_to_latlon(x, y, lat0, lon0)
        self.assertTrue(np.allclose(lat, self.doc["route_point"]["latitude"], atol=1e-9))
        self.assertTrue(np.allclose(lon, self.doc["route_point"]["longitude"], atol=1e-9))

    def test_audit_finds_flags(self):
        doc = {
            "first_point": {"latitude": 0.0, "longitude": 0.0},
            "last_point": {"latitude": 0.0, "longitude": 0.0},
            "route_point": {
                "latitude": [0.0, 0.0, 0.0, 0.0, 0.0],
                "longitude": [0.0, 0.00005, 0.0003, 0.0005, 0.0005001],
            },
            "origin": {"latitude": 0.0, "longitude": 0.0},
        }
        audit = core.compute_audit(
            doc,
            min_gap=2.0,
            max_gap=12.0,
            turn_threshold_deg=5.0,
            duplicate_gap=0.05,
        )
        types = [flag["type"] for flag in audit["flags"]]
        self.assertIn("duplicate", types)
        self.assertIn("gap", types)
        self.assertGreaterEqual(audit["summary"]["flag_count"], 2)
        self.assertEqual(len(audit["segment_distance"]), 4)

    def test_edit_operations(self):
        doc = self.doc
        updated = core.set_route_point(doc, 10, -7.75, 110.38)
        self.assertNotEqual(
            updated["route_point"]["latitude"][10],
            doc["route_point"]["latitude"][10],
        )
        self.assertAlmostEqual(updated["origin"]["latitude"], doc["origin"]["latitude"])

        inserted, index = core.insert_midpoint(doc, 10)
        self.assertEqual(len(inserted["route_point"]["latitude"]), 8104)
        self.assertEqual(index, 11)

        deleted, selected = core.delete_route_point(doc, 10)
        self.assertEqual(len(deleted["route_point"]["latitude"]), 8102)
        self.assertEqual(selected, 10)

        smoothed = core.smooth_route_point(doc, 10, radius=2)
        self.assertEqual(len(smoothed["route_point"]["latitude"]), 8103)

        moved_first = core.set_endpoint(doc, "first", -7.75, 110.38)
        self.assertAlmostEqual(moved_first["first_point"]["latitude"], -7.75)
        self.assertAlmostEqual(updated["origin"]["latitude"], -7.753441)

    def test_reference_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "utara_gnss.csv"
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["latitude", "longitude"])
                writer.writerow([-7.753441, 110.384121])
                writer.writerow([-7.7534, 110.3841])
            reference = core.load_reference_csv(path)
            self.assertEqual(len(reference["latitude"]), 2)
            self.assertEqual(reference["path"], str(path))

    def test_choose_osm_zoom_bounded(self):
        zoom = core.choose_osm_zoom(-7.8, -7.7, 110.3, 110.5, max_tiles=16)
        self.assertGreaterEqual(zoom, 3)
        self.assertLessEqual(zoom, 18)
        x0 = int(core._lon_to_tile_x(110.3, zoom))
        x1 = int(core._lon_to_tile_x(110.5, zoom))
        y0 = int(core._lat_to_tile_y(-7.7, zoom))
        y1 = int(core._lat_to_tile_y(-7.8, zoom))
        self.assertLessEqual((x1 - x0 + 1) * (y1 - y0 + 1), 16)


if __name__ == "__main__":
    unittest.main()
