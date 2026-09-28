import tempfile
import unittest
from pathlib import Path

import numpy as np

from sih_drone_pipeline.export_gis import (
    _clean_mesh,
    _fill_small_dsm_holes,
    _filter_statistical_outliers,
    _preserved_mesh_quality,
)


class EnuToProjectedTests(unittest.TestCase):
    ORIGIN = {"latitude": 30.261171, "longitude": -97.755681, "altitude_m": 445.3}

    def test_matches_direct_utm_projection_far_from_origin(self):
        from pyproj import Transformer

        from sih_drone_pipeline.export_gis import _utm_crs, enu_to_projected
        from sih_drone_pipeline.telemetry import TelemetrySample
        from sih_drone_pipeline.validation import _to_enu

        utm = _utm_crs(self.ORIGIN["latitude"], self.ORIGIN["longitude"])
        to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
        # Real landmarks around the 11-min flight, up to ~1 km from the origin.
        targets = [(30.260415, -97.756751, 420.0), (30.2701, -97.7526, 430.0), (30.2585, -97.7470, 410.0)]
        enu = np.array([
            _to_enu(TelemetrySample(time_s=0.0, latitude=lat, longitude=lon, altitude_m=alt), self.ORIGIN)
            for lat, lon, alt in targets
        ])
        projected = enu_to_projected(enu, self.ORIGIN, utm)
        for (lat, lon, alt), row in zip(targets, projected):
            easting, northing = to_utm.transform(lon, lat)
            self.assertAlmostEqual(row[0], easting, delta=0.001)
            self.assertAlmostEqual(row[1], northing, delta=0.001)
            self.assertAlmostEqual(row[2], alt, delta=0.001)

    def test_old_offset_method_was_off_by_metres(self):
        from pyproj import Transformer

        from sih_drone_pipeline.export_gis import _utm_crs, enu_to_projected

        utm = _utm_crs(self.ORIGIN["latitude"], self.ORIGIN["longitude"])
        to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
        e0, n0 = to_utm.transform(self.ORIGIN["longitude"], self.ORIGIN["latitude"])
        enu = np.array([[0.0, 800.0, 0.0]])
        correct = enu_to_projected(enu, self.ORIGIN, utm)[0]
        naive = np.array([e0, n0 + 800.0])
        # Grid convergence (~0.6 deg here) shifts a point 800 m north by ~8 m sideways.
        self.assertGreater(np.linalg.norm(correct[:2] - naive), 5.0)


class ExportQualityTests(unittest.TestCase):
    def test_fills_supported_interior_dsm_hole_but_not_unsupported_edge(self):
        dsm = np.array([
            [np.nan, 10.0, 10.0],
            [10.0, np.nan, 12.0],
            [10.0, 12.0, 12.0],
        ])
        filled = _fill_small_dsm_holes(dsm, iterations=1)
        self.assertTrue(np.isnan(filled[0, 0]))
        self.assertTrue(np.isfinite(filled[1, 1]))

    def test_mesh_cleanup_removes_tiny_disconnected_component(self):
        import trimesh

        large = trimesh.creation.box()
        tiny = trimesh.Trimesh(
            vertices=[[10, 0, 0], [10, 1, 0], [10, 0, 1]],
            faces=[[0, 1, 2]],
            process=False,
        )
        combined = trimesh.util.concatenate([large, tiny])
        cleaned, quality = _clean_mesh(combined)
        self.assertEqual(quality["components_before"], 2)
        self.assertEqual(quality["components_after"], 1)
        self.assertEqual(len(cleaned.faces), len(large.faces))

    def test_mesh_cleanup_removes_implausibly_long_bridge_face(self):
        import trimesh

        regular = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        vertices = np.vstack([regular.vertices, [[0, 0, 0], [20, 0, 0], [0, 20, 0]]])
        faces = np.vstack([regular.faces, [[len(regular.vertices), len(regular.vertices) + 1, len(regular.vertices) + 2]]])
        combined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        cleaned, quality = _clean_mesh(combined)
        self.assertEqual(quality["removed_oversized_faces"], 1)
        self.assertEqual(len(cleaned.faces), len(regular.faces))

    def test_mesh_cleanup_removes_face_without_point_support(self):
        import trimesh

        near = trimesh.creation.box(extents=[1, 1, 1])
        far = trimesh.Trimesh(
            vertices=[[20, 0, 0], [20, 0.2, 0], [20, 0, 0.2]],
            faces=[[0, 1, 2]],
            process=False,
        )
        combined = trimesh.util.concatenate([near, far])
        support = np.asarray(near.vertices)
        cleaned, quality = _clean_mesh(combined, support_points=support, support_distance_m=1.0)
        self.assertEqual(quality["removed_unsupported_faces"], 1)
        self.assertEqual(len(cleaned.faces), len(near.faces))
        self.assertEqual(quality["final_oversized_faces"], 0)

    def test_upstream_cleaned_mesh_is_reported_without_fragmenting_it_again(self):
        import trimesh

        first = trimesh.creation.box()
        second = trimesh.creation.box(transform=trimesh.transformations.translation_matrix([3, 0, 0]))
        mesh = trimesh.util.concatenate([first, second])
        vertices_before = np.asarray(mesh.vertices).copy()
        faces_before = np.asarray(mesh.faces).copy()

        quality = _preserved_mesh_quality(mesh, {"removed_unsupported_faces": 0})

        np.testing.assert_array_equal(mesh.vertices, vertices_before)
        np.testing.assert_array_equal(mesh.faces, faces_before)
        self.assertEqual(quality["components_after"], 2)
        self.assertFalse(quality["export_cleanup_applied"])
        self.assertEqual(quality["export_policy"], "preserve_upstream_cleaned_mesh")

    def test_statistical_filter_removes_isolated_point(self):
        rng = np.random.default_rng(42)
        cluster = rng.normal(0, 0.05, size=(200, 3))
        points = np.vstack([cluster, [[100, 100, 100]]])
        filtered, _, quality = _filter_statistical_outliers(points)
        self.assertLess(len(filtered), len(points))
        self.assertGreaterEqual(quality["removed_points"], 1)


if __name__ == "__main__":
    unittest.main()
