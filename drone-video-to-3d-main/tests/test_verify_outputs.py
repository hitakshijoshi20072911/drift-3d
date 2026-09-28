import tempfile
import unittest
from pathlib import Path

import trimesh

from sih_drone_pipeline.verify_outputs import (
    _at_most,
    _final_mesh_edge_max,
    _glb_mesh_count,
    _ply_header_counts,
    _surface_accuracy_results,
    _substantive_geometry_checks,
    footprint_coverage,
)


class FootprintCoverageTests(unittest.TestCase):
    @staticmethod
    def _l_shaped_flight():
        import numpy as np

        valid = np.zeros((400, 400), dtype=bool)       # 200 m x 200 m at 0.5 m cells
        valid[:, :200] = True                           # west half
        valid[200:, :] = True                           # plus south half: an L, 75% of the box
        return valid

    def test_non_rectangular_flight_with_full_data_scores_near_100_percent(self):
        valid = self._l_shaped_flight()
        result = footprint_coverage(valid, 0.5)
        self.assertAlmostEqual(valid.mean(), 0.75)      # the old bounding-box measure
        self.assertGreater(result["fraction"], 0.97)
        self.assertEqual(result["analysis_cell_m"], 5.0)

    def test_interior_hole_still_counts_against_coverage(self):
        valid = self._l_shaped_flight()
        valid[240:360, 40:160] = False                  # 60 m x 60 m "lake" inside the footprint
        result = footprint_coverage(valid, 0.5)
        self.assertLess(result["fraction"], 0.9)
        self.assertGreater(result["fraction"], 0.8)

    def test_sparse_speckle_is_not_counted_as_missing(self):
        import numpy as np

        rng = np.random.default_rng(0)
        valid = rng.random((400, 400)) < 0.15           # sparse points at 0.5 m cells
        self.assertGreater(footprint_coverage(valid, 0.5)["fraction"], 0.97)


class LightweightFormatTests(unittest.TestCase):
    def test_reads_ply_counts_without_loading_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            mesh = trimesh.creation.box()
            mesh.export(path)
            self.assertEqual(_ply_header_counts(path), (len(mesh.vertices), len(mesh.faces)))

    def test_validates_glb_without_decoding_binary_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.glb"
            trimesh.creation.box().export(path)
            self.assertGreater(_glb_mesh_count(path), 0)

    def test_zero_is_a_valid_at_most_measurement(self):
        self.assertTrue(_at_most(0.0, 1.0))
        self.assertFalse(_at_most(None, 1.0))
        self.assertFalse(_at_most(float("nan"), 1.0))

    def test_verifier_reports_post_cleanup_mesh_edge_maximum(self):
        self.assertEqual(_final_mesh_edge_max({
            "longest_edge_max_before_m": 87.7,
            "longest_edge_max_after_m": 7.9,
        }), 7.9)
        self.assertEqual(_final_mesh_edge_max({
            "longest_edge_max_before_m": 8.5,
        }), 8.5)

    def test_surface_accuracy_combines_distinct_dense_and_surveyed_evidence(self):
        validation = {
            "surveyed_checkpoints": {"passes_one_metre_target": True},
            "dense_geometry": {
                "passes_one_metre_target": False,
                "reference_is_distinct_from_reconstruction": True,
            },
        }
        self.assertEqual(_surface_accuracy_results(validation), (True, False))
        validation["dense_geometry"]["reference_is_distinct_from_reconstruction"] = False
        self.assertEqual(_surface_accuracy_results(validation), (True, True))

    def test_substantive_geometry_rejects_tiny_but_valid_artifacts(self):
        formats = {
            "las": {"valid": True, "points": 13},
            "geotiff": {"valid": True, "valid_cells": 13},
            "mesh.ply": {"valid": True, "vertices": 3, "faces": 1},
        }
        result = _substantive_geometry_checks(
            formats, {"sparse_metrics": {"registered_images": 78}}
        )
        self.assertFalse(result["passed"])
        self.assertFalse(all(result["checks"].values()))

    def test_substantive_geometry_scales_with_registered_camera_count(self):
        formats = {
            "las": {"valid": True, "points": 5000},
            "geotiff": {"valid": True, "valid_cells": 3000},
            "mesh.ply": {"valid": True, "vertices": 2500, "faces": 4000},
        }
        result = _substantive_geometry_checks(
            formats, {"sparse_metrics": {"registered_images": 100}}
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["thresholds"]["minimum_point_count"], 2000)


if __name__ == "__main__":
    unittest.main()
