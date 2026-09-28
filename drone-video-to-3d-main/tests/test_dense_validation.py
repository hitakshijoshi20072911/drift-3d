import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from sih_drone_pipeline.dense_validation import evaluate_dense_geometry
from sih_drone_pipeline.cli import _validate_geometry


class DenseValidationTests(unittest.TestCase):
    @staticmethod
    def _write_cloud(path: Path, points: np.ndarray) -> None:
        trimesh.points.PointCloud(np.asarray(points, dtype=np.float64)).export(path)

    def test_identical_point_clouds_score_perfectly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.random.default_rng(3).normal(size=(1_000, 3))
            self._write_cloud(root / "reconstruction.ply", points)
            self._write_cloud(root / "reference.ply", points)

            report = evaluate_dense_geometry(
                root / "reconstruction.ply",
                root / "reference.ply",
                thresholds_m=(0.01, 0.1),
            )

            self.assertEqual(report["reconstruction"]["geometry_type"], "point_cloud")
            self.assertEqual(report["distances_m"]["reconstruction_to_reference"]["rmse_m"], 0.0)
            self.assertEqual(report["distances_m"]["reference_to_reconstruction"]["p95_m"], 0.0)
            for metrics in report["threshold_metrics"]:
                self.assertEqual(metrics["precision"], 1.0)
                self.assertEqual(metrics["completeness"], 1.0)
                self.assertEqual(metrics["recall"], 1.0)
                self.assertEqual(metrics["f1"], 1.0)
            self.assertTrue(report["passes_one_metre_target"])
            json.dumps(report)

    def test_capped_identical_clouds_still_use_complete_lookup_clouds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.column_stack(
                (np.arange(200, dtype=np.float64) * 10.0, np.zeros((200, 2)))
            )
            self._write_cloud(root / "reconstruction.ply", points)
            self._write_cloud(root / "reference.ply", points)

            report = evaluate_dense_geometry(
                root / "reconstruction.ply",
                root / "reference.ply",
                thresholds_m=(0.01,),
                max_samples=20,
            )

            self.assertEqual(report["reconstruction"]["evaluated_samples"], 20)
            self.assertEqual(report["reference"]["nearest_neighbour_target_points"], 200)
            self.assertEqual(report["threshold_metrics"][0]["f1"], 1.0)

    def test_shifted_cloud_reports_distance_statistics_and_threshold_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
            reconstruction = reference + np.array([1.0, 0.0, 0.0])
            self._write_cloud(root / "reconstruction.ply", reconstruction)
            self._write_cloud(root / "reference.ply", reference)

            report = evaluate_dense_geometry(
                root / "reconstruction.ply",
                root / "reference.ply",
                thresholds_m=(0.5, 1.0),
            )

            for direction in report["distances_m"].values():
                self.assertEqual(direction["mean_m"], 1.0)
                self.assertEqual(direction["median_m"], 1.0)
                self.assertEqual(direction["rmse_m"], 1.0)
                self.assertEqual(direction["p95_m"], 1.0)
            self.assertEqual(report["threshold_metrics"][0]["f1"], 0.0)
            self.assertEqual(report["threshold_metrics"][1]["f1"], 1.0)

    def test_subset_reconstruction_distinguishes_precision_and_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cloud(
                root / "reconstruction.ply", np.array([[0.0, 0.0, 0.0]])
            )
            self._write_cloud(
                root / "reference.ply",
                np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            )

            metrics = evaluate_dense_geometry(
                root / "reconstruction.ply",
                root / "reference.ply",
                thresholds_m=(0.1,),
            )["threshold_metrics"][0]

            self.assertEqual(metrics["precision"], 1.0)
            self.assertEqual(metrics["completeness"], 0.5)
            self.assertEqual(metrics["recall"], 0.5)
            self.assertAlmostEqual(metrics["f1"], 2.0 / 3.0)

            report = evaluate_dense_geometry(
                root / "reconstruction.ply",
                root / "reference.ply",
                thresholds_m=(1.0,),
            )
            self.assertFalse(report["passes_one_metre_target"])

    def test_triangle_meshes_are_area_sampled_and_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vertices = np.array(
                [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]
            )
            faces = np.array([[0, 1, 2]])
            reference = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            reconstruction = reference.copy()
            reconstruction.apply_translation((0.0, 0.0, 3.0))
            reference.export(root / "reference.ply")
            reconstruction.export(root / "reconstruction.ply")

            report = evaluate_dense_geometry(
                root / "reconstruction.ply",
                root / "reference.ply",
                thresholds_m=(1.0,),
                max_samples=2_000,
                seed=7,
            )

            self.assertEqual(report["reference"]["geometry_type"], "mesh")
            self.assertEqual(report["reference"]["evaluated_samples"], 2_000)
            self.assertGreaterEqual(
                report["distances_m"]["reconstruction_to_reference"]["mean_m"], 3.0
            )
            self.assertEqual(report["threshold_metrics"][0]["precision"], 0.0)
            self.assertEqual(report["threshold_metrics"][0]["recall"], 0.0)

    def test_rejects_invalid_controls_and_missing_input(self):
        with self.assertRaises(ValueError):
            evaluate_dense_geometry("missing.ply", "also_missing.ply", thresholds_m=())
        with self.assertRaises(ValueError):
            evaluate_dense_geometry(
                "missing.ply", "also_missing.ply", thresholds_m=(0.1,), max_samples=0
            )
        with self.assertRaises(FileNotFoundError):
            evaluate_dense_geometry(
                "missing.ply", "also_missing.ply", thresholds_m=(0.1,)
            )

    def test_post_run_command_attaches_report_and_updates_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            self._write_cloud(output / "point_cloud.ply", points)
            self._write_cloud(output / "reference.ply", points)
            (output / "run_report.json").write_text("{}", encoding="utf-8")
            (output / "viewer_metadata.json").write_text("{}", encoding="utf-8")

            exit_code = _validate_geometry(SimpleNamespace(
                output=str(output),
                reference=str(output / "reference.ply"),
                reconstruction=None,
                threshold_m=[0.1],
                max_samples=100,
                seed=9,
            ))

            self.assertEqual(exit_code, 0)
            report = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
            attached = report["validation"]["dense_geometry"]
            self.assertEqual(attached["protocol"]["thresholds_m"], [0.1])
            self.assertEqual(len(attached["reference"]["sha256"]), 64)
            self.assertFalse(attached["reference_is_distinct_from_reconstruction"])
            self.assertIsNone(report["targets"]["one_metre_surface_accuracy"])
            self.assertTrue((output / "dense_geometry_validation.json").is_file())
            viewer = json.loads((output / "viewer_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(viewer["validation"]["dense_geometry"], attached)


if __name__ == "__main__":
    unittest.main()
