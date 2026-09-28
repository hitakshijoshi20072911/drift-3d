import csv
import math
import tempfile
import unittest
from pathlib import Path

from sih_drone_pipeline.validation import ground_truth_trajectory_report, surveyed_checkpoint_report


class ValidationTests(unittest.TestCase):
    def test_surveyed_checkpoint_report_requires_three_and_preserves_metric_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "checkpoint_id", "known_x_m", "known_y_m", "known_z_m",
                    "reconstructed_x_m", "reconstructed_y_m", "reconstructed_z_m",
                ])
                writer.writerows([
                    (1, 0, 0, 0, 0.2, 0, 0),
                    (2, 10, 0, 0, 10, 0.3, 0),
                    (3, 0, 10, 0, 0, 10, 0.4),
                    (4, 0, 0, 10, "", "", ""),
                ])
            report = surveyed_checkpoint_report(path)
            self.assertEqual(report["checks"], 3)
            self.assertEqual(report["checkpoint_count"], 3)
            self.assertAlmostEqual(report["maximum_3d_error_m"], 0.4)
            self.assertAlmostEqual(report["mean_3d_error_m"], 0.3)
            self.assertAlmostEqual(report["median_3d_error_m"], 0.3)
            self.assertAlmostEqual(report["std_3d_error_m"], math.sqrt(2 / 300))
            self.assertAlmostEqual(report["minimum_3d_error_m"], 0.2)
            self.assertAlmostEqual(report["p95_3d_error_m"], 0.39)
            self.assertAlmostEqual(report["rmse_x_m"], math.sqrt(0.04 / 3))
            self.assertAlmostEqual(report["rmse_y_m"], math.sqrt(0.09 / 3))
            self.assertAlmostEqual(report["rmse_z_m"], math.sqrt(0.16 / 3))
            self.assertAlmostEqual(report["rmse_horizontal_m"], math.sqrt(0.13 / 3))
            self.assertEqual(report["rmse_horizontal_m"], report["horizontal_rmse_m"])
            self.assertEqual(report["rmse_vertical_m"], report["vertical_rmse_m"])
            self.assertTrue(report["passes_one_metre_target"])
            self.assertFalse(report["asprs_checkpoint_count_compliant"])
            self.assertEqual(report["asprs_checkpoint_count_status"], "reduced_assessment")
            self.assertEqual(report["alignment_type"], "none_direct_metric_frame_comparison")

    def test_surveyed_checkpoint_report_marks_thirty_checks_count_compliant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "checkpoint_id", "known_x_m", "known_y_m", "known_z_m",
                    "reconstructed_x_m", "reconstructed_y_m", "reconstructed_z_m",
                ])
                writer.writerows(
                    (index, 0, 0, 0, 0.1, -0.2, 0.3)
                    for index in range(1, 31)
                )
            report = surveyed_checkpoint_report(path)
            self.assertEqual(report["checkpoint_count"], 30)
            self.assertEqual(report["asprs_minimum_checkpoints"], 30)
            self.assertTrue(report["asprs_checkpoint_count_compliant"])
            self.assertEqual(
                report["asprs_checkpoint_count_status"], "checkpoint_count_compliant"
            )
            self.assertEqual(report["asprs_compliance_scope"], "checkpoint_count_only")

    def test_surveyed_checkpoint_report_summarizes_optional_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "checkpoint_id", "known_x_m", "known_y_m", "known_z_m",
                    "reconstructed_x_m", "reconstructed_y_m", "reconstructed_z_m",
                    "horizontal_uncertainty_m", "vertical_uncertainty_m",
                ])
                writer.writerows([
                    ("cp1", 0, 0, 0, 0.1, 0, 0, 0.02, 0.04),
                    ("cp2", 1, 0, 0, 1.1, 0, 0, 0.04, 0.08),
                    ("cp3", 2, 0, 0, 2.1, 0, 0, "", ""),
                ])
            uncertainty = surveyed_checkpoint_report(path)["checkpoint_uncertainty"]
            self.assertEqual(uncertainty["checkpoints_with_any_uncertainty"], 2)
            self.assertEqual(uncertainty["source_fields"]["horizontal"], "horizontal_uncertainty_m")
            self.assertAlmostEqual(uncertainty["summary_m"]["horizontal"]["mean"], 0.03)
            self.assertAlmostEqual(uncertainty["summary_m"]["vertical"]["p95"], 0.078)
            self.assertEqual(uncertainty["per_checkpoint_m"]["cp1"]["vertical"], 0.04)

    def test_rigid_trajectory_alignment_preserves_metric_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images.txt"
            frames = root / "frames.csv"
            truth = root / "truth.csv"
            centers = [(0, 0, 0), (2, 0, 0), (2, 3, 0), (2, 3, 4)]
            image_lines = []
            for index, (x, y, z) in enumerate(centers):
                image_lines.extend([f"{index + 1} 1 0 0 0 {-x} {-y} {-z} 1 frame_{index:04d}.jpg", ""])
            images.write_text("\n".join(image_lines), encoding="utf-8")
            with frames.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream); writer.writerow(["image_name", "time_s"])
                writer.writerows((f"frame_{i:04d}.jpg", i) for i in range(4))
            with truth.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream); writer.writerow(["time_s", "x_m", "y_m", "z_m"])
                writer.writerows((i, x + 100, y + 200, z + 10) for i, (x, y, z) in enumerate(centers))
            report = ground_truth_trajectory_report(images, frames, truth)
            self.assertLess(report["trajectory_rmse_m"], 1e-9)
            self.assertAlmostEqual(report["trajectory_scale_ratio"], 1.0)
            self.assertAlmostEqual(report["trajectory_scale_error_percent"], 0.0)
            self.assertEqual(report["alignment_type"], "rigid_rotation_translation_no_scale")
            self.assertTrue(report["passes_one_metre_position_target"])


if __name__ == "__main__":
    unittest.main()
