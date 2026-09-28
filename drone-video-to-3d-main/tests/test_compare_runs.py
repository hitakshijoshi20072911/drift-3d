import json
import tempfile
import unittest
from pathlib import Path

from sih_drone_pipeline.compare_runs import (
    compare_output_directories,
    matches_protected_accepted_result,
    matches_protected_full_video_workload,
)


class RunComparisonTests(unittest.TestCase):
    def test_protected_baseline_match_requires_the_complete_fixed_workload(self):
        workload = {
            "video_duration_s": 251.517933,
            "target_frames": 253,
            "selected_frames": 253,
            "frame_resolution": [1920, 1080],
        }
        self.assertTrue(matches_protected_full_video_workload(workload))
        self.assertTrue(matches_protected_accepted_result(workload, 26.92 * 60))
        self.assertFalse(matches_protected_accepted_result(workload, 20 * 60))
        mutations = {
            "video_duration_s": 120.0,
            "target_frames": 66,
            "selected_frames": 66,
            "frame_resolution": [1280, 720],
        }
        for key, changed in mutations.items():
            with self.subTest(key=key):
                candidate = dict(workload)
                candidate[key] = changed
                self.assertFalse(matches_protected_full_video_workload(candidate))

    @staticmethod
    def _write_manifest(directory: Path, changed_frame: int | None = None) -> None:
        rows = ["image_name,source_index,time_s"]
        for index in range(253):
            source_index = index * 30 + (1 if index == changed_frame else 0)
            rows.append(f"frame_{index:04d}.jpg,{source_index},{index:.1f}")
        (directory / "frames.csv").write_text("\n".join(rows), encoding="utf-8")

    @staticmethod
    def _full_report(runtime: float) -> dict:
        return {
            "wall_clock_seconds": runtime,
            "input": {"frames": 253},
            "comparison_metadata": {
                "video_sha256": "a" * 64,
                "video_duration_s": 251.52,
                "target_frames": 253,
                "frame_resolution": [1920, 1080],
            },
            "sparse_metrics": {
                "input_frames": 253,
                "registered_images": 250,
                "registration_percent": 98.81,
                "mean_reprojection_error_px": 0.5,
            },
            "validation": {
                "gps_alignment_rmse_m": 3.0,
                "surveyed_checkpoints": {
                    "checks": 3,
                    "checkpoint_ids": ["CP1", "CP2", "CP3"],
                    "rmse_3d_m": 0.4,
                    "maximum_3d_error_m": 0.6,
                    "passes_one_metre_target": True,
                    "minimum_checks_required": 3,
                },
            },
            "products": {
                "point_count": 1000,
                "dsm": {"valid_fraction": 0.5},
                "mesh_quality": {"components_after": 4},
            },
            "stages": {"dense_stereo": {"seconds": 60.0}},
        }

    @staticmethod
    def _dense_report(runtime: float) -> dict:
        report = RunComparisonTests._full_report(runtime)
        summary = {
            "samples": 1000,
            "mean_m": 0.1,
            "median_m": 0.08,
            "rmse_m": 0.12,
            "p95_m": 0.2,
            "maximum_m": 0.4,
        }
        report["validation"] = {
            "gps_alignment_rmse_m": 3.0,
            "dense_geometry": {
                "alignment": "none_same_metric_coordinate_frame_required",
                "reference_is_distinct_from_reconstruction": True,
                "passes_one_metre_target": True,
                "protocol": {
                    "name": "bidirectional_nearest_neighbour_surface_distance",
                    "version": "1.0",
                    "thresholds_m": [0.25, 0.5, 1.0],
                    "mesh_sampling": "deterministic_area_weighted",
                },
                "reference": {
                    "path": "reference.ply",
                    "sha256": "d" * 64,
                },
                "max_samples": 100000,
                "seed": 26158,
                "distances_m": {
                    "reconstruction_to_reference": dict(summary),
                    "reference_to_reconstruction": dict(summary),
                },
                "threshold_metrics": [
                    {
                        "threshold_m": threshold,
                        "precision": 0.9,
                        "completeness": 0.9,
                        "f1": 0.9,
                    }
                    for threshold in (0.25, 0.5, 1.0)
                ],
            },
        }
        return report

    def test_reports_runtime_speedup_without_claiming_surface_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = root / "before"
            after = root / "after"
            before.mkdir()
            after.mkdir()
            base = {
                "wall_clock_seconds": 100.0,
                "sparse_metrics": {
                    "input_frames": 90,
                    "registered_images": 70,
                    "registration_percent": 77.78,
                    "mean_reprojection_error_px": 0.5,
                },
                "validation": {"gps_alignment_rmse_m": 3.0},
                "products": {
                    "point_count": 1000,
                    "dsm": {"valid_fraction": 0.5},
                    "mesh_quality": {"components_after": 4},
                },
                "stages": {"dense_stereo": {"seconds": 60.0}},
            }
            experimental = json.loads(json.dumps(base))
            experimental["wall_clock_seconds"] = 50.0
            experimental["sparse_metrics"]["registered_images"] = 75
            (before / "run_report.json").write_text(json.dumps(base))
            (after / "run_report.json").write_text(json.dumps(experimental))
            output = root / "comparison.json"
            result = compare_output_directories(before, after, output)
            self.assertEqual(result["speedup"], 2.0)
            self.assertFalse(result["surface_accuracy_comparable"])
            self.assertTrue(result["passes_quality_guardrails"])
            self.assertFalse(result["passes_fair_comparison_gates"])
            self.assertFalse(result["candidate_eligible_to_replace_baseline"])
            self.assertIn(
                "candidate_has_independent_surface_evidence",
                result["fair_comparison_failures"],
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".md").is_file())

    def test_fixed_253_workload_and_checkpoint_set_can_replace_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before", root / "after"
            before.mkdir()
            after.mkdir()
            baseline = self._full_report(26.92 * 60)
            candidate = self._full_report(14.5 * 60)
            self._write_manifest(before)
            self._write_manifest(after)
            (before / "run_report.json").write_text(json.dumps(baseline), encoding="utf-8")
            (after / "run_report.json").write_text(json.dumps(candidate), encoding="utf-8")

            result = compare_output_directories(before, after, root / "comparison.json")

            self.assertTrue(result["passes_quality_guardrails"])
            self.assertTrue(result["passes_fair_comparison_gates"])
            self.assertTrue(result["surface_accuracy_comparable"])
            self.assertTrue(result["candidate_eligible_to_replace_baseline"])

    def test_faster_run_cannot_hide_changed_workload_or_missing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before", root / "after"
            before.mkdir()
            after.mkdir()
            baseline = self._full_report(26.92 * 60)
            candidate = self._full_report(10.3 * 60)
            candidate["comparison_metadata"].update({
                "video_sha256": "b" * 64,
                "video_duration_s": 120.0,
                "target_frames": 66,
                "frame_resolution": [1280, 720],
            })
            candidate["validation"] = {"gps_alignment_rmse_m": 3.0}
            self._write_manifest(before)
            # Keep 253 rows but change one source frame so the exact-manifest
            # gate is tested independently from selected-frame count.
            self._write_manifest(after, changed_frame=100)
            (before / "run_report.json").write_text(json.dumps(baseline), encoding="utf-8")
            (after / "run_report.json").write_text(json.dumps(candidate), encoding="utf-8")

            result = compare_output_directories(before, after, root / "comparison.json")
            gates = result["fair_comparison_gates"]

            self.assertTrue(result["passes_quality_guardrails"])
            self.assertFalse(gates["same_video_identity"])
            self.assertFalse(gates["same_video_duration"])
            self.assertFalse(gates["same_target_frame_count"])
            self.assertTrue(gates["same_selected_frame_count"])
            self.assertFalse(gates["same_selected_frame_manifest"])
            self.assertFalse(gates["same_frame_resolution"])
            self.assertFalse(gates["candidate_has_independent_surface_evidence"])
            self.assertFalse(result["candidate_eligible_to_replace_baseline"])

    def test_faster_run_with_worse_checkpoint_accuracy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before", root / "after"
            before.mkdir()
            after.mkdir()
            baseline = self._full_report(26.92 * 60)
            candidate = self._full_report(14.0 * 60)
            candidate_checkpoints = candidate["validation"]["surveyed_checkpoints"]
            candidate_checkpoints.update({
                "rmse_3d_m": 0.9,
                "maximum_3d_error_m": 1.2,
                "passes_one_metre_target": False,
            })
            self._write_manifest(before)
            self._write_manifest(after)
            (before / "run_report.json").write_text(json.dumps(baseline), encoding="utf-8")
            (after / "run_report.json").write_text(json.dumps(candidate), encoding="utf-8")

            result = compare_output_directories(before, after, root / "comparison.json")

            self.assertTrue(result["passes_fair_comparison_gates"])
            self.assertFalse(result["passes_surface_quality_gates"])
            self.assertFalse(
                result["surface_quality_gates"]["candidate_passes_one_metre_surface_target"]
            )
            self.assertFalse(result["candidate_eligible_to_replace_baseline"])

    def test_dense_reference_evidence_locks_reference_and_evaluation_protocol(self):
        mutations = {
            "reference": lambda dense: dense["reference"].update({"sha256": "e" * 64}),
            "copied_reconstruction": lambda dense: dense.update(
                {"reference_is_distinct_from_reconstruction": False}
            ),
            "thresholds": lambda dense: (
                dense["protocol"].update({"thresholds_m": [0.2, 0.5, 1.0]}),
                dense["threshold_metrics"][0].update({"threshold_m": 0.2}),
            ),
            "max_samples": lambda dense: dense.update({"max_samples": 50000}),
            "seed": lambda dense: dense.update({"seed": 7}),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    before, after = root / f"before_{name}", root / f"after_{name}"
                    before.mkdir()
                    after.mkdir()
                    baseline = self._dense_report(26.92 * 60)
                    candidate = self._dense_report(14.0 * 60)
                    mutate(candidate["validation"]["dense_geometry"])
                    self._write_manifest(before)
                    self._write_manifest(after)
                    (before / "run_report.json").write_text(
                        json.dumps(baseline), encoding="utf-8"
                    )
                    (after / "run_report.json").write_text(
                        json.dumps(candidate), encoding="utf-8"
                    )

                    result = compare_output_directories(
                        before, after, root / f"comparison_{name}.json"
                    )
                    self.assertEqual(
                        result["comparison_metadata"]["before"]
                        ["independent_surface_evidence"]["type"],
                        "dense_reference_geometry",
                    )
                    self.assertFalse(
                        result["fair_comparison_gates"]
                        ["same_independent_surface_evidence_set"]
                    )
                    self.assertFalse(result["candidate_eligible_to_replace_baseline"])

    def test_dense_reference_f1_and_directional_errors_must_be_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before", root / "after"
            before.mkdir()
            after.mkdir()
            baseline = self._dense_report(26.92 * 60)
            candidate = self._dense_report(14.0 * 60)
            dense = candidate["validation"]["dense_geometry"]
            dense["threshold_metrics"][0]["f1"] = 0.7
            dense["distances_m"]["reconstruction_to_reference"].update({
                "rmse_m": 0.5,
                "p95_m": 0.7,
            })
            self._write_manifest(before)
            self._write_manifest(after)
            (before / "run_report.json").write_text(json.dumps(baseline), encoding="utf-8")
            (after / "run_report.json").write_text(json.dumps(candidate), encoding="utf-8")

            result = compare_output_directories(before, after, root / "comparison.json")

            self.assertTrue(result["passes_fair_comparison_gates"])
            self.assertFalse(result["passes_surface_quality_gates"])
            self.assertFalse(
                result["surface_quality_gates"]
                ["dense_f1_at_all_thresholds_not_materially_worse"]
            )
            self.assertFalse(
                result["surface_quality_gates"]
                ["reconstruction_to_reference_rmse_not_materially_worse"]
            )
            self.assertFalse(result["candidate_eligible_to_replace_baseline"])


if __name__ == "__main__":
    unittest.main()
