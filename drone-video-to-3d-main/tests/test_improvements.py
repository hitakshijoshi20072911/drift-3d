import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from sih_drone_pipeline.benchmark import build_benchmark
from sih_drone_pipeline.camera import (
    calibration_for_image_size, colmap_image_reader_options, load_camera_calibration,
)
from sih_drone_pipeline.control_points import apply_similarity, fit_control_point_transform
from sih_drone_pipeline.profiles import apply_processing_profile, resolve_adaptive_dense_settings
from sih_drone_pipeline.telemetry import interpolate, parse_csv


class CameraCalibrationTests(unittest.TestCase):
    def test_loads_valid_opencv_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            path.write_text(json.dumps({
                "model": "OPENCV",
                "params": [1000, 1001, 960, 540, 0.1, -0.1, 0, 0],
                "width": 1920,
                "height": 1080,
                "fixed_intrinsics": True,
            }), encoding="utf-8")
            profile = load_camera_calibration(path)
            options = colmap_image_reader_options(profile)
            self.assertEqual(profile["model"], "OPENCV")
            self.assertIn("--ImageReader.camera_params", options)
            self.assertTrue(profile["fixed_intrinsics"])
            scaled = calibration_for_image_size(profile, 960, 540)
            self.assertAlmostEqual(scaled["params"][0], 500)
            self.assertAlmostEqual(scaled["params"][2], 480)

    def test_rejects_wrong_parameter_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            path.write_text('{"model":"OPENCV","params":[1,2]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires 8"):
                load_camera_calibration(path)


class ControlPointTests(unittest.TestCase):
    def test_robust_similarity_rejects_outlier(self):
        source = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        target = 1.2 * source @ rotation + np.array([10, 20, 3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gcp.csv"
            rows = ["control_id,reconstructed_x_m,reconstructed_y_m,reconstructed_z_m,known_x_m,known_y_m,known_z_m"]
            for index, (left, right) in enumerate(zip(source, target)):
                rows.append(f"p{index},{left[0]},{left[1]},{left[2]},{right[0]},{right[1]},{right[2]}")
            rows.append("bad,5,5,5,100,100,100")
            path.write_text("\n".join(rows), encoding="utf-8")
            transform, report = fit_control_point_transform(path, inlier_threshold_m=0.05)
            np.testing.assert_allclose(apply_similarity(source, transform), target, atol=1e-8)
            self.assertEqual(report["inliers"], 4)
            self.assertEqual(report["outlier_ids"], ["bad"])


class TelemetryUncertaintyTests(unittest.TestCase):
    def test_parses_and_interpolates_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.csv"
            path.write_text(
                "time_s,latitude,longitude,altitude_m,h_accuracy_m,v_accuracy_m,fix_type\n"
                "0,1,2,3,0.1,0.2,RTK_FIXED\n"
                "1,2,3,4,0.3,0.6,RTK_FLOAT\n"
                "2,3,4,5,0.5,1.0,GPS\n",
                encoding="utf-8",
            )
            samples = parse_csv(path)
            middle = interpolate(samples, 0.5)
            self.assertAlmostEqual(middle.horizontal_accuracy_m, 0.2)
            self.assertAlmostEqual(middle.vertical_accuracy_m, 0.4)
            self.assertEqual(middle.position_source, "RTK_FLOAT")


class ProfileAndBenchmarkTests(unittest.TestCase):
    def test_explicit_command_line_option_overrides_profile(self):
        args = Namespace(profile="deadline-adaptive", spatial_matching=True,
                         explicit_options={"spatial_matching"})
        report = apply_processing_profile(args)
        self.assertTrue(args.spatial_matching)
        self.assertNotIn("spatial_matching", report["applied"])
        self.assertEqual(report["overridden_by_command_line"]["spatial_matching"],
                         {"profile": False, "used": True})
        self.assertEqual(args.sift_max_num_features, 4096)  # the rest of the profile still applies

    def test_typed_options_are_detected_in_both_flag_forms(self):
        from sih_drone_pipeline.cli import _explicit_options, build_parser

        argv = ["run", "--video", "v.mp4", "--telemetry", "t.csv", "--output", "out",
                "--profile", "deadline-adaptive", "--spatial-matching", "--spatial-max-distance-m=150"]
        parser = build_parser()
        args = parser.parse_args(argv)
        explicit = _explicit_options(parser, args, argv)
        self.assertIn("spatial_matching", explicit)
        self.assertIn("spatial_max_distance_m", explicit)
        self.assertNotIn("sift_max_num_features", explicit)
        args.explicit_options = explicit
        apply_processing_profile(args)
        self.assertTrue(args.spatial_matching)
        self.assertEqual(args.spatial_max_distance_m, 150.0)

    def test_deadline_preview_profile_has_bounded_workload_and_sparse_gate(self):
        args = Namespace(profile="deadline-preview")
        report = apply_processing_profile(args)
        self.assertEqual(report["name"], "deadline-preview")
        self.assertEqual(args.sift_max_num_features, 4096)
        self.assertEqual(args.dense_target_references, 72)
        self.assertEqual(args.dense_source_images, 8)
        self.assertEqual(args.dense_max_image_size, 896)
        self.assertEqual(args.sparse_min_registration_ratio, 0.95)
        self.assertEqual(args.sparse_max_reprojection_error_px, 0.75)
        self.assertEqual(args.sparse_max_gps_alignment_rmse_m, 2.0)
        self.assertFalse(args.ai_mask_dynamic)

    def test_deadline_reuse_profile_preserves_validated_frame_network(self):
        args = Namespace(profile="deadline-reuse14")
        report = apply_processing_profile(args)
        self.assertEqual(report["name"], "deadline-reuse14")
        self.assertTrue(args.reuse_keyframes)
        self.assertEqual(args.keyframe_decode_mode, "sequential")
        self.assertEqual(args.sift_max_num_features, 4096)
        self.assertEqual(args.dense_target_references, 64)
        self.assertEqual(args.dense_source_images, 6)
        self.assertEqual(args.dense_max_image_size, 896)

    def test_deadline_dynamic_profile_scales_dense_references(self):
        args = Namespace(profile="deadline-dynamic")
        report = apply_processing_profile(args)
        short = resolve_adaptive_dense_settings(args, 56)
        self.assertEqual(report["name"], "deadline-dynamic")
        self.assertTrue(args.reuse_keyframes)
        self.assertTrue(short["automatic"])
        self.assertEqual(args.dense_target_references, 20)

        args = Namespace(profile="deadline-dynamic")
        apply_processing_profile(args)
        medium = resolve_adaptive_dense_settings(args, 140)
        self.assertEqual(medium["dense_target_references"], 49)

        args = Namespace(profile="deadline-dynamic")
        apply_processing_profile(args)
        large = resolve_adaptive_dense_settings(args, 300)
        self.assertEqual(large["dense_target_references"], 64)

    def test_fast_profile_matches_accepted_26_92_minute_configuration(self):
        args = Namespace(profile="verified-fast", dense_target_references=None)
        report = apply_processing_profile(args)
        resolved = resolve_adaptive_dense_settings(args, 253)
        self.assertEqual(report["name"], "verified-fast")
        self.assertIsNone(resolved["dense_target_references"])
        self.assertEqual(resolved["reference_policy"], "adaptive_stride")
        self.assertEqual(args.dense_frame_stride, 2)
        self.assertEqual(args.dense_source_images, 10)
        self.assertEqual(args.dense_max_image_size, 1088)
        self.assertEqual(args.dense_num_iterations, 4)
        self.assertEqual(args.dense_num_samples, 15)
        self.assertEqual(args.dense_window_step, 1)
        self.assertEqual(args.delaunay_max_proj_dist, 20.0)
        self.assertEqual(args.delaunay_num_threads, -1)
        self.assertEqual(args.mapper, "pose-prior")
        self.assertFalse(args.mapper_single_model)
        self.assertTrue(args.mapper_no_extract_colors)
        self.assertEqual(args.mapper_ba_global_frames_ratio, 1.4)
        self.assertEqual(args.mapper_ba_global_points_ratio, 1.4)
        self.assertEqual(args.mapper_ba_global_max_refinements, 2)
        self.assertTrue(args.mapper_ba_global_ignore_redundant_points3d)

    def test_adaptive_profile_differs_from_verified_fast_only_in_keyframe_selection(self):
        from sih_drone_pipeline.profiles import PROFILES
        fast = dict(PROFILES["verified-fast"])
        adaptive = dict(PROFILES["verified-fast-adaptive"])
        self.assertEqual(fast.pop("keyframe_mode"), "geometry")
        fast.pop("keyframe_decode_mode")  # irrelevant to the adaptive selector
        self.assertEqual(adaptive.pop("keyframe_mode"), "adaptive")
        self.assertEqual(fast, adaptive)
        args = Namespace(profile="verified-fast-adaptive", dense_target_references=None)
        apply_processing_profile(args)
        resolved = resolve_adaptive_dense_settings(args, 120)
        self.assertEqual(resolved["reference_policy"], "adaptive_stride")

    def test_benchmark_selects_fastest_production_ready_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for label, runtime in (("slow", 800), ("fast", 500)):
                run = root / label
                run.mkdir()
                (run / "run_report.json").write_text(json.dumps({
                    "wall_clock_seconds": runtime,
                    "comparison_metadata": {
                        "video_sha256": "a" * 64,
                        "video_duration_s": 251.52,
                        "target_frames": 253,
                        "selected_frames": 253,
                        "frame_manifest_sha256": "b" * 64,
                        "frame_resolution": [1920, 1080],
                    },
                    "validation": {"surveyed_checkpoints": {
                        "checks": 3,
                        "checkpoint_ids": ["CP1", "CP2", "CP3"],
                        "rmse_3d_m": 0.4,
                        "maximum_3d_error_m": 0.6,
                        "passes_one_metre_target": True,
                    }},
                    "targets": {"processing_under_15_minutes": True, "one_metre_surface_accuracy": True},
                }), encoding="utf-8")
                (run / "verification_report.json").write_text(
                    json.dumps({"artifact_checks_pass": True, "production_ready": True}), encoding="utf-8"
                )
                entries.append((label, run))
            result = build_benchmark(entries, root / "benchmark.json")
            self.assertEqual(result["winner"], "fast")
            self.assertEqual(result["reference_run"], "slow")
            self.assertTrue(all(row["eligible"] for row in result["runs"]))
            self.assertTrue((root / "benchmark.csv").is_file())
            self.assertTrue((root / "benchmark.md").is_file())

    def test_benchmark_protects_full_253_frame_baseline_from_shortcuts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_run(label, runtime, *, frames=253, evidence=True):
                run = root / label
                run.mkdir()
                report = {
                    "wall_clock_seconds": runtime,
                    "comparison_metadata": {
                        "video_sha256": "a" * 64,
                        "video_duration_s": 251.52,
                        "target_frames": frames,
                        "selected_frames": frames,
                        "frame_manifest_sha256": ("b" if frames == 253 else "c") * 64,
                        "frame_resolution": [1920, 1080],
                    },
                    "targets": {
                        "processing_under_15_minutes": runtime < 900,
                        "one_metre_surface_accuracy": evidence,
                    },
                }
                if evidence:
                    report["validation"] = {"surveyed_checkpoints": {
                        "checks": 3,
                        "checkpoint_ids": ["CP1", "CP2", "CP3"],
                        "rmse_3d_m": 0.4,
                        "maximum_3d_error_m": 0.6,
                        "passes_one_metre_target": True,
                    }}
                (run / "run_report.json").write_text(json.dumps(report), encoding="utf-8")
                (run / "verification_report.json").write_text(json.dumps({
                    "artifact_checks_pass": True,
                    "production_ready": True,
                }), encoding="utf-8")
                return run

            entries = [
                ("accepted_full_253", write_run("accepted_full_253", 26.92 * 60)),
                ("reduced_66", write_run("reduced_66", 10.3 * 60, frames=66)),
                ("no_surface_evidence", write_run(
                    "no_surface_evidence", 12.0 * 60, evidence=False
                )),
            ]
            result = build_benchmark(entries, root / "benchmark.json")

            self.assertEqual(result["winner"], "accepted_full_253")
            self.assertTrue(result["reference_matches_fixed_253_frame_workload"])
            reduced = next(row for row in result["runs"] if row["label"] == "reduced_66")
            unverified = next(
                row for row in result["runs"] if row["label"] == "no_surface_evidence"
            )
            self.assertFalse(reduced["eligible"])
            self.assertIn("same_target_frame_count", reduced["exclusion_reasons"])
            self.assertIn("same_selected_frame_count", reduced["exclusion_reasons"])
            self.assertFalse(unverified["eligible"])
            self.assertIn(
                "candidate_has_independent_surface_evidence",
                unverified["exclusion_reasons"],
            )


if __name__ == "__main__":
    unittest.main()
