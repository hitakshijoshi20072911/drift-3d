import unittest

from sih_drone_pipeline.cli import _independent_surface_target, build_parser
from sih_drone_pipeline.colmap_pipeline import _feature_configuration


class FrontendSparseCliTests(unittest.TestCase):
    def test_accepts_precomputed_video_hash_and_capture_quality_reuse(self):
        digest = "a" * 64
        args = build_parser().parse_args([
            "run",
            "--video", "flight.mp4",
            "--telemetry", "telemetry.csv",
            "--output", "output",
            "--video-sha256", digest,
            "--reuse-capture-quality",
        ])
        self.assertEqual(args.video_sha256, digest)
        self.assertTrue(args.reuse_capture_quality)

    def test_adaptive_screening_timeout_defaults_above_library_limit(self):
        base = ["run", "--video", "flight.mp4", "--telemetry", "telemetry.csv", "--output", "output"]
        self.assertEqual(build_parser().parse_args(base).adaptive_screening_timeout_s, 900.0)
        args = build_parser().parse_args(base + ["--adaptive-screening-timeout-s", "1200"])
        self.assertEqual(args.adaptive_screening_timeout_s, 1200.0)

    def test_accepts_existing_sparse_model_reuse(self):
        args = build_parser().parse_args([
            "run",
            "--video", "flight.mp4",
            "--telemetry", "telemetry.csv",
            "--output", "output",
            "--reuse-sparse-model", "workspace/sparse/2",
        ])
        self.assertEqual(args.reuse_sparse_model, "workspace/sparse/2")

    def test_accepts_sequential_global_sparse_screen(self):
        args = build_parser().parse_args([
            "run",
            "--video", "flight.mp4",
            "--telemetry", "telemetry.csv",
            "--output", "output",
            "--keyframe-mode", "geometry",
            "--keyframe-decode-mode", "sequential",
            "--mapper", "global",
            "--mapper-ba-gpu",
            "--sparse-only",
        ])
        self.assertEqual(args.keyframe_decode_mode, "sequential")
        self.assertEqual(args.mapper, "global")
        self.assertTrue(args.mapper_ba_gpu)
        self.assertTrue(args.sparse_only)

    def test_accepts_incremental_mapper_ba_tuning(self):
        args = build_parser().parse_args([
            "run",
            "--video", "flight.mp4",
            "--telemetry", "telemetry.csv",
            "--output", "output",
            "--mapper", "pose-prior",
            "--mapper-single-model",
            "--mapper-no-extract-colors",
            "--mapper-ba-global-frames-ratio", "1.4",
            "--mapper-ba-global-points-ratio", "1.4",
            "--mapper-ba-global-max-refinements", "2",
            "--mapper-ba-global-ignore-redundant-points3d",
            "--mapper-ba-local-max-num-iterations", "20",
            "--mapper-ba-global-max-num-iterations", "35",
            "--mapper-random-seed", "7",
        ])
        self.assertTrue(args.mapper_single_model)
        self.assertTrue(args.mapper_no_extract_colors)
        self.assertEqual(args.mapper_ba_global_frames_ratio, 1.4)
        self.assertEqual(args.mapper_ba_global_points_ratio, 1.4)
        self.assertEqual(args.mapper_ba_global_max_refinements, 2)
        self.assertTrue(args.mapper_ba_global_ignore_redundant_points3d)
        self.assertEqual(args.mapper_ba_local_max_num_iterations, 20)
        self.assertEqual(args.mapper_ba_global_max_num_iterations, 35)
        self.assertEqual(args.mapper_random_seed, 7)

    def test_accepts_dense_and_delaunay_ab_controls(self):
        args = build_parser().parse_args([
            "run",
            "--video", "flight.mp4",
            "--telemetry", "telemetry.csv",
            "--output", "output",
            "--dense-target-references", "92",
            "--dense-source-images", "8",
            "--dense-num-iterations", "3",
            "--dense-num-samples", "12",
            "--dense-window-step", "2",
            "--delaunay-max-proj-dist", "24",
            "--delaunay-max-depth-dist", "0.05",
            "--delaunay-num-threads", "-1",
        ])
        self.assertEqual(args.dense_target_references, 92)
        self.assertEqual(args.dense_source_images, 8)
        self.assertEqual(args.dense_num_iterations, 3)
        self.assertEqual(args.dense_num_samples, 12)
        self.assertEqual(args.dense_window_step, 2)
        self.assertEqual(args.delaunay_max_proj_dist, 24.0)
        self.assertEqual(args.delaunay_max_depth_dist, 0.05)
        self.assertEqual(args.delaunay_num_threads, -1)

    def test_accepts_deadline_preview_frontend_and_sparse_gate(self):
        args = build_parser().parse_args([
            "run",
            "--video", "flight.mp4",
            "--telemetry", "telemetry.csv",
            "--output", "output",
            "--profile", "deadline-preview",
            "--sift-max-num-features", "4096",
            "--sparse-min-registration-ratio", "0.95",
            "--sparse-max-reprojection-error-px", "0.75",
            "--sparse-max-gps-alignment-rmse-m", "2.0",
        ])
        self.assertEqual(args.profile, "deadline-preview")
        self.assertEqual(args.sift_max_num_features, 4096)
        self.assertEqual(args.sparse_min_registration_ratio, 0.95)
        self.assertEqual(args.sparse_max_reprojection_error_px, 0.75)
        self.assertEqual(args.sparse_max_gps_alignment_rmse_m, 2.0)

    def test_accepts_dense_geometry_validation_controls(self):
        args = build_parser().parse_args([
            "validate-geometry",
            "--output", "output",
            "--reference", "truth.ply",
            "--reconstruction", "candidate.ply",
            "--threshold-m", "0.1",
            "--threshold-m", "0.5",
            "--max-samples", "25000",
            "--seed", "7",
        ])
        self.assertEqual(args.reference, "truth.ply")
        self.assertEqual(args.reconstruction, "candidate.ply")
        self.assertEqual(args.threshold_m, [0.1, 0.5])
        self.assertEqual(args.max_samples, 25_000)
        self.assertEqual(args.seed, 7)

    def test_accepts_learned_frontend_and_advancing_front_mesher(self):
        args = build_parser().parse_args([
            "run",
            "--video", "flight.mp4",
            "--telemetry", "telemetry.csv",
            "--output", "output",
            "--feature-type", "aliked-n16rot",
            "--feature-matcher", "lightglue",
            "--mesher", "advancing-front",
        ])
        self.assertEqual(args.feature_type, "aliked-n16rot")
        self.assertEqual(args.feature_matcher, "lightglue")
        self.assertEqual(args.mesher, "advancing-front")

    def test_resolves_compatible_colmap_frontend_enums(self):
        self.assertEqual(
            _feature_configuration("sift", "lightglue"),
            ("SIFT", "SIFT_LIGHTGLUE"),
        )
        self.assertEqual(
            _feature_configuration("aliked-n32", "bruteforce"),
            ("ALIKED_N32", "ALIKED_BRUTEFORCE"),
        )
        with self.assertRaises(ValueError):
            _feature_configuration("unknown", "bruteforce")

    def test_surface_target_combines_all_independent_evidence(self):
        validation = {
            "surveyed_checkpoints": {"passes_one_metre_target": True},
            "dense_geometry": {
                "passes_one_metre_target": False,
                "reference_is_distinct_from_reconstruction": True,
            },
        }
        passed, evidence = _independent_surface_target(validation)
        self.assertFalse(passed)
        self.assertIn("checkpoints", evidence)
        self.assertIn("reference geometry", evidence)

        validation["dense_geometry"]["reference_is_distinct_from_reconstruction"] = False
        passed, evidence = _independent_surface_target(validation)
        self.assertTrue(passed)
        self.assertNotIn("reference geometry", evidence)


if __name__ == "__main__":
    unittest.main()
