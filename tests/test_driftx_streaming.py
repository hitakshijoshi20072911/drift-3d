"""CPU-only tests for DRIFTX's streaming orchestration helpers."""

import unittest
import importlib.util
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from driftx.benchmark.runner import PROFILES, _resolve_options
from driftx.benchmark.streaming import (
    _apply_sim3_to_prediction,
    align_overlap,
    iter_chunks,
    oom_fallback_size,
)
from driftx.export.baseline import _depth_to_world_points


class StreamingTests(unittest.TestCase):
    def test_frame_metadata_uses_portable_relative_paths(self):
        from driftx.benchmark.runner import _write_frame_metadata

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark"
            output.mkdir()
            info = SimpleNamespace(source_frame_ids=(7,), source_fps=30.0)
            prediction = SimpleNamespace(
                extrinsics=np.eye(4, dtype=np.float32)[None],
                intrinsics=np.eye(3, dtype=np.float32)[None],
            )
            _write_frame_metadata(
                output, info, [str(output / "input_images" / "frame_000000007.png")], prediction
            )
            import json
            row = json.loads((output / "frames.json").read_text())[0]
            self.assertEqual(row["image"], "input_images/frame_000000007.png")

    def test_all_extra_does_not_force_native_gsplat_build(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        all_extra = text.split("all = [", 1)[1].split("]", 1)[0]
        self.assertIn("gradio>=5", all_extra)
        self.assertNotIn("gsplat", all_extra)

    def test_runtime_dependencies_include_imported_addict_package(self):
        project = Path("pyproject.toml").read_text(encoding="utf-8")
        requirements = Path("requirements.txt").read_text(encoding="utf-8")
        self.assertIn('"addict>=2.4.0"', project)
        self.assertIn("addict>=2.4.0", requirements)

    def test_chunks_cover_every_frame_and_overlap(self):
        chunks = list(iter_chunks(65, 16, 4))
        covered = []
        for chunk in chunks:
            covered.extend(range(chunk.start, chunk.end))
        self.assertEqual(set(covered), set(range(65)))
        self.assertEqual(chunks[0].start, 0)
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left.end - right.start, 4)
            self.assertLess(right.start, left.end)

    def test_single_short_chunk_has_no_duplicate_frames(self):
        chunks = list(iter_chunks(5, 8, 2))
        self.assertEqual([(c.start, c.end) for c in chunks], [(0, 5)])

    def test_weighted_overlap_sim3_recovers_transform(self):
        h = w = 32
        frames = 3
        depth = np.full((frames, h, w), 3.0, dtype=np.float32)
        confidence = np.ones_like(depth)
        k = np.array([[24.0, 0, 15.5], [0, 24.0, 15.5], [0, 0, 1]], dtype=np.float32)
        intrinsics = np.repeat(k[None], frames, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
        previous = SimpleNamespace(
            depth=depth.copy(), conf=confidence.copy(), intrinsics=intrinsics.copy(),
            extrinsics=extrinsics.copy(),
        )
        current = SimpleNamespace(
            depth=depth.copy(), conf=confidence.copy(), intrinsics=intrinsics.copy(),
            extrinsics=extrinsics.copy(),
        )
        angle = 0.17
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        scale = 1.25
        translation = np.array([0.4, -0.2, 0.7])
        inverse_scale = 1 / scale
        inverse_rotation = rotation.T
        inverse_translation = -inverse_scale * (inverse_rotation @ translation)
        _apply_sim3_to_prediction(current, inverse_scale, inverse_rotation, inverse_translation)
        result = align_overlap(previous, current, overlap=frames, stride=2)
        self.assertAlmostEqual(result["scale"], scale, places=4)
        self.assertLess(result["relative_median_error"], 1e-4)
        np.testing.assert_allclose(current.depth, previous.depth, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(current.extrinsics, previous.extrinsics, rtol=1e-4, atol=1e-4)

    def test_oom_fallback_is_monotonic(self):
        self.assertEqual(oom_fallback_size(32), 24)
        self.assertEqual(oom_fallback_size(16), 12)
        self.assertEqual(oom_fallback_size(8), 4)
        self.assertIsNone(oom_fallback_size(4))

    def test_profile_defaults_and_overrides(self):
        self.assertEqual(_resolve_options("smoke", None, None, None, None, None),
                         (1.0, 16, 504, 8, 2))
        self.assertEqual(_resolve_options("balanced", None, 64, 720, 12, 3),
                         (2.0, 64, 720, 12, 3))
        self.assertIsNone(_resolve_options("quality", None, 0, None, None, None)[1])
        self.assertEqual(set(PROFILES), {"smoke", "balanced", "quality"})

    def test_world_point_sampling_respects_global_cap_and_ids(self):
        n, h, w = 4, 10, 12
        depth = np.ones((n, h, w), dtype=np.float32)
        conf = np.ones_like(depth)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0)
        images = np.zeros((n, h, w, 3), dtype=np.uint8)
        points, colors = _depth_to_world_points(
            depth, intrinsics, extrinsics, images, conf, max_points=37
        )
        self.assertEqual(points.shape, (37, 3))
        self.assertEqual(colors.shape, (37, 4))
        self.assertTrue(np.isfinite(points).all())

    @unittest.skipUnless(
        importlib.util.find_spec("trimesh") and importlib.util.find_spec("imageio"),
        "Export smoke test requires trimesh and imageio",
    )
    def test_export_contract_creates_real_artifacts(self):
        from driftx.export.baseline import export_baseline

        modules = {
            "depth_anything_3": types.ModuleType("depth_anything_3"),
            "depth_anything_3.utils": types.ModuleType("depth_anything_3.utils"),
            "depth_anything_3.utils.visualize": types.ModuleType("depth_anything_3.utils.visualize"),
        }
        modules["depth_anything_3"].__path__ = []
        modules["depth_anything_3.utils"].__path__ = []
        def fake_visualize(depth):
            value = np.asarray(depth, dtype=np.float32)
            gray = np.clip(value / max(float(value.max()), 1e-6) * 255, 0, 255).astype(np.uint8)
            return np.repeat(gray[..., None], 3, axis=2)
        modules["depth_anything_3.utils.visualize"].visualize_depth = fake_visualize
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                frames, h, w = 2, 12, 16
                prediction = SimpleNamespace(
                    depth=np.full((frames, h, w), 2.0, dtype=np.float32),
                    conf=np.ones((frames, h, w), dtype=np.float32),
                    intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0),
                    extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0),
                    processed_images=np.full((frames, h, w, 3), 90, dtype=np.uint8),
                    frame_ids=np.array([5, 15]),
                )
                artifacts = export_baseline(prediction, tmp, max_points=100)
                for key in ("glb", "ply", "npz", "legacy_mini_npz"):
                    self.assertTrue(Path(artifacts[key]).is_file(), key)
                self.assertEqual(len(list(Path(artifacts["depth_vis"]).glob("*.jpg"))), frames)
                with np.load(artifacts["npz"]) as data:
                    self.assertEqual(data["depth"].shape, (frames, h, w))
                    self.assertEqual(data["frame_ids"].tolist(), [5, 15])
        finally:
            for name, previous_module in previous.items():
                if previous_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous_module

    def test_mocked_stream_attempt_merges_every_frame_once(self):
        from driftx.benchmark.runner import _run_stream_attempt

        fake_root = types.ModuleType("depth_anything_3")
        fake_specs = types.ModuleType("depth_anything_3.specs")
        class FakePrediction:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        fake_specs.Prediction = FakePrediction
        previous_root = sys.modules.get("depth_anything_3")
        previous_specs = sys.modules.get("depth_anything_3.specs")
        sys.modules["depth_anything_3"] = fake_root
        sys.modules["depth_anything_3.specs"] = fake_specs

        class FakeCuda:
            @staticmethod
            def is_available():
                return False
        fake_torch = SimpleNamespace(cuda=FakeCuda())

        class FakeModel:
            def inference(self, *, image, **kwargs):
                n = len(image)
                h = w = 32
                k = np.array([[24.0, 0, 15.5], [0, 24.0, 15.5], [0, 0, 1]], dtype=np.float32)
                return SimpleNamespace(
                    depth=np.full((n, h, w), 3.0, dtype=np.float32),
                    conf=np.ones((n, h, w), dtype=np.float32),
                    intrinsics=np.repeat(k[None], n, axis=0),
                    extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0),
                    processed_images=np.zeros((n, h, w, 3), dtype=np.uint8),
                    is_metric={"metric": True, "source": "checkpoint"},
                )

        try:
            with tempfile.TemporaryDirectory() as tmp:
                ids = tuple(range(100, 113))
                paths = [f"frame-{i}" for i in ids]
                started = time.perf_counter()
                result, chunks, alignment, _, temp_dir = _run_stream_attempt(
                    FakeModel(), fake_torch, paths, ids, __import__("pathlib").Path(tmp),
                    "cpu", 504, "fp32", 8, 2, 13.0, {}, started,
                )
                self.assertEqual([c.start for c in chunks], [0, 6])
                self.assertEqual(result.depth.shape[0], 13)
                self.assertEqual(result.frame_ids.tolist(), list(ids))
                self.assertEqual(result.is_metric, {"metric": True, "source": "checkpoint"})
                self.assertEqual(len(alignment), 1)
                self.assertEqual(alignment[0]["inliers"], alignment[0]["correspondences"])
                self.assertTrue((temp_dir / "depth.npy").is_file())
                for name in ("depth", "conf", "extrinsics", "intrinsics", "processed_images"):
                    arr = getattr(result, name)
                    getattr(arr, "_mmap", None).close()
        finally:
            if previous_root is None:
                sys.modules.pop("depth_anything_3", None)
            else:
                sys.modules["depth_anything_3"] = previous_root
            if previous_specs is None:
                sys.modules.pop("depth_anything_3.specs", None)
            else:
                sys.modules["depth_anything_3.specs"] = previous_specs

    @unittest.skipUnless(
        importlib.util.find_spec("trimesh") and importlib.util.find_spec("imageio"),
        "Benchmark integration requires trimesh and imageio",
    )
    def test_mocked_full_benchmark_writes_report_and_cleans_staging(self):
        from driftx.benchmark import runner

        fake_root = types.ModuleType("depth_anything_3")
        fake_root.__path__ = []
        fake_specs = types.ModuleType("depth_anything_3.specs")
        class FakePrediction:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        fake_specs.Prediction = FakePrediction
        fake_utils = types.ModuleType("depth_anything_3.utils")
        fake_utils.__path__ = []
        fake_visualize = types.ModuleType("depth_anything_3.utils.visualize")
        fake_visualize.visualize_depth = lambda depth: np.repeat(
            np.zeros((*np.asarray(depth).shape, 1), dtype=np.uint8), 3, axis=2
        )
        fake_modules = {
            "depth_anything_3": fake_root,
            "depth_anything_3.specs": fake_specs,
            "depth_anything_3.utils": fake_utils,
            "depth_anything_3.utils.visualize": fake_visualize,
        }
        previous_modules = {name: sys.modules.get(name) for name in fake_modules}
        sys.modules.update(fake_modules)

        class FakeCuda:
            @staticmethod
            def is_available():
                return False
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = FakeCuda()

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def inference(self, *, image, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("CUDA out of memory. synthetic fallback test")
                n, h, w = len(image), 32, 32
                k = np.array([[24.0, 0, 15.5], [0, 24.0, 15.5], [0, 0, 1]], dtype=np.float32)
                return SimpleNamespace(
                    depth=np.full((n, h, w), 2.5, dtype=np.float32),
                    conf=np.full((n, h, w), 0.9, dtype=np.float32),
                    intrinsics=np.repeat(k[None], n, axis=0),
                    extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0),
                    processed_images=np.full((n, h, w, 3), 120, dtype=np.uint8),
                    is_metric={"metric": True, "source": "checkpoint"},
                )

        info = SimpleNamespace(
            source_frames=5, frame_interval=1, duration_seconds=1.0,
            source_fps=5.0, sampled_fps=1.0, source_frame_ids=(0, 1, 2, 3, 4),
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "benchmark"
                paths = [str(out / "input_images" / f"frame_{i:09d}.png") for i in range(5)]
                fake_cv = lambda video, output, fps, max_frames: (info, paths)
                with patch.dict(sys.modules, {"torch": fake_torch}), \
                     patch("driftx.ingest.extract_video_frames", side_effect=fake_cv), \
                     patch.object(runner, "_load_da3", return_value=FakeModel()), \
                     patch.object(runner, "_install_driftx_runtime_controls"):
                    report = runner.run_benchmark(
                        "synthetic.mp4", str(out), device="cpu", profile="smoke", max_frames=5
                    )
                self.assertEqual(report["status"], "completed", report.get("error"))
                self.assertEqual(report["frames_processed"], 5)
                self.assertEqual(report["source_frames"], 5)
                self.assertEqual(report["final_precision"], "fp32")
                self.assertEqual(report["cuda_oom_retries"], 1)
                self.assertEqual(report["chunk_size"], 4)
                self.assertEqual(report["num_chunks"], 2)
                self.assertTrue((out / "run_report.json").is_file())
                self.assertTrue((out / "metrics.json").is_file())
                self.assertFalse((out / ".driftx_stream_tmp").exists())
                self.assertTrue(Path(report["artifacts"]["glb"]).is_file())
                self.assertEqual(len(list((out / "depth_vis").glob("*.jpg"))), 5)
        finally:
            for name, previous_module in previous_modules.items():
                if previous_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous_module

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") and importlib.util.find_spec("PIL"),
        "Figure integration requires the optional matplotlib and Pillow dependencies",
    )
    def test_trajectory_formats_and_figure_generation(self):
        from driftx.figures import _trajectory_centers, main
        from PIL import Image
        import json
        from pathlib import Path

        expected = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        ext34 = np.repeat(np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)[None], 2, axis=0)
        ext34[:, :3, 3] = -expected
        np.testing.assert_allclose(_trajectory_centers(ext34), expected)
        ext44 = np.repeat(np.eye(4)[None], 2, axis=0)
        ext44[:, :3, 3] = -expected
        np.testing.assert_allclose(_trajectory_centers(ext44), expected)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            images_dir = run / "input_images"
            images_dir.mkdir(parents=True)
            frames, h, w = 4, 12, 16
            for i in range(frames):
                Image.fromarray(np.full((h, w, 3), 100 + i, dtype=np.uint8)).save(
                    images_dir / f"frame_{i:09d}.png"
                )
            npz_path = run / "results.npz"
            np.savez_compressed(
                npz_path,
                depth=np.full((frames, h, w), 2.0, dtype=np.float32),
                conf=np.full((frames, h, w), 0.75, dtype=np.float32),
                extrinsics=ext34.astype(np.float32)[:1].repeat(frames, axis=0),
                intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0),
                frame_ids=np.array([0, 30, 60, 90]),
            )
            report_path = run / "run_report.json"
            report_path.write_text(json.dumps({
                "schema_version": 2,
                "status": "completed",
                "video": "synthetic.mp4",
                "output": str(run),
                "model_variant": "DA3 Large 1.1",
                "device": "cpu",
                "source_fps": 30,
                "source_frames": 120,
                "video_duration_s": 4.0,
                "sample_fps": 1.0,
                "frames_sampled": frames,
                "frames_processed": frames,
                "video_decode_seconds": 0.1,
                "inference_seconds": 1.0,
                "export_seconds": 0.2,
                "total_runtime_seconds": 1.5,
                "peak_gpu_memory_mb": "not measured",
                "mean_confidence": 0.75,
                "median_confidence": 0.75,
                "valid_depth_pixels": frames * h * w,
                "valid_depth_percentage": 100.0,
                "chunk_size": 8,
                "chunk_overlap": 2,
                "num_chunks": 1,
                "cuda_oom_retries": 0,
                "final_precision": "fp32",
                "artifacts": {"npz": str(npz_path)},
            }))
            figure_dir = root / "figures"
            self.assertEqual(main(["--report", str(report_path), "--output", str(figure_dir)]), 0)
            self.assertEqual(len(list(figure_dir.glob("fig*.png"))), 8)
            self.assertEqual(len(list(figure_dir.glob("fig*.svg"))), 8)
            self.assertTrue((figure_dir / "fig06_camera_trajectory.png").is_file())
            self.assertTrue((figure_dir / "baseline_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
