import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import cv2
except ImportError:  # The lightweight local test environment may omit video extras.
    cv2 = None

if cv2 is not None:
    from sih_drone_pipeline.extract_keyframes import extract_keyframes
    from sih_drone_pipeline.telemetry import TelemetrySample


class KeyframeSelectionTests(unittest.TestCase):
    @unittest.skipIf(cv2 is None, "opencv-python is not installed")
    def test_sequential_decode_preserves_seek_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "selection.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
            )
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(26158)
            texture = rng.integers(0, 256, size=(64, 180), dtype=np.uint8)
            for index in range(120):
                crop = texture[:, index // 2:index // 2 + 96]
                writer.write(np.repeat(crop[:, :, None], 3, axis=2))
            writer.release()
            telemetry = [
                TelemetrySample(float(index), 12.0, 77.0 + index * 0.00001, 100.0)
                for index in range(13)
            ]
            seek_manifest = root / "seek.csv"
            sequential_manifest = root / "sequential.csv"
            seek_records = extract_keyframes(
                video, root / "seek_images", target_frames=12, candidates_per_bin=4,
                manifest_path=seek_manifest, selection_mode="geometry",
                telemetry_samples=telemetry, decode_mode="seek",
            )
            sequential_records = extract_keyframes(
                video, root / "sequential_images", target_frames=12, candidates_per_bin=4,
                manifest_path=sequential_manifest, selection_mode="geometry",
                telemetry_samples=telemetry, decode_mode="sequential",
            )
            self.assertEqual(
                [record.source_frame for record in seek_records],
                [record.source_frame for record in sequential_records],
            )
            metadata = json.loads(sequential_manifest.with_suffix(".selection.json").read_text())
            self.assertEqual(metadata["keyframe_decode_mode"], "sequential")
            self.assertEqual(metadata["seek_operations"], 0)
            self.assertGreater(metadata["sequential_grabbed_frames"], 0)
            self.assertEqual(metadata["candidate_frames"], 48)

    @unittest.skipIf(cv2 is None, "opencv-python is not installed")
    def test_duration_adaptive_selection_respects_cap_and_records_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "flight.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for index in range(100):
                frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            manifest = root / "frames.csv"
            records = extract_keyframes(
                video, root / "images", target_frames=None, sample_fps=2.0,
                max_frames=15, candidates_per_bin=1, manifest_path=manifest,
            )
            metadata = json.loads(manifest.with_suffix(".selection.json").read_text())
            self.assertEqual(len(records), 15)
            self.assertEqual(metadata["selection_mode"], "duration_adaptive")
            self.assertEqual(metadata["selected_frames"], 15)

    @unittest.skipIf(cv2 is None, "opencv-python is not installed")
    def test_geometry_selection_records_baseline_and_parallax_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "moving.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
            )
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(26158)
            texture = rng.integers(0, 256, size=(64, 150), dtype=np.uint8)
            for index in range(100):
                crop = texture[:, index // 2:index // 2 + 96]
                writer.write(np.repeat(crop[:, :, None], 3, axis=2))
            writer.release()
            telemetry = [
                TelemetrySample(float(index), 12.0, 77.0 + index * 0.00001, 100.0)
                for index in range(11)
            ]
            manifest = root / "frames.csv"
            records = extract_keyframes(
                video,
                root / "images",
                target_frames=10,
                candidates_per_bin=3,
                manifest_path=manifest,
                selection_mode="geometry",
                telemetry_samples=telemetry,
            )
            metadata = json.loads(manifest.with_suffix(".selection.json").read_text())
            self.assertEqual(len(records), 10)
            self.assertEqual(metadata["keyframe_selection_mode"], "geometry")
            self.assertGreater(metadata["median_selected_baseline_m"], 0)
            self.assertGreaterEqual(metadata["median_selected_parallax_px"], 0)

    @unittest.skipIf(cv2 is None, "opencv-python is not installed")
    def test_validated_keyframe_cache_skips_jpeg_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "cached.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for index in range(100):
                writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
            writer.release()
            manifest = root / "frames.csv"
            kwargs = {
                "target_frames": 10,
                "candidates_per_bin": 1,
                "manifest_path": manifest,
                "decode_mode": "sequential",
            }
            original = extract_keyframes(video, root / "images", **kwargs)
            with patch(
                "sih_drone_pipeline.extract_keyframes.cv2.imwrite",
                side_effect=AssertionError("cache reuse must not rewrite JPEGs"),
            ):
                reused = extract_keyframes(
                    video, root / "images", reuse_existing=True, **kwargs
                )
            self.assertEqual(original, reused)

    @unittest.skipIf(cv2 is None, "opencv-python is not installed")
    def test_invalid_cache_with_extra_image_is_rebuilt_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "cached.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for index in range(100):
                writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
            writer.release()
            images = root / "images"
            manifest = root / "frames.csv"
            kwargs = {
                "target_frames": 10,
                "candidates_per_bin": 1,
                "manifest_path": manifest,
                "decode_mode": "sequential",
            }
            extract_keyframes(video, images, **kwargs)
            (images / "caller_supplied.jpg").write_bytes(
                next(images.glob("*.jpg")).read_bytes()
            )
            rebuilt = extract_keyframes(
                video, images, reuse_existing=True, **kwargs
            )
            self.assertEqual(len(rebuilt), 10)
            self.assertEqual(
                {path.name for path in images.glob("*.jpg")},
                {record.image_name for record in rebuilt},
            )
            self.assertFalse((images / "caller_supplied.jpg").exists())


if __name__ == "__main__":
    unittest.main()
