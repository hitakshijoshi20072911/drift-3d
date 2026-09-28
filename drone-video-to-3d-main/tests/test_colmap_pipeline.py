import tempfile
import unittest
from pathlib import Path

from sih_drone_pipeline.colmap_pipeline import _subsample_patch_match_references


class DenseReferenceSelectionTests(unittest.TestCase):
    @staticmethod
    def _write_config(dense: Path, count: int = 5) -> Path:
        stereo = dense / "stereo"
        stereo.mkdir(parents=True)
        config = stereo / "patch-match.cfg"
        config.write_text(
            "".join(f"frame_{index}.jpg\n__auto__, 20\n" for index in range(count)),
            encoding="utf-8",
        )
        return config

    def test_keeps_every_nth_reference_and_final_coverage_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory)
            config = self._write_config(dense)
            report = _subsample_patch_match_references(dense, 2)
            self.assertEqual(report["original_references"], 5)
            self.assertEqual(report["selected_references"], 3)
            self.assertEqual(
                config.read_text(encoding="utf-8").splitlines()[::2],
                ["frame_0.jpg", "frame_2.jpg", "frame_4.jpg"],
            )

    def test_adaptive_selection_keeps_temporal_coverage_and_limits_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory) / "dense"
            config = self._write_config(dense, count=6)
            manifest = Path(directory) / "frames.csv"
            manifest.write_text(
                "image_name,sharpness,geometry_score\n"
                "frame_0.jpg,10,0.1\n"
                "frame_1.jpg,10,0.9\n"
                "frame_2.jpg,10,0.2\n"
                "frame_3.jpg,10,0.8\n"
                "frame_4.jpg,10,0.7\n"
                "frame_5.jpg,10,0.3\n",
                encoding="utf-8",
            )
            report = _subsample_patch_match_references(
                dense,
                2,
                selection_mode="adaptive",
                frame_manifest=manifest,
                max_source_images=10,
            )
            lines = config.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[::2], ["frame_1.jpg", "frame_3.jpg", "frame_4.jpg"])
            self.assertEqual(lines[1::2], ["__auto__, 10"] * 3)
            self.assertEqual(report["selected_references"], 3)
            self.assertEqual(report["selection_mode"], "adaptive")

    def test_adaptive_exact_target_has_exact_size_and_temporal_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory) / "dense"
            config = self._write_config(dense, count=10)
            manifest = Path(directory) / "frames.csv"
            manifest.write_text(
                "image_name,sharpness,geometry_score\n"
                + "".join(
                    f"frame_{index}.jpg,10,{(index + 1) / 10:.1f}\n"
                    for index in range(10)
                ),
                encoding="utf-8",
            )
            report = _subsample_patch_match_references(
                dense,
                1,
                selection_mode="adaptive",
                frame_manifest=manifest,
                max_source_images=8,
                target_references=4,
            )
            lines = config.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[::2], [
                "frame_1.jpg", "frame_4.jpg", "frame_6.jpg", "frame_9.jpg",
            ])
            self.assertEqual(lines[1::2], ["__auto__, 8"] * 4)
            self.assertEqual(report["target_references"], 4)
            self.assertEqual(report["selected_references"], 4)

    def test_uniform_exact_target_includes_temporal_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory)
            config = self._write_config(dense, count=10)
            report = _subsample_patch_match_references(
                dense, 1, target_references=4
            )
            self.assertEqual(
                config.read_text(encoding="utf-8").splitlines()[::2],
                ["frame_0.jpg", "frame_3.jpg", "frame_6.jpg", "frame_9.jpg"],
            )
            self.assertEqual(report["selected_references"], 4)

    def test_exact_target_clamps_to_available_references(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory)
            self._write_config(dense, count=5)
            report = _subsample_patch_match_references(
                dense, 1, target_references=20
            )
            self.assertEqual(report["target_references"], 20)
            self.assertEqual(report["selected_references"], 5)

    @staticmethod
    def _write_nadir_model(model: Path, centres_x: list[float]) -> None:
        """Downward-looking cameras 100 m above flat ground at the given east offsets."""
        model.mkdir(parents=True)
        (model / "cameras.txt").write_text("1 PINHOLE 1000 1000 500 500 500 500\n", encoding="utf-8")
        # 180 degrees about x: camera x = east, y = south, z = down.
        (model / "images.txt").write_text(
            "".join(
                f"{index + 1} 0 1 0 0 {-x} 0 100 1 frame_{index}.jpg\n\n"
                for index, x in enumerate(centres_x)
            ),
            encoding="utf-8",
        )
        (model / "points3D.txt").write_text(
            "".join(f"{index} {index} 0 -100 0 0 0 0.5\n" for index in range(20)),
            encoding="utf-8",
        )

    def test_coverage_selection_reaches_ground_the_time_slices_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory) / "dense"
            config = self._write_config(dense, count=10)
            # Seven frames hover over one spot, then three frames each see new ground.
            model = Path(directory) / "aligned_text"
            self._write_nadir_model(model, [0.0] * 7 + [500.0, 1000.0, 1500.0])
            manifest = Path(directory) / "frames.csv"
            manifest.write_text(
                "image_name,sharpness,geometry_score\n"
                + "".join(f"frame_{index}.jpg,10,0.5\n" for index in range(10)),
                encoding="utf-8",
            )
            report = _subsample_patch_match_references(
                dense, 1, selection_mode="coverage", frame_manifest=manifest,
                target_references=4, model_dir=model, coverage_min_views=1,
            )
            chosen = config.read_text(encoding="utf-8").splitlines()[::2]
            self.assertEqual(report["selection_mode_used"], "coverage")
            self.assertEqual(len(chosen), 4)
            self.assertTrue({"frame_7.jpg", "frame_8.jpg", "frame_9.jpg"} <= set(chosen))
            self.assertEqual(report["coverage"]["cells_meeting_min_views_percent"], 100.0)

    def test_coverage_selection_falls_back_to_time_slices_without_poses(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory) / "dense"
            self._write_config(dense, count=10)
            manifest = Path(directory) / "frames.csv"
            manifest.write_text(
                "image_name,sharpness,geometry_score\n"
                + "".join(f"frame_{index}.jpg,10,0.5\n" for index in range(10)),
                encoding="utf-8",
            )
            report = _subsample_patch_match_references(
                dense, 1, selection_mode="coverage", frame_manifest=manifest,
                target_references=4, model_dir=Path(directory) / "missing",
            )
            self.assertEqual(report["selection_mode_used"], "adaptive")
            self.assertEqual(report["selected_references"], 4)
            self.assertEqual(report["coverage"]["fallback"], "adaptive")

    def test_exact_target_rejects_ambiguous_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = Path(directory)
            self._write_config(dense, count=5)
            with self.assertRaisesRegex(ValueError, "requires stride=1"):
                _subsample_patch_match_references(
                    dense, 2, target_references=4
                )


if __name__ == "__main__":
    unittest.main()
