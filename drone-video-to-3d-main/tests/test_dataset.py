import tempfile
import unittest
from pathlib import Path

from sih_drone_pipeline.dataset import (
    _enu_to_geodetic,
    _gps_altitude_is_below_sea_level,
    _gps_decimal,
    _numeric_rows,
    _normalized_pose_times,
    prepare_huggingface_flight_record,
    prepare_whu_dataset,
)


class DatasetTests(unittest.TestCase):
    def test_normalizes_longest_huggingface_video_segment_and_feet(self):
        header = [
            "OSD.flyTime [s]", "OSD.latitude", "OSD.longitude", "OSD.altitude [ft]",
            "OSD.gpsNum", "OSD.yaw [360]", "GIMBAL.pitch", "GIMBAL.roll",
            "GIMBAL.yaw [360]", "CAMERA.isVideo",
        ]
        rows = []
        for index in range(4):
            rows.append([index / 10, 37.0, -122.0, 100, 12, 90, -45, 0, 91, "True"])
        rows.append([0.5, 37.0, -122.0, 100, 12, 90, -45, 0, 91, "False"])
        for index in range(30):
            rows.append([10 + index / 10, 37.1, -122.1, 200, 15, 180, -60, 1, 181, "True"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "FlightRecord.csv"
            import csv
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
            result = prepare_huggingface_flight_record(source, root / "prepared")
            normalized = (root / "prepared" / "hf_drone_telemetry.csv").read_text(encoding="utf-8")
            self.assertEqual(result["telemetry_samples"], 30)
            self.assertEqual(result["selected_segment_policy"], "longest_continuous_CAMERA.isVideo_interval")
            self.assertIn("0.000,37.1000000000,-122.1000000000,60.960", normalized)

    def test_numeric_rows_ignore_official_trailing_empty_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "OnboardGPS.csv"
            path.write_text(
                "Timpstemp,imgid,lat,lon,alt,fix_type,,,,\n"
                "7009129,1,47.3843571,8.5451784,464.91,3,,,,\n",
                encoding="utf-8",
            )

            rows = _numeric_rows(path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], 1.0)
            self.assertAlmostEqual(rows[0][2], 47.3843571)

    def test_exif_dms_coordinates_are_converted_with_hemisphere(self):
        self.assertAlmostEqual(_gps_decimal((39, 45, 30), "N"), 39.7583333333)
        self.assertAlmostEqual(_gps_decimal((104, 59, 15), "W"), -104.9875)

    def test_parrot_byte_altitude_reference_is_decoded(self):
        self.assertFalse(_gps_altitude_is_below_sea_level(b"\x00"))
        self.assertTrue(_gps_altitude_is_below_sea_level(b"\x01"))

    def test_whu_pose_timestamps_are_normalized_from_microseconds(self):
        times, scale = _normalized_pose_times([5_000_000, 6_000_000, 7_000_000], 2.0)
        self.assertEqual(times, [0.0, 1.0, 2.0])
        self.assertEqual(scale, 1e-6)

    def test_enu_to_geodetic_preserves_origin(self):
        latitude, longitude, altitude = _enu_to_geodetic(0, 0, 0, 30.5, 114.3, 42.0)
        self.assertAlmostEqual(latitude, 30.5, places=7)
        self.assertAlmostEqual(longitude, 114.3, places=7)
        self.assertAlmostEqual(altitude, 42.0, places=4)

    def test_prepares_standard_whu_files_from_extracted_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "regular"
            sequence.mkdir()
            (sequence / "flight.mp4").write_bytes(b"")
            (sequence / "ground_truth.txt").write_text(
                "1000000 500 600 10 0 0 0 1\n"
                "2000000 501 600 10 0 0 0 1\n"
                "3000000 502 600 10 0 0 0 1\n",
                encoding="utf-8",
            )
            (sequence / "keyframe_GCP_obsevations.txt").write_text("# fixture\n", encoding="utf-8")
            (root / "points.txt").write_text("1 510 620 11\n2 520 630 12\n", encoding="utf-8")
            result = prepare_whu_dataset(root, root / "prepared", "regular", 30.5, 114.3, 40)
            telemetry = Path(result["telemetry"]).read_text(encoding="utf-8")
            checkpoints = Path(result["checkpoint_measurements_template"]).read_text(encoding="utf-8")
            self.assertIn("time_s,latitude,longitude,altitude_m", telemetry)
            self.assertIn("1,10.0,20.0,1.0", checkpoints)
            self.assertEqual(result["pose_origin_removed"], [500.0, 600.0, 10.0])


if __name__ == "__main__":
    unittest.main()
