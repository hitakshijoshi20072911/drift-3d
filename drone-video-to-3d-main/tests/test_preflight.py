import unittest

from sih_drone_pipeline.preflight import telemetry_diagnostics, video_stream_diagnostics
from sih_drone_pipeline.telemetry import TelemetrySample


class TelemetryPreflightTests(unittest.TestCase):
    def test_regular_telemetry_passes(self):
        samples = [
            TelemetrySample(float(index), 12.0, 77.0 + index * 0.00001, 100.0)
            for index in range(11)
        ]
        result = telemetry_diagnostics(samples, video_duration_s=10.0)
        self.assertTrue(result["critical_pass"])
        self.assertEqual(result["altitude_datum_status"], "unknown")

    def test_duplicate_timestamp_and_large_gap_fail(self):
        samples = [
            TelemetrySample(0.0, 12.0, 77.0, 100.0),
            TelemetrySample(0.0, 12.0, 77.00001, 100.0),
            TelemetrySample(10.0, 12.0, 77.00002, 100.0),
        ]
        result = telemetry_diagnostics(samples, video_duration_s=10.0)
        self.assertFalse(result["critical_pass"])
        self.assertFalse(result["checks"]["timestamps_strictly_increasing"])
        self.assertFalse(result["checks"]["maximum_gap_at_most_5s"])

    def test_implausible_position_jump_fails(self):
        samples = [
            TelemetrySample(0.0, 12.0, 77.0, 100.0),
            TelemetrySample(1.0, 13.0, 78.0, 100.0),
            TelemetrySample(2.0, 13.00001, 78.00001, 100.0),
        ]
        result = telemetry_diagnostics(samples, video_duration_s=2.0)
        self.assertFalse(result["checks"]["horizontal_speed_below_80mps"])


class VideoStreamPreflightTests(unittest.TestCase):
    def test_constant_rate_stream_passes(self):
        result = video_stream_diagnostics(
            {
                "width": 3840,
                "height": 2160,
                "r_frame_rate": "30000/1001",
                "avg_frame_rate": "30000/1001",
                "nb_frames": "300",
                "codec_name": "hevc",
                "pix_fmt": "yuv420p",
            },
            {"duration": "10.01"},
        )
        self.assertTrue(result["critical_pass"])
        self.assertFalse(result["variable_frame_rate"])

    def test_variable_rate_and_rotation_are_explicit(self):
        result = video_stream_diagnostics(
            {
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "60/1",
                "avg_frame_rate": "30000/1001",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "tags": {"rotate": "90"},
            },
            {"duration": "5"},
        )
        self.assertTrue(result["variable_frame_rate"])
        self.assertEqual(result["rotation_degrees"], 90)
        self.assertTrue(result["rotation_must_be_applied"])


if __name__ == "__main__":
    unittest.main()
