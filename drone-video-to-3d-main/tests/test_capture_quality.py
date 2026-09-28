import unittest

from sih_drone_pipeline.capture_quality import telemetry_quality
from sih_drone_pipeline.telemetry import TelemetrySample


def sample(time_s, latitude, longitude):
    return TelemetrySample(
        time_s=time_s,
        latitude=latitude,
        longitude=longitude,
        altitude_m=100.0,
    )


class TelemetryJumpTests(unittest.TestCase):
    def test_fast_high_rate_flight_is_not_a_position_jump(self):
        # Roughly 1.3 m per 1/30 s (about 39 m/s): fast, but continuous.
        samples = [
            sample(index / 30.0, 12.0 + index * 0.000012, 77.0)
            for index in range(120)
        ]
        report = telemetry_quality(samples)
        self.assertEqual(report["probable_position_jumps"], 0)

    def test_large_single_sample_coordinate_teleport_is_reported(self):
        samples = [
            sample(float(index), 12.0 + index * 0.00001, 77.0)
            for index in range(40)
        ]
        samples.extend([
            sample(40.0, 12.02, 77.02),
            sample(41.0, 12.02001, 77.02),
        ])
        report = telemetry_quality(samples)
        self.assertEqual(report["probable_position_jumps"], 1)
        self.assertEqual(len(report["probable_position_jump_events"]), 1)


if __name__ == "__main__":
    unittest.main()
