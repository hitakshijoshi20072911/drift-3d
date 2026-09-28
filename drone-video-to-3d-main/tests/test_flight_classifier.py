import math
import unittest

from sih_drone_pipeline.flight_classifier import VALIDATED_TARGET_FRAMES, classify
from sih_drone_pipeline.telemetry import TelemetrySample

LAT0, LON0 = 30.26, -97.75
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT0))


def _flight(points, dt=1.0):
    """points: list of (east_m, north_m, altitude_m) at uniform dt spacing."""
    return [
        TelemetrySample(
            time_s=i * dt,
            latitude=LAT0 + north / M_PER_DEG_LAT,
            longitude=LON0 + east / M_PER_DEG_LON,
            altitude_m=alt,
        )
        for i, (east, north, alt) in enumerate(points)
    ]


class FlightClassifierTests(unittest.TestCase):
    def test_planar_survey_keeps_fixed_budget_geometry_selector(self):
        # 400 m straight pass at constant altitude.
        samples = _flight([(x, 0.0, 100.0) for x in range(0, 401, 5)])
        result = classify(samples)
        self.assertLess(result.viewpoint_complexity_ratio, 0.05)
        self.assertEqual(result.recommended_keyframe_mode, "geometry")
        self.assertEqual(result.recommended_profile, "verified-fast")

    def test_climbing_orbit_switches_to_adaptive_selector(self):
        # Small 200 m orbit that climbs 60 m -- the HF-SFO failure pattern.
        points = []
        for i in range(120):
            angle = 2 * math.pi * i / 120
            points.append((100 * math.cos(angle), 100 * math.sin(angle), 10 + 60 * i / 119))
        result = classify(_flight(points))
        self.assertGreaterEqual(result.viewpoint_complexity_ratio, 0.05)
        self.assertEqual(result.recommended_keyframe_mode, "adaptive")
        self.assertEqual(result.recommended_profile, "verified-fast-adaptive")

    def test_frame_budget_never_scales_with_footprint(self):
        # Regression: footprint-scaled budgets picked 300 frames for DJI_1006
        # and overran badly. Every footprint must get the validated budget.
        for length_m in (20.0, 500.0, 2000.0):
            samples = _flight([(0.0, 0.0, 100.0), (length_m, 0.0, 100.0)])
            self.assertEqual(classify(samples).recommended_target_frames, VALIDATED_TARGET_FRAMES)

    def test_large_footprint_gets_a_coverage_note(self):
        samples = _flight([(0.0, 0.0, 100.0), (2000.0, 0.0, 100.0)])
        self.assertIsNotNone(classify(samples).coverage_feasibility_warning)

    def test_small_footprint_gets_no_coverage_note(self):
        samples = _flight([(0.0, 0.0, 30.0), (20.0, 0.0, 30.0)])
        self.assertIsNone(classify(samples).coverage_feasibility_warning)

    def test_60hz_rows_with_slow_gps_do_not_fake_hover_or_impossible_speed(self):
        # 60 Hz rows, GPS position only updating at 5 Hz, true speed 10 m/s.
        samples = []
        for i in range(60 * 30):
            t = i / 60.0
            east = 10.0 * math.floor(t * 5) / 5
            samples.append(TelemetrySample(
                time_s=t,
                latitude=LAT0,
                longitude=LON0 + east / M_PER_DEG_LON,
                altitude_m=80.0,
            ))
        result = classify(samples)
        self.assertLess(result.hover_fraction, 0.1)
        self.assertAlmostEqual(result.speed_median_mps, 10.0, delta=1.0)
        self.assertLess(result.speed_p95_mps, 12.0)

    def test_rejects_insufficient_telemetry(self):
        with self.assertRaises(ValueError):
            classify(_flight([(0.0, 0.0, 10.0)]))


if __name__ == "__main__":
    unittest.main()
