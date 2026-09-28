import tempfile
import unittest
from pathlib import Path

from sih_drone_pipeline.telemetry import (
    TelemetrySample,
    embed_frame_gps_exif,
    interpolate,
    load_telemetry,
    vertical_reference,
    write_frame_references,
)

try:
    import piexif
    from PIL import Image
except ImportError:
    piexif = None


class _Frame:
    def __init__(self, image_name, time_s):
        self.image_name = image_name
        self.time_s = time_s


class VerticalReferenceTests(unittest.TestCase):
    @staticmethod
    def _srt(tag):
        return "".join(
            f"{i + 1}\n00:00:0{i},000 --> 00:00:0{i},033\n"
            f"[latitude: 30.26{i}] [longitude: -97.75{i}] [{tag}: {100 + i}.0]\n\n"
            for i in range(4)
        )

    def _load(self, name, text, offset=0.0):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(text, encoding="utf-8")
            return load_telemetry(path, altitude_offset_m=offset)

    def test_dji_rel_alt_is_flagged_as_relative(self):
        reference = vertical_reference(self._load("flight.SRT", self._srt("rel_alt")))
        self.assertEqual(reference["status"], "relative_to_takeoff")
        self.assertFalse(reference["heights_are_absolute"])
        self.assertIn("--telemetry-altitude-offset-m", reference["message"])

    def test_dji_abs_alt_is_absolute(self):
        reference = vertical_reference(self._load("flight.SRT", self._srt("abs_alt")))
        self.assertEqual(reference["status"], "absolute")
        self.assertIsNone(reference["message"])

    def test_plain_csv_altitude_is_unknown_and_warned(self):
        rows = "time_s,latitude,longitude,altitude\n" + "".join(
            f"{i},30.26,-97.75,{445 + i}\n" for i in range(4)
        )
        reference = vertical_reference(self._load("log.csv", rows))
        self.assertEqual(reference["status"], "unknown")
        self.assertIsNotNone(reference["message"])

    def test_csv_relative_altitude_column_is_relative(self):
        rows = "time_s,latitude,longitude,rel_altitude\n" + "".join(
            f"{i},30.26,-97.75,{50 + i}\n" for i in range(4)
        )
        self.assertEqual(vertical_reference(self._load("log.csv", rows))["status"], "relative_to_takeoff")

    def test_offset_makes_heights_absolute(self):
        reference = vertical_reference(self._load("flight.SRT", self._srt("rel_alt"), offset=145.0))
        self.assertEqual(reference["status"], "offset_applied")
        self.assertTrue(reference["heights_are_absolute"])


class TelemetryTests(unittest.TestCase):
    def test_interpolates_position_by_video_time(self):
        samples = [
            TelemetrySample(0.0, 10.0, 20.0, 100.0),
            TelemetrySample(2.0, 12.0, 24.0, 120.0),
        ]
        result = interpolate(samples, 1.0)
        self.assertEqual((result.latitude, result.longitude, result.altitude_m), (11.0, 22.0, 110.0))

    def test_reads_required_csv_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flight.csv"
            path.write_text(
                "time_s,latitude,longitude,altitude_m\n0,10,20,100\n1,10.1,20.1,101\n2,10.2,20.2,102\n",
                encoding="utf-8",
            )
            samples = load_telemetry(path)
            self.assertEqual(len(samples), 3)
            self.assertAlmostEqual(samples[-1].altitude_m, 102.0)

    def test_reads_dji_style_srt(self):
        srt = """1
00:00:00,000 --> 00:00:00,100
[latitude: 10.0000] [longitude: 20.0000] [absolute_altitude: 100.0]

2
00:00:01,000 --> 00:00:01,100
[latitude: 10.0001] [longitude: 20.0001] [absolute_altitude: 101.0]

3
00:00:02,000 --> 00:00:02,100
[latitude: 10.0002] [longitude: 20.0002] [absolute_altitude: 102.0]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flight.srt"
            path.write_text(srt, encoding="utf-8")
            samples = load_telemetry(path)
            self.assertEqual(len(samples), 3)
            self.assertEqual(samples[1].time_s, 1.0)
            self.assertAlmostEqual(samples[2].longitude, 20.0002)

    def test_prefers_absolute_over_relative_srt_altitude(self):
        block = lambda i: f"""{i + 1}
00:00:0{i},000 --> 00:00:0{i},100
[latitude: 10.0] [longitude: 20.0] [rel_alt: 5.0] [abs_alt: {100 + i}.0]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flight.srt"
            path.write_text("\n".join(block(i) for i in range(3)), encoding="utf-8")
            samples = load_telemetry(path)
            self.assertEqual(samples[0].altitude_m, 100.0)
            self.assertEqual(samples[2].altitude_m, 102.0)

    def test_frame_reference_origin_matches_colmap_first_reference(self):
        samples = [
            TelemetrySample(0.0, 47.0, 8.0, 400.0),
            TelemetrySample(1.0, 47.1, 8.1, 410.0),
            TelemetrySample(2.0, 47.2, 8.2, 420.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame_gps.txt"
            report = write_frame_references(
                [_Frame("frame_0000.jpg", 0.5), _Frame("frame_0001.jpg", 1.5)],
                samples,
                output,
            )
            first = output.read_text(encoding="utf-8").splitlines()[0].split()
            self.assertEqual(report["origin_policy"], "first_frame_reference")
            self.assertAlmostEqual(report["origin"]["latitude"], float(first[1]))
            self.assertAlmostEqual(report["origin"]["longitude"], float(first[2]))
            self.assertAlmostEqual(report["origin"]["altitude_m"], float(first[3]))

    @unittest.skipIf(piexif is None, "piexif or Pillow is not installed")
    def test_embeds_synchronized_wgs84_pose_prior_without_reencoding(self):
        samples = [
            TelemetrySample(0.0, 12.0, 77.0, 100.0),
            TelemetrySample(1.0, 12.1, 77.1, 110.0),
            TelemetrySample(2.0, 12.2, 77.2, 120.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame_0000.jpg"
            Image.new("RGB", (32, 24), color="green").save(image)
            before_scan = image.read_bytes().split(b"\xff\xda", 1)[-1]
            report = embed_frame_gps_exif([_Frame(image.name, 1.0)], samples, root)
            exif = piexif.load(str(image))
            after_scan = image.read_bytes().split(b"\xff\xda", 1)[-1]
            self.assertEqual(report["embedded_images"], 1)
            self.assertEqual(exif["GPS"][piexif.GPSIFD.GPSLatitudeRef], b"N")
            self.assertEqual(exif["GPS"][piexif.GPSIFD.GPSLongitudeRef], b"E")
            self.assertEqual(before_scan, after_scan)


if __name__ == "__main__":
    unittest.main()
