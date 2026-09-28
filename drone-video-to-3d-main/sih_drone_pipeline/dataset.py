from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
import math
import statistics
from pathlib import Path


ZURICH_SAMPLE_URL = "https://download.ifi.uzh.ch/rpg/AGZ_data/AGZ_subset.zip"
BELLEVIEW_URL = "https://data.pix4d.com/misc/example_datasets/example_belleview.zip"
WHU_DATASET_PAGE = "https://gpcv.whu.edu.cn/data/WHU_Areial_Video_Dataset.html"
HF_DRONE_DATASET_PAGE = "https://huggingface.co/datasets/npeng/drone-telemetry/tree/main"
HF_DRONE_VIDEO_NAME = "20240629164410_0023_D.mp4"
HF_DRONE_FLIGHT_RECORD_NAME = "FlightRecord_2024-06-29.csv"
HF_DRONE_BASE_URL = "https://huggingface.co/datasets/npeng/drone-telemetry/resolve/main"
HF_DRONE_SHA256 = {
    HF_DRONE_VIDEO_NAME: "4908722e22c824943b3596407a4b8c7654173c2b5bc56882f76581a1a9b242c5",
    HF_DRONE_FLIGHT_RECORD_NAME: "1e1950ca4dc6cdce40beeaed7853421296875824fceb996d66f57824ca42ecb7",
}
FEET_TO_METRES = 0.3048


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe ZIP member: {member.filename}")
    archive.extractall(destination)


def download_zurich_sample(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "AGZ_subset.zip"
    extracted = destination / "zurich_mav_sample"
    if not archive_path.exists():
        print(f"Downloading official Zurich MAV sample: {ZURICH_SAMPLE_URL}")
        urllib.request.urlretrieve(ZURICH_SAMPLE_URL, archive_path)
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted)
    return extracted


def download_belleview_sample(destination: str | Path) -> Path:
    """Download and safely extract Pix4D's small geotagged rooftop dataset."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "example_belleview.zip"
    extracted = destination / "belleview"
    if not archive_path.exists():
        print(f"Downloading Pix4D Belleview aerial sample: {BELLEVIEW_URL}")
        urllib.request.urlretrieve(BELLEVIEW_URL, archive_path)
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted)
    return extracted


def download_huggingface_drone_sample(destination: str | Path) -> dict[str, Path]:
    """Download the public native DJI video and matching flight record."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in (HF_DRONE_VIDEO_NAME, HF_DRONE_FLIGHT_RECORD_NAME):
        target = destination / name
        expected = HF_DRONE_SHA256[name]
        if target.exists() and _sha256(target) != expected:
            raise ValueError(f"Cached download failed SHA256 validation: {target}. Remove it and retry.")
        if not target.exists():
            url = f"{HF_DRONE_BASE_URL}/{name}?download=true"
            print(f"Downloading {name} from {HF_DRONE_DATASET_PAGE}")
            partial = target.with_name(target.name + ".part")
            if partial.exists():
                partial.unlink()
            urllib.request.urlretrieve(url, partial)
            actual = _sha256(partial)
            if actual != expected:
                raise ValueError(f"Downloaded {name} SHA256 mismatch: expected {expected}, got {actual}")
            partial.replace(target)
        files["video" if name == HF_DRONE_VIDEO_NAME else "flight_record"] = target
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name)
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _recording_segments(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Return continuous CAMERA.isVideo intervals from a DJI Flight Reader CSV."""
    segments: list[list[dict[str, str]]] = []
    active: list[dict[str, str]] = []
    previous_time: float | None = None
    for row in rows:
        time_s = _as_float(row, "OSD.flyTime [s]")
        valid = (
            time_s is not None
            and _is_true(row.get("CAMERA.isVideo"))
            and _as_float(row, "OSD.latitude") is not None
            and _as_float(row, "OSD.longitude") is not None
        )
        # The logger normally samples at 10 Hz. A gap over two seconds marks a
        # new recording even if malformed CSV continuation lines hid a toggle.
        if valid and (previous_time is None or time_s - previous_time <= 2.0):
            active.append(row)
        elif valid:
            if active:
                segments.append(active)
            active = [row]
        elif active:
            segments.append(active)
            active = []
        previous_time = time_s if valid else None
    if active:
        segments.append(active)
    return [segment for segment in segments if len(segment) >= 3]


def prepare_huggingface_flight_record(
    flight_record_csv: str | Path,
    output_dir: str | Path,
    video_path: str | Path | None = None,
) -> dict:
    """Normalize npeng/drone-telemetry's DJI log for the reconstruction.

    The source flight contains more than one CAMERA.isVideo interval. The
    published MP4 is the longest interval, so this adapter deliberately selects
    that continuous segment and makes its first sample video time zero.
    """
    flight_record_csv, output_dir = Path(flight_record_csv), Path(output_dir)
    if not flight_record_csv.is_file():
        raise FileNotFoundError(flight_record_csv)
    if video_path is not None and not Path(video_path).is_file():
        raise FileNotFoundError(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with flight_record_csv.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        rows = list(csv.DictReader(stream))
    segments = _recording_segments(rows)
    if not segments:
        raise ValueError("No continuous CAMERA.isVideo telemetry segment was found")
    segment = max(
        segments,
        key=lambda item: (_as_float(item[-1], "OSD.flyTime [s]") or 0)
        - (_as_float(item[0], "OSD.flyTime [s]") or 0),
    )
    start_s = _as_float(segment[0], "OSD.flyTime [s]")
    end_s = _as_float(segment[-1], "OSD.flyTime [s]")
    if start_s is None or end_s is None:
        raise ValueError("Selected recording segment has invalid flight times")

    telemetry_path = output_dir / "hf_drone_telemetry.csv"
    normalized_rows = []
    gps_satellites = []
    with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "time_s", "latitude", "longitude", "altitude_m",
            "yaw_deg", "pitch_deg", "roll_deg",
        ])
        for row in segment:
            flight_time = _as_float(row, "OSD.flyTime [s]")
            latitude = _as_float(row, "OSD.latitude")
            longitude = _as_float(row, "OSD.longitude")
            altitude_ft = _as_float(row, "OSD.altitude [ft]")
            if None in (flight_time, latitude, longitude, altitude_ft):
                continue
            yaw = _as_float(row, "GIMBAL.yaw [360]")
            if yaw is None:
                yaw = _as_float(row, "OSD.yaw [360]")
            pitch = _as_float(row, "GIMBAL.pitch")
            roll = _as_float(row, "GIMBAL.roll")
            normalized = (
                flight_time - start_s, latitude, longitude,
                altitude_ft * FEET_TO_METRES, yaw, pitch, roll,
            )
            normalized_rows.append(normalized)
            writer.writerow([
                f"{normalized[0]:.3f}", f"{latitude:.10f}", f"{longitude:.10f}",
                f"{normalized[3]:.3f}",
                "" if yaw is None else f"{yaw:.3f}",
                "" if pitch is None else f"{pitch:.3f}",
                "" if roll is None else f"{roll:.3f}",
            ])
            satellites = _as_float(row, "OSD.gpsNum")
            if satellites is not None:
                gps_satellites.append(satellites)
    if len(normalized_rows) < 30:
        raise ValueError(f"Only {len(normalized_rows)} usable samples in the recording interval")

    video_duration = _video_duration(Path(video_path)) if video_path else None
    telemetry_duration = normalized_rows[-1][0]
    duration_difference = abs(video_duration - telemetry_duration) if video_duration is not None else None
    manifest = {
        "name": "npeng/drone-telemetry DJI Mini 4 Pro flight",
        "source": HF_DRONE_DATASET_PAGE,
        "source_video": HF_DRONE_VIDEO_NAME,
        "source_flight_record": HF_DRONE_FLIGHT_RECORD_NAME,
        "source_type": "native continuous drone MP4 with matching DJI flight telemetry",
        "video": str(Path(video_path).resolve()) if video_path else None,
        "telemetry": str(telemetry_path.resolve()),
        "telemetry_samples": len(normalized_rows),
        "recording_segments_found": len(segments),
        "selected_segment_policy": "longest_continuous_CAMERA.isVideo_interval",
        "source_segment_start_flight_s": start_s,
        "source_segment_end_flight_s": end_s,
        "telemetry_duration_s": telemetry_duration,
        "video_duration_s": video_duration,
        "video_telemetry_duration_difference_s": duration_difference,
        "altitude_conversion": "OSD.altitude [ft] * 0.3048 to WGS84-style altitude_m",
        "orientation_source": "gimbal yaw/pitch/roll, with aircraft yaw fallback",
        "gps_satellites_median": statistics.median(gps_satellites) if gps_satellites else None,
        "accuracy_evidence": "GPS telemetry only; no independent surveyed surface checkpoints are supplied",
        "license_note": "No license was declared on the dataset page when this adapter was authored; obtain permission before redistribution.",
        "privacy_note": "The raw flight log contains device identifiers; only normalized navigation fields are written to outputs.",
    }
    metadata_path = output_dir / "dataset_manifest.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _gps_decimal(values, reference: str) -> float:
    degrees, minutes, seconds = (float(value) for value in values)
    coordinate = degrees + minutes / 60.0 + seconds / 3600.0
    return -coordinate if reference.upper() in {"S", "W"} else coordinate


def _gps_altitude_is_below_sea_level(reference) -> bool:
    """Decode EXIF GPSAltitudeRef from either integer or raw-byte form."""
    if isinstance(reference, (bytes, bytearray)):
        return int.from_bytes(reference, byteorder="big") == 1
    return int(reference or 0) == 1


def _image_exif_position(path: Path) -> tuple[float, float, float, str]:
    from PIL import ExifTags, Image

    with Image.open(path) as image:
        exif = image.getexif()
        try:
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (AttributeError, KeyError):
            gps = exif.get(34853, {})
        if not gps or 2 not in gps or 4 not in gps:
            raise ValueError(f"Image has no complete EXIF GPS position: {path}")
        latitude = _gps_decimal(gps[2], str(gps.get(1, "N")))
        longitude = _gps_decimal(gps[4], str(gps.get(3, "E")))
        altitude = float(gps.get(6, 0.0))
        if _gps_altitude_is_below_sea_level(gps.get(5, 0)):
            altitude = -altitude
        captured = str(exif.get(36867) or exif.get(306) or "")
    return latitude, longitude, altitude, captured


def prepare_geotagged_images(
    extracted_root: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    max_width: int = 1920,
    max_images: int | None = None,
) -> dict:
    """Convert ordered, geotagged aerial JPGs to pipeline video and telemetry."""
    extracted_root, output_dir = Path(extracted_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in extracted_root.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        try:
            latitude, longitude, altitude, captured = _image_exif_position(path)
        except ValueError:
            continue
        records.append((captured, path.name.lower(), path, latitude, longitude, altitude))
    records.sort(key=lambda item: (item[0], item[1]))
    if max_images:
        records = records[:max_images]
    if len(records) < 10:
        raise ValueError(f"Only {len(records)} geotagged aerial images found under {extracted_root}")

    import cv2
    first = cv2.imread(str(records[0][2]))
    if first is None:
        raise RuntimeError(f"Could not read {records[0][2]}")
    source_height, source_width = first.shape[:2]
    width = min(max_width, source_width)
    width -= width % 2
    height = int(round(source_height * width / source_width))
    height -= height % 2
    frames_dir = output_dir / "source_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()
    for index, record in enumerate(records):
        image = cv2.imread(str(record[2]))
        if image is None:
            raise RuntimeError(f"Could not read {record[2]}")
        normalized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(frames_dir / f"frame_{index:06d}.jpg"), normalized):
            raise RuntimeError(f"Could not normalize {record[2]}")

    video = output_dir / "belleview_rooftops.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%06d.jpg"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(video),
    ], check=True)
    telemetry = output_dir / "belleview_telemetry.csv"
    with telemetry.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "latitude", "longitude", "altitude_m"])
        for index, record in enumerate(records):
            writer.writerow([
                f"{index / fps:.6f}", f"{record[3]:.10f}", f"{record[4]:.10f}", f"{record[5]:.3f}",
            ])
    manifest = {
        "name": "Pix4D Belleview Avenue rooftop demonstration",
        "source": BELLEVIEW_URL,
        "source_type": "38 geotagged aerial photographs from one grid flight",
        "frames": len(records),
        "fps": fps,
        "video": str(video),
        "telemetry": str(telemetry),
        "constructed_video": True,
        "attribution": "Courtesy of Pix4D / pix4d.com",
        "note": "Demonstration input constructed from capture-ordered geotagged photos; use a native continuous drone video for final SIH compliance.",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _find(root: Path, names: set[str]) -> Path:
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in names:
            return path
    raise FileNotFoundError(f"Could not find one of {sorted(names)} under {root}")


def _numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        for row in csv.reader(stream):
            if len(row) == 1:
                row = [value for value in re.split(r"[;,\s]+", row[0].strip()) if value]
            else:
                # The official sample CSVs contain unused trailing columns.
                # Ignore only trailing blanks so a valid numeric row is not
                # discarded by float("").
                while row and not row[-1].strip():
                    row.pop()
            try:
                rows.append([float(value.strip()) for value in row])
            except (ValueError, TypeError):
                continue
    return rows


def _image_id(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def _video_duration(path: Path) -> float | None:
    """Read duration without requiring ffprobe; OpenCV is already a dependency."""
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return frames / fps if frames > 0 and fps > 0 else None


def _normalized_pose_times(raw_times: list[float], video_duration_s: float | None) -> tuple[list[float], float]:
    """Normalize seconds/ms/us/ns timestamps using the video duration as evidence."""
    if len(raw_times) < 3:
        raise ValueError("WHU ground truth must contain at least three poses")
    relative = [value - raw_times[0] for value in raw_times]
    candidates = (1.0, 1e-3, 1e-6, 1e-9)
    if video_duration_s and relative[-1] > 0:
        scale = min(candidates, key=lambda item: abs(relative[-1] * item - video_duration_s))
    else:
        median_step = sorted(
            value for value in (relative[i] - relative[i - 1] for i in range(1, len(relative))) if value > 0
        )[max(0, (len(relative) - 2) // 2)]
        scale = 1.0 if median_step < 100 else 1e-3 if median_step < 100_000 else 1e-6
    return [value * scale for value in relative], scale


def _enu_to_geodetic(
    east_m: float,
    north_m: float,
    up_m: float,
    origin_latitude: float,
    origin_longitude: float,
    origin_altitude_m: float,
) -> tuple[float, float, float]:
    """Convert local ENU positions to WGS84 without treating metres as degrees."""
    from pyproj import Transformer

    geodetic_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    ecef_to_geodetic = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    x0, y0, z0 = geodetic_to_ecef.transform(origin_longitude, origin_latitude, origin_altitude_m)
    lat = math.radians(origin_latitude)
    lon = math.radians(origin_longitude)
    # Transpose of the ECEF->ENU rotation used by validation._to_enu.
    dx = -math.sin(lon) * east_m - math.sin(lat) * math.cos(lon) * north_m + math.cos(lat) * math.cos(lon) * up_m
    dy = math.cos(lon) * east_m - math.sin(lat) * math.sin(lon) * north_m + math.cos(lat) * math.sin(lon) * up_m
    dz = math.cos(lat) * north_m + math.sin(lat) * up_m
    longitude, latitude, altitude = ecef_to_geodetic.transform(x0 + dx, y0 + dy, z0 + dz)
    return latitude, longitude, altitude


def _whu_sequence_file(root: Path, sequence: str, kind: str) -> Path:
    sequence = sequence.lower()
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or sequence not in str(path.parent).lower():
            continue
        name = path.name.lower()
        if kind == "video" and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            candidates.append(path)
        elif kind == "ground_truth" and name == "ground_truth.txt":
            candidates.append(path)
        elif kind == "gcp_observations" and "gcp" in name and ("observation" in name or "obsev" in name):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Could not find WHU {sequence} {kind} under {root}")
    return min(candidates, key=lambda path: len(path.parts))


def prepare_whu_dataset(
    extracted_root: str | Path,
    output_dir: str | Path,
    sequence: str,
    origin_latitude: float,
    origin_longitude: float,
    origin_altitude_m: float = 0.0,
    pose_translation_convention: str = "camera_center",
) -> dict:
    """Standardize a manually downloaded WHU sequence for the main pipeline.

    WHU publishes its large files through Quark, so the user downloads the
    videos/keyframes, calibration, and ground-truth archives once and points
    this adapter at their common extracted directory.
    """
    if sequence not in {"regular", "irregular"}:
        raise ValueError("WHU sequence must be 'regular' or 'irregular'")
    if pose_translation_convention not in {"camera_center", "world_to_camera"}:
        raise ValueError("pose_translation_convention must be camera_center or world_to_camera")
    root, output_dir = Path(extracted_root), Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = _whu_sequence_file(root, sequence, "video")
    truth_path = _whu_sequence_file(root, sequence, "ground_truth")
    observations_path = _whu_sequence_file(root, sequence, "gcp_observations")
    points_path = _find(root, {"points.txt"})
    rows = [row for row in _numeric_rows(truth_path) if len(row) >= 8]
    if len(rows) < 3:
        raise ValueError(f"No usable WHU poses in {truth_path}")
    duration = _video_duration(video)
    times, time_scale = _normalized_pose_times([row[0] for row in rows], duration)
    positions = []
    for row in rows:
        translation = row[1:4]
        if pose_translation_convention == "world_to_camera":
            import numpy as np
            qx, qy, qz, qw = row[4:8]
            rotation = np.array([
                [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
                [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
                [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
            ])
            translation = (-(rotation.T @ np.asarray(translation))).tolist()
        positions.append(translation)
    pose_origin = positions[0]
    local_positions = [
        [position[axis] - pose_origin[axis] for axis in range(3)] for position in positions
    ]

    telemetry = output_dir / f"whu_{sequence}_telemetry.csv"
    trajectory = output_dir / f"whu_{sequence}_ground_truth.csv"
    with telemetry.open("w", newline="", encoding="utf-8") as stream, trajectory.open(
        "w", newline="", encoding="utf-8"
    ) as truth_stream:
        telemetry_writer = csv.writer(stream)
        truth_writer = csv.writer(truth_stream)
        telemetry_writer.writerow(["time_s", "latitude", "longitude", "altitude_m"])
        truth_writer.writerow(["time_s", "x_m", "y_m", "z_m"])
        for time_s, position in zip(times, local_positions):
            east, north, up = position
            latitude, longitude, altitude = _enu_to_geodetic(
                east, north, up, origin_latitude, origin_longitude, origin_altitude_m
            )
            telemetry_writer.writerow([
                f"{time_s:.9f}", f"{latitude:.10f}", f"{longitude:.10f}", f"{altitude:.4f}"
            ])
            truth_writer.writerow([f"{time_s:.9f}", east, north, up])

    gcp_points = output_dir / "whu_gcp_points.csv"
    point_rows = [row for row in _numeric_rows(points_path) if len(row) >= 4]
    local_point_rows = [
        [row[0], row[1] - pose_origin[0], row[2] - pose_origin[1], row[3] - pose_origin[2]]
        for row in point_rows
    ]
    with gcp_points.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint_id", "known_x_m", "known_y_m", "known_z_m"])
        writer.writerows((int(row[0]), row[1], row[2], row[3]) for row in local_point_rows)
    observations_copy = output_dir / f"whu_{sequence}_gcp_observations.txt"
    shutil.copy2(observations_path, observations_copy)
    checkpoint_template = output_dir / "whu_checkpoint_measurements.csv"
    with checkpoint_template.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "checkpoint_id", "known_x_m", "known_y_m", "known_z_m",
            "reconstructed_x_m", "reconstructed_y_m", "reconstructed_z_m",
            "horizontal_uncertainty_m", "vertical_uncertainty_m",
        ])
        writer.writerows(
            (int(row[0]), row[1], row[2], row[3], "", "", "", "", "")
            for row in local_point_rows
        )

    manifest = {
        "name": f"WHU Aerial Video Dataset - {sequence}",
        "source": WHU_DATASET_PAGE,
        "source_type": "native continuous UAV video with camera calibration, poses, and surveyed GCPs",
        "video": str(video.resolve()),
        "video_duration_s": duration,
        "telemetry": str(telemetry.resolve()),
        "ground_truth_trajectory": str(trajectory.resolve()),
        "gcp_points": str(gcp_points.resolve()),
        "gcp_observations": str(observations_copy.resolve()),
        "checkpoint_measurements_template": str(checkpoint_template.resolve()),
        "pose_translation_convention": pose_translation_convention,
        "pose_origin_removed": pose_origin,
        "normalized_pose_frame": "first_camera_origin_assumed_ENU_metres",
        "timestamp_scale_to_seconds": time_scale,
        "synthetic_geodetic_origin": {
            "latitude": origin_latitude,
            "longitude": origin_longitude,
            "altitude_m": origin_altitude_m,
        },
        "warning": "Confirm the WHU release's pose-axis and translation convention before using accuracy numbers in a submission. Populate reconstructed checkpoint coordinates before claiming surface accuracy.",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_zurich_sample(
    extracted_root: str | Path,
    output_dir: str | Path,
    frame_count: int = 300,
    start_index: int = 0,
    fps: float = 20.0,
) -> dict:
    """Create a deterministic MP4, GPS CSV and independent trajectory CSV."""
    extracted_root, output_dir = Path(extracted_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gps_path = _find(extracted_root, {"onbordgps.csv", "onboardgps.csv"})
    gt_path = _find(extracted_root, {"groundtruthagl.csv"})
    images = {}
    for path in extracted_root.rglob("*"):
        path_lower = str(path).lower()
        if (
            path.suffix.lower() not in {".jpg", ".jpeg", ".png"}
            or "street" in path_lower
            or "calib" in path_lower
            or "mav images" not in path_lower
        ):
            continue
        identifier = _image_id(path)
        if identifier is not None:
            images[identifier] = path

    gps = {}
    for row in _numeric_rows(gps_path):
        if len(row) < 5:
            continue
        timestamp_us, image_id, lat_e7, lon_e7, alt_mm = row[:5]
        fix_type = int(row[7]) if len(row) > 7 else 3
        if int(image_id) in images and fix_type >= 3:
            # Some releases store WGS84 coordinates directly in degrees and
            # altitude in metres; others use scaled integer telemetry units.
            latitude = lat_e7 / 1e7 if abs(lat_e7) > 90 else lat_e7
            longitude = lon_e7 / 1e7 if abs(lon_e7) > 180 else lon_e7
            altitude = alt_mm / 1e3 if abs(alt_mm) > 10_000 else alt_mm
            gps[int(image_id)] = (timestamp_us, latitude, longitude, altitude)
    ground_truth = {int(row[0]): tuple(row[1:4]) for row in _numeric_rows(gt_path) if len(row) >= 4}
    common = sorted(set(images) & set(gps), key=lambda identifier: gps[identifier][0])
    selected = common[start_index:start_index + frame_count]
    if len(selected) < min(30, frame_count):
        raise ValueError(f"Only {len(selected)} synchronized images found; choose a different start index")

    frames_dir = output_dir / "source_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()
    import cv2
    for index, identifier in enumerate(selected):
        image = cv2.imread(str(images[identifier]))
        if image is None or not cv2.imwrite(str(frames_dir / f"frame_{index:06d}.jpg"), image):
            raise RuntimeError(f"Could not normalize source image {images[identifier]}")

    video = output_dir / "zurich_mav_sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%06d.jpg"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    telemetry = output_dir / "zurich_mav_telemetry.csv"
    with telemetry.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "latitude", "longitude", "altitude_m"])
        for index, identifier in enumerate(selected):
            _, lat, lon, alt = gps[identifier]
            writer.writerow([f"{index / fps:.6f}", f"{lat:.10f}", f"{lon:.10f}", f"{alt:.3f}"])
    trajectory = output_dir / "zurich_mav_ground_truth.csv"
    with trajectory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "x_m", "y_m", "z_m"])
        for index, identifier in enumerate(selected):
            if identifier in ground_truth:
                writer.writerow([f"{index / fps:.6f}", *ground_truth[identifier]])
    manifest = {
        "name": "Zurich Urban Micro Aerial Vehicle sample",
        "source": ZURICH_SAMPLE_URL,
        "citation": "A. L. Majdik, C. Till, D. Scaramuzza, The Zurich Urban Micro Aerial Vehicle Dataset, IJRR, 2017",
        "frames": len(selected),
        "fps": fps,
        "video": str(video),
        "telemetry": str(telemetry),
        "ground_truth_trajectory": str(trajectory),
        "note": "Constructed test video from the time-ordered synchronized MAV image sequence.",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare a public aerial verification dataset")
    parser.add_argument("--dataset", choices=("zurich", "belleview", "whu", "hf-drone"), default="zurich")
    parser.add_argument("--download-dir", default="/content/datasets")
    parser.add_argument("--input-root", help="Extracted WHU videos/keyframes/calibration/ground-truth directory")
    parser.add_argument("--flight-record", help="Existing DJI Flight Reader CSV for --dataset hf-drone")
    parser.add_argument("--video", help="Existing MP4 for --dataset hf-drone")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sequence", choices=("regular", "irregular"), default="regular")
    parser.add_argument("--origin-latitude", type=float)
    parser.add_argument("--origin-longitude", type=float)
    parser.add_argument("--origin-altitude-m", type=float, default=0.0)
    parser.add_argument(
        "--pose-translation-convention", choices=("camera_center", "world_to_camera"),
        default="camera_center",
    )
    args = parser.parse_args()
    if args.dataset == "hf-drone":
        if args.flight_record or args.video:
            if not (args.flight_record and args.video):
                parser.error("hf-drone requires both --flight-record and --video when using existing files")
            files = {"flight_record": Path(args.flight_record), "video": Path(args.video)}
        else:
            files = download_huggingface_drone_sample(args.download_dir)
        result = prepare_huggingface_flight_record(
            files["flight_record"], args.output, video_path=files["video"]
        )
    elif args.dataset == "whu":
        if not args.input_root or args.origin_latitude is None or args.origin_longitude is None:
            parser.error("WHU requires --input-root, --origin-latitude, and --origin-longitude")
        result = prepare_whu_dataset(
            args.input_root, args.output, args.sequence,
            args.origin_latitude, args.origin_longitude, args.origin_altitude_m,
            args.pose_translation_convention,
        )
    elif args.dataset == "belleview":
        root = download_belleview_sample(args.download_dir)
        result = prepare_geotagged_images(root, args.output, max_images=args.frames)
    else:
        root = download_zurich_sample(args.download_dir)
        result = prepare_zurich_sample(root, args.output, args.frames, args.start_index)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
