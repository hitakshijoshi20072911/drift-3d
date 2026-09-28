from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TelemetrySample:
    time_s: float
    latitude: float
    longitude: float
    altitude_m: float
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    horizontal_accuracy_m: float | None = None
    vertical_accuracy_m: float | None = None
    position_source: str | None = None
    altitude_datum: str | None = None


ALIASES = {
    "time_s": ("time_s", "timestamp_s", "seconds", "video_time_s"),
    "latitude": ("latitude", "lat", "gps_latitude"),
    "longitude": ("longitude", "lon", "lng", "gps_longitude"),
    "altitude_m": ("altitude_m", "alt_m", "altitude", "gps_altitude"),
    "yaw_deg": ("yaw_deg", "yaw", "heading"),
    "pitch_deg": ("pitch_deg", "pitch"),
    "roll_deg": ("roll_deg", "roll"),
    "horizontal_accuracy_m": (
        "horizontal_accuracy_m", "h_accuracy_m", "gps_horizontal_accuracy_m", "eph_m",
    ),
    "vertical_accuracy_m": (
        "vertical_accuracy_m", "v_accuracy_m", "gps_vertical_accuracy_m", "epv_m",
    ),
}


def _field(row: dict[str, str], logical: str, required: bool = True) -> float | None:
    normalized = {key.strip().lower(): value for key, value in row.items() if key}
    for alias in ALIASES[logical]:
        if alias in normalized and normalized[alias] not in (None, ""):
            return float(normalized[alias])
    # Fallback to substring match for dynamic columns (e.g. "GPS.Lat", "Time(s)")
    for col_name, value in normalized.items():
        if value not in (None, ""):
            for alias in ALIASES[logical]:
                # Require word boundary or reasonable substring match to avoid 'flat' matching 'lat'
                if alias in col_name.replace(".", "_").replace("(", "_").replace("-", "_").split("_") or alias in col_name:
                    try:
                        return float(value)
                    except ValueError:
                        continue
    if required:
        raise ValueError(f"Telemetry is missing required column: {ALIASES[logical][0]}\nProvided columns: {list(normalized.keys())}")
    return None


def _text_field(row: dict[str, str], *aliases: str) -> str | None:
    normalized = {key.strip().lower(): value for key, value in row.items() if key}
    for alias in aliases:
        if alias in normalized and normalized[alias] not in (None, ""):
            return str(normalized[alias]).strip()
    for col_name, value in normalized.items():
        if value not in (None, ""):
            for alias in aliases:
                if alias in col_name.replace(".", "_").replace("(", "_").replace("-", "_").split("_") or alias in col_name:
                    return str(value).strip()
    return None


_RELATIVE_HEIGHT_HINTS = ("rel", "takeoff", "take_off", "agl", "above_ground", "height_above")
_ABSOLUTE_HEIGHT_HINTS = ("abs", "msl", "amsl", "sea", "ellips", "wgs", "egm", "geoid", "hae")


def _altitude_column_datum(columns: Iterable[str]) -> str | None:
    """Guess the altitude reference from the name of the column the parser reads."""
    names = [name.strip().lower() for name in columns if name]
    chosen = next((name for name in names if name in ALIASES["altitude_m"]), None)
    if chosen is None:
        chosen = next((name for name in names if any(alias in name for alias in ALIASES["altitude_m"])), None)
    if chosen is None:
        return None
    if any(hint in chosen for hint in _RELATIVE_HEIGHT_HINTS):
        return "relative_to_takeoff"
    if any(hint in chosen for hint in _ABSOLUTE_HEIGHT_HINTS):
        return f"absolute ({chosen})"
    return None


def vertical_reference(samples: list[TelemetrySample]) -> dict:
    """Say whether telemetry heights are true elevations or relative to takeoff.

    Many drone logs (DJI rel_alt, most CSV exports) record height above the
    takeoff point. Treated as an elevation, that puts the whole model tens or
    hundreds of metres too low or high, while positions stay correct.
    """
    datums = sorted({sample.altitude_datum for sample in samples if sample.altitude_datum})
    text = " ".join(datums).lower()
    if "offset" in text:
        status = "offset_applied"
    elif "relative" in text:
        status = "relative_to_takeoff"
    elif datums and all(any(hint in datum.lower() for hint in _ABSOLUTE_HEIGHT_HINTS) for datum in datums):
        status = "absolute"
    else:
        status = "unknown"
    absolute = status in {"absolute", "offset_applied"}
    message = None
    if not absolute:
        message = (
            "Telemetry heights are "
            + ("relative to the takeoff point" if status == "relative_to_takeoff" else "of unknown reference")
            + ": DSM/LAS heights are not true elevations (positions are unaffected). "
            "Pass --telemetry-altitude-offset-m <takeoff elevation> to convert them."
        )
    return {"datums": datums, "status": status, "heights_are_absolute": absolute, "message": message}


def parse_csv(path: Path) -> list[TelemetrySample]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    column_datum = _altitude_column_datum(rows[0].keys()) if rows else None
    samples = []
    for row in rows:
        try:
            samples.append(
                TelemetrySample(
                    time_s=float(_field(row, "time_s")),
                    latitude=float(_field(row, "latitude")),
                    longitude=float(_field(row, "longitude")),
                    altitude_m=float(_field(row, "altitude_m")),
                    yaw_deg=_field(row, "yaw_deg", False),
                    pitch_deg=_field(row, "pitch_deg", False),
                    roll_deg=_field(row, "roll_deg", False),
                    horizontal_accuracy_m=_field(row, "horizontal_accuracy_m", False),
                    vertical_accuracy_m=_field(row, "vertical_accuracy_m", False),
                    position_source=_text_field(row, "position_source", "gps_source", "fix_type"),
                    altitude_datum=(
                        _text_field(row, "altitude_datum", "vertical_datum", "height_datum") or column_datum
                    ),
                )
            )
        except ValueError:
            continue
    if not samples:
        raise ValueError("Telemetry CSV contains no rows with valid GPS data.")
    return samples


def _srt_seconds(value: str) -> float:
    hours, minutes, remainder = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(remainder)


def parse_srt(path: Path) -> list[TelemetrySample]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    samples: list[TelemetrySample] = []
    for block in blocks:
        timestamp = re.search(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->", block)
        lat = re.search(r"(?:latitude|lat)\s*[:=]\s*(-?\d+(?:\.\d+)?)", block, re.I)
        lon = re.search(r"(?:longitude|longtitude|lon|lng)\s*[:=]\s*(-?\d+(?:\.\d+)?)", block, re.I)
        alt = re.search(
            r"(?:absolute[_ ]?altitude|abs[_ ]?alt|gps[_ ]?altitude)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            block,
            re.I,
        )
        datum = "absolute (drone abs_alt)" if alt else None
        if not alt:
            alt = re.search(r"(?:relative[_ ]?altitude|rel[_ ]?alt)\s*[:=]\s*(-?\d+(?:\.\d+)?)", block, re.I)
            datum = "relative_to_takeoff" if alt else None
        if not alt:
            alt = re.search(r"(?:altitude|alt)\s*[:=]\s*(-?\d+(?:\.\d+)?)", block, re.I)
        if not (timestamp and lat and lon and alt):
            gps = re.search(
                r"GPS\s*\(?\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*(-?\d+(?:\.\d+)?)",
                block,
                re.I,
            )
            if timestamp and gps:
                lat, lon, alt = gps.group(1), gps.group(2), gps.group(3)
                samples.append(TelemetrySample(_srt_seconds(timestamp.group(1)), float(lat), float(lon), float(alt)))
            continue
        samples.append(
            TelemetrySample(
                _srt_seconds(timestamp.group(1)),
                float(lat.group(1)),
                float(lon.group(1)),
                float(alt.group(1)),
                altitude_datum=datum,
            )
        )
    return samples


def load_telemetry(path: str | Path, altitude_offset_m: float = 0.0) -> list[TelemetrySample]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".srt" or "-->" in text[:2000]:
        samples = parse_srt(path)
    else:
        samples = parse_csv(path)
    if altitude_offset_m:
        from dataclasses import replace

        samples = [
            replace(
                sample,
                altitude_m=sample.altitude_m + altitude_offset_m,
                altitude_datum=f"{sample.altitude_datum or 'unknown'} + offset {altitude_offset_m:g} m",
            )
            for sample in samples
        ]
    samples = sorted(samples, key=lambda sample: sample.time_s)
    if len(samples) < 3:
        raise ValueError("At least three telemetry samples with time, GPS and altitude are required")
    return samples


def is_dji_flight_record(path: str | Path) -> bool:
    """True for a raw DJI flight-record CSV export (OSD.* column names)."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        header = stream.readline()
    return "OSD.latitude" in header and "OSD.altitude" in header


def normalize_dji_flight_record(
    raw_path: str | Path,
    output_path: str | Path,
    video_duration_s: float,
    margin_s: float = 2.0,
) -> dict:
    """Convert a raw DJI flight record to the pipeline's CSV on the *video* clock.

    A flight record's clock is OSD.flyTime -- seconds since takeoff -- and one
    record can span several recordings. Using flyTime as video time attaches every
    GPS fix to the wrong frame by however long the drone flew before recording
    started (9.1 s on the reference flight, worth 160+ m of apparent error).
    CAMERA.isVideo marks when the camera was recording, so the recording whose
    length matches the video defines video t=0.
    """
    if video_duration_s <= 0:
        raise ValueError("video_duration_s must be positive")
    with Path(raw_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)

    def column(predicate, description):
        found = next((name for name in fields if predicate(name.strip().lower())), None)
        if found is None:
            raise ValueError(f"DJI flight record has no {description} column; columns: {fields}")
        return found

    fly_col = column(lambda c: c.startswith("osd.flytime") and "[s]" in c, "OSD.flyTime [s]")
    lat_col = column(lambda c: c == "osd.latitude", "OSD.latitude")
    lon_col = column(lambda c: c == "osd.longitude", "OSD.longitude")
    alt_col = column(lambda c: c.startswith("osd.altitude"), "OSD.altitude")
    alt_scale = 0.3048 if "[ft]" in alt_col.lower() else 1.0
    video_col = next((name for name in fields if "isvideo" in re.sub(r"[^a-z]", "", name.lower())), None)
    if video_col is None:
        raise ValueError(
            "DJI flight record has no CAMERA.isVideo column, so the video start cannot be "
            "located in flight time. Export the record with camera fields, or supply the "
            "video's SRT / a CSV already on the video clock.")

    samples, recording = [], []
    for row in rows:
        try:
            fly = float(row[fly_col])
            lat, lon = float(row[lat_col]), float(row[lon_col])
            alt = float(row[alt_col]) * alt_scale
        except (TypeError, ValueError):
            continue
        if lat == 0.0 and lon == 0.0:      # logged before GPS lock
            continue
        samples.append((fly, lat, lon, alt))
        recording.append(str(row.get(video_col, "")).strip().lower() in {"true", "1", "yes"})

    segments, start = [], None
    for index, on in enumerate(recording + [False]):
        if on and start is None:
            start = index
        elif not on and start is not None:
            segments.append((samples[start][0], samples[index - 1][0]))
            start = None
    if not segments:
        raise ValueError("DJI flight record never shows the camera recording (CAMERA.isVideo)")
    tolerance = max(3.0, 0.03 * video_duration_s)
    listed = [(round(a, 1), round(b, 1)) for a, b in segments]
    matches = [s for s in segments if abs((s[1] - s[0]) - video_duration_s) <= tolerance]
    if not matches:
        raise ValueError(
            f"No recording in the flight record matches the {video_duration_s:.1f} s video "
            f"(recordings, in flight seconds: {listed}). If the camera split one long recording "
            f"into several files, this file does not start where the recording does; supply its "
            f"SRT or a CSV already on the video clock.")
    if len(matches) > 1:
        # Picking the closest length would silently attach another clip's GPS to this video.
        raise ValueError(
            f"{len(matches)} recordings in the flight record match the {video_duration_s:.1f} s "
            f"video (recordings, in flight seconds: {listed}); cannot tell which one this video is. "
            f"Supply the video's SRT or a CSV already on the video clock.")
    begin, end = matches[0]

    synced, last_time = [], None
    for fly, lat, lon, alt in sorted(samples, key=lambda sample: sample[0]):
        time_s = fly - begin
        if -margin_s <= time_s <= video_duration_s + margin_s and time_s != last_time:
            synced.append((time_s, lat, lon, alt))
            last_time = time_s
    if len(synced) < 3:
        raise ValueError("Fewer than three GPS fixes fall inside the recording")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "latitude", "longitude", "altitude_m"])
        writer.writerows((f"{t:.3f}", f"{lat:.10f}", f"{lon:.10f}", f"{alt:.3f}")
                         for t, lat, lon, alt in synced)
    return {
        "sync_method": "dji_camera_isVideo",
        "recording_segments_flytime_s": [(round(a, 3), round(b, 3)) for a, b in segments],
        "video_start_flytime_s": round(begin, 3),
        "recording_duration_s": round(end - begin, 3),
        "video_duration_s": round(video_duration_s, 3),
        "rows": len(synced),
    }


def interpolate(samples: list[TelemetrySample], time_s: float) -> TelemetrySample:
    if time_s <= samples[0].time_s:
        return samples[0]
    if time_s >= samples[-1].time_s:
        return samples[-1]
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid].time_s <= time_s:
            lo = mid
        else:
            hi = mid
    left, right = samples[lo], samples[hi]
    ratio = (time_s - left.time_s) / (right.time_s - left.time_s)

    def mix(a: float, b: float) -> float:
        return a + ratio * (b - a)

    def mix_optional(a: float | None, b: float | None) -> float | None:
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return mix(a, b)

    return TelemetrySample(
        time_s=time_s,
        latitude=mix(left.latitude, right.latitude),
        longitude=mix(left.longitude, right.longitude),
        altitude_m=mix(left.altitude_m, right.altitude_m),
        yaw_deg=mix_optional(left.yaw_deg, right.yaw_deg),
        pitch_deg=mix_optional(left.pitch_deg, right.pitch_deg),
        roll_deg=mix_optional(left.roll_deg, right.roll_deg),
        horizontal_accuracy_m=mix_optional(
            left.horizontal_accuracy_m, right.horizontal_accuracy_m
        ),
        vertical_accuracy_m=mix_optional(
            left.vertical_accuracy_m, right.vertical_accuracy_m
        ),
        position_source=left.position_source if ratio < 0.5 else right.position_source,
        altitude_datum=left.altitude_datum if ratio < 0.5 else right.altitude_datum,
    )


def write_frame_references(
    frame_records: Iterable,
    samples: list[TelemetrySample],
    output_path: str | Path,
) -> dict:
    """Write COLMAP's image-name/latitude/longitude/altitude reference file."""
    references = []
    for frame in frame_records:
        sample = interpolate(samples, float(frame.time_s))
        references.append((frame.image_name, sample.latitude, sample.longitude, sample.altitude_m, frame.time_s))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        for name, lat, lon, alt, _ in references:
            stream.write(f"{name} {lat:.10f} {lon:.10f} {alt:.3f}\n")
    if not references:
        raise ValueError("No frame references could be generated")

    # COLMAP's model_aligner uses the first GPS reference as the origin when
    # --alignment_type enu is selected. Every later conversion must use that
    # exact origin too; using an average position introduces a constant map
    # translation even when the reconstruction itself is accurate.
    _, origin_lat, origin_lon, origin_alt, _ = references[0]
    return {
        "count": len(references),
        "origin": {
            "latitude": origin_lat,
            "longitude": origin_lon,
            "altitude_m": origin_alt,
        },
        "origin_policy": "first_frame_reference",
        "uncertainty": {
            "horizontal_accuracy_m_median": (
                float(__import__("statistics").median(values))
                if (values := [sample.horizontal_accuracy_m for sample in samples if sample.horizontal_accuracy_m is not None])
                else None
            ),
            "vertical_accuracy_m_median": (
                float(__import__("statistics").median(values))
                if (values := [sample.vertical_accuracy_m for sample in samples if sample.vertical_accuracy_m is not None])
                else None
            ),
            "position_sources": sorted({sample.position_source for sample in samples if sample.position_source}),
            "altitude_datums": sorted({sample.altitude_datum for sample in samples if sample.altitude_datum}),
        },
        "vertical_reference": vertical_reference(samples),
    }


def _degrees_to_exif(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    absolute = abs(value)
    degrees = int(absolute)
    minutes_value = (absolute - degrees) * 60.0
    minutes = int(minutes_value)
    seconds = (minutes_value - minutes) * 60.0
    return ((degrees, 1), (minutes, 1), (round(seconds * 1_000_000), 1_000_000))


def embed_frame_gps_exif(
    frame_records: Iterable,
    samples: list[TelemetrySample],
    images_dir: str | Path,
) -> dict:
    """Embed synchronized WGS84 positions so COLMAP can ingest pose priors.

    ``piexif.insert`` modifies JPEG metadata without decoding and re-encoding
    pixels, so the baseline and pose-prior experiments use identical imagery.
    """
    try:
        import piexif
    except ImportError as error:
        raise RuntimeError(
            "GPS pose priors require piexif; install the project requirements"
        ) from error

    images_dir = Path(images_dir)
    embedded = 0
    for frame in frame_records:
        image_path = images_dir / frame.image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        sample = interpolate(samples, float(frame.time_s))
        try:
            exif = piexif.load(str(image_path))
        except piexif.InvalidImageDataError:
            exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif["GPS"] = {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: b"N" if sample.latitude >= 0 else b"S",
            piexif.GPSIFD.GPSLatitude: _degrees_to_exif(sample.latitude),
            piexif.GPSIFD.GPSLongitudeRef: b"E" if sample.longitude >= 0 else b"W",
            piexif.GPSIFD.GPSLongitude: _degrees_to_exif(sample.longitude),
            piexif.GPSIFD.GPSAltitudeRef: 0 if sample.altitude_m >= 0 else 1,
            piexif.GPSIFD.GPSAltitude: (round(abs(sample.altitude_m) * 1000), 1000),
        }
        piexif.insert(piexif.dump(exif), str(image_path))
        embedded += 1
    return {"embedded_images": embedded, "coordinate_system": "WGS84"}
