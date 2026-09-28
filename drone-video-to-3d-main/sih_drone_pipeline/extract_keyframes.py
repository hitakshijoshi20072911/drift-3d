from __future__ import annotations

import csv
import hashlib
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    image_name: str
    source_frame: int
    time_s: float
    sharpness: float
    motion_score: float
    baseline_m: float = 0.0
    parallax_px: float = 0.0
    tracked_fraction: float = 0.0
    geometry_score: float = 0.0


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _distance_m(left, right) -> float:
    """Approximate short WGS84 distances, including altitude, in metres."""
    latitude = math.radians((left.latitude + right.latitude) / 2.0)
    north = math.radians(right.latitude - left.latitude) * 6378137.0
    east = math.radians(right.longitude - left.longitude) * 6378137.0 * math.cos(latitude)
    up = right.altitude_m - left.altitude_m
    return float(math.sqrt(east * east + north * north + up * up))


def _trajectory_length_m(samples: Sequence) -> float:
    # Consumer flight logs can be 10 Hz or faster. Summing every jittery GPS
    # sample greatly exaggerates path length, so estimate it at ~1 Hz.
    if not samples:
        return 0.0
    coarse = [samples[0]]
    for sample in samples[1:]:
        if sample.time_s - coarse[-1].time_s >= 1.0:
            coarse.append(sample)
    if coarse[-1] is not samples[-1]:
        coarse.append(samples[-1])
    return sum(_distance_m(left, right) for left, right in zip(coarse, coarse[1:]))


def _telemetry_path(samples: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Return a jitter-resistant time/cumulative-distance trajectory."""
    if not samples:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    coarse = [samples[0]]
    for sample in samples[1:]:
        if sample.time_s - coarse[-1].time_s >= 1.0:
            coarse.append(sample)
    if coarse[-1] is not samples[-1]:
        coarse.append(samples[-1])
    times = np.asarray([sample.time_s for sample in coarse], dtype=float)
    distance = np.zeros(len(coarse), dtype=float)
    if len(coarse) > 1:
        distance[1:] = np.cumsum([
            _distance_m(left, right) for left, right in zip(coarse, coarse[1:])
        ])
    # np.interp requires increasing x values. Telemetry validation reports
    # duplicate/reversed timestamps separately; keep one deterministic sample.
    unique_times, unique_indices = np.unique(times, return_index=True)
    return unique_times, distance[unique_indices]


def _hybrid_anchor_indices(
    total: int,
    fps: float,
    target_frames: int,
    telemetry_samples: Sequence,
    spatial_fraction: float = 0.60,
) -> np.ndarray:
    """Allocate anchors by flown distance and time, preserving both coverage types.

    Spatial anchors prevent fast portions of a flight from being undersampled;
    time anchors retain coverage while hovering or moving only a few metres.
    """
    target_frames = min(total, target_frames)
    spatial_count = max(2, int(round(target_frames * spatial_fraction)))
    temporal_count = max(2, target_frames - spatial_count)
    temporal = np.linspace(0, total - 1, temporal_count).round().astype(int)
    path_times, cumulative = _telemetry_path(telemetry_samples)
    if len(path_times) < 2 or cumulative[-1] <= 0:
        return np.unique(
            np.linspace(0, total - 1, target_frames).round().astype(int)
        )
    unique_distance, unique_positions = np.unique(cumulative, return_index=True)
    distance_times = path_times[unique_positions]
    spatial_distances = np.linspace(0.0, unique_distance[-1], spatial_count)
    spatial_times = np.interp(spatial_distances, unique_distance, distance_times)
    spatial = np.clip(np.rint(spatial_times * fps), 0, total - 1).astype(int)
    anchors = set(int(value) for value in np.concatenate((temporal, spatial)))
    anchors.update((0, total - 1))

    # Quantization and shared end points can collapse anchors. Fill holes by
    # repeatedly choosing the timeline point farthest from an existing anchor.
    pool = np.unique(
        np.linspace(0, total - 1, min(total, target_frames * 8)).round().astype(int)
    )
    while len(anchors) < target_frames:
        existing = np.asarray(sorted(anchors), dtype=int)
        remaining = np.asarray([value for value in pool if int(value) not in anchors])
        if not len(remaining):
            break
        nearest = np.min(np.abs(remaining[:, None] - existing[None, :]), axis=1)
        anchors.add(int(remaining[int(np.argmax(nearest))]))
    return np.asarray(sorted(anchors)[:target_frames], dtype=int)


def _candidate_bins_from_anchors(
    anchors: np.ndarray, total: int, candidates_per_bin: int
) -> list[np.ndarray]:
    """Create non-overlapping local candidate windows around ordered anchors."""
    bins: list[np.ndarray] = []
    for index, anchor in enumerate(anchors):
        left = 0 if index == 0 else int((anchors[index - 1] + anchor) // 2 + 1)
        right = total - 1 if index == len(anchors) - 1 else int((anchor + anchors[index + 1]) // 2)
        count = min(max(1, candidates_per_bin), max(1, right - left + 1))
        candidates = np.unique(np.linspace(left, right, count).round().astype(int))
        bins.append(candidates)
    return bins


def _candidate_bins_from_uniform_pool(
    anchors: np.ndarray, candidate_indices: np.ndarray
) -> list[np.ndarray]:
    """Partition a fast uniform decode pool around non-uniform anchors.

    FFmpeg's ``fps`` filter is much faster than evaluating hundreds of
    ``eq(n, frame)`` expressions for every 4K source frame. The pool stays
    uniform for decoding efficiency, while its non-overlapping bins follow
    the telemetry/time anchors. Every anchor receives at least one unique
    candidate when the pool is larger than the requested output budget.
    """
    pool_size = len(candidate_indices)
    anchor_count = len(anchors)
    if pool_size < anchor_count:
        raise ValueError("Candidate pool must contain at least one frame per anchor")
    ranks = np.searchsorted(candidate_indices, anchors, side="left")
    ranks = np.clip(ranks, 0, pool_size - 1).astype(int)
    for index in range(1, anchor_count):
        ranks[index] = max(ranks[index], ranks[index - 1] + 1)
    for index in range(anchor_count - 1, -1, -1):
        maximum = pool_size - (anchor_count - index)
        ranks[index] = min(ranks[index], maximum)
        if index < anchor_count - 1:
            ranks[index] = min(ranks[index], ranks[index + 1] - 1)
    bins: list[np.ndarray] = []
    for index, rank in enumerate(ranks):
        left = 0 if index == 0 else int((ranks[index - 1] + rank) // 2 + 1)
        right = pool_size - 1 if index == anchor_count - 1 else int((rank + ranks[index + 1]) // 2)
        bins.append(candidate_indices[left:right + 1])
    return bins


def _parallax(previous: np.ndarray | None, current: np.ndarray) -> tuple[float, float]:
    """Return median tracked-feature displacement and surviving track fraction."""
    if previous is None:
        return 0.0, 1.0
    points = cv2.goodFeaturesToTrack(
        previous, maxCorners=160, qualityLevel=0.01, minDistance=5, blockSize=5
    )
    if points is None or len(points) < 8:
        return 0.0, 0.0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if tracked is None or status is None:
        return 0.0, 0.0
    valid = status.reshape(-1).astype(bool)
    if valid.sum() < 6:
        return 0.0, float(valid.mean())
    displacement = np.linalg.norm(tracked[valid] - points[valid], axis=2).reshape(-1)
    return float(np.median(displacement)), float(valid.mean())


def extract_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    target_frames: int | None = None,
    max_width: int = 1920,
    candidates_per_bin: int = 3,
    manifest_path: str | Path | None = None,
    sample_fps: float = 1.0,
    max_frames: int = 1200,
    selection_mode: str = "uniform",
    telemetry_samples: Sequence | None = None,
    decode_mode: str = "seek",
    reuse_existing: bool = False,
    adaptive_screening_timeout_s: float | None = None,
    adaptive_target_candidates: int = 700,
) -> list[FrameRecord]:
    """Select sharp, evenly distributed frames while preserving flight order.

    Each timeline bin contributes one frame. ``uniform`` chooses its sharpest
    candidate. ``geometry`` balances sharpness, GPS baseline, optical-flow
    parallax, and surviving tracks. Sequential order is never changed.
    """
    if selection_mode == "adaptive":
        from .adaptive_selection import extract_adaptive_keyframes
        return extract_adaptive_keyframes(
            video_path, output_dir, manifest_path=manifest_path,
            telemetry_samples=telemetry_samples, target_frames=target_frames,
            max_width=max_width, screening_timeout_s=adaptive_screening_timeout_s,
            target_candidate_count=adaptive_target_candidates,
        )
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if target_frames is not None and target_frames < 10:
        raise ValueError("target_frames must be at least 10")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if max_frames < 10:
        raise ValueError("max_frames must be at least 10")
    if selection_mode not in {"uniform", "geometry"}:
        raise ValueError("selection_mode must be 'uniform' or 'geometry'")
    if decode_mode not in {"seek", "sequential", "ffmpeg"}:
        raise ValueError("decode_mode must be 'seek', 'sequential', or 'ffmpeg'")
    requested_decode_mode = decode_mode
    if selection_mode == "geometry" and not telemetry_samples:
        raise ValueError("geometry selection requires telemetry_samples")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path) if manifest_path else output_dir / "frames.csv"
    metadata_path = manifest.with_suffix(".selection.json")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if total <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError("Video has invalid frame count or frame rate")

    # A fixed frame budget undersamples long flights. When the caller does not
    # force a target, derive it from duration and cap it explicitly so compute
    # remains predictable. One frame/second is a safe draft baseline; operators
    # can increase --sample-fps for faster or lower-altitude flights.
    duration_s = total / fps
    video_fingerprint = _source_file_fingerprint(video_path)
    telemetry_fingerprint = _telemetry_fingerprint(telemetry_samples)
    frame_budget_mode = "fixed"
    if target_frames is None:
        target_frames = min(max_frames, max(10, int(round(duration_s * sample_fps)) + 1))
        frame_budget_mode = "duration_adaptive"

    if reuse_existing and manifest.is_file() and metadata_path.is_file():
        try:
            metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
            with manifest.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            cached = [
                FrameRecord(
                    image_name=row["image_name"],
                    source_frame=int(row["source_frame"]),
                    time_s=float(row["time_s"]),
                    sharpness=float(row["sharpness"]),
                    motion_score=float(row["motion_score"]),
                    baseline_m=float(row.get("baseline_m") or 0.0),
                    parallax_px=float(row.get("parallax_px") or 0.0),
                    tracked_fraction=float(row.get("tracked_fraction") or 0.0),
                    geometry_score=float(row.get("geometry_score") or 0.0),
                )
                for row in rows
            ]
            cache_matches = (
                len(cached) == target_frames
                and metadata.get("cache_schema_version") == KEYFRAME_CACHE_SCHEMA_VERSION
                and metadata.get("selected_frames") == target_frames
                and metadata.get("selection_algorithm_version") == KEYFRAME_SELECTION_ALGORITHM_VERSION
                and metadata.get("keyframe_selection_mode") == selection_mode
                and metadata.get("keyframe_decode_mode_requested", metadata.get("keyframe_decode_mode")) == requested_decode_mode
                and metadata.get("max_width") == max_width
                and metadata.get("source_video_size_bytes") == video_path.stat().st_size
                and metadata.get("source_video_fingerprint") == video_fingerprint
                and metadata.get("telemetry_fingerprint") == telemetry_fingerprint
                and _records_are_strictly_ordered(cached)
                and _cache_has_exact_image_set(output_dir, cached)
            )
            if cache_matches:
                cap.release()
                print(f"Reusing {len(cached)} validated ordered keyframes")
                return cached
        except (KeyError, OSError, TypeError, ValueError):
            # An incomplete/stale cache is never fatal; rebuild it below.
            pass

    # A rejected cache is rebuilt in a clean, dedicated image directory. This
    # prevents caller-provided names from being mixed with regenerated
    # ``frame_*.jpg`` files and counted twice by COLMAP.
    _clear_cached_images(output_dir)

    if selection_mode == "geometry" and telemetry_samples:
        anchors = _hybrid_anchor_indices(total, fps, target_frames, telemetry_samples)
        anchor_policy = "60_percent_gps_arclength_40_percent_time"
    else:
        anchors = np.unique(np.linspace(0, total - 1, target_frames).round().astype(int))
        anchor_policy = "uniform_time"
    if decode_mode == "ffmpeg":
        candidate_count = min(
            total, max(target_frames, target_frames * max(4, candidates_per_bin))
        )
        candidate_indices = np.unique(
            np.linspace(0, total - 1, candidate_count).round().astype(int)
        )
        bins = _candidate_bins_from_uniform_pool(anchors, candidate_indices)
    else:
        bins = _candidate_bins_from_anchors(anchors, total, candidates_per_bin)
        candidate_indices = np.unique(np.concatenate(bins))
    decode_seconds = 0.0
    scoring_seconds = 0.0
    selection_seconds = 0.0
    jpeg_write_seconds = 0.0
    seek_operations = 0
    sequential_grabs = 0
    ffmpeg_candidate_paths: dict[int, Path] = {}
    ffmpeg_decode_backend = None
    ffmpeg_temp_dir: Path | None = None
    if decode_mode == "ffmpeg":
        # FFmpeg's decoder is substantially faster than thousands of OpenCV
        # grab/read operations on long H.264/H.265 videos. Try NVDEC first and
        # transparently fall back to FFmpeg's CPU decoder when unavailable.
        cap.release()
        ffmpeg_temp_dir = Path(tempfile.mkdtemp(prefix="sih_keyframe_candidates_"))
        output_pattern = ffmpeg_temp_dir / "candidate_%06d.jpg"
        # Decode a single uniform candidate stream efficiently. Telemetry-aware
        # allocation happens in ``bins`` above, so fast path sections still
        # receive more output anchors without an O(frames*candidates) filter.
        candidate_fps = len(candidate_indices) / max(duration_s, 1e-6)
        video_filter = (
            f"fps={candidate_fps:.12f},"
            f"scale=w=min({max_width}\\,iw):h=-2"
        )
        base_command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        ]
        attempts = (
            ("nvdec", ["-hwaccel", "cuda"]),
            ("cpu", []),
        )
        ffmpeg_started = time.perf_counter()
        last_error = ""
        candidate_files: list[Path] = []
        for backend, acceleration in attempts:
            for old_candidate in ffmpeg_temp_dir.glob("candidate_*.jpg"):
                old_candidate.unlink()
            completed = subprocess.run(
                [
                    *base_command, *acceleration, "-i", str(video_path),
                    "-vf", video_filter, "-fps_mode", "vfr",
                    "-frames:v", str(len(candidate_indices)),
                    "-q:v", "2", str(output_pattern),
                ],
                capture_output=True, text=True, errors="replace", check=False,
            )
            candidate_files = sorted(ffmpeg_temp_dir.glob("candidate_*.jpg"))
            minimum_expected = max(10, int(math.floor(0.90 * len(candidate_indices))))
            if completed.returncode == 0 and len(candidate_files) >= minimum_expected:
                ffmpeg_decode_backend = backend
                break
            last_error = (completed.stderr or completed.stdout or "")[-2000:]
        decode_seconds += time.perf_counter() - ffmpeg_started
        if ffmpeg_decode_backend is None:
            shutil.rmtree(ffmpeg_temp_dir, ignore_errors=True)
            ffmpeg_temp_dir = None
            decode_mode = "sequential"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(
                    "FFmpeg candidate decoding failed and OpenCV could not open "
                    f"the sequential fallback: {last_error}"
                )
            print(
                "WARNING: FFmpeg candidate extraction failed; using ordered "
                "OpenCV sequential decoding. Telemetry anchor indices are preserved."
            )
        else:
            # Map the ordered FFmpeg outputs monotonically across the requested
            # source indices. Normally this is one-to-one; the mapping also
            # tolerates small timestamp-related output-count differences.
            mapped_positions = np.linspace(
                0, len(candidate_files) - 1, len(candidate_indices)
            ).round().astype(int)
            ffmpeg_candidate_paths = {
                int(source_index): candidate_files[int(file_index)]
                for source_index, file_index in zip(candidate_indices, mapped_positions)
            }
    records: list[FrameRecord] = []
    previous_small: np.ndarray | None = None
    previous_sample = None
    next_sequential_frame = 0
    baseline_target_m = (
        max(0.5, _trajectory_length_m(telemetry_samples) / max(target_frames - 1, 1))
        if selection_mode == "geometry"
        else 0.0
    )

    if telemetry_samples:
        from .telemetry import interpolate

    for output_index, frame_bin in enumerate(bins):
        candidates = []
        for source_index in frame_bin:
            decode_started = time.perf_counter()
            source_index = int(source_index)
            if decode_mode == "ffmpeg":
                frame = cv2.imread(str(ffmpeg_candidate_paths[source_index]))
                ok = frame is not None
            else:
                if decode_mode == "seek":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                    seek_operations += 1
                else:
                    if source_index < next_sequential_frame:
                        cap.release()
                        raise RuntimeError("Sequential candidate indices must be monotonically increasing")
                    while next_sequential_frame < source_index:
                        if not cap.grab():
                            break
                        next_sequential_frame += 1
                        sequential_grabs += 1
                ok, frame = cap.read()
                if decode_mode == "sequential" and ok:
                    next_sequential_frame += 1
            decode_seconds += time.perf_counter() - decode_started
            if not ok:
                continue
            scoring_started = time.perf_counter()
            height, width = frame.shape[:2]
            if width > max_width:
                scale = max_width / width
                frame = cv2.resize(frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            sharpness = _sharpness(gray)
            sample = interpolate(telemetry_samples, int(source_index) / fps) if telemetry_samples else None
            baseline_m = _distance_m(previous_sample, sample) if previous_sample is not None else 0.0
            parallax_px, tracked_fraction = _parallax(previous_small, small)
            scoring_seconds += time.perf_counter() - scoring_started
            candidates.append({
                "sharpness": sharpness,
                "source_index": int(source_index),
                "frame": frame,
                "gray": gray,
                "small": small,
                "sample": sample,
                "baseline_m": baseline_m,
                "parallax_px": parallax_px,
                "tracked_fraction": tracked_fraction,
            })
        if not candidates:
            continue

        selection_started = time.perf_counter()
        maximum_sharpness = max(candidate["sharpness"] for candidate in candidates) or 1.0
        for candidate in candidates:
            sharpness_score = math.log1p(candidate["sharpness"]) / math.log1p(maximum_sharpness)
            if previous_small is None or selection_mode == "uniform":
                geometry_score = sharpness_score
            else:
                baseline_score = min(candidate["baseline_m"] / baseline_target_m, 1.0)
                # Useful temporal stereo has visible displacement but retains
                # enough tracks for overlap. Penalize extremely large flow.
                flow = candidate["parallax_px"]
                parallax_score = min(flow / 8.0, 1.0) * min(35.0 / max(flow, 1e-6), 1.0)
                overlap_score = min(candidate["tracked_fraction"] / 0.65, 1.0)
                geometry_score = (
                    0.40 * sharpness_score
                    + 0.30 * baseline_score
                    + 0.20 * parallax_score
                    + 0.10 * overlap_score
                )
            candidate["geometry_score"] = geometry_score

        best = max(
            candidates,
            key=(
                (lambda candidate: (candidate["geometry_score"], candidate["sharpness"]))
                if selection_mode == "geometry"
                else (lambda candidate: candidate["sharpness"])
            ),
        )
        sharpness = best["sharpness"]
        source_index = best["source_index"]
        frame = best["frame"]
        small = best["small"]
        motion = 0.0 if previous_small is None else float(np.mean(cv2.absdiff(previous_small, small)))
        previous_small = small
        previous_sample = best["sample"]
        image_name = f"frame_{output_index:04d}.jpg"
        selection_seconds += time.perf_counter() - selection_started
        write_started = time.perf_counter()
        if not cv2.imwrite(str(output_dir / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Failed to write {image_name}")
        jpeg_write_seconds += time.perf_counter() - write_started
        records.append(FrameRecord(
            image_name,
            source_index,
            source_index / fps,
            sharpness,
            motion,
            best["baseline_m"],
            best["parallax_px"],
            best["tracked_fraction"],
            best["geometry_score"],
        ))
    cap.release()
    if ffmpeg_temp_dir is not None:
        shutil.rmtree(ffmpeg_temp_dir, ignore_errors=True)
    if len(records) < 10:
        raise RuntimeError(f"Only {len(records)} usable frames were extracted")

    selected_times = np.asarray([record.time_s for record in records], dtype=float)
    selected_time_gaps = np.diff(selected_times)
    path_times, cumulative_path = _telemetry_path(telemetry_samples or [])
    selected_path_gaps = np.asarray([], dtype=float)
    if len(path_times) >= 2 and len(selected_times) >= 2:
        selected_distances = np.interp(selected_times, path_times, cumulative_path)
        selected_path_gaps = np.diff(selected_distances)

    def gap_summary(values: np.ndarray) -> dict:
        return {
            "median": float(np.median(values)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None,
            "maximum": float(np.max(values)) if len(values) else None,
        }

    # Keep non-image files outside the COLMAP image directory. COLMAP scans
    # every entry in that directory and otherwise tries to decode frames.csv.
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(records[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    # Keep selection metadata next to the frame manifest without putting it in
    # COLMAP's image directory.
    metadata_path.write_text(
        __import__("json").dumps({
            # Keep the historical field for report compatibility and record
            # the new selection algorithm separately.
            "selection_mode": frame_budget_mode,
            "cache_schema_version": KEYFRAME_CACHE_SCHEMA_VERSION,
            "selection_algorithm_version": KEYFRAME_SELECTION_ALGORITHM_VERSION,
            "keyframe_selection_mode": selection_mode,
            "keyframe_decode_mode_requested": requested_decode_mode,
            "keyframe_decode_mode": decode_mode,
            "video_duration_s": duration_s,
            "source_fps": fps,
            "requested_sample_fps": sample_fps,
            "max_frames": max_frames,
            "max_width": max_width,
            "source_video_size_bytes": video_path.stat().st_size,
            "source_video_fingerprint": video_fingerprint,
            "telemetry_fingerprint": telemetry_fingerprint,
            "selected_frames": len(records),
            "anchor_count": len(anchors),
            "anchor_policy": anchor_policy,
            "spatial_anchor_fraction": 0.60 if anchor_policy.startswith("60_percent") else 0.0,
            "candidate_frames": len(candidate_indices),
            "seek_operations": seek_operations,
            "sequential_grabbed_frames": sequential_grabs,
            "ffmpeg_decode_backend": ffmpeg_decode_backend,
            "timing_seconds": {
                "decode": round(decode_seconds, 2),
                "score_candidates": round(scoring_seconds, 2),
                "select_candidates": round(selection_seconds, 2),
                "write_selected_jpegs": round(jpeg_write_seconds, 2),
            },
            "effective_sample_fps": len(records) / duration_s if duration_s else None,
            "baseline_target_m": baseline_target_m if selection_mode == "geometry" else None,
            "median_selected_baseline_m": float(np.median([record.baseline_m for record in records[1:]])) if len(records) > 1 else 0.0,
            "median_selected_parallax_px": float(np.median([record.parallax_px for record in records[1:]])) if len(records) > 1 else 0.0,
            "median_tracked_fraction": float(np.median([record.tracked_fraction for record in records[1:]])) if len(records) > 1 else 0.0,
            "selected_time_gap_s": gap_summary(selected_time_gaps),
            "selected_path_gap_m": gap_summary(selected_path_gaps),
        }, indent=2),
        encoding="utf-8",
    )
    return records
KEYFRAME_CACHE_SCHEMA_VERSION = 2
KEYFRAME_SELECTION_ALGORITHM_VERSION = "telemetry_hybrid_v2_cache_identity"


def _source_file_fingerprint(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    """Return a stable, inexpensive identity for a potentially large video.

    The file size and samples from both ends detect replaced/truncated inputs
    without adding a second full-video hash pass to every keyframe operation.
    The final release manifest still records the full SHA-256 digest.
    """
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if size > sample_bytes:
            stream.seek(max(0, size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()


def _telemetry_fingerprint(samples: Sequence | None) -> str | None:
    """Hash the telemetry values that influence geometry frame selection."""
    if not samples:
        return None
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            (
                f"{float(sample.time_s):.9f},"
                f"{float(sample.latitude):.12f},"
                f"{float(sample.longitude):.12f},"
                f"{float(sample.altitude_m):.6f}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def _cache_has_exact_image_set(output_dir: Path, records: Sequence[FrameRecord]) -> bool:
    """Reject caches containing missing, extra, duplicate, or unsafe images."""
    names = [record.image_name for record in records]
    if len(names) != len(set(names)):
        return False
    if any(Path(name).name != name for name in names):
        return False
    expected = set(names)
    actual = {
        path.name for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    return actual == expected


def _records_are_strictly_ordered(records: Sequence[FrameRecord]) -> bool:
    if not records:
        return False
    frames = [record.source_frame for record in records]
    times = [record.time_s for record in records]
    return (
        len(frames) == len(set(frames))
        and all(left < right for left, right in zip(frames, frames[1:]))
        and all(left < right for left, right in zip(times, times[1:]))
    )


def _clear_cached_images(output_dir: Path) -> None:
    """Clear only image files from the dedicated keyframe cache directory."""
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            path.unlink()
