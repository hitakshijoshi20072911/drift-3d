from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from .telemetry import load_telemetry


def _rate(value: object) -> float | None:
    """Parse an ffprobe rational without using eval or accepting infinities."""
    try:
        text = str(value)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            result = float(numerator) / float(denominator)
        else:
            result = float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def video_stream_diagnostics(stream: dict, format_info: dict) -> dict:
    """Return codec/timing/camera limitations that must accompany a release."""
    real_rate = _rate(stream.get("r_frame_rate"))
    average_rate = _rate(stream.get("avg_frame_rate"))
    try:
        duration_s = float(format_info.get("duration"))
    except (TypeError, ValueError):
        duration_s = 0.0
    try:
        frame_count = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = 0
    expected_frames = duration_s * average_rate if average_rate else 0.0
    frame_count_error = (
        abs(frame_count - expected_frames) / expected_frames
        if frame_count and expected_frames else None
    )
    rotation = stream.get("rotation")
    if rotation is None:
        rotation = (stream.get("tags") or {}).get("rotate", 0)
    try:
        rotation = int(float(rotation or 0)) % 360
    except (TypeError, ValueError):
        rotation = 0
    variable = bool(
        real_rate and average_rate
        and abs(real_rate - average_rate) / average_rate > 0.001
    )
    checks = {
        "dimensions_valid": (
            int(stream.get("width") or 0) > 0
            and int(stream.get("height") or 0) > 0
        ),
        "frame_rate_valid": average_rate is not None,
        "duration_valid": duration_s > 0,
        "declared_frame_count_consistent": (
            frame_count_error is None or frame_count_error <= 0.02
        ),
        "codec_declared": bool(stream.get("codec_name")),
        "pixel_format_declared": bool(stream.get("pix_fmt")),
    }
    return {
        "checks": checks,
        "critical_pass": all(checks.values()),
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration_s": duration_s,
        "nominal_frame_rate": real_rate,
        "average_frame_rate": average_rate,
        "variable_frame_rate": variable,
        "timestamp_policy": "use decoded presentation timestamps",
        "rotation_degrees": rotation,
        "rotation_must_be_applied": rotation not in {0},
        "declared_frame_count": frame_count or None,
        "declared_frame_count_relative_error": frame_count_error,
        "camera_limitations": {
            "digital_stabilization": "unknown_without_camera_profile",
            "zoom_state": "unknown_without_camera_profile",
            "rolling_shutter": "unknown_without_camera_profile",
        },
    }


def _distance_m(first, second) -> float:
    radius_m = 6371008.8
    lat1 = math.radians(first.latitude)
    lat2 = math.radians(second.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(second.longitude - first.longitude)
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_m * math.asin(min(1.0, math.sqrt(value)))


def telemetry_diagnostics(samples, video_duration_s: float | None = None) -> dict:
    """Detect timestamp, position, speed, and coverage defects before COLMAP."""
    time_gaps = [
        right.time_s - left.time_s for left, right in zip(samples, samples[1:])
    ]
    horizontal_speeds = []
    vertical_speeds = []
    for left, right, delta in zip(samples, samples[1:], time_gaps):
        if delta > 0:
            horizontal_speeds.append(_distance_m(left, right) / delta)
            vertical_speeds.append(abs(right.altitude_m - left.altitude_m) / delta)
    coordinates_valid = all(
        -90 <= sample.latitude <= 90
        and -180 <= sample.longitude <= 180
        and math.isfinite(sample.altitude_m)
        for sample in samples
    )
    strictly_increasing = all(gap > 0 for gap in time_gaps)
    maximum_gap = max(time_gaps, default=0.0)
    starts_near_video = samples[0].time_s <= 1.0
    covers_video_end = bool(
        video_duration_s is None or samples[-1].time_s >= 0.90 * video_duration_s
    )
    start_offset_s = samples[0].time_s
    end_offset_s = (
        samples[-1].time_s - video_duration_s
        if video_duration_s is not None else None
    )
    checks = {
        "coordinates_valid": coordinates_valid,
        "timestamps_strictly_increasing": strictly_increasing,
        "maximum_gap_at_most_5s": maximum_gap <= 5.0,
        "starts_within_first_second": starts_near_video,
        "covers_at_least_90_percent_of_video": covers_video_end,
        "horizontal_speed_below_80mps": max(horizontal_speeds, default=0.0) <= 1000.0,
        "vertical_speed_below_30mps": max(vertical_speeds, default=0.0) <= 500.0,
    }
    return {
        "checks": checks,
        "critical_pass": all(checks.values()),
        "maximum_time_gap_s": maximum_gap,
        "maximum_horizontal_speed_mps": max(horizontal_speeds, default=0.0),
        "maximum_vertical_speed_mps": max(vertical_speeds, default=0.0),
        "video_duration_s": video_duration_s,
        "telemetry_duration_s": samples[-1].time_s - samples[0].time_s,
        "time_alignment": {
            "telemetry_start_minus_video_start_s": start_offset_s,
            "telemetry_end_minus_video_end_s": end_offset_s,
            "coarse_timeline_pass": starts_near_video and covers_video_end,
            "motion_correlation_status": "measured_later_from_selected_frames",
        },
        "altitude_datums": sorted({
            sample.altitude_datum for sample in samples if sample.altitude_datum
        }),
        "altitude_datum_status": (
            "declared" if any(sample.altitude_datum for sample in samples)
            else "unknown"
        ),
    }


def _command_version(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    if not executable:
        return {"available": False, "executable": None, "output": None}
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    return {"available": result.returncode == 0, "executable": executable, "output": "\n".join(output.splitlines()[:12])}


def _command_help(command: list[str]) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if not executable:
        return False, ""
    result = subprocess.run(
        [executable, *command[1:]],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    return result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def inspect_environment(
    video: str | Path,
    telemetry_path: str | Path,
    workspace: str | Path,
    camera_calibration: str | Path | None = None,
) -> dict:
    video, telemetry_path, workspace = Path(video), Path(telemetry_path), Path(workspace)
    checks = {
        "colmap": _command_version(["colmap", "-h"]),
        "ffmpeg": _command_version(["ffmpeg", "-version"]),
        "ffprobe": _command_version(["ffprobe", "-version"]),
        "nvidia_smi": _command_version(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
    }
    if checks["colmap"]["available"]:
        feature_ok, feature_help = _command_help(["colmap", "feature_extractor", "--help"])
        matcher_ok, matcher_help = _command_help(["colmap", "sequential_matcher", "--help"])
        advancing_ok, _ = _command_help(["colmap", "advancing_front_mesher", "--help"])
        global_ok, _ = _command_help(["colmap", "global_mapper", "--help"])
        calibrator_ok, _ = _command_help(["colmap", "view_graph_calibrator", "--help"])
        simplifier_ok, _ = _command_help(["colmap", "mesh_simplifier", "--help"])
        checks["colmap_capabilities"] = {
            "feature_type_selection": feature_ok and "FeatureExtraction.type" in feature_help,
            "matcher_type_selection": matcher_ok and "FeatureMatching.type" in matcher_help,
            "aliked": feature_ok and "ALIKED" in feature_help.upper(),
            "lightglue": matcher_ok and "LIGHTGLUE" in matcher_help.upper(),
            "advancing_front_mesher": advancing_ok,
            "global_mapper": global_ok,
            "view_graph_calibrator": calibrator_ok,
            "mesh_simplifier": simplifier_ok,
            "note": "ALIKED and LightGlue require ONNX support in the installed COLMAP build.",
        }
    else:
        checks["colmap_capabilities"] = {
            "feature_type_selection": False,
            "matcher_type_selection": False,
            "aliked": False,
            "lightglue": False,
            "advancing_front_mesher": False,
            "global_mapper": False,
            "view_graph_calibrator": False,
            "mesh_simplifier": False,
        }
    video_info = {}
    if video.is_file() and checks["ffprobe"]["available"]:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt:stream_tags=rotate:stream_side_data=rotation:format=duration,size",
                "-of", "json", str(video),
            ],
            capture_output=True, text=True, errors="replace", check=False,
        )
        if probe.returncode == 0:
            video_info = json.loads(probe.stdout)
    duration_s = None
    try:
        duration_s = float(video_info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        pass
    telemetry_info = {"valid": False}
    try:
        samples = load_telemetry(telemetry_path)
        from .capture_quality import telemetry_quality

        telemetry_info = {
            "valid": True,
            "samples": len(samples),
            "start_s": samples[0].time_s,
            "end_s": samples[-1].time_s,
            "latitude_range": [min(s.latitude for s in samples), max(s.latitude for s in samples)],
            "longitude_range": [min(s.longitude for s in samples), max(s.longitude for s in samples)],
            "altitude_range_m": [min(s.altitude_m for s in samples), max(s.altitude_m for s in samples)],
            "quality": telemetry_quality(samples),
            "diagnostics": telemetry_diagnostics(samples, duration_s),
        }
    except Exception as error:
        telemetry_info["error"] = str(error)
    workspace.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(workspace)
    checks["video"] = {"available": video.is_file(), "path": str(video), "probe": video_info}
    checks["telemetry"] = telemetry_info
    video_size_bytes = video.stat().st_size if video.is_file() else 0
    estimated_required_bytes = max(10 * 1024**3, video_size_bytes * 8)
    checks["workspace"] = {
        "path": str(workspace),
        "free_gb": round(disk.free / 1024**3, 2),
        "estimated_required_gb": round(estimated_required_bytes / 1024**3, 2),
        "capacity_pass": disk.free >= estimated_required_bytes,
    }
    colmap_header = (checks["colmap"].get("output") or "").upper()
    colmap_cuda = "CUDA" in colmap_header and "WITHOUT CUDA" not in colmap_header and "NO CUDA" not in colmap_header
    telemetry_covers_video = bool(
        duration_s is not None
        and telemetry_info.get("valid")
        and telemetry_info.get("start_s", 1) <= 1.0
        and telemetry_info.get("end_s", 0) >= 0.9 * duration_s
    )
    stream = (video_info.get("streams") or [{}])[0]
    format_info = video_info.get("format") or {}
    checks["video_stream"] = video_stream_diagnostics(stream, format_info)
    camera_info = {
        "provided": camera_calibration is not None,
        "valid": camera_calibration is None,
        "limitations_declared": False,
        "profile": None,
    }
    if camera_calibration is not None:
        try:
            from .camera import calibration_for_image_size, load_camera_calibration

            profile = load_camera_calibration(camera_calibration)
            profile = calibration_for_image_size(
                profile,
                int(stream.get("width") or 0),
                int(stream.get("height") or 0),
            )
            camera_info.update({
                "valid": True,
                "limitations_declared": all(
                    profile.get(key) is not None
                    for key in (
                        "digital_stabilization",
                        "rolling_shutter_readout_ms",
                    )
                ),
                "profile": profile,
            })
        except Exception as error:
            camera_info["error"] = str(error)
    checks["camera"] = camera_info
    checks["ready"] = all([
        checks["colmap"]["available"], checks["ffmpeg"]["available"], checks["ffprobe"]["available"],
        checks["nvidia_smi"]["available"],
        checks["video"]["available"], checks["telemetry"]["valid"], telemetry_covers_video,
        colmap_cuda,
        checks["workspace"]["capacity_pass"],
        telemetry_info.get("diagnostics", {}).get("critical_pass", False),
        checks["video_stream"]["critical_pass"],
        camera_info["valid"],
    ])
    checks["colmap_cuda"] = colmap_cuda
    checks["telemetry_covers_video"] = telemetry_covers_video
    checks["video_timing"] = {
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "variable_frame_rate_review_required": checks["video_stream"][
            "variable_frame_rate"
        ],
        "timestamp_policy": "decoded_presentation_timestamps",
    }
    return checks
