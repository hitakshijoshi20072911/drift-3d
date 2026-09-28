from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .utils import write_json


PROTECTED_FULL_VIDEO_BASELINE = {
    "name": "accepted_full_253_frame_run",
    "video_duration_s": 251.517933,
    "video_duration_tolerance_s": 0.5,
    "target_frames": 253,
    "selected_frames": 253,
    "frame_resolution": [1920, 1080],
    "reference_runtime_seconds": round(26.92 * 60, 2),
    "reference_runtime_minutes": 26.92,
    "reference_runtime_tolerance_seconds": 0.6,
    "principle": (
        "A faster configuration does not replace the accepted full-video baseline unless "
        "it uses the same workload and independent surface evidence."
    ),
}

SURFACE_DISTANCE_ABSOLUTE_TOLERANCE_M = 0.02
SURFACE_DISTANCE_RELATIVE_TOLERANCE = 0.05
DENSE_F1_ABSOLUTE_TOLERANCE = 0.02


def _read_report(output_dir: str | Path) -> dict:
    path = Path(output_dir) / "run_report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _get(payload: dict, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first(payload: dict, paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value = _get(payload, *path)
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_int(value: Any) -> int | None:
    number = _number(value)
    if number is None or number <= 0 or not number.is_integer():
        return None
    return int(number)


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _normalise_digest(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    return digest


def _normalise_manifest_value(field: str, value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if field == "image_name":
        return Path(text.replace("\\", "/")).name
    if field in {"source_frame", "source_index"}:
        number = _number(text)
        return str(int(number)) if number is not None and number.is_integer() else text
    if field == "time_s":
        number = _number(text)
        return format(number, ".12g") if number is not None else text
    return text


def _manifest_metadata(directory: Path, explicit_digest: Any = None) -> dict:
    """Fingerprint frame identity, excluding quality scores that may be recomputed.

    Older outputs already contain ``frames.csv`` but no stored digest.  Hashing the
    image name, source-frame index and timestamp lets those reports participate in
    strict A/B comparisons without requiring their run reports to be rewritten.
    """
    path = directory / "frames.csv"
    reported_digest = _normalise_digest(explicit_digest)
    if not path.is_file():
        return {
            "sha256": reported_digest,
            "selected_frames": None,
            "first_time_s": None,
            "last_time_s": None,
            "source": "report" if explicit_digest else None,
            "digest_consistent": True,
        }

    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    canonical = []
    for row in rows:
        identity = {}
        if "image_name" in row:
            identity["image_name"] = _normalise_manifest_value(
                "image_name", row.get("image_name")
            )
        source_value = row.get("source_index", row.get("source_frame"))
        if source_value is not None:
            # ``source_frame`` was renamed to ``source_index``; normalize both
            # schemas so unchanged historical manifests still compare equally.
            identity["source_frame"] = _normalise_manifest_value(
                "source_frame", source_value
            )
        if "time_s" in row:
            identity["time_s"] = _normalise_manifest_value("time_s", row.get("time_s"))
        if not identity:
            identity = {
                field: _normalise_manifest_value(field, value)
                for field, value in sorted(row.items())
            }
        canonical.append(identity)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    computed_digest = hashlib.sha256(payload).hexdigest()
    times = [
        number for number in (_number(row.get("time_s")) for row in rows)
        if number is not None
    ]
    return {
        "sha256": computed_digest,
        "selected_frames": len(rows),
        "first_time_s": min(times) if times else None,
        "last_time_s": max(times) if times else None,
        "source": "frames.csv",
        "digest_consistent": reported_digest is None or reported_digest == computed_digest,
    }


def _coerce_resolution(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict):
        width = _positive_int(
            value.get("width_px", value.get("width", value.get("frame_width_px")))
        )
        height = _positive_int(
            value.get("height_px", value.get("height", value.get("frame_height_px")))
        )
        return (width, height) if width is not None and height is not None else None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = (_positive_int(part) for part in value)
        return (width, height) if width is not None and height is not None else None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _reported_resolutions(report: dict, metadata: dict) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []

    direct = _coerce_resolution(
        _first(
            metadata,
            (
                ("frame_resolution",),
                ("input_resolution",),
                ("resolution",),
            ),
        )
    )
    if direct is not None:
        values.append(direct)

    width = _positive_int(
        _first(
            metadata,
            (("frame_width_px",), ("image_width_px",), ("width_px",), ("width",)),
        )
    )
    height = _positive_int(
        _first(
            metadata,
            (("frame_height_px",), ("image_height_px",), ("height_px",), ("height",)),
        )
    )
    if width is not None and height is not None:
        values.append((width, height))

    for path in (
        ("input", "frame_resolution"),
        ("input", "resolution"),
        ("input", "camera_calibration"),
    ):
        result = _coerce_resolution(_get(report, *path))
        if result is not None:
            values.append(result)

    input_width = _positive_int(
        _first(report, (("input", "frame_width_px"), ("input", "image_width_px")))
    )
    input_height = _positive_int(
        _first(report, (("input", "frame_height_px"), ("input", "image_height_px")))
    )
    if input_width is not None and input_height is not None:
        values.append((input_width, input_height))

    # Historical reports include COLMAP's actual decoded dimensions in the
    # feature-extraction log tail even when no camera profile was supplied.
    log_tail = _get(report, "stages", "feature_extraction", "log_tail")
    if isinstance(log_tail, str):
        matches = {
            (int(width_value), int(height_value))
            for width_value, height_value in re.findall(
                r"Dimensions:\s*(\d+)\s*x\s*(\d+)", log_tail
            )
        }
        if len(matches) == 1:
            values.append(next(iter(matches)))
    return values


def _video_identity(report: dict, metadata: dict) -> dict:
    digest = _normalise_digest(
        _first(
            metadata,
            (("video_sha256",), ("video_hash",), ("input_video_sha256",)),
        )
        or _first(report, (("input", "video_sha256"), ("input", "video_hash")))
    )
    if digest:
        return {"value": f"sha256:{digest}", "basis": "sha256", "strong": True}

    identifier = _first(
        metadata,
        (("video_id",), ("video_identity",), ("dataset_video_id",), ("dataset_id",)),
    ) or _first(report, (("input", "video_id"), ("input", "dataset_video_id")))
    if identifier is not None and str(identifier).strip():
        return {
            "value": str(identifier).strip().casefold(),
            "basis": "reported_identifier",
            "strong": True,
        }

    video_path = _get(report, "input", "video")
    if isinstance(video_path, str) and video_path.strip():
        # Colab/workstation roots often differ for the same downloaded asset.
        # The exact selected-frame digest and duration remain separate gates.
        return {
            "value": Path(video_path.replace("\\", "/")).name.casefold(),
            "basis": "legacy_path_name",
            "strong": False,
        }
    return {"value": None, "basis": None, "strong": False}


def _dense_reference_identity(dense: dict) -> tuple[str | None, str | None]:
    reference = dense.get("reference")
    if not isinstance(reference, dict):
        return None, None
    digest = _normalise_digest(reference.get("sha256"))
    if digest:
        return f"sha256:{digest}", "sha256"
    identifier = reference.get("id", reference.get("identity"))
    if identifier is not None and str(identifier).strip():
        return str(identifier).strip().casefold(), "reported_identifier"
    path = reference.get("path")
    if isinstance(path, str) and path.strip():
        return path.strip().replace("\\", "/").casefold(), "reported_path"
    return None, None


def _valid_distance_summary(value: Any) -> bool:
    if not isinstance(value, dict) or _positive_int(value.get("samples")) is None:
        return False
    required = ("mean_m", "median_m", "rmse_m", "p95_m", "maximum_m")
    values = [_number(value.get(key)) for key in required]
    return all(number is not None and number >= 0 for number in values)


def _dense_geometry_evidence(dense: dict) -> dict:
    threshold_rows = dense.get("threshold_metrics")
    metric_thresholds = []
    threshold_quality = []
    metrics_valid = isinstance(threshold_rows, list) and bool(threshold_rows)
    if metrics_valid:
        for row in threshold_rows:
            threshold = _number(row.get("threshold_m")) if isinstance(row, dict) else None
            precision = _number(row.get("precision")) if isinstance(row, dict) else None
            completeness = (
                _number(row.get("completeness", row.get("recall")))
                if isinstance(row, dict)
                else None
            )
            f1 = _number(row.get("f1")) if isinstance(row, dict) else None
            if (
                threshold is None
                or threshold <= 0
                or precision is None
                or completeness is None
                or f1 is None
                or any(value < 0 or value > 1 for value in (precision, completeness, f1))
            ):
                metrics_valid = False
                break
            metric_thresholds.append(threshold)
            threshold_quality.append({"threshold_m": threshold, "f1": f1})

    protocol = dense.get("protocol") if isinstance(dense.get("protocol"), dict) else {}
    protocol_thresholds = protocol.get("thresholds_m")
    thresholds = list(metric_thresholds)
    if isinstance(protocol_thresholds, (list, tuple)) and protocol_thresholds:
        normalized_protocol_thresholds = [_number(value) for value in protocol_thresholds]
        if all(value is not None and value > 0 for value in normalized_protocol_thresholds):
            thresholds = [float(value) for value in normalized_protocol_thresholds]
            if sorted(thresholds) != sorted(metric_thresholds):
                metrics_valid = False
        else:
            metrics_valid = False

    distances = dense.get("distances_m")
    forward = distances.get("reconstruction_to_reference") if isinstance(distances, dict) else None
    reverse = distances.get("reference_to_reconstruction") if isinstance(distances, dict) else None
    directions_valid = _valid_distance_summary(forward) and _valid_distance_summary(reverse)
    reference_identity, reference_basis = _dense_reference_identity(dense)
    max_samples = _positive_int(dense.get("max_samples"))
    seed = _integer(dense.get("seed"))
    evaluation_protocol = {
        "name": protocol.get("name"),
        "version": protocol.get("version"),
        "thresholds_m": sorted(set(thresholds)),
        "mesh_sampling": protocol.get("mesh_sampling"),
        "alignment": dense.get("alignment"),
        "max_samples": max_samples,
        "seed": seed,
    }
    protocol_valid = bool(thresholds and max_samples is not None and seed is not None)
    reference_samples = _positive_int(reverse.get("samples")) if isinstance(reverse, dict) else None
    return {
        "type": "dense_reference_geometry",
        "checks": reference_samples,
        "valid": bool(
            metrics_valid
            and directions_valid
            and protocol_valid
            and reference_identity is not None
            and dense.get("reference_is_distinct_from_reconstruction") is not False
        ),
        "set_identity": reference_identity,
        "set_identity_basis": reference_basis,
        "evaluation_protocol": evaluation_protocol,
        "quality": {
            "passes_one_metre_target": dense.get("passes_one_metre_target"),
            "threshold_f1": sorted(threshold_quality, key=lambda row: row["threshold_m"]),
            "reconstruction_to_reference": {
                "rmse_m": _number(forward.get("rmse_m")) if isinstance(forward, dict) else None,
                "p95_m": _number(forward.get("p95_m")) if isinstance(forward, dict) else None,
                "maximum_m": _number(forward.get("maximum_m")) if isinstance(forward, dict) else None,
            },
            "reference_to_reconstruction": {
                "rmse_m": _number(reverse.get("rmse_m")) if isinstance(reverse, dict) else None,
                "p95_m": _number(reverse.get("p95_m")) if isinstance(reverse, dict) else None,
                "maximum_m": _number(reverse.get("maximum_m")) if isinstance(reverse, dict) else None,
            },
        },
    }


def _surface_evidence(report: dict, metadata: dict) -> dict:
    candidates = []
    dense = _get(report, "validation", "dense_geometry")
    if isinstance(dense, dict):
        dense_evidence = _dense_geometry_evidence(dense)
        if dense_evidence["valid"]:
            return dense_evidence
        candidates.append(dense_evidence)

    checkpoints = _get(report, "validation", "surveyed_checkpoints")
    if isinstance(checkpoints, dict):
        checks = _positive_int(checkpoints.get("checks"))
        metric = _number(checkpoints.get("rmse_3d_m"))
        minimum = _positive_int(checkpoints.get("minimum_checks_required")) or 3
        identifiers = checkpoints.get("checkpoint_ids")
        if not isinstance(identifiers, list):
            per_checkpoint = checkpoints.get("per_checkpoint_3d_error_m")
            identifiers = list(per_checkpoint) if isinstance(per_checkpoint, dict) else None
        identifier_digest = None
        if identifiers:
            normalized = sorted(str(identifier).strip() for identifier in identifiers)
            identifier_digest = hashlib.sha256(
                json.dumps(normalized, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        explicit_set = _normalise_digest(
            _first(
                metadata,
                (("surface_evidence_sha256",), ("checkpoint_set_sha256",)),
            )
        )
        checkpoint_evidence = {
            "type": "surveyed_checkpoints",
            "checks": checks,
            "valid": bool(checks is not None and checks >= minimum and metric is not None),
            "set_identity": explicit_set or identifier_digest,
            "set_identity_basis": "sha256" if explicit_set else "checkpoint_ids" if identifier_digest else "count",
            "quality": {
                "passes_one_metre_target": checkpoints.get("passes_one_metre_target"),
                "rmse_m": metric,
                "maximum_m": _number(checkpoints.get("maximum_3d_error_m")),
            },
        }
        if checkpoint_evidence["valid"]:
            return checkpoint_evidence
        candidates.append(checkpoint_evidence)

    distances = _get(report, "validation", "independent_distances")
    if isinstance(distances, dict):
        checks = _positive_int(distances.get("checks"))
        metric = _number(distances.get("distance_rmse_m"))
        explicit_set = _normalise_digest(
            _first(
                metadata,
                (("surface_evidence_sha256",), ("distance_set_sha256",)),
            )
        )
        distance_evidence = {
            "type": "independent_distances",
            "checks": checks,
            "valid": bool(checks is not None and metric is not None),
            "set_identity": explicit_set,
            "set_identity_basis": "sha256" if explicit_set else "count",
            "quality": {
                "passes_one_metre_target": distances.get("passes_one_metre_target"),
                "rmse_m": metric,
                "maximum_m": _number(distances.get("maximum_absolute_error_m")),
            },
        }
        if distance_evidence["valid"]:
            return distance_evidence
        candidates.append(distance_evidence)

    return candidates[0] if candidates else {
        "type": None,
        "checks": None,
        "valid": False,
        "set_identity": None,
        "set_identity_basis": None,
        "quality": None,
    }


def extract_comparison_metadata(report: dict, output_dir: str | Path) -> dict:
    """Extract normalized workload/evidence metadata from new or legacy reports.

    Normalized third-party reports may provide fields in ``comparison_metadata``
    (or the legacy alias ``benchmark_metadata``). Native legacy outputs are
    enriched from ``frames.csv`` and COLMAP's recorded feature-extraction log.
    """
    directory = Path(output_dir)
    metadata: dict = {}
    for key in ("benchmark_metadata", "comparison_metadata"):
        block = report.get(key)
        if isinstance(block, dict):
            metadata.update(block)
    selection_path = directory / "frames.selection.json"
    selection = {}
    if selection_path.is_file():
        try:
            payload = json.loads(selection_path.read_text(encoding="utf-8"))
            selection = payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            selection = {}

    explicit_manifest_digest = _first(
        metadata,
        (("frame_manifest_sha256",), ("selected_frame_manifest_sha256",)),
    ) or _first(
        report,
        (
            ("input", "frame_manifest_sha256"),
            ("input", "selected_frame_manifest_sha256"),
        ),
    )
    manifest = _manifest_metadata(directory, explicit_manifest_digest)
    reported_counts = [
        _positive_int(value)
        for value in (
            _first(metadata, (("selected_frames",), ("input_frames",))),
            _get(report, "input", "frames"),
            _get(report, "sparse_metrics", "input_frames"),
            _get(report, "workflow_steps", "keyframe_extraction", "selected_frames"),
            _get(report, "workflow_steps", "keyframe_extraction", "frames"),
            selection.get("selected_frames"),
        )
    ]
    reported_counts = [value for value in reported_counts if value is not None]
    count_values = set(reported_counts)
    if manifest["selected_frames"] is not None:
        count_values.add(manifest["selected_frames"])
    selected_frames = (
        manifest["selected_frames"]
        if manifest["selected_frames"] is not None
        else reported_counts[0] if reported_counts else None
    )

    duration_values = [
        _number(value)
        for value in (
            _first(metadata, (("video_duration_s",), ("input_duration_s",))),
            _get(report, "input", "video_duration_s"),
            _get(report, "workflow_steps", "keyframe_extraction", "video_duration_s"),
            selection.get("video_duration_s"),
        )
    ]
    duration_values = [value for value in duration_values if value is not None and value > 0]
    video_duration = duration_values[0] if duration_values else None
    duration_consistent = not duration_values or all(
        math.isclose(value, duration_values[0], rel_tol=1e-4, abs_tol=0.5)
        for value in duration_values[1:]
    )

    target_values = [
        _positive_int(value)
        for value in (
            _first(
                metadata,
                (("target_frames",), ("requested_target_frames",), ("frame_budget",)),
            ),
            _get(report, "input", "target_frames"),
            _get(report, "experiment", "target_frames"),
            _get(report, "workflow_steps", "keyframe_extraction", "target_frames"),
            selection.get("target_frames"),
        )
    ]
    target_values = [value for value in target_values if value is not None]
    target_frames = target_values[0] if target_values else None
    # Legacy selection reports did not retain the resolved target. In fixed
    # mode, the selected count is its best recoverable value. ``max_frames`` is
    # deliberately not used: it is only a safety cap, not the requested target.
    frame_budget_mode = _get(
        report, "workflow_steps", "keyframe_extraction", "selection_mode"
    ) or selection.get("selection_mode")
    if target_frames is None and frame_budget_mode == "fixed":
        target_frames = selected_frames
    resolutions = _reported_resolutions(report, metadata)
    resolution = resolutions[0] if resolutions else None
    video = _video_identity(report, metadata)
    evidence = _surface_evidence(report, metadata)
    consistency_issues = []
    if len(count_values) > 1:
        consistency_issues.append("selected_frame_count_conflict")
    if not duration_consistent:
        consistency_issues.append("video_duration_conflict")
    if len(set(target_values)) > 1:
        consistency_issues.append("target_frame_count_conflict")
    if len(set(resolutions)) > 1:
        consistency_issues.append("frame_resolution_conflict")
    if not manifest["digest_consistent"]:
        consistency_issues.append("selected_frame_manifest_digest_conflict")
    return {
        "video_identity": video["value"],
        "video_identity_basis": video["basis"],
        "video_identity_strong": video["strong"],
        "video_duration_s": video_duration,
        "target_frames": target_frames,
        "selected_frames": selected_frames,
        "selected_frame_manifest_sha256": manifest["sha256"],
        "selected_frame_manifest_source": manifest["source"],
        "selected_time_start_s": manifest["first_time_s"],
        "selected_time_end_s": manifest["last_time_s"],
        "frame_resolution": list(resolution) if resolution is not None else None,
        "independent_surface_evidence": evidence,
        "metadata_consistent": not consistency_issues,
        "consistency_issues": consistency_issues,
    }


def _same_duration(before: Any, after: Any) -> bool:
    first, second = _number(before), _number(after)
    return bool(
        first is not None
        and second is not None
        and first > 0
        and second > 0
        and math.isclose(first, second, rel_tol=1e-4, abs_tol=0.5)
    )


def matches_protected_full_video_workload(metadata: dict) -> bool:
    """Match every fixed workload field of the accepted four-minute run."""
    duration = _number(metadata.get("video_duration_s"))
    expected_duration = float(PROTECTED_FULL_VIDEO_BASELINE["video_duration_s"])
    duration_tolerance = float(
        PROTECTED_FULL_VIDEO_BASELINE["video_duration_tolerance_s"]
    )
    return bool(
        duration is not None
        and math.isclose(duration, expected_duration, rel_tol=0.0, abs_tol=duration_tolerance)
        and metadata.get("target_frames") == PROTECTED_FULL_VIDEO_BASELINE["target_frames"]
        and metadata.get("selected_frames") == PROTECTED_FULL_VIDEO_BASELINE["selected_frames"]
        and metadata.get("frame_resolution")
        == PROTECTED_FULL_VIDEO_BASELINE["frame_resolution"]
    )


def matches_protected_accepted_result(metadata: dict, runtime_seconds: Any) -> bool:
    runtime = _number(runtime_seconds)
    return bool(
        matches_protected_full_video_workload(metadata)
        and runtime is not None
        and math.isclose(
            runtime,
            float(PROTECTED_FULL_VIDEO_BASELINE["reference_runtime_seconds"]),
            rel_tol=0.0,
            abs_tol=float(
                PROTECTED_FULL_VIDEO_BASELINE["reference_runtime_tolerance_seconds"]
            ),
        )
    )
def _same_surface_evidence(before: dict, after: dict) -> bool:
    if not before.get("valid") or not after.get("valid"):
        return False
    if before.get("type") != after.get("type") or before.get("checks") != after.get("checks"):
        return False
    if before.get("type") == "dense_reference_geometry":
        return bool(
            before.get("set_identity") is not None
            and before.get("set_identity") == after.get("set_identity")
            and before.get("evaluation_protocol") == after.get("evaluation_protocol")
        )
    before_identity = before.get("set_identity")
    after_identity = after.get("set_identity")
    if before_identity is not None or after_identity is not None:
        return bool(before_identity is not None and before_identity == after_identity)
    # Historical independent-distance reports did not retain measurement IDs.
    # Equal type/count is the strongest comparison available for those reports.
    return True


def _not_materially_higher(before: Any, after: Any) -> bool:
    baseline, candidate = _number(before), _number(after)
    if baseline is None or candidate is None:
        return False
    allowance = max(
        SURFACE_DISTANCE_ABSOLUTE_TOLERANCE_M,
        abs(baseline) * SURFACE_DISTANCE_RELATIVE_TOLERANCE,
    )
    return bool(candidate <= baseline + allowance + 1e-12)


def _checkpoint_surface_quality(evidence: dict) -> tuple[bool, float | None, float | None]:
    quality = evidence.get("quality") if isinstance(evidence.get("quality"), dict) else {}
    rmse = _number(quality.get("rmse_m"))
    maximum = _number(quality.get("maximum_m"))
    reported_pass = quality.get("passes_one_metre_target")
    passes_target = bool(
        maximum is not None
        and maximum <= 1.0
        and (reported_pass is not False)
    )
    return passes_target, rmse, maximum


def surface_quality_report(before: dict, after: dict) -> dict:
    """Ensure a fair candidate also retains independently measured quality."""
    before_evidence = before["independent_surface_evidence"]
    after_evidence = after["independent_surface_evidence"]
    same_evidence = _same_surface_evidence(before_evidence, after_evidence)
    evidence_type = before_evidence.get("type") if same_evidence else None

    if evidence_type in {"surveyed_checkpoints", "independent_distances"}:
        _, before_rmse, before_maximum = _checkpoint_surface_quality(before_evidence)
        candidate_target, after_rmse, after_maximum = _checkpoint_surface_quality(after_evidence)
        gates = {
            "same_surface_evidence_for_quality": True,
            "candidate_passes_one_metre_surface_target": candidate_target,
            "surface_rmse_not_materially_worse": _not_materially_higher(
                before_rmse, after_rmse
            ),
            "surface_maximum_error_not_materially_worse": _not_materially_higher(
                before_maximum, after_maximum
            ),
        }
    elif evidence_type == "dense_reference_geometry":
        before_quality = before_evidence.get("quality", {})
        after_quality = after_evidence.get("quality", {})
        candidate_target = after_quality.get("passes_one_metre_target")
        if candidate_target is None:
            candidate_target = all(
                _number(after_quality.get(direction, {}).get("maximum_m")) is not None
                and _number(after_quality[direction]["maximum_m"]) <= 1.0
                for direction in (
                    "reconstruction_to_reference", "reference_to_reconstruction"
                )
            )
        before_f1 = {
            _number(row.get("threshold_m")): _number(row.get("f1"))
            for row in before_quality.get("threshold_f1", [])
            if isinstance(row, dict)
        }
        after_f1 = {
            _number(row.get("threshold_m")): _number(row.get("f1"))
            for row in after_quality.get("threshold_f1", [])
            if isinstance(row, dict)
        }
        f1_retained = bool(
            before_f1
            and set(before_f1) == set(after_f1)
            and all(
                before_f1[threshold] is not None
                and after_f1[threshold] is not None
                and after_f1[threshold] + DENSE_F1_ABSOLUTE_TOLERANCE
                >= before_f1[threshold]
                for threshold in before_f1
            )
        )
        gates = {
            "same_surface_evidence_for_quality": True,
            "candidate_passes_one_metre_surface_target": candidate_target is True,
            "dense_f1_at_all_thresholds_not_materially_worse": f1_retained,
        }
        for direction in ("reconstruction_to_reference", "reference_to_reconstruction"):
            before_direction = before_quality.get(direction, {})
            after_direction = after_quality.get(direction, {})
            gates[f"{direction}_rmse_not_materially_worse"] = _not_materially_higher(
                before_direction.get("rmse_m"), after_direction.get("rmse_m")
            )
            gates[f"{direction}_p95_not_materially_worse"] = _not_materially_higher(
                before_direction.get("p95_m"), after_direction.get("p95_m")
            )
    else:
        gates = {"same_surface_evidence_for_quality": False}

    failures = [name for name, passed in gates.items() if not passed]
    return {
        "gates": gates,
        "passes": not failures,
        "failures": failures,
        "tolerances": {
            "distance_absolute_m": SURFACE_DISTANCE_ABSOLUTE_TOLERANCE_M,
            "distance_relative_fraction": SURFACE_DISTANCE_RELATIVE_TOLERANCE,
            "dense_f1_absolute": DENSE_F1_ABSOLUTE_TOLERANCE,
        },
        "policy": (
            "Checkpoint/distance RMSE and maximum error may increase by at most the larger "
            "of 0.02 m or 5%, and the candidate must retain the one-metre target. Dense "
            "directional RMSE/p95 use the same tolerance and F1 may drop by at most 0.02 "
            "at every locked threshold."
        ),
    }


def fair_comparison_report(before: dict, after: dict) -> dict:
    """Return strict workload and independent-evidence gates for an A/B run."""
    before_evidence = before["independent_surface_evidence"]
    after_evidence = after["independent_surface_evidence"]
    gates = {
        "baseline_metadata_consistent": before.get("metadata_consistent") is True,
        "candidate_metadata_consistent": after.get("metadata_consistent") is True,
        "same_video_identity": bool(
            before.get("video_identity") is not None
            and before.get("video_identity") == after.get("video_identity")
        ),
        "same_video_duration": _same_duration(
            before.get("video_duration_s"), after.get("video_duration_s")
        ),
        "same_target_frame_count": bool(
            before.get("target_frames") is not None
            and before.get("target_frames") == after.get("target_frames")
        ),
        "same_selected_frame_count": bool(
            before.get("selected_frames") is not None
            and before.get("selected_frames") == after.get("selected_frames")
        ),
        "same_selected_frame_manifest": bool(
            before.get("selected_frame_manifest_sha256") is not None
            and before.get("selected_frame_manifest_sha256")
            == after.get("selected_frame_manifest_sha256")
        ),
        "same_frame_resolution": bool(
            before.get("frame_resolution") is not None
            and before.get("frame_resolution") == after.get("frame_resolution")
        ),
        "baseline_has_independent_surface_evidence": before_evidence.get("valid") is True,
        "candidate_has_independent_surface_evidence": after_evidence.get("valid") is True,
        "same_independent_surface_evidence_set": _same_surface_evidence(
            before_evidence, after_evidence
        ),
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "gates": gates,
        "passes": not failures,
        "failures": failures,
        "policy": (
            "A candidate may replace the baseline only when video identity/duration, target "
            "and exact selected-frame manifest, frame resolution, and independent surface "
            "evidence are all fixed. Missing metadata fails closed."
        ),
    }


def _change(before: Any, after: Any) -> dict:
    result = {"before": before, "after": after, "delta": None, "change_percent": None}
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        result["delta"] = after - before
        if before:
            result["change_percent"] = 100.0 * (after - before) / before
    return result


METRICS = {
    "wall_clock_seconds": ("wall_clock_seconds",),
    "input_frames": ("sparse_metrics", "input_frames"),
    "registered_images": ("sparse_metrics", "registered_images"),
    "registration_percent": ("sparse_metrics", "registration_percent"),
    "reprojection_error_px": ("sparse_metrics", "mean_reprojection_error_px"),
    "gps_alignment_rmse_m": ("validation", "gps_alignment_rmse_m"),
    "point_count": ("products", "point_count"),
    "dsm_valid_fraction": ("products", "dsm", "valid_fraction"),
    "mesh_vertices": ("products", "mesh_vertices"),
    "mesh_faces": ("products", "mesh_faces"),
    "mesh_components": ("products", "mesh_quality", "components_after"),
    "oversized_mesh_faces": ("products", "mesh_quality", "removed_oversized_faces"),
    "dense_stereo_seconds": ("stages", "dense_stereo", "seconds"),
    "sparse_mapping_seconds": ("stages", "sparse_mapping", "seconds"),
    "feature_extraction_seconds": ("stages", "feature_extraction", "seconds"),
    "sequential_matching_seconds": ("stages", "sequential_matching", "seconds"),
    "image_undistortion_seconds": ("stages", "image_undistortion", "seconds"),
    "stereo_fusion_seconds": ("stages", "stereo_fusion", "seconds"),
    "delaunay_meshing_seconds": ("stages", "delaunay_meshing", "seconds"),
    "mesh_cleanup_seconds": ("stages", "mesh_cleanup", "seconds"),
    "product_export_seconds": ("workflow_steps", "product_export", "seconds"),
    "dense_reference_views": ("dense_reference_selection", "selected_references"),
    "selected_baseline_median_m": (
        "workflow_steps", "keyframe_extraction", "median_selected_baseline_m"
    ),
    "selected_parallax_median_px": (
        "workflow_steps", "keyframe_extraction", "median_selected_parallax_px"
    ),
    "selected_track_fraction_median": (
        "workflow_steps", "keyframe_extraction", "median_tracked_fraction"
    ),
    "selected_sharpness_median": (
        "workflow_steps", "keyframe_extraction", "sharpness_median"
    ),
}


def compare_output_directories(
    before_dir: str | Path,
    after_dir: str | Path,
    output_path: str | Path,
) -> dict:
    before = _read_report(before_dir)
    after = _read_report(after_dir)
    before_comparison_metadata = extract_comparison_metadata(before, before_dir)
    after_comparison_metadata = extract_comparison_metadata(after, after_dir)
    fairness = fair_comparison_report(
        before_comparison_metadata, after_comparison_metadata
    )
    surface_quality = surface_quality_report(
        before_comparison_metadata, after_comparison_metadata
    )
    metrics = {
        name: _change(_get(before, *path), _get(after, *path))
        for name, path in METRICS.items()
    }
    runtime_before = metrics["wall_clock_seconds"]["before"]
    runtime_after = metrics["wall_clock_seconds"]["after"]
    speedup = (
        runtime_before / runtime_after
        if isinstance(runtime_before, (int, float))
        and isinstance(runtime_after, (int, float))
        and runtime_after > 0
        else None
    )
    surface_accuracy_comparable = bool(
        fairness["gates"]["baseline_has_independent_surface_evidence"]
        and fairness["gates"]["candidate_has_independent_surface_evidence"]
        and fairness["gates"]["same_independent_surface_evidence_set"]
    )
    guardrails = {
        "runtime_improved": (
            runtime_after < runtime_before
            if isinstance(runtime_before, (int, float)) and isinstance(runtime_after, (int, float))
            else None
        ),
        "registration_drop_at_most_1_point": _within_limit(
            metrics["registration_percent"]["before"],
            metrics["registration_percent"]["after"],
            lower_delta=-1.0,
        ),
        "reprojection_increase_at_most_0_03_px": _within_limit(
            metrics["reprojection_error_px"]["before"],
            metrics["reprojection_error_px"]["after"],
            upper_delta=0.03,
        ),
        "gps_rmse_increase_at_most_0_15_m": _within_limit(
            metrics["gps_alignment_rmse_m"]["before"],
            metrics["gps_alignment_rmse_m"]["after"],
            upper_delta=0.15,
        ),
        "point_count_retains_75_percent": _retains_ratio(
            metrics["point_count"]["before"], metrics["point_count"]["after"], 0.75
        ),
        "dsm_coverage_retains_95_percent": _retains_ratio(
            metrics["dsm_valid_fraction"]["before"],
            metrics["dsm_valid_fraction"]["after"],
            0.95,
        ),
        "mesh_components_increase_at_most_3": _within_limit(
            metrics["mesh_components"]["before"],
            metrics["mesh_components"]["after"],
            upper_delta=3.0,
        ),
    }
    evaluated_guardrails = [value for value in guardrails.values() if value is not None]
    passes_quality_guardrails = bool(
        len(evaluated_guardrails) == len(guardrails) and all(evaluated_guardrails)
    )
    result = {
        "before": str(Path(before_dir)),
        "after": str(Path(after_dir)),
        "before_experiment": before.get("experiment", {}),
        "after_experiment": after.get("experiment", {}),
        "metrics": metrics,
        "speedup": speedup,
        "quality_guardrails": guardrails,
        "passes_quality_guardrails": passes_quality_guardrails,
        "comparison_metadata": {
            "before": before_comparison_metadata,
            "after": after_comparison_metadata,
        },
        "fair_comparison_gates": fairness["gates"],
        "fair_comparison_failures": fairness["failures"],
        "passes_fair_comparison_gates": fairness["passes"],
        "surface_quality_gates": surface_quality["gates"],
        "surface_quality_failures": surface_quality["failures"],
        "passes_surface_quality_gates": surface_quality["passes"],
        "surface_quality_tolerances": surface_quality["tolerances"],
        "surface_quality_policy": surface_quality["policy"],
        "candidate_eligible_to_replace_baseline": bool(
            passes_quality_guardrails and fairness["passes"] and surface_quality["passes"]
        ),
        "protected_baseline_policy": PROTECTED_FULL_VIDEO_BASELINE,
        "baseline_matches_fixed_253_frame_workload": bool(
            matches_protected_full_video_workload(before_comparison_metadata)
        ),
        "baseline_matches_accepted_full_253_result": bool(
            matches_protected_accepted_result(
                before_comparison_metadata, runtime_before
            )
        ),
        "fair_comparison_policy": fairness["policy"],
        "surface_accuracy_comparable": surface_accuracy_comparable,
        "surface_accuracy_note": (
            "Compared using the same independent surface-evidence set."
            if surface_accuracy_comparable
            else "NOT EVALUATED: both runs need the same independent surveyed evidence "
            "or dense reference geometry under a locked evaluation protocol; "
            "reprojection and GPS residual do not prove surface accuracy."
        ),
    }
    output_path = Path(output_path)
    write_json(output_path, result)

    rows = []
    for name, values in metrics.items():
        percent = values["change_percent"]
        rows.append(
            f"| {name} | {values['before']} | {values['after']} | "
            f"{percent:+.2f}% |" if percent is not None else
            f"| {name} | {values['before']} | {values['after']} | n/a |"
        )
    markdown = (
        "# Reconstruction A/B comparison\n\n"
        f"- Before: `{Path(before_dir)}`\n"
        f"- After: `{Path(after_dir)}`\n"
        f"- Runtime speedup: `{speedup:.2f}x`\n\n" if speedup is not None else
        "# Reconstruction A/B comparison\n\nRuntime speedup could not be calculated.\n\n"
    )
    markdown += (
        "| Metric | Before | After | Change |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n## Accuracy boundary\n\n"
        + result["surface_accuracy_note"]
        + "\n\n## Fair-comparison gates\n\n"
        + "\n".join(
            f"- {name}: `{value}`"
            for name, value in fairness["gates"].items()
        )
        + f"\n\nFair-comparison result: `{fairness['passes']}`\n"
        + "\n\n## Independent surface-quality gates\n\n"
        + "\n".join(
            f"- {name}: `{value}`"
            for name, value in surface_quality["gates"].items()
        )
        + f"\n\nSurface-quality result: `{surface_quality['passes']}`\n"
        + "\n\n## Runtime/quality guardrails\n\n"
        + "\n".join(
            f"- {name}: `{value if value is not None else 'not evaluated'}`"
            for name, value in guardrails.items()
        )
        + f"\n\nOverall guardrail result: `{result['passes_quality_guardrails']}`\n"
        + "\nCandidate eligible to replace baseline: "
        + f"`{result['candidate_eligible_to_replace_baseline']}`\n"
    )
    output_path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    return result


def _within_limit(
    before: Any,
    after: Any,
    lower_delta: float | None = None,
    upper_delta: float | None = None,
) -> bool | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    delta = after - before
    return bool(
        (lower_delta is None or delta >= lower_delta)
        and (upper_delta is None or delta <= upper_delta)
    )


def _retains_ratio(before: Any, after: Any, ratio: float) -> bool | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    if before <= 0:
        return None
    return bool(after >= ratio * before)
