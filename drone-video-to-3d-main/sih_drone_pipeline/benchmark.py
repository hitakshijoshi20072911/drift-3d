from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .compare_runs import (
    PROTECTED_FULL_VIDEO_BASELINE,
    extract_comparison_metadata,
    fair_comparison_report,
    matches_protected_accepted_result,
    matches_protected_full_video_workload,
    surface_quality_report,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _runtime_seconds(report: dict) -> float | None:
    value = report.get("wall_clock_seconds")
    if value is None:
        value = report.get("processing_seconds")
    if isinstance(value, bool):
        return None
    try:
        runtime = float(value)
    except (TypeError, ValueError):
        return None
    return runtime if math.isfinite(runtime) and runtime > 0 else None


def build_benchmark(entries: list[tuple[str, str | Path]], output_path: str | Path) -> dict:
    """Compare normalized runs against the first, protected reference workload.

    Existing reports remain readable. Missing comparison metadata is represented
    explicitly and fails closed instead of allowing a reduced workload to win on
    runtime alone.
    """
    if len(entries) < 2:
        raise ValueError("Benchmarking requires at least two labelled runs")
    loaded = []
    for label, directory_value in entries:
        directory = Path(directory_value)
        report = _read_json(directory / "run_report.json")
        verification = _read_json(directory / "verification_report.json")
        comparison_metadata = extract_comparison_metadata(report, directory)
        loaded.append((label, directory, report, verification, comparison_metadata))

    reference_label, _, _, _, reference_metadata = loaded[0]
    rows = []
    for label, directory, report, verification, comparison_metadata in loaded:
        sparse = report.get("sparse_metrics", {})
        targets = report.get("targets", {})
        validation = report.get("validation", {})
        checkpoints = validation.get("surveyed_checkpoints", {})
        products = report.get("products", {})
        mesh_quality = products.get("mesh_quality", {})
        fairness = fair_comparison_report(reference_metadata, comparison_metadata)
        surface_quality = surface_quality_report(reference_metadata, comparison_metadata)
        surface_evidence = comparison_metadata["independent_surface_evidence"]
        runtime_seconds = _runtime_seconds(report)
        exclusion_reasons = []
        if verification.get("production_ready") is not True:
            exclusion_reasons.append("not_production_ready")
        if runtime_seconds is None:
            exclusion_reasons.append("missing_valid_runtime")
        exclusion_reasons.extend(fairness["failures"])
        exclusion_reasons.extend(surface_quality["failures"])
        row = {
            "label": label,
            "directory": str(directory.resolve()),
            "runtime_seconds": runtime_seconds,
            "registration_percent": sparse.get("registration_percent"),
            "reprojection_error_px": sparse.get("mean_reprojection_error_px"),
            "checkpoint_rmse_3d_m": checkpoints.get("rmse_3d_m"),
            "checkpoint_max_error_m": checkpoints.get("maximum_3d_error_m"),
            "point_count": products.get("point_count"),
            "mesh_faces": products.get("mesh_faces"),
            "dsm_valid_fraction": products.get("dsm", {}).get("valid_fraction"),
            "confidence_high_fraction": products.get("confidence", {}).get("high_confidence_fraction"),
            "final_oversized_face_fraction": mesh_quality.get("final_oversized_face_fraction"),
            "artifact_checks_pass": verification.get("artifact_checks_pass"),
            "production_ready": verification.get("production_ready"),
            "under_15_minutes": targets.get("processing_under_15_minutes"),
            "surface_accuracy_pass": targets.get("one_metre_surface_accuracy"),
            "video_identity": comparison_metadata["video_identity"],
            "video_duration_s": comparison_metadata["video_duration_s"],
            "target_frames": comparison_metadata["target_frames"],
            "selected_frames": comparison_metadata["selected_frames"],
            "selected_frame_manifest_sha256": comparison_metadata[
                "selected_frame_manifest_sha256"
            ],
            "frame_resolution": (
                "x".join(str(value) for value in comparison_metadata["frame_resolution"])
                if comparison_metadata["frame_resolution"] is not None
                else None
            ),
            "independent_surface_evidence": surface_evidence["valid"],
            "surface_evidence_type": surface_evidence["type"],
            "surface_evidence_checks": surface_evidence["checks"],
            "fair_comparison_pass": fairness["passes"],
            "fair_comparison_failures": fairness["failures"],
            "surface_quality_pass": surface_quality["passes"],
            "surface_quality_failures": surface_quality["failures"],
            "eligible": not exclusion_reasons,
            "exclusion_reasons": exclusion_reasons,
            "comparison_metadata": comparison_metadata,
            "fair_comparison_gates": fairness["gates"],
            "surface_quality_gates": surface_quality["gates"],
        }
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda row: (
            row["runtime_seconds"],
        ),
    )
    result = {
        "runs": rows,
        "winner": ranked[0]["label"] if ranked else None,
        "reference_run": reference_label,
        "protected_baseline_policy": PROTECTED_FULL_VIDEO_BASELINE,
        "reference_matches_fixed_253_frame_workload": bool(
            matches_protected_full_video_workload(reference_metadata)
        ),
        "reference_matches_accepted_full_253_result": bool(
            matches_protected_accepted_result(
                reference_metadata, rows[0]["runtime_seconds"]
            )
        ),
        "selection_rule": (
            "fastest production-ready run that passes all fair-comparison gates against "
            "the first (reference) run"
        ),
        "note": (
            "Video identity/duration, target and exact selected-frame manifest, frame "
            "resolution, and independent surface evidence must remain fixed. Missing "
            "metadata is not treated as a pass."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output.with_suffix(".md")
    columns = [
        "label", "runtime_seconds", "registration_percent", "checkpoint_rmse_3d_m",
        "selected_frames", "frame_resolution", "point_count", "dsm_valid_fraction",
        "production_ready", "fair_comparison_pass", "surface_quality_pass", "eligible",
    ]
    lines = [
        "# Reconstruction benchmark",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    lines += [
        "",
        f"Reference run: **{reference_label}**",
        f"Winner: **{result['winner'] or 'none (no eligible fair comparison)'}**",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    result["outputs"] = {"json": str(output), "csv": str(csv_path), "markdown": str(markdown_path)}
    return result
