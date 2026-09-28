"""Measured requirement scorecard; unknown accuracy/coverage never becomes a pass."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path


def duration_deadline_minutes(duration_s: float, fixed_minutes=None) -> float:
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    value = float(fixed_minutes) if fixed_minutes is not None else max(3.5, 1.5 * duration_s / 60)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("deadline must be finite and positive")
    return value


def requirement_report(*, duration_s, elapsed_s, deadline_minutes, selection,
                       component_reports, artifacts, merge_pass=False, checkpoints=None):
    if not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("Invalid measured runtime")
    reconstructed = [r for r in component_reports if r.get("status") == "reconstructed"]
    accuracy = (checkpoints or {}).get("rmse_3d_m")
    count = (checkpoints or {}).get("checkpoint_count", 0)
    accuracy_pass = (bool(math.isfinite(accuracy) and accuracy <= 1.0)
                     if isinstance(accuracy, (float, int)) and count >= 3 else None)
    # No image-count or vertex-occupancy proxy can establish entire-visible-scene
    # completeness. Keep this unknown until a defensible surface audit exists.
    coverage_pass = None
    formats = {}
    for kind in ("obj", "ply", "las", "geotiff", "glb", "gltf", "fbx"):
        paths = [Path(p) for p in artifacts.get(kind, [])]
        formats[kind] = {"files": [str(p) for p in paths],
                         "present": bool(paths) and all(p.is_file() and p.stat().st_size > 0 for p in paths)}
    runtime_pass = elapsed_s < deadline_minutes * 60
    checks = {"runtime": runtime_pass, "spatial_accuracy": accuracy_pass,
              "entire_visible_scene": coverage_pass, "merge": bool(merge_pass),
              "all_formats": all(item["present"] for item in formats.values()),
              "reconstruction": bool(reconstructed)}
    return {
        "schema_version": 1, "source_duration_minutes": duration_s/60,
        "processing_minutes": elapsed_s/60, "deadline_minutes": deadline_minutes,
        "runtime_includes": "screening, reconstruction, merge, artifact export; setup and manual review excluded",
        "runtime_per_video_minute": elapsed_s/duration_s,
        "ten_minute_sla_applicable": abs(duration_s - 600) <= 1,
        "reconstruction_type": "3D mesh and point cloud" if reconstructed else "unavailable",
        "spatial_accuracy_rmse_3d_m": accuracy, "independent_checkpoint_count": count,
        "spatial_accuracy_evidence": "independent_checkpoints" if accuracy_pass is not None else "unmeasured",
        "entire_visible_scene_coverage_fraction": None,
        "candidate_visual_support_fraction": selection.get("candidate_visual_support_fraction"),
        "selected_frames": selection.get("selected_frames"),
        "verified_graph_components": selection.get("verified_graph_components"),
        "unverified_adjacent_links": selection.get("unverified_adjacent_links"),
        "selection_budget_conflict": selection.get("budget_conflict"),
        "successful_components": len(reconstructed), "formats": formats,
        "format_validation_scope": "presence and nonzero size; component verification is separate",
        "checks": checks, "production_ready": all(value is True for value in checks.values()),
        "unmet_or_unverified_targets": [key for key,value in checks.items() if value is not True],
        "coverage_disclosure": "Registration, graph connectivity and camera-path support do not measure surface completeness.",
    }


def export_interchange(glb_path, output_dir, timeout_s=60.0) -> dict:
    """Export actual glTF and FBX through Blender; preserve failures explicitly."""
    source, output = Path(glb_path).resolve(), Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    blender = shutil.which("blender")
    if blender is None:
        return {"status": "unavailable", "reason": "Blender is not installed", "files": []}
    if timeout_s <= 0:
        return {"status": "deadline_limited", "files": []}
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("blender_interchange.py")
    try:
        result = subprocess.run([blender, "--background", "--factory-startup", "--python", str(script),
                                 "--", str(source), str(output)], timeout=timeout_s,
                                capture_output=True, text=True, errors="replace")
    except subprocess.TimeoutExpired:
        return {"status": "deadline_limited", "files": []}
    files = [output / "scene.gltf", output / "scene.fbx"]
    valid = result.returncode == 0 and all(p.is_file() and p.stat().st_size > 0 for p in files)
    return {"status": "exported" if valid else "failed",
            "files": [str(p) for p in files] if valid else [],
            "log_tail": (result.stderr + result.stdout)[-1500:] if not valid else None}
