from __future__ import annotations

import csv
import json
import math
import struct
from datetime import datetime
from pathlib import Path

from .production_status import derive_terminal_state


REQUIRED_STAGES = [
    "feature_extraction", "sequential_matching", "sparse_mapping", "sparse_analysis",
    "gps_alignment", "aligned_model_text_export", "image_undistortion", "dense_stereo",
    "stereo_fusion", "mesh_cleanup",
]
REQUIRED_FILES = [
    "point_cloud.ply", "mesh.ply", "point_cloud.las", "dsm.tif", "model.glb", "model.obj",
    "confidence.tif", "confidence_summary.json", "capture_quality.json",
    "georeference.json", "run_report.json", "viewer_metadata.json", "frames.csv", "gpu_usage.csv",
]
REQUIRED_WORKFLOW_STEPS = [
    "keyframe_extraction", "capture_quality", "telemetry_sync", "ai_dynamic_masking", "validation", "product_export",
]

DEFAULT_SUBSTANTIVE_GEOMETRY_THRESHOLDS = {
    "minimum_point_count": 1000,
    "minimum_mesh_vertices": 500,
    "minimum_mesh_faces": 500,
    "minimum_dsm_valid_cells": 500,
}


def footprint_coverage(valid, resolution_m: float, cell_m: float = 5.0, bridge_m: float = 30.0) -> dict:
    """Share of the flown footprint that actually has DSM data.

    The raster's bounding rectangle includes corners the drone never saw, so
    filled-cells / all-cells fails any non-rectangular flight. Instead the DSM
    is summarised on a coarse grid (sparse points leave speckle at 0.5 m), the
    footprint is the outline of the data with gaps up to ``bridge_m`` closed,
    and coverage is data cells / footprint cells. Interior holes (water,
    shadow, missed ground) still count against it.
    """
    import numpy as np
    from scipy import ndimage

    valid = np.asarray(valid, dtype=bool)
    factor = max(1, int(round(cell_m / resolution_m)))
    rows = -(-valid.shape[0] // factor) * factor
    cols = -(-valid.shape[1] // factor) * factor
    padded = np.zeros((rows, cols), dtype=bool)
    padded[:valid.shape[0], :valid.shape[1]] = valid
    coarse = padded.reshape(rows // factor, factor, cols // factor, factor).any(axis=(1, 3))
    radius = max(1, int(math.ceil(bridge_m / (factor * resolution_m))))
    coarse = np.pad(coarse, radius + 1)
    coarse = ndimage.binary_closing(coarse, iterations=1)      # ignore single-cell speckle
    disk = np.hypot(*np.mgrid[-radius:radius + 1, -radius:radius + 1]) <= radius
    footprint = ndimage.binary_erosion(
        ndimage.binary_fill_holes(ndimage.binary_dilation(coarse, disk)), disk
    )
    cell_area = (factor * resolution_m) ** 2
    footprint_cells = int(footprint.sum())
    covered_cells = int((coarse & footprint).sum())
    return {
        "analysis_cell_m": factor * resolution_m,
        "bridge_m": bridge_m,
        "footprint_area_m2": footprint_cells * cell_area,
        "covered_area_m2": covered_cells * cell_area,
        "fraction": covered_cells / footprint_cells if footprint_cells else 0.0,
    }


def _substantive_geometry_checks(formats: dict, report: dict) -> dict:
    """Reject syntactically valid but operationally empty reconstructions."""
    configured = report.get("quality_thresholds", {}).get(
        "substantive_geometry", {}
    )
    registered = int(report.get("sparse_metrics", {}).get("registered_images") or 0)
    thresholds = {
        **DEFAULT_SUBSTANTIVE_GEOMETRY_THRESHOLDS,
        **{
            key: int(value)
            for key, value in configured.items()
            if key in DEFAULT_SUBSTANTIVE_GEOMETRY_THRESHOLDS
            and not isinstance(value, bool)
            and int(value) >= 0
        },
    }
    # Scale modestly with camera count while retaining useful minima for small
    # bounded components. These are anti-degeneracy gates, not completeness or
    # independent-accuracy claims.
    required = {
        "minimum_point_count": max(thresholds["minimum_point_count"], registered * 20),
        "minimum_mesh_vertices": max(thresholds["minimum_mesh_vertices"], registered * 10),
        "minimum_mesh_faces": max(thresholds["minimum_mesh_faces"], registered * 15),
        "minimum_dsm_valid_cells": max(thresholds["minimum_dsm_valid_cells"], registered * 10),
    }
    las = formats.get("las", {})
    mesh = formats.get("mesh.ply", {})
    dsm = formats.get("geotiff", {})
    checks = {
        "point_cloud_is_substantive": _at_least(
            las.get("points"), required["minimum_point_count"]
        ),
        "mesh_vertices_are_substantive": _at_least(
            mesh.get("vertices"), required["minimum_mesh_vertices"]
        ),
        "mesh_faces_are_substantive": _at_least(
            mesh.get("faces"), required["minimum_mesh_faces"]
        ),
        "dsm_has_substantive_valid_area": _at_least(
            dsm.get("valid_cells"), required["minimum_dsm_valid_cells"]
        ),
    }
    return {
        "thresholds": required,
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "anti_degeneracy_only_not_surface_completeness",
    }


def _at_least(value: object, minimum: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= minimum


def _at_most(value: object, maximum: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number <= maximum


def _final_mesh_edge_max(mesh_quality: dict) -> float | None:
    """Prefer the measured post-cleanup edge maximum, with legacy fallback."""
    after = mesh_quality.get("longest_edge_max_after_m")
    return after if after is not None else mesh_quality.get("longest_edge_max_before_m")


def _surface_accuracy_results(validation: dict) -> tuple[bool | None, bool | None]:
    """Return surveyed-only and all-independent one-metre outcomes."""
    checkpoint_surface = validation.get("surveyed_checkpoints", {}).get(
        "passes_one_metre_target"
    )
    distance_surface = validation.get("independent_distances", {}).get(
        "passes_one_metre_target"
    )
    surveyed_outcomes = [
        value for value in (checkpoint_surface, distance_surface) if isinstance(value, bool)
    ]
    surveyed_surface = all(surveyed_outcomes) if surveyed_outcomes else None

    dense_validation = validation.get("dense_geometry", {})
    dense_surface = dense_validation.get("passes_one_metre_target")
    if dense_validation.get("reference_is_distinct_from_reconstruction") is False:
        dense_surface = None
    independent_outcomes = list(surveyed_outcomes)
    if isinstance(dense_surface, bool):
        independent_outcomes.append(dense_surface)
    independent_surface = all(independent_outcomes) if independent_outcomes else None
    return surveyed_surface, independent_surface


def _ply_header_counts(path: Path) -> tuple[int, int]:
    """Read vertex/face counts without allocating a full dense mesh."""
    vertices = faces = 0
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Invalid PLY signature")
        for _ in range(10_000):
            line = stream.readline()
            if not line:
                raise ValueError("PLY end_header not found")
            fields = line.decode("ascii", errors="replace").strip().split()
            if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
                vertices = int(fields[2])
            elif len(fields) == 3 and fields[:2] == ["element", "face"]:
                faces = int(fields[2])
            elif fields == ["end_header"]:
                break
    return vertices, faces


def _glb_mesh_count(path: Path) -> int:
    """Validate GLB structure and count declared meshes without decoding buffers."""
    with path.open("rb") as stream:
        magic, version, declared_length = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2 or declared_length != path.stat().st_size:
            raise ValueError("Invalid GLB header")
        chunk_length, chunk_type = struct.unpack("<II", stream.read(8))
        if chunk_type != 0x4E4F534A:
            raise ValueError("GLB first chunk is not JSON")
        document = json.loads(stream.read(chunk_length).decode("utf-8").rstrip(" \t\r\n\0"))
    return len(document.get("meshes", []))


def verify_output_directory(output_dir: str | Path) -> dict:
    import laspy
    import rasterio

    output_dir = Path(output_dir)
    files = {name: (output_dir / name).is_file() for name in REQUIRED_FILES}
    report_path = output_dir / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    stages = report.get("stages", {})
    workflow = report.get("workflow_steps", {})
    sparse_reuse = report.get("sparse_reuse", {})
    reused_sparse_stages = set(sparse_reuse.get("skipped_stages", []))
    reused_sparse_valid = bool(
        sparse_reuse.get("enabled")
        and Path(sparse_reuse.get("source_model", "")).is_dir()
    )
    stage_checks = {
        name: bool(
            (name in reused_sparse_stages and reused_sparse_valid)
            or (
                name in stages
                and stages[name].get("return_code") == 0
                and stages[name].get("seconds") is not None
                and stages[name].get("log_file")
                and Path(stages[name]["log_file"]).is_file()
            )
        )
        for name in REQUIRED_STAGES
    }
    successful_meshers = [
        name
        for name in (
            "poisson_meshing", "delaunay_meshing", "advancing_front_meshing"
        )
        if name in stages
        and stages[name].get("return_code") == 0
        and stages[name].get("seconds") is not None
        and stages[name].get("log_file")
        and Path(stages[name]["log_file"]).is_file()
    ]
    stage_checks["meshing"] = bool(successful_meshers)
    formats = {}
    try:
        with laspy.open(output_dir / "point_cloud.las") as reader:
            point_count = reader.header.point_count
            crs = reader.header.parse_crs()
        formats["las"] = {"valid": point_count > 0 and crs is not None, "points": point_count, "crs": str(crs)}
    except Exception as error:
        formats["las"] = {"valid": False, "error": str(error)}
    try:
        with rasterio.open(output_dir / "dsm.tif") as dataset:
            band = dataset.read(1, masked=True)
            valid_cells = int(band.count())
            total_cells = int(band.size)
            import numpy as np

            formats["geotiff"] = {
                "valid": dataset.width > 0 and dataset.height > 0 and dataset.crs is not None,
                "width": dataset.width, "height": dataset.height, "crs": str(dataset.crs),
                "valid_cells": valid_cells,
                # Kept for comparison with earlier runs; the gate uses footprint coverage.
                "valid_fraction": valid_cells / total_cells if total_cells else 0.0,
                "footprint_coverage": footprint_coverage(
                    ~np.ma.getmaskarray(band), abs(float(dataset.res[0]))
                ),
                "resolution": list(dataset.res),
            }
    except Exception as error:
        formats["geotiff"] = {"valid": False, "error": str(error)}
    try:
        with rasterio.open(output_dir / "confidence.tif") as dataset:
            band = dataset.read(1, masked=True)
            values = band.compressed()
            formats["confidence_geotiff"] = {
                "valid": bool(
                    dataset.width > 0 and dataset.height > 0 and dataset.crs is not None
                    and len(values) and values.min() >= 0 and values.max() <= 1
                ),
                "width": dataset.width,
                "height": dataset.height,
                "crs": str(dataset.crs),
                "mean_confidence": float(values.mean()) if len(values) else None,
            }
    except Exception as error:
        formats["confidence_geotiff"] = {"valid": False, "error": str(error)}
    mesh_quality = report.get("products", {}).get("mesh_quality", {})
    try:
        vertices, faces = _ply_header_counts(output_dir / "mesh.ply")
        formats["mesh.ply"] = {
            "valid": vertices > 0 and faces > 0,
            "vertices": vertices,
            "faces": faces,
            "components": mesh_quality.get("components_after"),
            "watertight": mesh_quality.get("watertight"),
            "textured": mesh_quality.get("textured", False),
            "longest_edge_max_m": _final_mesh_edge_max(mesh_quality),
            "robust_edge_limit_m": mesh_quality.get("longest_edge_limit_m"),
            "oversized_face_fraction": mesh_quality.get("final_oversized_face_fraction"),
            "removed_unsupported_faces": mesh_quality.get("removed_unsupported_faces"),
            "validation_method": "ply_header_plus_recorded_export_metrics",
        }
    except Exception as error:
        formats["mesh.ply"] = {"valid": False, "error": str(error)}
    try:
        mesh_count = _glb_mesh_count(output_dir / "model.glb")
        formats["model.glb"] = {
            "valid": mesh_count > 0,
            "meshes": mesh_count,
            "textured": report.get("products", {}).get("textured_glb", False),
            "validation_method": "glb_header_and_json_chunk",
        }
    except Exception as error:
        formats["model.glb"] = {"valid": False, "error": str(error)}
    try:
        gpu_summary = json.loads((output_dir / "gpu_usage.summary.json").read_text(encoding="utf-8"))
        formats["gpu_log"] = {"valid": bool(gpu_summary.get("available") and gpu_summary.get("samples", 0) > 0), **gpu_summary}
    except Exception as error:
        formats["gpu_log"] = {"valid": False, "error": str(error)}
    gpu_by_stage = {}
    try:
        with (output_dir / "gpu_usage.csv").open(newline="", encoding="utf-8") as stream:
            gpu_rows = list(csv.DictReader(stream))
        for name, stage in stages.items():
            if not stage.get("started_at_utc") or not stage.get("ended_at_utc"):
                continue
            started = datetime.fromisoformat(stage["started_at_utc"])
            ended = datetime.fromisoformat(stage["ended_at_utc"])
            selected = [
                row for row in gpu_rows
                if started <= datetime.fromisoformat(row["sample_time_utc"]) <= ended
            ]
            utilization = [float(row["utilization_gpu_percent"]) for row in selected if row["utilization_gpu_percent"] not in {"", "N/A", "[N/A]"}]
            memory = [float(row["memory_used_mb"]) for row in selected if row["memory_used_mb"] not in {"", "N/A", "[N/A]"}]
            gpu_by_stage[name] = {
                "samples": len(selected),
                "gpu_utilization_mean_percent": sum(utilization) / len(utilization) if utilization else None,
                "gpu_utilization_max_percent": max(utilization) if utilization else None,
                "gpu_memory_max_mb": max(memory) if memory else None,
            }
    except Exception as error:
        gpu_by_stage = {"error": str(error)}
    checkpoint_report = report.get("validation", {}).get("surveyed_checkpoints", {})
    dense_report = report.get("validation", {}).get("dense_geometry", {})
    dense_one_metre = {}
    for row in dense_report.get("threshold_metrics", []):
        try:
            threshold_m = float(row.get("threshold_m"))
        except (AttributeError, TypeError, ValueError):
            continue
        if abs(threshold_m - 1.0) < 1e-9:
            dense_one_metre = row
            break
    result = {
        "files": files,
        "stages": stage_checks,
        "workflow_steps": {
            name: bool(
                workflow.get(name, {}).get("status")
                in (
                    {"passed", "skipped"}
                    if name == "ai_dynamic_masking"
                    else {"passed", "warning"}
                    if name == "capture_quality"
                    else {"passed"}
                )
                and workflow[name].get("seconds") is not None
            )
            for name in REQUIRED_WORKFLOW_STEPS
        },
        "formats": formats,
        "gpu_by_stage": gpu_by_stage,
        "validation": report.get("validation", {}),
        "metrics": {
            "processing_under_15_minutes": report.get("targets", {}).get("processing_under_15_minutes"),
            "one_metre_accuracy": report.get("targets", {}).get("one_metre_accuracy"),
            "registration_percent": report.get("sparse_metrics", {}).get("registration_percent"),
            "reprojection_error_px": report.get("sparse_metrics", {}).get("mean_reprojection_error_px"),
            "checkpoint_rmse_horizontal_m": checkpoint_report.get("rmse_horizontal_m"),
            "checkpoint_rmse_vertical_m": checkpoint_report.get("rmse_vertical_m"),
            "checkpoint_rmse_3d_m": checkpoint_report.get("rmse_3d_m"),
            "dense_precision_at_1m": dense_one_metre.get("precision"),
            "dense_completeness_at_1m": dense_one_metre.get("completeness"),
            "dense_f1_at_1m": dense_one_metre.get("f1"),
            "dense_reconstruction_to_reference_rmse_m": dense_report.get(
                "distances_m", {}
            ).get("reconstruction_to_reference", {}).get("rmse_m"),
            "dense_reference_to_reconstruction_rmse_m": dense_report.get(
                "distances_m", {}
            ).get("reference_to_reconstruction", {}).get("rmse_m"),
        },
        "standards": {
            "asprs_checkpoint_count_status": checkpoint_report.get(
                "asprs_checkpoint_count_status", "not_evaluated"
            ),
            "asprs_checkpoint_count_compliant": checkpoint_report.get(
                "asprs_checkpoint_count_compliant", False
            ),
            "asprs_compliance_scope": checkpoint_report.get("asprs_compliance_scope"),
            "note": (
                "Checkpoint-count status alone is not a complete ASPRS compliance claim; "
                "distribution, survey quality, coordinate reference system, and reporting "
                "requirements must also be verified."
            ),
        },
    }
    # GPU telemetry is useful performance evidence, but it is not a geometry
    # artifact. A missing monitor must not reject an otherwise valid mesh.
    core_files = {
        name: passed for name, passed in files.items()
        if name != "gpu_usage.csv"
    }
    core_formats = {
        name: details for name, details in formats.items()
        if name != "gpu_log"
    }
    result["monitoring_checks"] = {
        "gpu_usage_csv_present": files.get("gpu_usage.csv", False),
        "gpu_log_valid": formats.get("gpu_log", {}).get("valid", False),
    }
    result["structural_artifact_checks_pass"] = (
        all(core_files.values())
        and all(stage_checks.values())
        and all(result["workflow_steps"].values())
        and all(item.get("valid", False) for item in core_formats.values())
    )
    result["substantive_geometry"] = _substantive_geometry_checks(formats, report)
    result["substantive_geometry_checks_pass"] = result[
        "substantive_geometry"
    ]["passed"]
    result["artifact_checks_pass"] = bool(
        result["structural_artifact_checks_pass"]
        and result["substantive_geometry_checks_pass"]
    )
    targets = report.get("targets", {})
    geotiff = formats.get("geotiff", {})
    mesh = formats.get("mesh.ply", {})
    textured_assets = (output_dir / "textured" / "mesh.ply").is_file() and (
        output_dir / "textured" / "texture.png"
    ).is_file()
    validation = report.get("validation", {})
    surveyed_surface, independent_surface = _surface_accuracy_results(validation)
    result["quality_checks"] = {
        "capture_quality_passed": report.get("capture_quality", {}).get("ready") is True,
        "gcp_alignment_fit_passed_if_used": (
            report.get("control_point_alignment") is None
            or report.get("control_point_alignment", {}).get("passes_fit_gate") is True
        ),
        "registration_at_least_90_percent": _at_least(
            report.get("sparse_metrics", {}).get("registration_percent"), 90
        ),
        "reprojection_error_at_most_2px": _at_most(
            report.get("sparse_metrics", {}).get("mean_reprojection_error_px"), 2
        ),
        "absolute_camera_alignment_at_most_1m": _at_most(
            report.get("validation", {}).get("gps_alignment_rmse_m"), 1
        ),
        "dsm_coverage_at_least_75_percent": (
            geotiff.get("footprint_coverage", {}).get("fraction", 0) >= 0.75
        ),
        "mesh_has_at_most_20_components": 0 < mesh.get("components", 0) <= 20,
        "mesh_has_at_most_0_1_percent_oversized_faces": (
            mesh.get("oversized_face_fraction") is not None
            and mesh["oversized_face_fraction"] <= 0.001
        ),
        "mesh_is_textured": mesh.get("textured") is True or textured_assets,
        "processing_under_15_minutes": targets.get("processing_under_15_minutes") is True,
        "surveyed_surface_accuracy_at_most_1m": surveyed_surface,
        "independent_surface_accuracy_at_most_1m": independent_surface,
        "consistent_enu_origin": report.get("georeference", {}).get("origin_policy") == "first_frame_reference",
    }
    # GPS-camera agreement remains visible as a diagnostic, but it is not an
    # independent surface-accuracy measurement. A run that passes surveyed
    # checkpoints must not be rejected merely because consumer-grade flight
    # telemetry is noisier than the reconstructed product.
    result["diagnostic_checks"] = {
        "gps_camera_agreement_at_most_1m": result["quality_checks"][
            "absolute_camera_alignment_at_most_1m"
        ],
    }
    production_gate_names = [
        name for name in result["quality_checks"]
        if name not in {
            "absolute_camera_alignment_at_most_1m",
            "surveyed_surface_accuracy_at_most_1m",
        }
    ]
    result["production_gate_names"] = production_gate_names
    result["production_ready"] = result["artifact_checks_pass"] and all(
        result["quality_checks"][name] is True for name in production_gate_names
    )
    telemetry_step = workflow.get("telemetry_sync", {})
    terminal_state, terminal_reasons = derive_terminal_state(
        telemetry_valid=telemetry_step.get("status") == "passed",
        capture_sufficient=report.get("capture_quality", {}).get("ready") is True,
        structural_artifacts_valid=result["structural_artifact_checks_pass"],
        substantive_geometry_valid=result["substantive_geometry_checks_pass"],
        production_gates_pass=result["production_ready"],
        deadline_met=targets.get("processing_under_15_minutes") is True,
        stopped_for_deadline=report.get("runtime", {}).get(
            "stopped_for_deadline", False
        ) is True,
    )
    result["terminal_state"] = terminal_state.value
    result["terminal_reasons"] = terminal_reasons
    (output_dir / "verification_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
