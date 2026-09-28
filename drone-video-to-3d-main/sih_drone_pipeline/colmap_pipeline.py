from __future__ import annotations

import csv
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .camera import colmap_image_reader_options
from .utils import PipelineError, run_command, write_json


_FEATURE_TYPES = {
    "sift": "SIFT",
    "aliked-n16rot": "ALIKED_N16ROT",
    "aliked-n32": "ALIKED_N32",
}


def _feature_configuration(feature_type: str, feature_matcher: str) -> tuple[str, str]:
    """Resolve a compatible COLMAP extractor/matcher enum pair."""
    if feature_type not in _FEATURE_TYPES:
        raise ValueError(f"Unsupported feature_type: {feature_type}")
    if feature_matcher not in {"bruteforce", "lightglue"}:
        raise ValueError(f"Unsupported feature_matcher: {feature_matcher}")
    family = "SIFT" if feature_type == "sift" else "ALIKED"
    matcher = f"{family}_{'LIGHTGLUE' if feature_matcher == 'lightglue' else 'BRUTEFORCE'}"
    return _FEATURE_TYPES[feature_type], matcher


def _clean_mesher_output(mesh: Path, cleaned_mesh: Path, report: dict) -> dict:
    """Record deterministic Python mesh cleanup as a timed pipeline stage."""
    from .export_gis import clean_mesh_file

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    quality = clean_mesh_file(mesh, cleaned_mesh)
    elapsed = round(time.perf_counter() - started, 2)
    ended_at = datetime.now(timezone.utc).isoformat()
    log_dir = Path(report["_log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mesh_cleanup.log"
    log_file.write_text(str(quality), encoding="utf-8")
    report["stages"]["mesh_cleanup"] = {
        "seconds": elapsed,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "return_code": 0,
        "command": ["python", "robust_mesh_cleanup"],
        "log_tail": str(quality),
        "log_file": str(log_file),
    }
    return quality


def _require_colmap() -> str:
    executable = shutil.which("colmap")
    if not executable:
        raise PipelineError("COLMAP was not found on PATH. Run this pipeline from the supplied Colab notebook.")
    return executable


def _help_has(colmap: str, command: str, option: str) -> bool:
    import subprocess

    result = subprocess.run(
        [colmap, command, "--help"], capture_output=True, text=True, errors="replace", check=False
    )
    return option.lstrip("-") in ((result.stdout or "") + (result.stderr or ""))


# COLMAP camera model id -> number of leading focal-length parameters.
_FOCAL_PARAMS = {0: 1, 1: 2, 2: 1, 3: 1, 4: 2}   # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV


def _apply_camera_prior(database: Path, default_hfov_deg: float | None, calibration_provided: bool) -> dict:
    """Give the matcher a focal length it can trust before geometric verification.

    Without one, COLMAP guesses fx = 1.2 x image width -- a ~45 deg field of view --
    and marks it untrusted, so every pair is verified with an uncalibrated
    fundamental matrix. Drone cameras are much wider (the reference flight's lens
    refined to 68.7 deg, fx 1404 px against the 2304 px guess). A supplied
    calibration is trusted as-is; otherwise a typical drone field of view is used.
    Either way bundle adjustment still refines the intrinsics.
    """
    import math
    import sqlite3
    import struct

    applied = []
    connection = sqlite3.connect(database)
    with connection:
        cameras = connection.execute("select camera_id, model, width, height, params from cameras").fetchall()
        for camera_id, model, width, height, blob in cameras:
            params = list(struct.unpack(f"{len(blob) // 8}d", blob))
            if calibration_provided:
                source = "supplied calibration"
            elif default_hfov_deg is None:
                continue
            elif model not in _FOCAL_PARAMS:
                applied.append({"camera_id": camera_id, "skipped": f"camera model {model} not handled"})
                continue
            else:
                focal = (width / 2) / math.tan(math.radians(default_hfov_deg / 2))
                count = _FOCAL_PARAMS[model]
                params[:count] = [focal] * count
                params[count:count + 2] = [width / 2, height / 2]
                params[count + 2:] = [0.0] * (len(params) - count - 2)
                source = f"default {default_hfov_deg:g} deg horizontal field of view"
            connection.execute(
                "update cameras set params = ?, prior_focal_length = 1 where camera_id = ?",
                (struct.pack(f"{len(params)}d", *params), camera_id))
            applied.append({"camera_id": camera_id, "source": source, "focal_px": round(params[0], 2),
                            "horizontal_fov_deg": round(2 * math.degrees(math.atan(width / 2 / params[0])), 2)})
    connection.close()
    return {"default_hfov_deg": default_hfov_deg, "calibration_provided": calibration_provided,
            "cameras": applied}


def _largest_model(sparse_root: Path) -> Path:
    models = [path for path in sparse_root.iterdir() if path.is_dir()]
    if not models:
        raise PipelineError("COLMAP mapper did not create a sparse model")
    return max(models, key=lambda path: (path / "images.bin").stat().st_size if (path / "images.bin").exists() else 0)


def _frame_geometry_scores(frame_manifest: str | Path | None) -> dict[str, float]:
    if frame_manifest is None:
        return {}
    manifest = Path(frame_manifest)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    scores = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            # Geometry-selected manifests already contain the combined score.
            # Sharpness is only a deterministic fallback for older manifests.
            geometry = float(row.get("geometry_score") or 0.0)
            sharpness = float(row.get("sharpness") or 0.0)
            scores[row["image_name"]] = geometry if geometry > 0 else sharpness
    return scores


def _limit_patch_match_sources(line: str, maximum: int | None) -> str:
    if maximum is None:
        return line
    if line.startswith("__auto__"):
        return f"__auto__, {maximum}"
    sources = [value.strip() for value in line.split(",") if value.strip()]
    return ", ".join(sources[:maximum])


def _read_text_model_views(model_dir: Path) -> tuple[dict, float]:
    """Camera centre, world-from-camera rotation and intrinsics per image, plus the
    median height of the sparse points (used as the ground plane)."""
    import numpy as np

    cameras = {}
    for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        model, width, height = parts[1], int(parts[2]), int(parts[3])
        params = [float(value) for value in parts[4:]]
        two_focal = model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}
        fx = params[0]
        fy = params[1] if two_focal else params[0]
        cx, cy = (params[2], params[3]) if two_focal else (params[1], params[2])
        cameras[int(parts[0])] = (width, height, fx, fy, cx, cy)
    views = {}
    # Every image uses two lines; the second (2D points) may be empty, so pair
    # raw lines instead of filtering blanks.
    lines = [line for line in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    for line in lines[0::2]:
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        translation = np.array(list(map(float, parts[5:8])))
        rotation = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        views[" ".join(parts[9:])] = {
            "centre": -(rotation.T @ translation),
            "world_from_camera": rotation.T,
            "camera": cameras[int(parts[8])],
        }
    heights = []
    points_file = model_dir / "points3D.txt"
    if points_file.is_file():
        for line in points_file.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                heights.append(float(line.split()[3]))
    if not heights:
        raise PipelineError("No sparse points available to estimate the ground plane")
    return views, float(np.median(heights))


def _ground_footprint_cells(model_dir: Path, names: list[str]) -> tuple[list, dict]:
    """Ground-grid cells inside each named view's footprint on the sparse ground plane."""
    import numpy as np
    from matplotlib.path import Path as Polygon

    views, ground_z = _read_text_model_views(model_dir)
    flying_height = float(np.median([views[name]["centre"][2] - ground_z for name in names]))
    if flying_height <= 0:
        raise PipelineError("Cameras are not above the estimated ground plane")
    max_range = 2.0 * flying_height
    cell = max(5.0, flying_height / 40.0)

    def footprint(view) -> np.ndarray:
        width, height, fx, fy, cx, cy = view["camera"]
        samples = np.linspace(0.0, 1.0, 13)
        border = np.concatenate([
            np.c_[samples * width, np.zeros_like(samples)],
            np.c_[np.full_like(samples, width), samples * height],
            np.c_[(1 - samples) * width, np.full_like(samples, height)],
            np.c_[np.zeros_like(samples), (1 - samples) * height],
        ])
        rays = np.c_[(border[:, 0] - cx) / fx, (border[:, 1] - cy) / fy, np.ones(len(border))]
        rays = rays @ view["world_from_camera"].T
        centre = view["centre"]
        outline = []
        for ray in rays:
            horizontal = np.hypot(ray[0], ray[1])
            scale = (ground_z - centre[2]) / ray[2] if ray[2] < 0 else np.inf
            reach = scale * horizontal
            if reach > max_range or not np.isfinite(reach):
                if horizontal < 1e-9:
                    continue
                scale = max_range / horizontal
            outline.append(centre[:2] + scale * ray[:2])
        return np.asarray(outline)

    outlines = [footprint(views[name]) for name in names]
    stacked = np.vstack([outline for outline in outlines if len(outline) >= 3])
    x0, y0 = stacked.min(axis=0)
    columns = int((stacked[:, 0].max() - x0) / cell) + 1
    rows = int((stacked[:, 1].max() - y0) / cell) + 1
    centres_x = x0 + (np.arange(columns) + 0.5) * cell
    centres_y = y0 + (np.arange(rows) + 0.5) * cell
    seen_cells = []
    for outline in outlines:
        if len(outline) < 3:
            seen_cells.append(np.array([], dtype=np.int64))
            continue
        (bx0, by0), (bx1, by1) = outline.min(axis=0), outline.max(axis=0)
        c0, c1 = np.searchsorted(centres_x, [bx0, bx1])
        r0, r1 = np.searchsorted(centres_y, [by0, by1])
        gx, gy = np.meshgrid(centres_x[c0:c1], centres_y[r0:r1])
        inside = Polygon(outline).contains_points(np.c_[gx.ravel(), gy.ravel()])
        cols, rws = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        seen_cells.append((rws.ravel() * columns + cols.ravel())[inside])
    return seen_cells, {
        "cell_count": rows * columns,
        "ground_plane_z": round(ground_z, 2),
        "flying_height_m": round(flying_height, 1),
        "grid_cell_m": round(cell, 2),
    }


def _coverage_reference_selection(
    model_dir: Path,
    reference_names: list[str],
    target: int,
    min_views: int,
) -> tuple[list[int], dict]:
    """Choose depth-map reference views so every patch of ground gets depth maps.

    Each view's ground footprint is projected onto the sparse-point ground plane
    and rasterised. Views are then picked greedily, each time taking the one that
    adds the most ground still seen by fewer than ``min_views`` chosen references
    (capped at how many views can see that ground at all). Time-sliced selection
    ignores what each frame looks at, so on long flights over large areas whole
    patches end up below the fusion step's minimum and vanish from the model.
    """
    import numpy as np

    views, _ = _read_text_model_views(model_dir)
    names = [name for name in reference_names if name in views]
    if len(names) < target:
        raise PipelineError("Too few reference views have poses for coverage selection")
    seen_cells, grid = _ground_footprint_cells(model_dir, names)
    available = np.zeros(grid["cell_count"], dtype=np.int32)
    for cells in seen_cells:
        available[cells] += 1
    wanted = np.minimum(available, min_views)
    covered = np.zeros_like(available)
    chosen: list[int] = []
    remaining = set(range(len(names)))
    while len(chosen) < target and remaining:
        best = max(remaining, key=lambda index: (
            int(np.count_nonzero(covered[seen_cells[index]] < wanted[seen_cells[index]])),
            -index,
        ))
        covered[seen_cells[best]] += 1
        chosen.append(best)
        remaining.discard(best)
    ground = available > 0
    satisfied = ground & (covered >= wanted)
    chosen_names = {names[index] for index in chosen}
    selected_indices = [index for index, name in enumerate(reference_names) if name in chosen_names]
    return selected_indices, {
        **{key: value for key, value in grid.items() if key != "cell_count"},
        "min_views_per_cell": min_views,
        "ground_cells": int(ground.sum()),
        "cells_meeting_min_views_percent": round(100.0 * satisfied.sum() / max(1, ground.sum()), 1),
        "cells_with_any_reference_percent": round(
            100.0 * np.count_nonzero(ground & (covered > 0)) / max(1, ground.sum()), 1),
    }


def _subsample_patch_match_references(
    dense: Path,
    stride: int,
    selection_mode: str = "uniform",
    frame_manifest: str | Path | None = None,
    max_source_images: int | None = None,
    target_references: int | None = None,
    model_dir: Path | None = None,
    coverage_min_views: int = 6,
) -> dict:
    """Configure dense anchors while retaining temporal coverage and source views."""
    if selection_mode not in {"uniform", "adaptive", "coverage"}:
        raise ValueError("selection_mode must be 'uniform', 'adaptive' or 'coverage'")
    coverage_report = None
    if max_source_images is not None and max_source_images < 2:
        raise ValueError("max_source_images must be at least 2")
    if target_references is not None and target_references < 2:
        raise ValueError("target_references must be at least 2")
    if target_references is not None and stride != 1:
        raise ValueError("target_references requires stride=1 to avoid ambiguous selection")
    config = dense / "stereo" / "patch-match.cfg"
    if not config.is_file():
        raise FileNotFoundError(config)
    lines = [line.strip() for line in config.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 2:
        raise PipelineError(f"Unexpected PatchMatch configuration format: {config}")
    pairs = [(lines[index], lines[index + 1]) for index in range(0, len(lines), 2)]
    mode_used = selection_mode
    if selection_mode == "coverage":
        mode_used = "adaptive"          # used when coverage selection is not possible
        if target_references is not None and target_references < len(pairs):
            try:
                if model_dir is None:
                    raise PipelineError("coverage selection needs the aligned text model")
                indices, coverage_report = _coverage_reference_selection(
                    Path(model_dir), [Path(pair[0]).name for pair in pairs],
                    target_references, coverage_min_views,
                )
                mode_used = "coverage"
            except (PipelineError, OSError, ValueError, KeyError, IndexError) as error:
                coverage_report = {"fallback": "adaptive", "reason": str(error)}
    if mode_used == "coverage":
        selected = [pairs[index] for index in indices]
    elif target_references is not None:
        target = min(target_references, len(pairs))
        if target >= len(pairs):
            selected = pairs
        elif mode_used == "uniform":
            # Inclusive endpoints retain the start and end of the flight,
            # while rounding gives a deterministic exact-size selection.
            indices = [round(index * (len(pairs) - 1) / (target - 1)) for index in range(target)]
            selected = [pairs[index] for index in indices]
        else:
            scores = _frame_geometry_scores(frame_manifest)
            if not scores:
                raise ValueError("adaptive dense anchors require a non-empty frame manifest")
            selected = []
            for index in range(target):
                start = index * len(pairs) // target
                end = max(start + 1, (index + 1) * len(pairs) // target)
                candidates = pairs[start:end]
                selected.append(max(
                    candidates,
                    key=lambda pair: (scores.get(Path(pair[0]).name, 0.0), pair[0]),
                ))
    elif stride <= 1:
        selected = pairs
    elif mode_used == "uniform":
        selected = pairs[::stride]
        if pairs and pairs[-1] not in selected:
            selected.append(pairs[-1])
    else:
        scores = _frame_geometry_scores(frame_manifest)
        if not scores:
            raise ValueError("adaptive dense anchors require a non-empty frame manifest")
        target = max(1, (len(pairs) + stride - 1) // stride)
        selected = []
        # Pick the strongest geometry view inside each temporal bin. Binning
        # prevents a high-score region from consuming all anchors and leaving
        # other parts of the flight without dense coverage.
        for index in range(target):
            start = index * len(pairs) // target
            end = max(start + 1, (index + 1) * len(pairs) // target)
            candidates = pairs[start:end]
            selected.append(max(
                candidates,
                key=lambda pair: (scores.get(Path(pair[0]).name, 0.0), pair[0]),
            ))
    selected = [
        (reference, _limit_patch_match_sources(sources, max_source_images))
        for reference, sources in selected
    ]
    config.write_text(
        "".join(f"{reference}\n{sources}\n" for reference, sources in selected),
        encoding="utf-8",
    )
    return {
        "stride": stride,
        "selection_mode": selection_mode,
        "selection_mode_used": mode_used,
        "max_source_images": max_source_images,
        "target_references": target_references,
        "original_references": len(pairs),
        "selected_references": len(selected),
        "coverage": coverage_report,
    }


def reconstruct(
    images_dir: str | Path,
    references_path: str | Path,
    workspace: str | Path,
    quality: str = "draft",
    make_mesh: bool = True,
    masks_dir: str | Path | None = None,
    mapper_mode: str = "standard",
    gps_prior_std_m: float = 5.0,
    gps_prior_horizontal_std_m: float | None = None,
    gps_prior_vertical_std_m: float | None = None,
    camera_calibration: dict | None = None,
    spatial_matching: bool = False,
    spatial_max_distance_m: float = 100.0,
    feature_type: str = "sift",
    feature_matcher: str = "bruteforce",
    sift_max_num_features: int | None = None,
    dense_frame_stride: int = 1,
    dense_anchor_mode: str = "uniform",
    dense_target_references: int | None = None,
    dense_source_images: int | None = None,
    dense_max_image_size: int | None = None,
    dense_num_iterations: int | None = None,
    dense_num_samples: int | None = None,
    dense_window_step: int | None = None,
    fusion_min_num_pixels: int = 5,
    delaunay_max_proj_dist: float | None = None,
    delaunay_max_depth_dist: float | None = None,
    delaunay_num_threads: int | None = None,
    mesher: str = "auto",
    mapper_ba_gpu: bool = False,
    mapper_single_model: bool = False,
    mapper_no_extract_colors: bool = False,
    mapper_ba_global_frames_ratio: float | None = None,
    mapper_ba_global_points_ratio: float | None = None,
    mapper_ba_global_max_refinements: int | None = None,
    mapper_ba_global_ignore_redundant_points3d: bool = False,
    mapper_ba_local_max_num_iterations: int | None = None,
    mapper_ba_global_max_num_iterations: int | None = None,
    mapper_random_seed: int | None = None,
    mapper_init_num_trials: int | None = None,
    mapper_fallback: str | None = None,
    global_mapper_min_registration_ratio: float = 0.6,
    global_mapper_options: dict | None = None,
    global_gps_refinement: dict | None = None,
    default_hfov_deg: float | None = None,
    single_model_min_registration_ratio: float = 0.5,
    min_fused_points_per_image: int = 100,
    frame_manifest: str | Path | None = None,
    reuse_sparse_model: str | Path | None = None,
    shared_feature_database: str | Path | None = None,
    sparse_only: bool = False,
    sparse_min_registration_ratio: float | None = None,
    sparse_max_reprojection_error_px: float | None = None,
    sparse_max_gps_alignment_rmse_m: float | None = None,
    telemetry_samples: list | None = None,
    georeference_origin: dict | None = None,
) -> dict:
    """Run an ordered, GPS-aligned COLMAP reconstruction suitable for Colab GPU."""
    images_dir, references_path, workspace = Path(images_dir), Path(references_path), Path(workspace)
    if quality not in {"draft", "full"}:
        raise ValueError("quality must be 'draft' or 'full'")
    if mapper_mode not in {"standard", "pose-prior", "global"}:
        raise ValueError("mapper_mode must be 'standard', 'pose-prior', or 'global'")
    if gps_prior_std_m <= 0:
        raise ValueError("gps_prior_std_m must be positive")
    gps_prior_horizontal_std_m = gps_prior_horizontal_std_m or gps_prior_std_m
    gps_prior_vertical_std_m = gps_prior_vertical_std_m or gps_prior_std_m
    if gps_prior_horizontal_std_m <= 0 or gps_prior_vertical_std_m <= 0:
        raise ValueError("GPS prior horizontal and vertical uncertainties must be positive")
    if spatial_max_distance_m <= 0:
        raise ValueError("spatial_max_distance_m must be positive")
    feature_type_enum, feature_matcher_enum = _feature_configuration(
        feature_type, feature_matcher
    )
    if sift_max_num_features is not None and sift_max_num_features < 512:
        raise ValueError("sift_max_num_features must be at least 512")
    if sift_max_num_features is not None and feature_type != "sift":
        raise ValueError("sift_max_num_features can only be used with SIFT")
    if not (
        sparse_min_registration_ratio is None
        or 0 < sparse_min_registration_ratio <= 1
    ):
        raise ValueError("sparse_min_registration_ratio must be in (0, 1]")
    if not (
        sparse_max_reprojection_error_px is None
        or sparse_max_reprojection_error_px > 0
    ):
        raise ValueError("sparse_max_reprojection_error_px must be positive")
    if not (
        sparse_max_gps_alignment_rmse_m is None
        or sparse_max_gps_alignment_rmse_m > 0
    ):
        raise ValueError("sparse_max_gps_alignment_rmse_m must be positive")
    if sparse_max_gps_alignment_rmse_m is not None and (
        telemetry_samples is None or georeference_origin is None or frame_manifest is None
    ):
        raise ValueError(
            "GPS sparse gating requires telemetry_samples, georeference_origin, and frame_manifest"
        )
    if mesher not in {"auto", "delaunay", "poisson", "advancing-front"}:
        raise ValueError(
            "mesher must be 'auto', 'delaunay', 'poisson', or 'advancing-front'"
        )
    if not make_mesh and mesher != "auto":
        raise ValueError("an explicit mesher cannot be used when make_mesh is false")
    if mesher in {"poisson", "advancing-front"} and any(
        value is not None
        for value in (
            delaunay_max_proj_dist,
            delaunay_max_depth_dist,
            delaunay_num_threads,
        )
    ):
        raise ValueError("Delaunay tuning options require --mesher auto or delaunay")
    if mapper_ba_global_frames_ratio is not None and mapper_ba_global_frames_ratio <= 1:
        raise ValueError("mapper_ba_global_frames_ratio must be greater than 1")
    if mapper_ba_global_points_ratio is not None and mapper_ba_global_points_ratio <= 1:
        raise ValueError("mapper_ba_global_points_ratio must be greater than 1")
    if mapper_ba_global_max_refinements is not None and mapper_ba_global_max_refinements < 1:
        raise ValueError("mapper_ba_global_max_refinements must be at least 1")
    for option_name, option_value in (
        ("mapper_ba_local_max_num_iterations", mapper_ba_local_max_num_iterations),
        ("mapper_ba_global_max_num_iterations", mapper_ba_global_max_num_iterations),
    ):
        if option_value is not None and option_value < 1:
            raise ValueError(f"{option_name} must be at least 1")
    if mapper_random_seed is not None and mapper_random_seed < 0:
        raise ValueError("mapper_random_seed must be non-negative")
    mapper_tuning_requested = any((
        mapper_single_model,
        mapper_no_extract_colors,
        mapper_ba_global_frames_ratio is not None,
        mapper_ba_global_points_ratio is not None,
        mapper_ba_global_max_refinements is not None,
        mapper_ba_global_ignore_redundant_points3d,
        mapper_ba_local_max_num_iterations is not None,
        mapper_ba_global_max_num_iterations is not None,
        mapper_random_seed is not None,
    ))
    if mapper_fallback not in {None, "standard", "pose-prior"}:
        raise ValueError("mapper_fallback must be None, 'standard', or 'pose-prior'")
    if mapper_fallback is not None and mapper_mode != "global":
        raise ValueError("mapper_fallback only applies to mapper_mode='global'")
    if not 0 < global_mapper_min_registration_ratio <= 1:
        raise ValueError("global_mapper_min_registration_ratio must be in (0, 1]")
    if default_hfov_deg is not None and not 20 <= default_hfov_deg <= 170:
        raise ValueError("default_hfov_deg must be between 20 and 170 degrees")
    if not 0 < single_model_min_registration_ratio <= 1:
        raise ValueError("single_model_min_registration_ratio must be in (0, 1]")
    if min_fused_points_per_image < 0:
        raise ValueError("min_fused_points_per_image must be non-negative")
    if global_gps_refinement is not None:
        from .gps_refinement import DEFAULT_OPTIONS as _gps_refinement_defaults

        if mapper_mode != "global":
            raise ValueError("global_gps_refinement only applies to mapper_mode='global'")
        unknown = set(global_gps_refinement) - set(_gps_refinement_defaults)
        if unknown:
            raise ValueError(f"Unknown global_gps_refinement options: {sorted(unknown)}")
        if any(float(value) <= 0 for value in global_gps_refinement.values()):
            raise ValueError("global_gps_refinement values must be positive")
    if mapper_mode == "global" and mapper_tuning_requested and mapper_fallback is None:
        raise ValueError("incremental Mapper tuning options cannot be used with mapper_mode='global'")
    if dense_frame_stride < 1:
        raise ValueError("dense_frame_stride must be at least 1")
    if dense_anchor_mode not in {"uniform", "adaptive", "coverage"}:
        raise ValueError("dense_anchor_mode must be 'uniform', 'adaptive' or 'coverage'")
    if dense_anchor_mode in {"adaptive", "coverage"} and frame_manifest is None:
        raise ValueError(f"{dense_anchor_mode} dense anchors require frame_manifest")
    if fusion_min_num_pixels < 2:
        raise ValueError("fusion_min_num_pixels must be at least 2")
    if dense_target_references is not None and dense_target_references < 2:
        raise ValueError("dense_target_references must be at least 2")
    if dense_target_references is not None and dense_frame_stride != 1:
        raise ValueError("dense_target_references requires dense_frame_stride=1")
    if dense_source_images is not None and dense_source_images < 2:
        raise ValueError("dense_source_images must be at least 2")
    if dense_max_image_size is not None and dense_max_image_size < 320:
        raise ValueError("dense_max_image_size must be at least 320")
    if dense_num_iterations is not None and dense_num_iterations < 1:
        raise ValueError("dense_num_iterations must be positive")
    if dense_num_samples is not None and dense_num_samples < 1:
        raise ValueError("dense_num_samples must be positive")
    if dense_window_step is not None and dense_window_step < 1:
        raise ValueError("dense_window_step must be positive")
    if delaunay_max_proj_dist is not None and delaunay_max_proj_dist < 0:
        raise ValueError("delaunay_max_proj_dist must be non-negative")
    if not (
        delaunay_max_depth_dist is None or 0 <= delaunay_max_depth_dist <= 1
    ):
        raise ValueError("delaunay_max_depth_dist must be between 0 and 1")
    if delaunay_num_threads is not None and delaunay_num_threads < -1:
        raise ValueError("delaunay_num_threads must be -1 or a positive integer")
    if delaunay_num_threads == 0:
        raise ValueError("delaunay_num_threads cannot be zero")
    if not images_dir.is_dir() or not any(images_dir.glob("*.jpg")):
        raise FileNotFoundError(f"No JPG frames found in {images_dir}")
    if not references_path.is_file():
        raise FileNotFoundError(references_path)
    reused_sparse_model = Path(reuse_sparse_model) if reuse_sparse_model else None
    if reused_sparse_model is not None and not reused_sparse_model.is_dir():
        raise FileNotFoundError(reused_sparse_model)
    workspace.mkdir(parents=True, exist_ok=True)
    colmap = _require_colmap()
    report: dict = {
        "quality": quality,
        "mapper_mode": mapper_mode,
        "mapper_fallback": mapper_fallback,
        "gps_prior_std_m": gps_prior_std_m if mapper_mode == "pose-prior" else None,
        "gps_prior_uncertainty_m": {
            "x": gps_prior_horizontal_std_m,
            "y": gps_prior_horizontal_std_m,
            "z": gps_prior_vertical_std_m,
        } if mapper_mode == "pose-prior" else None,
        "camera_calibration": camera_calibration,
        "spatial_matching": spatial_matching,
        "spatial_max_distance_m": spatial_max_distance_m if spatial_matching else None,
        "feature_type": feature_type,
        "feature_matcher": feature_matcher,
        "feature_type_colmap": feature_type_enum,
        "feature_matcher_colmap": feature_matcher_enum,
        "sift_max_num_features": sift_max_num_features,
        "dense_frame_stride": dense_frame_stride,
        "dense_anchor_mode": dense_anchor_mode,
        "dense_target_references": dense_target_references,
        "dense_source_images": dense_source_images,
        "dense_max_image_size": dense_max_image_size,
        "dense_num_iterations": dense_num_iterations,
        "dense_num_samples": dense_num_samples,
        "dense_window_step": dense_window_step,
        "fusion_min_num_pixels": fusion_min_num_pixels,
        "delaunay_max_proj_dist": delaunay_max_proj_dist,
        "delaunay_max_depth_dist": delaunay_max_depth_dist,
        "delaunay_num_threads": delaunay_num_threads,
        "mesher_requested": mesher,
        "mapper_ba_gpu_requested": mapper_ba_gpu,
        "mapper_tuning": {
            "single_model": mapper_single_model,
            "extract_colors": not mapper_no_extract_colors,
            "ba_global_frames_ratio": mapper_ba_global_frames_ratio,
            "ba_global_points_ratio": mapper_ba_global_points_ratio,
            "ba_global_max_refinements": mapper_ba_global_max_refinements,
            "ba_global_ignore_redundant_points3d": (
                mapper_ba_global_ignore_redundant_points3d
            ),
            "ba_local_max_num_iterations": mapper_ba_local_max_num_iterations,
            "ba_global_max_num_iterations": mapper_ba_global_max_num_iterations,
            "random_seed": mapper_random_seed,
        },
        "sparse_only": sparse_only,
        "reused_sparse_model": str(reused_sparse_model) if reused_sparse_model else None,
        "workspace": str(workspace),
        "stages": {},
        "_log_dir": str(workspace / "logs"),
        "_report_path": str(workspace / "run_report.partial.json"),
    }
    database = workspace / "database.db"
    sparse_root = workspace / "sparse"
    aligned = workspace / "aligned"
    dense = workspace / "dense"
    for directory in (sparse_root, aligned):
        directory.mkdir(parents=True, exist_ok=True)

    feature_gpu = (
        "--FeatureExtraction.use_gpu"
        if _help_has(colmap, "feature_extractor", "--FeatureExtraction.use_gpu")
        else "--SiftExtraction.use_gpu"
    )
    matching_gpu = (
        "--FeatureMatching.use_gpu"
        if _help_has(colmap, "sequential_matcher", "--FeatureMatching.use_gpu")
        else "--SiftMatching.use_gpu"
    )
    mask_options = ["--ImageReader.mask_path", str(Path(masks_dir))] if masks_dir else []
    image_reader_options = colmap_image_reader_options(camera_calibration)
    feature_type_options = []
    if feature_type != "sift" or _help_has(
        colmap, "feature_extractor", "--FeatureExtraction.type"
    ):
        if not _help_has(colmap, "feature_extractor", "--FeatureExtraction.type"):
            raise PipelineError(
                "Installed COLMAP does not support --FeatureExtraction.type; "
                "use SIFT or install a current ONNX-enabled COLMAP build"
            )
        feature_type_options = ["--FeatureExtraction.type", feature_type_enum]
    sift_limit_options = []
    if sift_max_num_features is not None:
        option = "--SiftExtraction.max_num_features"
        if not _help_has(colmap, "feature_extractor", option):
            raise PipelineError(f"Installed COLMAP does not support {option}")
        sift_limit_options = [option, str(sift_max_num_features)]
    matching_type_options = []
    if feature_matcher != "bruteforce" or feature_type != "sift" or _help_has(
        colmap, "sequential_matcher", "--FeatureMatching.type"
    ):
        if not _help_has(colmap, "sequential_matcher", "--FeatureMatching.type"):
            raise PipelineError(
                "Installed COLMAP does not support --FeatureMatching.type; "
                "use SIFT brute-force or install a current ONNX-enabled COLMAP build"
            )
        matching_type_options = ["--FeatureMatching.type", feature_matcher_enum]
    if shared_feature_database and reused_sparse_model is None:
        from .feature_cache import isolate_database, image_digest
        import json
        if (camera_calibration or masks_dir or feature_type != "sift"
                or feature_matcher != "bruteforce" or quality != "draft"
                or sift_max_num_features != 4096):
            raise ValueError("Shared frontend requires unmasked draft SIFT/4096 with default OPENCV camera")
        master = Path(shared_feature_database)
        metadata = json.loads((master.parent / "complete.json").read_text())
        names = sorted(p.name for p in images_dir.glob("*.jpg"))
        if metadata.get("sha256") != image_digest(master):
            raise ValueError("Shared frontend database checksum mismatch")
        identity = metadata["identity"]
        if any(identity.get(k) != v for k, v in
               {"schema": 1, "camera": "OPENCV", "features": 4096, "overlap": 10}.items()):
            raise ValueError("Shared frontend configuration mismatch")
        cached_images = identity["images"]
        if any(cached_images.get(name) != image_digest(images_dir / name) for name in names):
            raise ValueError("Shared frontend image content mismatch")
        isolate_database(master, database, names)
        report["feature_cache"] = {"source": str(master), "images": len(names),
                                   "skipped_stages": ["feature_extraction", "sequential_matching"]}
    elif reused_sparse_model is None:
        run_command(
            "feature_extraction",
            [
                colmap,
                "feature_extractor",
                "--database_path",
                str(database),
                "--image_path",
                str(images_dir),
                *image_reader_options,
                *mask_options,
                *feature_type_options,
                *sift_limit_options,
                feature_gpu,
                "1",
            ],
            report,
        )
        report["camera_prior"] = _apply_camera_prior(
            database, default_hfov_deg, camera_calibration is not None)
        run_command(
            "sequential_matching",
            [
                colmap,
                "sequential_matcher",
                "--database_path",
                str(database),
                "--SequentialMatching.overlap",
                "10" if quality == "draft" else "20",
                *matching_type_options,
                matching_gpu,
                "1",
            ],
            report,
        )
    else:
        report["sparse_reuse"] = {
            "enabled": True,
            "source_model": str(reused_sparse_model),
            "skipped_stages": ["feature_extraction", "sequential_matching", "sparse_mapping"],
        }
    if spatial_matching and reused_sparse_model is None:
        import subprocess

        spatial_help_result = subprocess.run(
            [colmap, "spatial_matcher", "--help"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if spatial_help_result.returncode != 0:
            raise PipelineError("This COLMAP build does not support spatial_matcher")
        spatial_help = (spatial_help_result.stdout or "") + (spatial_help_result.stderr or "")
        spatial_options = []
        if "SpatialMatching.max_distance" in spatial_help:
            spatial_options += ["--SpatialMatching.max_distance", str(spatial_max_distance_m)]
        run_command(
            "spatial_matching",
            [
                colmap,
                "spatial_matcher",
                "--database_path",
                str(database),
                *matching_type_options,
                *spatial_options,
            ],
            report,
        )

    def _run_mapper(mode: str, stage_name: str, allow_multiple_models: bool = False) -> None:
        mapper_command_name = (
            "pose_prior_mapper" if mode == "pose-prior"
            else "global_mapper" if mode == "global"
            else "mapper"
        )
        # Forcing --Mapper.init_image_id1/2 from the pair with the most raw inlier
        # matches was tried and reverted: COLMAP rejects a pair as an initializer
        # for reasons a match count alone can't see (mainly insufficient
        # triangulation angle -- two near-adjacent, barely-moved frames can have
        # huge inlier counts with almost no parallax to triangulate from). Once a
        # specific pair is forced, COLMAP no longer falls back to searching for a
        # different one if it's rejected; it retries the same bad pair until the
        # trial cap and then fails outright, instead of the working reconstruction
        # its own free search would have found. Leave the pair search to COLMAP;
        # only bound its per-attempt trial count below.
        mapper_help = ""
        if mode in {"pose-prior", "global"} or mapper_ba_gpu or mapper_tuning_requested:
            import subprocess

            mapper_help_result = subprocess.run(
                [colmap, mapper_command_name, "--help"],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            mapper_help = (mapper_help_result.stdout or "") + (mapper_help_result.stderr or "")
            if mapper_help_result.returncode != 0 and mode in {"pose-prior", "global"}:
                raise PipelineError(
                    f"This COLMAP build does not support {mapper_command_name}; use a current COLMAP build"
                )
        if mode == "global" and reused_sparse_model is None:
            calibrator_help = __import__("subprocess").run(
                [colmap, "view_graph_calibrator", "--help"],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            if calibrator_help.returncode != 0:
                raise PipelineError(
                    "Global SfM requires COLMAP view_graph_calibrator for this controlled experiment"
                )
            run_command(
                "view_graph_calibration",
                [colmap, "view_graph_calibrator", "--database_path", str(database)],
                report,
            )

        mapper_options = [] if mode == "global" else [
            "--Mapper.ba_refine_focal_length",
            "0" if camera_calibration and camera_calibration.get("fixed_intrinsics") else "1",
        ]
        if mode != "global" and camera_calibration and camera_calibration.get("fixed_intrinsics"):
            mapper_options += [
                "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0",
            ]
        if mode == "pose-prior" and "overwrite_priors_covariance" in mapper_help:
            mapper_options += ["--overwrite_priors_covariance", "1"]
            for axis, uncertainty in (
                ("x", gps_prior_horizontal_std_m),
                ("y", gps_prior_horizontal_std_m),
                ("z", gps_prior_vertical_std_m),
            ):
                mapper_options += [f"--prior_position_std_{axis}", str(uncertainty)]
        requested_mapper_options: list[tuple[str, str]] = []
        if mapper_single_model and not allow_multiple_models:
            requested_mapper_options.append(("--Mapper.multiple_models", "0"))
        if mapper_no_extract_colors:
            requested_mapper_options.append(("--Mapper.extract_colors", "0"))
        if mapper_ba_global_frames_ratio is not None:
            requested_mapper_options.append(
                ("--Mapper.ba_global_frames_ratio", str(mapper_ba_global_frames_ratio))
            )
        if mapper_ba_global_points_ratio is not None:
            requested_mapper_options.append(
                ("--Mapper.ba_global_points_ratio", str(mapper_ba_global_points_ratio))
            )
        if mapper_ba_global_max_refinements is not None:
            requested_mapper_options.append(
                ("--Mapper.ba_global_max_refinements", str(mapper_ba_global_max_refinements))
            )
        if mapper_ba_global_ignore_redundant_points3d:
            requested_mapper_options.append(
                ("--Mapper.ba_global_ignore_redundant_points3D", "1")
            )
        if mapper_ba_local_max_num_iterations is not None:
            requested_mapper_options.append(
                ("--Mapper.ba_local_max_num_iterations", str(mapper_ba_local_max_num_iterations))
            )
        if mapper_ba_global_max_num_iterations is not None:
            requested_mapper_options.append(
                ("--Mapper.ba_global_max_num_iterations", str(mapper_ba_global_max_num_iterations))
            )
        if mapper_random_seed is not None:
            requested_mapper_options.append(
                ("--Mapper.random_seed", str(mapper_random_seed))
            )
        if mode == "global":
            # Incremental-only knobs; with a fallback configured they still
            # apply to the incremental retry.
            requested_mapper_options = []
        unsupported_mapper_options = []
        for option, value in requested_mapper_options:
            if option.lstrip("-") not in mapper_help:
                # These knobs are optional performance hints.  COLMAP occasionally
                # renames or removes them between releases; the standard mapper is
                # still valid without them, so retain the quality gates and record
                # the compatibility fallback instead of aborting before mapping.
                unsupported_mapper_options.append(option)
                continue
            mapper_options += [option, value]
        report["mapper_tuning"]["unsupported_options_skipped"] = (
            unsupported_mapper_options
        )
        if unsupported_mapper_options:
            print(
                "Skipping mapper options unsupported by this COLMAP build: "
                + ", ".join(unsupported_mapper_options),
                flush=True,
            )
        init_num_trials_applied = None
        if mapper_init_num_trials is not None and mode != "global":
            if _help_has(colmap, mapper_command_name, "--Mapper.init_num_trials"):
                mapper_options += ["--Mapper.init_num_trials", str(mapper_init_num_trials)]
                init_num_trials_applied = mapper_init_num_trials
            else:
                unsupported_mapper_options.append("--Mapper.init_num_trials")
        report["mapper_tuning"]["init_num_trials"] = init_num_trials_applied
        ba_gpu_options = []
        if mode == "global":
            global_applied, global_skipped = {}, []
            for key, value in (global_mapper_options or {}).items():
                option = f"--GlobalMapper.{key}"
                if option.lstrip("-") in mapper_help:
                    mapper_options += [option, str(value)]
                    global_applied[key] = value
                else:
                    global_skipped.append(option)
            report["mapper_tuning"]["global_options_applied"] = global_applied
            report["mapper_tuning"]["global_options_skipped"] = global_skipped
            for option in ("--GlobalMapper.gp_use_gpu", "--GlobalMapper.ba_ceres_use_gpu"):
                if option.lstrip("-") in mapper_help:
                    ba_gpu_options.append(option)
            if mapper_ba_gpu:
                for option in ba_gpu_options:
                    mapper_options += [option, "1"]
            ba_gpu_option_name = ",".join(ba_gpu_options) if ba_gpu_options else None
        else:
            ba_gpu_option = re.search(r"(--[\w.]*ba_use_gpu)\b", mapper_help)
            if mapper_ba_gpu and ba_gpu_option:
                mapper_options += [ba_gpu_option.group(1), "1"]
            ba_gpu_options = [ba_gpu_option.group(1)] if ba_gpu_option else []
            ba_gpu_option_name = ba_gpu_option.group(1) if ba_gpu_option else None
        report["mapper_ba_gpu"] = {
            "requested": mapper_ba_gpu,
            "supported": bool(ba_gpu_options),
            "enabled": bool(mapper_ba_gpu and ba_gpu_options),
            "option": ba_gpu_option_name,
        }
        run_command(
            stage_name,
            [
                colmap,
                mapper_command_name,
                "--database_path",
                str(database),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_root),
                *mapper_options,
            ],
            report,
        )

    mapper_attempts = [mapper_mode]
    if mapper_mode == "global" and mapper_fallback is not None:
        mapper_attempts.append(mapper_fallback)
    report["mapper_attempts"] = []
    report["mapper_mode_effective"] = mapper_mode
    if reused_sparse_model is None:
        image_count = len(list(images_dir.glob("*.jpg")))
        database_backup = workspace / "database.before_global_mapper.db"
        for attempt_index, mode in enumerate(mapper_attempts):
            is_last = attempt_index == len(mapper_attempts) - 1
            # verify_outputs.py's REQUIRED_STAGES hardcodes the literal stage
            # name "sparse_mapping". Whichever attempt is ultimately accepted
            # -- global on the first try, or the fallback after it -- must be
            # the one left behind under that exact key, or artifact
            # verification fails even on a fully successful run. Every
            # attempt therefore runs under "sparse_mapping"; a rejected
            # attempt's stage entry is renamed out of the way before retrying.
            stage_name = "sparse_mapping"
            attempt = {"mode": mode, "stage": stage_name}
            report["mapper_attempts"].append(attempt)
            if not is_last:
                # view_graph_calibrator rewrites the database; the incremental
                # retry must start from the untouched matches.
                shutil.copy2(database, database_backup)
            try:
                _run_mapper(mode, stage_name)
                if is_last:
                    attempt["accepted"] = True
                    report["mapper_mode_effective"] = mode
                    break
                analysis = run_command(
                    f"sparse_mapping_{mode}_analysis",
                    [colmap, "model_analyzer", "--path", str(_largest_model(sparse_root))],
                    report,
                )
                registered_match = re.search(r"Registered images:\s*(\d+)", analysis)
                ratio = int(registered_match.group(1)) / image_count if registered_match else 0.0
                attempt["registration_ratio"] = round(ratio, 4)
                attempt["accepted"] = ratio >= global_mapper_min_registration_ratio
                if attempt["accepted"]:
                    report["mapper_mode_effective"] = mode
                    break
                print(
                    f"Global mapper registered {ratio:.1%} of frames (< "
                    f"{global_mapper_min_registration_ratio:.0%}); retrying with {mapper_attempts[-1]}",
                    flush=True,
                )
            except PipelineError as error:
                if is_last:
                    raise
                attempt["accepted"] = False
                attempt["error"] = str(error)
                print(f"Global mapper unavailable or failed ({error}); retrying with "
                      f"{mapper_attempts[-1]}", flush=True)
            report["stages"][f"sparse_mapping_{mode}_rejected"] = report["stages"].pop(stage_name)
            shutil.rmtree(sparse_root)
            sparse_root.mkdir(parents=True)
            shutil.copy2(database_backup, database)

    # Stopping after the first model saves the minutes COLMAP otherwise spends
    # starting doomed extra models from frames it cannot place (hover, takeoff),
    # but only helps if that first model is the real one. If it is small -- a bad
    # initial pair -- map again with extra models allowed.
    report["single_model_retry"] = None
    if (reused_sparse_model is None and mapper_single_model
            and report["mapper_mode_effective"] in {"standard", "pose-prior"}):
        try:
            first_model = run_command(
                "sparse_mapping_single_model_check",
                [colmap, "model_analyzer", "--path", str(_largest_model(sparse_root))], report)
            registered_match = re.search(r"Registered images:\s*(\d+)", first_model)
            first_ratio = int(registered_match.group(1)) / image_count if registered_match else 0.0
        except PipelineError:
            first_ratio = 0.0
        report["single_model_retry"] = {
            "first_model_registration_ratio": round(first_ratio, 4),
            "threshold": single_model_min_registration_ratio,
            "retried": first_ratio < single_model_min_registration_ratio,
        }
        if first_ratio < single_model_min_registration_ratio:
            print(f"First model registered {first_ratio:.1%} of frames (< "
                  f"{single_model_min_registration_ratio:.0%}); mapping again with extra models allowed",
                  flush=True)
            report["stages"]["sparse_mapping_single_model_rejected"] = report["stages"].pop("sparse_mapping")
            shutil.rmtree(sparse_root)
            sparse_root.mkdir(parents=True)
            _run_mapper(report["mapper_mode_effective"], "sparse_mapping", allow_multiple_models=True)

    report["gps_refinement"] = {
        "requested": global_gps_refinement,
        "enabled": (global_gps_refinement is not None and reused_sparse_model is None
                    and report["mapper_mode_effective"] == "global"),
        "applied": False,
    }
    if report["gps_refinement"]["enabled"]:
        from .gps_refinement import refine_global_model

        global_model = _largest_model(sparse_root)
        refine_dir = workspace / "gps_refinement"
        try:
            refined_model = refine_global_model(
                colmap, database, images_dir, global_model, references_path,
                refine_dir, report, global_gps_refinement)
        except Exception as error:  # the unrefined global model is still a valid result
            report["gps_refinement"]["error"] = f"{type(error).__name__}: {error}"
            print(f"GPS refinement not applied ({error}); keeping the global-mapper model", flush=True)
        else:
            # Keep the unrefined model for diagnosis; the refined one becomes the only model.
            shutil.copytree(global_model, refine_dir / "global_model")
            shutil.rmtree(sparse_root)
            sparse_root.mkdir(parents=True)
            shutil.copytree(refined_model, sparse_root / "0")
            report["gps_refinement"]["applied"] = True

    sparse_model = reused_sparse_model or _largest_model(sparse_root)
    analyzer_output = run_command(
        "sparse_analysis", [colmap, "model_analyzer", "--path", str(sparse_model)], report
    )
    registered = re.search(r"Registered images:\s*(\d+)", analyzer_output)
    reprojection = re.search(r"Mean reprojection error:\s*([\d.]+)", analyzer_output)
    report["sparse_metrics"] = {
        "registered_images": int(registered.group(1)) if registered else None,
        "mean_reprojection_error_px": float(reprojection.group(1)) if reprojection else None,
    }
    selected_images = len(list(images_dir.glob("*.jpg")))
    registered_images = report["sparse_metrics"]["registered_images"]
    reprojection_error = report["sparse_metrics"]["mean_reprojection_error_px"]
    registration_ratio = (
        registered_images / selected_images
        if registered_images is not None and selected_images else 0.0
    )
    sparse_gate = {
        "selected_images": selected_images,
        "registered_images": registered_images,
        "registration_ratio": registration_ratio,
        "mean_reprojection_error_px": reprojection_error,
        "minimum_registration_ratio": sparse_min_registration_ratio,
        "maximum_reprojection_error_px": sparse_max_reprojection_error_px,
        "maximum_gps_alignment_rmse_m": sparse_max_gps_alignment_rmse_m,
        "registration_passed": (
            sparse_min_registration_ratio is None
            or registration_ratio >= sparse_min_registration_ratio
        ),
        "reprojection_passed": (
            sparse_max_reprojection_error_px is None
            or (
                reprojection_error is not None
                and reprojection_error <= sparse_max_reprojection_error_px
            )
        ),
    }
    align_command = [
        colmap,
        "model_aligner",
        "--input_path",
        str(sparse_model),
        "--output_path",
        str(aligned),
        "--ref_images_path",
        str(references_path),
        "--ref_is_gps",
        "1",
        "--alignment_type",
        "enu",
    ]
    if _help_has(colmap, "model_aligner", "--alignment_max_error"):
        align_command += ["--alignment_max_error", "5.0"]
    run_command("gps_alignment", align_command, report)
    aligned_text = workspace / "aligned_text"
    aligned_text.mkdir(parents=True, exist_ok=True)
    run_command(
        "aligned_model_text_export",
        [
            colmap,
            "model_converter",
            "--input_path",
            str(aligned),
            "--output_path",
            str(aligned_text),
            "--output_type",
            "TXT",
        ],
        report,
    )
    gps_alignment = None
    if telemetry_samples is not None and georeference_origin is not None and frame_manifest is not None:
        from .validation import gps_alignment_report

        gps_alignment = gps_alignment_report(
            aligned_text / "images.txt",
            frame_manifest,
            telemetry_samples,
            georeference_origin,
        )
    gps_rmse = gps_alignment.get("gps_alignment_rmse_m") if gps_alignment else None
    sparse_gate["gps_alignment"] = gps_alignment
    sparse_gate["gps_alignment_passed"] = (
        sparse_max_gps_alignment_rmse_m is None
        or (gps_rmse is not None and gps_rmse <= sparse_max_gps_alignment_rmse_m)
    )
    sparse_gate["passed"] = bool(
        sparse_gate["registration_passed"]
        and sparse_gate["reprojection_passed"]
        and sparse_gate["gps_alignment_passed"]
    )
    report["sparse_quality_gate"] = sparse_gate
    write_json(
        Path(report["_report_path"]),
        {key: value for key, value in report.items() if not key.startswith("_")},
    )
    if not sparse_gate["passed"]:
        raise PipelineError(
            "Sparse quality gate failed before dense stereo: "
            f"registration_ratio={registration_ratio:.3f}, "
            f"reprojection_error_px={reprojection_error}, gps_alignment_rmse_m={gps_rmse}"
        )
    if sparse_only:
        report["outputs"] = {
            "sparse_model": str(sparse_model),
            "aligned_model": str(aligned),
            "aligned_images_txt": str(aligned_text / "images.txt"),
        }
        report["total_seconds"] = round(
            sum(stage["seconds"] for stage in report["stages"].values()), 2
        )
        public_report = {key: value for key, value in report.items() if not key.startswith("_")}
        write_json(workspace / "run_report.json", public_report)
        return public_report
    run_command(
        "image_undistortion",
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(aligned),
            "--output_path",
            str(dense),
            "--output_type",
            "COLMAP",
        ],
        report,
    )
    report["dense_reference_selection"] = _subsample_patch_match_references(
        dense,
        dense_frame_stride,
        selection_mode=dense_anchor_mode,
        frame_manifest=frame_manifest,
        max_source_images=dense_source_images,
        target_references=dense_target_references,
        model_dir=aligned_text,
        # Fusion keeps a point only when this many depth maps agree, so aim for
        # one more reference than that over every patch of ground.
        coverage_min_views=fusion_min_num_pixels + 1,
    )
    patch_size = dense_max_image_size or (1200 if quality == "draft" else 2000)
    patch_options = [
        "--PatchMatchStereo.max_image_size", str(patch_size),
        "--PatchMatchStereo.geom_consistency", "false" if quality == "draft" else "true",
        "--PatchMatchStereo.filter", "true",
    ]
    if dense_num_iterations is not None:
        patch_options += ["--PatchMatchStereo.num_iterations", str(dense_num_iterations)]
    if dense_num_samples is not None:
        patch_options += ["--PatchMatchStereo.num_samples", str(dense_num_samples)]
    if dense_window_step is not None:
        if not _help_has(colmap, "patch_match_stereo", "--PatchMatchStereo.window_step"):
            raise PipelineError("Installed COLMAP does not support --PatchMatchStereo.window_step")
        patch_options += ["--PatchMatchStereo.window_step", str(dense_window_step)]
    fusion_options = []
    if dense_max_image_size is not None and _help_has(
        colmap, "stereo_fusion", "--StereoFusion.max_image_size"
    ):
        fusion_options = ["--StereoFusion.max_image_size", str(dense_max_image_size)]
    run_command(
        "dense_stereo",
        [
            colmap,
            "patch_match_stereo",
            "--workspace_path",
            str(dense),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.gpu_index",
            "0",
            *patch_options,
        ],
        report,
    )
    fused = dense / "fused.ply"
    fusion_log = run_command(
        "stereo_fusion",
        [
            colmap,
            "stereo_fusion",
            "--workspace_path",
            str(dense),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric" if quality == "full" else "photometric",
            "--output_path",
            str(fused),
            "--StereoFusion.min_num_pixels",
            str(fusion_min_num_pixels),
            *fusion_options,
        ],
        report,
    )
    # A few hundred fused points from a hundred-plus frames means the camera
    # poses are wrong (collapsed sparse model), not a sparse scene. Meshing that
    # would only waste minutes and ship a meaningless model, so stop here.
    fused_match = re.search(r"Number of fused points:\s*(\d+)", fusion_log or "")
    fused_points = int(fused_match.group(1)) if fused_match else None
    minimum_fused = max(5000, min_fused_points_per_image * registered_images)
    report["dense_metrics"] = {"fused_points": fused_points, "minimum_fused_points": minimum_fused}
    if fused_points is not None and fused_points < minimum_fused:
        raise PipelineError(
            f"Dense reconstruction collapsed: {fused_points} fused points from {registered_images} "
            f"registered images (minimum {minimum_fused}). The sparse camera poses are most likely wrong."
        )
    mesh = dense / "mesh.ply"
    textured_mesh = None
    texture_image = None
    if make_mesh:
        delaunay_options = []
        requested_delaunay_options = [
            ("--DelaunayMeshing.max_proj_dist", delaunay_max_proj_dist),
            ("--DelaunayMeshing.max_depth_dist", delaunay_max_depth_dist),
            ("--DelaunayMeshing.num_threads", delaunay_num_threads),
        ]
        for option, value in requested_delaunay_options:
            if value is None:
                continue
            if not _help_has(colmap, "delaunay_mesher", option):
                raise PipelineError(f"Installed COLMAP does not support requested option {option}")
            delaunay_options += [option, str(value)]
        selected_mesher = (
            "delaunay" if mesher == "auto" and quality == "draft"
            else "poisson" if mesher == "auto"
            else mesher
        )
        if selected_mesher == "delaunay":
            # Delaunay is the fast, robust preview mesher. COLMAP 4.2's
            # Poisson surface trimmer can segfault on otherwise valid clouds.
            run_command(
                "delaunay_meshing",
                [
                    colmap,
                    "delaunay_mesher",
                    "--input_path",
                    str(dense),
                    "--output_path",
                    str(mesh),
                    "--input_type",
                    "dense",
                    *delaunay_options,
                ],
                report,
            )
            report["meshing_method"] = (
                "delaunay_draft" if mesher == "auto" else "delaunay"
            )
        elif selected_mesher == "advancing-front":
            import subprocess

            advancing_help = subprocess.run(
                [colmap, "advancing_front_mesher", "--help"],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            if advancing_help.returncode != 0:
                raise PipelineError(
                    "Installed COLMAP does not provide advancing_front_mesher; "
                    "install a current CGAL-enabled build"
                )
            run_command(
                "advancing_front_meshing",
                [
                    colmap,
                    "advancing_front_mesher",
                    "--input_path",
                    str(dense),
                    "--output_path",
                    str(mesh),
                ],
                report,
            )
            report["meshing_method"] = "advancing_front"
        else:
            try:
                run_command(
                    "poisson_meshing",
                    [colmap, "poisson_mesher", "--input_path", str(fused), "--output_path", str(mesh)],
                    report,
                )
                report["meshing_method"] = "poisson"
            except PipelineError as error:
                if mesher == "poisson":
                    raise
                report["poisson_meshing_error"] = str(error)
                run_command(
                    "delaunay_meshing",
                    [
                        colmap,
                        "delaunay_mesher",
                        "--input_path",
                        str(dense),
                        "--output_path",
                        str(mesh),
                        "--input_type",
                        "dense",
                        *delaunay_options,
                    ],
                    report,
                )
                report["meshing_method"] = "delaunay_fallback"
        cleaned_mesh = dense / "mesh_cleaned.ply"
        report["mesh_cleanup"] = _clean_mesher_output(mesh, cleaned_mesh, report)
        mesh = cleaned_mesh
        if _help_has(colmap, "mesh_texturer", "--output_path"):
            textured_dir = dense / "textured"
            try:
                run_command(
                    "mesh_texturing",
                    [
                        colmap,
                        "mesh_texturer",
                        "--workspace_path",
                        str(dense),
                        "--input_path",
                        str(mesh),
                        "--output_path",
                        str(textured_dir),
                    ],
                    report,
                )
                candidate_mesh = textured_dir / "mesh.ply"
                candidate_texture = textured_dir / "texture.png"
                textured_mesh = candidate_mesh if candidate_mesh.is_file() else None
                texture_image = candidate_texture if candidate_texture.is_file() else None
            except PipelineError as error:
                # Geometry remains useful if a particular COLMAP build cannot
                # texture a cleaned PLY; report the failure without losing it.
                report["mesh_texturing_error"] = str(error)
    report["outputs"] = {
        "point_cloud_ply": str(fused),
        "mesh_ply": str(mesh) if make_mesh else None,
        "textured_mesh_ply": str(textured_mesh) if textured_mesh else None,
        "texture_image": str(texture_image) if texture_image else None,
        "aligned_images_txt": str(aligned_text / "images.txt"),
    }
    report["total_seconds"] = round(sum(stage["seconds"] for stage in report["stages"].values()), 2)
    public_report = {key: value for key, value in report.items() if not key.startswith("_")}
    write_json(workspace / "run_report.json", public_report)
    return public_report
