from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .utils import PipelineError, write_json


def _sha256_file(path: str | Path) -> str:
    """Return a stable content identity for strict benchmark comparisons."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _independent_surface_target(validation: dict) -> tuple[bool | None, str]:
    """Combine every available independent surface check conservatively."""
    outcomes: list[bool] = []
    evidence: list[str] = []
    for key, label in (
        ("surveyed_checkpoints", "independent surveyed 3D checkpoints"),
        ("independent_distances", "independent surveyed distances"),
    ):
        value = validation.get(key, {}).get("passes_one_metre_target")
        if isinstance(value, bool):
            outcomes.append(value)
            evidence.append(label)

    dense = validation.get("dense_geometry", {})
    dense_value = dense.get("passes_one_metre_target")
    if (
        isinstance(dense_value, bool)
        and dense.get("reference_is_distinct_from_reconstruction") is not False
    ):
        outcomes.append(dense_value)
        evidence.append("independent dense reference geometry")

    if not outcomes:
        return (
            None,
            "none - camera trajectory is diagnostic only; supply surveyed measurements or reference geometry",
        )
    return all(outcomes), "; ".join(evidence)


def _run(args: argparse.Namespace) -> int:
    from .ai_masking import mask_dynamic_objects
    from .camera import calibration_for_image_size, load_camera_calibration
    from .capture_quality import analyze_selected_frames
    from .colmap_pipeline import reconstruct
    from .control_points import fit_control_point_transform
    from .export_gis import export_products
    from .extract_keyframes import extract_keyframes
    from .profiles import apply_processing_profile, resolve_adaptive_dense_settings
    from .production_status import RuntimeLedger
    from .telemetry import embed_frame_gps_exif, load_telemetry, write_frame_references
    from .validation import (
        gps_alignment_report, ground_truth_trajectory_report,
        independent_distance_report, surveyed_checkpoint_report,
    )

    started = time.perf_counter()
    runtime_ledger = RuntimeLedger()
    output_dir = Path(args.output).resolve()
    workspace = Path(args.workspace).resolve()
    frames_dir = workspace / "images"
    frames_manifest = workspace / "frames.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow_steps = {}
    if args.gcp_max_scale_change_percent <= 0:
        raise ValueError("--gcp-max-scale-change-percent must be positive")
    if (
        args.gcp_control_points
        and args.validation_checkpoints
        and Path(args.gcp_control_points).resolve() == Path(args.validation_checkpoints).resolve()
    ):
        raise ValueError(
            "GCP control points and independent validation checkpoints must be different files"
        )
    processing_profile = apply_processing_profile(args)
    telemetry = load_telemetry(args.telemetry, altitude_offset_m=args.telemetry_altitude_offset_m)
    horizontal_accuracy = [
        sample.horizontal_accuracy_m for sample in telemetry
        if sample.horizontal_accuracy_m is not None
    ]
    vertical_accuracy = [
        sample.vertical_accuracy_m for sample in telemetry
        if sample.vertical_accuracy_m is not None
    ]
    gps_prior_horizontal_std_m = (
        args.gps_prior_horizontal_std_m
        or (float(np.median(horizontal_accuracy)) if horizontal_accuracy else args.gps_prior_std_m)
    )
    gps_prior_vertical_std_m = (
        args.gps_prior_vertical_std_m
        or (float(np.median(vertical_accuracy)) if vertical_accuracy else args.gps_prior_std_m)
    )
    camera_calibration = load_camera_calibration(args.camera_calibration)
    step_started = time.perf_counter()
    print("1/5 Extracting ordered keyframes")
    frames = extract_keyframes(
        args.video,
        frames_dir,
        target_frames=args.target_frames,
        max_width=args.max_width,
        manifest_path=frames_manifest,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames or (600 if args.quality == "draft" else 1200),
        selection_mode=args.keyframe_mode,
        # Uniform mode ignores telemetry when scoring but still records the
        # selected GPS baselines for a fair A/B report.
        telemetry_samples=telemetry,
        decode_mode=args.keyframe_decode_mode,
        reuse_existing=args.reuse_keyframes,
        adaptive_screening_timeout_s=args.adaptive_screening_timeout_s,
        adaptive_target_candidates=args.adaptive_target_candidates,
    )
    import cv2

    metadata_frame = cv2.imread(str(frames_dir / frames[0].image_name))
    if metadata_frame is None:
        raise PipelineError("Could not read the first extracted frame")
    frame_height, frame_width = metadata_frame.shape[:2]
    if args.video_sha256 is not None:
        normalized_hash = args.video_sha256.strip().lower()
        if len(normalized_hash) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_hash
        ):
            raise ValueError("--video-sha256 must contain exactly 64 hexadecimal characters")
        video_sha256 = normalized_hash
    else:
        video_sha256 = _sha256_file(args.video)
    if camera_calibration is not None:
        camera_calibration = calibration_for_image_size(
            camera_calibration, frame_width, frame_height
        )
    selection_path = frames_manifest.with_suffix(".selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
    workflow_steps["keyframe_extraction"] = {
        "status": "passed", "seconds": round(time.perf_counter() - step_started, 2), "frames": len(frames),
        "sharpness_median": float(np.median([frame.sharpness for frame in frames])),
        "motion_score_median": float(np.median([frame.motion_score for frame in frames])),
        **selection,
    }
    adaptive_settings = resolve_adaptive_dense_settings(args, len(frames))
    capture_quality_path = workspace / "capture_quality.json"
    step_started = time.perf_counter()
    if args.reuse_capture_quality and capture_quality_path.is_file():
        capture_quality = json.loads(capture_quality_path.read_text(encoding="utf-8"))
        capture_quality_reused = True
    else:
        capture_quality = analyze_selected_frames(
            frames_dir, frames, telemetry, capture_quality_path, camera_calibration
        )
        capture_quality_reused = False
    workflow_steps["capture_quality"] = {
        "status": "passed" if capture_quality["ready"] else "warning",
        "seconds": round(time.perf_counter() - step_started, 2),
        "ready": capture_quality["ready"],
        "reused": capture_quality_reused,
    }
    shutil.copy2(capture_quality_path, output_dir / "capture_quality.json")
    if args.strict_capture_quality and not capture_quality["ready"]:
        raise PipelineError(
            "Capture-quality gate failed; inspect capture_quality.json or rerun without "
            "--strict-capture-quality"
        )
    step_started = time.perf_counter()
    references = workspace / "frame_gps.txt"
    geo = write_frame_references(frames, telemetry, references)
    coordinate_transform = None
    control_point_alignment = None
    if args.gcp_control_points:
        coordinate_transform, control_point_alignment = fit_control_point_transform(
            args.gcp_control_points,
            origin=geo["origin"],
            inlier_threshold_m=args.gcp_inlier_threshold_m,
        )
        if control_point_alignment["scale_change_percent"] > args.gcp_max_scale_change_percent:
            raise PipelineError(
                "GCP correction changes scale by "
                f"{control_point_alignment['scale_change_percent']:.2f}%, above the allowed "
                f"{args.gcp_max_scale_change_percent:.2f}%; check coordinate frames and GCP labels"
            )
        write_json(output_dir / "gcp_alignment.json", control_point_alignment)
    pose_prior_report = None
    # GPS EXIF tags become the database pose priors that pose-prior mapping and the
    # global model's GPS refinement both depend on.
    if (args.mapper == "pose-prior" or args.mapper_fallback == "pose-prior" or args.spatial_matching
            or getattr(args, "global_gps_refinement", None) is not None):
        pose_prior_report = embed_frame_gps_exif(frames, telemetry, frames_dir)
    workflow_steps["telemetry_sync"] = {
        "status": "passed", "seconds": round(time.perf_counter() - step_started, 2),
        "samples": len(telemetry), "frame_references": geo["count"],
        "pose_prior_exif": pose_prior_report,
    }
    print(f"Prepared {len(frames)} frames and {geo['count']} GPS references")
    height_note = geo.get("vertical_reference", {}).get("message")
    if height_note:
        print(f"NOTE: {height_note}")

    masks_dir = None
    ai_mask_report = None
    if args.ai_mask_dynamic:
        step_started = time.perf_counter()
        print("Checking for dynamic objects (people, vehicles, animals) before masking")
        candidate_masks_dir = workspace / "masks"
        ai_mask_report = mask_dynamic_objects(frames_dir, candidate_masks_dir)
        if not ai_mask_report.get("skipped_after_precheck"):
            masks_dir = candidate_masks_dir
        workflow_steps["ai_dynamic_masking"] = {
            "status": (
                "skipped_after_precheck" if ai_mask_report.get("skipped_after_precheck") else "passed"
            ),
            "seconds": round(time.perf_counter() - step_started, 2), **ai_mask_report,
        }
    else:
        workflow_steps["ai_dynamic_masking"] = {"status": "skipped", "seconds": 0.0}

    print("2/5 Reconstructing and GPS-aligning with COLMAP")
    report = reconstruct(
        frames_dir,
        references,
        workspace,
        args.quality,
        make_mesh=not args.no_mesh,
        masks_dir=masks_dir,
        mapper_mode=args.mapper,
        gps_prior_std_m=args.gps_prior_std_m,
        gps_prior_horizontal_std_m=gps_prior_horizontal_std_m,
        gps_prior_vertical_std_m=gps_prior_vertical_std_m,
        camera_calibration=camera_calibration,
        spatial_matching=args.spatial_matching,
        spatial_max_distance_m=args.spatial_max_distance_m,
        feature_type=args.feature_type,
        feature_matcher=args.feature_matcher,
        sift_max_num_features=args.sift_max_num_features,
        dense_frame_stride=args.dense_frame_stride,
        dense_anchor_mode=args.dense_anchor_mode,
        dense_target_references=args.dense_target_references,
        dense_source_images=args.dense_source_images,
        dense_max_image_size=args.dense_max_image_size,
        dense_num_iterations=args.dense_num_iterations,
        dense_num_samples=args.dense_num_samples,
        dense_window_step=args.dense_window_step,
        fusion_min_num_pixels=args.fusion_min_num_pixels,
        delaunay_max_proj_dist=args.delaunay_max_proj_dist,
        delaunay_max_depth_dist=args.delaunay_max_depth_dist,
        delaunay_num_threads=args.delaunay_num_threads,
        mesher=args.mesher,
        mapper_ba_gpu=args.mapper_ba_gpu,
        mapper_single_model=args.mapper_single_model,
        mapper_no_extract_colors=args.mapper_no_extract_colors,
        mapper_ba_global_frames_ratio=args.mapper_ba_global_frames_ratio,
        mapper_ba_global_points_ratio=args.mapper_ba_global_points_ratio,
        mapper_ba_global_max_refinements=args.mapper_ba_global_max_refinements,
        mapper_ba_global_ignore_redundant_points3d=(
            args.mapper_ba_global_ignore_redundant_points3d
        ),
        mapper_ba_local_max_num_iterations=args.mapper_ba_local_max_num_iterations,
        mapper_ba_global_max_num_iterations=args.mapper_ba_global_max_num_iterations,
        mapper_random_seed=args.mapper_random_seed,
        mapper_init_num_trials=args.mapper_init_num_trials,
        mapper_fallback=args.mapper_fallback,
        global_mapper_min_registration_ratio=args.global_mapper_min_registration_ratio,
        global_mapper_options=getattr(args, "global_mapper_options", None),
        global_gps_refinement=getattr(args, "global_gps_refinement", None),
        default_hfov_deg=getattr(args, "default_hfov_deg", None),
        sparse_only=args.sparse_only,
        sparse_min_registration_ratio=args.sparse_min_registration_ratio,
        sparse_max_reprojection_error_px=args.sparse_max_reprojection_error_px,
        sparse_max_gps_alignment_rmse_m=args.sparse_max_gps_alignment_rmse_m,
        telemetry_samples=telemetry,
        georeference_origin=geo["origin"],
        frame_manifest=frames_manifest,
        reuse_sparse_model=args.reuse_sparse_model,
        shared_feature_database=args.shared_feature_database,
    )
    logs_source = workspace / "logs"
    logs_target = output_dir / "logs"
    if logs_source.is_dir():
        shutil.copytree(logs_source, logs_target, dirs_exist_ok=True)
        for stage in report.get("stages", {}).values():
            if stage.get("log_file"):
                stage["log_file"] = str(logs_target / Path(stage["log_file"]).name)
    report["input"] = {
        "video": str(Path(args.video).resolve()),
        "video_sha256": video_sha256,
        "video_size_bytes": Path(args.video).stat().st_size,
        "video_duration_s": selection.get("video_duration_s"),
        "telemetry": str(Path(args.telemetry).resolve()),
        "frames": len(frames),
        "target_frames": args.target_frames,
        "frame_resolution": [frame_width, frame_height],
        "camera_calibration": camera_calibration,
        "gcp_control_points": (
            str(Path(args.gcp_control_points).resolve()) if args.gcp_control_points else None
        ),
        "telemetry_altitude_offset_m": args.telemetry_altitude_offset_m,
    }
    report["experiment"] = {
        "processing_profile": processing_profile,
        "adaptive_settings": adaptive_settings,
        "keyframe_mode": args.keyframe_mode,
        "adaptive_screening_timeout_s": args.adaptive_screening_timeout_s,
        "adaptive_target_candidates": args.adaptive_target_candidates,
        "keyframe_decode_mode": args.keyframe_decode_mode,
        "reuse_keyframes": args.reuse_keyframes,
        "mapper": args.mapper,
        "mapper_fallback": args.mapper_fallback,
        "mapper_effective": report.get("mapper_mode_effective"),
        "global_mapper_options": getattr(args, "global_mapper_options", None),
        "global_gps_refinement": getattr(args, "global_gps_refinement", None),
        "gps_refinement_applied": report.get("gps_refinement", {}).get("applied"),
        "default_hfov_deg": getattr(args, "default_hfov_deg", None),
        "camera_prior": report.get("camera_prior"),
        "mapper_single_model": args.mapper_single_model,
        "single_model_retry": report.get("single_model_retry"),
        "dense_metrics": report.get("dense_metrics"),
        "spatial_matching": args.spatial_matching,
        "spatial_max_distance_m": args.spatial_max_distance_m,
        "feature_type": args.feature_type,
        "feature_matcher": args.feature_matcher,
        "sift_max_num_features": args.sift_max_num_features,
        "sparse_quality_gate": {
            "minimum_registration_ratio": args.sparse_min_registration_ratio,
            "maximum_reprojection_error_px": args.sparse_max_reprojection_error_px,
            "maximum_gps_alignment_rmse_m": args.sparse_max_gps_alignment_rmse_m,
        },
        "sparse_only": args.sparse_only,
        "reuse_sparse_model": args.reuse_sparse_model,
        "gps_prior_std_m": args.gps_prior_std_m if args.mapper == "pose-prior" else None,
        "gps_prior_horizontal_std_m": (
            gps_prior_horizontal_std_m if args.mapper == "pose-prior" else None
        ),
        "gps_prior_vertical_std_m": (
            gps_prior_vertical_std_m if args.mapper == "pose-prior" else None
        ),
        "dense_frame_stride": args.dense_frame_stride,
        "dense_anchor_mode": args.dense_anchor_mode,
        "dense_target_references": args.dense_target_references,
        "dense_source_images": args.dense_source_images,
        "dense_max_image_size": args.dense_max_image_size,
        "dense_num_iterations": args.dense_num_iterations,
        "dense_num_samples": args.dense_num_samples,
        "dense_window_step": args.dense_window_step,
        "fusion_min_num_pixels": args.fusion_min_num_pixels,
        "delaunay_max_proj_dist": args.delaunay_max_proj_dist,
        "delaunay_max_depth_dist": args.delaunay_max_depth_dist,
        "delaunay_num_threads": args.delaunay_num_threads,
        "mesher": args.mesher,
        "mapper_ba_gpu": args.mapper_ba_gpu,
        "mapper_single_model": args.mapper_single_model,
        "mapper_no_extract_colors": args.mapper_no_extract_colors,
        "mapper_ba_global_frames_ratio": args.mapper_ba_global_frames_ratio,
        "mapper_ba_global_points_ratio": args.mapper_ba_global_points_ratio,
        "mapper_ba_global_max_refinements": args.mapper_ba_global_max_refinements,
        "mapper_ba_global_ignore_redundant_points3d": (
            args.mapper_ba_global_ignore_redundant_points3d
        ),
        "mapper_ba_local_max_num_iterations": args.mapper_ba_local_max_num_iterations,
        "mapper_ba_global_max_num_iterations": args.mapper_ba_global_max_num_iterations,
        "mapper_random_seed": args.mapper_random_seed,
        "mapper_init_num_trials": args.mapper_init_num_trials,
    }
    report["georeference"] = geo
    report["control_point_alignment"] = control_point_alignment
    report["capture_quality"] = capture_quality
    report["ai_dynamic_masking"] = ai_mask_report
    report["workflow_steps"] = workflow_steps
    registered = report.get("sparse_metrics", {}).get("registered_images")
    report["sparse_metrics"]["input_frames"] = len(frames)
    report["sparse_metrics"]["registration_percent"] = (
        round(100 * registered / len(frames), 2) if registered is not None else None
    )

    print("3/5 Calculating GPS alignment diagnostics")
    step_started = time.perf_counter()
    validation = gps_alignment_report(
        report["outputs"]["aligned_images_txt"], frames_manifest, telemetry, geo["origin"]
    )
    if args.validation_distances:
        validation["independent_distances"] = independent_distance_report(args.validation_distances)
    if args.validation_checkpoints:
        validation["surveyed_checkpoints"] = surveyed_checkpoint_report(args.validation_checkpoints)
    if args.ground_truth_trajectory:
        validation["ground_truth_trajectory"] = ground_truth_trajectory_report(
            report["outputs"]["aligned_images_txt"], frames_manifest, args.ground_truth_trajectory
        )
    workflow_steps["validation"] = {"status": "passed", "seconds": round(time.perf_counter() - step_started, 2)}
    report["validation"] = validation

    if args.sparse_only:
        shutil.copy2(frames_manifest, output_dir / "frames.csv")
        if selection_path.is_file():
            shutil.copy2(selection_path, output_dir / "frames.selection.json")
        shutil.copy2(capture_quality_path, output_dir / "capture_quality.json")
        workflow_steps["product_export"] = {"status": "skipped", "seconds": 0.0}
        report["products"] = {}
        report["wall_clock_seconds"] = round(time.perf_counter() - started, 2)
        runtime_ledger.add_seconds(
            "normal_pipeline", report["wall_clock_seconds"], "sparse_pipeline"
        )
        report["runtime"] = runtime_ledger.summary()
        report["targets"] = {
            "sparse_screen_only": True,
            "surface_accuracy": None,
            "accuracy_evidence": "none - sparse screening does not prove surface accuracy",
        }
        from .compare_runs import extract_comparison_metadata

        report["comparison_metadata"] = extract_comparison_metadata(report, output_dir)
        write_json(output_dir / "run_report.json", report)
        print("4/5 Product export skipped for sparse-screen run")
        print("5/5 Sparse screen complete")
        print(f"Outputs: {output_dir}")
        return 0

    print("4/5 Exporting GIS and viewer products")
    step_started = time.perf_counter()
    products = export_products(
        report["outputs"]["point_cloud_ply"],
        report["outputs"]["mesh_ply"],
        output_dir,
        geo["origin"],
        dsm_resolution_m=args.dsm_resolution,
        origin_policy=geo["origin_policy"],
        textured_mesh_path=report["outputs"].get("textured_mesh_ply"),
        texture_path=report["outputs"].get("texture_image"),
        coordinate_transform=coordinate_transform,
        source_mesh_quality=report.get("mesh_cleanup"),
        vertical_reference=geo.get("vertical_reference"),
    )
    shutil.copy2(frames_manifest, output_dir / "frames.csv")
    if selection_path.is_file():
        shutil.copy2(selection_path, output_dir / "frames.selection.json")
    shutil.copy2(capture_quality_path, output_dir / "capture_quality.json")
    workflow_steps["product_export"] = {"status": "passed", "seconds": round(time.perf_counter() - step_started, 2)}
    report["products"] = products
    report["wall_clock_seconds"] = round(time.perf_counter() - started, 2)
    runtime_ledger.add_seconds(
        "normal_pipeline", report["wall_clock_seconds"], "reconstruction_pipeline"
    )
    report["runtime"] = runtime_ledger.summary()
    checkpoint_count_status = validation.get("surveyed_checkpoints", {}).get(
        "asprs_checkpoint_count_status"
    )
    checkpoint_count_compliant = validation.get("surveyed_checkpoints", {}).get(
        "asprs_checkpoint_count_compliant"
    )
    independent_pass, accuracy_evidence = _independent_surface_target(validation)
    trajectory_pass = validation.get("ground_truth_trajectory", {}).get("passes_one_metre_position_target")
    absolute_gps_pass = validation.get("gps_alignment_rmse_m", float("inf")) <= 1.0
    report["targets"] = {
        "processing_under_15_minutes": report["wall_clock_seconds"] < 900,
        # Camera-trajectory agreement is useful validation, but it cannot prove
        # surface accuracy. Only independent scene measurements can.
        "one_metre_accuracy": independent_pass,
        "one_metre_surface_accuracy": independent_pass,
        "one_metre_absolute_camera_alignment": absolute_gps_pass,
        "one_metre_relative_trajectory": trajectory_pass,
        "asprs_checkpoint_count_status": checkpoint_count_status,
        "asprs_checkpoint_count_compliant": checkpoint_count_compliant,
        "accuracy_evidence": accuracy_evidence,
    }
    from .compare_runs import extract_comparison_metadata

    report["comparison_metadata"] = extract_comparison_metadata(report, output_dir)
    write_json(output_dir / "run_report.json", report)
    write_json(output_dir / "viewer_metadata.json", {
        "origin": geo["origin"],
        "crs": products["crs"],
        "validation": validation,
        "capture_quality": capture_quality,
        "control_point_alignment": control_point_alignment,
        "processing_seconds": report["wall_clock_seconds"],
        "registration_percent": report["sparse_metrics"]["registration_percent"],
        "sparse_metrics": report["sparse_metrics"],
        "vertical_reference": geo.get("vertical_reference"),
        "products": {
            **products,
            "las": "point_cloud.las",
            "geotiff_dsm": "dsm.tif",
            "confidence_geotiff": "confidence.tif",
            "glb": "model.glb" if products.get("glb") else None,
            "obj": "model.obj" if products.get("obj") else None,
            "mesh_ply": "mesh.ply" if products.get("mesh_ply") else None,
        },
    })
    print("5/5 Complete")
    print(f"Outputs: {output_dir}")
    _print_time_breakdown(report)
    return 0


def _print_time_breakdown(report: dict) -> None:
    rows = [(f"step: {name}", step.get("seconds") or 0.0)
            for name, step in report.get("workflow_steps", {}).items()]
    rows += [(f"colmap: {name}", stage.get("seconds") or 0.0)
             for name, stage in report.get("stages", {}).items()]
    total = report.get("wall_clock_seconds") or 0.0
    accounted = sum(seconds for _, seconds in rows)
    rows.append(("unaccounted (hashing, report, I/O)", max(0.0, total - accounted)))
    print(f"\nTime breakdown (mapper used: {report.get('mapper_mode_effective')}):")
    for name, seconds in sorted(rows, key=lambda row: -row[1]):
        share = 100 * seconds / total if total else 0.0
        print(f"  {name:<42} {seconds / 60:6.2f} min  {share:5.1f}%")
    print(f"  {'TOTAL':<42} {total / 60:6.2f} min")


def _preflight(args: argparse.Namespace) -> int:
    from .preflight import inspect_environment

    result = inspect_environment(
        args.video,
        args.telemetry,
        args.workspace,
        camera_calibration=args.camera_calibration,
    )
    write_json(Path(args.output), result)
    print(Path(args.output).read_text(encoding="utf-8"))
    return 0 if result["ready"] else 2


def _verify(args: argparse.Namespace) -> int:
    from .verify_outputs import verify_output_directory

    result = verify_output_directory(args.output)
    print(__import__("json").dumps(result, indent=2))
    return 0 if result["artifact_checks_pass"] else 2


def _validate_checkpoints(args: argparse.Namespace) -> int:
    """Attach post-reconstruction surveyed checkpoint evidence without rerunning MVS."""
    import json
    from .validation import surveyed_checkpoint_report

    output_dir = Path(args.output).resolve()
    report_path = output_dir / "run_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_report = surveyed_checkpoint_report(args.checkpoints)
    report.setdefault("validation", {})["surveyed_checkpoints"] = checkpoint_report
    surface_pass, accuracy_evidence = _independent_surface_target(report["validation"])
    report.setdefault("targets", {})["one_metre_accuracy"] = surface_pass
    report["targets"]["one_metre_surface_accuracy"] = surface_pass
    report["targets"]["asprs_checkpoint_count_status"] = checkpoint_report[
        "asprs_checkpoint_count_status"
    ]
    report["targets"]["asprs_checkpoint_count_compliant"] = checkpoint_report[
        "asprs_checkpoint_count_compliant"
    ]
    report["targets"]["accuracy_evidence"] = accuracy_evidence
    from .compare_runs import extract_comparison_metadata

    report["comparison_metadata"] = extract_comparison_metadata(report, output_dir)
    write_json(report_path, report)
    viewer_path = output_dir / "viewer_metadata.json"
    if viewer_path.is_file():
        viewer = json.loads(viewer_path.read_text(encoding="utf-8"))
        viewer.setdefault("validation", {})["surveyed_checkpoints"] = checkpoint_report
        write_json(viewer_path, viewer)
    print(__import__("json").dumps(checkpoint_report, indent=2))
    return 0 if checkpoint_report["passes_one_metre_target"] else 2


def _validate_geometry(args: argparse.Namespace) -> int:
    """Attach reference-geometry evidence to an existing reconstruction."""
    import json

    from .dense_validation import DEFAULT_THRESHOLDS_M, evaluate_dense_geometry

    output_dir = Path(args.output).resolve()
    report_path = output_dir / "run_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    reconstruction_path = (
        Path(args.reconstruction).resolve()
        if args.reconstruction
        else output_dir / "point_cloud.ply"
    )
    thresholds_m = args.threshold_m or DEFAULT_THRESHOLDS_M
    geometry_report = evaluate_dense_geometry(
        reconstruction_path,
        args.reference,
        thresholds_m=thresholds_m,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("validation", {})["dense_geometry"] = geometry_report
    surface_pass, accuracy_evidence = _independent_surface_target(report["validation"])
    report.setdefault("targets", {})["dense_reference_geometry_evaluated"] = True
    report["targets"]["dense_geometry_alignment"] = geometry_report["alignment"]
    report["targets"]["one_metre_accuracy"] = surface_pass
    report["targets"]["one_metre_surface_accuracy"] = surface_pass
    report["targets"]["accuracy_evidence"] = accuracy_evidence
    from .compare_runs import extract_comparison_metadata

    report["comparison_metadata"] = extract_comparison_metadata(report, output_dir)
    write_json(report_path, report)
    write_json(output_dir / "dense_geometry_validation.json", geometry_report)

    viewer_path = output_dir / "viewer_metadata.json"
    if viewer_path.is_file():
        viewer = json.loads(viewer_path.read_text(encoding="utf-8"))
        viewer.setdefault("validation", {})["dense_geometry"] = geometry_report
        write_json(viewer_path, viewer)

    print(json.dumps(geometry_report, indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    from .compare_runs import compare_output_directories

    result = compare_output_directories(args.before, args.after, args.output)
    print(json.dumps(result, indent=2))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .benchmark import build_benchmark

    entries = []
    for value in args.run_entry:
        if "=" not in value:
            raise ValueError("Each --run-entry must use LABEL=OUTPUT_DIRECTORY")
        label, directory = value.split("=", 1)
        if not label.strip() or not directory.strip():
            raise ValueError("Each --run-entry must use LABEL=OUTPUT_DIRECTORY")
        entries.append((label.strip(), directory.strip()))
    result = build_benchmark(entries, args.output)
    print(json.dumps(result, indent=2))
    return 0


def _assess(args: argparse.Namespace) -> int:
    from .camera import load_camera_calibration
    from .capture_quality import analyze_selected_frames
    from .extract_keyframes import extract_keyframes
    from .telemetry import load_telemetry

    telemetry = load_telemetry(
        args.telemetry, altitude_offset_m=args.telemetry_altitude_offset_m
    )
    camera_calibration = load_camera_calibration(args.camera_calibration)
    workspace = Path(args.workspace).resolve()
    frames_dir = workspace / "capture_assessment_frames"
    manifest = workspace / "capture_assessment_frames.csv"
    frames = extract_keyframes(
        args.video,
        frames_dir,
        target_frames=args.target_frames,
        max_width=args.max_width,
        manifest_path=manifest,
        selection_mode=getattr(args, "keyframe_mode", "geometry"),
        telemetry_samples=telemetry,
        decode_mode=args.keyframe_decode_mode,
    )
    result = analyze_selected_frames(
        frames_dir, frames, telemetry, args.output, camera_calibration
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 2


def _classify(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .flight_classifier import classify
    from .telemetry import load_telemetry

    telemetry = load_telemetry(
        args.telemetry, altitude_offset_m=args.telemetry_altitude_offset_m
    )
    result = asdict(classify(telemetry))
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sih-drone",
        description="Single-pass drone video to georeferenced 3D products (SIH26158)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="Run the complete Colab reconstruction workflow")
    run.add_argument("--video", required=True, help="1080p/4K drone video")
    run.add_argument("--telemetry", required=True, help="DJI SRT or normalized telemetry CSV")
    run.add_argument(
        "--telemetry-altitude-offset-m", type=float, default=0.0,
        help="Known vertical-datum correction added to every telemetry altitude",
    )
    run.add_argument("--workspace", default="/content/sih_workspace", help="Fast temporary Colab workspace")
    run.add_argument("--output", required=True, help="Persistent output directory, normally in Google Drive")
    run.add_argument(
        "--target-frames",
        type=int,
        help="Fixed frame budget; omit for duration-adaptive selection",
    )
    run.add_argument("--max-width", type=int, default=1920)
    run.add_argument(
        "--sample-fps", type=float, default=1.0,
        help="Adaptive sampling rate when --target-frames is omitted (default: 1 frame/second)",
    )
    run.add_argument(
        "--max-frames", type=int,
        help="Cap adaptive selection (default: 600 draft, 1200 full)",
    )
    run.add_argument("--quality", choices=("draft", "full"), default="draft")
    run.add_argument(
        "--profile",
        choices=(
            "manual", "verified-fast", "verified-fast-adaptive", "high-detail",
            "deadline-preview", "deadline-reuse14", "deadline-dynamic", "deadline-adaptive",
        ),
        default="manual",
        help="Apply a reproducible processing profile; explicit profile settings override related flags",
    )
    run.add_argument(
        "--keyframe-mode", choices=("uniform", "geometry", "adaptive"), default="uniform",
        help="Select frames by sharpness alone or by sharpness, GPS baseline, and optical-flow parallax",
    )
    run.add_argument(
        "--keyframe-decode-mode", choices=("seek", "sequential", "ffmpeg"), default="seek",
        help="Decode candidates with random seeks, OpenCV sequential scan, or FFmpeg/NVDEC",
    )
    run.add_argument(
        "--adaptive-screening-timeout-s", type=float, default=900.0,
        help="Time limit for adaptive keyframe screening. The library default of 180 s "
             "timed out on a 5.5-minute video; actual screening time is recorded in "
             "frames.selection.json.",
    )
    run.add_argument(
        "--adaptive-target-candidates", type=int, default=700,
        help="Target candidate-frame count for adaptive screening; candidate_fps is "
             "derived from this divided by video duration (clamped to 0.5-3.0 fps), so "
             "screening cost stays roughly flat as video length grows instead of "
             "scaling with duration at a fixed sampling rate.",
    )
    run.add_argument(
        "--reuse-keyframes", action="store_true",
        help="Reuse a validated ordered frame cache already present in the workspace",
    )
    run.add_argument(
        "--video-sha256",
        help="Precomputed 64-character SHA-256 digest; avoids rehashing a shared large video",
    )
    run.add_argument(
        "--reuse-capture-quality", action="store_true",
        help="Reuse workspace/capture_quality.json instead of recomputing image-quality metrics",
    )
    run.add_argument(
        "--mapper", choices=("standard", "pose-prior", "global"), default="standard",
        help="Use incremental, GPS pose-prior incremental, or calibrated global SfM",
    )
    run.add_argument(
        "--mapper-fallback", choices=("standard", "pose-prior"), default=None,
        help="With --mapper global: retry with this incremental mapper if global SfM is "
             "unavailable, fails, or registers too few frames",
    )
    run.add_argument(
        "--global-mapper-min-registration-ratio", type=float, default=0.6,
        help="With --mapper-fallback: minimum fraction of frames the global mapper must "
             "register before its model is accepted (default: 0.6)",
    )
    run.add_argument(
        "--global-gps-refinement", action="store_const", const={}, default=None,
        help="With --mapper global: re-anchor cameras that disagree with GPS, re-triangulate, "
             "and run one GPS-anchored bundle adjustment pass (the deadline-adaptive profile "
             "enables this)",
    )
    run.add_argument(
        "--spatial-matching", action="store_true",
        help="Add GPS-neighbour matches to reconnect revisited or crossing flight paths",
    )
    run.add_argument(
        "--spatial-max-distance-m", type=float, default=100.0,
        help="Maximum GPS distance for spatial matching (default: 100 m)",
    )
    run.add_argument(
        "--feature-type",
        choices=("sift", "aliked-n16rot", "aliked-n32"),
        default="sift",
        help="COLMAP local feature extractor (learned ALIKED requires an ONNX-enabled build)",
    )
    run.add_argument(
        "--feature-matcher",
        choices=("bruteforce", "lightglue"),
        default="bruteforce",
        help="Compatible brute-force or LightGlue matcher (LightGlue requires ONNX)",
    )
    run.add_argument(
        "--sift-max-num-features", type=int,
        help="Maximum SIFT features per image (COLMAP default is normally 8192)",
    )
    run.add_argument(
        "--default-hfov-deg", type=float, default=None,
        help="Without --camera-calibration: start from this horizontal field of view "
             "(trusted, refined in BA) instead of COLMAP's 45-degree guess "
             "(the deadline-adaptive profile uses 70)",
    )
    run.add_argument(
        "--gps-prior-std-m", type=float, default=5.0,
        help="One-sigma GPS uncertainty used by pose-prior mapping (default: 5 m)",
    )
    run.add_argument(
        "--gps-prior-horizontal-std-m", type=float,
        help="Horizontal pose-prior uncertainty; defaults to telemetry accuracy or --gps-prior-std-m",
    )
    run.add_argument(
        "--gps-prior-vertical-std-m", type=float,
        help="Vertical pose-prior uncertainty; defaults to telemetry accuracy or --gps-prior-std-m",
    )
    run.add_argument(
        "--camera-calibration",
        help="JSON camera profile with COLMAP model/params and optional fixed_intrinsics",
    )
    run.add_argument(
        "--dense-frame-stride", type=int, default=1,
        help="Compute dense depth for every Nth registered reference frame (default: 1)",
    )
    run.add_argument(
        "--dense-anchor-mode", choices=("uniform", "adaptive", "coverage"), default="uniform",
        help="Choose dense references by fixed stride, strongest geometry within temporal bins, "
             "or ground coverage (every patch of ground gets enough depth maps)",
    )
    run.add_argument(
        "--dense-target-references", type=int,
        help="Select an exact number of temporally distributed dense reference views",
    )
    run.add_argument(
        "--dense-source-images", type=int,
        help="Maximum source views per dense reference (default: COLMAP automatic value)",
    )
    run.add_argument(
        "--dense-max-image-size", type=int,
        help="Override PatchMatch and fusion maximum image dimension",
    )
    run.add_argument(
        "--dense-num-iterations", type=int,
        help="Override PatchMatch coordinate-descent iterations",
    )
    run.add_argument(
        "--dense-num-samples", type=int,
        help="Override PatchMatch Monte Carlo samples",
    )
    run.add_argument(
        "--dense-window-step", type=int,
        help="Override PatchMatch correlation-window step (larger values may lose fine detail)",
    )
    run.add_argument(
        "--fusion-min-num-pixels", type=int, default=5,
        help="Depth maps that must agree before stereo fusion keeps a point (default: 5)",
    )
    run.add_argument(
        "--delaunay-max-proj-dist", type=float,
        help="Override Delaunay projected point-unification radius in pixels",
    )
    run.add_argument(
        "--delaunay-max-depth-dist", type=float,
        help="Override Delaunay relative depth-unification threshold",
    )
    run.add_argument(
        "--delaunay-num-threads", type=int,
        help="Override Delaunay CPU thread count (-1 uses all available threads)",
    )
    run.add_argument(
        "--mesher",
        choices=("auto", "delaunay", "poisson", "advancing-front"),
        default="auto",
        help="Surface mesher; auto keeps the quality-profile default and safe fallback",
    )
    run.add_argument(
        "--mapper-ba-gpu", action="store_true",
        help="Enable mapper GPU bundle adjustment when supported by the installed COLMAP build",
    )
    run.add_argument(
        "--mapper-single-model", action="store_true",
        help="Experimental: stop after the first retained model; unsafe when initialization varies",
    )
    run.add_argument(
        "--mapper-no-extract-colors", action="store_true",
        help="Skip sparse-point color extraction inside the mapper",
    )
    run.add_argument(
        "--mapper-ba-global-frames-ratio", type=float,
        help="Frame-growth ratio that triggers incremental global bundle adjustment",
    )
    run.add_argument(
        "--mapper-ba-global-points-ratio", type=float,
        help="Point-growth ratio that triggers incremental global bundle adjustment",
    )
    run.add_argument(
        "--mapper-ba-global-max-refinements", type=int,
        help="Maximum iterative refinements per global bundle-adjustment pass",
    )
    run.add_argument(
        "--mapper-ba-global-ignore-redundant-points3d", action="store_true",
        help="Use COLMAP's reduced-landmark global bundle-adjustment path",
    )
    run.add_argument(
        "--mapper-ba-local-max-num-iterations", type=int,
        help="Maximum Ceres iterations for each local bundle-adjustment solve",
    )
    run.add_argument(
        "--mapper-ba-global-max-num-iterations", type=int,
        help="Maximum Ceres iterations for each global bundle-adjustment solve",
    )
    run.add_argument(
        "--mapper-random-seed", type=int,
        help="Deterministic COLMAP incremental-mapper seed (V10 passes 0)",
    )
    run.add_argument(
        "--mapper-init-num-trials", type=int, default=50,
        help="Cap on the mapper's initial-pair search trials per model. A weakly "
             "connected flight (e.g. two spatially separated clusters) can otherwise "
             "spend most of the runtime cycling through COLMAP's default trial count "
             "before giving up on each failed component. The strongest already-verified "
             "pair (from feature matching) is tried first regardless of this cap.",
    )
    run.add_argument("--dsm-resolution", type=float, default=0.5)
    run.add_argument(
        "--strict-capture-quality", action="store_true",
        help="Stop before COLMAP when selected-frame or telemetry quality gates fail",
    )
    run.add_argument("--no-mesh", action="store_true", help="Generate a point cloud only")
    run.add_argument(
        "--sparse-only", action="store_true",
        help="Stop after sparse reconstruction, GPS alignment, and sparse validation",
    )
    run.add_argument("--shared-feature-database", help="Completed immutable SIFT cache database")
    run.add_argument(
        "--reuse-sparse-model",
        help=(
            "Reuse an existing COLMAP sparse model and skip feature extraction, "
            "matching, and sparse mapping; the selected images must match the model"
        ),
    )
    run.add_argument(
        "--sparse-min-registration-ratio", type=float,
        help="Stop before dense stereo when the registered/selected image ratio is lower",
    )
    run.add_argument(
        "--sparse-max-reprojection-error-px", type=float,
        help="Stop before dense stereo when mean sparse reprojection error is higher",
    )
    run.add_argument(
        "--sparse-max-gps-alignment-rmse-m", type=float,
        help="Stop before dense stereo when aligned camera/GPS RMSE is higher",
    )
    run.add_argument("--validation-distances", help="Optional independent metric-distance CSV")
    run.add_argument(
        "--gcp-control-points",
        help=(
            "CSV with reconstructed XYZ and surveyed local XYZ or geodetic coordinates; "
            "used for robust similarity correction before export"
        ),
    )
    run.add_argument(
        "--gcp-inlier-threshold-m", type=float, default=1.0,
        help="Maximum GCP residual retained by robust alignment (default: 1 m)",
    )
    run.add_argument(
        "--gcp-max-scale-change-percent", type=float, default=25.0,
        help="Safety limit preventing a mismatched GCP file from severely distorting the model",
    )
    run.add_argument(
        "--validation-checkpoints",
        help="CSV with surveyed and reconstructed XYZ checkpoint coordinates (at least three measured rows)",
    )
    run.add_argument("--ground-truth-trajectory", help="Optional independent time_s,x_m,y_m,z_m trajectory CSV")
    run.add_argument(
        "--ai-mask-dynamic",
        action="store_true",
        help="Use YOLO segmentation to exclude moving people, vehicles and animals",
    )
    run.set_defaults(handler=_run)
    preflight = subcommands.add_parser("preflight", help="Verify GPU, COLMAP, video, telemetry and disk space")
    preflight.add_argument("--video", required=True)
    preflight.add_argument("--telemetry", required=True)
    preflight.add_argument("--workspace", default="/content/sih_workspace")
    preflight.add_argument("--camera-calibration")
    preflight.add_argument("--output", required=True, help="Preflight JSON path")
    preflight.set_defaults(handler=_preflight)
    verify = subcommands.add_parser("verify", help="Validate every expected output and stage log")
    verify.add_argument("--output", required=True, help="Completed pipeline output directory")
    verify.set_defaults(handler=_verify)
    checkpoints = subcommands.add_parser(
        "validate-checkpoints",
        help="Attach surveyed surface-checkpoint evidence to an existing reconstruction",
    )
    checkpoints.add_argument("--output", required=True, help="Completed pipeline output directory")
    checkpoints.add_argument("--checkpoints", required=True, help="Populated surveyed/reconstructed XYZ CSV")
    checkpoints.set_defaults(handler=_validate_checkpoints)
    geometry = subcommands.add_parser(
        "validate-geometry",
        help="Measure dense accuracy and completeness against independent reference geometry",
    )
    geometry.add_argument("--output", required=True, help="Completed pipeline output directory")
    geometry.add_argument(
        "--reference", required=True,
        help="Independent metric point cloud or mesh in the reconstruction coordinate frame",
    )
    geometry.add_argument(
        "--reconstruction",
        help="Reconstructed point cloud or mesh (default: OUTPUT/point_cloud.ply)",
    )
    geometry.add_argument(
        "--threshold-m", action="append", type=float,
        help="Distance threshold in metres; repeat for multiple values (defaults: 0.05, 0.1, 0.25, 0.5, 1.0)",
    )
    geometry.add_argument(
        "--max-samples", type=int, default=100_000,
        help="Maximum deterministic evaluation samples per geometry (default: 100000)",
    )
    geometry.add_argument("--seed", type=int, default=26158)
    geometry.set_defaults(handler=_validate_geometry)
    comparison = subcommands.add_parser(
        "compare", help="Compare baseline and experimental reconstruction reports"
    )
    comparison.add_argument("--before", required=True, help="Baseline output directory")
    comparison.add_argument("--after", required=True, help="Experimental output directory")
    comparison.add_argument("--output", required=True, help="Comparison JSON path; Markdown is written beside it")
    comparison.set_defaults(handler=_compare)
    assessment = subcommands.add_parser(
        "assess", help="Screen video, frame overlap, exposure, blur, and telemetry before reconstruction"
    )
    assessment.add_argument("--video", required=True)
    assessment.add_argument("--telemetry", required=True)
    assessment.add_argument("--telemetry-altitude-offset-m", type=float, default=0.0)
    assessment.add_argument("--camera-calibration")
    assessment.add_argument("--workspace", default="/content/sih_capture_assessment")
    assessment.add_argument("--output", required=True, help="Capture-quality JSON path")
    assessment.add_argument("--target-frames", type=int, default=60)
    assessment.add_argument("--keyframe-mode", choices=("geometry", "adaptive"), default="geometry")
    assessment.add_argument("--max-width", type=int, default=1280)
    assessment.add_argument(
        "--keyframe-decode-mode",
        choices=("seek", "sequential", "ffmpeg"),
        default="ffmpeg",
    )
    assessment.set_defaults(handler=_assess)
    classification = subcommands.add_parser(
        "classify",
        help="Recommend a processing profile and frame budget from telemetry alone (no video needed)",
    )
    classification.add_argument("--telemetry", required=True)
    classification.add_argument("--telemetry-altitude-offset-m", type=float, default=0.0)
    classification.add_argument("--output", help="Optional JSON path for the classification report")
    classification.set_defaults(handler=_classify)
    benchmark = subcommands.add_parser(
        "benchmark", help="Rank two or more normalized reconstruction outputs"
    )
    benchmark.add_argument(
        "--run-entry", action="append", required=True, metavar="LABEL=DIR",
        help="Labelled output directory; repeat at least twice",
    )
    benchmark.add_argument("--output", required=True, help="Benchmark JSON path")
    benchmark.set_defaults(handler=_benchmark)
    return parser


def _explicit_options(parser: argparse.ArgumentParser, args, argv: list[str]) -> set[str]:
    """Destinations of the options actually typed for the chosen subcommand."""
    typed = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    command_parser = subparsers[0].choices.get(getattr(args, "command", None)) if subparsers else None
    actions = command_parser._actions if command_parser else parser._actions
    return {action.dest for action in actions if typed.intersection(action.option_strings)}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    args.explicit_options = _explicit_options(parser, args, argv)
    try:
        raise SystemExit(args.handler(args))
    except (PipelineError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
