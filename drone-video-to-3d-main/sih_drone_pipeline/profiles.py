from __future__ import annotations


PROFILES = {
    "manual": {},
    "verified-fast": {
        "quality": "draft",
        "keyframe_mode": "geometry",
        "keyframe_decode_mode": "sequential",
        "mapper": "pose-prior",
        # Frozen from the accepted full253_a3_robust_multimodel_1088 run.
        # Do not silently fold rejected dense-screen settings into this profile.
        "spatial_matching": False,
        "dense_frame_stride": 2,
        "dense_anchor_mode": "adaptive",
        "dense_target_references": None,
        "dense_source_images": 10,
        "dense_max_image_size": 1088,
        "dense_num_iterations": 4,
        "dense_num_samples": 15,
        "dense_window_step": 1,
        "delaunay_max_proj_dist": 20.0,
        "delaunay_num_threads": -1,
        "mapper_ba_gpu": True,
        "mapper_single_model": False,
        "mapper_no_extract_colors": True,
        "mapper_ba_global_frames_ratio": 1.4,
        "mapper_ba_global_points_ratio": 1.4,
        "mapper_ba_global_max_refinements": 2,
        "mapper_ba_global_ignore_redundant_points3d": True,
    },
    "verified-fast-adaptive": {
        # Identical to verified-fast except keyframe selection: instead of a
        # fixed target-frame budget, adaptive_selection.py visually verifies
        # every frame-to-frame connection (ORB+SIFT, RANSAC) and inserts
        # extra frames wherever a link is weak or unverified -- e.g. during
        # a rapid altitude change, where a fixed-budget selector can leave a
        # gap too wide for the mapper to track through. target_frames here
        # is an affordable-count estimate, not a hard cap; the algorithm can
        # exceed it and reports budget_conflict=true when it does, rather
        # than silently under-sampling a hard segment.
        "quality": "draft",
        "keyframe_mode": "adaptive",
        "mapper": "pose-prior",
        "spatial_matching": False,
        "dense_frame_stride": 2,
        "dense_anchor_mode": "adaptive",
        "dense_target_references": None,
        "dense_source_images": 10,
        "dense_max_image_size": 1088,
        "dense_num_iterations": 4,
        "dense_num_samples": 15,
        "dense_window_step": 1,
        "delaunay_max_proj_dist": 20.0,
        "delaunay_num_threads": -1,
        "mapper_ba_gpu": True,
        "mapper_single_model": False,
        "mapper_no_extract_colors": True,
        "mapper_ba_global_frames_ratio": 1.4,
        "mapper_ba_global_points_ratio": 1.4,
        "mapper_ba_global_max_refinements": 2,
        "mapper_ba_global_ignore_redundant_points3d": True,
    },
    "high-detail": {
        "quality": "full",
        "keyframe_mode": "geometry",
        "keyframe_decode_mode": "sequential",
        "mapper": "pose-prior",
        "spatial_matching": True,
        "dense_anchor_mode": "adaptive",
        "dense_source_images": 16,
        "dense_max_image_size": 1920,
        "dense_num_iterations": 5,
        "dense_num_samples": 15,
        "dense_window_step": 1,
    },
    "deadline-preview": {
        "quality": "draft",
        "keyframe_mode": "geometry",
        "keyframe_decode_mode": "sequential",
        "mapper": "pose-prior",
        "spatial_matching": False,
        "sift_max_num_features": 4096,
        "dense_frame_stride": 1,
        "dense_anchor_mode": "adaptive",
        "dense_target_references": 72,
        "dense_source_images": 8,
        "dense_max_image_size": 896,
        "dense_num_iterations": 3,
        "dense_num_samples": 10,
        "dense_window_step": 1,
        "delaunay_max_proj_dist": 20.0,
        "delaunay_num_threads": -1,
        "mapper_ba_gpu": True,
        "mapper_single_model": False,
        "mapper_no_extract_colors": True,
        "mapper_ba_global_frames_ratio": 1.4,
        "mapper_ba_global_points_ratio": 1.4,
        "mapper_ba_global_max_refinements": 2,
        "mapper_ba_global_ignore_redundant_points3d": True,
        "sparse_min_registration_ratio": 0.95,
        "sparse_max_reprojection_error_px": 0.75,
        "sparse_max_gps_alignment_rmse_m": 2.0,
        "ai_mask_dynamic": False,
    },
    "deadline-reuse14": {
        "quality": "draft",
        "keyframe_mode": "geometry",
        "keyframe_decode_mode": "sequential",
        "reuse_keyframes": True,
        "mapper": "pose-prior",
        "spatial_matching": False,
        "sift_max_num_features": 4096,
        "dense_frame_stride": 1,
        "dense_anchor_mode": "adaptive",
        "dense_target_references": 64,
        "dense_source_images": 6,
        "dense_max_image_size": 896,
        "dense_num_iterations": 3,
        "dense_num_samples": 10,
        "dense_window_step": 1,
        "delaunay_max_proj_dist": 20.0,
        "delaunay_num_threads": -1,
        "mapper_ba_gpu": True,
        "mapper_single_model": False,
        "mapper_no_extract_colors": True,
        "mapper_ba_global_frames_ratio": 1.4,
        "mapper_ba_global_points_ratio": 1.4,
        "mapper_ba_global_max_refinements": 2,
        "mapper_ba_global_ignore_redundant_points3d": True,
        "sparse_min_registration_ratio": 0.95,
        "sparse_max_reprojection_error_px": 0.75,
        "sparse_max_gps_alignment_rmse_m": 2.0,
        "ai_mask_dynamic": False,
    },
    "deadline-dynamic": {
        "quality": "draft",
        "keyframe_mode": "geometry",
        "keyframe_decode_mode": "sequential",
        "reuse_keyframes": True,
        "mapper": "pose-prior",
        "spatial_matching": False,
        "sift_max_num_features": 4096,
        "dense_frame_stride": 1,
        "dense_anchor_mode": "adaptive",
        # Resolved after extraction as 35% of selected frames, bounded 20..64.
        "dense_target_references": None,
        "dense_source_images": 6,
        "dense_max_image_size": 896,
        "dense_num_iterations": 3,
        "dense_num_samples": 10,
        "dense_window_step": 1,
        "delaunay_max_proj_dist": 20.0,
        "delaunay_num_threads": -1,
        "mapper_ba_gpu": True,
        "mapper_single_model": False,
        "mapper_no_extract_colors": True,
        "mapper_ba_global_frames_ratio": 1.4,
        "mapper_ba_global_points_ratio": 1.4,
        "mapper_ba_global_max_refinements": 2,
        "mapper_ba_global_ignore_redundant_points3d": True,
        "sparse_min_registration_ratio": 0.95,
        "sparse_max_reprojection_error_px": 0.75,
        "sparse_max_gps_alignment_rmse_m": 2.0,
        "ai_mask_dynamic": False,
    },
    "deadline-adaptive": {
        # Built for the 15-minute budget. Adaptive keyframes and the
        # deadline-dynamic dense budget: dense references are resolved from the
        # actual frame count (35%, bounded 20..64), so dense cost stays flat as
        # videos get longer. Sparse quality gates are left unset so a weak
        # flight still produces outputs and a report rather than aborting
        # before dense; the dense-collapse guard stops a meaningless model.
        "quality": "draft",
        "keyframe_mode": "adaptive",
        # Screening cost scales with candidate count; 400 keeps ~1.7 fps on a
        # 4-min clip and the verifier still inserts frames at weak links.
        "adaptive_target_candidates": 400,
        # Global SfM was dropped: on the synced reference flight it collapsed
        # the whole path (every camera within metres of each other) while
        # reporting sub-pixel reprojection error, so no registration or
        # reprojection gate could catch it. Incremental mapping anchored to GPS
        # priors placed 122 of 170 frames at 1.2 m median / 2.2 m RMSE in
        # 3.2 min on the T4 Colab runtime with the settings below.
        "mapper": "pose-prior",
        "gps_prior_horizontal_std_m": 2.0,
        "gps_prior_vertical_std_m": 3.0,
        # COLMAP's own focal guess (45 deg) is far from any drone camera and is
        # untrusted, so every pair verifies as "uncalibrated". A generic 70 deg
        # default, trusted and refined in BA, is close enough for any DJI-class
        # camera; a --camera-calibration file overrides it.
        "default_hfov_deg": 70.0,
        # Stop after the first model: extra-model attempts from the takeoff and
        # hover frames cost ~2 min and never join. If the first model is small
        # (< 50% of frames) the mapper reruns with extra models allowed.
        "mapper_single_model": True,
        "spatial_matching": False,
        "dense_frame_stride": 1,
        # Depth-map references are chosen by ground coverage: on the 11-min
        # flight, time-sliced references left large patches seen by fewer
        # depth maps than fusion requires, so they vanished from the mesh.
        "dense_anchor_mode": "coverage",
        "dense_target_references": None,
        "dense_source_images": 6,
        "dense_max_image_size": 896,
        "dense_num_iterations": 3,
        "dense_num_samples": 10,
        "dense_window_step": 1,
        "delaunay_max_proj_dist": 20.0,
        "delaunay_num_threads": -1,
        "mapper_ba_gpu": True,
        "mapper_no_extract_colors": True,
        # Measured on the 11-min video (236 frames): 4096 features plus lighter
        # BA cut extraction+matching+mapping from 13.1 to 5.1 min with all
        # frames still registered and GPS residual 2.5 -> 2.2 m.
        "sift_max_num_features": 4096,
        "mapper_ba_global_frames_ratio": 2.0,
        "mapper_ba_global_points_ratio": 2.0,
        "mapper_ba_global_max_refinements": 1,
        "mapper_ba_local_max_num_iterations": 15,
        "mapper_ba_global_max_num_iterations": 30,
        "mapper_ba_global_ignore_redundant_points3d": True,
    },
}


def apply_processing_profile(args) -> dict:
    """Apply an explicit reproducible profile and return the resolved settings.

    Options the user typed on the command line (``args.explicit_options``) keep
    their value and are recorded as overrides, so a single setting can be tested
    against a profile without editing it.
    """
    profile = getattr(args, "profile", "manual")
    if profile not in PROFILES:
        raise ValueError(f"Unknown processing profile: {profile}")
    explicit = set(getattr(args, "explicit_options", ()) or ())
    applied, overridden = {}, {}
    for name, value in PROFILES[profile].items():
        if name in explicit and hasattr(args, name):
            overridden[name] = {"profile": value, "used": getattr(args, name)}
            continue
        setattr(args, name, value)
        applied[name] = value
    report = {"name": profile, "applied": applied}
    if overridden:
        report["overridden_by_command_line"] = overridden
    return report


def resolve_adaptive_dense_settings(args, frame_count: int) -> dict:
    """Record how the verified profile derives dense references from registered views."""
    if getattr(args, "profile", "manual") in ("deadline-dynamic", "deadline-adaptive"):
        # deadline-adaptive allows up to 96 references so long flights keep the
        # same 35% share (the 11-min video hit the old cap at 64 of 236 frames).
        cap = 96 if args.profile == "deadline-adaptive" else 64
        target = min(frame_count, max(20, min(cap, int(round(frame_count * 0.35)))))
        args.dense_target_references = target
        return {
            "automatic": True,
            "input_frames": frame_count,
            "dense_target_references": target,
            "reference_policy": f"35_percent_bounded_20_{cap}",
            "dense_source_images": args.dense_source_images,
            "dense_max_image_size": args.dense_max_image_size,
            "dense_num_iterations": args.dense_num_iterations,
            "dense_num_samples": args.dense_num_samples,
            "dense_window_step": args.dense_window_step,
        }
    if getattr(args, "profile", "manual") not in ("verified-fast", "verified-fast-adaptive"):
        return {"automatic": False}
    return {
        "automatic": False,
        "input_frames": frame_count,
        "dense_target_references": args.dense_target_references,
        "reference_policy": "adaptive_stride",
        "dense_frame_stride": args.dense_frame_stride,
        "dense_source_images": args.dense_source_images,
        "dense_max_image_size": args.dense_max_image_size,
        "dense_num_iterations": args.dense_num_iterations,
        "dense_num_samples": args.dense_num_samples,
        "dense_window_step": args.dense_window_step,
    }
