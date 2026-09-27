"""Thin orchestration around DA3's vendored Gaussian branch.

This module deliberately does not implement a second 3DGS system.  It asks the
existing DA3 API for ``Prediction.gaussians`` and delegates PLY/video writing
to the vendored DA3 exporters.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


def run_gaussian_reconstruction(
    model: Any,
    frame_paths: list[str],
    output_dir: str | Path,
    *,
    process_res: int,
    stage_timings: dict[str, float] | None = None,
) -> tuple[Any, dict[str, str]]:
    """Run one DA3 Gaussian inference and write stable DRIFTX artifact names.

    The Gaussian branch is intentionally a single DA3 call: unlike the depth
    baseline, its vendored renderer expects one coherent ``Prediction`` with a
    single Gaussian set.  The baseline streaming/GLB path remains separate.
    """
    if not frame_paths:
        raise ValueError("Gaussian reconstruction requires at least one frame")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timings = stage_timings if stage_timings is not None else {}
    inference_started = time.perf_counter()
    prediction = model.inference(
        image=frame_paths,
        infer_gs=True,
        export_dir=None,
        export_format="mini_npz",
        process_res=process_res,
    )
    timings["Gaussian_inference"] = timings.get("Gaussian_inference", 0.0) + (
        time.perf_counter() - inference_started
    )
    if getattr(prediction, "gaussians", None) is None:
        raise RuntimeError(
            "DA3 did not return Prediction.gaussians. Use a checkpoint that supports "
            "the vendored Gaussian branch and keep infer_gs=True."
        )

    # The upstream exporters create their own subdirectories.  Keep their
    # implementation and output format intact, but expose stable DRIFTX names.
    with tempfile.TemporaryDirectory(prefix="driftx-gaussian-", dir=output_path) as tmp:
        tmp_path = Path(tmp)
        export_started = time.perf_counter()
        try:
            from depth_anything_3.utils.export.gs import (
                export_to_gs_ply,
                export_to_gs_video,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Gaussian export requires the vendored DA3 export dependencies. "
                "Install the project requirements and the optional gsplat renderer."
            ) from exc
        export_to_gs_ply(prediction, str(tmp_path))
        export_to_gs_video(
            prediction,
            str(tmp_path),
            enable_tqdm=False,
            output_name="gaussian_preview",
            video_quality="medium",
        )
        timings["Gaussian_export"] = timings.get("Gaussian_export", 0.0) + (
            time.perf_counter() - export_started
        )
        ply_candidates = sorted((tmp_path / "gs_ply").glob("*.ply"))
        video_candidates = sorted((tmp_path / "gs_video").glob("*.mp4"))
        if not ply_candidates or not video_candidates:
            raise RuntimeError(
                "Gaussian export completed without both a PLY and preview video: "
                f"ply={len(ply_candidates)}, video={len(video_candidates)}"
            )
        gaussian_ply = output_path / "gaussian.ply"
        preview_video = output_path / "gaussian_preview.mp4"
        shutil.copyfile(ply_candidates[0], gaussian_ply)
        shutil.copyfile(video_candidates[0], preview_video)

    artifacts = {
        "gaussian_ply": str(gaussian_ply),
        "gaussian_preview": str(preview_video),
    }
    missing = [path for path in map(Path, artifacts.values()) if not path.is_file()]
    if missing:
        raise RuntimeError("Gaussian export incomplete; missing: " + ", ".join(map(str, missing)))
    return prediction, artifacts
