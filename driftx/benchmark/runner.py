"""Reproducible DRIFTX baseline benchmark runner."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any


def _not_measured() -> str:
    return "not measured"


def _base_report(video: str, output: str, model: str, device: str, sample_fps: float) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    reference_dir = repo_root / "outputs" / "large_576_dense"
    return {
        "schema_version": 1,
        "status": "not measured",
        "video": str(Path(video).expanduser().resolve()),
        "output": str(Path(output).expanduser().resolve()),
        "model_variant": model,
        "device": device,
        "requested_sampled_fps": sample_fps,
        "source_fps": _not_measured(),
        "sampled_fps": _not_measured(),
        "video_duration_seconds": _not_measured(),
        "frames_extracted": _not_measured(),
        "frames_processed": _not_measured(),
        "inference_time_seconds": _not_measured(),
        "export_time_seconds": _not_measured(),
        "total_wall_clock_seconds": _not_measured(),
        "valid_depth_pixels": _not_measured(),
        "mean_confidence": _not_measured(),
        "median_confidence": _not_measured(),
        "gpu_memory_peak_bytes": _not_measured(),
        "artifacts": {},
        "sanity_check": {
            "reference_output": str(reference_dir),
            "reference_glb_exists": (reference_dir / "scene.glb").is_file(),
            "reference_depth_vis_frames": len(list((reference_dir / "depth_vis").glob("*.jpg")))
            if (reference_dir / "depth_vis").is_dir()
            else 0,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def run_benchmark(video: str, output: str, model: str, device: str, sample_fps: float, process_res: int) -> dict[str, Any]:
    """Run a baseline and always persist ``run_report.json``."""
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    report = _base_report(video, output_path, model, device, sample_fps)
    report_path = output_path / "run_report.json"
    started = time.perf_counter()

    try:
        from driftx.ingest import extract_video_frames

        info, frame_paths = extract_video_frames(video, output_path, sample_fps)
        report.update(
            {
                "source_fps": info.source_fps,
                "video_duration_seconds": info.duration_seconds,
                "sampled_fps": info.sampled_fps,
                "frames_extracted": len(frame_paths),
            }
        )

        import numpy as np
        import torch
        from third_party.depth_anything_3.services.inference_service import run_inference

        actual_device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        report["device"] = actual_device
        if actual_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is available")

        if actual_device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(actual_device)
        inference_started = time.perf_counter()
        prediction = run_inference(
            image_paths=frame_paths,
            export_dir=None,
            model_dir=model,
            device=actual_device,
            export_format="glb-mini_npz",
            process_res=process_res,
        )
        report["inference_time_seconds"] = time.perf_counter() - inference_started
        report["frames_processed"] = int(np.asarray(prediction.depth).shape[0])

        depth = np.asarray(prediction.depth)
        conf = np.asarray(prediction.conf) if prediction.conf is not None else None
        valid_depth = np.isfinite(depth) & (depth > 0)
        report["valid_depth_pixels"] = int(valid_depth.sum())
        if conf is not None:
            valid_conf = conf[np.isfinite(conf)]
            if valid_conf.size:
                report["mean_confidence"] = float(valid_conf.mean())
                report["median_confidence"] = float(np.median(valid_conf))

        export_started = time.perf_counter()
        from driftx.export import export_baseline

        report["artifacts"] = export_baseline(prediction, output_path)
        report["export_time_seconds"] = time.perf_counter() - export_started
        if actual_device.startswith("cuda") and torch.cuda.is_available():
            report["gpu_memory_peak_bytes"] = int(torch.cuda.max_memory_allocated(actual_device))
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "not measured"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["total_wall_clock_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return report
