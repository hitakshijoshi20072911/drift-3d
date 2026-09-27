"""Streaming-style DRIFTX benchmark runner around the DA3 inference API."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import shutil
import time
import traceback
from pathlib import Path
from typing import Any


PROFILES = {
    "smoke": {"sample_fps": 1.0, "max_frames": 16, "process_res": 504, "chunk_size": 8, "chunk_overlap": 2},
    "balanced": {"sample_fps": 2.0, "max_frames": 0, "process_res": 630, "chunk_size": 16, "chunk_overlap": 4},
    "quality": {"sample_fps": 2.0, "max_frames": 0, "process_res": 756, "chunk_size": 12, "chunk_overlap": 4},
}


def _validate_large_model(model: str) -> None:
    """Reject the Giant/Nested checkpoints previously found to OOM this baseline."""
    model_text = str(model).lower()
    if "giant" in model_text or "nested" in model_text:
        raise ValueError(
            "This benchmark is configured for DA3 Large 1.1, not a Giant/Nested checkpoint. "
            "Use depth-anything/DA3-LARGE-1.1 or a local copy of that checkpoint."
        )
    config_path = Path(model).expanduser() / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_name = str(config.get("model_name", "")).lower()
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read model configuration: {config_path}") from exc
        if model_name and ("giant" in model_name or "nested" in model_name):
            raise ValueError(f"Local checkpoint {model_name!r} is not DA3 Large 1.1.")


def _resolve_options(profile, sample_fps, max_frames, process_res, chunk_size, chunk_overlap):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {', '.join(PROFILES)}")
    defaults = PROFILES[profile]
    fps = defaults["sample_fps"] if sample_fps is None else sample_fps
    frame_limit = defaults["max_frames"] if max_frames is None else max_frames
    resolution = defaults["process_res"] if process_res is None else process_res
    size = defaults["chunk_size"] if chunk_size is None else chunk_size
    overlap = defaults["chunk_overlap"] if chunk_overlap is None else chunk_overlap
    if fps <= 0 or resolution <= 0 or size < 4 or overlap < 0 or overlap >= size:
        raise ValueError("FPS/resolution must be positive, chunk size must be >= 4, and overlap smaller than the chunk")
    if frame_limit < 0:
        raise ValueError("max_frames must be >= 0; use 0 for all sampled frames")
    return fps, (None if frame_limit == 0 else frame_limit), resolution, size, overlap


def _base_report(video, output, model, device, profile, sample_fps, max_frames, process_res,
                 chunk_size, chunk_overlap, precision):
    try:
        gs_available = importlib.util.find_spec("gsplat") is not None
    except (ImportError, ValueError):
        gs_available = False
    return {
        "schema_version": 2,
        "status": "not measured",
        "video": str(Path(video).expanduser().resolve()),
        "output": str(Path(output).expanduser().resolve()),
        "model_variant": model,
        "device": device,
        "profile": profile,
        "requested_sample_fps": sample_fps,
        "sample_fps": sample_fps,
        "max_frames": max_frames if max_frames is not None else "all",
        "process_resolution": process_res,
        "requested_chunk_size": chunk_size,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": 0,
        "precision_requested": precision,
        "final_precision": "not measured",
        "video_duration_s": "not measured",
        "source_fps": "not measured",
        "source_frames": "not measured",
        "frames_available": "not measured",
        "frames_sampled": "not measured",
        "frames_processed": 0,
        "video_decode_seconds": "not measured",
        "preprocessing_seconds": 0.0,
        "inference_seconds": 0.0,
        "export_seconds": "not measured",
        "total_runtime_seconds": "not measured",
        "frames_per_second": "not measured",
        "peak_gpu_memory_mb": "not measured",
        "mean_gpu_memory_mb": "not measured",
        "cuda_oom_retries": 0,
        "mean_confidence": "not measured",
        "median_confidence": "not measured",
        "valid_depth_pixels": "not measured",
        "valid_depth_percentage": "not measured",
        "model_size_mb": _model_size_mb(model),
        "gs_available": gs_available,
        "gs_error": None if gs_available else "Optional gsplat package is not installed; baseline exports are unaffected.",
        "gs_enabled": False,
        "stage_timings_seconds": {
            "video_decode": "not measured", "preprocessing": 0.0,
            "DA3_inference": 0.0, "chunk_alignment": 0.0,
            "point_fusion": 0.0, "GLB_export": 0.0,
            "PLY_export": 0.0, "NPZ_export": 0.0,
        },
        "alignment": [],
        "artifacts": {},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def _model_size_mb(model: str) -> float | str:
    path = Path(model).expanduser()
    if path.is_file():
        return round(path.stat().st_size / 1024**2, 2)
    if path.is_dir():
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return round(size / 1024**2, 2)
    return "not measured"


def _load_da3(model_name: str, device: str):
    try:
        from third_party.depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        if not str(exc).startswith("No module named 'third_party"):
            raise
        from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_name).to(device)
    model.eval()
    return model


def _install_driftx_runtime_controls(model: Any, torch: Any, device: str, precision: str,
                                     stage_timings: dict[str, float]) -> None:
    """Apply DRIFTX-only precision and timing wrappers to one loaded DA3 instance."""
    if str(device).startswith("cuda"):
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    else:
        dtype = None
    def controlled_forward(image, extrinsics=None, intrinsics=None, export_feat_layers=None,
                           infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced"):
        if dtype is None or precision == "fp32":
            with torch.inference_mode():
                return model.model(image, extrinsics, intrinsics, export_feat_layers,
                                   infer_gs, use_ray_pose, ref_view_strategy)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            return model.model(image, extrinsics, intrinsics, export_feat_layers,
                               infer_gs, use_ray_pose, ref_view_strategy)
    model.forward = controlled_forward

    original_preprocess = model._preprocess_inputs
    def timed_preprocess(*args, **kwargs):
        stage_started = time.perf_counter()
        result = original_preprocess(*args, **kwargs)
        stage_timings["preprocessing"] = stage_timings.get("preprocessing", 0.0) + (
            time.perf_counter() - stage_started
        )
        return result
    model._preprocess_inputs = timed_preprocess

    original_forward_stage = model._run_model_forward
    def timed_forward_stage(*args, **kwargs):
        stage_started = time.perf_counter()
        result = original_forward_stage(*args, **kwargs)
        stage_timings["DA3_inference"] = stage_timings.get("DA3_inference", 0.0) + (
            time.perf_counter() - stage_started
        )
        return result
    model._run_model_forward = timed_forward_stage


def _normalize_prediction(prediction: Any) -> None:
    import numpy as np

    for name in ("depth", "conf", "extrinsics", "intrinsics", "processed_images"):
        value = getattr(prediction, name, None)
        if value is None:
            continue
        setattr(prediction, name, np.asarray(value))
    if prediction.depth is None or prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("DA3 prediction is missing depth or camera parameters")
    if prediction.conf is None:
        prediction.conf = np.ones_like(prediction.depth, dtype=np.float32)
    if prediction.processed_images is None:
        raise RuntimeError("DA3 prediction did not return processed RGB images needed for exports")
    if prediction.extrinsics.shape[-2:] == (3, 4):
        ext = np.broadcast_to(np.eye(4, dtype=np.float32), (len(prediction.extrinsics), 4, 4)).copy()
        ext[:, :3] = prediction.extrinsics
        prediction.extrinsics = ext


def _run_stream_attempt(model: Any, torch: Any, frame_paths: list[str], source_frame_ids: tuple[int, ...],
                        output_dir: Path, device: str, process_res: int, precision: str,
                        chunk_size: int, chunk_overlap: int, duration: float,
                        stage_timings: dict[str, float], progress_started: float,
                        progress_callback=None):
    import numpy as np
    from depth_anything_3.specs import Prediction
    from driftx.benchmark.streaming import align_overlap, iter_chunks, report_progress

    chunks = list(iter_chunks(len(frame_paths), chunk_size, chunk_overlap))
    if len(chunks) > 1 and chunk_overlap == 0:
        raise ValueError("chunk_overlap must be at least 1 when a video spans multiple chunks")
    temp_dir = output_dir / ".driftx_stream_tmp"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True)
    maps = None
    cursor = 0
    previous = None
    previous_chunk = None
    alignments = []
    memory_samples = []

    try:
        for chunk_index, chunk in enumerate(chunks, 1):
            prediction = model.inference(
                image=frame_paths[chunk.start:chunk.end],
                export_dir=None,
                export_format="glb",
                process_res=process_res,
            )
            _normalize_prediction(prediction)
            if prediction.depth.shape[0] != chunk.end - chunk.start:
                raise RuntimeError("DA3 returned a different number of predictions than input frames")
            if maps is None:
                n, h, w = len(frame_paths), *prediction.depth.shape[1:]
                maps = {
                    "depth": np.lib.format.open_memmap(temp_dir / "depth.npy", mode="w+", dtype=np.float32, shape=(n, h, w)),
                    "conf": np.lib.format.open_memmap(temp_dir / "conf.npy", mode="w+", dtype=np.float32, shape=(n, h, w)),
                    "intrinsics": np.lib.format.open_memmap(temp_dir / "intrinsics.npy", mode="w+", dtype=np.float32, shape=(n, 3, 3)),
                    "extrinsics": np.lib.format.open_memmap(temp_dir / "extrinsics.npy", mode="w+", dtype=np.float32, shape=(n, 4, 4)),
                    "images": np.lib.format.open_memmap(temp_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(n, h, w, 3)),
                }

            take_from = 0
            if previous is not None:
                aligned_start = time.perf_counter()
                overlap_now = previous_chunk.end - chunk.start
                alignment = align_overlap(previous, prediction, overlap_now)
                alignment.update({"previous_chunk": chunk_index - 2, "current_chunk": chunk_index - 1})
                alignments.append(alignment)
                stage_timings["chunk_alignment"] = stage_timings.get("chunk_alignment", 0.0) + (
                    time.perf_counter() - aligned_start
                )
                take_from = overlap_now
            amount = prediction.depth.shape[0] - take_from
            target_slice = slice(cursor, cursor + amount)
            local_slice = slice(take_from, None)
            maps["depth"][target_slice] = prediction.depth[local_slice]
            maps["conf"][target_slice] = prediction.conf[local_slice]
            maps["intrinsics"][target_slice] = prediction.intrinsics[local_slice]
            maps["extrinsics"][target_slice] = prediction.extrinsics[local_slice]
            maps["images"][target_slice] = prediction.processed_images[local_slice]
            cursor += amount
            if progress_callback is not None:
                progress_callback(cursor)
            if previous is not None:
                del previous
            previous = prediction
            previous_chunk = chunk

            if str(device).startswith("cuda") and torch.cuda.is_available():
                memory_samples.append(float(torch.cuda.memory_allocated(device)) / 1024**2)
            report_progress(
                video_duration=duration, sampled=len(frame_paths), processed=cursor,
                chunk_index=chunk_index, chunk_count=len(chunks), chunk_size=chunk_size,
                resolution=process_res, precision=precision,
                device_name=(torch.cuda.get_device_name(device) if str(device).startswith("cuda") and torch.cuda.is_available() else str(device)),
                started=progress_started, torch_module=torch, device=device,
            )
            # The previous window remains live only until this window is aligned.
            # Python reference counting releases it without a costly per-window GC.

        if cursor != len(frame_paths):
            raise RuntimeError(f"Streaming merge covered {cursor} of {len(frame_paths)} frames")
        for arr in maps.values():
            arr.flush()
        output = Prediction(
            depth=maps["depth"], conf=maps["conf"], extrinsics=maps["extrinsics"],
            intrinsics=maps["intrinsics"], processed_images=maps["images"],
            is_metric=int(getattr(previous, "is_metric", 0)),
        )
        output.frame_ids = np.asarray(source_frame_ids, dtype=np.int64)
        return output, chunks, alignments, memory_samples, temp_dir
    except Exception:
        for arr in (maps or {}).values():
            try:
                arr.flush()
                mmap = getattr(arr, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            except Exception:
                pass
        raise


def run_benchmark(
    video: str,
    output: str,
    model: str = "depth-anything/DA3-LARGE-1.1",
    device: str = "auto",
    sample_fps: float | None = None,
    process_res: int | None = None,
    max_frames: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    profile: str = "smoke",
    precision: str = "auto",
    auto_memory: bool = True,
) -> dict[str, Any]:
    """Run a reproducible benchmark, staging each DA3 inference window on CPU."""
    sample_fps, max_frames, process_res, requested_chunk, requested_overlap = _resolve_options(
        profile, sample_fps, max_frames, process_res, chunk_size, chunk_overlap
    )
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    report = _base_report(
        video, output_path, model, device, profile, sample_fps, max_frames,
        process_res, requested_chunk, requested_overlap, precision,
    )
    report_path = output_path / "run_report.json"
    started = time.perf_counter()
    final_tmp = None
    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _validate_large_model(model)
        from driftx.ingest import extract_video_frames
        from driftx.benchmark.streaming import (
            amp_precision, choose_memory_chunk_size, oom_fallback_size, is_cuda_oom,
        )
        import torch

        decode_started = time.perf_counter()
        info, frame_paths = extract_video_frames(video, output_path, sample_fps, max_frames)
        report["video_decode_seconds"] = time.perf_counter() - decode_started
        report["frames_available"] = len(range(0, info.source_frames, info.frame_interval))
        report["source_frames"] = info.source_frames
        report["frames_sampled"] = len(frame_paths)
        report["frames_processed"] = 0
        report["video_duration_s"] = info.duration_seconds
        report["source_fps"] = info.source_fps
        report["sample_fps"] = info.sampled_fps
        report["stage_timings_seconds"]["video_decode"] = report["video_decode_seconds"]
        report["stage_timings_seconds"]["preprocessing"] = 0.0
        actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        report["device"] = actual_device
        if actual_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch has no available CUDA device")
        final_precision = amp_precision(torch, actual_device, precision)
        report["final_precision"] = final_precision

        model_obj = _load_da3(model, actual_device)
        _install_driftx_runtime_controls(
            model_obj, torch, actual_device, final_precision, report["stage_timings_seconds"]
        )
        active_chunk = choose_memory_chunk_size(torch, actual_device, requested_chunk, auto_memory)
        active_overlap = min(requested_overlap, active_chunk - 1)
        report["chunk_size"] = active_chunk
        report["chunk_overlap"] = active_overlap
        if actual_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(actual_device)

        while True:
            try:
                report["frames_processed"] = 0
                result, chunks, alignments, gpu_samples, final_tmp = _run_stream_attempt(
                    model_obj, torch, frame_paths, info.source_frame_ids, output_path,
                    actual_device, process_res, final_precision, active_chunk, active_overlap,
                    info.duration_seconds, report["stage_timings_seconds"], started,
                    progress_callback=lambda processed: report.__setitem__("frames_processed", processed),
                )
                report["num_chunks"] = len(chunks)
                report["alignment"] = alignments
                if gpu_samples:
                    report["mean_gpu_memory_mb"] = round(sum(gpu_samples) / len(gpu_samples), 2)
                break
            except Exception as exc:
                if not is_cuda_oom(exc, torch):
                    raise
                retry_size = oom_fallback_size(active_chunk)
                if retry_size is None or retry_size <= active_overlap:
                    raise RuntimeError(
                        f"CUDA OOM persisted at chunk_size={active_chunk}; no safe smaller window remains. "
                        "Lower process resolution or close other GPU applications."
                    ) from exc
                report["cuda_oom_retries"] += 1
                print(f"CUDA OOM at chunk_size={active_chunk}; retrying the full sequence with chunk_size={retry_size}", flush=True)
                active_chunk = retry_size
                active_overlap = min(active_overlap, active_chunk - 1)
                report["chunk_size"] = active_chunk
                report["chunk_overlap"] = active_overlap
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                shutil.rmtree(output_path / ".driftx_stream_tmp", ignore_errors=True)

        report["stage_timings_seconds"]["DA3_inference"] = report["stage_timings_seconds"].get("DA3_inference", 0.0)
        report["stage_timings_seconds"]["preprocessing"] = report["stage_timings_seconds"].get("preprocessing", 0.0)
        report["preprocessing_seconds"] = report["stage_timings_seconds"]["preprocessing"]
        report["inference_seconds"] = (
            report["stage_timings_seconds"].get("preprocessing", 0.0)
            + report["stage_timings_seconds"]["DA3_inference"]
        )

        import numpy as np
        depth = np.asarray(result.depth)
        confidence = np.asarray(result.conf)
        valid_depth = np.isfinite(depth) & (depth > 0)
        report["valid_depth_pixels"] = int(valid_depth.sum())
        report["valid_depth_percentage"] = float(100.0 * valid_depth.mean())
        finite_conf = confidence[np.isfinite(confidence)]
        if finite_conf.size:
            report["mean_confidence"] = float(finite_conf.mean())
            report["median_confidence"] = float(np.median(finite_conf))

        _write_frame_metadata(output_path, info, frame_paths, result)
        export_timings: dict[str, float] = {}
        export_started = time.perf_counter()
        from driftx.export import export_baseline
        report["artifacts"] = export_baseline(result, output_path, stage_timings=export_timings)
        report["export_seconds"] = time.perf_counter() - export_started
        report["stage_timings_seconds"].update(export_timings)
        report["artifacts"].update({
            "frames": str(output_path / "frames.json"),
            "camera_poses": str(output_path / "camera_poses.json"),
            "metrics": str(output_path / "metrics.json"),
        })
        report["status"] = "completed"
        if actual_device.startswith("cuda") and torch.cuda.is_available():
            report["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated(actual_device) / 1024**2, 2)
        report["total_runtime_seconds"] = time.perf_counter() - started
    except Exception as exc:
        report["status"] = "not measured"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["error_traceback"] = traceback.format_exc()
        report["total_runtime_seconds"] = time.perf_counter() - started
    finally:
        result_obj = locals().get("result")
        if result_obj is not None:
            for name in ("depth", "conf", "extrinsics", "intrinsics", "processed_images"):
                arr = getattr(result_obj, name, None)
                mmap = getattr(arr, "_mmap", None)
                if mmap is not None:
                    try:
                        arr.flush()
                        mmap.close()
                    except Exception:
                        pass
        if final_tmp is not None:
            shutil.rmtree(final_tmp, ignore_errors=True)
        if isinstance(report.get("total_runtime_seconds"), (float, int)):
            report["frames_per_second"] = report["frames_processed"] / max(
                report["total_runtime_seconds"], 1e-9
            )
        metrics = {
            "schema_version": report["schema_version"],
            "status": report["status"],
            "error": report.get("error"),
            "model_variant": report["model_variant"],
            "profile": report["profile"],
            "precision_requested": report["precision_requested"],
            "requested_chunk_size": report["requested_chunk_size"],
            **{key: report.get(key) for key in (
                "video_duration_s", "source_fps", "source_frames", "sample_fps", "frames_available",
                "frames_sampled", "frames_processed", "chunk_size", "chunk_overlap",
                "num_chunks", "process_resolution", "inference_seconds", "export_seconds",
                "total_runtime_seconds", "peak_gpu_memory_mb", "mean_gpu_memory_mb",
                "frames_per_second",
                "mean_confidence", "median_confidence", "valid_depth_pixels",
                "valid_depth_percentage", "model_size_mb", "cuda_oom_retries",
                "final_precision", "device", "gs_available", "gs_error", "gs_enabled",
            )},
            "stage_timings_seconds": report["stage_timings_seconds"],
        }
        (output_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _write_frame_metadata(output_dir: Path, info: Any, frame_paths: list[str], prediction: Any) -> None:
    import numpy as np
    frame_rows = []
    pose_rows = []
    for i, (source_id, image_path) in enumerate(zip(info.source_frame_ids, frame_paths)):
        frame_rows.append({"frame_id": int(source_id), "timestamp_s": source_id / info.source_fps, "image": str(Path(image_path).relative_to(output_dir))})
        c2w = np.linalg.inv(np.asarray(prediction.extrinsics[i], dtype=np.float64))
        pose_rows.append({
            "frame_id": int(source_id),
            "intrinsics": np.asarray(prediction.intrinsics[i]).tolist(),
            "camera_to_world": c2w.tolist(),
        })
    (output_dir / "frames.json").write_text(json.dumps(frame_rows, indent=2) + "\n", encoding="utf-8")
    (output_dir / "camera_poses.json").write_text(json.dumps(pose_rows, indent=2) + "\n", encoding="utf-8")
