"""Memory-bounded chunk inference and overlap-based Sim(3) alignment for DRIFTX.

This module is DRIFTX orchestration code. It reuses the DA3 public inference
API and implements its own conservative CPU-side overlap alignment; it does not
vendor or modify the official DA3-Streaming pipeline.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int


def iter_chunks(frame_count: int, chunk_size: int, overlap: int) -> Iterator[Chunk]:
    """Yield half-open overlapping windows that cover each frame at least once."""
    if frame_count < 0:
        raise ValueError("frame_count cannot be negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    start = 0
    while start < frame_count:
        end = min(frame_count, start + chunk_size)
        yield Chunk(start, end)
        if end == frame_count:
            break
        start = end - overlap


def _as_44(extrinsic: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsic, dtype=np.float64)
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3] = ext
        return out
    raise ValueError(f"Expected a 3x4 or 4x4 extrinsic matrix, got {ext.shape}")


def _sampled_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    confidence: np.ndarray | None,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a sparse regular pixel grid to world coordinates on CPU."""
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    pixels = np.stack((xx.reshape(-1), yy.reshape(-1), np.ones(xx.size)), axis=0)
    values = np.asarray(depth)[::stride, ::stride].reshape(-1).astype(np.float64)
    valid = np.isfinite(values) & (values > 0)
    if confidence is not None:
        conf = np.asarray(confidence)[::stride, ::stride].reshape(-1).astype(np.float64)
        valid &= np.isfinite(conf) & (conf > 0)
    else:
        conf = np.ones_like(values)
    world = np.full((len(values), 3), np.nan, dtype=np.float64)
    weights = np.zeros(len(values), dtype=np.float64)
    if np.count_nonzero(valid) < 12:
        return world, weights

    ray = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64)) @ pixels[:, valid]
    camera = ray * values[valid][None, :]
    c2w = np.linalg.inv(_as_44(extrinsics))
    world[valid] = (c2w[:3, :3] @ camera + c2w[:3, 3:4]).T
    weights[valid] = np.sqrt(np.clip(conf[valid], 1e-6, None))
    return world, weights


def _weighted_sim3(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    """Estimate source->target Sim(3) using weighted Umeyama least squares."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-9, None)
    weights /= weights.sum()
    src_mean = np.sum(source * weights[:, None], axis=0)
    tgt_mean = np.sum(target * weights[:, None], axis=0)
    src_center = source - src_mean
    tgt_center = target - tgt_mean
    covariance = (tgt_center * weights[:, None]).T @ src_center
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    variance = np.sum(weights * np.sum(src_center * src_center, axis=1))
    if not np.isfinite(variance) or variance <= 1e-12:
        raise ValueError("Degenerate overlap: source points have no spatial variance")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate overlap: estimated Sim(3) scale is invalid")
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def align_overlap(
    previous: Any,
    current: Any,
    overlap: int,
    *,
    stride: int = 8,
) -> dict[str, float]:
    """Align ``current`` into the previous chunk's already-global coordinate frame.

    Correspondences are same-frame/same-pixel 3D points from the overlapping
    images, weighted by confidence and robustly reweighted by residual. A failed
    estimate is an explicit error: chunks are never silently concatenated in
    incompatible coordinate systems.
    """
    n = min(overlap, int(previous.depth.shape[0]), int(current.depth.shape[0]))
    if n <= 0:
        raise ValueError("At least one shared frame is required for chunk alignment")
    src_parts, tgt_parts, weight_parts = [], [], []
    for offset in range(n):
        previous_index = previous.depth.shape[0] - n + offset
        target_points, target_weights = _sampled_world_points(
            previous.depth[previous_index], previous.intrinsics[previous_index],
            previous.extrinsics[previous_index],
            previous.conf[previous_index] if previous.conf is not None else None,
            stride,
        )
        source_points, source_weights = _sampled_world_points(
            current.depth[offset], current.intrinsics[offset], current.extrinsics[offset],
            current.conf[offset] if current.conf is not None else None, stride,
        )
        if source_points.shape != target_points.shape:
            raise ValueError("Overlapping chunk predictions have different processed resolutions")
        shared = (source_weights > 0) & (target_weights > 0)
        if np.any(shared):
            src_parts.append(source_points[shared])
            tgt_parts.append(target_points[shared])
            weight_parts.append(np.sqrt(source_weights[shared] * target_weights[shared]))
    if not src_parts:
        raise ValueError("No valid depth/confidence correspondences in overlapping frames")
    source = np.concatenate(src_parts)
    target = np.concatenate(tgt_parts)
    weights = np.concatenate(weight_parts)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target, weights = source[valid], target[valid], weights[valid]
    if len(source) < 30:
        raise ValueError(f"Only {len(source)} valid overlap correspondences; at least 30 are required")

    keep = np.ones(len(source), dtype=bool)
    scale = 1.0
    for _ in range(5):
        scale, rotation, translation = _weighted_sim3(source[keep], target[keep], weights[keep])
        residual = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
        active = residual[keep]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        cutoff = median + max(3.0 * 1.4826 * mad, 1e-6)
        new_keep = residual <= cutoff
        if new_keep.sum() < 30 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    scale, rotation, translation = _weighted_sim3(source[keep], target[keep], weights[keep])
    residual = np.linalg.norm(scale * (source[keep] @ rotation.T) + translation - target[keep], axis=1)
    median_error = float(np.median(residual))
    target_scale = float(np.median(np.linalg.norm(target[keep], axis=1)))
    relative_error = median_error / max(target_scale, 1e-6)
    if not 0.02 <= scale <= 50.0:
        raise ValueError(f"Unreliable overlap scale estimate {scale:.4g}; refusing an unsafe merge")
    if not np.isfinite(relative_error) or relative_error > 0.35:
        raise ValueError(
            f"Overlap alignment is inconsistent (relative median error {relative_error:.3f}); "
            "increase overlap, use a lower sample FPS, or inspect the video/model."
        )

    _apply_sim3_to_prediction(current, scale, rotation, translation)
    return {
        "scale": float(scale),
        "median_error": median_error,
        "relative_median_error": relative_error,
        "correspondences": int(len(source)),
        "inliers": int(keep.sum()),
    }


def _apply_sim3_to_prediction(prediction: Any, scale: float, rotation: np.ndarray, translation: np.ndarray):
    """Apply a world Sim(3) while keeping camera-space depth and pose consistent."""
    prediction.depth *= scale
    transformed = []
    for extrinsic in prediction.extrinsics:
        c2w = np.linalg.inv(_as_44(extrinsic))
        center = scale * (rotation @ c2w[:3, 3]) + translation
        c2w[:3, :3] = rotation @ c2w[:3, :3]
        c2w[:3, 3] = center
        transformed.append(np.linalg.inv(c2w).astype(np.float32))
    prediction.extrinsics = np.stack(transformed)


def choose_memory_chunk_size(torch_module: Any, device: str, requested: int, enabled: bool = True) -> int:
    """Choose conservative VRAM-aware chunk sizing after the model is loaded."""
    if not enabled or not str(device).startswith("cuda") or not torch_module.cuda.is_available():
        return requested
    try:
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3
    # The model is already resident; leave headroom for activations and allocator variance.
    if free_gib < 1.5:
        estimate = 4
    elif free_gib < 2.5:
        estimate = 8
    elif free_gib < 4.0:
        estimate = 12
    else:
        estimate = 16
    return max(4, min(requested, estimate))


def oom_fallback_size(current: int) -> int | None:
    """Return the next conservative chunk size after an OOM, or None at floor."""
    for candidate in (32, 24, 16, 12, 8, 4):
        if candidate < current:
            return candidate
    return None


def is_cuda_oom(error: BaseException, torch_module: Any) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", ())
    return (bool(oom_type) and isinstance(error, oom_type)) or (
        isinstance(error, RuntimeError)
        and any(token in str(error).lower() for token in ("cuda out of memory", "cuda error: out of memory", "out of memory"))
    )


def amp_precision(torch_module: Any, device: str, requested: str) -> str:
    if requested != "auto":
        if requested == "bf16" and str(device).startswith("cuda"):
            if not torch_module.cuda.is_bf16_supported():
                raise ValueError("BF16 was requested but this PyTorch/CUDA device does not support it")
        if requested in ("fp16", "bf16") and not str(device).startswith("cuda"):
            raise ValueError(f"{requested.upper()} inference is only enabled for CUDA in this runner")
        return requested
    if not str(device).startswith("cuda"):
        return "fp32"
    return "bf16" if torch_module.cuda.is_bf16_supported() else "fp16"


def report_progress(
    *,
    video_duration: float,
    sampled: int,
    processed: int,
    chunk_index: int,
    chunk_count: int,
    chunk_size: int,
    resolution: int,
    precision: str,
    device_name: str,
    started: float,
    torch_module: Any,
    device: str,
) -> None:
    elapsed = time.perf_counter() - started
    speed = processed / elapsed if elapsed > 0 else 0.0
    eta = max(0.0, (sampled - processed) / speed) if speed > 0 else 0.0
    memory = "n/a"
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        try:
            allocated = torch_module.cuda.memory_allocated(device) / 1024**3
            total = torch_module.cuda.get_device_properties(device).total_memory / 1024**3
            memory = f"{allocated:.1f}/{total:.1f} GiB"
        except Exception:
            pass
    print(
        "\nDRIFTX STREAMING BENCHMARK\n"
        f"Duration          : {video_duration:.1f} s\n"
        f"Frames sampled    : {sampled}\n"
        f"Frames processed  : {processed} / {sampled}\n"
        f"Chunks            : {chunk_index} / {chunk_count}\n"
        f"Chunk size        : {chunk_size}\n"
        f"Resolution        : {resolution}\n"
        f"Precision         : {precision.upper()}\n"
        f"GPU / device      : {device_name}\n"
        f"VRAM              : {memory}\n"
        f"Elapsed / ETA     : {elapsed:.1f} / {eta:.1f} s",
        flush=True,
    )
