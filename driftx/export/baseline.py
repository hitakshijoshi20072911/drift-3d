"""DRIFTX exports from the merged CPU-side DA3 prediction."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any


def export_baseline(
    prediction: Any,
    output_dir: str | Path,
    *,
    stage_timings: dict[str, float] | None = None,
    max_points: int = 1_000_000,
) -> dict[str, str]:
    """Write a global point-cloud GLB/PLY, compressed NPZ, and all depth views.

    Point generation is CPU-side and deterministically capped before geometry is
    materialized, so export memory is bounded by ``max_points`` rather than the
    video pixel count. The legacy ``exports/mini_npz/results.npz`` path remains.
    """
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timings = stage_timings if stage_timings is not None else {}

    fusion_started = time.perf_counter()
    points, colors = _depth_to_world_points(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        prediction.processed_images,
        prediction.conf,
        max_points=max_points,
    )
    timings["point_fusion"] = timings.get("point_fusion", 0.0) + time.perf_counter() - fusion_started

    glb_started = time.perf_counter()
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("GLB/PLY export requires the declared trimesh dependency") from exc

    # Keep the DA3 CV world frame for PLY/NPZ; align the GLB to conventional
    # glTF axes and center it using the same vendor utility as the DA3 exporter.
    try:
        from depth_anything_3.utils.export.glb import (
            _compute_alignment_transform_first_cam_glTF_center_by_points,
        )
        transform = _compute_alignment_transform_first_cam_glTF_center_by_points(
            np.asarray(prediction.extrinsics[0]), points
        )
        glb_points = trimesh.transform_points(points, transform) if len(points) else points
    except (ImportError, AttributeError):
        transform = np.eye(4)
        glb_points = points
    scene = trimesh.Scene()
    if len(glb_points):
        scene.add_geometry(trimesh.points.PointCloud(vertices=glb_points, colors=colors))
    scene.metadata["driftx_global_frame_count"] = int(prediction.depth.shape[0])
    scene.metadata["driftx_alignment_transform"] = np.asarray(transform).tolist()
    scene.export(output_path / "scene.glb")
    timings["GLB_export"] = timings.get("GLB_export", 0.0) + time.perf_counter() - glb_started

    ply_started = time.perf_counter()
    trimesh.PointCloud(vertices=points, colors=colors).export(output_path / "scene.ply")
    timings["PLY_export"] = timings.get("PLY_export", 0.0) + time.perf_counter() - ply_started

    npz_started = time.perf_counter()
    nested_npz = output_path / "exports" / "mini_npz" / "results.npz"
    nested_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "depth": prediction.depth,
        "conf": prediction.conf,
        "extrinsics": prediction.extrinsics,
        "intrinsics": prediction.intrinsics,
        "frame_ids": getattr(prediction, "frame_ids", np.arange(prediction.depth.shape[0])),
    }
    np.savez_compressed(nested_npz, **payload)
    root_npz = output_path / "results.npz"
    try:
        root_npz.unlink(missing_ok=True)
        os.link(nested_npz, root_npz)
    except OSError:
        shutil.copyfile(nested_npz, root_npz)
    timings["NPZ_export"] = timings.get("NPZ_export", 0.0) + time.perf_counter() - npz_started

    depth_vis_started = time.perf_counter()
    _write_depth_visualizations(prediction, output_path / "depth_vis")
    timings["depth_vis_export"] = time.perf_counter() - depth_vis_started

    artifacts = {
        "glb": str(output_path / "scene.glb"),
        "ply": str(output_path / "scene.ply"),
        "npz": str(root_npz),
        "legacy_mini_npz": str(nested_npz),
        "depth_vis": str(output_path / "depth_vis"),
    }
    missing = [p for p in (Path(artifacts["glb"]), Path(artifacts["ply"]), root_npz, nested_npz)
               if not p.is_file()]
    if not Path(artifacts["depth_vis"]).is_dir():
        missing.append(Path(artifacts["depth_vis"]))
    if missing:
        raise RuntimeError("Benchmark export incomplete; missing: " + ", ".join(map(str, missing)))
    return artifacts


def _write_depth_visualizations(prediction: Any, output_dir: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np
    from depth_anything_3.utils.visualize import visualize_depth

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.jpg"):
        old.unlink(missing_ok=True)
    for i, (depth, image) in enumerate(zip(prediction.depth, prediction.processed_images)):
        depth_vis = np.asarray(visualize_depth(depth), dtype=np.uint8)
        rgb = np.asarray(image, dtype=np.uint8)
        imageio.imwrite(output_dir / f"{i:06d}.jpg", np.concatenate((rgb, depth_vis), axis=1), quality=92)


def _depth_to_world_points(depth, intrinsics, extrinsics, images, conf, max_points: int = 1_000_000):
    """Vectorized per-frame back-projection with a deterministic global point cap."""
    import numpy as np

    n, h, w = depth.shape
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        valid = np.isfinite(depth[i]) & (depth[i] > 0) & np.isfinite(conf[i])
        counts[i] = int(np.count_nonzero(valid))
    total_valid = int(counts.sum())
    if total_valid == 0 or max_points <= 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.uint8)

    budget = min(int(max_points), total_valid)
    shares = counts.astype(np.float64) * (budget / total_valid)
    take = np.floor(shares).astype(np.int64)
    remainder = budget - int(take.sum())
    if remainder:
        order = np.argsort(-(shares - take))
        for index in order[:remainder]:
            take[index] += 1

    points, colors = [], []
    for i, count in enumerate(take):
        if count == 0:
            continue
        valid = np.isfinite(depth[i]) & (depth[i] > 0) & np.isfinite(conf[i])
        indices = np.flatnonzero(valid)
        if len(indices) > count:
            indices = indices[np.linspace(0, len(indices) - 1, count, dtype=np.int64)]
        yy, xx = np.divmod(indices, w)
        pixels = np.stack((xx, yy, np.ones(len(indices), dtype=np.float64)), axis=0)
        rays = np.linalg.inv(np.asarray(intrinsics[i], dtype=np.float64)) @ pixels
        camera = rays * np.asarray(depth[i]).reshape(-1)[indices][None, :]
        ext = np.asarray(extrinsics[i], dtype=np.float64)
        if ext.shape == (3, 4):
            ext44 = np.eye(4, dtype=np.float64)
            ext44[:3] = ext
        else:
            ext44 = ext
        c2w = np.linalg.inv(ext44)
        world = (c2w[:3, :3] @ camera + c2w[:3, 3:4]).T.astype(np.float32)
        points.append(world)
        color = np.asarray(images[i]).reshape(-1, np.asarray(images[i]).shape[-1])[indices]
        colors.append(color)
    pts = np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=np.float32)
    cols = np.concatenate(colors, axis=0).astype(np.uint8, copy=False) if colors else np.empty((0, 3), dtype=np.uint8)
    if cols.shape[1] == 3:
        cols = np.column_stack((cols, np.full(len(cols), 255, dtype=np.uint8)))
    return pts, cols
