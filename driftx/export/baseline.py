"""Baseline export helpers around the vendored prediction/export interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_baseline(prediction: Any, output_dir: str | Path) -> dict[str, str]:
    """Write vendor-compatible GLB/NPZ exports and a point-cloud PLY."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep GLB and mini-NPZ generation on the vendored dispatcher so the
    # benchmark uses the same export semantics as the existing validated runs.
    from third_party.depth_anything_3.utils.export import export

    export(prediction, "glb-mini_npz", str(output_path))

    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise RuntimeError("PLY export requires the declared numpy and trimesh dependencies") from exc

    depth = np.asarray(prediction.depth)
    conf = np.asarray(prediction.conf) if prediction.conf is not None else np.ones_like(depth)
    images = np.asarray(prediction.processed_images)
    intrinsics = np.asarray(prediction.intrinsics)
    extrinsics = np.asarray(prediction.extrinsics)

    points, colors = _depth_to_world_points(depth, intrinsics, extrinsics, images, conf)
    ply_path = output_path / "scene.ply"
    trimesh.PointCloud(vertices=points, colors=colors).export(ply_path)

    npz_path = output_path / "exports" / "mini_npz" / "results.npz"
    return {
        "glb": str(output_path / "scene.glb"),
        "ply": str(ply_path),
        "npz": str(npz_path),
        "depth_vis": str(output_path / "depth_vis"),
    }


def _depth_to_world_points(depth, intrinsics, extrinsics, images, conf, max_points: int = 1_000_000):
    """Back-project valid depth pixels to a compact PLY point cloud."""
    import numpy as np

    points_all = []
    colors_all = []
    for depth_frame, K, ext, image, conf_frame in zip(
        depth, intrinsics, extrinsics, images, conf
    ):
        h, w = depth_frame.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        valid = np.isfinite(depth_frame) & (depth_frame > 0) & np.isfinite(conf_frame)
        if not np.any(valid):
            continue
        pixels = np.stack((xx[valid], yy[valid], np.ones(valid.sum(), dtype=np.float32)), axis=1)
        camera_points = (np.linalg.inv(K) @ pixels.T).T * depth_frame[valid, None]
        ext44 = np.eye(4, dtype=np.float32) if ext.shape == (3, 4) else ext
        if ext.shape == (3, 4):
            ext44[:3] = ext
        world = (np.linalg.inv(ext44) @ np.c_[camera_points, np.ones(len(camera_points))].T).T[:, :3]
        points_all.append(world)
        colors_all.append(image[valid])

    if not points_all:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.uint8)
    points = np.concatenate(points_all)
    colors = np.concatenate(colors_all).astype(np.uint8, copy=False)
    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[keep]
        colors = colors[keep]
    if colors.shape[1] == 3:
        colors = np.c_[colors, np.full(len(colors), 255, dtype=np.uint8)]
    return points, colors
