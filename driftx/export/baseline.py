"""Baseline export helpers around the vendored prediction/export interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_baseline(prediction: Any, output_dir: str | Path) -> dict[str, str]:
    """Write vendor-compatible GLB/NPZ exports and a point-cloud PLY."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import numpy as np

    # The vendored mini-NPZ helper is asynchronous. Use its synchronous-safe
    # GLB/depth-visualization path, then write the small mini-NPZ here so the
    # benchmark never reports success before all artifacts exist.
    try:
        from third_party.depth_anything_3.utils.export import export
    except ModuleNotFoundError as exc:
        if not str(exc).startswith("No module named 'third_party"):
            raise
        from depth_anything_3.utils.export import export

    export(prediction, "glb", str(output_path), export_depth_vis=True)

    npz_path = output_path / "exports" / "mini_npz" / "results.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    mini_npz = {"depth": np.round(prediction.depth, 8)}
    if prediction.conf is not None:
        mini_npz["conf"] = np.round(prediction.conf, 2)
    if prediction.extrinsics is not None:
        mini_npz["extrinsics"] = prediction.extrinsics
    if prediction.intrinsics is not None:
        mini_npz["intrinsics"] = prediction.intrinsics
    np.savez_compressed(npz_path, **mini_npz)

    try:
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

    artifacts = {
        "glb": str(output_path / "scene.glb"),
        "ply": str(ply_path),
        "npz": str(npz_path),
        "depth_vis": str(output_path / "depth_vis"),
    }
    missing = [
        path
        for path in (Path(artifacts["glb"]), Path(artifacts["ply"]), Path(artifacts["npz"]))
        if not path.is_file()
    ]
    if not Path(artifacts["depth_vis"]).is_dir():
        missing.append(Path(artifacts["depth_vis"]))
    if missing:
        raise RuntimeError("Benchmark export incomplete; missing: " + ", ".join(map(str, missing)))
    return artifacts


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
