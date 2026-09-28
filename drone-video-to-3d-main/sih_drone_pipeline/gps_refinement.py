"""GPS-anchored refinement of a global-SfM model.

Global positioning is fast but ill-conditioned wherever the flight gives it little
parallax (low altitude, hover, near-vertical climbs): whole sections of the camera
path can come out stretched by several times while the rest of the model is
accurate to about a metre. Bundle adjustment with GPS priors can repair that, but
only slowly when it has to drag a section hundreds of metres, because it is a
local optimiser. So the repair is done in three steps:

1. re-anchor: cameras that disagree with their GPS position by more than a
   threshold (after a robust similarity fit) are moved onto it, keeping their
   orientation, and every 3D point is cleared;
2. re-triangulate the points with those poses held fixed;
3. one short GPS-anchored bundle adjustment pass, now starting near the answer.
"""
from __future__ import annotations

import math
import shutil
import sqlite3
from pathlib import Path

import numpy as np

from .utils import PipelineError, run_command

# Measured on the 4-minute reference flight: 0.75 m / 1.5 m kept every camera
# within GPS noise without raising reprojection error; 2 m / 3 m let a weakly
# constrained section drift by ~25 m.
DEFAULT_OPTIONS = {
    "prior_std_horizontal_m": 0.75,
    "prior_std_vertical_m": 1.5,
    "reanchor_threshold_m": 5.0,
}

_REFINE_FLAGS = (
    ("--Mapper.ba_global_max_refinements", "1"),
    ("--Mapper.ba_global_frames_ratio", "1.4"),
    ("--Mapper.ba_global_points_ratio", "1.4"),
    ("--Mapper.ba_refine_focal_length", "0"),
    ("--Mapper.ba_refine_principal_point", "0"),
    ("--Mapper.ba_refine_extra_params", "0"),
    ("--Mapper.multiple_models", "0"),
    ("--Mapper.extract_colors", "0"),
)


def _quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_reference_enu(references_path: str | Path) -> dict[str, np.ndarray]:
    """Read the pipeline's `name lat lon alt` reference file as local east/north/up metres."""
    rows = []
    for line in Path(references_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 4 and not line.startswith("#"):
            rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    if not rows:
        raise ValueError(f"No GPS references in {references_path}")
    lat0, lon0 = rows[0][1], rows[0][2]
    east_scale = 111320.0 * math.cos(math.radians(lat0))
    return {name: np.array([(lon - lon0) * east_scale, (lat - lat0) * 110540.0, alt])
            for name, lat, lon, alt in rows}


def reanchor_text_model(src_dir: str | Path, dst_dir: str | Path,
                        gps_enu: dict[str, np.ndarray], threshold_m: float) -> dict:
    """Move cameras that disagree with GPS by more than threshold_m onto their GPS
    position (orientation kept) and clear all 3D points. Works on COLMAP TXT models,
    including COLMAP 4 frames.txt, whose per-frame pose is kept identical to the image's."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    lines = (src_dir / "images.txt").read_text(encoding="utf-8").split("\n")
    image_rows = [i for i, line in enumerate(lines)
                  if not line.startswith("#") and len(line.split()) == 10
                  and line.split()[9].lower().endswith((".jpg", ".jpeg", ".png"))]
    ids, names, rotations, original_pose = [], [], [], {}
    for i in image_rows:
        parts = lines[i].split()
        ids.append(parts[0]); names.append(parts[9])
        original_pose[parts[0]] = np.array([float(v) for v in parts[1:8]])
        rotations.append(_quat_to_rotation(*[float(v) for v in parts[1:5]]))
    if len(image_rows) < 3:
        raise ValueError("Model has fewer than three registered images")
    centers = np.array([-R.T @ original_pose[i][4:7] for R, i in zip(rotations, ids)])
    targets = np.array([gps_enu.get(n, np.full(3, np.nan)) for n in names])
    have = np.isfinite(targets).all(1)
    if have.sum() < 3:
        raise ValueError("Fewer than three registered images have GPS references")

    # Robust similarity (model -> GPS): refit on the best 70% so a stretched
    # section cannot drag the fit.
    keep = have.copy()
    for _ in range(8):
        mx, my = centers[keep].mean(0), targets[keep].mean(0)
        U, S, Vt = np.linalg.svd((targets[keep] - my).T @ (centers[keep] - mx) / keep.sum())
        D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        rotation = U @ D @ Vt
        scale = np.trace(np.diag(S) @ D) / ((centers[keep] - mx) ** 2).sum(1).mean()
        residual = np.full(len(centers), np.inf)
        residual[have] = np.linalg.norm(
            targets[have] - (scale * (rotation @ (centers[have] - mx).T).T + my), axis=1)
        keep = have & (residual <= np.quantile(residual[have], 0.7))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate model-to-GPS similarity")
    move = have & (residual > threshold_m)

    new_translation = {}
    for k, i in enumerate(image_rows):
        parts = lines[i].split()
        if move[k]:
            center = rotation.T @ (targets[k] - my) / scale + mx   # GPS position in model coordinates
            translation = -rotations[k] @ center
            new_translation[ids[k]] = translation
            parts[5:8] = [repr(float(v)) for v in translation]
            lines[i] = " ".join(parts)
        points = lines[i + 1].split()           # POINTS2D: x y point3D_id triples
        points[2::3] = ["-1"] * len(points[2::3])
        lines[i + 1] = " ".join(points)

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    (dst_dir / "images.txt").write_text("\n".join(lines), encoding="utf-8")
    (dst_dir / "points3D.txt").write_text("# 3D point list with one line of data per point\n",
                                          encoding="utf-8")
    shutil.copy2(src_dir / "cameras.txt", dst_dir / "cameras.txt")
    if (src_dir / "rigs.txt").exists():
        shutil.copy2(src_dir / "rigs.txt", dst_dir / "rigs.txt")
    if (src_dir / "frames.txt").exists():
        frame_lines = (src_dir / "frames.txt").read_text(encoding="utf-8").split("\n")
        edited = 0
        for j, line in enumerate(frame_lines):
            parts = line.split()
            if not parts or line.startswith("#"):
                continue
            try:
                pose = np.array([float(v) for v in parts[2:9]])
            except ValueError as error:
                raise ValueError(f"frames.txt line not understood: {line[:160]}") from error
            match = [image_id for image_id, image_pose in original_pose.items()
                     if np.allclose(pose, image_pose, atol=1e-7)]
            if len(match) > 1:
                raise ValueError(f"Frame pose matches {len(match)} images: {line[:160]}")
            if match and match[0] in new_translation:
                parts[6:9] = [repr(float(v)) for v in new_translation[match[0]]]
                frame_lines[j] = " ".join(parts)
                edited += 1
        (dst_dir / "frames.txt").write_text("\n".join(frame_lines), encoding="utf-8")
        if edited != len(new_translation):
            raise ValueError(f"Updated {edited} frame poses but moved {len(new_translation)} images")
    return {
        "cameras": len(image_rows),
        "moved": int(move.sum()),
        "without_gps": int((~have).sum()),
        "median_residual_before_m": round(float(np.median(residual[have])), 3),
        "worst_residual_before_m": round(float(residual[have].max()), 3),
    }


def _supported(help_text: str, flags) -> tuple[list[str], list[str]]:
    applied, skipped = [], []
    for flag, value in flags:
        if flag.lstrip("-") in help_text:
            applied += [flag, value]
        else:
            skipped.append(flag)
    return applied, skipped


def _help(colmap: str, command: str) -> str:
    import subprocess

    result = subprocess.run([colmap, command, "--help"], capture_output=True, text=True,
                            errors="replace", check=False)
    return (result.stdout or "") + (result.stderr or "")


def _largest(root: Path) -> Path:
    models = [p.parent for p in root.rglob("images.bin")]
    if not models:
        raise PipelineError(f"No model written under {root}")
    return max(models, key=lambda m: (m / "images.bin").stat().st_size)


def refine_global_model(colmap: str, database: Path, images_dir: Path, model_dir: Path,
                        references_path: Path, work_dir: Path, report: dict,
                        options: dict | None = None) -> Path:
    """Return the path of the GPS-refined model. Raises on any failure; the caller
    keeps the unrefined global model in that case."""
    options = {**DEFAULT_OPTIONS, **(options or {})}
    with sqlite3.connect(database) as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        priors = (connection.execute("select count(*) from pose_priors").fetchone()[0]
                  if "pose_priors" in tables else 0)
    if priors < 3:
        raise PipelineError("database has no GPS pose priors (frames were not GPS-tagged)")

    if work_dir.exists():
        shutil.rmtree(work_dir)
    global_text, anchored, triangulated, refined = (
        work_dir / "global_text", work_dir / "anchored_text", work_dir / "triangulated", work_dir / "refined")
    for directory in (global_text, triangulated, refined):
        directory.mkdir(parents=True)

    run_command("gps_refine_export", [colmap, "model_converter", "--input_path", str(model_dir),
                                      "--output_path", str(global_text), "--output_type", "TXT"], report)
    anchor = reanchor_text_model(global_text, anchored, read_reference_enu(references_path),
                                 options["reanchor_threshold_m"])
    report["gps_refinement"].update(anchor)
    print(f"GPS re-anchor: moved {anchor['moved']} of {anchor['cameras']} cameras "
          f"(median residual before {anchor['median_residual_before_m']:.1f} m)", flush=True)

    triangulator_flags, _ = _supported(_help(colmap, "point_triangulator"), (
        ("--refine_intrinsics", "0"), ("--Mapper.ba_global_max_refinements", "1")))
    run_command("gps_refine_triangulation", [
        colmap, "point_triangulator", "--database_path", str(database), "--image_path", str(images_dir),
        "--input_path", str(anchored), "--output_path", str(triangulated), *triangulator_flags], report)

    refine_help = _help(colmap, "pose_prior_mapper")
    if "input_path" not in refine_help or "prior_position_std_x" not in refine_help:
        raise PipelineError("pose_prior_mapper cannot refine an existing model in this COLMAP build")
    refine_flags, skipped = _supported(refine_help, _REFINE_FLAGS)
    report["gps_refinement"]["options_skipped"] = skipped
    horizontal, vertical = options["prior_std_horizontal_m"], options["prior_std_vertical_m"]
    run_command("gps_refine_bundle_adjustment", [
        colmap, "pose_prior_mapper", "--database_path", str(database), "--image_path", str(images_dir),
        "--input_path", str(_largest(triangulated)), "--output_path", str(refined),
        "--overwrite_priors_covariance", "1",
        "--prior_position_std_x", str(horizontal), "--prior_position_std_y", str(horizontal),
        "--prior_position_std_z", str(vertical), *refine_flags], report)
    report["gps_refinement"].update({"prior_std_horizontal_m": horizontal, "prior_std_vertical_m": vertical,
                                     "reanchor_threshold_m": options["reanchor_threshold_m"]})
    return _largest(refined)
