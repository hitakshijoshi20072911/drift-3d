from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import numpy as np


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> dict:
    """Fit target = scale * source @ rotation + translation (Umeyama/Kabsch)."""
    if len(source) < 3:
        raise ValueError("At least three control points are required")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError("Control points are collinear or coincident")
    u, singular, vt = np.linalg.svd(source_centered.T @ target_centered / len(source))
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Control points have insufficient spatial spread")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * source_mean @ rotation
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }


def apply_similarity(points: np.ndarray, transform: dict | None) -> np.ndarray:
    if transform is None:
        return points
    rotation = np.asarray(transform["rotation"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    return float(transform["scale"]) * np.asarray(points) @ rotation + translation


def _known_point(row: dict[str, str], origin: dict | None) -> np.ndarray:
    xyz_fields = [f"known_{axis}_m" for axis in "xyz"]
    if all(row.get(field, "").strip() for field in xyz_fields):
        return np.asarray([float(row[field]) for field in xyz_fields], dtype=float)
    gps_fields = ["known_latitude", "known_longitude", "known_altitude_m"]
    if all(row.get(field, "").strip() for field in gps_fields):
        if origin is None:
            raise ValueError("Geodetic control points require the reconstruction ENU origin")
        from .telemetry import TelemetrySample
        from .validation import _to_enu

        return _to_enu(TelemetrySample(
            0.0,
            float(row["known_latitude"]),
            float(row["known_longitude"]),
            float(row["known_altitude_m"]),
        ), origin)
    raise ValueError(
        "Each control point needs known_x_m/known_y_m/known_z_m or "
        "known_latitude/known_longitude/known_altitude_m"
    )


def fit_control_point_transform(
    path: str | Path,
    origin: dict | None = None,
    inlier_threshold_m: float = 1.0,
) -> tuple[dict, dict]:
    """Robustly align reconstructed GCP coordinates to surveyed coordinates."""
    if inlier_threshold_m <= 0:
        raise ValueError("Control-point inlier threshold must be positive")
    source, target, identifiers = [], [], []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for index, row in enumerate(csv.DictReader(stream), start=1):
            reconstructed_fields = [f"reconstructed_{axis}_m" for axis in "xyz"]
            if not all(row.get(field, "").strip() for field in reconstructed_fields):
                continue
            source.append([float(row[field]) for field in reconstructed_fields])
            target.append(_known_point(row, origin))
            identifiers.append(row.get("control_id") or row.get("checkpoint_id") or str(index))
    if len(source) < 3:
        raise ValueError("At least three populated control-point rows are required")
    source_array, target_array = np.asarray(source), np.asarray(target)
    candidate_count = math.comb(len(source_array), 3)
    if candidate_count <= 256:
        candidates = list(itertools.combinations(range(len(source_array)), 3))
    else:
        rng = np.random.default_rng(26158)
        candidates_set: set[tuple[int, int, int]] = set()
        while len(candidates_set) < 256:
            candidates_set.add(tuple(sorted(rng.choice(len(source_array), 3, replace=False).tolist())))
        candidates = sorted(candidates_set)
    best_indices: np.ndarray | None = None
    best_rmse = float("inf")
    for sample_indices in candidates:
        try:
            candidate = _fit_similarity(source_array[list(sample_indices)], target_array[list(sample_indices)])
        except ValueError:
            continue
        residuals = np.linalg.norm(apply_similarity(source_array, candidate) - target_array, axis=1)
        inliers = np.flatnonzero(residuals <= inlier_threshold_m)
        if len(inliers) < 3:
            continue
        rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        if best_indices is None or (len(inliers), -rmse) > (len(best_indices), -best_rmse):
            best_indices, best_rmse = inliers, rmse
    if best_indices is None:
        raise ValueError("No consistent set of at least three control points was found")
    fitted = _fit_similarity(source_array[best_indices], target_array[best_indices])
    residuals = np.linalg.norm(apply_similarity(source_array, fitted) - target_array, axis=1)
    inlier_mask = residuals <= inlier_threshold_m
    inlier_values = residuals[inlier_mask]
    serializable = {
        "scale": fitted["scale"],
        "rotation": fitted["rotation"].tolist(),
        "translation": fitted["translation"].tolist(),
    }
    report = {
        "control_points": len(source_array),
        "inliers": int(inlier_mask.sum()),
        "outliers": int((~inlier_mask).sum()),
        "inlier_ids": [identifier for identifier, keep in zip(identifiers, inlier_mask) if keep],
        "outlier_ids": [identifier for identifier, keep in zip(identifiers, inlier_mask) if not keep],
        "inlier_threshold_m": inlier_threshold_m,
        "fit_rmse_m": float(np.sqrt(np.mean(inlier_values**2))),
        "fit_median_error_m": float(np.median(inlier_values)),
        "maximum_inlier_error_m": float(np.max(inlier_values)),
        "scale": fitted["scale"],
        "scale_change_percent": abs(fitted["scale"] - 1.0) * 100.0,
        "transform": serializable,
        "passes_fit_gate": bool(len(inlier_values) >= 3 and np.max(inlier_values) <= inlier_threshold_m),
        "note": (
            "Control points influence alignment and therefore are not independent accuracy evidence. "
            "Use separate surveyed checkpoints for the final accuracy claim."
        ),
    }
    return serializable, report
