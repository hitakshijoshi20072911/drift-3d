"""Benchmark dense reconstructions against reference point clouds or meshes.

The evaluator deliberately performs no registration or scale alignment.  Inputs
must already be expressed in the same metric coordinate system so the reported
distances retain their geospatial meaning.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from scipy.spatial import cKDTree


DEFAULT_THRESHOLDS_M = (0.05, 0.1, 0.25, 0.5, 1.0)
EVALUATOR_VERSION = 1


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_geometry(loaded: object, path: Path) -> trimesh.Trimesh | trimesh.points.PointCloud:
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in {path}")
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, (trimesh.Trimesh, trimesh.points.PointCloud)):
        raise ValueError(f"Unsupported geometry in {path}: {type(loaded).__name__}")
    return loaded


def _load_geometry(path: str | Path) -> tuple[trimesh.Trimesh | trimesh.points.PointCloud, str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    geometry = _as_geometry(trimesh.load(source, process=False), source)
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"No 3D vertices found in {source}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Geometry contains non-finite vertices: {source}")
    faces = np.asarray(getattr(geometry, "faces", ()))
    geometry_type = "mesh" if faces.ndim == 2 and len(faces) else "point_cloud"
    return geometry, geometry_type


def _sample_rows(points: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.random.default_rng(seed).choice(len(points), maximum, replace=False)
    return points[np.sort(indices)]


def _sample_mesh_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
    cross_products = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    valid = np.isfinite(areas) & (areas > np.finfo(np.float64).eps)
    triangles = triangles[valid]
    areas = areas[valid]
    if not len(areas):
        raise ValueError("Mesh has no non-degenerate surface triangles")

    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(areas), size=count, replace=True, p=areas / areas.sum())
    chosen = triangles[face_indices]
    first = np.sqrt(rng.random(count))
    second = rng.random(count)
    return (
        (1.0 - first)[:, None] * chosen[:, 0]
        + (first * (1.0 - second))[:, None] * chosen[:, 1]
        + (first * second)[:, None] * chosen[:, 2]
    )


def _evaluation_points(
    geometry: trimesh.Trimesh | trimesh.points.PointCloud,
    geometry_type: str,
    maximum: int,
    seed: int,
) -> np.ndarray:
    if geometry_type == "mesh":
        return _sample_mesh_surface(geometry, maximum, seed)  # type: ignore[arg-type]
    return _sample_rows(np.asarray(geometry.vertices, dtype=np.float64), maximum, seed)


def _lookup_points(
    geometry: trimesh.Trimesh | trimesh.points.PointCloud,
    geometry_type: str,
    surface_samples: np.ndarray,
) -> np.ndarray:
    # Preserve every measured point when building a point-cloud search tree.
    # Meshes have no discrete surface samples, so their area-weighted evaluation
    # samples also form the nearest-neighbour approximation of the surface.
    if geometry_type == "point_cloud":
        return np.asarray(geometry.vertices, dtype=np.float64)
    return surface_samples


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    distances, _ = cKDTree(target).query(source, k=1, workers=-1)
    distances = np.asarray(distances, dtype=np.float64)
    if not len(distances) or not np.isfinite(distances).all():
        raise ValueError("Nearest-neighbour evaluation produced invalid distances")
    return distances


def _distance_summary(distances: np.ndarray) -> dict[str, int | float]:
    return {
        "samples": int(len(distances)),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "rmse_m": float(np.sqrt(np.mean(np.square(distances)))),
        "p95_m": float(np.percentile(distances, 95)),
        "maximum_m": float(np.max(distances)),
    }


def _validated_thresholds(thresholds_m: Iterable[float]) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(value) for value in thresholds_m)
    except TypeError as error:
        raise ValueError("thresholds_m must be a non-empty iterable") from error
    if not thresholds:
        raise ValueError("thresholds_m must not be empty")
    if any(not np.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("Every distance threshold must be finite and positive")
    return thresholds


def evaluate_dense_geometry(
    reconstruction_path: str | Path,
    reference_path: str | Path,
    *,
    thresholds_m: Iterable[float] = DEFAULT_THRESHOLDS_M,
    max_samples: int = 100_000,
    seed: int = 26158,
) -> dict:
    """Evaluate reconstructed geometry against reference geometry in metres.

    Both paths may contain a point cloud or triangle mesh readable by ``trimesh``
    (including PLY).  Meshes are converted to deterministic area-weighted surface
    samples; point clouds are deterministically capped at ``max_samples``.

    Precision is the fraction of reconstruction samples within a threshold of
    the reference.  Completeness/recall is the fraction of reference samples
    within the same threshold of the reconstruction.  F1 is their harmonic mean.
    No alignment is performed.
    """
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    thresholds = _validated_thresholds(thresholds_m)

    reconstruction, reconstruction_type = _load_geometry(reconstruction_path)
    reference, reference_type = _load_geometry(reference_path)
    reconstruction_points = _evaluation_points(
        reconstruction, reconstruction_type, max_samples, seed
    )
    reference_points = _evaluation_points(reference, reference_type, max_samples, seed + 1)

    reconstruction_lookup = _lookup_points(
        reconstruction, reconstruction_type, reconstruction_points
    )
    reference_lookup = _lookup_points(reference, reference_type, reference_points)

    reconstruction_to_reference = _nearest_distances(
        reconstruction_points, reference_lookup
    )
    reference_to_reconstruction = _nearest_distances(
        reference_points, reconstruction_lookup
    )
    forward_summary = _distance_summary(reconstruction_to_reference)
    reverse_summary = _distance_summary(reference_to_reconstruction)
    reconstruction_sha256 = _sha256_file(reconstruction_path)
    reference_sha256 = _sha256_file(reference_path)

    threshold_metrics = []
    for threshold in thresholds:
        precision = float(np.mean(reconstruction_to_reference <= threshold))
        recall = float(np.mean(reference_to_reconstruction <= threshold))
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        threshold_metrics.append(
            {
                "threshold_m": threshold,
                "precision": precision,
                "completeness": recall,
                "recall": recall,
                "f1": float(f1),
            }
        )

    reconstruction_vertices = int(len(reconstruction.vertices))
    reference_vertices = int(len(reference.vertices))
    return {
        "alignment": "none_same_metric_coordinate_frame_required",
        "protocol": {
            "name": "bidirectional_nearest_neighbour_surface_distance",
            "version": EVALUATOR_VERSION,
            "thresholds_m": list(thresholds),
            "mesh_sampling": "deterministic_area_weighted",
        },
        "reconstruction": {
            "path": str(Path(reconstruction_path)),
            "sha256": reconstruction_sha256,
            "geometry_type": reconstruction_type,
            "vertices": reconstruction_vertices,
            "faces": int(len(getattr(reconstruction, "faces", ()))),
            "evaluated_samples": int(len(reconstruction_points)),
            "nearest_neighbour_target_points": int(len(reconstruction_lookup)),
        },
        "reference": {
            "path": str(Path(reference_path)),
            "sha256": reference_sha256,
            "geometry_type": reference_type,
            "vertices": reference_vertices,
            "faces": int(len(getattr(reference, "faces", ()))),
            "evaluated_samples": int(len(reference_points)),
            "nearest_neighbour_target_points": int(len(reference_lookup)),
        },
        "max_samples": max_samples,
        "seed": int(seed),
        "reference_is_distinct_from_reconstruction": (
            reference_sha256 != reconstruction_sha256
        ),
        "distances_m": {
            "reconstruction_to_reference": forward_summary,
            "reference_to_reconstruction": reverse_summary,
        },
        "threshold_metrics": threshold_metrics,
        "passes_one_metre_target": bool(
            forward_summary["maximum_m"] <= 1.0
            and reverse_summary["maximum_m"] <= 1.0
        ),
        "one_metre_target_definition": (
            "maximum sampled distance is at most 1 m in both directions"
        ),
    }
