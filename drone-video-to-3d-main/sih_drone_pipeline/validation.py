from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from pyproj import Transformer

from .telemetry import TelemetrySample, interpolate


def _camera_centers(images_txt: Path) -> dict[str, np.ndarray]:
    centers: dict[str, np.ndarray] = {}
    lines = [line for line in images_txt.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            int(parts[0])
            int(parts[8])
        except ValueError:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        translation = np.array(list(map(float, parts[5:8])))
        rotation = np.array([
            [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
        ])
        centers[" ".join(parts[9:])] = -(rotation.T @ translation)
    return centers


def _to_enu(sample: TelemetrySample, origin: dict) -> np.ndarray:
    transform = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x, y, z = transform.transform(sample.longitude, sample.latitude, sample.altitude_m)
    x0, y0, z0 = transform.transform(origin["longitude"], origin["latitude"], origin["altitude_m"])
    lat, lon = math.radians(origin["latitude"]), math.radians(origin["longitude"])
    rotation = np.array([
        [-math.sin(lon), math.cos(lon), 0],
        [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
        [math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)],
    ])
    return rotation @ np.array([x - x0, y - y0, z - z0])


def gps_alignment_report(
    images_txt: str | Path,
    frames_csv: str | Path,
    telemetry: list[TelemetrySample],
    origin: dict,
) -> dict:
    centers = _camera_centers(Path(images_txt))
    residuals = []
    residual_vectors = []
    per_camera_residuals = []
    normalized_residuals = []
    with Path(frames_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["image_name"] not in centers:
                continue
            sample = interpolate(telemetry, float(row["time_s"]))
            expected = _to_enu(sample, origin)
            vector = centers[row["image_name"]] - expected
            residual = float(np.linalg.norm(vector))
            residuals.append(residual)
            residual_vectors.append(vector)
            per_camera_residuals.append({
                "image_name": row["image_name"],
                "time_s": float(row["time_s"]),
                "residual_m": residual,
                "east_m": float(vector[0]),
                "north_m": float(vector[1]),
                "up_m": float(vector[2]),
            })
            horizontal_std = sample.horizontal_accuracy_m
            vertical_std = sample.vertical_accuracy_m
            if horizontal_std and vertical_std and horizontal_std > 0 and vertical_std > 0:
                normalized_residuals.append(float(np.sqrt(
                    (vector[0] / horizontal_std) ** 2
                    + (vector[1] / horizontal_std) ** 2
                    + (vector[2] / vertical_std) ** 2
                )))
    if not residuals:
        return {"matched_cameras": 0, "warning": "No registered image names matched the frame manifest"}
    values = np.asarray(residuals)
    vectors = np.asarray(residual_vectors)
    return {
        "matched_cameras": len(values),
        "gps_alignment_rmse_m": float(np.sqrt(np.mean(values**2))),
        "gps_alignment_median_m": float(np.median(values)),
        "gps_alignment_p95_m": float(np.percentile(values, 95)),
        "gps_alignment_rmse_east_m": float(np.sqrt(np.mean(vectors[:, 0] ** 2))),
        "gps_alignment_rmse_north_m": float(np.sqrt(np.mean(vectors[:, 1] ** 2))),
        "gps_alignment_rmse_up_m": float(np.sqrt(np.mean(vectors[:, 2] ** 2))),
        "normalized_position_residual_median_sigma": (
            float(np.median(normalized_residuals)) if normalized_residuals else None
        ),
        "per_camera_residuals": per_camera_residuals,
        "note": "GPS alignment residual is not an independent ground-control accuracy test.",
    }


def independent_distance_report(path: str | Path) -> dict:
    errors = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            measured = np.linalg.norm(
                np.array([float(row[f"x2_m"]), float(row[f"y2_m"]), float(row[f"z2_m"])])
                - np.array([float(row[f"x1_m"]), float(row[f"y1_m"]), float(row[f"z1_m"])])
            )
            errors.append(float(measured - float(row["known_distance_m"])))
    values = np.asarray(errors)
    return {
        "checks": len(values),
        "distance_rmse_m": float(np.sqrt(np.mean(values**2))) if len(values) else None,
        "maximum_absolute_error_m": float(np.max(np.abs(values))) if len(values) else None,
        "passes_one_metre_target": bool(len(values) and np.max(np.abs(values)) <= 1.0),
    }


_ASPRS_MINIMUM_CHECKPOINTS = 30
_CHECKPOINT_UNCERTAINTY_FIELDS = {
    "x": (
        "x_uncertainty_m", "uncertainty_x_m", "checkpoint_x_uncertainty_m",
        "survey_x_uncertainty_m",
    ),
    "y": (
        "y_uncertainty_m", "uncertainty_y_m", "checkpoint_y_uncertainty_m",
        "survey_y_uncertainty_m",
    ),
    "z": (
        "z_uncertainty_m", "uncertainty_z_m", "checkpoint_z_uncertainty_m",
        "survey_z_uncertainty_m",
    ),
    "horizontal": (
        "horizontal_uncertainty_m", "checkpoint_horizontal_uncertainty_m",
        "survey_horizontal_uncertainty_m", "horizontal_accuracy_m",
    ),
    "vertical": (
        "vertical_uncertainty_m", "checkpoint_vertical_uncertainty_m",
        "survey_vertical_uncertainty_m", "vertical_accuracy_m",
    ),
}


def _error_statistics(values: np.ndarray) -> dict:
    """Return population statistics for one checkpoint-error component."""
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "p95": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _checkpoint_uncertainty(row: dict[str, str]) -> tuple[dict[str, float], dict[str, str]]:
    """Read optional survey uncertainty columns without converting confidence levels."""
    values: dict[str, float] = {}
    source_fields: dict[str, str] = {}
    for component, aliases in _CHECKPOINT_UNCERTAINTY_FIELDS.items():
        for field in aliases:
            raw = row.get(field, "")
            if raw is None or str(raw).strip() == "":
                continue
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Checkpoint uncertainty {field} must be a finite non-negative number")
            values[component] = value
            source_fields[component] = field
            break
    return values, source_fields


def surveyed_checkpoint_report(path: str | Path) -> dict:
    """Compare reconstructed surface checkpoints with independent surveyed XYZ.

    Coordinates must already use the reconstruction's metric ENU frame. This
    intentionally performs no similarity alignment: allowing scale fitting here
    would hide the metric/georeferencing error the check is intended to expose.
    """
    errors = []
    error_vectors = []
    identifiers = []
    per_checkpoint_uncertainty = {}
    uncertainty_sources = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            required = (
                "known_x_m", "known_y_m", "known_z_m",
                "reconstructed_x_m", "reconstructed_y_m", "reconstructed_z_m",
            )
            if any(row.get(field, "").strip() == "" for field in required):
                continue
            known = np.array([float(row[f"known_{axis}_m"]) for axis in "xyz"])
            reconstructed = np.array([float(row[f"reconstructed_{axis}_m"]) for axis in "xyz"])
            vector = reconstructed - known
            errors.append(float(np.linalg.norm(vector)))
            error_vectors.append(vector)
            identifier = row.get("checkpoint_id") or str(len(identifiers) + 1)
            identifiers.append(identifier)
            uncertainty, source_fields = _checkpoint_uncertainty(row)
            if uncertainty:
                per_checkpoint_uncertainty[identifier] = uncertainty
                uncertainty_sources.update(source_fields)
    values = np.asarray(errors)
    vectors = np.asarray(error_vectors)
    horizontal_errors = np.linalg.norm(vectors[:, :2], axis=1) if len(vectors) else np.asarray([])
    vertical_absolute_errors = np.abs(vectors[:, 2]) if len(vectors) else np.asarray([])
    x_errors = vectors[:, 0] if len(vectors) else np.asarray([])
    y_errors = vectors[:, 1] if len(vectors) else np.asarray([])
    z_errors = vectors[:, 2] if len(vectors) else np.asarray([])

    x_statistics = _error_statistics(x_errors)
    y_statistics = _error_statistics(y_errors)
    z_statistics = _error_statistics(z_errors)
    horizontal_statistics = _error_statistics(horizontal_errors)
    vertical_absolute_statistics = _error_statistics(vertical_absolute_errors)
    three_dimensional_statistics = _error_statistics(values)

    uncertainty_statistics = {}
    for component in _CHECKPOINT_UNCERTAINTY_FIELDS:
        component_values = np.asarray([
            uncertainty[component]
            for uncertainty in per_checkpoint_uncertainty.values()
            if component in uncertainty
        ])
        if len(component_values):
            uncertainty_statistics[component] = _error_statistics(component_values)

    checkpoint_count = len(values)
    asprs_count_compliant = checkpoint_count >= _ASPRS_MINIMUM_CHECKPOINTS
    return {
        "checks": checkpoint_count,
        "checkpoint_count": checkpoint_count,
        "checkpoint_ids": identifiers,
        "rmse_3d_m": float(np.sqrt(np.mean(values**2))) if len(values) else None,
        "rmse_x_m": float(np.sqrt(np.mean(x_errors**2))) if len(x_errors) else None,
        "rmse_y_m": float(np.sqrt(np.mean(y_errors**2))) if len(y_errors) else None,
        "rmse_z_m": float(np.sqrt(np.mean(z_errors**2))) if len(z_errors) else None,
        "rmse_horizontal_m": (
            float(np.sqrt(np.mean(horizontal_errors**2))) if len(horizontal_errors) else None
        ),
        "rmse_vertical_m": (
            float(np.sqrt(np.mean(z_errors**2))) if len(z_errors) else None
        ),
        "mean_3d_error_m": three_dimensional_statistics["mean"],
        "median_3d_error_m": three_dimensional_statistics["median"],
        "std_3d_error_m": three_dimensional_statistics["standard_deviation"],
        "minimum_3d_error_m": three_dimensional_statistics["minimum"],
        "maximum_3d_error_m": three_dimensional_statistics["maximum"],
        "p95_3d_error_m": three_dimensional_statistics["p95"],
        "horizontal_rmse_m": (
            float(np.sqrt(np.mean(horizontal_errors**2))) if len(horizontal_errors) else None
        ),
        "vertical_rmse_m": (
            float(np.sqrt(np.mean(z_errors**2))) if len(z_errors) else None
        ),
        "bias_east_north_up_m": vectors.mean(axis=0).tolist() if len(vectors) else None,
        "error_statistics_m": {
            "x_signed": x_statistics,
            "y_signed": y_statistics,
            "z_signed": z_statistics,
            "horizontal": horizontal_statistics,
            "vertical_absolute": vertical_absolute_statistics,
            "three_dimensional": three_dimensional_statistics,
        },
        "per_checkpoint_3d_error_m": {
            identifier: error for identifier, error in zip(identifiers, errors)
        },
        "passes_one_metre_target": bool(len(values) >= 3 and np.max(values) <= 1.0),
        "minimum_checks_required": 3,
        "asprs_minimum_checkpoints": _ASPRS_MINIMUM_CHECKPOINTS,
        "asprs_checkpoint_count_compliant": asprs_count_compliant,
        "asprs_checkpoint_count_status": (
            "checkpoint_count_compliant"
            if asprs_count_compliant
            else "reduced_assessment"
            if checkpoint_count >= 3
            else "insufficient_checkpoints"
        ),
        "asprs_compliance_scope": "checkpoint_count_only",
        "checkpoint_uncertainty": {
            "checkpoints_with_any_uncertainty": len(per_checkpoint_uncertainty),
            "source_fields": uncertainty_sources,
            "summary_m": uncertainty_statistics,
            "per_checkpoint_m": per_checkpoint_uncertainty,
            "note": "Values are reported as provided; confidence levels are not converted.",
        },
        "alignment_type": "none_direct_metric_frame_comparison",
        "note": (
            "Surface/checkpoint accuracy evidence; blank unmeasured template rows are ignored. "
            "ASPRS status assesses checkpoint count only; distribution, survey quality, and other "
            "standard requirements require separate verification."
        ),
    }


def ground_truth_trajectory_report(
    images_txt: str | Path,
    frames_csv: str | Path,
    ground_truth_csv: str | Path,
) -> dict:
    """Compare camera centres with independent positions using rigid alignment only."""
    centers = _camera_centers(Path(images_txt))
    ground_truth = []
    with Path(ground_truth_csv).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            ground_truth.append((float(row["time_s"]), np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])))
    ground_truth.sort(key=lambda item: item[0])
    if len(ground_truth) < 3:
        return {"matched_cameras": 0, "warning": "Ground-truth trajectory has fewer than three positions"}

    def truth_at(time_s: float) -> np.ndarray:
        if time_s <= ground_truth[0][0]:
            return ground_truth[0][1]
        if time_s >= ground_truth[-1][0]:
            return ground_truth[-1][1]
        for index in range(1, len(ground_truth)):
            if ground_truth[index][0] >= time_s:
                t0, p0 = ground_truth[index - 1]
                t1, p1 = ground_truth[index]
                return p0 + (time_s - t0) / (t1 - t0) * (p1 - p0)
        return ground_truth[-1][1]

    estimated, expected = [], []
    with Path(frames_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            center = centers.get(row["image_name"])
            if center is not None:
                estimated.append(center)
                expected.append(truth_at(float(row["time_s"])))
    if len(estimated) < 3:
        return {"matched_cameras": len(estimated), "warning": "Fewer than three registered cameras matched ground truth"}
    source, target = np.asarray(estimated), np.asarray(expected)
    source_centered, target_centered = source - source.mean(axis=0), target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = source_centered @ rotation + target.mean(axis=0)
    errors = np.linalg.norm(aligned - target, axis=1)
    estimated_path = np.linalg.norm(np.diff(source, axis=0), axis=1).sum()
    truth_path = np.linalg.norm(np.diff(target, axis=0), axis=1).sum()
    scale_ratio = float(estimated_path / truth_path) if truth_path else None
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {
        "matched_cameras": len(errors),
        "trajectory_rmse_m": rmse,
        "trajectory_median_error_m": float(np.median(errors)),
        "trajectory_p95_error_m": float(np.percentile(errors, 95)),
        "trajectory_scale_ratio": scale_ratio,
        "trajectory_scale_error_percent": abs(scale_ratio - 1.0) * 100 if scale_ratio is not None else None,
        "alignment_type": "rigid_rotation_translation_no_scale",
        "passes_one_metre_position_target": rmse <= 1.0,
        "note": "Independent relative camera-trajectory diagnostic; it does not prove absolute georeferencing or surface accuracy.",
    }
