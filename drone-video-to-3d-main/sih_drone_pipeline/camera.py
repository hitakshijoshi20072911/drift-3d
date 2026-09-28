from __future__ import annotations

import json
from pathlib import Path


PARAMETER_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
}


def load_camera_calibration(path: str | Path | None) -> dict | None:
    """Load and validate a reusable COLMAP camera-calibration profile."""
    if path is None:
        return None
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    model = str(payload.get("model", "")).upper()
    if model not in PARAMETER_COUNTS:
        raise ValueError(
            f"Unsupported camera model {model!r}; choose one of {sorted(PARAMETER_COUNTS)}"
        )
    params = payload.get("params")
    if not isinstance(params, list) or len(params) != PARAMETER_COUNTS[model]:
        raise ValueError(
            f"Camera model {model} requires {PARAMETER_COUNTS[model]} numeric parameters"
        )
    params = [float(value) for value in params]
    if params[0] <= 0 or (model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"} and params[1] <= 0):
        raise ValueError("Camera focal length parameters must be positive")
    width = payload.get("width")
    height = payload.get("height")
    if width is not None and int(width) <= 0 or height is not None and int(height) <= 0:
        raise ValueError("Camera calibration width and height must be positive")
    return {
        "path": str(calibration_path.resolve()),
        "model": model,
        "params": params,
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "fixed_intrinsics": bool(payload.get("fixed_intrinsics", False)),
        "rolling_shutter_readout_ms": (
            float(payload["rolling_shutter_readout_ms"])
            if payload.get("rolling_shutter_readout_ms") is not None
            else None
        ),
        "digital_stabilization": payload.get("digital_stabilization"),
        "source": payload.get("source", "user_supplied"),
    }


def colmap_image_reader_options(calibration: dict | None) -> list[str]:
    if calibration is None:
        return ["--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV"]
    return [
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", calibration["model"],
        "--ImageReader.camera_params", ",".join(f"{value:.12g}" for value in calibration["params"]),
    ]


def calibration_for_image_size(calibration: dict | None, width: int, height: int) -> dict | None:
    """Scale pixel-based intrinsics when extraction resizes the source video."""
    if calibration is None:
        return None
    result = dict(calibration)
    original_width, original_height = result.get("width"), result.get("height")
    if original_width is None or original_height is None:
        result["applied_width"] = width
        result["applied_height"] = height
        result["resolution_scaling"] = "not_available"
        return result
    scale_x, scale_y = width / original_width, height / original_height
    if abs(scale_x - scale_y) > 0.005:
        raise ValueError(
            "Extracted frame aspect ratio does not match the camera calibration resolution"
        )
    params = list(result["params"])
    model = result["model"]
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        params[0] *= scale_x
        params[1] *= scale_x
        params[2] *= scale_y
    else:
        params[0] *= scale_x
        params[1] *= scale_y
        params[2] *= scale_x
        params[3] *= scale_y
    result["params"] = params
    result["applied_width"] = width
    result["applied_height"] = height
    result["resolution_scale"] = float((scale_x + scale_y) / 2.0)
    return result
