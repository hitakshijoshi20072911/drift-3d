"""DRIFTX model metadata and capability detection.

Gaussian support is determined from checkpoint architecture metadata, never from
whether the optional gsplat package happens to be installed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelCapabilities:
    model_name: str
    model_path: str
    parameter_scale: str
    supports_depth: bool
    supports_pose: bool
    supports_gaussian: bool
    license: str
    expected_memory_profile: str
    recommended_device: str
    capability_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "da3-large": {
        "parameter_scale": "~0.35B",
        "expected_memory_profile": "local GPU baseline; tune with auto-memory",
        "recommended_device": "local GPU or CPU",
        "license": "See upstream Depth Anything 3 model license",
    },
    "da3-giant": {
        "parameter_scale": "~1.15B",
        "expected_memory_profile": "large GPU; local RTX 3050 is not expected to fit Gaussian mode",
        "recommended_device": "cloud GPU (T4/A10G or larger)",
        "license": "See official DA3-GIANT model license and usage restrictions",
    },
}


def _read_checkpoint_config(model: str | Path) -> dict[str, Any]:
    path = Path(model).expanduser()
    candidates = [path / "config.json", path / "config.yaml", path / "model_config.json"]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.suffix == ".json":
                return json.loads(candidate.read_text(encoding="utf-8"))
            import yaml
            return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def detect_model_capabilities(model: str | Path) -> ModelCapabilities:
    raw = str(model)
    lowered = raw.lower()
    config = _read_checkpoint_config(raw)
    architecture = json.dumps(config, sort_keys=True).lower()
    has_gs_architecture = "gs_head" in architecture or "gsdpt" in architecture or "gs_adapter" in architecture
    is_giant = "giant" in lowered or "nested" in lowered or "giant" in architecture
    supports_gaussian = bool(has_gs_architecture or is_giant) and not ("large" in lowered and not is_giant)
    if supports_gaussian:
        key = "da3-giant"
        source = "checkpoint architecture metadata" if has_gs_architecture else "official model family name"
    else:
        key = "da3-large"
        source = "checkpoint architecture metadata/model family name"
    metadata = MODEL_REGISTRY[key]
    return ModelCapabilities(
        model_name=raw,
        model_path=str(Path(raw).expanduser().resolve()) if Path(raw).expanduser().exists() else raw,
        supports_depth=True,
        supports_pose=True,
        supports_gaussian=supports_gaussian,
        capability_source=source,
        **metadata,
    )


def require_gaussian_capability(model: str | Path) -> ModelCapabilities:
    capabilities = detect_model_capabilities(model)
    if not capabilities.supports_gaussian:
        raise RuntimeError(
            "Selected checkpoint does not expose Gaussian prediction. "
            "Use an official Gaussian-capable DA3 checkpoint such as DA3-GIANT-1.1. "
            f"Detected model: {capabilities.model_name}."
        )
    return capabilities
