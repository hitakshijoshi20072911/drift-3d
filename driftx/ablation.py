"""Evidence-preserving ablation summary utilities."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

FIELDS = (
    "runtime_seconds", "frames_per_second", "peak_gpu_memory_mb", "valid_depth_percentage",
    "mean_confidence", "point_count", "coverage", "depth_metrics", "pose_metrics",
)


def summarize_reports(reports: Iterable[str | Path], output: str | Path) -> dict:
    rows = []
    for report_path in reports:
        path = Path(report_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "run": path.parent.name,
            "report": str(path),
            "status": data.get("status", "not measured"),
            "model": data.get("model_variant", "not measured"),
            "reconstruction_mode": data.get("reconstruction_mode", "not measured"),
            "runtime_seconds": data.get("total_runtime_seconds", "not measured"),
            "frames_per_second": data.get("frames_per_second", "not measured"),
            "peak_gpu_memory_mb": data.get("peak_gpu_memory_mb", "not measured"),
            "valid_depth_percentage": data.get("valid_depth_percentage", "not measured"),
            "mean_confidence": data.get("mean_confidence", "not measured"),
            "point_count": data.get("point_count", "not measured"),
            "coverage": "unavailable without reference geometry",
            "depth_metrics": "unavailable without GT depth",
            "pose_metrics": "unavailable without GT trajectory",
        })
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "completed" if rows else "not measured",
        "ground_truth_policy": "Metrics requiring ground truth remain unavailable; no values are fabricated.",
        "runs": rows,
    }
    (out / "ablation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        with (out / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return payload
