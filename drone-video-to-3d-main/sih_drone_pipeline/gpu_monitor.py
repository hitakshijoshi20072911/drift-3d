from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "sample_time_utc",
    "timestamp",
    "gpu_name",
    "utilization_gpu_percent",
    "memory_used_mb",
    "memory_total_mb",
    "temperature_c",
    "power_draw_w",
]
QUERY = "timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def monitor(output: Path, stop_file: Path, interval_s: float) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    if not shutil.which("nvidia-smi"):
        summary = {"available": False, "reason": "nvidia-smi not found"}
        output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        while not stop_file.exists():
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
                if len(values) == len(FIELDS) - 1:
                    row = {"sample_time_utc": datetime.now(timezone.utc).isoformat(), **dict(zip(FIELDS[1:], values))}
                    writer.writerow(row)
                    stream.flush()
                    samples.append(row)
            time.sleep(interval_s)
    numeric = FIELDS[3:]
    summary = {"available": bool(samples), "samples": len(samples), "interval_s": interval_s}
    for field in numeric:
        values = [float(row[field]) for row in samples if row[field] not in {"", "[N/A]", "N/A"}]
        if values:
            summary[f"{field}_mean"] = sum(values) / len(values)
            summary[f"{field}_max"] = max(values)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll NVIDIA GPU metrics until a stop file appears")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-file", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    raise SystemExit(0 if monitor(args.output, args.stop_file, args.interval).get("available") else 1)


if __name__ == "__main__":
    main()
