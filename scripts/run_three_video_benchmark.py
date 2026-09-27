"""Run the documented three-video DRIFTX reconstruction and generate figures.

This script deliberately runs videos sequentially so GPU memory is released
between jobs and every report remains independently inspectable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--profile", default="smoke", choices=("smoke", "balanced", "quality"))
    parser.add_argument("--reconstruction-mode", default="baseline", choices=("baseline", "gaussian", "both"))
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks/three_video"))
    parser.add_argument("--figures-dir", type=Path, default=Path("output_figures/three_video"))
    parser.add_argument("--videos", nargs=3, default=("test3.mp4", "test6.mp4", "test7.mp4"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    for video_name in args.videos:
        video = args.input_dir / video_name
        run_dir = args.output_dir / video.stem
        if not video.is_file():
            parser.error(f"Input video does not exist: {video}")
        command = [
            sys.executable, "-m", "driftx", "benchmark",
            "--video", str(video), "--output", str(run_dir),
            "--model", args.model, "--device", args.device, "--profile", args.profile,
            "--reconstruction-mode", args.reconstruction_mode,
        ]
        print("\n$", " ".join(command), flush=True)
        completed = subprocess.run(command)
        report_path = run_dir / "run_report.json"
        if not report_path.is_file():
            raise SystemExit(f"Benchmark did not write {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append(report_path)
        if completed.returncode != 0 or report.get("status") != "completed":
            raise SystemExit(
                f"Benchmark failed for {video}: {report.get('error', 'see run_report.json')}"
            )

    figure_command = [
        sys.executable, "-m", "driftx.figures",
        *sum((["--report", str(path)] for path in reports), []),
        "--output", str(args.figures_dir),
    ]
    print("\n$", " ".join(figure_command), flush=True)
    subprocess.run(figure_command, check=True)
    print(f"\nCompleted {len(reports)} sequential benchmarks.")
    print(f"Figures and summaries: {args.figures_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
