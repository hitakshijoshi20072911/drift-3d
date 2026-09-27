"""DRIFTX command-line surface.

The existing ``da3`` command remains available for compatibility. This small
product-facing entry point is intentionally dependency-light so ``driftx
--help`` works before model or GPU dependencies are loaded.
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the DRIFTX product CLI parser."""
    parser = argparse.ArgumentParser(
        prog="driftx",
        description="DRIFTX geometry engine for single-pass UAV video reconstruction.",
    )
    subparsers = parser.add_subparsers(dest="command")
    benchmark = subparsers.add_parser(
        "benchmark", help="Run a reproducible video-to-3D baseline benchmark."
    )
    benchmark.add_argument("--video", required=True, help="Path to an input video.")
    benchmark.add_argument("--output", required=True, help="Directory for frames, exports, and run_report.json.")
    benchmark.add_argument(
        "--model",
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Pretrained model variant or local model directory.",
    )
    benchmark.add_argument(
        "--device", default="auto", help="Inference device: auto, cpu, or cuda (default: auto)."
    )
    benchmark.add_argument(
        "--sample-fps", type=float, default=1.0, help="Video sampling rate (default: 1 FPS)."
    )
    benchmark.add_argument(
        "--process-res", type=int, default=504, help="Vendor preprocessing resolution (default: 504)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the DRIFTX product CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        from driftx.benchmark import run_benchmark

        report = run_benchmark(
            video=args.video,
            output=args.output,
            model=args.model,
            device=args.device,
            sample_fps=args.sample_fps,
            process_res=args.process_res,
        )
        print(f"DRIFTX benchmark status: {report['status']}")
        report_path = Path(args.output).expanduser().resolve() / "run_report.json"
        print(f"Report: {report_path}")
        if report["status"] == "completed":
            print(f"Frames processed: {report['frames_processed']}")
            print(f"Valid depth pixels: {report['valid_depth_pixels']}")
            print(f"Mean confidence: {report['mean_confidence']}")
            print(f"Inference seconds: {report['inference_time_seconds']:.3f}")
            print(f"Artifacts: {report['artifacts']}")
        elif "error" in report:
            print(f"Error: {report['error']}")
        return 0 if report["status"] == "completed" else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
