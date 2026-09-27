"""DRIFTX command-line surface.

The existing ``da3`` command remains available and is not routed through the
DRIFTX benchmark-specific streaming wrapper.
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
        "benchmark", help="Run a memory-bounded, overlap-aligned video benchmark."
    )
    benchmark.add_argument("--video", required=True, help="Path to an input video.")
    benchmark.add_argument("--output", required=True, help="Output folder for reconstruction and benchmark report.")
    benchmark.add_argument(
        "--model", default="depth-anything/DA3-LARGE-1.1",
        help="DA3 Large 1.1 model ID or local checkpoint folder; Giant/Nested are rejected.",
    )
    benchmark.add_argument("--device", default="auto", help="Inference device: auto, cpu, or cuda (default: auto).")
    benchmark.add_argument("--profile", choices=("smoke", "balanced", "quality"), default="smoke",
                           help="Starting settings profile. Explicit numeric options override profile values.")
    benchmark.add_argument("--sample-fps", type=float, default=None,
                           help="Video sampling rate; profile defaults to 1 FPS (smoke) or 2 FPS.")
    benchmark.add_argument(
        "--max-frames", type=int, default=None,
        help="TOTAL sampled frames to process (not a GPU batch size); 0 processes all sampled frames. "
             "Smoke defaults to 16; other profiles default to all.",
    )
    benchmark.add_argument("--process-res", type=int, default=None,
                           help="DA3 preprocessing resolution; defaults to the selected profile.")
    benchmark.add_argument("--chunk-size", type=int, default=None,
                           help="Maximum frames per DA3 inference window; profile/VRAM-derived by default.")
    benchmark.add_argument("--chunk-overlap", type=int, default=None,
                           help="Shared frames used for adjacent-window geometric alignment.")
    benchmark.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"), default="auto",
                           help="Mixed precision mode. Auto selects BF16 when supported, otherwise FP16 on CUDA.")
    benchmark.add_argument("--auto-memory", action=argparse.BooleanOptionalAction, default=True,
                           help="Choose a conservative initial window from free VRAM (default: on); OOM retries remain enabled.")

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
            max_frames=args.max_frames,
            process_res=args.process_res,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            profile=args.profile,
            precision=args.precision,
            auto_memory=args.auto_memory,
        )
        report_path = Path(args.output).expanduser().resolve() / "run_report.json"
        print(f"DRIFTX benchmark status: {report['status']}")
        print(f"Report: {report_path}")
        if report["status"] == "completed":
            print(f"Frames processed: {report['frames_processed']} in {report['num_chunks']} chunks")
            print(f"Effective window/overlap: {report['chunk_size']}/{report['chunk_overlap']}")
            print(f"Precision: {report['final_precision']}; peak GPU memory: {report['peak_gpu_memory_mb']} MB")
            print(f"Runtime: {report['total_runtime_seconds']:.2f} s; OOM retries: {report['cuda_oom_retries']}")
            print(f"Artifacts: {report['artifacts']}")
        elif "error" in report:
            print(f"Error: {report['error']}")
        return 0 if report["status"] == "completed" else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
