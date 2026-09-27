"""Generate documentation figures from completed DRIFTX benchmark runs.

Usage:
    python -m driftx.figures --results-root outputs/benchmarks --output output_figures

Only values present in run_report.json and exported mini-NPZ artifacts are used.
Unavailable values are shown as ``not measured`` rather than fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


COLORS = ["#1565C0", "#00897B", "#EF6C00", "#6A1B9A", "#C62828"]
FIG_NAMES = [
    "fig01_performance_dashboard",
    "fig02_runtime_breakdown",
    "fig03_frame_pipeline",
    "fig04_confidence_distribution",
    "fig05_temporal_quality",
    "fig06_camera_trajectory",
    "fig07_rgb_depth_confidence",
    "fig08_runtime_scaling",
]


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any, unit: str = "", digits: int = 2) -> str:
    number = _number(value)
    if math.isnan(number):
        return "not measured"
    return f"{number:.{digits}f}{unit}"


def _metric(report: dict[str, Any], current: str, legacy: str | None = None) -> Any:
    """Read a schema-v2 value, falling back to a schema-v1 report key."""
    value = report.get(current)
    if value is None and legacy is not None:
        value = report.get(legacy)
    return value


def _trajectory_centers(extrinsics: np.ndarray | None) -> np.ndarray | None:
    """Return camera centers for DA3 (N,3,4) or homogeneous (N,4,4) w2c poses."""
    if extrinsics is None:
        return None
    ext = np.asarray(extrinsics)
    if ext.ndim != 3:
        return None
    if ext.shape[1:] == (4, 4):
        if not np.isfinite(ext).all():
            return None
        return np.linalg.inv(ext)[:, :3, 3]
    if ext.shape[1:] == (3, 4):
        rotation = ext[:, :3, :3]
        translation = ext[:, :3, 3]
        if not np.isfinite(ext).all():
            return None
        # DA3 extrinsics are world-to-camera: C_world = -R.T @ t.
        return -np.einsum("nji,nj->ni", rotation, translation)
    return None


def _short_name(report: dict[str, Any], index: int) -> str:
    return Path(str(report.get("video", "video"))).stem or f"video_{index + 1}"


def _load_npz(report: dict[str, Any]) -> dict[str, np.ndarray]:
    candidate = report.get("artifacts", {}).get("npz")
    if candidate is None:
        candidate = Path(str(report.get("output", ""))) / "exports" / "mini_npz" / "results.npz"
    path = Path(candidate)
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except Exception:
        return {}


def _load_reports(root: Path, report_paths: list[str]) -> list[dict[str, Any]]:
    paths = [Path(p) for p in report_paths] if report_paths else sorted(root.rglob("run_report.json"))
    reports = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        report["_report_path"] = str(path.resolve())
        report["_npz"] = _load_npz(report)
        reports.append(report)
    if not reports:
        raise SystemExit(f"No run_report.json files found under {root}")
    return reports


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _labels(reports: list[dict[str, Any]]) -> list[str]:
    return [_short_name(r, i) for i, r in enumerate(reports)]


def fig01_dashboard(reports, output):
    labels = _labels(reports)
    fig, axes = plt.subplots(1, len(reports), figsize=(5.2 * len(reports), 6.2), squeeze=False)
    fig.suptitle("DRIFTX Phase 1 Baseline — End-to-End Performance Dashboard", fontsize=16, weight="bold")
    keys = [
        ("Duration", "video_duration_s", " s"),
        ("Source FPS", "source_fps", " fps"),
        ("Sampled FPS", "sample_fps", " fps"),
        ("Frames extracted", "frames_sampled", ""),
        ("Frames processed", "frames_processed", ""),
        ("Total runtime", "total_runtime_seconds", " s"),
        ("Inference runtime", "inference_seconds", " s"),
        ("Export runtime", "export_seconds", " s"),
        ("GLB size", "glb_size_mb", " MB"),
        ("PLY size", "ply_size_mb", " MB"),
        ("GPU peak", "gpu_peak_gb", " GB"),
    ]
    for i, (ax, report, label) in enumerate(zip(axes[0], reports, labels)):
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#F4F7FB", ec=COLORS[i % len(COLORS)], lw=2))
        ax.text(0.5, 0.94, label, ha="center", va="top", fontsize=14, weight="bold", transform=ax.transAxes)
        y = 0.84
        for title, key, unit in keys:
            ax.text(0.08, y, title, fontsize=10, color="#455A64", transform=ax.transAxes)
            legacy = {
                "video_duration_s": "video_duration_seconds", "sample_fps": "sampled_fps",
                "frames_sampled": "frames_extracted", "total_runtime_seconds": "total_wall_clock_seconds",
                "inference_seconds": "inference_time_seconds", "export_seconds": "export_time_seconds",
            }.get(key)
            value = _metric(report, key, legacy)
            if key == "gpu_peak_gb":
                value = report.get(key)
            ax.text(0.92, y, _fmt(value, unit), fontsize=10, ha="right", weight="bold", transform=ax.transAxes)
            y -= 0.072
    _save(fig, output, FIG_NAMES[0])


def _artifact_sizes(reports):
    for report in reports:
        for key, field in (("glb", "glb_size_mb"), ("ply", "ply_size_mb")):
            path = report.get("artifacts", {}).get(key)
            size = Path(path).stat().st_size / 1e6 if path and Path(path).is_file() else float("nan")
            report[field] = size
        gpu_mb = _number(report.get("peak_gpu_memory_mb"))
        if not math.isnan(gpu_mb):
            report["gpu_peak_gb"] = gpu_mb / 1024
        else:
            gpu = _number(report.get("gpu_memory_peak_bytes"))
            report["gpu_peak_gb"] = gpu / (1024 ** 3) if not math.isnan(gpu) else float("nan")


def fig02_runtime(reports, output):
    labels = _labels(reports)
    extraction = np.array([_number(_metric(r, "video_decode_seconds", "frame_extraction_time_seconds")) for r in reports])
    inference = np.array([_number(_metric(r, "inference_seconds", "inference_time_seconds")) for r in reports])
    export = np.array([_number(_metric(r, "export_seconds", "export_time_seconds")) for r in reports])
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(reports))
    for values, name, color in zip((extraction, inference, export), ("Frame extraction", "DA3 inference", "Export"), COLORS[:3]):
        safe = np.nan_to_num(values, nan=0.0)
        ax.bar(labels, safe / 60, bottom=bottom / 60, label=name, color=color)
        bottom += safe
    for x, report in enumerate(reports):
        total = _number(_metric(report, "total_runtime_seconds", "total_wall_clock_seconds"))
        if not math.isnan(total):
            ax.text(x, (bottom[x] / 60) + 0.03, f"total {_fmt(total, ' s')}", ha="center", fontsize=9)
    ax.set_title("DRIFTX Baseline Runtime Breakdown", weight="bold")
    ax.set_xlabel("Input video")
    ax.set_ylabel("Runtime (minutes)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output, FIG_NAMES[1])


def fig03_pipeline(reports, output):
    labels = _labels(reports)
    fields = [("source_frames", "Original frames"), ("frames_sampled", "Sampled frames"), ("frames_processed", "Processed frames")]
    fig, axes = plt.subplots(1, len(reports), figsize=(5 * len(reports), 5), squeeze=False)
    fig.suptitle("Frame Reduction and Processing Efficiency", fontsize=15, weight="bold")
    for ax, report, label in zip(axes[0], reports, labels):
        values = [_number(_metric(report, key, "frames_extracted" if key == "frames_sampled" else None)) for key, _ in fields]
        npz = report.get("_npz", {})
        depth = npz.get("depth")
        valid_frames = int(np.sum(np.any(np.isfinite(depth) & (depth > 0), axis=(1, 2)))) if depth is not None and depth.ndim == 3 else float("nan")
        values.append(valid_frames)
        names = [name for _, name in fields] + ["Valid-depth frames"]
        y = np.arange(len(names))[::-1]
        safe = [v if not math.isnan(v) else 0 for v in values]
        ax.barh(y, safe, color=COLORS[:4])
        for yi, value, original in zip(y, safe, values):
            text = _fmt(original, "", 0)
            pct = ""
            if values[0] and not math.isnan(values[0]) and not math.isnan(original):
                pct = f" ({100 * original / values[0]:.1f}%)"
            ax.text(max(value, 0) + max(safe or [1]) * 0.02, yi, text + pct, va="center", fontsize=8)
        ax.set_title(label, weight="bold")
        ax.set_xlabel("Frame count")
        ax.set_yticks(y, names)
        ax.grid(axis="x", alpha=0.25)
    _save(fig, output, FIG_NAMES[2])


def _confidence_samples(report):
    conf = report.get("_npz", {}).get("conf")
    if conf is None:
        return np.array([])
    values = np.asarray(conf, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) > 100_000:
        values = values[np.linspace(0, len(values) - 1, 100_000, dtype=np.int64)]
    return values


def fig04_confidence(reports, output):
    labels = _labels(reports)
    samples = [_confidence_samples(r) for r in reports]
    fig, ax = plt.subplots(figsize=(10, 6))
    usable = [s if len(s) else np.array([np.nan]) for s in samples]
    ax.violinplot(usable, showmeans=False, showmedians=True, showextrema=True)
    ax.boxplot(usable, widths=0.16, showfliers=False)
    for i, (label, report, sample) in enumerate(zip(labels, reports, samples), 1):
        mean = np.mean(sample) if len(sample) else _number(report.get("mean_confidence"))
        median = np.median(sample) if len(sample) else _number(report.get("median_confidence"))
        ax.text(i, ax.get_ylim()[1], f"mean {_fmt(mean)}\nmedian {_fmt(median)}", ha="center", va="top", fontsize=8)
    ax.set_title("DA3 Depth-Confidence Distribution", weight="bold")
    ax.set_xlabel("Input video")
    ax.set_ylabel("Confidence (vendor scale)")
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output, FIG_NAMES[3])


def _temporal(report):
    data = report.get("_npz", {})
    conf, depth = data.get("conf"), data.get("depth")
    if conf is None or depth is None or conf.ndim != 3 or depth.ndim != 3:
        return None
    frame_conf = np.nanmean(conf, axis=(1, 2))
    valid = np.mean(np.isfinite(depth) & (depth > 0), axis=(1, 2)) * 100
    frame_ids = data.get("frame_ids")
    source_fps = _number(report.get("source_fps"))
    if frame_ids is not None and len(frame_ids) == len(frame_conf) and source_fps > 0:
        time_s = (np.asarray(frame_ids, dtype=float) - float(frame_ids[0])) / source_fps
        return time_s, frame_conf, valid
    fps = _number(_metric(report, "sample_fps", "sampled_fps"))
    if math.isnan(fps) or fps <= 0:
        fps = 1.0
    return np.arange(len(frame_conf)) / fps, frame_conf, valid


def fig05_temporal(reports, output):
    labels = _labels(reports)
    fig, axes = plt.subplots(len(reports), 1, figsize=(11, max(3.2 * len(reports), 5)), squeeze=False)
    fig.suptitle("Temporal Depth Quality and Confidence", fontsize=15, weight="bold")
    for ax, report, label in zip(axes[:, 0], reports, labels):
        series = _temporal(report)
        if series is None:
            ax.text(0.5, 0.5, "not measured: exported confidence/depth arrays unavailable", ha="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        time_s, confidence, valid = series
        ax2 = ax.twinx()
        ax.plot(time_s, confidence, color=COLORS[0], label="Mean confidence")
        ax2.plot(time_s, valid, color=COLORS[1], label="Valid depth (%)")
        ax.set_ylabel("Mean confidence", color=COLORS[0])
        ax2.set_ylabel("Valid depth (%)", color=COLORS[1])
        ax.set_xlabel("Video time (s)")
        ax.set_title(label, loc="left", weight="bold")
        ax.grid(alpha=0.2)
    _save(fig, output, FIG_NAMES[4])


def fig06_trajectory(reports, output):
    labels = _labels(reports)
    fig = plt.figure(figsize=(6 * len(reports), 5.5))
    for i, (report, label) in enumerate(zip(reports, labels), 1):
        ax = fig.add_subplot(1, len(reports), i, projection="3d")
        ext = report.get("_npz", {}).get("extrinsics")
        poses = _trajectory_centers(ext)
        if poses is None or len(poses) == 0:
            ax.text2D(0.5, 0.5, "not measured: poses unavailable", transform=ax.transAxes, ha="center")
        else:
            colors = np.linspace(0, 1, len(poses))
            ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], color=COLORS[0], lw=1.5)
            ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c=colors, cmap="viridis", s=16)
            ax.scatter(poses[0, 0], poses[0, 1], poses[0, 2], color=COLORS[2], s=45, label="Start")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        ax.set_title(label, weight="bold")
        ax.legend(loc="best") if poses is not None and len(poses) else None
    fig.suptitle("DA3 Reconstructed Camera Trajectory", fontsize=15, weight="bold")
    _save(fig, output, FIG_NAMES[5])


def _read_rgb(path: Path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def fig07_qualitative(reports, output):
    labels = _labels(reports)
    fig, axes = plt.subplots(3, len(reports), figsize=(5 * len(reports), 10), squeeze=False)
    fig.suptitle("RGB → Depth → Confidence Qualitative Comparison", fontsize=15, weight="bold")
    for i, (report, label) in enumerate(zip(reports, labels)):
        npz = report.get("_npz", {})
        depth = npz.get("depth")
        conf = npz.get("conf")
        frames = sorted((Path(report.get("output", "")) / "input_images").glob("*.png"))
        index = min(len(frames) // 2, depth.shape[0] // 2 - 1) if frames and depth is not None and depth.ndim == 3 else 0
        if frames:
            axes[0, i].imshow(_read_rgb(frames[index]))
        else:
            axes[0, i].text(0.5, 0.5, "not measured", ha="center")
        if depth is not None and depth.ndim == 3:
            axes[1, i].imshow(depth[index], cmap="magma")
        else:
            axes[1, i].text(0.5, 0.5, "not measured", ha="center")
        if conf is not None and conf.ndim == 3:
            axes[2, i].imshow(conf[index], cmap="viridis")
        else:
            axes[2, i].text(0.5, 0.5, "not measured", ha="center")
        axes[0, i].set_title(label, weight="bold")
        for row, name in enumerate(("RGB", "Depth", "Confidence")):
            axes[row, i].set_ylabel(name)
            axes[row, i].axis("off")
    _save(fig, output, FIG_NAMES[6])


def fig08_scaling(reports, output):
    labels = _labels(reports)
    frames = np.array([_number(r.get("frames_processed")) for r in reports])
    duration = np.array([_number(_metric(r, "video_duration_s", "video_duration_seconds")) for r in reports])
    runtime = np.array([_number(_metric(r, "total_runtime_seconds", "total_wall_clock_seconds")) for r in reports])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, x, xlabel in ((axes[0], frames, "Processed frames"), (axes[1], duration, "Video duration (s)")):
        valid = np.isfinite(x) & np.isfinite(runtime)
        point_colors = np.asarray(COLORS[: len(reports)])[valid]
        ax.scatter(x[valid], runtime[valid] / 60, c=point_colors, s=70)
        for j, label in enumerate(labels):
            if valid[j]:
                ax.annotate(label, (x[j], runtime[j] / 60), xytext=(5, 5), textcoords="offset points")
        if valid.sum() >= 2:
            coeff = np.polyfit(x[valid], runtime[valid] / 60, 1)
            line_x = np.linspace(x[valid].min(), x[valid].max(), 100)
            ax.plot(line_x, np.polyval(coeff, line_x), "--", color="#455A64", label="Least-squares fit")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Total runtime (minutes)")
        ax.grid(alpha=0.25)
    axes[0].set_title("Runtime vs processed frames")
    axes[1].set_title("Runtime vs video duration")
    if np.isfinite(frames).sum() >= 2 and np.isfinite(runtime).sum() >= 2:
        axes[0].legend()
    fig.suptitle("Baseline Runtime Scaling — Diagnostic Only", fontsize=15, weight="bold")
    _save(fig, output, FIG_NAMES[7])


def _summary(reports):
    rows = []
    for report in reports:
        row = {"video": _short_name(report, len(rows)), "report_path": report.get("_report_path", "")}
        row.update({
            "status": report.get("status", "not measured"),
            "model_variant": report.get("model_variant", "not measured"),
            "device": report.get("device", "not measured"),
            "video_duration_seconds": _metric(report, "video_duration_s", "video_duration_seconds"),
            "source_fps": report.get("source_fps", "not measured"),
            "source_frames": report.get("source_frames", "not measured"),
            "sampled_fps": _metric(report, "sample_fps", "sampled_fps"),
            "frames_extracted": _metric(report, "frames_sampled", "frames_extracted"),
            "frames_processed": report.get("frames_processed", "not measured"),
            "frame_extraction_time_seconds": _metric(report, "video_decode_seconds", "frame_extraction_time_seconds"),
            "inference_time_seconds": _metric(report, "inference_seconds", "inference_time_seconds"),
            "export_time_seconds": _metric(report, "export_seconds", "export_time_seconds"),
            "total_wall_clock_seconds": _metric(report, "total_runtime_seconds", "total_wall_clock_seconds"),
            "frames_per_second": report.get("frames_per_second", "not measured"),
            "chunk_size": report.get("chunk_size", "not measured"),
            "chunk_overlap": report.get("chunk_overlap", "not measured"),
            "num_chunks": report.get("num_chunks", "not measured"),
            "precision": report.get("final_precision", "not measured"),
            "cuda_oom_retries": report.get("cuda_oom_retries", "not measured"),
            "valid_depth_pixels": report.get("valid_depth_pixels", "not measured"),
            "valid_depth_percentage": report.get("valid_depth_percentage", "not measured"),
            "mean_confidence": report.get("mean_confidence", "not measured"),
            "median_confidence": report.get("median_confidence", "not measured"),
            "gpu_memory_peak_bytes": report.get("gpu_memory_peak_bytes", "not measured"),
            "peak_gpu_memory_mb": report.get("peak_gpu_memory_mb", "not measured"),
        })
        row["glb_size_mb"] = report.get("glb_size_mb", "not measured")
        row["ply_size_mb"] = report.get("ply_size_mb", "not measured")
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate eight DRIFTX benchmark figures from run reports.")
    parser.add_argument("--results-root", default="outputs/benchmarks", help="Folder containing benchmark run directories.")
    parser.add_argument("--report", action="append", default=[], help="Specific run_report.json path; repeat for multiple reports.")
    parser.add_argument("--output", default="output_figures", help="Figure and summary output folder.")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = _load_reports(Path(args.results_root).expanduser().resolve(), args.report)
    _artifact_sizes(reports)
    fig01_dashboard(reports, output)
    fig02_runtime(reports, output)
    fig03_pipeline(reports, output)
    fig04_confidence(reports, output)
    fig05_temporal(reports, output)
    fig06_trajectory(reports, output)
    fig07_qualitative(reports, output)
    fig08_scaling(reports, output)
    rows = _summary(reports)
    (output / "baseline_summary.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    with (output / "baseline_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(FIG_NAMES)} figures in {output}")
    print(f"Summary: {output / 'baseline_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
