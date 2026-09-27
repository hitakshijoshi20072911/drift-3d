"""Video probing and deterministic frame extraction for DRIFTX benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    duration_seconds: float
    source_fps: float
    source_frames: int
    sampled_fps: float
    frame_interval: int
    source_frame_ids: tuple[int, ...] = ()


def _evenly_spaced_indices(count: int, limit: int) -> list[int]:
    if limit <= 0 or count <= limit:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)})


def extract_video_frames(
    video: str | Path,
    output_dir: str | Path,
    sample_fps: float,
    max_frames: int | None = 16,
) -> tuple[VideoInfo, list[str]]:
    """Extract a deterministic set of sampled frames on CPU.

    ``max_frames`` is a total output-frame budget (``None`` means all sampled
    frames); it never controls model/GPU batch size. When capped, coverage is
    preserved by selecting evenly across the complete sampled sequence.
    ``VideoInfo.source_frame_ids`` maps every returned path back to its source
    video frame number.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be zero/None for all frames or a positive count")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video benchmarks require the declared opencv-python dependency") from exc

    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Video has no usable source FPS: {video_path}")

    frame_interval = max(1, int(source_fps / sample_fps))
    sampled_fps = source_fps / frame_interval
    duration_seconds = source_frames / source_fps if source_frames else 0.0
    sampled_source_ids = list(range(0, source_frames, frame_interval))
    cap_indices = _evenly_spaced_indices(
        len(sampled_source_ids), max_frames if max_frames not in (None, 0) else len(sampled_source_ids)
    )
    selected_source_ids = tuple(sampled_source_ids[i] for i in cap_indices)
    selected_set = set(selected_source_ids)

    frames_dir = Path(output_dir) / "input_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Avoid stale frames from an earlier run changing inference order.
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink(missing_ok=True)

    frame_paths: list[str] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index in selected_set:
                frame_path = frames_dir / f"frame_{frame_index:09d}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
                frame_paths.append(str(frame_path))
            frame_index += 1
    finally:
        cap.release()

    if not frame_paths:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    if len(frame_paths) != len(selected_source_ids):
        raise RuntimeError(
            f"Decoded {len(frame_paths)} of {len(selected_source_ids)} selected frames; "
            "the video may be truncated or have inaccurate frame-count metadata."
        )

    return (
        VideoInfo(
            duration_seconds=duration_seconds,
            source_fps=source_fps,
            source_frames=source_frames,
            sampled_fps=sampled_fps,
            frame_interval=frame_interval,
            source_frame_ids=selected_source_ids,
        ),
        frame_paths,
    )
