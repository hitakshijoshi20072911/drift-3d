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


def extract_video_frames(
    video: str | Path,
    output_dir: str | Path,
    sample_fps: float,
    max_frames: int | None = 16,
) -> tuple[VideoInfo, list[str]]:
    """Extract deterministic frames using the same interval rule as the vendor CLI."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero or None")

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

    frames_dir = Path(output_dir) / "input_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                frame_path = frames_dir / f"{len(frame_paths):06d}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
                frame_paths.append(str(frame_path))
            frame_index += 1
    finally:
        cap.release()

    if not frame_paths:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    if max_frames is not None and len(frame_paths) > max_frames:
        # Preserve coverage across the whole video rather than taking only the
        # first frames. The DA3 Large model sees one multi-view batch, so this
        # cap is the primary high-resolution VRAM safeguard.
        keep_indices = {
            round(i * (len(frame_paths) - 1) / (max_frames - 1))
            for i in range(max_frames)
        } if max_frames > 1 else {len(frame_paths) // 2}
        selected = []
        for index, frame_path in enumerate(frame_paths):
            if index in keep_indices:
                selected.append(frame_path)
            else:
                Path(frame_path).unlink(missing_ok=True)
        frame_paths = selected

    return (
        VideoInfo(
            duration_seconds=duration_seconds,
            source_fps=source_fps,
            source_frames=source_frames,
            sampled_fps=sampled_fps,
            frame_interval=frame_interval,
        ),
        frame_paths,
    )
