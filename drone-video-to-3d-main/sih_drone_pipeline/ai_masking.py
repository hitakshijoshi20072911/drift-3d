from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DYNAMIC_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
}


def mask_dynamic_objects(
    images_dir: str | Path, masks_dir: str | Path, model_name: str = "yolov8n-seg.pt",
    precheck_sample_size: int = 20,
) -> dict:
    """Create COLMAP feature masks using a lightweight segmentation model.

    White pixels remain eligible for features; detected people, vehicles and
    animals are blacked out so they cannot distort the static-scene solve.

    Most drone surveys (rural, agricultural, infrastructure) contain no
    people, vehicles or animals at all, so paying the full per-frame
    segmentation cost on every run is pure overhead for them. A small,
    evenly-spaced sample is checked first; the full per-frame pass only
    runs when that sample actually finds something to mask.
    """
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("AI masking requires `pip install ultralytics`") from error

    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        return {
            "model": model_name, "frames": 0, "masked_frames": 0, "dynamic_instances": 0,
            "skipped_after_precheck": True, "precheck_reason": "no frames to check",
        }
    model = YOLO(model_name)

    sample_stride = max(1, len(images) // precheck_sample_size)
    sample = images[::sample_stride]
    found_dynamic = False
    for result in model.predict(
        [str(path) for path in sample], imgsz=640, device=0, verbose=False, stream=True
    ):
        classes = result.boxes.cls.int().tolist() if result.boxes is not None else []
        if any(result.names[int(class_id)] in DYNAMIC_CLASSES for class_id in classes):
            found_dynamic = True
            break
    if not found_dynamic:
        return {
            "model": model_name, "frames": len(images), "masked_frames": 0, "dynamic_instances": 0,
            "skipped_after_precheck": True,
            "precheck_reason": f"no dynamic objects found in a {len(sample)}-frame sample",
        }

    masks_dir.mkdir(parents=True, exist_ok=True)
    masked_frames, instances = 0, 0
    for image_path, result in zip(
        images,
        model.predict([str(path) for path in images], imgsz=640, device=0, verbose=False, stream=True),
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        keep = np.full((height, width), 255, dtype=np.uint8)
        if result.masks is not None and result.boxes is not None:
            frame_instances = 0
            for class_id, raw_mask in zip(result.boxes.cls.int().tolist(), result.masks.data.cpu().numpy()):
                if result.names[int(class_id)] not in DYNAMIC_CLASSES:
                    continue
                resized = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_LINEAR) > 0.35
                keep[resized] = 0
                frame_instances += 1
            if frame_instances:
                keep = cv2.erode(keep, np.ones((9, 9), np.uint8), iterations=1)
                masked_frames += 1
                instances += frame_instances
        cv2.imwrite(str(masks_dir / f"{image_path.name}.png"), keep)
    return {
        "model": model_name, "frames": len(images), "masked_frames": masked_frames,
        "dynamic_instances": instances, "skipped_after_precheck": False,
    }
