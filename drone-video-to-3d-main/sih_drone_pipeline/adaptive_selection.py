"""Motion-triggered, visually verified keyframes; no fixed-count bin allocation.

Visual support is a screening proxy, not a measurement of reconstructed surface
coverage. All thresholds are configurable engineering defaults, not guarantees.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    candidate_fps: float = 3.0
    preview_width: int = 640
    max_features: int = 1200
    minimum_inliers: int = 24
    minimum_inlier_ratio: float = 0.45
    minimum_grid_fraction: float = 0.25
    motion_fraction: float = 0.06
    maximum_time_gap_s: float = 4.0
    maximum_path_gap_m: float = 16.25
    ratio_test: float = 0.75
    screening_timeout_s: float = 180.0

    def __post_init__(self):
        for key in ("candidate_fps", "screening_timeout_s", "maximum_time_gap_s", "maximum_path_gap_m"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.preview_width < 64 or self.max_features < 32 or self.minimum_inliers < 8:
            raise ValueError("Invalid feature extraction dimensions/counts")
        for key in ("minimum_inlier_ratio", "minimum_grid_fraction", "motion_fraction", "ratio_test"):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in (0, 1]")


def _features(gray, orb):
    points, descriptors = orb.detectAndCompute(gray, None)
    return np.asarray([p.pt for p in points], dtype=np.float32).reshape(-1, 2), descriptors, gray.shape


def verify_pair(left, right, config: SelectionConfig) -> dict:
    """Mutual ratio matches + RANSAC + image-wide support, not LK status bits."""
    a, da, shape_a = left
    b, db, shape_b = right
    result = {"verified": False, "matches": 0, "inliers": 0,
              "inlier_ratio": 0.0, "grid_fraction": 0.0,
              "motion_fraction": 0.0, "model": None}
    if da is None or db is None or min(len(da), len(db)) < 8:
        return result
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if da.dtype == np.uint8 else cv2.NORM_L2)
    def ratio_pairs(x, y):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(x, y, k=2)
                if len(pair) == 2 for m, n in [pair]
                if m.distance < config.ratio_test * n.distance}
    forward, backward = ratio_pairs(da, db), ratio_pairs(db, da)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    result["matches"] = len(pairs)
    if len(pairs) < 8:
        return result
    pa = np.float32([a[i] for i, _ in pairs])
    pb = np.float32([b[j] for _, j in pairs])
    models = []
    for kind in ("homography", "fundamental"):
        try:
            if kind == "homography":
                model, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 2.5, maxIters=1000)
            else:
                model, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.99, 1000)
            if model is not None and mask is not None and mask.size == len(pairs):
                models.append((kind, mask.ravel().astype(bool)))
        except cv2.error:
            continue
    if not models:
        return result
    # Homography supports near-planar views; F supports depth variation.
    kind, mask = max(models, key=lambda item: int(item[1].sum()))
    count = int(mask.sum())
    if not count:
        return result
    def grid_fraction(points, shape):
        h, w = shape
        cells = np.clip((points / [w, h] * 4).astype(int), 0, 3)
        return len(np.unique(cells, axis=0)) / 16.0
    grid = min(grid_fraction(pa[mask], shape_a), grid_fraction(pb[mask], shape_b))
    ratio = count / len(pairs)
    motion = float(np.median(np.linalg.norm(pb[mask] - pa[mask], axis=1))) / shape_a[1]
    result.update(inliers=count, inlier_ratio=ratio, grid_fraction=grid,
                  motion_fraction=motion, model=kind,
                  verified=bool(count >= config.minimum_inliers and
                                ratio >= config.minimum_inlier_ratio and
                                grid >= config.minimum_grid_fraction))
    return result


def choose_indices(candidates, pair, config: SelectionConfig):
    """Retain last verified bridge before overlap is lost; record real breaks."""
    if not candidates:
        return [], []
    selected, reasons = [0], ["start"]
    last_good = 0
    for index in range(1, len(candidates)):
        anchor = selected[-1]
        edge = pair(anchor, index)
        if not edge["verified"]:
            if last_good > anchor:
                selected.append(last_good)
                reasons.append("bridge_before_overlap_loss")
                anchor = last_good
                edge = pair(anchor, index)
            if not edge["verified"]:
                # No invention of connectivity across cuts, textureless regions,
                # or failed candidate matches. Preserve the new region explicitly.
                selected.append(index)
                reasons.append("unresolved_visual_break")
                last_good = index
                continue
        last_good = index
        dt = candidates[index]["time_s"] - candidates[anchor]["time_s"]
        dp = candidates[index]["path_m"] - candidates[anchor]["path_m"]
        reason = None
        if edge["motion_fraction"] >= config.motion_fraction:
            reason = "visual_motion"
        elif dp >= config.maximum_path_gap_m:
            reason = "trajectory_gap"
        elif dt >= config.maximum_time_gap_s:
            reason = "temporal_guard"
        if reason:
            # Choose a sharp nearby view without moving behind the last anchor.
            # Candidate diversity is triggered by motion; sharpness only ranks
            # the local, geometrically usable alternatives.
            local = [i for i in range(max(anchor+1, index-2), index+1)
                     if pair(anchor, i)["verified"]]
            chosen = max(local, key=lambda i: candidates[i].get("sharpness", 0.0) *
                         (0.85 + 0.15 * (i-anchor)/(index-anchor)))
            selected.append(chosen)
            reasons.append(reason)
    if selected[-1] != len(candidates) - 1:
        selected.append(len(candidates) - 1)
        reasons.append("end")
    return selected, reasons


def _ffmpeg_has_cuda() -> bool:
    """True when FFmpeg lists the CUDA hwaccel and NVDEC isn't disabled via SIH_DISABLE_NVDEC."""
    if os.environ.get("SIH_DISABLE_NVDEC", "").strip() not in ("", "0"):
        return False
    try:
        listing = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True,
                                 text=True, errors="replace", timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return "cuda" in listing.stdout.split()


def extract_adaptive_keyframes(video_path, output_dir, *, manifest_path=None,
                               telemetry_samples=None, target_frames=None,
                               max_width=1800, config=None,
                               screening_timeout_s=None, target_candidate_count=700):
    """One decode pass writes the candidates; the selected ones are copied, not re-decoded.

    target_frames is an estimated affordable count, never permission to silently
    delete necessary bridges. A budget conflict is written into the report.
    FFmpeg select preserves frame identity; showinfo supplies observed PTS.
    """
    from .extract_keyframes import FrameRecord, _telemetry_path, _clear_cached_images
    started = time.monotonic()
    video_path, output_dir = Path(video_path), Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Adaptive selection requires FFmpeg on PATH")
    cap = cv2.VideoCapture(str(video_path))
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if total < 10 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video must have at least 10 frames and a valid frame rate")
    if config is None:
        # Pairwise visual verification cost is driven by candidate *count*,
        # not video duration -- a fixed candidate_fps means a 10-minute
        # flight screens ~2.5x more candidates (and takes proportionally
        # longer) than the 4-minute flight this default was tuned on.
        # Holding the candidate count roughly constant instead keeps
        # screening cost near-flat as videos get longer, at the cost of
        # coarser temporal sampling (still caught by the adaptive verifier
        # inserting extra frames wherever a link turns out weak).
        duration_s = total / fps
        candidate_fps = max(0.5, min(3.0, target_candidate_count / duration_s))
        config = SelectionConfig(candidate_fps=candidate_fps)
    if screening_timeout_s is not None:
        from dataclasses import replace
        config = replace(config, screening_timeout_s=screening_timeout_s)
    step = max(1, int(math.floor(fps / config.candidate_fps)))
    source_indices = sorted(set(range(0, total, step)) | {total - 1})
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path) if manifest_path else output_dir.parent / "frames.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    def run_decode(filter_text, destination, hardware=False):
        remaining = config.screening_timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Frame screening exhausted its time allocation")
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "info",
            *(["-hwaccel", "cuda"] if hardware else []),
            "-i", str(video_path), "-vf", filter_text, "-fps_mode", "passthrough",
            "-q:v", "2", str(destination)], capture_output=True, text=True,
            errors="replace", timeout=remaining)
        if result.returncode:
            raise RuntimeError(f"Frame decode failed: {result.stderr[-2000:]}")
        return [float(value) for value in re.findall(r"\bpts_time:([\d.eE+\-]+)", result.stderr)]
    with tempfile.TemporaryDirectory(prefix="adaptive_candidates_", dir=manifest.parent) as temp:
        temp = Path(temp)
        # One decode pass only: candidates are written at output resolution and
        # screened from in-memory previews, so the selected frames are copied
        # rather than decoded a second time (a full-video decode on 2 CPUs,
        # especially H.265, is the dominant cost of this step).
        # The full-video decode runs on the GPU's NVDEC when FFmpeg has CUDA
        # (103 s -> 14.5 s for the 4-minute reference video on a T4). Its output
        # is only kept if it yields exactly the expected frames; otherwise the
        # CPU decode runs as before.
        phase_started = time.monotonic()
        decode_filter = (f"select='not(mod(n,{step}))+eq(n,{total - 1})',"
                         f"scale=w=min({max_width}\\,iw):h=-2,showinfo")
        decoder, decoder_fallback = "cpu", None
        if _ffmpeg_has_cuda():
            try:
                times = run_decode(decode_filter, temp / "candidate_%07d.jpg", hardware=True)
                produced = len(list(temp.glob("candidate_*.jpg")))
                if produced != len(source_indices) or len(times) != produced:
                    raise RuntimeError(f"NVDEC produced {produced} frames / {len(times)} timestamps, "
                                       f"expected {len(source_indices)}")
                decoder = "nvdec"
            except (RuntimeError, subprocess.SubprocessError, OSError) as error:
                decoder_fallback = str(error)[-300:]
                for partial in temp.glob("candidate_*.jpg"):
                    partial.unlink()
        if decoder == "cpu":
            times = run_decode(decode_filter, temp / "candidate_%07d.jpg")
        phase_seconds = {"decode": time.monotonic() - phase_started}
        phase_started = time.monotonic()
        paths = sorted(temp.glob("candidate_*.jpg"))
        if len(paths) != len(source_indices) or len(times) != len(paths):
            raise RuntimeError("Decoder frame count/PTS mismatch; refusing guessed timestamps")
        times = np.asarray(times) - times[0]
        if np.any(np.diff(times) <= 0):
            raise ValueError("Decoded timestamps are not strictly increasing")
        pt, pd = _telemetry_path(telemetry_samples or [])
        if len(pt) < 2 or pt[0] > times[0] + 1 or pt[-1] < times[-1] - 1:
            raise ValueError("Telemetry must cover the video timeline; check synchronization")
        distances = np.interp(times, pt, pd)
        orb = cv2.ORB_create(nfeatures=config.max_features)
        candidates, features, previews = [], [], []
        for index, path in enumerate(paths):
            if time.monotonic() - started > config.screening_timeout_s:
                raise TimeoutError("Candidate analysis exceeded screening deadline")
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"Unreadable candidate: {path}")
            if gray.shape[1] > config.preview_width:
                preview_height = int(round(gray.shape[0] * config.preview_width / gray.shape[1]))
                gray = cv2.resize(gray, (config.preview_width, preview_height), interpolation=cv2.INTER_AREA)
            previews.append(gray)
            features.append(_features(gray, orb))
            candidates.append({"source_frame": source_indices[index], "time_s": float(times[index]),
                               "path_m": float(distances[index]),
                               "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())})
        cache = {}
        fallback_features = {}
        fallback_pairs = 0
        maximum_fallback_pairs = max(16, min(512, int(total/fps/600*128)))
        sift = cv2.SIFT_create(nfeatures=config.max_features)
        def pair(a, b):
            nonlocal fallback_pairs
            if time.monotonic() - started > config.screening_timeout_s:
                raise TimeoutError("Visual matching exceeded screening deadline")
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = verify_pair(features[key[0]], features[key[1]], config)
                cache[key]['matcher'] = 'ORB_mutual_ratio_RANSAC'
                # Bounded robust fallback on difficult views; preserve failed
                # evidence if the stronger descriptor still cannot connect them.
                if not cache[key]['verified'] and fallback_pairs < maximum_fallback_pairs:
                    for i in key:
                        if i not in fallback_features:
                            fallback_features[i] = _features(previews[i], sift)
                    fallback = verify_pair(fallback_features[key[0]],fallback_features[key[1]],config)
                    fallback_pairs += 1
                    if fallback['verified']:
                        fallback['matcher'] = 'SIFT_mutual_ratio_RANSAC'
                        cache[key] = fallback
            return cache[key]
        selected, reasons = choose_indices(candidates, pair, config)
        # A mapper needs a minimum set, including low-motion clips. Such clips
        # still have to pass the separate parallax/geometry feasibility gates.
        if len(selected) < min(12, len(candidates)):
            selected = sorted(set(selected) | set(np.linspace(0, len(candidates)-1, min(12, len(candidates))).astype(int)))
            reasons = ["minimum_mapper_context" for _ in selected]
        edges = []
        for right in range(1, len(selected)):
            a, b = selected[right-1], selected[right]
            edges.append({"left_index": right-1, "right_index": right,
                          "time_gap_s": candidates[b]["time_s"]-candidates[a]["time_s"],
                          "path_gap_m": candidates[b]["path_m"]-candidates[a]["path_m"],
                          **pair(a, b)})
        # Limited nonconsecutive connections make the saved graph useful for
        # diagnosing weak boundaries without quadratic all-pairs matching.
        extra_edges = []
        for right in range(2, len(selected)):
            extra_edges.append({"left_index": right-2, "right_index": right,
                                **pair(selected[right-2], selected[right])})
        # GPS retrieves likely revisits; it never declares a visual connection.
        from scipy.spatial import cKDTree
        from .telemetry import interpolate
        poses = [interpolate(telemetry_samples, candidates[i]["time_s"]) for i in selected]
        origin = poses[0]
        xy = np.asarray([[(p.longitude-origin.longitude)*111320*math.cos(math.radians(origin.latitude)),
                          (p.latitude-origin.latitude)*111320] for p in poses])
        tree = cKDTree(xy)
        seen_pairs = {(e["left_index"],e["right_index"]) for e in edges+extra_edges}
        for right, point in enumerate(xy):
            _, nearby = tree.query(point, k=min(8,len(selected)), distance_upper_bound=40.0)
            checked = 0
            for left in np.atleast_1d(nearby):
                left = int(left)
                if left >= right-2 or (left,right) in seen_pairs:
                    continue
                extra_edges.append({"left_index":left, "right_index":right,
                                    "retrieval":"gps_revisit", **pair(selected[left],selected[right])})
                seen_pairs.add((left,right))
                checked += 1
                if checked >= 3:
                    break
        supported = set(selected)
        for index in range(len(candidates)):
            rank = int(np.searchsorted(selected, index))
            neighbors = selected[max(0, rank-1):min(len(selected), rank+1)]
            if index not in supported and any(pair(index, other)["verified"] for other in neighbors):
                supported.add(index)
        parent = list(range(len(selected)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for edge in edges + extra_edges:
            if edge["verified"]:
                parent[find(edge["right_index"])] = find(edge["left_index"])
        component_count = len({find(i) for i in range(len(selected))})
        phase_seconds["screening"] = time.monotonic() - phase_started
        phase_started = time.monotonic()
        # Selected frames are the already-decoded full-resolution candidates, so
        # their timestamps are the candidate PTS by construction.
        final_paths = [paths[i] for i in selected]
        _clear_cached_images(output_dir)
        records = []
        for output_index, (index, path) in enumerate(zip(selected, final_paths)):
            candidate = candidates[index]
            edge = edges[output_index-1] if output_index else None
            name = f"frame_{output_index:04d}.jpg"
            shutil.copy2(path, output_dir / name)
            records.append(FrameRecord(name, candidate["source_frame"], candidate["time_s"],
                candidate["sharpness"], edge["motion_fraction"] if edge else 0.0,
                edge["path_gap_m"] if edge else 0.0,
                edge["motion_fraction"]*160 if edge else 0.0,
                edge["inlier_ratio"] if edge and edge["verified"] else (1.0 if not edge else 0.0),
                edge["grid_fraction"] if edge else 1.0))
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=asdict(records[0]))
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
        phase_seconds["copy"] = time.monotonic() - phase_started
        report = {
            "selection_algorithm_version": "adaptive_visual_graph_v1",
            "keyframe_selection_mode": "adaptive", "config": asdict(config),
            "video_duration_s": total/fps, "observed_last_time_s": float(times[-1]),
            "source_fps": fps, "source_frames": total, "candidate_frames": len(candidates),
            "selected_frames": len(records), "affordable_frame_estimate": target_frames,
            "budget_conflict": target_frames is not None and len(records) > target_frames,
            "selection_reasons": reasons, "adjacent_edges": edges, "context_edges": extra_edges,
            "verified_graph_components": component_count,
            "unverified_adjacent_links": sum(not e["verified"] for e in edges),
            "candidate_visual_support_fraction": len(supported)/len(candidates),
            "unsupported_candidate_times_s": [c["time_s"] for i,c in enumerate(candidates) if i not in supported],
            "visible_surface_coverage_fraction": None,
            "coverage_disclosure": "Candidate matching is a visual-support proxy, not surface completeness.",
            "timing_seconds": {"screening_and_extraction": time.monotonic()-started,
                               **phase_seconds},
            "decoder": decoder,
            "decoder_fallback_reason": decoder_fallback,
            "timestamp_method": "FFmpeg select source indices and observed showinfo PTS",
            "sift_fallback_pairs": fallback_pairs,
            "sift_fallback_pair_limit": maximum_fallback_pairs,
        }
        manifest.with_suffix(".selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return records
