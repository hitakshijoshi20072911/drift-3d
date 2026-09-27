# DRIFTX Streaming Benchmark Design

## Scope

`python -m driftx benchmark` is a DRIFTX orchestration layer around the public Depth Anything 3 (DA3) inference API. It processes a deterministic sequence of sampled video frames in bounded overlapping windows. The existing `da3` command and DA3 streaming implementation remain separate and are not redirected through this runner.

The design is intended to reduce activation memory by limiting the number of views resident in a single model forward pass. It does **not** claim that every GPU/model/resolution combination is OOM-proof; it retries smaller windows and reports a hard failure if the configured safe floor still does not fit.

## Pipeline

```text
Video
  ↓ CPU decode + deterministic FPS sampling + total-frame budget
Sampled frame paths + original source frame IDs
  ↓ bounded overlapping windows
DA3 public inference API (one window on GPU at a time)
  ↓ confidence-weighted overlap correspondences
Robust Sim(3) alignment into one global world frame
  ↓ CPU NumPy staging (temporary memory-mapped arrays)
Ordered global result + frame/camera identity metadata
  ↓ bounded point-cloud sampling and export
scene.glb · scene.ply · results.npz · depth_vis/ · metrics.json · run_report.json
```

## Sampling and chunk semantics

- `--sample-fps` chooses the deterministic source-frame interval on the CPU.
- `--max-frames` is a **total cap on selected sampled frames**. It is never a GPU batch size. `--max-frames 64` selects at most 64 frames spread across the full video; `--max-frames 0` selects all sampled frames.
- The extractor stores selected frames on disk and records each original source frame number and timestamp. Frames are not silently dropped after extraction.
- `--chunk-size` bounds the number of views passed to one DA3 inference call.
- For `N` frames, chunk size `C`, and overlap `O`, windows advance by `C-O` frames. For example, `C=16, O=4` yields `[0,16)`, `[12,28)`, `[24,40)`, … . Each frame is exported once; overlapping predictions are alignment evidence, not duplicate final frames.

## Global geometry and overlap alignment

Each window initially has its own DA3 coordinate frame. DRIFTX therefore **does not concatenate independent chunk reconstructions directly**.

For adjacent windows, DRIFTX back-projects a sparse regular grid of valid depth pixels from corresponding overlapping images using each prediction's depth, confidence, intrinsics, and world-to-camera extrinsics. It estimates a confidence-weighted Sim(3) transform from the new window into the already-global previous window, trims high-residual correspondences, validates the result, and applies the transform to the new chunk's camera poses and depth scale. The first chunk defines the global gauge. The aligned chunk is then merged in source-frame order, with the overlap emitted only from the preceding chunk.

If overlap data are insufficient, degenerate, or inconsistent, the run stops with an explanatory error instead of producing a silently incoherent scene. Increase overlap, reduce sampling FPS, or inspect the footage/model if alignment is rejected. Loop closure/global pose-graph optimization is **not** part of this baseline phase.

## Memory and OOM behavior

- Video decoding and image-path preparation stay CPU-side.
- The model is loaded once. Each inference call contains only one configured window.
- DA3 inference uses `torch.inference_mode()` and DRIFTX's resolved CUDA autocast dtype: BF16 when PyTorch reports support, otherwise FP16; FP32 is also selectable. On CPU, FP32 is used.
- Auto-memory estimates a conservative starting window after model load. OOM triggers a full-sequence retry with a smaller window so all adjacent alignments use the same window geometry (fallback ladder: 32 → 24 → 16 → 12 → 8 → 4, skipping values above the current size).
- Only the current/previous prediction windows need to remain live for alignment. Merged CPU arrays use temporary NumPy memory-mapped files under `.driftx_stream_tmp`; those intermediates are removed after a successful export. GPU cache clearing is reserved for OOM recovery, not every chunk.
- Point-cloud export uses a deterministic global cap (one million points by default), assigned across frames in proportion to their valid depth pixels. This keeps geometry export bounded while `results.npz`, depth visualizations, and camera metadata cover every processed frame.
- `gsplat` is optional and is never invoked by the baseline. `gs_available`, `gs_error`, and `gs_enabled` are recorded in the report; no Gaussian output is fabricated.

## Outputs and observability

A successful run produces:

- `scene.glb` and `scene.ply` from the aligned global reconstruction.
- `results.npz` at the output root, plus the backwards-compatible `exports/mini_npz/results.npz` path. Arrays include depth, confidence, intrinsics, extrinsics, and `frame_ids`.
- `depth_vis/` for every processed frame.
- `frames.json` and `camera_poses.json` preserving source-frame identity and ordered global poses.
- `metrics.json` and `run_report.json`, including input counts, requested/effective window settings, chunk count, overlap alignment residuals, precision, OOM retries, confidence/depth coverage, CUDA peak/mean allocated memory when available, export timing, stage timings, and environment metadata.

A failed run still writes its error and traceback to `run_report.json` and persists the effective configuration in `metrics.json`. It never reports completion without checking required artifacts.

## Performance profiles

| Profile | Sample FPS | Total frame cap | Resolution | Initial chunk | Overlap | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `smoke` | 1 | 16 | 504 | 8 | 2 | Reproducible low-quality baseline / installation check |
| `balanced` | 2 | all | 630 | 16 | 4 | General long-video run; VRAM auto-sizing may lower the window |
| `quality` | 2 | all | 756 | 12 | 4 | Higher-resolution run; may require more VRAM or smaller overrides |

All values can be overridden. These are conservative starting points, not measured optima for every GPU. The historical smoke baseline remains the default to avoid changing its frame budget or resolution after collection starts.

## Technical reference and provenance

The implementation was informed by the repository's checked-in `da3_streaming/da3_streaming.py`, its base/KITTI configuration, and the `loop_utils` alignment utilities. Relevant ideas include chunk windows, shared overlap, sequential coordinate alignment, CPU/disk staging, mixed precision, and explicit cleanup. The DRIFTX runner does **not** import or copy the DA3 streaming classes or its loop detector/optimizer; DRIFTX-specific sampling, robust overlap estimation, retry policy, reports, and exports live under `driftx/`.

The checked-in official DA3 streaming sources retain their existing ByteDance/VGGT-Long attribution and Apache-2.0 notices. DA3 model/API files and command behavior are unchanged by the streaming implementation. Users must continue to comply with [`LICENSE`](../LICENSE), [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md), and the notices in the referenced source files.

## What remains out of scope

- Global loop closure, temporal/learned refinement, segmentation/confidence refinement, new meshing, and fine-tuning.
- CUDA performance measurements on a specific remote GPU. The supplied Linux sandbox has no PyTorch/CUDA runtime, so RTX 3050 measurements must be completed on the target Windows machine and saved with the test matrix.
- Automatic model downloads to local Windows paths; users may supply an already-downloaded DA3 Large 1.1 checkpoint.
