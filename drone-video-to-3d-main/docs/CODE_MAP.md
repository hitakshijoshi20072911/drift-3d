# Source code map

The production implementation is the `sih_drone_pipeline` Python package. Notebooks configure and invoke this package; they do not contain the main reconstruction logic.

## Execution path

```text
python -m sih_drone_pipeline
    -> __main__.py
    -> cli.py
    -> colmap_pipeline.py
    -> extraction, telemetry, validation, and export modules
```

## Important modules

| File | Responsibility |
|---|---|
| `cli.py` | Defines the `run`, `preflight`, `verify`, checkpoint, geometry-validation, comparison, and benchmark commands |
| `colmap_pipeline.py` | Orchestrates sparse/dense COLMAP reconstruction and output generation |
| `extract_keyframes.py` | Decodes the video and selects usable frames |
| `adaptive_selection.py` | Parallax-aware keyframe selection with RANSAC-verified matches |
| `flight_classifier.py` | Recommends a processing profile and frame budget from telemetry |
| `feature_cache.py` | Reuses validated feature databases between runs |
| `telemetry.py` | Reads and synchronizes CSV/SRT flight telemetry |
| `gps_refinement.py` | GPS-anchored repair of global-SfM camera paths distorted by low-parallax sections |
| `camera.py` | Loads calibrated camera profiles and fixed-intrinsic policy |
| `capture_quality.py` | Measures blur, exposure, overlap, and telemetry readiness before reconstruction |
| `control_points.py` | Loads independent GCP/control-point observations for metric alignment |
| `ai_masking.py` | Masks dynamic objects before reconstruction |
| `export_gis.py` | Cleans geometry and exports GIS/model formats |
| `validation.py` | Calculates trajectory, known-distance, and surveyed-checkpoint evidence |
| `dense_validation.py` | Computes reproducible bidirectional point-cloud/mesh precision, completeness, F1, and distance metrics |
| `compare_runs.py` | Applies locked workload and quality-retention gates to a baseline/candidate pair |
| `benchmark.py` | Ranks only comparable, production-ready runs and exports JSON/CSV/Markdown results |
| `profiles.py` | Named processing profiles, including the v14 `deadline-adaptive` profile |
| `verify_outputs.py` | Checks that a completed run is structurally and metrically valid |
| `production_status.py` | Derives the terminal run state (e.g. `PARTIAL_VALID`) from the quality gates |
| `target_metrics.py` | Problem-statement target scorecard; optional glTF/FBX export via Blender (`blender_interchange.py`) |
| `preflight.py` | Checks GPU, COLMAP, input media, telemetry, and disk readiness |
| `dataset.py` | Adapts the optional public verification datasets |
| `gpu_monitor.py` | Records GPU utilization during a run |

The browser viewer is isolated under `sih_drone_pipeline/viewer/`. Automated tests mirror the Python modules under `tests/`.
