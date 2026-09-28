# DRIFTX Evidence Report

## Scope

DRIFTX is a single-UAV-video reconstruction system. This report separates measured runtime/output facts from metrics that require ground truth. The public demo scenes are **PRECOMPUTED** repository artifacts; they are not live inference.

## Application scenes

| Demo | Description | Provenance | Georeference |
|---|---|---|---|
| test3 | IGDTUW COE Lab & New Building Road | Precomputed `outputs/DA3_LARGE_1.1_SAFE` | Unavailable; local reconstruction frame |
| test6 | Anveshan Foundation Construction Site | Precomputed `outputs/DA3_LARGE_1.1_SAFEtest6` | Unavailable; local reconstruction frame |
| test7 | IGDTUW Basketball Court, Kashmere Gate | Precomputed `outputs/DA3_LARGE_1.1_SAFEtest7` | Unavailable; local reconstruction frame |

## Evaluation contract

| Component | Metric | Ground truth needed | Current status |
|---|---|---|---|
| Depth | AbsRel, RMSE, δ1 | GT depth | Unavailable until GT depth is supplied |
| Camera pose | ATE, RPE | GT trajectory | Unavailable until surveyed trajectory is supplied |
| 3D reconstruction | Chamfer, RMSE, completeness | Reference point cloud/mesh | Unavailable until reference geometry is supplied |
| Georeferencing | position/control-point error | surveyed/RTK coordinates | Unavailable; no coordinates invented |
| Dynamic masking | precision/recall/F1/IoU | labeled masks | Not enabled in baseline |
| Runtime | seconds, FPS, VRAM | none | Measured per `run_report.json` when a run executes |
| Reconstruction coverage | visible-scene completeness | reference geometry | Unavailable until reference geometry is supplied |

The project's ≤1 m spatial-accuracy statement remains **TARGET / NOT YET VALIDATED**.

## Gaussian policy

DA3-LARGE is depth/pose-capable but not classified as Gaussian-capable. Gaussian runs fail before model inference unless checkpoint metadata exposes a Gaussian head, such as the official DA3-GIANT family. Cloud notebooks execute C0/C1 before larger experiments and persist reports/artifacts.

## Reproduction

1. Run `python -m driftx doctor --json`.
2. Run the baseline smoke benchmark with `--reconstruction-mode baseline`.
3. For Gaussian mode, use an official GS-capable checkpoint and the Kaggle/Colab notebook.
4. Preserve `run_manifest.json`, `run_report.json`, `metrics.json`, and all artifacts.
