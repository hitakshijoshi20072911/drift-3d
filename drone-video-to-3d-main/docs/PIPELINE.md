# Pipeline reference

Complete usage reference for the `sih_drone_pipeline` package: inputs, the Colab workflow, processing profiles, validation commands, outputs and verification datasets. For a project overview, see the [README](../README.md).

## Start here

You only need **one notebook** for a normal run:

> `notebooks/SIH26158_Colab.ipynb`

That notebook is the user interface. The actual implementation lives in `sih_drone_pipeline/`; do not copy code out of the notebook. Optional dataset notebooks are local experiment material and are not part of the core Git repository.

The command entry point is `sih_drone_pipeline/__main__.py`, which dispatches through `cli.py`; the reconstruction workflow itself is orchestrated by `colmap_pipeline.py`.

To inspect the local command interface:

```bash
python -m sih_drone_pipeline --help
```

## Inputs

- Drone video: 1080p or 4K, one continuous flight path.
- Telemetry: DJI-style `.srt` or CSV containing `time_s, latitude, longitude, altitude_m`.
- Optional independent distance checks for proving the one-metre accuracy target.

Camera intrinsics, IMU, barometric altitude, and RTK/PPK are useful but are not required by this implementation.

## Colab workflow

1. Upload this cleaned project folder to `MyDrive/SIH26158/`.
2. Upload the video and matching `.srt`/`.csv` telemetry to `MyDrive/SIH26158/input/`.
3. Open `notebooks/SIH26158_Colab.ipynb` in Google Colab.
4. Select a T4 GPU runtime and execute cells in order. The Conda installation cell intentionally restarts the runtime once.
5. Edit only the input filenames in the configuration cell.
6. Run the pipeline cell. Final files are written to `MyDrive/SIH26158/outputs/<run-name>/`.

Optional dataset demonstrations and verification material are kept locally but are intentionally excluded from the core-code Git repository.

### v14 (`--profile deadline-adaptive`)

- Speed: SIFT 4096 features, lighter bundle adjustment (local/global 15/30 iterations, one refinement).
- Coverage-guided depth maps: references are picked greedily by the ground footprint of each camera (`--dense-anchor-mode coverage`), not by time slices, capped at 96 references / 35% of frames. Fusion keeps points seen in `--fusion-min-num-pixels` (default 5) depth maps.
- Exact UTM export: ENU -> ECEF -> WGS84 -> UTM, so grid convergence is included (the old offset method drifted ~1 m per 100 m).
- DSM coverage gate is measured over the flown footprint (5 m cells, 30 m gap bridging) instead of the bounding box.
- Vertical reference check: DJI `rel_alt` heights are relative to takeoff. The run prints a `NOTE:` and `georeference.json` records it; pass `--telemetry-altitude-offset-m <takeoff elevation>` (notebook field `TELEMETRY_ALTITUDE_OFFSET_M`) for absolute DSM heights.

### Validated results (v14, Colab T4)

| Video length | Run time | Frames registered | Reprojection error | GPS alignment RMSE | DSM ground coverage |
|---|---|---|---|---|---|
| 4:11 | 8:44 | 121 / 170 (71.2%) | 0.554 px | 1.92 m | 84.8% |
| 8:37 | 11:11 | 131 / 131 (100%) | 0.543 px | 1.59 m | 91.9% |
| 11:18 | 18:05 | 236 / 236 (100%) | 0.532 px | 2.25 m | 80.6% |

GPS alignment RMSE compares reconstructed camera positions with the drone's own telemetry; it is
not a surveyed surface-accuracy test (see [Accuracy evidence](#accuracy-evidence)). On the 4:11
flight the unregistered frames are the take-off and hover/yaw segments, which have no parallax.

When `--target-frames` is omitted, selection is duration-aware at one frame per second, capped at 600 frames for `draft` and 1200 for `full`. Use `--sample-fps 2` for fast/low flights or set an explicit frame budget for quick debugging. `full` enables geometric dense consistency and tries Poisson meshing with an automatic Delaunay fallback. Processing time still depends strongly on scene length, motion and hardware; the verifier reports the 15-minute target instead of assuming it passes.

## Pipeline

```text
video + timed GPS telemetry
        |
sharp ordered keyframes + per-frame GPS
        |
COLMAP sequential SfM
        |
GPS alignment in local East-North-Up metres
        |
GPU dense stereo + fusion
        |
filtered point cloud + cleaned mesh + optional texture atlas
        |
PLY / LAS / GLB / OBJ / GeoTIFF DSM + reports
```

The one-command interface used by the notebook is:

```bash
python -m sih_drone_pipeline run \
  --video /content/input/flight.mp4 \
  --telemetry /content/input/flight.srt \
  --workspace /content/sih_work \
  --output /content/drive/MyDrive/SIH26158/outputs/demo \
  --target-frames 120 \
  --quality draft \
  --ai-mask-dynamic
```

For the optimized, reproducible path use `--profile verified-fast`. It enables geometry-aware
keyframes, forward-only decoding, robust multi-model GPS pose-prior mapping, the accepted reduced
bundle-adjustment schedule, adaptive dense stride two with `1088 / 4 / 15 / 10` PatchMatch settings,
and Delaunay `max_proj_dist=20`. It deliberately does not use the rejected 96/92-reference cap,
1024-pixel dense images, fewer iterations/samples/sources, or `window_step=2`. `--profile
high-detail` retains more dense evidence when runtime is not the primary constraint. Every run
writes `capture_quality.json`; add `--strict-capture-quality` to stop before COLMAP when blur,
exposure, overlap, or telemetry gates fail.

For controlled deadline experiments, `--profile deadline-preview` caps SIFT at 4096 features,
uses 72 adaptive dense references with eight source views and a reduced 896/3/10 PatchMatch
workload, disables dynamic-object masking, and stops before dense stereo when sparse registration
is below 95%, reprojection exceeds 0.75 px, or camera/GPS alignment RMSE exceeds 2 m. This profile is intentionally labelled preview:
promote it only after comparing DSM coverage, dense points, mesh integrity, and independent
accuracy evidence with a verified run.

The `deadline-reuse14` profile retains sequential geometry-aware selection, uses 64 adaptive
dense references with six source views at 896 px, and can reuse an already validated ordered-frame
cache. Reuse requires matching frame count, selection/decode modes, maximum width, source-video
size, manifest, and selected JPEGs.

For arbitrary video lengths, `deadline-dynamic` keeps the same sparse and dense safety settings but
derives dense references as 35% of accepted keyframes, bounded to 20–64.

Capture can also be screened without running reconstruction:

```bash
python -m sih_drone_pipeline assess \
  --video flight.mp4 --telemetry flight.csv \
  --output capture_quality.json
```

Current COLMAP builds can also be screened with learned local features and the newer surface
mesher, without changing the protected defaults:

```bash
python -m sih_drone_pipeline run \
  --video flight.mp4 --telemetry flight.csv --output outputs/learned_frontend \
  --target-frames 253 --max-width 1920 \
  --feature-type aliked-n16rot --feature-matcher lightglue \
  --mesher advancing-front
```

ALIKED and LightGlue require an ONNX-enabled COLMAP build; advancing-front meshing requires CGAL.
`preflight` reports these capabilities before a long run. SIFT plus brute-force matching and the
profile-selected Delaunay/Poisson path remain the defaults. Treat these switches as controlled A/B
candidates and promote them only through the fixed-workload and independent-evidence gates below.

Add `--validation-distances validation_distances.csv` for known-distance validation, or `--validation-checkpoints checkpoint_measurements.csv` for direct surveyed-vs-reconstructed XYZ checkpoint validation. At least three populated checkpoint rows are required for this project's minimal one-metre evidence gate; this is not a standards-compliance claim. The expected CSV columns are documented by each command's `--help` output.

### Camera calibration and positioning uncertainty

Normalized telemetry CSV may additionally contain `horizontal_accuracy_m`,
`vertical_accuracy_m`, `position_source` (for example `RTK_FIXED`, `RTK_FLOAT`, `PPK`, or
`GPS`), and `altitude_datum`. Pose-prior mapping automatically uses the median supplied uncertainty. It can be overridden
with `--gps-prior-horizontal-std-m` and `--gps-prior-vertical-std-m`.
Use `--telemetry-altitude-offset-m` only for a known geoid/vertical-datum correction and record the
datum in the CSV; the software cannot infer a missing datum safely.

Use `--camera-calibration camera.json` to supply stable intrinsics instead of relying only on
self-calibration:

```json
{
  "model": "OPENCV",
  "params": [1450.2, 1448.9, 960.0, 540.0, -0.08, 0.02, 0.0, 0.0],
  "width": 1920,
  "height": 1080,
  "fixed_intrinsics": true,
  "rolling_shutter_readout_ms": 12.4,
  "digital_stabilization": false,
  "source": "checkerboard calibration 2026-09-13"
}
```

### GCP correction versus independent checkpoints

`--gcp-control-points gcp.csv` robustly fits a 3D similarity correction before point-cloud, mesh,
LAS, GLB, OBJ, and DSM export. These control points influence the result and therefore cannot prove
accuracy. Keep at least three different surveyed points in `--validation-checkpoints` as a holdout
test.

GCP rows require `control_id,reconstructed_x_m,reconstructed_y_m,reconstructed_z_m` plus either
`known_x_m,known_y_m,known_z_m` in the reconstruction ENU frame or
`known_latitude,known_longitude,known_altitude_m`. Outliers are rejected using
`--gcp-inlier-threshold-m`. A 25% scale-change safety gate rejects likely coordinate-frame or
point-labelling mistakes; adjust it deliberately with `--gcp-max-scale-change-percent` only when
the initial model scale is known to be worse.

### Post-run accuracy validation

Independent checkpoints can be added to a completed reconstruction without rerunning COLMAP or
dense stereo:

```bash
python -m sih_drone_pipeline validate-checkpoints \
  --output outputs/demo \
  --checkpoints checkpoint_measurements.csv
```

Checkpoint validation performs a direct comparison in the existing metric coordinate frame; it
does not align or rescale the measurements. The report includes RMSE for X, Y, Z, horizontal,
vertical and 3D error, East/North/Up bias, per-checkpoint 3D error, and mean, median, population
standard deviation, minimum, maximum and p95 distributions. CSV rows may also include the optional
`x_uncertainty_m`, `y_uncertainty_m`, `z_uncertainty_m`, `horizontal_uncertainty_m`, and
`vertical_uncertainty_m` fields. Those uncertainty values are summarized exactly as supplied; the
software does not infer or convert their confidence level.

Three independent checkpoints are only this project's minimum evidence gate for the one-metre
target. The separate `asprs_checkpoint_count_status` requires at least 30 checkpoints, but its
scope is explicitly `checkpoint_count_only`. A passing count is not full ASPRS compliance:
checkpoint distribution, survey quality, confidence level, product class and the other standard
requirements must still be verified separately.

When an independently surveyed/reference point cloud or mesh is available, evaluate the entire
dense surface with:

```bash
python -m sih_drone_pipeline validate-geometry \
  --output outputs/demo \
  --reference reference_surface.ply \
  --reconstruction outputs/demo/point_cloud.ply \
  --threshold-m 0.10 \
  --threshold-m 0.25 \
  --max-samples 100000
```

`--reconstruction` is optional and defaults to `OUTPUT/point_cloud.ply`; `--threshold-m` may be
repeated. The command records bidirectional distance summaries
plus precision, completeness/recall and F1 at every requested distance threshold. Both inputs must
already share the same metric coordinate frame: no registration, similarity alignment or scale
correction is performed during validation. Their SHA-256 identities and the complete sampling
protocol are stored for reproducibility. A byte-identical reconstruction/reference pair is retained
as a diagnostic but rejected as independent benchmark evidence. The project's conservative
one-metre dense gate requires the maximum sampled distance in both directions to be at most 1 m;
this is a project acceptance rule, not a claim that a universal dense-reconstruction standard uses
that threshold.

The notebooks also run these verification commands automatically:

```bash
python -m sih_drone_pipeline preflight --video flight.mp4 --telemetry flight.srt \
  --workspace /content/sih_work --output preflight.json
python -m sih_drone_pipeline verify --output /path/to/completed/output
```

`preflight` checks CUDA COLMAP, NVIDIA visibility, video metadata, telemetry coverage and free disk space. `verify` separately reports structural artifact validity, quality thresholds, and `production_ready`. A readable file is not treated as proof of map quality.

## Outputs

| File | Purpose |
|---|---|
| `point_cloud.ply` | Coloured dense point cloud |
| `mesh.ply` | Full mesh |
| `textured/mesh.ply` + `texture.png` | COLMAP UV mesh and texture atlas when supported |
| `point_cloud.las` | UTM GIS point cloud with CRS metadata |
| `dsm.tif` | UTM GeoTIFF digital surface model |
| `confidence.tif` | UTM point-density confidence proxy from 0 to 1 |
| `confidence_summary.json` | Coverage and high-confidence-area statistics |
| `capture_quality.json` | Blur, exposure, overlap, telemetry and GPS-jump gates |
| `gcp_alignment.json` | Robust control-point fit, residuals and scale change when GCPs are used |
| `model.glb` | Browser-ready 3D model in local metric ENU coordinates |
| `model.obj` | Interoperable mesh |
| `georeference.json` | ENU origin and projected CRS |
| `run_report.json` | Stage timings, registration, reprojection and validation metrics |
| `viewer_metadata.json` | Compact metadata for the web viewer |
| `logs/*.log` | Full console output for every COLMAP stage |
| `gpu_usage.csv` | NVIDIA utilization, VRAM, temperature and power sampled every second |
| `gpu_usage.summary.json` | Mean and maximum GPU statistics |
| `preflight.json` | CUDA, COLMAP, video, telemetry coverage and disk checks |
| `verification_report.json` | Pass/fail checks plus GPU utilization/VRAM statistics for each stage |

A normal run writes OBJ, PLY and GLB for mesh interoperability and LAS and GeoTIFF for GIS use. glTF and FBX are produced by an optional helper, `target_metrics.export_interchange(model.glb, output_dir)`, which converts `model.glb` through Blender when Blender is installed; it is not called by the default run.

## Telemetry CSV

```csv
time_s,latitude,longitude,altitude_m,yaw_deg,pitch_deg,roll_deg,horizontal_accuracy_m,vertical_accuracy_m,position_source,altitude_datum
0.000,28.613900,77.209000,122.4,90.0,-35.0,0.3,0.03,0.06,RTK_FIXED,WGS84_ELLIPSOID
0.100,28.613901,77.209003,122.5,90.2,-35.1,0.2,0.03,0.06,RTK_FIXED,WGS84_ELLIPSOID
```

Only the first four columns are mandatory. Times are seconds from the start of the video. DJI `.srt` telemetry is parsed directly when it contains timestamped latitude, longitude and altitude fields.

## Capture protocol

- Fly slowly and continuously; do not rotate from a stationary point.
- Target about 80% forward overlap and 70% side overlap; increase it for vegetation or complex terrain.
- Keep the camera angle stable and avoid sudden yaw.
- Prefer daylight without strong moving shadows.
- Include oblique views of facades; a purely nadir path cannot recover vertical surfaces.
- Keep GPS recording and video recording synchronized from the same flight.
- Avoid moving vehicles and crowds where possible.

The Colab notebook enables AI dynamic-object segmentation by default. It masks people, vehicles, and animals before feature extraction, directly addressing one of the problem statement's key challenges.

Surfaces never visible in the single pass cannot be measured reliably. The system reports observable reconstruction quality rather than silently inventing geometry.

Before export, isolated dense-cloud points are rejected with a robust local-neighbour test. Mesh faces whose longest edge is an extreme outlier are removed before disconnected-fragment cleanup and COLMAP texturing. The report preserves the removed point/face counts; this prevents the large bridging triangles seen when Delaunay closes unsupported gaps.

## Viewer

```bash
cd sih_drone_pipeline/viewer
npm install
npm run dev
```

Open a run's output folder (or drop it on the page). The viewer picks up `model.glb`, `point_cloud.ply`, `georeference.json` and the run/verification reports automatically, and works fully offline.

| Tool | Answers |
|---|---|
| Point | Latitude/longitude, MGRS, UTM and height of any location |
| Distance | Ground and 3D length of a path |
| Height | Base-to-top height of a structure |
| Area | Plan and surface area, perimeter, height range |
| Volume | Fill and cut against a base fitted to the outline (stockpiles, debris, pits) |
| Profile | Surface elevation chart with climb and slope along a route |
| Sight line | Whether one position can see another, and where the view is blocked |
| Note | Labelled markers |

Display modes: photo texture, colour by height, point cloud. Results export as an HTML analysis report, GeoJSON, KML or CSV, and screenshots include a scale bar and north arrow. The **Model quality** tab explains how far absolute coordinates can be trusted; measurements inside the model are never rescaled. Press `?` in the viewer for shortcuts.

`npm run build` produces a static `dist/` that can be served from any web server or opened offline. Put a run's files in `public/demo/` before building to ship a preloaded demo (`?demo`).

### Hosting on GitHub Pages

`.github/workflows/pages.yml` builds and deploys the viewer on every push that touches `sih_drone_pipeline/viewer/`. The demo model is not stored in the repository: attach a run's `model.glb`, `point_cloud.ply`, `georeference.json`, `run_report.json`, `verification_report.json`, `viewer_metadata.json` and an optional `manifest.json` (`{"name": ..., "credit": ..., "creditUrl": ...}`) to a release tagged `demo-data`. The workflow downloads them at build time; to change the demo, replace the release assets and re-run the workflow. Without that release the site still deploys, and visitors open their own output folder.

## Accuracy evidence

GPS alignment residual indicates whether reconstructed camera positions agree with flight telemetry, but it is not an independent accuracy test. For final judging, measure several known site distances or ground-control checkpoints and add them using the provided CSV template. The report only declares the one-metre surface target passed when these independent checks pass. Three checkpoints satisfy only the project's minimal evidence gate; the separate 30-checkpoint status covers count alone and does not assert complete ASPRS compliance. Relative camera-trajectory accuracy is reported separately and never substitutes for surveyed surface evidence.

## Public verification dataset

### WHU aerial-video benchmark (recommended metric test)

WHU publishes native 4K/60-fps regular and irregular UAV videos, camera calibration, ground-truth camera poses, and surveyed GCP observations. Its large archives are hosted through a browser download service, so download and extract the three official packages from [the WHU dataset page](https://gpcv.whu.edu.cn/data/WHU_Areial_Video_Dataset.html) into one Drive directory, for example `MyDrive/WHU_Aerial_Video/`.

Prepare WHU data with `python -m sih_drone_pipeline.dataset --dataset whu ...`. The adapter discovers the selected sequence, normalizes its pose timestamps, converts its local metric camera centers to GPS references using the supplied geographic origin, and writes standardized trajectory, GCP, and checkpoint files. Confirm whether the release stores translations as camera centers or world-to-camera extrinsics; the `--pose-translation-convention` option records this choice in `dataset_manifest.json`.

The first reconstruction run verifies the native video, camera trajectory, products, and runtime. For the project's minimal surface-accuracy evidence gate, identify at least three reconstructed GCP centers in the metric viewer or CloudCompare, enter their XYZ values in `whu_checkpoint_measurements.csv`, then rerun with `--validation-checkpoints`. Use at least 30 properly distributed checkpoints before treating the count status as ASPRS-style count evidence, and still verify the rest of the standard separately. Blank template rows are ignored and camera-trajectory accuracy never substitutes for checkpoint accuracy.

The WHU capture is strongest for terrain, roads, rooftops, and vegetation. Only claim façade coverage if oblique frames visibly observe those façades; use a separate oblique development dataset and one final mixed-angle continuous flight for the complete SIH demonstration.

### Zurich mechanics test

The verification notebook uses the University of Zurich Urban Micro Aerial Vehicle sample:

- Official page: https://rpg.ifi.uzh.ch/zurichmavdataset.html
- Official sample archive: https://download.ifi.uzh.ch/rpg/AGZ_data/AGZ_subset.zip
- Sample size: under 200 MB; it is downloaded to `/content`, not committed to this repository.
- Contents used: time-synchronized 1920x1080 MAV images, onboard GPS and independent metric camera trajectory.
- Required academic citation: A. L. Majdik, C. Till and D. Scaramuzza, *The Zurich Urban Micro Aerial Vehicle Dataset*, IJRR, 2017.

The public sequence verifies software behavior, GPU utilization and camera-trajectory scale. Final surface-accuracy evidence must still come from surveyed distances or checkpoints in the SIH evaluation scene.

### Comparing runs

Completed output directories can be compared under locked workload and quality gates:

```bash
python -m sih_drone_pipeline compare   --before outputs/baseline   --after outputs/candidate   --output outputs/comparison.json
```

For two or more systems or configurations, normalize each result into this pipeline's report schema
and create a ranked JSON/CSV/Markdown benchmark:

```bash
python -m sih_drone_pipeline benchmark   --run-entry baseline=outputs/baseline   --run-entry ours=outputs/ours   --output outputs/benchmark.json
```

The first `--run-entry` is the protected reference. A candidate is eligible only when it is
production-ready and uses the same video, selected-frame manifest, frame resolution and
independent surface-evidence set; missing metadata or evidence fails closed. Only then does the
benchmark select the fastest eligible run, so a shorter or lower-resolution run cannot replace the
baseline merely because it finishes sooner.
