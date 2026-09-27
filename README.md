# DRIFTX

**AI-enabled, georeferenced 3D reconstruction from a single UAV flight.**

[![SIH 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-0b6e4f)](https://www.sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/PS-26158-1f6feb)](https://www.sih.gov.in/)
[![Team ZeroError](https://img.shields.io/badge/team-ZeroError-111827)](#team)

DRIFTX turns one continuous drone video, with limited views and optional flight metadata, into a measurable 3D representation for inspection, disaster response, mapping, and infrastructure analysis. It is designed for the **Smart India Hackathon 2026** problem statement **26158: “Single-Pass Drone Video to Accurate 3D Model Generation System”**, submitted under **Robotics & Drones** for **NTRO**.

> **One pass + limited views → complete 3D geometry.**

## Why DRIFTX

Conventional aerial mapping commonly depends on high image overlap, repeated flights, hundreds of frames, and long processing cycles. That workflow is difficult to use when a disaster zone, damaged structure, or inaccessible terrain allows only one safe flight.

DRIFTX is designed around that constraint:

- **One flight is enough:** a continuous UAV video replaces a large planned image set.
- **Video becomes structured data:** frame extraction and quality filtering produce a usable image sequence.
- **Geometry is recovered:** camera pose and depth are estimated before reconstruction.
- **The result is decision-ready:** point clouds, meshes, textures, and 3D assets can be inspected and measured.
- **Real-world context is retained:** GPS/IMU metadata can be fused with the visual trajectory for scale and georeferencing.

The target described by the project proposal is **≤1 m spatial accuracy** and **under 15 minutes for a 10-minute video**, with full visible-scene coverage. These are engineering targets, not a guarantee for every scene or hardware configuration.

## Pipeline

```text
Single UAV video + GPS/IMU metadata
                │
                ▼
Frame extraction → quality filtering → feature extraction
                │
                ▼
Depth + camera-pose estimation
                │
                ▼
SfM / MVS reconstruction → point cloud → mesh + texture
                │
                ▼
Georeferencing + validation → GLB / inspection-ready 3D output
```

### Current repository layout

This checkout remains compatible with the existing upstream package and CLI. The runtime package is under `src/depth_anything_3/`; `app_live.py` is a lightweight GLB viewer/inspection entry point.

| Area | Role |
|---|---|
| `frozen_ml/3d/` | Frozen third-party depth, pose, video, export, and service implementation |
| `src/depth_anything_3/` | Backwards-compatible import path for existing deployments |
| `driftx/` | Product namespace for ingest, preprocessing, geometry, reconstruction, georeferencing, validation, export, and benchmarking |
| `da3_streaming/` | Sliding-window streaming inference for long sequences |
| `app_live.py` | Local inspection of an existing `.glb` scene |
| `assets/examples/` | Small example inputs |
| `docs/` | Existing CLI, API, and benchmark documentation |
| `outputs/` | Existing generated example artifacts |

The DRIFTX product namespace is scaffolded for ingest, preprocessing, reconstruction, georeferencing, validation, export, and benchmarking. The frozen model remains isolated from that product code so future phases can evolve without changing the stable inference package in place.

## Quick start

### Install

```bash
pip install xformers "torch>=2" torchvision
pip install -e .
```

Optional capabilities:

```bash
pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install -e ".[app]"   # Gradio web application
pip install -e ".[all]"   # all optional capabilities
```

### Reconstruct example images

```bash
export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
export GALLERY_DIR=workspace/gallery
mkdir -p "$GALLERY_DIR"

da3 auto assets/examples/SOH \
  --model-dir "$MODEL_DIR" \
  --export-format glb \
  --export-dir "$GALLERY_DIR/DRIFTX_SOH"
```

### Process a video

```bash
da3 video assets/examples/robot_unitree.mp4 \
  --fps 15 \
  --model-dir "$MODEL_DIR" \
  --export-dir "$GALLERY_DIR/DRIFTX_VIDEO" \
  --export-format glb
```

The original `da3` entry point is intentionally retained for compatibility. The new `driftx` entry point exposes the DRIFTX product CLI, while the existing inference implementation remains available through `da3`. See [docs/CLI.md](docs/CLI.md) for all supported modes and [docs/API.md](docs/API.md) for Python usage.

### Inspect a generated GLB

```bash
python app_live.py path/to/scene.glb
```

### Run the baseline benchmark

The benchmark reuses the vendored inference and export paths while measuring
the video, sampling, inference, confidence, and export facts that are
available on the current machine. It writes `run_report.json` alongside the
extracted frames and artifacts.

```bash
python -m driftx benchmark \
  --video input/test.mp4 \
  --output outputs/benchmark_test
```

Optional controls include `--model`, `--device auto|cpu|cuda`,
`--sample-fps`, and `--process-res`. The report uses **`"not measured"`** for
values unavailable because of missing dependencies, hardware, or failed
inference; it never substitutes an invented accuracy or improvement number.

## Applications and impact

DRIFTX is aimed at rapid spatial intelligence where repeat inspection is expensive, slow, or unsafe:

- **Disaster management:** damage, debris, landslide, flood, and post-earthquake assessment.
- **Infrastructure inspection:** roads, bridges, buildings, and asset-condition monitoring.
- **Urban planning:** 3D city mapping and digital-twin inputs.
- **Archaeology and terrain:** rapid site documentation and discovery in difficult-to-access areas.
- **Aerial inspection:** measurement-ready geometry from a single mission.

The wider proposal also explores multimodal sensing with GPS, thermal data, and an ESP32/MPU-based vibration monitor. Its proposed event logic uses 200 Hz sampling, RMS/peak features, baseline comparison, timestamped records, and persistent high-vibration windows. Those capabilities are part of the product direction and should not be interpreted as already wired into this repository unless the relevant implementation is present.

## Feasibility and roadmap

The concept is designed to work with existing drone hardware and a cloud-compatible deployment model, avoiding a requirement for new flight equipment. Key production concerns are processing cost, blur and lighting, occlusion, telemetry accuracy, data security, and reliable validation.

Planned expansion areas include:

1. Robust video ingest and scene-quality checks.
2. GPS/IMU fusion, scale estimation, and georeferencing.
3. Accuracy and completeness validation with benchmark reports.
4. Secure dashboard delivery for measurement and collaboration.
5. Optional NeRF / 3D Gaussian Splatting export for higher-fidelity viewing.
6. Live data analysis, autonomous-drone support, and predictive maintenance workflows.

## Team

**ZeroError** — Smart India Hackathon 2026

Hitakshi Joshi · Ridhima Kulashri · Mannu · Renaissance Das · Saranya Jogi · Risha Rastogi

## Research and references

The proposal references the following research directions and datasets:

- [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)
- Towards Fast and Fully Automatic Drone Mapping — ACM, 2026
- E3D-Bench — 3D geometric foundation models
- Model-based Analysis of Multi-UAV Path Planning for Surveying
- Postdisaster Building Damage — *Scientific Reports*, 2021
- Accuracy and Effectiveness of Orthophotos Obtained from Low-Cost UAS Video Imagery
- AIGC-Enhanced UAV-Based 3D Mapping and Trajectory Planning for Rapid Disaster Response — ACM, 2025
- Datasets considered in the proposal: [DL3DV-10K](https://github.com/DL3DV-10K/DL3DV-10K), [ARKitScenes](https://github.com/apple/ARKitScenes), and [Waymo Open Dataset](https://waymo.com/open/)

## Built on

DRIFTX's geometry backbone is a frozen, pretrained open-source foundation model. Full attribution and license terms: see [CITATIONS.md](CITATIONS.md) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). The upstream code is licensed under [Apache-2.0](LICENSE); pretrained model weights may have separate terms, including **CC BY-NC 4.0** for the default nested checkpoint. Review the upstream [repository](https://github.com/ByteDance-Seed/Depth-Anything-3) and [model card](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) before redistribution or commercial use.

## Repository status

The DRIFTX refactor keeps the existing inference package, dependencies, licenses, model weights, generated outputs, and `da3` deployment entry point available through compatibility links. The vendored source is isolated under `frozen_ml/3d/`; DRIFTX product code lives under `driftx/`.
