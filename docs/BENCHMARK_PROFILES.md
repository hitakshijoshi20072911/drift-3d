# DRIFTX Benchmark Profiles and Local Runbook

## Before you run

Run from the repository root and use the Python executable in your active virtual environment. The **baseline** benchmark uses **DA3 Large 1.1**; Gaussian mode may use an official Gaussian-capable DA3-GIANT checkpoint. Keep the DA3 weights outside Git.

### Windows PowerShell

```powershell
# 1. Update the checkout (stop if Git reports conflicts or local-file overwrite errors)
git pull origin main

# 2. Create/activate the virtual environment (skip creation if it already exists)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install the repository and optional figure dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[figures]"

# 4. Confirm the environment and CLI before downloading/running weights
python -c "import sys; print(sys.executable)"
python -m driftx benchmark --help

# 5. Download DA3 Large 1.1 to a Windows-safe local folder if not present
python -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3-LARGE-1.1', local_dir='models/DA3_LARGE_1.1_SAFE')"

# 6. Check CUDA and the model path
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU')"
Test-Path "models\DA3_LARGE_1.1_SAFE\config.json"
```

If you already downloaded the model, point `$MODEL` to that local DA3 Large 1.1 directory for baseline mode. Do not point baseline mode at a Nested Giant/Large directory; use `--reconstruction-mode gaussian` or `both` for an official Gaussian-capable checkpoint. You can set a specific device string (for example `cuda:0`) if needed.

## Reproducible smoke baseline (recommended first run)

This retains the baseline's 1 FPS, 16-total-frame budget, and 504 resolution, but runs at most 8 views per DA3 inference window and aligns each adjacent pair with 2 shared frames:

```powershell
$MODEL = "models\DA3_LARGE_1.1_SAFE"
python -m driftx benchmark `
  --video "input\test3.mp4" `
  --output "outputs\benchmarks\test3_smoke" `
  --model $MODEL `
  --device cuda `
  --profile smoke
```

Expected output tree:

```text
outputs/benchmarks/test3_smoke/
  input_images/
  scene.glb
  scene.ply
  results.npz
  exports/mini_npz/results.npz
  depth_vis/
  frames.json
  camera_poses.json
  metrics.json
  run_report.json
```

Do not judge success by the CLI status alone. Verify `status` and every path in `artifacts`:

```powershell
$report = Get-Content "outputs\benchmarks\test3_smoke\run_report.json" -Raw | ConvertFrom-Json
$report | Format-List status, frames_sampled, frames_processed, chunk_size, chunk_overlap, num_chunks, final_precision, cuda_oom_retries, peak_gpu_memory_mb, error
$report.artifacts | Format-List
```

Check the required files:

```powershell
@(
  "scene.glb", "scene.ply", "results.npz", "frames.json",
  "camera_poses.json", "metrics.json", "run_report.json"
) | ForEach-Object {
  $p = Join-Path "outputs\benchmarks\test3_smoke" $_
  if (!(Test-Path $p)) { throw "Missing artifact: $p" }
}
(Get-ChildItem "outputs\benchmarks\test3_smoke\depth_vis\*.jpg").Count
```

## Sequential target-GPU test matrix

Run **one command at a time**. Keep the same local model path. Auto-memory is on unless passed `--no-auto-memory`; CUDA OOM retries remain enabled either way. Use fresh output folders so runs do not overwrite each other.

```powershell
# A: 1 FPS, 16 total sampled frames, 504 resolution
python -m driftx benchmark --video "input\test7.mp4" --output "outputs\benchmarks\test7_A" --model $MODEL --device cuda --sample-fps 1 --max-frames 16 --process-res 504 --chunk-size 8 --chunk-overlap 2

# B: 1 FPS, 32 total sampled frames, 504 resolution
python -m driftx benchmark --video "input\test7.mp4" --output "outputs\benchmarks\test7_B" --model $MODEL --device cuda --sample-fps 1 --max-frames 32 --process-res 504 --chunk-size 8 --chunk-overlap 2

# C: 2 FPS, 32 total sampled frames, 630 resolution
python -m driftx benchmark --video "input\test7.mp4" --output "outputs\benchmarks\test7_C" --model $MODEL --device cuda --sample-fps 2 --max-frames 32 --process-res 630 --chunk-size 8 --chunk-overlap 2

# D: 2 FPS, 48 total sampled frames, 630 resolution
python -m driftx benchmark --video "input\test7.mp4" --output "outputs\benchmarks\test7_D" --model $MODEL --device cuda --sample-fps 2 --max-frames 48 --process-res 630 --chunk-size 8 --chunk-overlap 2

# E: 2 FPS, 64 total sampled frames, 630 resolution
python -m driftx benchmark --video "input\test7.mp4" --output "outputs\benchmarks\test7_E" --model $MODEL --device cuda --sample-fps 2 --max-frames 64 --process-res 630 --chunk-size 8 --chunk-overlap 2
```

The explicit `8/2` window settings make the A–E comparison reproducible for a 4 GB target. A CUDA OOM is retried automatically with smaller windows across the full sequence; if the run still fails at the four-frame floor, keep the report and retry with a lower resolution or fewer total sampled frames. Do not run multiple jobs in parallel. `run_report.json` records runtime, frames per second, peak/mean VRAM (when CUDA is available), processed frames, windows, OOM retry count, confidence/depth coverage, and per-stage times. Record visual quality by opening each GLB/PLY in the same viewer and note failures rather than inferring quality from runtime.

## Recommended commands

**Reproducible baseline (run once measurements begin and keep settings fixed):**

```powershell
python -m driftx benchmark --video "input\test7.mp4" --output "outputs\benchmarks\test7_baseline" --model $MODEL --device cuda --profile smoke
```

**Higher-quality all-sampled-frame run:**

```powershell
python -m driftx benchmark --video "input\test7.mp4" --output "outputs\benchmarks\test7_quality" --model $MODEL --device cuda --profile quality --max-frames 0 --precision auto
```

On 4–6 GB cards, start the quality profile with `--chunk-size 8 --chunk-overlap 2`; 756 resolution may still exceed available memory, in which case the automatic OOM fallback reduces the window. Resolution and frame cap remain user-controlled.

## Test this checkout without a model/GPU

The repository includes CPU-only unit tests for chunk coverage, overlap geometry, profile defaults, OOM ladder, and point cap:

```powershell
python -m unittest discover -s tests -v
python -m driftx benchmark --help
python -m compileall -q driftx
```

The provided Linux sandbox used for implementation has no PyTorch, OpenCV, CUDA, or RTX 3050 device. Therefore no model inference, target-GPU performance score, or target hardware quality result is claimed here. Run the A–E matrix above on the actual Windows GPU and share its five `run_report.json` files if you want me to compare stable settings.
