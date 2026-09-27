param(
    [string]$Model = "depth-anything/DA3-LARGE-1.1",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [ValidateSet("smoke", "balanced", "quality")]
    [string]$Profile = "smoke"
)

$ErrorActionPreference = "Stop"

python scripts\run_three_video_benchmark.py `
    --model $Model `
    --device $Device `
    --profile $Profile `
    --input-dir input `
    --output-dir outputs\benchmarks\three_video `
    --figures-dir output_figures\three_video `
    --videos test3.mp4 test4.mp4 test5.mp4

if ($LASTEXITCODE -ne 0) {
    throw "Three-video benchmark failed with exit code $LASTEXITCODE"
}
