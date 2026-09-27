param(
    [string]$CudaHome = $env:CUDA_PATH
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CudaHome)) {
    $cudaRoot = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $CudaHome = Get-ChildItem $cudaRoot -Directory |
            Sort-Object Name -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
}

if ([string]::IsNullOrWhiteSpace($CudaHome)) {
    throw "CUDA Toolkit was not found. Install the NVIDIA CUDA Toolkit, restart PowerShell, and run this script again."
}

$CudaHome = (Resolve-Path $CudaHome).Path
$nvcc = Join-Path $CudaHome "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) {
    throw "CUDA_HOME '$CudaHome' is not a CUDA Toolkit root because '$nvcc' does not exist. Pass -CudaHome 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'."
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "MSVC cl.exe was not found. Install Visual Studio 2022 Build Tools with the Desktop development with C++ workload, then run this script from a Developer PowerShell."
}

$env:CUDA_HOME = $CudaHome
$env:CUDA_PATH = $CudaHome
$env:PATH = "$CudaHome\bin;$CudaHome\lib\x64;$env:PATH"

Write-Host "CUDA_HOME: $env:CUDA_HOME"
& $nvcc --version
python -c "import torch; print('Torch:', torch.__version__); print('Torch CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

python -m pip install --no-build-isolation `
    "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"

python -c "import gsplat; print('gsplat import: OK')"
