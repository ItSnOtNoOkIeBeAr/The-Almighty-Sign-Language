# Nsight Compute (full metric set) for the two custom kernels, one launch each, plus the key numbers.
#   powershell -File server\scripts\profile_kernels.ps1 [-Build release]
# Needs GPU performance-counter access: NVIDIA Control Panel -> Desktop -> Enable Developer Settings -> Manage GPU Performance Counters
# -> "Allow access to the GPU performance counters to all users" (then reboot), or run this script from an administrator shell.
# Without it ncu stops with ERR_NVGPUCTRPERM. Reports go to training\runs\ncu_<kernel>.txt (gitignored).
param([string]$Build = "release")
$ErrorActionPreference = "Continue"
$root = Split-Path (Split-Path $PSScriptRoot)
$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" }
$env:PATH = "$cuda\bin;$cuda\bin\x64;C:\dev\TensorRT-10.16.1.11\bin;$env:PATH"
$ncu = Get-ChildItem "C:\Program Files\NVIDIA Corporation" -Directory -Filter "Nsight Compute*" | Sort-Object Name -Descending | Select-Object -First 1
$ncu = Join-Path $ncu.FullName "ncu.bat"
$exe = Join-Path $root "server\cpp\build\$Build\signlang_server.exe"
$frame = Join-Path $root "training\runs\bench_frame.jpg"   # any 640x480 live-format JPEG; paths with spaces break ncu.bat, this repo has none
if (-not (Test-Path $frame)) { throw "put a 640x480 JPEG at $frame (e.g. one frame from server\cpp\tests\parity_keypoints.py's temp folder)" }
foreach ($k in "preprocess_kernel", "simcc_kernel") {
    $out = Join-Path $root "training\runs\ncu_$k.txt"
    & $ncu --kernel-name $k --launch-skip 4 --launch-count 1 --set full $exe --bench 3 $frame > $out 2>&1
    Write-Host "== $k  (full report: $out)"
    Select-String -Path $out -Pattern "ERR_NVGPUCTRPERM|Duration|Memory Throughput|DRAM Throughput|L1/TEX Cache Throughput|L2 Cache Throughput|Compute \(SM\) Throughput|Achieved Occupancy|Theoretical Occupancy|Registers Per Thread|Block Size|Grid Size|Waves Per SM|Executed Ipc Active" |
        ForEach-Object { Write-Host "   $($_.Line.Trim())" }
    Select-String -Path $out -Pattern "^\s+(OPT|INF|WRN)\s" | ForEach-Object { Write-Host "   $($_.Line.Trim())" }
}
