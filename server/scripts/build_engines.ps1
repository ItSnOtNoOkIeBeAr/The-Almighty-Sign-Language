# Builds the TensorRT FP16 engines the C++ server loads (server/models/*.engine, gitignored). Engines are tied to the
# GPU and the TensorRT version: rerun this after a driver or TensorRT update.
#   powershell -File server\scripts\build_engines.ps1 [-Profile primary|fallback|all]
param([ValidateSet("primary", "fallback", "all")][string]$Profile = "all")
$ErrorActionPreference = "Continue"   # PS 5.1 turns trtexec stderr warnings into terminating errors under "Stop"
$root = Split-Path (Split-Path $PSScriptRoot)
$trt = if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { "C:\dev\TensorRT-10.16.1.11" }
$trtexec = Join-Path $trt "bin\trtexec.exe"
if (-not (Test-Path $trtexec)) { throw "trtexec not found at '$trtexec'. Set TENSORRT_ROOT." }
$env:PATH = "$trt\bin;$env:PATH"
$models = Join-Path $root "server\models"

function Build($name, $onnx, $shape) {
    $engine = Join-Path $models "$name.engine"
    Write-Host "== $name : $onnx -> $engine ($shape)"
    # One fixed shape (batch 1): the server runs one frame at a time.
    & $trtexec "--onnx=$(Join-Path $models $onnx)" --fp16 "--saveEngine=$engine" `
        "--minShapes=$shape" "--optShapes=$shape" "--maxShapes=$shape" 2>&1 | Select-String -Pattern "PASSED|FAILED|Error" | Select-Object -Last 3
    if (-not (Test-Path $engine)) { throw "engine $name was not built" }
}

if ($Profile -in "primary", "all")  { Build "keypoint_primary"  "rtmw_x_384x288.onnx"   "input:1x3x384x288" }
if ($Profile -in "fallback", "all") { Build "keypoint_fallback" "rtmw_l_m_256x192.onnx" "input:1x3x256x192" }
# The temporal engines (fixed [1,30,233]) are built by training\export_onnx.py together with their ONNX files.
