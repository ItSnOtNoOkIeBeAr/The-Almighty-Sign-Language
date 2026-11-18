# Phase 3 helper: runs trtexec on every candidate engine, once with the shared 64 crops (outputs are exported for the
# TRT-vs-ORT comparison in training/phase3_eval.py trtcmp) and once for GPU latency percentiles.
# Needs the engines from the Phase 3 build (server/models/candidates/engines) and `phase3_eval.py trtprep` crops.
# "Continue": Windows PowerShell 5.1 turns trtexec's stderr warnings into terminating errors under "Stop".
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$env:PATH = "C:\dev\TensorRT-10.16.1.11\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;" + $env:PATH
$eng = Join-Path $root "server\models\candidates\engines"
$out = Join-Path $root "data\public\asl_citizen\phase3"
# trtexec splits --loadInputs on ":", so a drive letter breaks absolute paths: run inside $out with relative input names.
Set-Location $out

$pose = @(
    @{n = "rtmw_xl_384x288"; hw = "384x288"}, @{n = "rtmw_lm_256x192"; hw = "256x192"}, @{n = "rtmw_xl_256x192"; hw = "256x192"},
    @{n = "dwpose_l_384x288"; hw = "384x288"}, @{n = "dwpose_m_256x192"; hw = "256x192"}
)
foreach ($p in $pose) {
    $e = Join-Path $eng "$($p.n).engine"
    & trtexec.exe "--loadEngine=$e" "--shapes=input:64x3x$($p.hw)" "--loadInputs=input:crops_$($p.hw).bin" `
        "--exportOutput=$out\trt_$($p.n).json" --iterations=1 --warmUp=0 --duration=0 *> "$out\trt_$($p.n)_run.log"
    & trtexec.exe "--loadEngine=$e" "--shapes=input:1x3x$($p.hw)" --warmUp=1000 --duration=15 --iterations=500 --percentile=50,95,99 --noDataTransfers `
        *> "$out\trt_$($p.n)_lat.log"
    "$($p.n) done"
}
foreach ($d in @(@{n = "yolox_m"; s = "3x640x640"}, @{n = "yolox_tiny"; s = "3x416x416"})) {
    $e = Join-Path $eng "$($d.n).engine"
    & trtexec.exe "--loadEngine=$e" --warmUp=1000 --duration=15 --iterations=500 --percentile=50,95,99 --noDataTransfers *> "$out\trt_$($d.n)_lat.log"
    for ($i = 0; $i -lt 16; $i++) {
        $f = "{0:D2}" -f $i
        & trtexec.exe "--loadEngine=$e" "--shapes=input:1x$($d.s)" "--loadInputs=input:det_$($d.n)_$f.bin" `
            "--exportOutput=$out\trt_$($d.n)_$f.json" --iterations=1 --warmUp=0 --duration=0 *> "$out\trt_$($d.n)_$f.log"
    }
    "$($d.n) done"
}
