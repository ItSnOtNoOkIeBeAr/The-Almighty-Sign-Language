# Memory checks of the C++ code:
#   * compute-sanitizer (memcheck, racecheck, initcheck, synccheck) on the GPU tests, and memcheck on the full pipeline (--extract)
#   * AddressSanitizer on the CPU tests (network robustness, features, decision): host memory
#   powershell -File server\scripts\sanitize.ps1 [-Build release] [-Jpegs <folder of .jpg frames>]
# The ASan build is made with: server\scripts\build_cpp.ps1 -Dir asan -Asan
param([string]$Jpegs = "", [string]$Build = "release")
$ErrorActionPreference = "Continue"
$root = Split-Path (Split-Path $PSScriptRoot)
$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" }
$bin = Join-Path $root "server\cpp\build\$Build"
$env:PATH = "$cuda\bin;$cuda\bin\x64;C:\dev\TensorRT-10.16.1.11\bin;$env:PATH"
$san = Join-Path $cuda "compute-sanitizer\compute-sanitizer.exe"
# --launch-timeout 0: the default attach timeout expires while the CUDA context is created on this driver ("No attachable process found")
$opts = @("--launch-timeout", "0", "--target-processes", "application-only", "--error-exitcode", "9")
$failed = $false

foreach ($tool in "memcheck", "racecheck", "initcheck", "synccheck") {
    Write-Host "== compute-sanitizer --tool $tool : signlang_tests (GPU kernels)"
    $leak = if ($tool -eq "memcheck") { @("--leak-check", "full") } else { @() }
    $out = & $san --tool $tool @opts @leak (Join-Path $bin "signlang_tests.exe") 2>&1
    $out | Select-String -Pattern "ERROR SUMMARY|PASSED|FAILED|leaked" | ForEach-Object { Write-Host "   $_" }
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}
if ($Jpegs) {
    Write-Host "== compute-sanitizer --tool memcheck : full pipeline (--extract)"
    $out = & $san --tool memcheck @opts (Join-Path $bin "signlang_server.exe") --extract $Jpegs (Join-Path $env:TEMP "sanitize_extract.json") 2>&1
    $out | Select-String -Pattern "ERROR SUMMARY|extracted|fatal" | ForEach-Object { Write-Host "   $_" }
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}

$asanExe = Join-Path $root "server\cpp\build\asan\signlang_cpu_tests.exe"
if (Test-Path $asanExe) {
    $asanDll = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" -Recurse -Filter "clang_rt.asan_dynamic-x86_64.dll" | Select-Object -First 1
    $env:PATH = "$($asanDll.DirectoryName);$env:PATH"
    $env:ASAN_OPTIONS = "detect_leaks=0:halt_on_error=1"
    Write-Host "== AddressSanitizer : signlang_cpu_tests (network robustness, features, decision)"
    $out = & $asanExe 2>&1
    $out | Select-String -Pattern "ERROR: AddressSanitizer|SUMMARY|PASSED|FAILED" | ForEach-Object { Write-Host "   $_" }
    if ($LASTEXITCODE -ne 0) { $failed = $true }
} else {
    Write-Host "(no ASan build found: server\scripts\build_cpp.ps1 -Dir asan -Asan)"
}

if ($failed) { Write-Host "SANITIZER: ERRORS FOUND"; exit 1 } else { Write-Host "SANITIZER: clean" }
