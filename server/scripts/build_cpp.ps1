# Configures (first time) and builds the C++ server and its tests into server\cpp\build\<Dir> (default: release).
#   powershell -File server\scripts\build_cpp.ps1 [-Clean] [-Dir release] [-Asan]
# -Asan builds only the CPU tests, with AddressSanitizer (use -Dir asan). Build into another -Dir while a server from build\release is running (the exe is locked).
param([switch]$Clean, [string]$Dir = "release", [switch]$Asan)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\vsenv.ps1"
$cpp = Join-Path (Split-Path (Split-Path $PSScriptRoot)) "server\cpp"
$build = Join-Path $cpp "build\$Dir"
$asanFlag = if ($Asan) { "ON" } else { "OFF" }
if (-not $env:TENSORRT_ROOT) { $env:TENSORRT_ROOT = "C:\dev\TensorRT-10.16.1.11" }
if ($Clean -and (Test-Path $build)) { Remove-Item $build -Recurse -Force }
if (-not (Test-Path (Join-Path $build "build.ninja"))) {
    cmake -S $cpp -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake `
        -DVCPKG_INSTALLED_DIR="$cpp/vcpkg_installed" -DVCPKG_TARGET_TRIPLET=x64-windows "-DSIGNLANG_ASAN=$asanFlag"
    if ($LASTEXITCODE) { throw "cmake configure failed" }
}
if ($Asan) { cmake --build $build --target signlang_cpu_tests } else { cmake --build $build }
if ($LASTEXITCODE) { throw "build failed" }
Write-Host "built in $build"
