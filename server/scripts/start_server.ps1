# Starts the C++/CUDA server (primary) in the background with CUDA + TensorRT on PATH. Logs: logs\cpp_server.log
#   powershell -File server\scripts\start_server.ps1 [-Mode full|safe] [-Foreground] [-Diag]
# -Mode safe: small keypoint model (RTMW-l-m 256x192) from the start. -Foreground: run in this window (Ctrl+C stops it).
# -Diag: also write logs\diag_<time>.csv, one line per frame (class probabilities, top 3, decision state, hand detection): numbers only, no images.
# Only one server can own port 8765: use switch_server.ps1 to change between this and the Python fallback.
param([ValidateSet("full", "safe")][string]$Mode = "full", [switch]$Foreground, [switch]$Diag)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"
$exe = Join-Path $Root "server\cpp\build\release\signlang_server.exe"
if (-not (Test-Path $exe)) { throw "$exe not found. Build it: powershell -File server\scripts\build_cpp.ps1" }
$running = Get-ServerProcesses
if ($running.Cpp.Count -or $running.Python.Count) { throw "a server is already running (port $Port). Use: powershell -File server\scripts\switch_server.ps1 -To cpp" }
$env:SIGNLANG_TOKEN = Get-AppToken
Set-RuntimePath
New-Item -ItemType Directory -Force $LogDir | Out-Null
$extra = @()
if ($Diag) { $diagFile = Join-Path $LogDir ("diag_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".csv"); $extra = @("--diag-log", $diagFile); Write-Host "diagnostic log: $diagFile" }
if ($Foreground) { & $exe --mode $Mode @extra; exit $LASTEXITCODE }
$log = Join-Path $LogDir "cpp_server.log"
Start-Process -FilePath $exe -ArgumentList (@("--mode", $Mode) + $extra) -RedirectStandardOutput $log -RedirectStandardError (Join-Path $LogDir "cpp_server.err") -WindowStyle Hidden
Write-Host "starting the C++ server ($Mode mode), log: $log"
Show-Health
