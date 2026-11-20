# Starts the Python fallback server (onnxruntime-gpu, same protocol and models) in the background. Logs: logs\fallback_server.log
#   powershell -File server\scripts\start_fallback.ps1 [-Profile primary|fallback] [-Foreground] [-Diag]
# Takes ~20 s to load the models. Only one server can own port 8765: use switch_server.ps1 to change between servers.
param([ValidateSet("primary", "fallback")][string]$Profile = "primary", [switch]$Foreground, [switch]$Diag)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"
$py = Join-Path $Root "training\.venv\Scripts\python.exe"
$script = Join-Path $Root "server\fallback\fallback_server.py"
$running = Get-ServerProcesses
if ($running.Cpp.Count -or $running.Python.Count) { throw "a server is already running (port $Port). Use: powershell -File server\scripts\switch_server.ps1 -To python" }
$env:SIGNLANG_TOKEN = Get-AppToken
New-Item -ItemType Directory -Force $LogDir | Out-Null
$extra = @()
if ($Diag) { $diagFile = Join-Path $LogDir ("diag_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".csv"); $extra = @("--diag-log", "`"$diagFile`""); Write-Host "diagnostic log: $diagFile" }
if ($Foreground) { & $py $script --profile $Profile @extra; exit $LASTEXITCODE }
$log = Join-Path $LogDir "fallback_server.log"
Start-Process -FilePath $py -ArgumentList (@("`"$script`"", "--profile", $Profile) + $extra) -RedirectStandardOutput $log -RedirectStandardError (Join-Path $LogDir "fallback_server.err") -WindowStyle Hidden
Write-Host "starting the Python fallback server (profile $Profile), log: $log (about 20 s to load)"
Show-Health
