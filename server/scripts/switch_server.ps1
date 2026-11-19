# One command to choose which server owns port 8765 (the tunnel points there).
#   powershell -File server\scripts\switch_server.ps1 -To cpp       C++/CUDA server, full quality (the normal demo setup)
#   powershell -File server\scripts\switch_server.ps1 -To safe      C++ server on the small keypoint model (if the GPU is struggling)
#   powershell -File server\scripts\switch_server.ps1 -To python    Python fallback server (safety net, ~20 s to load)
#   powershell -File server\scripts\switch_server.ps1 -To stop      stop whatever is running
#   powershell -File server\scripts\switch_server.ps1 -To status    what is running now
# Clients reconnect by themselves (the laptop client retries with a back-off of 1-10 s), so a switch costs a few seconds.
param([Parameter(Mandatory)][ValidateSet("cpp", "safe", "python", "stop", "status")][string]$To)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"
$s = Get-ServerProcesses
if ($To -eq "status") {
    Write-Host ("C++ server: " + $(if ($s.Cpp.Count) { "running (pid $($s.Cpp[0].Id))" } else { "not running" }))
    Write-Host ("Python fallback: " + $(if ($s.Python.Count) { "running (pid $($s.Python[0].ProcessId))" } else { "not running" }))
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; Write-Host "port ${Port}: answering ($($h.server))" } catch { Write-Host "port ${Port}: nothing answering" }
    exit 0
}
Stop-Servers
switch ($To) {
    "cpp"    { & "$PSScriptRoot\start_server.ps1" -Mode full }
    "safe"   { & "$PSScriptRoot\start_server.ps1" -Mode safe }
    "python" { & "$PSScriptRoot\start_fallback.ps1" }
    "stop"   { Write-Host "no server running" }
}
