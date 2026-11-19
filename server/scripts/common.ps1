# Helpers shared by start_server.ps1, start_fallback.ps1 and switch_server.ps1 (dot-source this file).
$script:Root = Split-Path (Split-Path $PSScriptRoot)
$script:Port = 8765
$script:LogDir = Join-Path $Root "logs"

function Get-AppToken {
    # The token comes from SIGNLANG_TOKEN only. The Windows user variable wins over this process's copy: a terminal opened before
    # the token was rotated still carries the OLD value, and a server started with it would answer every client with HTTP 401.
    # It is never printed or written to a file.
    $t = [Environment]::GetEnvironmentVariable("SIGNLANG_TOKEN", "User")
    if (-not $t) { $t = $env:SIGNLANG_TOKEN }
    if (-not $t) { throw "SIGNLANG_TOKEN is not set. Run: setx SIGNLANG_TOKEN <value>   (then open a NEW terminal)" }
    return $t
}

function Set-RuntimePath {
    $cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2" }
    $trt = if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { "C:\dev\TensorRT-10.16.1.11" }
    $env:PATH = "$cuda\bin\x64;$cuda\bin;$trt\bin;$env:PATH"
}

function Get-ServerProcesses {
    $cpp = @(Get-Process signlang_server -ErrorAction SilentlyContinue)
    $py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*fallback_server.py*" })
    [pscustomobject]@{ Cpp = $cpp; Python = $py }
}

function Stop-Servers {
    $s = Get-ServerProcesses
    foreach ($p in $s.Cpp) { Stop-Process -Id $p.Id -Force; Write-Host "stopped C++ server (pid $($p.Id))" }
    foreach ($p in $s.Python) { Stop-Process -Id $p.ProcessId -Force; Write-Host "stopped Python fallback server (pid $($p.ProcessId))" }
    Start-Sleep -Milliseconds 800
}

function Wait-Health([int]$Seconds = 60) {
    $end = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $end) {
        try { return (Invoke-RestMethod -Uri "http://127.0.0.1:$script:Port/health" -TimeoutSec 2) } catch { Start-Sleep -Milliseconds 500 }
    }
    return $null
}

function Show-Health {
    $h = Wait-Health 60
    if ($h) { Write-Host "OK: $($h.server) server v$($h.version) answering on 127.0.0.1:$script:Port" -ForegroundColor Green }
    else { Write-Host "NOT answering on port $script:Port after 60 s: see the log in $script:LogDir" -ForegroundColor Red; exit 1 }
    try {
        $t = Invoke-RestMethod -Uri "https://signlang.dheynorlanda.me/health" -TimeoutSec 8 -UserAgent "signlang-start/1.0"
        Write-Host "tunnel: wss://signlang.dheynorlanda.me/ws reaches the $($t.server) server" -ForegroundColor Green
    } catch { Write-Host "tunnel: not reachable from this PC right now ($($_.Exception.Message)). Check the cloudflared service / internet." -ForegroundColor Yellow }
}
