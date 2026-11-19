# Runs every automated check that does not need a camera or the laptop, and prints one summary line per suite.
#   powershell -File server\scripts\run_tests.ps1 [-Build release]
# Uses port 8771 for its own throw-away server, so a running demo server on 8765 is not disturbed.
param([string]$Build = "release")
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\common.ps1"
Set-RuntimePath
$env:SIGNLANG_TOKEN = Get-AppToken
$bin = Join-Path $Root "server\cpp\build\$Build"
$results = @()
function Run($name, [scriptblock]$cmd, $pattern) {
    $out = & $cmd 2>&1 | Out-String
    $line = ($out -split "`n" | Select-String -Pattern $pattern | Select-Object -Last 1)
    $ok = ($LASTEXITCODE -eq 0)
    $script:results += [pscustomobject]@{ Suite = $name; Result = $(if ($ok) { "PASS" } else { "FAIL" }); Detail = "$line".Trim() }
}

Run "C++ CPU tests (decision, features, protocol, network robustness)" { & "$bin\signlang_cpu_tests.exe" } "PASSED|FAILED"
Run "C++ GPU tests (kernels vs CPU and cv2/numpy references)" { & "$bin\signlang_tests.exe" } "PASSED|FAILED"
Run "Python decision tests" { & "$Root\training\.venv\Scripts\python.exe" -m unittest "$Root\server\fallback\test_decision.py" } "^OK|FAILED"

# conformance against a throw-away C++ server
$p = Start-Process -FilePath "$bin\signlang_server.exe" -ArgumentList "--port", "8771" -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $env:TEMP "rt_server.log") -RedirectStandardError (Join-Path $env:TEMP "rt_server.err")
$up = $false
for ($i = 0; $i -lt 60 -and -not $up; $i++) { try { Invoke-RestMethod "http://127.0.0.1:8771/health" -TimeoutSec 2 | Out-Null; $up = $true } catch { Start-Sleep -Milliseconds 500 } }
if ($up) { Run "Conformance against the C++ server" { & "$Root\client\.venv\Scripts\python.exe" "$Root\protocol\conformance_test.py" --url ws://127.0.0.1:8771/ws } "passed" }
else { $results += [pscustomobject]@{ Suite = "Conformance against the C++ server"; Result = "FAIL"; Detail = "server did not start" } }
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue

$results | Format-Table -AutoSize -Wrap
if ($results | Where-Object Result -eq "FAIL") { Write-Host "SOME SUITES FAILED" -ForegroundColor Red; exit 1 } else { Write-Host "ALL SUITES PASSED" -ForegroundColor Green }
