# Usage: .\run.ps1 [server|client|both|headless|test] [extra client args]
#   server   fallback server on 127.0.0.1:8765
#   client   webcam client against the server in client\config.json, i.e. the tunnel (default)
#   headless 10 second client run without a window, same server
#   both     local server in a new window, then the client against it (needs training\.venv)
#   test     protocol conformance test against the local server (expects 9/9)
# Extra args go to the client, e.g. .\run.ps1 client --url ws://127.0.0.1:8765/ws
param(
    [ValidateSet('server', 'client', 'both', 'headless', 'test')]
    [string]$Mode = 'client',
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$ClientArgs
)

$root = $PSScriptRoot
$local = 'ws://127.0.0.1:8765/ws'
$server = Join-Path $root 'server\fallback\fallback_server.py'
$serverPy = Join-Path $root 'training\.venv\Scripts\python.exe'
$clientPy = Join-Path $root 'client\.venv\Scripts\python.exe'
$client = Join-Path $root 'client\client.py'

if (-not $env:SIGNLANG_TOKEN) {
    Write-Error 'SIGNLANG_TOKEN is not set in this terminal. Open a NEW terminal after setx.'
    exit 1
}

function Start-Client($more, $url) {
    $urlArgs = if ($url) { @('--url', $url) } else { @() }
    & $clientPy $client @urlArgs --width 640 --height 480 @more
}

switch ($Mode) {
    'server'   { & $serverPy $server }
    'client'   { Start-Client $ClientArgs }
    'both'     {
        Start-Process powershell -ArgumentList '-NoExit', '-Command', "& '$serverPy' '$server'"
        Start-Sleep -Seconds 3
        Start-Client $ClientArgs $local
    }
    'headless' { Start-Client @('--headless', '--duration', '10') }
    'test'     { & $clientPy (Join-Path $root 'protocol\conformance_test.py') --url $local }
}
