# Imports the VS 2022 x64 developer environment (cl, cmake, ninja) into the current PowerShell session.
# Dot-source it:  . "$PSScriptRoot\vsenv.ps1"
# Why: cl/cmake are not on the default PATH on this PC, and reopening VS Code from the
# Native Tools prompt is easy to forget.

if (Get-Command cl -ErrorAction SilentlyContinue) { return }

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at '$vswhere'. Is Visual Studio 2022 installed?" }

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "No Visual Studio install with the C++ x64 tools was found." }

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at '$vcvars'." }

# Run vcvars in cmd, dump the resulting environment, and copy it into this session.
$envDump = cmd /c "`"$vcvars`" >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed (exit $LASTEXITCODE)." }
foreach ($line in $envDump) {
    $i = $line.IndexOf('=')
    if ($i -gt 0) { Set-Item -Path "env:$($line.Substring(0, $i))" -Value $line.Substring($i + 1) }
}

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) { throw "cl still not on PATH after vcvars64.bat." }
