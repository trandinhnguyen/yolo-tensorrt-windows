# Build dist\yolo-tensorrt-windows\yolo-tensorrt-windows.exe with PyInstaller
# (see app.spec). Uses the active virtual environment, or .venv if none is active.
$ErrorActionPreference = "Stop"

Push-Location (Split-Path $PSScriptRoot)
try {
    $python = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" }
              else { ".venv\Scripts\python.exe" }
    if (-not (Test-Path $python)) {
        throw "No Python environment found: run scripts\setup.ps1 first."
    }
    $ErrorActionPreference = "Continue"  # PyInstaller logs progress on stderr
    & $python -m PyInstaller --noconfirm app.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    Write-Host "Built dist\yolo-tensorrt-windows\yolo-tensorrt-windows.exe" -ForegroundColor Green
}
finally {
    Pop-Location
}
