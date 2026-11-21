# Build the executable into dist\ml_app with PyInstaller (see app.spec).
Push-Location (Split-Path $PSScriptRoot)
try {
    pyinstaller --noconfirm app.spec
}
finally {
    Pop-Location
}
