# Install Python dependencies (Python 3.10) into the active virtual environment,
# or into .venv in the repository root (created if needed) when none is active.
# PyTorch comes from its CUDA 11.8 index; everything else from pyproject.toml.
$ErrorActionPreference = "Stop"

function Invoke-Native([scriptblock]$Command) {
    # Run native commands and fail on a non-zero exit code. Output goes to the console.
    $ErrorActionPreference = "Continue"  # native tools report progress on stderr
    & $Command | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Command" }
}

Push-Location (Split-Path $PSScriptRoot)
try {
    if (-not $env:VIRTUAL_ENV -and -not (Test-Path .venv)) {
        Invoke-Native { uv venv --python 3.10 .venv }
    }
    # uv uses $env:VIRTUAL_ENV if set, otherwise finds .venv in this directory.
    Invoke-Native {
        uv pip install `
            torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 `
            --index-url https://download.pytorch.org/whl/cu118
    }
    Invoke-Native { uv pip install -r pyproject.toml --extra dev }
}
finally {
    Pop-Location
}
