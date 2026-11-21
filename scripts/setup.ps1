<#
.SYNOPSIS
    One-shot project setup: TensorRT 10.0 + 8.5, CUDA 11 DLLs, FFmpeg, the
    YOLOv8n ONNX model and the Python environment.

.DESCRIPTION
    Each step is skipped when its result already exists, so the script is safe
    to re-run. TensorRT 8.5.3 sits behind an NVIDIA Developer login: if its zip
    is not in lib\ (or your Downloads folder), the download page is opened in
    your browser; save the zip into lib\ and the script picks it up, either
    later in this run or the next time you run it.

.PARAMETER SkipPythonDeps
    Don't create/update the Python environment (scripts\install.ps1).

.EXAMPLE
    .\scripts\setup.ps1
#>
param(
    [switch]$SkipPythonDeps
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Root = Split-Path $PSScriptRoot
$Lib = Join-Path $Root "lib"
$Models = Join-Path $Root "models"
$TrtPackages = Join-Path $Root "yolo_tensorrt_windows\inference"
$Tmp = Join-Path $Root ".setup-tmp"

$TensorRT = @(
    @{
        Dir        = "TensorRT-10.0.1.6"
        Zip        = "TensorRT-10.0.1.6.Windows10.win10.cuda-11.8.zip"
        Url        = "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.0.1/zip/TensorRT-10.0.1.6.Windows10.win10.cuda-11.8.zip"
        Wheel      = "tensorrt-10.0.1-cp310-none-win_amd64.whl"
        Package    = "tensorrt_1001"
        NeedsLogin = $false
    },
    @{
        Dir        = "TensorRT-8.5.3.1"
        Zip        = "TensorRT-8.5.3.1.Windows10.x86_64.cuda-11.8.cudnn8.6.zip"
        Url        = "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.5.3/zip/TensorRT-8.5.3.1.Windows10.x86_64.cuda-11.8.cudnn8.6.zip"
        Wheel      = "tensorrt-8.5.3.1-cp310-none-win_amd64.whl"
        Package    = "tensorrt_853"
        NeedsLogin = $true
    }
)

# CUDA 11 libraries TensorRT needs (cuBLAS, cuDNN, ...), from NVIDIA's pip wheels.
$NvidiaWheels = @(
    "nvidia-cudnn-cu11==8.9.5.29",
    "nvidia-cublas-cu11==11.11.3.6",
    "nvidia-cuda-nvrtc-cu11==11.8.89",
    "nvidia-cuda-runtime-cu11==11.8.89",
    "nvidia-cufft-cu11==10.9.0.58"
)
$NvidiaMarkers = @(  # one DLL per wheel, to tell whether it's unpacked
    "cudnn\bin\cudnn64_8.dll",
    "cublas\bin\cublas64_11.dll",
    "cuda_nvrtc\bin\nvrtc64_112_0.dll",
    "cuda_runtime\bin\cudart64_110.dll",
    "cufft\bin\cufft64_10.dll"
)

# Shared FFmpeg 6 build with NVDEC (h264_cuvid), for the GPU pipelines' decoder.
$FFmpeg = @{
    Url    = "https://github.com/GyanD/codexffmpeg/releases/download/6.1.1/ffmpeg-6.1.1-full_build-shared.zip"
    Bin    = "ffmpeg-6.1.1-full_build-shared/bin/"  # folder inside the zip
    Marker = "avcodec-60.dll"
}

# --- Helpers -----------------------------------------------------------------

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Done([string]$Message) { Write-Host "    $Message" -ForegroundColor DarkGray }

function Invoke-Native([scriptblock]$Command) {
    # Run native commands and fail on a non-zero exit code. Output goes to the
    # console, not into the caller's return value.
    $ErrorActionPreference = "Continue"  # native tools report progress on stderr
    & $Command | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Command" }
}

function Expand-ZipEntries {
    # Extract the entries of $Zip under $Prefix into $Destination, keeping
    # their relative paths (minus $Prefix when -StripPrefix). Skips $Exclude.
    param([string]$Zip, [string]$Prefix, [string]$Destination,
          [switch]$StripPrefix, [string[]]$Exclude = @())
    $archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        foreach ($entry in $archive.Entries) {
            $name = $entry.FullName
            if (-not $name.StartsWith($Prefix) -or $name.EndsWith("/")) { continue }
            $relative = if ($StripPrefix) { $name.Substring($Prefix.Length) } else { $name }
            if ($Exclude -contains $relative) { continue }
            $target = Join-Path $Destination ($relative -replace "/", "\")
            New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
        }
    }
    finally {
        $archive.Dispose()
    }
}

$script:ToolVenv = $null
function Get-ToolPython {
    # Throwaway venv with pip, for downloading wheels and exporting the model.
    if (-not $script:ToolVenv) {
        $script:ToolVenv = Join-Path $Tmp "venv"
        Invoke-Native { uv venv --python 3.10 --seed --quiet $script:ToolVenv }
    }
    return Join-Path $script:ToolVenv "Scripts\python.exe"
}

# --- Steps -------------------------------------------------------------------

function Install-TensorRT($t) {
    # Returns $false if the zip still has to be downloaded by hand.
    $dir = Join-Path $Lib $t.Dir
    $zip = Join-Path $Lib $t.Zip
    $pyd = Join-Path $TrtPackages "$($t.Package)\tensorrt.cp310-win_amd64.pyd"

    if (-not (Test-Path $dir)) {
        if (-not (Test-Path $zip)) {
            $downloaded = Join-Path $env:USERPROFILE "Downloads\$($t.Zip)"
            if (Test-Path $downloaded) {
                Write-Done "Moving $($t.Zip) from Downloads to lib\"
                Move-Item $downloaded $zip
            }
            elseif ($t.NeedsLogin) {
                return $false
            }
            else {
                Write-Done "Downloading $($t.Zip)"
                Invoke-Native { curl.exe -fL --retry 3 -o "$zip.part" $t.Url }
                Move-Item -Force "$zip.part" $zip
            }
        }
        Write-Done "Extracting $($t.Zip)"
        $staging = Join-Path $Lib ".extract-tmp"
        if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $staging)
        Move-Item (Join-Path $staging $t.Dir) $dir
        Remove-Item -Recurse -Force $staging
    }

    if (-not (Test-Path $pyd)) {
        # Same result as "pip install the wheel, copy site-packages\tensorrt,
        # uninstall": take the package files straight from the wheel, keeping
        # the vendored __init__.py.
        Write-Done "Vendoring Python bindings into yolo_tensorrt_windows\inference\$($t.Package)"
        Expand-ZipEntries -Zip (Join-Path $dir "python\$($t.Wheel)") -Prefix "tensorrt/" `
            -Destination (Join-Path $TrtPackages $t.Package) -StripPrefix -Exclude "__init__.py"
    }

    Write-Done "$($t.Dir) ready"
    return $true
}

function Install-NvidiaDlls {
    $missing = @($NvidiaMarkers | Where-Object { -not (Test-Path (Join-Path $Lib "nvidia\$_")) })
    if ($missing.Count -eq 0) {
        Write-Done "lib\nvidia ready"
        return
    }
    # Same result as "pip install, copy site-packages\nvidia into lib\,
    # uninstall", without touching any environment.
    $wheels = Join-Path $Tmp "nvidia-wheels"
    $python = Get-ToolPython
    Write-Done "Downloading $($NvidiaWheels.Count) NVIDIA wheels"
    Invoke-Native {
        & $python -m pip download --quiet --no-deps "--only-binary=:all:" `
            --platform win_amd64 --python-version 3.10 -d $wheels @NvidiaWheels
    }
    foreach ($wheel in Get-ChildItem $wheels -Filter *.whl) {
        Write-Done "Unpacking $($wheel.Name)"
        Expand-ZipEntries -Zip $wheel.FullName -Prefix "nvidia/" -Destination $Lib
    }
    Write-Done "lib\nvidia ready"
}

function Install-FFmpeg {
    $bin = Join-Path $Lib "ffmpeg\bin"
    if (Test-Path (Join-Path $bin $FFmpeg.Marker)) {
        Write-Done "lib\ffmpeg ready"
        return
    }
    New-Item -ItemType Directory -Force $Tmp | Out-Null
    $zip = Join-Path $Tmp "ffmpeg.zip"
    Write-Done "Downloading $(Split-Path -Leaf $FFmpeg.Url)"
    Invoke-Native { curl.exe -fL --retry 3 -o $zip $FFmpeg.Url }
    Expand-ZipEntries -Zip $zip -Prefix $FFmpeg.Bin -Destination $bin -StripPrefix
    if (-not (Test-Path (Join-Path $bin $FFmpeg.Marker))) {
        throw "Unexpected FFmpeg archive layout: $($FFmpeg.Marker) not found"
    }
    Write-Done "lib\ffmpeg ready"
}

function Export-OnnxModel {
    $onnx = Join-Path $Models "yolov8n.onnx"
    if (Test-Path $onnx) {
        Write-Done "models\yolov8n.onnx ready"
        return
    }
    $python = Get-ToolPython
    $work = Join-Path $Tmp "export"
    New-Item -ItemType Directory -Force $work, $Models | Out-Null

    # Run export_onnx.ps1 inside the throwaway venv so ultralytics never lands
    # in the project environment.
    $savedPath, $savedVenv = $env:PATH, $env:VIRTUAL_ENV
    Push-Location $work
    try {
        $env:VIRTUAL_ENV = Split-Path (Split-Path $python)
        $env:PATH = "$(Split-Path $python);$savedPath"
        Invoke-Native { & (Join-Path $PSScriptRoot "export_onnx.ps1") }
    }
    finally {
        $env:PATH = $savedPath
        $env:VIRTUAL_ENV = $savedVenv
        Pop-Location
    }
    $exported = Join-Path $work "yolov8n.onnx"
    if (-not (Test-Path $exported)) { throw "ONNX export failed: $exported was not created" }
    Move-Item $exported $onnx
    Write-Done "models\yolov8n.onnx ready"
}

# --- Main --------------------------------------------------------------------

New-Item -ItemType Directory -Force $Lib | Out-Null
$pending = @()
try {
    Write-Step "TensorRT"
    foreach ($t in $TensorRT) {
        if (-not (Install-TensorRT $t)) {
            Write-Warning ("$($t.Zip) needs an NVIDIA Developer login. Opening the download page; " +
                "save the zip into $Lib.")
            Start-Process $t.Url
            $pending += $t
        }
    }

    Write-Step "CUDA 11 libraries (lib\nvidia)"
    Install-NvidiaDlls

    Write-Step "FFmpeg (lib\ffmpeg)"
    Install-FFmpeg

    Write-Step "ONNX model"
    Export-OnnxModel

    if ($SkipPythonDeps) {
        Write-Step "Python environment (skipped)"
    }
    else {
        Write-Step "Python environment"
        & (Join-Path $PSScriptRoot "install.ps1")
    }

    if ($pending) {
        Write-Step "TensorRT (waiting for manual downloads)"
        $pending = @($pending | Where-Object { -not (Install-TensorRT $_) })
    }
}
finally {
    if (Test-Path $Tmp) { Remove-Item -Recurse -Force $Tmp }
}

if ($pending) {
    Write-Warning ("Not finished: save $($pending.Zip -join ', ') into $Lib " +
        "(or your Downloads folder), then run this script again.")
    exit 1
}
Write-Host "Setup complete. Start the app with: python app.py" -ForegroundColor Green
