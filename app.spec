# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: builds dist\yolo-tensorrt-windows\yolo-tensorrt-windows.exe.
# Run scripts\setup.ps1 first: it provides the DLLs and the model listed here.
import glob
import os

from PyInstaller.utils.hooks import collect_all

name = "yolo-tensorrt-windows"
console = False  # a desktop app; logs go to logs\ next to the working directory

# The ONNX model and any sample videos. TensorRT engines (*.trt) are specific to
# a GPU and TensorRT version, so each machine builds its own on first start.
datas = [
    (path, "models")
    for path in glob.glob(r"models\*")
    if os.path.isfile(path) and not path.endswith(".trt")
]
if not any(path.endswith(".onnx") for path, _ in datas):
    raise SystemExit(r"models\yolov8n.onnx is missing: run scripts\setup.ps1 first")

dll_files = [
    # CUDA 11 libraries TensorRT 8.5 loads by name (unpacked by scripts\setup.ps1)
    r"lib\nvidia\cublas\bin\cublas64_11.dll",
    r"lib\nvidia\cublas\bin\cublasLt64_11.dll",
    r"lib\nvidia\cuda_runtime\bin\cudart64_110.dll",
    r"lib\nvidia\cudnn\bin\cudnn64_8.dll",
    r"lib\nvidia\cudnn\bin\cudnn_ops_infer64_8.dll",
    r"lib\nvidia\cudnn\bin\cudnn_cnn_infer64_8.dll",
    r"lib\nvidia\cufft\bin\cufft64_10.dll",
    r"lib\TensorRT-8.5.3.1\lib\nvinfer.dll",
    r"lib\TensorRT-8.5.3.1\lib\nvinfer_plugin.dll",
    r"lib\TensorRT-8.5.3.1\lib\nvonnxparser.dll",
    r"lib\TensorRT-8.5.3.1\lib\nvparsers.dll",
    r"lib\TensorRT-8.5.3.1\lib\nvinfer_builder_resource.dll",
    r"lib\TensorRT-10.0.1.6\lib\nvinfer_10.dll",
    r"lib\TensorRT-10.0.1.6\lib\nvinfer_plugin_10.dll",
    r"lib\TensorRT-10.0.1.6\lib\nvonnxparser_10.dll",
    r"lib\TensorRT-10.0.1.6\lib\nvinfer_builder_resource_10.dll",
] + glob.glob(r"lib\ffmpeg\bin\*.dll")
missing = [f for f in dll_files if not os.path.isfile(f)]
if missing:
    raise SystemExit("Missing DLLs (run scripts\\setup.ps1 first):\n" + "\n".join(missing))

binaries = [(file, ".") for file in dll_files]

hiddenimports = []
excludes = ["onnxruntime", "ultralytics", "matplotlib", "IPython", "pytest"]

# Packages PyInstaller can't fully discover on its own (PySide6 has built-in hooks).
packages = [
    "cuda.bindings",
    "cuda.pathfinder",
    "torio",
]
for package in packages:
    data, binary, hiddenimport = collect_all(package)
    datas.extend(data)
    binaries.extend(binary)
    hiddenimports.extend(hiddenimport)


a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# UPX stays off: compressing CUDA/TensorRT DLLs can break them.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=name,
)
