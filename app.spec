# -*- mode: python ; coding: utf-8 -*-
import glob
import os
from PyInstaller.utils.hooks import collect_all

name = "ml_app"
console = True

datas = [(r"models", "models")]
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

binaries = []
for file in dll_files:
    binaries.append((file, "."))

hiddenimports = []
excludes = ["onnxruntime", "ultralytics", "matplotlib", "IPython"]

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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name=name,
)
