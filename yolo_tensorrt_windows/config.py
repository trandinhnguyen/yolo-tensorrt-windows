"""Tunable settings for the detection pipelines, tracker and UI.

Filesystem locations live in :mod:`yolo_tensorrt_windows.paths`.
"""

# --- Model -------------------------------------------------------------------
MODEL_INPUT_SIZE = 640  # YOLOv8 square input, in pixels
PAD_VALUE = 114  # letterbox fill colour (0-255)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5  # NMS

# Model files, relative to paths.MODELS_DIR
ONNX_MODEL_FILE = "yolov8n.onnx"
TRT8_ENGINE_FILE = "yolov8n_trt8.trt"
TRT10_ENGINE_FILE = "yolov8n_trt10.trt"

# --- TensorRT ----------------------------------------------------------------
# GPUs whose compute capability major version is <= this (Pascal and older) use
# TensorRT 8.5; newer GPUs use TensorRT 10.0.
TRT8_MAX_SM_MAJOR = 6
TRT_WORKSPACE_BYTES = 1 << 30

# --- Pipelines ---------------------------------------------------------------
PIPELINE_QUEUE_SIZE = 25  # max frames buffered between async stages
LOG_EVERY_N_FRAMES = 500
GPU_DEVICE = "cuda:0"
GPU_DECODER = "h264_cuvid"
GPU_DECODER_BUFFER_CHUNKS = 15
OUTPUT_FOURCC = "mp4v"

# --- Tracking (ByteTrack) ----------------------------------------------------
TRACK_ACTIVATION_THRESHOLD = 0.25
LOST_TRACK_BUFFER = 30  # frames (at 30 fps) a lost track is kept for re-matching
MINIMUM_MATCHING_THRESHOLD = 0.8
MINIMUM_CONSECUTIVE_FRAMES = 1

# --- UI ----------------------------------------------------------------------
UI_POLL_MS = 200  # how often the window checks for worker updates
