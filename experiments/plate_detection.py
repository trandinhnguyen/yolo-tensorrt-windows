# Work in progress: licence-plate detection with landmarks (YOLOv5-face style
# model, DeepStream-compatible pre/post-processing). Not wired into the app.
import onnxruntime as ort

IMAGE_PATH = r"path\to\test_image.png"  # set to a local test image

onnx_session = ort.InferenceSession(
    r"models\plate_detect.onnx",
    providers=["CUDAExecutionProvider"],
)
input_name = onnx_session.get_inputs()[0].name
infer_func = lambda img_data: onnx_session.run(None, {input_name: img_data})

import cv2
import numpy as np


def preprocess_deepstream(image_path, target_shape=(320, 320)):
    """
    Preprocesses an image matching the DeepStream configuration:
    - Resize to 320x320
    - maintain-aspect-ratio: 1
    - symmetric-padding: 0 (Top-Left aligned padding)
    - net-scale-factor: 1/255 (0.003921569...)
    """
    # 1. Load the image (DeepStream typically processes in BGR or RGB depending on network-mode)
    # Assuming standard BGR layout for OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open or find the image: {image_path}")

    h_org, w_org = img.shape[:2]
    target_w, target_h = target_shape

    # 2. Calculate scale factor to maintain aspect ratio
    # DeepStream fits the image entirely inside the target canvas
    scale = min(target_w / w_org, target_h / h_org)
    new_w = int(round(w_org * scale))
    new_h = int(round(h_org * scale))

    # 3. Resize the image with the calculated dimensions
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 4. Create the black canvas for padding
    # DeepStream initializes memory to 0 (black background)
    preprocessed_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 5. Apply asymmetric padding (symmetric-padding: 0)
    # This places the resized image at the top-left corner (0, 0)
    preprocessed_img[0:new_h, 0:new_w] = resized_img

    # 6. Scale and convert data type (net-scale-factor: 0.003921569...)
    # 1 / 255 = 0.00392156862...
    scale_factor = 0.0039215697906911373
    preprocessed_img = preprocessed_img.astype(np.float32) * scale_factor

    # 7. HWC to CHW format (DeepStream/TensorRT standard layout)
    # If your model expects RGB, flip channels here:
    preprocessed_img = preprocessed_img[:, :, ::-1]
    preprocessed_img = np.transpose(preprocessed_img, (2, 0, 1))

    # 8. Add batch dimension (1, C, H, W)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

    return preprocessed_img


# Example Usage:
tensor = preprocess_deepstream(IMAGE_PATH)
print("Input tensor shape:", tensor.shape)  # Expected: (1, 3, 320, 320)
import numpy as np

# Global Configurations
IMAGE_WIDTH_LMs = 320
IMAGE_HEIGHT_LMs = 320
NMS_THRESH = 0.45
OBJ_THRESHOLD_LMs = 0.4
STRIDES_LMs = [8, 16, 32]
ANCHORS_LMs = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]


class DetectRes:
    def __init__(self):
        self.classes = 0
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0
        self.prob = 0.0
        self.landmark = []


def sigmoid_x(x):
    # Clips to prevent overflow errors in exp
    x = np.clip(x, -88.7, 88.7)
    return 1.0 / (1.0 + np.exp(-x))


def compute_iou(bbox1, bbox2):
    def overlap_1d(x1min, x1max, x2min, x2max):
        if x1min > x2min:
            x1min, x2min = x2min, x1min
            x1max, x2max = x2max, x1max
        return 0.0 if x1max < x2min else min(x1max, x2max) - x2min

    overlap_x = overlap_1d(bbox1.x, bbox1.x + bbox1.w, bbox2.x, bbox2.x + bbox2.w)
    overlap_y = overlap_1d(bbox1.y, bbox1.y + bbox1.h, bbox2.y, bbox2.y + bbox2.h)

    area1 = bbox1.w * bbox1.h
    area2 = bbox2.w * bbox2.h

    overlap_2d = overlap_x * overlap_y
    u = area1 + area2 - overlap_2d

    return 0.0 if u == 0 else overlap_2d / u


def nms_detect_lms(bboxes: list):
    # Sort by probability descending
    bboxes.sort(key=lambda box: box.prob, reverse=True)
    result_boxes = []

    for box1 in bboxes:
        keep = True
        for box2 in result_boxes:
            if keep:
                overlap = compute_iou(box1, box2)
                keep = overlap <= NMS_THRESH
            else:
                break
        if keep:
            result_boxes.append(box1)

    return result_boxes


def post_process_landmark(out_tensor: np.ndarray):
    """
    Expects out_tensor to be a flattened or 1D/2D numpy array containing the raw model outputs.
    Shape is typically (Total_Anchors, 16)
    """
    # Flatten if it comes in as multi-dimensional
    out = out_tensor.flatten()
    vec_result = []
    row_ind = 0
    num_elements_per_row = 16  # 1 + 15 from C++ code

    for n in range(3):
        num_grid_x = int(IMAGE_WIDTH_LMs / STRIDES_LMs[n])
        num_grid_y = int(IMAGE_HEIGHT_LMs / STRIDES_LMs[n])

        for q in range(3):
            anchor_w = ANCHORS_LMs[n][q * 2]
            anchor_h = ANCHORS_LMs[n][q * 2 + 1]

            for i in range(num_grid_y):
                for j in range(num_grid_x):
                    start_idx = row_ind * num_elements_per_row
                    pdata = out[start_idx : start_idx + num_elements_per_row]

                    box_score = sigmoid_x(pdata[4]) * sigmoid_x(pdata[15])

                    if box_score > OBJ_THRESHOLD_LMs:
                        box = DetectRes()
                        box.x = (sigmoid_x(pdata[0]) * 2.0 - 0.5 + j) * STRIDES_LMs[n]
                        box.y = (sigmoid_x(pdata[1]) * 2.0 - 0.5 + i) * STRIDES_LMs[n]
                        box.w = (sigmoid_x(pdata[2]) * 2.0) ** 2.0 * anchor_w
                        box.h = (sigmoid_x(pdata[3]) * 2.0) ** 2.0 * anchor_h
                        box.classes = 0
                        box.prob = box_score

                        landmark = [0.0] * 10
                        for k in range(5, 15, 2):
                            ind = k - 5
                            landmark[ind] = pdata[k] * anchor_w + j * STRIDES_LMs[n]
                            landmark[ind + 1] = (
                                pdata[k + 1] * anchor_h + i * STRIDES_LMs[n]
                            )

                        box.landmark = landmark
                        vec_result.append(box)

                    row_ind += 1

    return nms_detect_lms(vec_result)


def nv_ds_infer_parse_custom_yolo_v5_landmark(output_layer_buffer):
    """
    Python equivalent of NvDsInferParseCustomYoloV5Landmark.
    Returns a list of parsed object dictionaries mimicking NvDsInferInstanceMaskInfo.
    """
    boxes = post_process_landmark(output_layer_buffer)
    object_list = []

    for r in boxes:
        obj = {}
        obj["classId"] = r.classes

        # Calculate Bounding Box coordinates matching C++ CLIP logic
        left = int(np.clip(r.x - r.w / 2.0 - 3, 0, IMAGE_WIDTH_LMs - 1))
        top = int(np.clip(r.y - r.h / 2.0 - 3, 0, IMAGE_HEIGHT_LMs - 1))
        right = int(np.clip(r.x + r.w / 2.0 + 3, 0, IMAGE_WIDTH_LMs - 1))
        bottom = int(np.clip(r.y + r.h / 2.0 + 3, 0, IMAGE_HEIGHT_LMs - 1))

        obj["left"] = left
        obj["top"] = top
        obj["width"] = right - left
        obj["height"] = bottom - top
        obj["detectionConfidence"] = float(r.prob)

        # Handle Landmark mapping (skipping landmark index 4 x/y pairs exactly like C++)
        # This reduces the 5 landmarks down to 4 landmarks (8 float values)
        landmark_box = [0.0] * 8
        ind = 0
        for k in range(0, 10, 2):
            if k == 4:
                continue
            landmark_box[ind] = float(np.clip(r.landmark[k], 0, IMAGE_WIDTH_LMs - 1))
            landmark_box[ind + 1] = float(
                np.clip(r.landmark[k + 1], 0, IMAGE_HEIGHT_LMs - 1)
            )
            ind += 2

        obj["mask"] = landmark_box
        obj["mask_width"] = IMAGE_WIDTH_LMs
        obj["mask_height"] = IMAGE_HEIGHT_LMs

        object_list.append(obj)

    return object_list


det_res = nv_ds_infer_parse_custom_yolo_v5_landmark(output[0])
import numpy as np


def scale_bbox_to_original(left, top, width, height, org_w, org_h, target_dim=320):
    """
    Scales bounding box coordinates from the 320x320 DeepStream canvas
    back to the original image dimensions, accounting for asymmetric top-left padding.
    """
    # 1. Re-calculate the exact scale factor DeepStream used during preprocessing
    scale = min(target_dim / org_w, target_dim / org_h)

    # 2. Convert from Left/Top/Width/Height to absolute pixel coordinates
    right = left + width
    bottom = top + height

    # 3. Since padding is asymmetric (top-left aligned), the offset is (0, 0).
    # We simply divide the coordinates by the scale factor to reverse the resize.
    org_left = left / scale
    org_top = top / scale
    org_right = right / scale
    org_bottom = bottom / scale

    # 4. Clip coordinates to the original image boundaries to prevent overflow
    org_left = int(np.clip(round(org_left), 0, org_w - 1))
    org_top = int(np.clip(round(org_top), 0, org_h - 1))
    org_right = int(np.clip(round(org_right), 0, org_w - 1))
    org_bottom = int(np.clip(round(org_bottom), 0, org_h - 1))

    # 5. Recompute width and height for the original image space
    org_width = max(0, org_right - org_left)
    org_height = max(0, org_bottom - org_top)

    return org_left, org_top, org_width, org_height


# --- Example Usage ---
# Suppose your network processed a 1920x1080 stream squeezed into the 320x320 canvas.
# The network detected a box at left=50, top=40, width=30, height=60 on the canvas.
img = cv2.imread(IMAGE_PATH)

original_w, original_h = img.shape[1], img.shape[0]
obj = det_res[1]
c_left, c_top, c_width, c_height = obj["left"], obj["top"], obj["width"], obj["height"]

scaled_l, scaled_t, scaled_w, scaled_h = scale_bbox_to_original(
    c_left, c_top, c_width, c_height, original_w, original_h
)

print(
    f"Original Frame Box -> Left: {scaled_l}, Top: {scaled_t}, Width: {scaled_w}, Height: {scaled_h}"
)