# Export YOLOv8n to ONNX. Run in a separate environment so ultralytics doesn't
# end up in the build. Move the resulting yolov8n.onnx into models\.
pip install -U ultralytics
yolo export model=yolov8n.pt format=onnx
pip uninstall -y ultralytics
