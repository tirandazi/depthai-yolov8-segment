from ultralytics import YOLO

height = 480
width = 640

# Load a model
model = YOLO("models/yolov8n-seg.pt")
# model = YOLO("path/to/custom_yolo_model.pt")

# Export the model
model.export(format="onnx", imgsz=(height, width))
