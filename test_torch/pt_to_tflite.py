from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('finger_best.pt')

# Export the model to TFLite format
model.export(format='tflite') # creates 'yolov8n_float32.tflite'

# # Load the exported TFLite model
# tflite_model = YOLO('yolov8n_saved_model/yolov8n_float32.tflite')

# # Run inference
# results = tflite_model('https://ultralytics.com/images/bus.jpg')

    