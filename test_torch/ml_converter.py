

import tflite2onnx

tflite_path = '/Users/user/test_torch/tflite/best_float32.tflite'
onnx_path = '/Users/user/test_torch/onnx'

tflite2onnx.convert(tflite_path, onnx_path)