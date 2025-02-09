import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 加载 ONNX 模型
onnx_model = onnx.load("ffdnet.onnx")

# 对模型进行量化
quantized_model = quantize_dynamic(onnx_model, "ffdnet_quantized.onnx", weight_type=QuantType.QInt8)
