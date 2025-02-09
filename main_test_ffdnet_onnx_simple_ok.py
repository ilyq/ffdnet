import time
import onnx
import onnxruntime as ort
import numpy as np
import cv2  # OpenCV 用于图像处理

# 加载 ONNX 模型
# onnx_model = onnx.load("ffdnet.onnx")

# 创建 ONNX Runtime 会话
ort_session = ort.InferenceSession(
    "ffdnet_correct.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# 加载输入图像并预处理
input_image_path = "input_image01.png"  # 输入图像路径
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
# image = cv2.resize(image, (1920, 1080))  # 调整大小

st = time.time() * 1000
image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1] 范围
x = image[np.newaxis, np.newaxis, :, :]  # 添加批次和通道维度

sigma_val = np.full((1, 1, 1, 1), 15 / 255.0, dtype=np.float32)

# 准备 ONNX Runtime 所需的输入字典
inputs = {"input_image": x, "sigma": sigma_val}

# 进行推理
outputs = ort_session.run(None, inputs)

# 获取输出图像并处理
output_tensor = outputs[0]
output_image = output_tensor[0, 0]  # 取出第一个批次和通道
output_image = np.clip(output_image, 0, 1) * 255.0  # 反归一化到 [0, 255]
output_image = output_image.astype(np.uint8)  # 转换为 uint8

print(time.time() * 1000 - st)
# 保存输出图像
output_image_path = "output_image.png"
cv2.imwrite(output_image_path, output_image)

print(f"输出图像已保存为 {output_image_path}")
