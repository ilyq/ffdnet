#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
此脚本加载 FFDNet 模型，并将其导出为 ONNX 模型。
请确保：
 - FFDNet 类及其依赖模块（例如 PixelUnshuffle、conv 和 sequential）已经正确实现；
 - 模型权重文件（如 model_zoo/ffdnet_gray.pth）存在
"""

import torch
import os

# 假设你有如下接口，这里直接从 models.network_ffdnet 导入 FFDNet
from models.network_ffdnet import FFDNet

model_name = "ffdnet_gray_clip.pth"

if "color" in model_name:
    n_channels = 3  # setting for color image
    nc = 96  # setting for color image
    nb = 12  # setting for color image
else:
    n_channels = 1  # setting for grayscale image
    nc = 64  # setting for grayscale image
    nb = 15  # setting for grayscale image

# 模型参数设置（此处以灰度图为例，如果是彩色图像请调整 in_nc/out_nc、nc、nb 参数）
# in_nc = 1
# out_nc = 1
# nc = 64
# nb = 15
act_mode = "R"

# 实例化模型并加载预训练权重
model = FFDNet(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode=act_mode)
model_path = os.path.join("model_zoo", model_name)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型权重文件不存在：{model_path}")
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

# 创建两个 dummy 输入：一张 dummy 图像及 sigma 噪声水平
# 注意：FFDNet 的 forward 函数内部会对图像做 Padding 以及 PixelUnshuffle/PixelShuffle 操作，
# 因此我们选择一个任意尺寸的 dummy 图像。这里选择 256x256，你也可以选择其他尺寸（会自动 padding）。
dummy_image = torch.randn(1, n_channels, 256, 256, requires_grad=False)
# sigma 的形状与 export 时保持一致（forward 内部会通过 repeat 调整尺寸）
dummy_sigma = torch.full((1, 1, 1, 1), 15 / 255.0, dtype=torch.float32)

# 导出为 ONNX 模型
onnx_filename = "ffdnet_gray_clip.onnx"
torch.onnx.export(
    model,  # 要导出的模型
    (dummy_image, dummy_sigma),  # 模型的输入（支持多个输入）
    onnx_filename,  # 导出的文件路径
    input_names=["input_image", "sigma"],
    output_names=["output"],
    dynamic_axes={
        "input_image": {2: "height", 3: "width"},
        "output": {2: "height", 3: "width"},
    },
    opset_version=11,
    do_constant_folding=True,
)

print(f"ONNX 模型已导出至: {onnx_filename}")
