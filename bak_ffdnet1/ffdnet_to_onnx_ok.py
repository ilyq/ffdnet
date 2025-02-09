import torch
import os

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

act_mode = "R"

# 实例化模型并加载预训练权重
model = FFDNet(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode=act_mode)
model_path = os.path.join("model_zoo", model_name)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型权重文件不存在：{model_path}")
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

dummy_image = torch.randn(1, n_channels, 256, 256, requires_grad=False)
dummy_sigma = torch.full((1, 1, 1, 1), 15 / 255.0, dtype=torch.float32)

onnx_filename = "ffdnet_gray_clip.onnx"
torch.onnx.export(
    model,  # 要导出的模型
    (dummy_image, dummy_sigma),  # 模型的输入（支持多个输入）
    onnx_filename,  # 导出  qwertyuyt`的文件路径
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
