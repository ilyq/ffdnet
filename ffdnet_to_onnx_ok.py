import torch
import os
from pathlib import Path
from models.network_ffdnet import FFDNet


def get_model_params(model_name):
    if "color" in model_name:
        return 3, 96, 12  # color image settings
    else:
        return 1, 64, 15  # grayscale image settings


model_name = "ffdnet_gray_clip.pth"
n_channels, num_channels, num_blocks = get_model_params(model_name)
act_mode = "R"

# 实例化模型并加载预训练权重
model = FFDNet(
    in_nc=n_channels,
    out_nc=n_channels,
    nc=num_channels,
    nb=num_blocks,
    act_mode=act_mode,
)
model_path = Path("model_zoo") / model_name
if not model_path.exists():
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
