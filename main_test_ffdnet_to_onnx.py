import time
import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util

from models.network_ffdnet import FFDNet as net

def prepare_paths(testset_name, model_name):
    model_pool = 'model_zoo'
    testsets = 'testsets'
    results = 'results'
    result_name = f"{testset_name}_{model_name}"
    model_path = os.path.join(model_pool, f"{model_name}.pth")
    L_path = os.path.join(testsets, testset_name)
    E_path = os.path.join(results, result_name)
    util.mkdir(E_path)
    return model_path, L_path, E_path, result_name

def load_model(model_path, n_channels, nc, nb, device):
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)

def process_image(img, n_channels, noise_level_img, use_clip, device):
    img_L = util.imread_uint(img, n_channels=n_channels)
    img_L = util.uint2single(img_L)
    np.random.seed(seed=0)
    img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
    if use_clip:
        img_L = util.uint2single(util.single2uint(img_L))
    img_L = util.single2tensor4(img_L).to(device)  # 确保数据在GPU上
    return img_L

def main():
    noise_level_img = 15
    model_name = 'ffdnet_gray_clip'
    testset_name = 'barcode'
    n_channels, nc, nb = (3, 96, 12) if 'color' in model_name else (1, 64, 15)
    use_clip = 'clip' in model_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path, L_path, E_path, result_name = prepare_paths(testset_name, model_name)
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, f"{logger_name}.log"))
    logger = logging.getLogger(logger_name)

    logger.info(device)

    model = load_model(model_path, n_channels, nc, nb, device)
    logger.info(f'Model path: {model_path}')

    # 导出 ONNX 模型
    x = torch.randn((2, 1, 1080, 1920)).to(device)  # 输入图像
    sigma = torch.randn(2, 1, 1, 1).to(device)    # 噪声水平
    torch.onnx.export(
        model,
        (x, sigma),  # 模型输入
        "ffdnet.onnx",  # 输出文件名
        export_params=True,  # 是否导出参数
        opset_version=11,  # ONNX 的操作集版本
        do_constant_folding=True,  # 是否执行常量折叠
        input_names=['input', 'sigma'],  # 输入名称
        output_names=['output'],  # 输出名称
        dynamic_axes={
            'input': {0: 'batch_size'},  # 动态批次大小
            'sigma': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("ONNX 模型已成功导出为 ffdnet.onnx")


    test_results = OrderedDict(psnr=[], ssim=[])
    logger.info(f'model_name:{model_name}, model sigma:{noise_level_img}, image sigma:{noise_level_img}')
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):
        st = time.time() * 1000
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = process_image(img, n_channels, noise_level_img, use_clip, device)
        sigma = torch.full((1,1,1,1), noise_level_img/255.).type_as(img_L).to(device)  # 确保sigma在GPU上
        img_E = model(img_L, sigma)  # 模型推理在GPU上
        img_E = util.tensor2uint(img_E)
        print(time.time() * 1000 - st)
        util.imsave(img_E, os.path.join(E_path, img_name+ext))

if __name__ == '__main__':
    main()

