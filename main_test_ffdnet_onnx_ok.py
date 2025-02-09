import onnxruntime as ort
import numpy as np
import cv2
import os
import glob
import time

def preprocess_image(image_path, n_channels, noise_level, use_clip):
    """
    1. 读取图片，同时支持灰度或彩色图像
    2. 转为 float32 并归一化到 [0,1]
    3. 添加高斯噪声（标准差 = noise_level/255.0）
    4. 如果 use_clip 为 True，则对添加噪声后的图片 clip 到 [0,1]
    5. 转换数据格式，从 HWC 转 CHW，并增加 batch 维度
    """
    if n_channels == 1:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Fail to read image {image_path}")
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Fail to read image {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32) / 255.0
    noise_std = noise_level / 255.0
    # 固定种子以便复现实验（也可去掉）
    np.random.seed(0)
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    img_noisy = img + noise
    if use_clip:
        img_noisy = np.clip(img_noisy, 0, 1)
    
    img_noisy = np.transpose(img_noisy, (2, 0, 1))  # (C, H, W)
    img_noisy = np.expand_dims(img_noisy, axis=0)     # (1, C, H, W)
    return img_noisy

def postprocess_image(output, n_channels):
    """
    1. 去除 batch 维度，得到 (C, H, W)
    2. 将输出 clip 到 [0,1] 后转换为 [0,255] 的 uint8 图像
    3. 根据通道数转换格式，确保 cv2.imwrite 正常保存
    """
    output = np.squeeze(output, axis=0)  # (C, H, W)
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    
    if n_channels == 1:
        output = output[0]
    else:
        output = output.transpose(1, 2, 0)  # (H, W, C)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def main():
    # 参数设置
    noise_level = 15  # 噪声标准差（针对 uint8 图片尺度 0~255）
    model_name = 'ffdnet_gray_clip'  # 模型名称，包含 'color' 则认为是彩色模型，否则为灰度模型
    testset_dir = "testsets/barcode"  # 测试图片所在目录
    results_dir = f"results/barcode_{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    n_channels = 3 if 'color' in model_name else 1
    use_clip = 'clip' in model_name

    # 加载 ONNX 模型
    session = ort.InferenceSession("ffdnet.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    sigma_name = session.get_inputs()[1].name
    print("ONNX 模型输入名称：", input_name, sigma_name)

    # 构造 sigma，形状固定为 (1, 1, 1, 1)
    sigma_val = np.full((1, 1, 1, 1), noise_level / 255.0, dtype=np.float32)

    # 获取所有测试图片（支持 png, jpg, jpeg, bmp 格式）
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(testset_dir, ext)))
    
    if not image_paths:
        print("目录中未发现图片，请检查路径。")
        return

    for img_path in image_paths:
        print("正在处理图片：", img_path)
        st = time.time() * 1000
        # 预处理：读取、归一化、加噪
        input_img = preprocess_image(img_path, n_channels, noise_level, use_clip)
        # 构造模型输入
        ort_inputs = {input_name: input_img, sigma_name: sigma_val}
        # 执行推理
        ort_outs = session.run(None, ort_inputs)
        output = ort_outs[0]
        # 后处理：转换回 uint8 图像
        output_img = postprocess_image(output, n_channels)
        # 保存图片
        output_path = os.path.join(results_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, output_img)
        print(time.time() * 1000 - st)
        print("图片已保存至：", output_path)

if __name__ == '__main__':
    main()