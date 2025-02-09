import time
import cv2
import numpy as np
import onnxruntime


def preprocess_image_with_padding(image_path, in_nc=1):
    if in_nc == 1:
        # 读取灰度图片
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        if img is None:
            raise ValueError(f"无法读取图片：{image_path}")
        img = img.astype(np.float32) / 255.0
        # add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)
    else:
        # 读取彩色图片
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        print(img.shape)
        if img is None:
            raise ValueError(f"无法读取图片：{image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

    # 记录原始高宽
    orig_h, orig_w = img.shape[1], img.shape[2]

    # 计算需要补充的行和列
    new_h = int(np.ceil(orig_h / 2) * 2)
    new_w = int(np.ceil(orig_w / 2) * 2)
    pad_bottom = new_h - orig_h
    pad_right = new_w - orig_w

    # 使用 np.pad 做简单的复制边缘 padding
    # 这里只对 H 和 W 维度进行 Padding，要求数组的格式为 (C, H, W)
    img = np.pad(img, ((0, 0), (0, pad_bottom), (0, pad_right)), mode="edge")

    # 添加 batch 维度: (1, C, H, W)
    img = np.expand_dims(img, axis=0)
    return img, orig_h, orig_w


def postprocess_output(output):
    """
    把输出的结果转换为 uint8 图像格式：
      - 裁剪数值到 [0,1]
      - 乘以 255 转换为 uint8
      - 如果是灰度图，去掉 channel 维度
    """
    output = np.clip(output, 0, 1)
    output = output[0]  # 去除 batch 维度
    if output.shape[0] == 1:
        # 灰度图：去除 channel 维度
        output = output[0]
    else:
        # 如果需要转换为 BGR，可调整通道顺序
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # 添加这行代码

    output = (output * 255.0).astype(np.uint8)
    return output


def main():
    onnx_model_path = "ffdnet_color_clip.onnx"
    in_nc = 3
    noise_level = 15 / 255.0

    ort_session = onnxruntime.InferenceSession(
        onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # image_path = "input_image.png"
    # image_path = "input_image01.png"
    image_path = "input_color.jpg"
    img, orig_h, orig_w = preprocess_image_with_padding(image_path, in_nc=in_nc)

    sigma = np.full((1, 1, 1, 1), noise_level, dtype=np.float32)
    ort_inputs = {"input_image": img, "sigma": sigma}
    ort_outs = ort_session.run(None, ort_inputs)
    output = ort_outs[0]

    # 使用 postprocess_output 函数处理输出
    output = postprocess_output(output)

    # 裁剪回原始尺寸
    if len(output.shape) == 2:
        output = output[:orig_h, :orig_w]
    else:
        output = output[:orig_h, :orig_w, :]

    cv2.imwrite("output_color.png", output)
    print("推理结果已保存至: output_color.png")


if __name__ == "__main__":
    st = time.time() * 1000
    main()
    print(time.time() * 1000 - st)
