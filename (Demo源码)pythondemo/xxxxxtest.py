import os
import cv2
import numpy as np

def overlay_image(ct_path, mask_path, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取原始CT图像的文件名
    ct_file_name = os.path.basename(ct_path)

    # 读取CT原图和Mask图
    ct_image = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像尺寸是否匹配
    if ct_image.shape != mask_image.shape:
        raise ValueError(f"图像尺寸不匹配: {ct_path} 和 {mask_path}")

    # 将CT图像转为三通道灰度图
    ct_image_color = cv2.cvtColor(ct_image, cv2.COLOR_GRAY2BGR)

    # 创建一个掩码，仅保留 Mask 中非黑色的部分
    mask_binary = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)[1]

    # 将Mask图像中非黑色部分改为红色
    mask_colored = np.zeros_like(ct_image_color)
    mask_colored[:, :, 2] = mask_binary  # 将红色通道设置为掩码的二值图

    # 添加边缘描边
    edges = cv2.Canny(mask_image, 100, 200)  # 检测边缘
    edges_colored = np.zeros_like(ct_image_color)
    edges_colored[:, :, 2] = edges  # 将边缘也设置为红色

    # 叠加Mask和边缘
    overlay_image = cv2.addWeighted(ct_image_color, 1.0, mask_colored, 0.5, 0)
    overlay_image = cv2.add(overlay_image, edges_colored)

    # 使用原始CT图像的文件名保存叠加后的图像
    output_path = os.path.join(output_folder, ct_file_name)
    cv2.imwrite(output_path, overlay_image)

    print(f"处理完成，结果已保存到: {output_path}")
    return output_path
