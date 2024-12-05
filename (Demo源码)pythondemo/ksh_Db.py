import os
import cv2
import numpy as np

def visualize_segmentation(original_img_path, ground_truth_path, prediction_path, output_path):
    # 读取原图、金标准图和预测图
    original_img = cv2.imread(original_img_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像尺寸是否一致
    if original_img.shape[:2] != ground_truth.shape[:2] or original_img.shape[:2] != prediction.shape[:2]:
        raise ValueError("图像尺寸不一致，请确保所有输入图像的尺寸相同")

    # 将金标准图和预测图规范化为二值图
    ground_truth = (ground_truth > 0).astype(np.uint8)
    prediction = (prediction > 0).astype(np.uint8)

    # 检查金标准图和预测图中的唯一值
    print(f"Ground truth unique values after binarization: {np.unique(ground_truth)}")
    print(f"Prediction unique values after binarization: {np.unique(prediction)}")

    # 创建颜色掩码
    correct_mask = (ground_truth == 1) & (prediction == 1)
    incorrect_mask = (ground_truth == 0) & (prediction == 1)
    missed_mask = (ground_truth == 1) & (prediction == 0)

    # 检查掩码
    print(f"Correct mask has {np.sum(correct_mask)} pixels")
    print(f"Incorrect mask has {np.sum(incorrect_mask)} pixels")
    print(f"Missed mask has {np.sum(missed_mask)} pixels")

    # 创建一个彩色掩码图像
    color_mask = np.zeros_like(original_img)

    # 正确预测为绿色
    color_mask[correct_mask] = [0, 255, 0]

    # 错误预测为红色
    color_mask[incorrect_mask] = [0, 0, 255]

    # 误判预测为蓝色
    color_mask[missed_mask] = [255, 0, 0]

    # 定义透明度
    alpha = 0.5  # 透明度

    # 创建一个叠加图像
    overlayed_img = original_img.copy()
    overlayed_img[color_mask.any(axis=-1)] = cv2.addWeighted(
        original_img[color_mask.any(axis=-1)], 1 - alpha, color_mask[color_mask.any(axis=-1)], alpha, 0
    )

    # 将图像右旋转90度
    rotated_img = cv2.rotate(overlayed_img, cv2.ROTATE_90_CLOCKWISE)

    # 保存结果图像
    cv2.imwrite(output_path, rotated_img)

    return rotated_img

def batch_visualize_segmentation(original_dir, ground_truth_dir, prediction_dir, output_dir):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件名
    original_files = os.listdir(original_dir)
    ground_truth_files = os.listdir(ground_truth_dir)
    prediction_files = os.listdir(prediction_dir)

    # 遍历所有图像文件
    for file_name in original_files:
        # 构建每个图像的完整路径
        original_img_path = os.path.join(original_dir, file_name)
        ground_truth_path = os.path.join(ground_truth_dir, file_name)
        prediction_path = os.path.join(prediction_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # 检查文件是否存在于所有目录中
        if file_name in ground_truth_files and file_name in prediction_files:
            try:
                # 可视化分割并保存结果
                visualize_segmentation(original_img_path, ground_truth_path, prediction_path, output_path)
                print(f"Processed {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"Skipping {file_name}: Missing ground truth or prediction file")

# 使用示例
original_dir = r'D:\test\mmlab\mmseg\data\Covid\Covid_MedSeg_1\val\img'
ground_truth_dir = r'D:\test\mmlab\mmseg\data\Covid\Covid_MedSeg_1\val\masks'
prediction_dir = r'D:\test\mmlab\mmseg\data\Covid\Covid_MedSeg_1\val\ksh\deeplabv3'
output_dir = r'D:\test\mmlab\mmseg\data\Covid\Covid_MedSeg_1\val\ksh\deeplabv3\output_images'

batch_visualize_segmentation(original_dir, ground_truth_dir, prediction_dir, output_dir)
