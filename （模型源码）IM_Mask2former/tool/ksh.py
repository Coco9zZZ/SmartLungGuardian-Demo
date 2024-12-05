import cv2
import os

def process_image(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 检查图像是否成功加载
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # 创建掩膜，所有非黑色像素设置为白色（255），黑色像素设置为黑色（0）
    mask = cv2.inRange(image, (1, 1, 1), (255, 255, 255))

    # 创建一个全白的图像
    white_image = cv2.bitwise_not(cv2.inRange(image, (0, 0, 0), (0, 0, 0)))

    # 将掩膜应用到图像上，使非黑色像素变为白色
    result = cv2.bitwise_or(mask, white_image)

    # 保存处理后的图像
    cv2.imwrite(output_path, result)
    print(f"Processed and saved image: {output_path}")

def process_directory(input_dir):
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        # 构造完整的文件路径
        file_path = os.path.join(input_dir, filename)

        # 检查文件是否为图像文件（根据文件扩展名）
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 处理图像并保存到同一目录，保持文件名不变
            process_image(file_path, file_path)

if __name__ == "__main__":
    input_directory = r"D:\test\mmlab\mmseg\data\Covid\Covid_MedSeg_1\val\ksh_xr\up"  # 替换为你的文件夹路径
    process_directory(input_directory)
