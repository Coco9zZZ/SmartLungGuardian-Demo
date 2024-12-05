import os
from mmseg.apis import inference_model, init_model, show_result_pyplot


def run():
    # 初始化模型
    checkpoint_file = r'D:\BaiduNetdiskDownload\MeSeg1\lung_swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512\iter_28000.pth'
    config_file = r'D:\BaiduNetdiskDownload\MeSeg1\lung_swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512\lung_swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    # 输入文件夹路径
    input_folder = r'D:\test\mmlab\mmseg\data\Covid\Covid_MedSeg_1\val\images'
    # 输出文件夹路径
    output_folder = r'D:\test\mmlab\mmseg\data\Covid\Covid_MedSeg_1\val\ksh\Swin'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的每一张图片
    for img_name in os.listdir(input_folder):
        if img_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(input_folder, img_name)
            # 读取并推理
            print(img_path + '--------------------')
            result = inference_model(model, img_path)
            # 输出结果路径
            out_file = os.path.join(output_folder, img_name)
            # 保存可视化结果
            show_result_pyplot(model, img_path, result, show=False, withLabels=False, out_file=out_file, opacity=1)
            print(f'Processed {img_name} and saved result to {out_file}')


if __name__ == '__main__':
    run()
