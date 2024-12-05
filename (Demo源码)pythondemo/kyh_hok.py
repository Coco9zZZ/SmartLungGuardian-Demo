from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

checkpoint_file = r'D:\BaiduNetdiskDownload\MeSeg2\lung_segformer_mit-b0_8xb2-160k_ade20k-512x512\iter_24000.pth'
config_file= r'D:\BaiduNetdiskDownload\MeSeg2\lung_segformer_mit-b0_8xb2-160k_ade20k-512x512\lung_segformer_mit-b0_8xb2-160k_ade20k-512x512.py'


# 根据配置文件和模型文件建立模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 在单张图像上测试并可视化
img = r'D:\test\mmlab\mmseg\data\Covid\Covid_MedSeg_1\val\images\bjorke_100.png'  # or img = mmcv.imread(img), 这样仅需下载一次
result = inference_model(model, img)
# 在新的窗口可视化结果
show_result_pyplot(model, img, result, show=True,withLabels=False,out_file='result.jpg',opacity=1)