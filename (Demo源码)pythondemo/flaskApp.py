import time
import uuid
import zipfile
import xxxxxtest
from flask import Flask, request

from result.result import Result
from pathlib import Path
import os
from mmseg.apis import inference_model, init_model, show_result_pyplot
from config.config import Config

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'Hello'


@app.route('/upload', methods=['POST','GET'])
def upload():
    file = request.files['file']

    extension = file.filename.rsplit('.', 1)[1].lower()
    if extension not in Config.ALLOWED_EXTENSIONS:
        return Result.error('文件类型不允许')

    if extension == 'zip':
        files_name = []
        files1_name = []
        extract_to_path = app.config['UPLOAD_FOLDER']
        # 确保目标文件夹存在，如果不存在则创建
        os.makedirs(extract_to_path, exist_ok=True)

        # 打开ZIP文件
        with zipfile.ZipFile(file, 'r') as zip_ref:
            # 遍历ZIP文件中的所有文件
            for i, file_info in enumerate(zip_ref.infolist()):
                # 获取原始文件的扩展名
                file_extension = os.path.splitext(file_info.filename)[-1]

                # 定义新的文件名（例如，使用UUID）
                new_filename = f'{uuid.uuid4()}{file_extension}'

                # 生成完整的输出路径
                new_file_path = os.path.join(extract_to_path, new_filename)

                # 打开并读取ZIP中的文件内容
                with zip_ref.open(file_info) as source_file:
                    # 将内容写入新的文件
                    with open(new_file_path, 'wb') as target_file:
                        files_name.append(new_file_path)
                        target_file.write(source_file.read())
                        files1_name.append(handle1(new_file_path))
        zip_name = str(time.time()).replace('.', '')
        create_zip(files_name, files1_name, zip_name)
        get_url = f"http://{app.config['HOST']}:{app.config['PORT']}/static/zips/{zip_name}.zip".replace('\'','')
        files_name2 =[]
        for name in files_name:
            files_name2.append(f"http://{app.config['HOST']}:{app.config['PORT']}/{name}")
        return Result.success([files_name2,get_url])
        #return Result.success([get_url])

    else:
        # 生成唯一文件名并保存文件
        filename = f'{uuid.uuid4()}.{extension}'
        file_path = app.config['UPLOAD_FOLDER'] + '/' + filename
        file.save(file_path)
        relative_path = Path(file_path).relative_to("./")
        # url_path = f"http://{socket.gethostbyname(socket.gethostname())}:{app.config['PORT']}/{relative_path}"
        url_path = [f"http://{app.config['HOST']}:{app.config['PORT']}/{relative_path}"]
        #print('-----------------'+relative_path+'------------------------')
        # 处理文件
        zip_name = str(time.time()).replace('.', '')
        #create_zip([f'{relative_path}'], [f'{relative_path}'], zip_name)
        result = handle1(str(relative_path))
        create_zip([f'{relative_path}'], [result], zip_name)
        return Result.success([url_path,f"http://{app.config['HOST']}:{app.config['PORT']}/static/zips/{zip_name}.zip"])


def create_zip(origin_files, result_files, output_filename):
    # 确保目标文件夹存在
    output_dir = 'static/zips'
    os.makedirs(output_dir, exist_ok=True)

    # 创建ZIP文件的完整路径
    zip_path = os.path.join(output_dir, output_filename + '.zip')

    # 创建ZIP文件
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # 添加 origin_files 到压缩包中的 origin 文件夹
        for file in origin_files:
            if os.path.isfile(file):
                zipf.write(file, os.path.join('origin', os.path.basename(file)))

        # 添加 result_files 到压缩包中的 result 文件夹
        for file in result_files:
            if os.path.isfile(file):
                zipf.write(file, os.path.join('result', os.path.basename(file)))

    print(f"压缩包已创建，路径为: {zip_path}")
    return zip_path

#图像融合
def handle1(image_name):
    # image_name = request.args.get('imageName')
    # image_path =  r'C:\Users\ISLab_Stu\Desktop\pythonProject-xx\static\upload' + image_name
    result = inference_model(model, image_name)
    result_path = image_name.replace('upload', 'output')
    show_result_pyplot(model, image_name, result, show=False, withLabels=False, out_file=result_path, opacity=1)

    # return result_path
    return xxxxxtest.overlay_image(image_name, result_path, 'static/output')


if __name__ == '__main__':
    # 初始化模型
    #checkpoint_file = r'D:\BaiduNetdiskDownload\MeSeg1\lung_swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512\iter_28000.pth'
    checkpoint_file = r'model/best_mIoU_iter_6000.pth'
    config_file = r'model/mask2former_swin-t_8xb2-160k_ade20k-512x512_datasetA.py'
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    app.config.from_object(Config)
    app.run(debug=True, host=Config.HOST, port=Config.PORT)
