import subprocess
import os

def run_osformer_demo(input_file):
    # 定义OSFormer文件夹的路径
    osformer_path = 'OSFormer'  # 修改为实际路径

    # 固定配置文件和权重文件路径
    config_file = os.path.join(osformer_path, 'configs/CIS_R50.yaml')
    weights_file = os.path.join(osformer_path, 'OSFormer-Pretrain/osformer-r50.pth')

    # 输入文件夹路径
    input_file_abs = os.path.abspath(input_file)
    output_folder = os.path.dirname(input_file_abs)
    output_file = os.path.join(output_folder, 'osformer.jpg')

    # 定义要运行的脚本路径
    script_path = os.path.join(osformer_path, 'demo1.py')
    python_path = "/root/miniconda3/bin/python"
    # 构建命令
    command = [
        python_path, script_path,
        '--config-file', config_file,
        '--input', input_file_abs,
        '--output', output_file,
        '--opts', 'MODEL.WEIGHTS', weights_file
    ]

    # 运行命令
    subprocess.run(command)

if __name__ == "__main__":
    # 输入文件路径（这里可以是任何路径）
    input_file = 'sample4.jpg'  # 修改为实际路径

    # 调用函数运行demo1
    run_osformer_demo(input_file)
