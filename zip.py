import zipfile
import os

def zip_folder(folder_path, zip_path):
    """
    压缩文件夹到指定的zip文件中
    :param folder_path: 要压缩的文件夹路径
    :param zip_path: 压缩后的zip文件路径
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)

# 调用函数压缩文件夹
folder_to_zip = './t1'
output_zip_path = 'test.zip'
zip_folder(folder_to_zip, output_zip_path)