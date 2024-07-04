import cv2
import numpy as np
import os
#t1 for testdataset,tr for traindataset，notice that tv is for validation
def get_all_file_paths(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths
def to1(color_image_path,segmentation_image_path,tag):#这个函数用于转换至半二值函数
    #读取图片
    color_image = cv2.imread(color_image_path)
    segmentation_image = cv2.imread(segmentation_image_path, cv2.IMREAD_GRAYSCALE)
    save_path = 'C:/python/UI/result1.png'
    # 将黑白分割图转换为二值图像
    _, binary_mask = cv2.threshold(segmentation_image, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)  # 将掩码转换为 8 位无符号整数类型

    # 创建一个全黑的背景图像
    background = np.zeros_like(color_image)

    # 将二值图像作为掩码，将彩色图像中的前景部分提取出来
    foreground = cv2.bitwise_and(color_image, color_image, mask=binary_mask)

    # 将前景部分与背景合并
    res = cv2.add(background, foreground)
    #保存图片
    if tag==0:
        cv2.imwrite(save_path, res)
    elif tag==1:
        cv2.imwrite('./tv/'+os.path.basename(segmentation_image_path),res)
    return save_path

  #这个函数用于将图片识别的部分用淡红色框住，但整体仍然保留彩色
def to2(color_image_path, segmentation_image_path):
    # 读取彩色图片和分割图像
    color_image = cv2.imread(color_image_path)
    segmentation_image = cv2.imread(segmentation_image_path, cv2.IMREAD_GRAYSCALE)

    # 将黑白分割图转换为二值图像
    _, binary_mask = cv2.threshold(segmentation_image, 127, 255, cv2.THRESH_BINARY)

    # 创建一个淡红色的图层
    red_layer = np.zeros_like(color_image, dtype=np.uint8)
    red_layer[:, :, 2] = 255  # 设置红色通道为255（完全红色）

    # 将淡红色图层与二值掩码相乘以产生淡红色的遮罩
    red_mask = cv2.bitwise_and(red_layer, red_layer, mask=binary_mask)

    # 将淡红色遮罩与彩色图像相加以突出显示物体
    highlighted_image = cv2.addWeighted(color_image, 1, red_mask, 0.5, 0)

    # 保存结果图像
    save_path = 'C:/python/UI/result2.png'
    cv2.imwrite(save_path, highlighted_image)
    return save_path


if __name__ == '__main__':
    #导入文件夹路径
    print("hi")
    folder_path = './SINetV2/res3'  # 替换为你的文件夹路径
    all_file_paths = get_all_file_paths(folder_path)
    i=0
    for file_path in all_file_paths:
        i=i+1
        #要除去前缀和尾缀
        basename = os.path.basename(file_path)
        filename, file_extension = os.path.splitext(basename)

        rt=to1('./SINetV2/CAMO/image/'+filename+'.jpg',file_path,1)
        print('picutre{}already processed'.format(i))
