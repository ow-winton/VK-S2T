

import cv2
import os
import numpy as np

# 指定文件夹路径
folder_path = r'D:\VK-S2T\data\output\_DLMidC-b8w_27-8-rgb_front'

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 创建一个字典来存储每张图片中满足条件的像素数
result = {}

for image_file in image_files:
    # 读取图片
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    # 检查R通道值大于100的像素
    r_channel = image[:, :, 0]
    count = np.sum(r_channel > 0)

    # 保存结果
    result[image_file] = count

# 打印结果
for image_file, count in result.items():
    print(f"{image_file}: {count} pixels with R channel value > 100")

