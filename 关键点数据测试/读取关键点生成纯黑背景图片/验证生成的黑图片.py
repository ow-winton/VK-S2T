import cv2
import numpy as np

def print_non_zero_points(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # 查找所有非零点
    non_zero_indices = np.argwhere(image[:, :, 0] )
    for y, x in non_zero_indices:
        value = image[y, x]
        print(f"Coordinate: ({x}, {y}), Value: {value}")

# 设置已生成图像的路径
image_path = r'D:\VK-S2T\data\output\test\_g0fpC8aiME_6-8-rgb_front\frame_0001.png'

print_non_zero_points(image_path)
img= cv2.imread(image_path)
print(img.shape)
