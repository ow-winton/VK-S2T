import cv2
from PIL import Image
import numpy as np

# 读取图像路径
image_path = 'frame_0000.jpg'

# 使用 OpenCV 读取图像
img_bgr = cv2.imread(image_path)

# 转换为 RGB 格式
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 打印转换前部分像素的RGB值
print("Before PIL conversion (RGB values):")
print(img_rgb[0:5, 0:5, :])

# 将 NumPy 数组转换为 PIL 图像
img_pil = Image.fromarray(img_rgb)

# 将 PIL 图像转换回 NumPy 数组
img_back_to_numpy = np.array(img_pil)

# 打印转换回 NumPy 数组后部分像素的RGB值
print("\nAfter PIL conversion (RGB values):")
print(img_back_to_numpy[0:5, 0:5, :])

# 验证两者是否相同
if np.array_equal(img_rgb, img_back_to_numpy):
    print("\nThe RGB values are unchanged after converting to and from PIL.")
else:
    print("\nThe RGB values have changed after converting to and from PIL.")
