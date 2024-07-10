import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms

image_path = 'norm_0000.jpg'

keypoints = [(50, 60), (80, 90), (120, 140)]

image = Image.open(image_path)
plt.subplot(1, 2, 1)
plt.imshow(image)
for x, y in keypoints:
    plt.scatter(x, y, c='red', s=10)
plt.title('Original Image with Keypoints')




image = Image.open(image_path).convert('RGB')

# 转换为numpy数组并打印部分原始RGB矩阵
image_np = np.array(image)
print("Original RGB values (part of the image):")
print(image_np[0:5, 0:5, :])  # 打印图像前5x5个像素的RGB值


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image_tensor = data_transform(image)


image_normalized_np = image_tensor.permute(1, 2, 0).numpy()
print("Normalized RGB values (part of the image):")
print(image_normalized_np[0:5, 0:5, :])  # 打印归一化后前5x5个像素的RGB值

# 结论 ： transformer.normalize 本身就是w1 *， 而且它是大模型上训练得出的
# 归一化不影响 rgb