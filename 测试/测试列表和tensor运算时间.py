import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import time
import numpy as np

# 定义图像变换
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 示例图像路径列表
paths = ['frame_0000.jpg', 'norm_0000.jpg']

# 测试使用列表存储和处理图像
batch_image = []
start_time = time.time()
for img_path in paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    batch_image.append(data_transform(img))
batch_tensor = torch.stack(batch_image)
end_time = time.time()
list_time = end_time - start_time

# 打印列表方式的时间
print(f"Time using list: {list_time:.6f} seconds")

# 测试直接使用张量存储和处理图像
start_time = time.time()
imgs = torch.zeros(len(paths), 3, 224, 224)
for i, img_path in enumerate(paths):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tensor = data_transform(img)
    imgs[i] = img_tensor
end_time = time.time()
tensor_time = end_time - start_time

# 打印张量方式的时间
print(f"Time using tensor: {tensor_time:.6f} seconds")

# 比较两种方法的时间差异
print(f"Speedup: {list_time / tensor_time:.2f}x")
