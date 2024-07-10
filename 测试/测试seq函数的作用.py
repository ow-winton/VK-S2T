
import vidaug.augmentors as va
from skimage.transform import resize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 自定义增强类
class CustomRandomResize(va.RandomResize):
    def __call__(self, clip):
        new_h = int(self.scale * clip[0].shape[0])
        new_w = int(self.scale * clip[0].shape[1])
        return [np.uint8(resize(img, (new_h, new_w)) * 255) for img in clip]

# 使用自定义增强类
sometimes = lambda aug: va.Sometimes(0.5, aug)

seq = va.Sequential([
    sometimes(va.RandomRotate(30)),
    sometimes(CustomRandomResize(0.2)),
    sometimes(va.RandomTranslate(x=10, y=10)),
])

# 示例图像路径
img_path = 'frame_0000.jpg'

# 读取图像并转换为 NumPy 数组
img = Image.open(img_path).convert('RGB')
img_np = np.array(img)

# 将图像放入列表
batch_image = [img_np]

# 应用数据增强
augmented_images = seq(batch_image)

# 显示原始图像和增强后的图像
plt.figure(figsize=(10, 5))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(batch_image[0])
plt.title('Original Image')

# 显示增强后的图像
plt.subplot(1, 2, 2)
plt.imshow(augmented_images[0])
plt.title('Augmented Image')

plt.show()

# 验证输出的类型和形状
print("Original image type:", type(batch_image[0]))
print("Original image shape:", batch_image[0].shape)
print("Augmented image type:", type(augmented_images[0]))
print("Augmented image shape:", augmented_images[0].shape)
