import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
class rgb_fusion(nn.Module):
    def __init__(self, frozen=False):
        super(rgb_fusion, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))  # 设置初始值为0.5
        if frozen:
            self.alpha.requires_grad = False

    def forward(self, x1, x2):
        alpha_expanded = self.alpha.view(1, -1, 1, 1)  # 调整alpha形状以匹配输入
        out = x1 + alpha_expanded * x2
        return out

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image

def show_image(image_tensor, title=""):
    image = image_tensor.numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 加载示例图像
image1_path = r'D:\VK-S2T\data\frame\train\0ACeI3_jN3k_6-8-rgb_front\frame_0000.jpg'
image2_path = r'D:\VK-S2T\data\output\train\0ACeI3_jN3k_6-8-rgb_front\frame_0000.png'
image1 = load_image(image1_path, transform)
image2 = load_image(image2_path, transform)

# 确保 x2 中的关键点有实际值
image2[:, 30, 30] = torch.tensor([1.0, 0.0, 0.0])  # 设置关键点的值
image2[:, 50, 50] = torch.tensor([0.0, 1.0, 0.0])  # 设置关键点的值
# 创建模型实例
model = rgb_fusion(frozen=False)

# 将图像添加批次维度，并传递给模型
image1_batch = image1.unsqueeze(0)  # 添加批次维度
image2_batch = image2.unsqueeze(0)  # 添加批次维度

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练步骤
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(image1_batch, image2_batch)

    # 假设我们有一个目标张量，可以是任意值，这里假设为image1
    target = image1_batch
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Alpha: {model.alpha.data}")

# 前向传播
output = model(image1_batch, image2_batch)

# 去掉批次维度
output_image = output.squeeze(0)

# 将张量转换为图像
output_image_np = output_image.permute(1, 2, 0).detach().numpy()  # 转换为 (H, W, C) 并转换为 numpy 数组

# 显示图像
plt.imshow(output_image_np)
plt.axis('off')
plt.show()

# 假设关键点在特定位置，比如 (30, 30) 和 (50, 50)
keypoints = [(30, 30), (50, 50)]

# 检查关键点的值
for (row, col) in keypoints:
    print(f"关键点 ({row}, {col})")
    print("原始图像1中的像素值:", image1[:, row, col])
    print("原始图像2中的像素值:", image2[:, row, col])
    print("融合后的图像中的像素值:", output[0, :, row, col])
    print()