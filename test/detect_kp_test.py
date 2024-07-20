import torch
import cv2
import os
from torchvision.transforms import ToTensor


def detect_keypoints(x):
    keypoints = []
    non_zero = (x != 0).any(dim=1)
    batch_indices, rows, cols = torch.nonzero(non_zero, as_tuple=True)
    keypoints = list(zip(batch_indices.tolist(), rows.tolist(), cols.tolist()))
    return keypoints


def process_image(image_path):
    # 加载图像，转换为RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = ToTensor()(img).unsqueeze(0)

    # 检测关键点
    keypoints = detect_keypoints(tensor)
    return len(keypoints)


# 遍历指定文件夹内的所有图像文件并计算关键点总数
def process_single_directory(directory):
    total_keypoints = 0
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # 确保只处理图像文件
            image_path = os.path.join(directory, filename)
            num_keypoints = process_image(image_path)
            total_keypoints += num_keypoints
    return total_keypoints


# 指定文件夹路径
directory_path = r"F:\test_train\how2sign\frame\key\train\00kppw3aqus_3-3-rgb_front"  # 请确保路径是正确的
total_keypoints = process_single_directory(directory_path)
print(f"Total number of keypoints in the folder: {total_keypoints}")
