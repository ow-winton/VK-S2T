import cv2
import os
import numpy as np
import json
import re


def load_keypoint(json_file, type="all"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    keypoints = data['people'][0]['pose_keypoints_2d']
    face_keypoints = data['people'][0]['face_keypoints_2d']
    hand_left_keypoints = data['people'][0]['hand_left_keypoints_2d']
    hand_right_keypoints = data['people'][0]['hand_right_keypoints_2d']

    keypoints = [int(round(keypoint)) if idx % 3 != 2 else keypoint for idx, keypoint in enumerate(keypoints)]
    face_keypoints = [int(round(keypoint)) if idx % 3 != 2 else keypoint for idx, keypoint in enumerate(face_keypoints)]
    hand_left_keypoints = [int(round(keypoint)) if idx % 3 != 2 else keypoint for idx, keypoint in
                           enumerate(hand_left_keypoints)]
    hand_right_keypoints = [int(round(keypoint)) if idx % 3 != 2 else keypoint for idx, keypoint in
                            enumerate(hand_right_keypoints)]

    kpoints = []
    if type == "all":
        for i in range(0, len(keypoints), 3):
            kpoints.append([int(keypoints[i]), int(keypoints[i + 1]), keypoints[i + 2]])
        for i in range(0, len(face_keypoints), 3):
            kpoints.append([int(face_keypoints[i]), int(face_keypoints[i + 1]), face_keypoints[i + 2]])
        for i in range(0, len(hand_left_keypoints), 3):
            kpoints.append([int(hand_left_keypoints[i]), int(hand_left_keypoints[i + 1]), hand_left_keypoints[i + 2]])
        for i in range(0, len(hand_right_keypoints), 3):
            kpoints.append(
                [int(hand_right_keypoints[i]), int(hand_right_keypoints[i + 1]), hand_right_keypoints[i + 2]])
    elif type == "hand":
        for i in range(0, len(hand_left_keypoints), 3):
            kpoints.append([int(hand_left_keypoints[i]), int(hand_left_keypoints[i + 1]), hand_left_keypoints[i + 2]])
        for i in range(0, len(hand_right_keypoints), 3):
            kpoints.append(
                [int(hand_right_keypoints[i]), int(hand_right_keypoints[i + 1]), hand_right_keypoints[i + 2]])
    return kpoints

# 四维

# def imgto4d(file_path,filtered_keypoints):
#     # 假设原始图像是 image_rgb，关键点是 filtered_keypoints
#
#     image_rgb = cv2.imread(file_path)
#
#     image_with_keypoints = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
#     image_with_keypoints[:, :, :3] = image_rgb
#
#     # for x, y, confidence in filtered_keypoints:
#     #
#     #     # 检查关键点的位置是否在图像范围内
#     #     if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
#     #         image_with_keypoints[y, x, 3] = 1
#     #
#     # print(image_with_keypoints[678, 790, 3])  # 结果 1
#     for x, y, confidence in filtered_keypoints:
#
#         # 检查关键点的位置是否在图像范围内
#         if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
#             image_with_keypoints[y, x, 3] = int(confidence * 255)
#     # print(image_with_keypoints[678, 790, 3])  # 结果132
#
#             print(int(confidence * 255), end=" ")
#     print()
#     return image_with_keypoints
#半径
def imgto4d(file_path, filtered_keypoints, radius=5, alpha=0.5):
    """
    将图像文件路径读取成RGB图像，并在关键点周围进行增强。
    参数:
    - file_path: 图像文件的路径。
    - filtered_keypoints: 关键点列表，每个关键点是 (x, y, confidence) 形式。
    - radius: 增强区域的半径，默认为 5。
    - alpha: 增强强度的系数，默认为 0.5。

    返回:
    - image_with_keypoints: 增强后的图像。
    """
    image_rgb = cv2.imread(file_path)
    if image_rgb is None:
        print(f"Failed to load image: {file_path}")
        return None

    # 初始化图像
    image_with_keypoints = image_rgb / 15

    # 遍历关键点
    for x, y, confidence in filtered_keypoints:
        if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
            conf = int(confidence * 200)

            # 遍历半径内的像素
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if 0 <= x + i < image_rgb.shape[1] and 0 <= y + j < image_rgb.shape[0]:
                        distance = np.sqrt(i ** 2 + j ** 2)
                        if distance <= radius:
                            # 增强附近的像素值
                            image_with_keypoints[y + j, x + i] += alpha * conf
                            image_with_keypoints[y + j, x + i] = np.clip(image_with_keypoints[y + j, x + i], 0, 255)

    return image_with_keypoints


# 高斯
# def imgto4d(file_path, filtered_keypoints, sigma=5, alpha=0.5):
#     """
#     将图像文件路径读取成RGB图像，并在关键点周围进行高斯模糊增强。
#     参数:
#     - file_path: 图像文件的路径。
#     - filtered_keypoints: 关键点列表，每个关键点是 (x, y, confidence) 形式。
#     - sigma: 高斯模糊的标准差，默认为 5。
#     - alpha: 增强强度的系数，默认为 0.5。
#
#     返回:
#     - image_with_keypoints: 增强后的图像。
#     """
#     image_rgb = cv2.imread(file_path)
#     if image_rgb is None:
#         print(f"Failed to load image: {file_path}")
#         return None
#
#     # 初始化图像
#     image_with_keypoints = image_rgb / 10
#
#     for x, y, confidence in filtered_keypoints:
#         if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
#             conf = int(confidence * 255)
#
#             # 创建一个与图像相同大小的空白高斯分布图
#             gaussian = np.zeros(image_rgb.shape[:2], dtype=np.float32)
#             cv2.circle(gaussian, (x, y), sigma * 3, conf, -1)  # 使用 sigma*3 作为半径覆盖大部分高斯分布
#             gaussian = cv2.GaussianBlur(gaussian, (0, 0), sigma)
#
#             # 叠加高斯分布到图像
#             for c in range(3):
#                 image_with_keypoints[:, :, c] += alpha * gaussian
#                 image_with_keypoints[:, :, c] = np.clip(image_with_keypoints[:, :, c], 0, 255)
#
#     return image_with_keypoints





# 单个图片生成对应的四维数据
# def imgto4d(file_path, filtered_keypoints, kernel_size=5, alpha=0.5):
#     """
#     将图像文件路径读取成RGB图像，并在关键点周围进行卷积核增强。
#     参数:
#     - file_path: 图像文件的路径。
#     - filtered_keypoints: 关键点列表，每个关键点是 (x, y, confidence) 形式。
#     - kernel_size: 卷积核的大小，默认为 5。
#     - alpha: 增强强度的系数，默认为 0.5。
#
#     返回:
#     - image_with_keypoints: 增强后的图像。
#     """
#     image_rgb = cv2.imread(file_path)
#     if image_rgb is None:
#         print(f"Failed to load image: {file_path}")
#         return None
#
#     # 初始化图像
#     image_with_keypoints = image_rgb / 10
#
#     # 创建卷积核
#     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
#
#     for x, y, confidence in filtered_keypoints:
#         if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
#             conf = int(confidence * 255)
#             for c in range(3):
#                 region = image_with_keypoints[
#                          max(0, y - kernel_size // 2):min(image_rgb.shape[0], y + kernel_size // 2 + 1),
#                          max(0, x - kernel_size // 2):min(image_rgb.shape[1], x + kernel_size // 2 + 1),
#                          c]
#                 filtered = cv2.filter2D(region, -1, kernel)
#                 filtered += alpha * conf
#                 filtered = np.clip(filtered, 0, 255)
#                 image_with_keypoints[max(0, y - kernel_size // 2):min(image_rgb.shape[0], y + kernel_size // 2 + 1),
#                 max(0, x - kernel_size // 2):min(image_rgb.shape[1], x + kernel_size // 2 + 1),
#                 c] = filtered
#
#     return image_with_keypoints


def process_folder(video_name, image_folder, keypoints_folder, output_folder,type="all"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])

    for image_file in image_files:
        # 提取帧编号
        frame_number = int(os.path.splitext(image_file)[0].split('_')[-1])#downsample_Rate
        image_path = os.path.join(image_folder, image_file)
        json_filename = os.path.join(keypoints_folder, f'{video_name}_{frame_number:012d}_keypoints.json')

        if not os.path.exists(json_filename):
            print("JSON file does not exist:", json_filename)
            continue


        keypoints = load_keypoint(json_filename,type)
        frame_with_keypoints = imgto4d(image_path, keypoints)

        outputfile = os.path.splitext(image_file)[0] + '.jpg'
        output_path = os.path.join(output_folder, outputfile)
        # outputfile = os.path.join(output_folder, f'{video_name}_{frame_number:012d}_keypoints')
      # outputfile = os.path.join(output_folder, f"{video_name}_{frame_number:012d}.png")

        cv2.imwrite(output_path, frame_with_keypoints)



def process_image_to_4d(base_image_path, base_keypoints_path, base_output_path):
    type = input("type")
    image_folders = sorted([d for d in os.listdir(base_image_path) if os.path.isdir(os.path.join(base_image_path, d))])
    for folder in image_folders:
        video_name = folder  # 使用文件夹名称作为视频名称
        image_folder = os.path.join(base_image_path, folder)
        keypoints_folder = os.path.join(base_keypoints_path, folder)
        output_folder = os.path.join(base_output_path, folder)
        process_folder(video_name, image_folder, keypoints_folder, output_folder,type)
