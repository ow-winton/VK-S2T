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


# 单个图片生成对应的四维数据
def imgto4d(file_path,filtered_keypoints):
    # 假设原始图像是 image_rgb，关键点是 filtered_keypoints

    image_rgb = cv2.imread(file_path)

    image_with_keypoints = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
    image_with_keypoints[:, :, :3] = image_rgb

    # for x, y, confidence in filtered_keypoints:
    #
    #     # 检查关键点的位置是否在图像范围内
    #     if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
    #         image_with_keypoints[y, x, 3] = 1
    #
    # print(image_with_keypoints[678, 790, 3])  # 结果 1
    for x, y, confidence in filtered_keypoints:

        # 检查关键点的位置是否在图像范围内
        if 0 <= x < image_rgb.shape[1] and 0 <= y < image_rgb.shape[0]:
            image_with_keypoints[y, x, 3] = int(confidence * 255)
    # print(image_with_keypoints[678, 790, 3])  # 结果132

            print(int(confidence * 255), end=" ")
    print()
    return image_with_keypoints


def process_folder(video_name, image_folder, keypoints_folder, output_folder,type="all"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])

    for image_file in image_files:
        # 提取帧编号
        frame_number = int(os.path.splitext(image_file)[0].split('_')[-1])*5#downsample_Rate
        image_path = os.path.join(image_folder, image_file)
        json_filename = os.path.join(keypoints_folder, f'{video_name}_{frame_number:012d}_keypoints.json')

        if not os.path.exists(json_filename):
            print("JSON file does not exist:", json_filename)
            continue


        keypoints = load_keypoint(json_filename,type)
        frame_with_keypoints = imgto4d(image_path, keypoints)

        outputfile = os.path.join(output_folder, f'{video_name}_{frame_number:012d}_keypoints')
        np.save(outputfile, frame_with_keypoints)


def process_image_to_4d(base_image_path, base_keypoints_path, base_output_path):
    type = input("type")
    image_folders = sorted([d for d in os.listdir(base_image_path) if os.path.isdir(os.path.join(base_image_path, d))])
    for folder in image_folders:
        video_name = folder  # 使用文件夹名称作为视频名称
        image_folder = os.path.join(base_image_path, folder)
        keypoints_folder = os.path.join(base_keypoints_path, folder)
        output_folder = os.path.join(base_output_path, folder)
        process_folder(video_name, image_folder, keypoints_folder, output_folder,type)

