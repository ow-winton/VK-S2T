
import cv2
import numpy as np
import mediapipe as mp
import os
import csv
from mypackage import framedownsample as ds
import shutil



def mp_draw2raw(input_dir, output_dir):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)


    # 处理每个jpg文件
    for root, dirs, files in os.walk(input_dir):
        for img_file in files:
            if img_file.endswith('.jpg'):
                image_path = os.path.join(root, img_file)  # 使用root替代input_dir来生成正确的路径
                image = cv2.imread(image_path)
                if image is None:
                    print(f"图像未加载: {image_path}")
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image,
                                              results.pose_landmarks,
                                              mp_pose.POSE_CONNECTIONS,
                                              landmark_drawing_spec=landmark_drawing_spec,  # 使用自定义关键点样式
                                               connection_drawing_spec=connection_drawing_spec  )
                    rel_path = os.path.relpath(root, input_dir)
                    output_folder = os.path.join(output_dir, rel_path)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    new_filename = os.path.splitext(img_file)[0] + '_drawn.jpg'
                    output_path = os.path.join(output_folder, new_filename)
                    cv2.imwrite(output_path, image)
                   #print(f"Processed image saved to {output_path}")
                else:
                    print("未检测到任何关键点。")
### 方法二：在纯黑背景上绘制关键点
def mp_draw2blk_bg(input_dir, output_dir):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    # 处理每个jpg文件
    for root, dirs, files in os.walk(input_dir):
        for img_file in files:
            if img_file.endswith('.jpg'):
                image_path = os.path.join(root, img_file)  # 使用root替代input_dir来生成正确的路径

                image = cv2.imread(image_path)
                if image is None:
                    print(f"图像未加载: {image_path}")
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    height, width, _ = image.shape
                    black_image = np.zeros((height, width, 3), dtype=np.uint8)
                    mp_drawing.draw_landmarks(
                        black_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=landmark_drawing_spec,  # 使用自定义关键点样式
                        connection_drawing_spec=connection_drawing_spec
                    )
                    rel_path = os.path.relpath(root, input_dir)
                    output_folder = os.path.join(output_dir, rel_path)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    new_filename = os.path.splitext(img_file)[0] + '_black.jpg'
                    output_path = os.path.join(output_folder, new_filename)
                    cv2.imwrite(output_path, black_image)
                    #print(f"Processed image saved to {output_path}")

                else:
                    print("未检测到任何关键点。")

def save_keypoints(input_dir, output_dir):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                        enable_segmentation=False, min_detection_confidence=0.5)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有图片
    for root, dirs, files in os.walk(input_dir):
        for img_file in files:
            if img_file.endswith('.jpg'):
                img_path = os.path.join(root, img_file)  # 使用root替代input_dir来生成正确的路径

                image = cv2.imread(img_path)
                if image is None:
                    print(f"图像未加载: {img_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # 准备保存关键点数据的 CSV 文件
                rel_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, rel_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                new_filename = os.path.splitext(img_file)[0] + '_keypoints.csv'
                csv_path = os.path.join(output_folder, new_filename)

                with open(csv_path, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['id', 'x', 'y', 'z', 'visibility'])

                    if results.pose_landmarks:
                        for id, landmark in enumerate(results.pose_landmarks.landmark):
                            # 将关键点的 x, y 坐标转换为图像尺寸的实际位置
                            x = landmark.x * image.shape[1]
                            y = landmark.y * image.shape[0]
                            z = landmark.z
                            visibility = landmark.visibility
                            csvwriter.writerow([id, x, y, z, visibility])
                    else:
                        print(f"No keypoints detected in {img_file}.")
                print(f"Keypoints for {img_file} saved to {csv_path}")
def find_matching_files(parent_dir, child_dir):
    child_files = {file for file in os.listdir(child_dir)}
    parent_files = {file for file in os.listdir(parent_dir)}
    matching_files = parent_files.intersection(child_files)
    return matching_files
def process_matching_files(parent_dir, child_dir, target_dir):
    # 获取子目录中所有文件的文件名（不包括路径）
    matching_files = find_matching_files(parent_dir, child_dir)
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # 复制匹配的文件到目标目录
    for file_name in matching_files:
        full_file_path = os.path.join(parent_dir, file_name)
        target_file_path = os.path.join(target_dir, file_name)
        shutil.copy(full_file_path, target_file_path)  # 复制文件
        print(f"Copied {file_name} to {target_dir}")
def clip_exist_kpvideo( keypoint_folder,video_path,target_dir,output_base_folder,downsample_rate):
    process_matching_files(parent_dir=keypoint_folder,child_dir=video_path,target_dir=target_dir)
    ds.process_dataset(input_folder=target_dir,output_base_folder=output_base_folder,downsample_rate=downsample_rate)