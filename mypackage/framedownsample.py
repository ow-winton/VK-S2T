import cv2
import os

# 视频分割成帧并且down sample
def extract_frames_from_video(video_path, output_base_folder, downsample_rate=3):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_base_folder, video_name)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}.")
        return

    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有帧了就退出循环

        # 只保存每第 downsample_rate 帧
        if frame_count % downsample_rate == 0:
            frame_file = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            if frame is not None and frame.size > 0:
                success = cv2.imwrite(frame_file, frame)
                # print(f"Attempt to save frame {saved_frame_count}: {success}")
                if not success:
                    print(f"Error: Failed to save frame {saved_frame_count}")
                # else:
                    # print(f"Saved frame {saved_frame_count} to {frame_file}")
                    # print(f"Data type: {frame.dtype}")  # 显示数据类型
                    # print(f"Image dimensions: {frame.shape}")  # 显示图像维度
                saved_frame_count += 1
            else:
                print(f"Warning: Frame {frame_count} is empty or corrupted.")

        frame_count += 1

    cap.release()
    # print(f"Extracted {saved_frame_count} frames from {video_name} and saved in {output_folder}")



def process_dataset(input_folder, output_base_folder, downsample_rate=3):
    # 遍历数据集文件夹中的所有视频文件
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                extract_frames_from_video(video_path, output_base_folder, downsample_rate)
def calculate_dataset_amount(input_folder):
    # 遍历数据集文件夹中的所有视频文件
    n = 0
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                n+=1
    return n

def calculate_out_amount(input_folder):
    # 遍历数据集文件夹中的所有文件夹数量
    n = 0
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.jpg')):
                n+=1
    return  n



