import os
import pandas as pd
import json
import csv

def collect_frame_path(output_base_folder,datatype):
    video_frames = {}



    for video_name in os.listdir(output_base_folder):
        video_folder = os.path.join(output_base_folder, video_name)
        if os.path.isdir(video_folder):
            frames = sorted(
                [os.path.join(video_folder, frame) for frame in os.listdir(video_folder) if frame.endswith('.jpg')])
            # 将绝对路径转换为相对路径并添加必要的前缀

            prefix = os.path.join(f'{datatype}')
            relative_frames = [os.path.join(prefix, os.path.relpath(frame, output_base_folder)).replace('\\', '/') for frame in frames]
            video_frames[video_name] = relative_frames

    return video_frames

def collect_frame_paths(output_base_folder):
    datatype = input("数据类型 train,dev,test  ")
    if datatype == 'train':
        video_frames = collect_frame_path(output_base_folder,datatype)
    elif datatype == 'dev':
        video_frames = collect_frame_path(output_base_folder,datatype)
    elif datatype == 'test':
        video_frames = collect_frame_path(output_base_folder,datatype)
    else:
        print("error ,终止程序")

    return video_frames

def process_frames_and_text(output_base_folder, csv_file):
    # 读取 CSV 文件，指定分隔符为制表符
    data = pd.read_csv(csv_file, sep='\t')

    video_frames = collect_frame_paths(output_base_folder)
    output_data = []
    shujuji_type = input("输入数据集类型train，dev或者test")
    for video_name, frames in video_frames.items():
        # 尝试找到对应的文本信息
        matching_rows = data.loc[data['SENTENCE_NAME'] == video_name, 'SENTENCE']
        if len(matching_rows) == 0:
            print(f"Warning: No matching record found for video {video_name}")
            continue

        text_info = matching_rows.values[0]

        if shujuji_type == "train":
            video_name_with_prefix = os.path.join('train', video_name).replace('\\', '/')  # 使用 os.path.join 添加前缀
        elif shujuji_type == "dev":
            video_name_with_prefix = os.path.join('dev', video_name).replace('\\', '/')
        elif shujuji_type == "test":
            video_name_with_prefix = os.path.join('test', video_name).replace('\\', '/')
        else:
            print("输入错误，请终止")
        keypoint_paths = [frame.replace(shujuji_type, f'key/{shujuji_type}').replace('.jpg', '.png') for frame in
                          frames]

        # 创建键值对
        entry = {
            'Key': video_name_with_prefix,
            'Value': {
                'name': video_name_with_prefix,
                'gloss': '',  # 如果有其他信息需要添加在这里
                'text': text_info,
                'length': len(frames),
                'imgs_path': frames,
                'keypoint_path': keypoint_paths
            }
        }
        output_data.append(entry)

    return output_data



def save_to_csv(output_data, output_csv):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_csv, 'w', encoding='utf-8') as f:
        for entry in output_data:
            key = entry['Key']
            value = json.dumps(entry['Value'], ensure_ascii=False)
            f.write(f'{key},{value}\n')
# 示例用法


def biaotou(csv_file_path):
    # 用于存储展开后的数据
    data = []

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            # 检查行中是否包含正确数量的字段（假设应为2个字段）
            if len(row) >= 2:
                # 将多余的部分合并到第二个字段中
                key = row[0]
                value = ','.join(row[1:])
                data.append([key, value])
            else:
                print(f"Skipping malformed row: {row}")
    # 将数据转换为DataFrame并添加表头
    df = pd.DataFrame(data, columns=['Key', 'Value'])

    # 保存带表头的文件
    df.to_csv(csv_file_path, index=False, encoding='utf-8')


