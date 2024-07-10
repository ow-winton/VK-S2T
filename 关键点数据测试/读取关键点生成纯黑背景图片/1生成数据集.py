import cv2
import json

import os
import numpy as np
'''
{"version":1.3,"people":[{"person_id":[-1],
""hand_left_keypoints_2d":[858.567,674.237,0.473745,834.68,679.75,0.605761,809.567,688.325,0.79109,
791.192,704.25,0.840096,778.33,719.562,0.844054,814.467,720.174,0.623984,799.155,742.837,0.480212,
790.58,752.637,0.324374,784.455,762.437,0.183557,829.167,731.812,0.552808,813.855,748.962,0.346242,
804.055,763.662,0.254145,796.092,772.849,0.175303,843.867,737.937,0.466871,829.167,753.862,0.223132,
822.43,769.174,0.163297,810.18,779.587,0.138426,856.73,741.612,0.258997,843.867,758.762,0.18492,
835.292,769.174,0.152476,829.167,777.749,0.107277]

"hand_right_keypoints_2d":[533.077,669.471,0.569786,554.877,675.7,0.660163,575.432,690.026,0.66982,
586.021,709.335,0.715781,596.61,720.546,0.857545,557.369,715.563,0.6677,571.695,740.478,0.574502,
580.415,751.69,0.424673,583.529,762.279,0.26638,541.797,724.906,0.678123,553.632,746.084,0.419789,
562.352,764.147,0.304396,567.958,773.49,0.239336,526.848,726.152,0.625442,536.814,746.707,0.437199,
542.42,763.524,0.319722,551.14,775.359,0.187086,515.014,724.906,0.579241,522.488,746.084,0.31496,
526.848,759.787,0.32866,531.831,764.147,0.230654],
'''

import os
import json
import cv2
import numpy as np

import os
import json
import cv2
import numpy as np


def draw_keypoints(image, keypoints, confidence_threshold=0.1):
    for i in range(0, len(keypoints), 3):
        x, y, confidence = keypoints[i:i + 3]
        if confidence > confidence_threshold:
            scaled_x, scaled_y = int(x ), int(y )
            color = (float(confidence)*255, 0, 0)  # (B, G, R)

            if 0 <= scaled_x < 1280 and 0 <= scaled_y < 720:
                image[scaled_y, scaled_x] = color

def process_json_file(json_file, image_size):
    with open(json_file, 'r') as f:
        data = json.load(f)

    image = np.zeros((int(image_size[1] ), int(image_size[0] ), 3), dtype=np.uint8)


    for person in data['people']:

        if 'hand_left_keypoints_2d' in person:
            draw_keypoints(image, person['hand_left_keypoints_2d'])
        if 'hand_right_keypoints_2d' in person:
            draw_keypoints(image, person['hand_right_keypoints_2d'])

    return image


def process_directory(directory, output_directory, image_size=(1280, 720)):
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            json_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
            total_files = len(json_files)
            selected_frames = list(range(0, total_files, 3))  # 每三个文件选一个文件

            for frame_idx,idx in enumerate(selected_frames):
                json_file = os.path.join(dir_path, json_files[idx])
                image = process_json_file(json_file, image_size)
                frame_name = f"frame_{str(frame_idx).zfill(4)}.png"
                output_path = os.path.join(output_directory, dir_name, frame_name)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, image)
                print(f'Saved: {output_path}')



# 设置输入目录和输出目录
input_directory = r'D:\VK-S2T\data\json\test'
output_directory = r'D:\VK-S2T\data\output\test'

# 开始处理目录
process_directory(input_directory, output_directory)
