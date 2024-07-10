from  mypackage import framedownsample as down
from  mypackage import pair2text as pair
from mypackage import csv2train as dabao
import os

datatype = input("请输入要生成的数据集类型 train，dev，test    ")
if datatype == "train":
    csv_file = r'..\save\how2sign_realigned_train.csv'
    output_csv = r'F:\Data_preprocessing\output_save\train\paired_text-frame_data.csv'
elif datatype == "dev":
    csv_file = r'..\save\how2sign_realigned_val.csv'
    output_csv = r'F:\Data_preprocessing\output_save\dev\paired_text-frame_data.csv'
elif datatype == "test":
    csv_file = r'..\save\how2sign_realigned_test.csv'
    output_csv = r'F:\Data_preprocessing\output_save\test\paired_text-frame_data.csv'


frame_folder = input("请输入生成的帧文件存放在的位置")
frame_folder = frame_folder if frame_folder else r"F:\Data_preprocessing\output_save\frame_tem_save"
output_data = pair.process_frames_and_text(frame_folder, csv_file)
pair.save_to_csv(output_data, output_csv)
print(f"\033[91m  帧数据与文本配队结束 Done. \033[0m")
# 添加表头
# 读取CSV文件并解析JSON

final_df = pair.biaotou(output_csv)
print(f"\033[91m  表头添加，保存在output_save种，每次重新生成记得删除原文件 Done. \033[0m")
'''
这部分留一步给识别关键点并且打印到原始图像上来 融合关键点信息

'''
# 打包压缩为训练数据
dabao_wenjianming = input("请输出本次生成的训练数据类型， 如labels.train,labels.dev,labels.test  ")
output_dev_file_path = r'D:\VK-S2T\data\打包save'
dabao_baocun_lujing = os.path.join(output_dev_file_path, dabao_wenjianming)

data_dict = dabao.load_csv_to_dict(output_csv)
dabao.save_dict_to_gz(data_dict, dabao_baocun_lujing)
print(f"\033[91m  打包为训练数据格式成功 Done. \033[0m")