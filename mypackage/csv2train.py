import csv
import gzip
import pickle
import json

# 读取CSV文件并转换为字典
def load_csv_to_dict(csv_file):
    data = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['Key']
            # 尝试解析JSON，如果失败则尝试替换引号并重新解析
            try:
                value = json.loads(row['Value'])
            except json.JSONDecodeError:
                corrected_value = row['Value'].replace("'", '"')
                try:
                    value = json.loads(corrected_value)
                except json.JSONDecodeError:
                    print(f"无法解析行：{row['Value']}")
                    continue
            data[key] = value
    return data

# 保存字典为gzip压缩的pickle文件
def save_dict_to_gz(data, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)

# 示例用法

