#对数据进行标准化后归一化
import os
import json
import numpy as np


# 定义标准化函数
def standardize_data(ir_spectra):
    mean = np.mean(ir_spectra)
    std = np.std(ir_spectra)
    return (ir_spectra - mean) / std


# 定义归一化函数
def normalize_data(ir_spectra):
    min_val = np.min(ir_spectra)
    max_val = np.max(ir_spectra)
    return (ir_spectra - min_val) / (max_val - min_val)


# 读取原始文件夹中的所有文件
input_folder = r'/home/ubuntu/data/zhenganyang/MAE_data/mix/raw_data/ir_json_output'  # 请替换为你的输入文件夹路径
output_folder = r'/home/ubuntu/data/zhenganyang/MAE_data/mix/raw_data/ir_json_minmax'  # 请替换为你希望保存预处理数据的输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的每个文件
for filename in os.listdir(input_folder):
    print(f"正在处理{filename}!")
    if filename.endswith('.json'):
        file_path = os.path.join(input_folder, filename)

        # 读取JSON文件
        with open(file_path, 'r') as f:
            data = json.load(f)

        # 对每个数据项进行标准化和归一化
        for item in data:
            ir_spectra = item['ir_spectra']
            # 标准化
            # standardized_spectra = standardize_data(ir_spectra)
            # 归一化
            normalized_spectra = normalize_data(ir_spectra)
            # 更新数据项
            item['ir_spectra'] = normalized_spectra.tolist()

        # 将处理后的数据保存到新的文件中
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

print("数据预处理完成！")
