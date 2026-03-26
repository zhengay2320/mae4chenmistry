import os
import shutil
import random

def split_files(input_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')

    # 创建训练集、验证集、测试集文件夹
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 获取文件夹中的所有文件
    all_files = os.listdir(input_folder)
    all_files = [f for f in all_files if os.path.isfile(os.path.join(input_folder, f))]  # 只考虑文件

    # 随机打乱文件列表
    random.shuffle(all_files)

    # 计算拆分的数量
    total_files = len(all_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count  # 剩下的就是测试集

    # 分配文件
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    # 移动文件到对应的文件夹
    for f in train_files:
        shutil.move(os.path.join(input_folder, f), os.path.join(train_folder, f))

    for f in val_files:
        shutil.move(os.path.join(input_folder, f), os.path.join(val_folder, f))

    for f in test_files:
        shutil.move(os.path.join(input_folder, f), os.path.join(test_folder, f))

    print(f"文件已拆分为：\n训练集：{len(train_files)} 个文件\n验证集：{len(val_files)} 个文件\n测试集：{len(test_files)} 个文件")

# 使用示例
input_folder = '/home/ubuntu/data/zhenganyang/MAE_data/mix/raw_data/ir_json_minmax'  # 请替换为你的输入文件夹路径
output_folder = '/home/ubuntu/data/zhenganyang/MAE_data/mix/raw_data/ir_json_dataset'  # 请替换为你希望存储拆分文件的输出文件夹路径

split_files(input_folder, output_folder)
