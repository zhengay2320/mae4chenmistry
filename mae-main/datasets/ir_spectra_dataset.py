#


import pickle
import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

#简易版
class IRSpectraDataset(Dataset):
    def __init__(self, folder_path):
        """
        数据集初始化，加载文件夹中所有的 JSON 文件
        :param folder_path: 文件夹路径，包含多个 JSON 文件
        :param batch_size: 每次加载的数据量（批大小）
        """
        """
                初始化时直接读取预处理好的文件
                """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"{folder_path} not found. Please run preprocessing first.")

        print(f"Loading preprocessed data from {folder_path} ...")
        with open(folder_path, 'rb') as f:
            self.total_data = pickle.load(f)
        print(f"Loaded {len(self.total_data)} data items.")

    def __len__(self):
        """返回数据集大小"""
        return len(self.total_data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据，包括 SMILES 和 IR 光谱
        :param idx: 数据索引
        :return: 包含 SMILES 和 IR 光谱的元组
        """
        # 检查当前批次的大小，确保 idx 不会超出当前批次范围
        # idy = idx - self.batch_size * self.batch_n

        smiles, ir_spectra = self.total_data[idx]
        x = ir_spectra
        x = (x - np.mean(x)) / (np.std(x) + 1e-6)
        x = torch.tensor(x, dtype=torch.float32)

        return x