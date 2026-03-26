#
# import os
# import json
# import torch
# from torch.utils.data import Dataset
# import numpy as np
#
# class IRSpectraDataset(Dataset):
#     def __init__(self, folder_path, batch_size=1000):
#         """
#         数据集初始化，加载文件夹中所有的 JSON 文件
#         :param folder_path: 文件夹路径，包含多个 JSON 文件
#         :param batch_size: 每次加载的数据量（批大小）
#         """
#         self.folder_path = folder_path
#         self.batch_size = batch_size
#         self.file_list = [file_name for file_name in os.listdir(folder_path)
#                           if file_name.endswith('.json') and os.path.isfile(os.path.join(folder_path, file_name))]
#
#         if not self.file_list:
#             raise ValueError("文件夹中没有有效的 JSON 文件")
#
#         self.current_file_idx = 0
#         self.current_data_idx = 0
#         self.current_batch = []
#         self.total_data = []  # 全部数据列表，用于最终一次性加载
#         self.batch_n = 0
#         self.idy = 0
#
#         # 将所有数据加载到内存中以便于验证集
#         for file_name in self.file_list:
#             file_path = os.path.join(self.folder_path, file_name)
#             try:
#                 with open(file_path, 'r') as f:
#                     json_data = json.load(f)
#                     for entry in json_data:
#                         smiles = entry['smiles']
#                         ir_spectra = np.array(entry['ir_spectra'], dtype=np.float32)
#                         self.total_data.append((smiles, ir_spectra))
#             except Exception as e:
#                 print(f"无法加载文件 {file_name}: {e}")
#
#     def __len__(self):
#         """返回数据集大小"""
#         return len(self.total_data)
#
#     def _load_next_batch(self):
#         """加载下一个批次的数据"""
#         self.current_batch.clear()
#
#         # 检查是否还有数据
#         if self.current_data_idx < len(self.total_data):
#             while len(self.current_batch) < self.batch_size and self.current_data_idx < len(self.total_data):
#                 self.current_batch.append(self.total_data[self.current_data_idx])
#                 self.current_data_idx += 1
#         else:
#             raise StopIteration  # 所有数据加载完成
#
#     def __getitem__(self, idx):
#         """
#         获取指定索引的数据，包括 SMILES 和 IR 光谱
#         :param idx: 数据索引
#         :return: 包含 SMILES 和 IR 光谱的元组
#         """
#         # 检查当前批次的大小，确保 idx 不会超出当前批次范围
#         # idy = idx - self.batch_size * self.batch_n
#
#         if len(self.current_batch) == 0 or self.idy >= (len(self.current_batch)):
#             self._load_next_batch()
#             self.idy = 0
#
#         # print(f'batch_n:{self.batch_n}')
#         #
#         # print(f'这是第{idx}个数据')
#         # print(f'这是修改后的第{self.idy}个数据')
#         # idy = idx - self.batch_size * self.batch_n
#         smiles, ir_spectra = self.current_batch[self.idy]
#         self.idy = self.idy +1
#         return torch.tensor(ir_spectra, dtype=torch.float32)


import pickle
import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


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