import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class IRSpectraDataset(Dataset):
    """
    读取预处理好的 pkl 数据集。

    pkl 文件格式要求：
        [
            (smiles, ir_spectra),
            (smiles, ir_spectra),
            ...
        ]

    参数
    ----
    data_path : str
        pkl 文件路径
    normalize : bool
        是否标准化
    return_dict : bool
        False 时只返回 spectrum tensor
        True 时返回完整字典
    use_global_stats : bool
        True: 使用整个数据集的全局均值方差
        False: 每条样本单独标准化（与你现在的生成/读取方式更一致）
    """

    def __init__(
        self,
        data_path,
        normalize=True,
        return_dict=False,
        use_global_stats=False,
    ):
        super().__init__()

        self.data_path = data_path
        self.normalize = normalize
        self.return_dict = return_dict
        self.use_global_stats = use_global_stats

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found.")

        if not self.data_path.endswith(".pkl"):
            raise ValueError(f"Expected a .pkl file, but got: {self.data_path}")

        file_size = os.path.getsize(self.data_path)
        print(f"Loading preprocessed data from {self.data_path} ...")
        print(f"File size: {file_size} bytes")

        if file_size == 0:
            raise ValueError(f"{self.data_path} is empty.")


        with open(self.data_path, "rb") as f:
            self.total_data = pickle.load(f)
            # self.total_data = pickle.load(f)[0:100000]


        if not isinstance(self.total_data, list):
            raise ValueError(
                f"Expected loaded object to be a list, but got {type(self.total_data)}"
            )

        if len(self.total_data) == 0:
            raise ValueError(f"{self.data_path} contains no samples.")

        print(f"Loaded {len(self.total_data)} data items.")

        # 检查前几个样本结构，避免后面训练时才报错
        check_num = min(5, len(self.total_data))
        for i in range(check_num):
            item = self.total_data[i]
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                raise ValueError(
                    f"Invalid sample at index {i}. "
                    f"Expected (smiles, ir_spectra), got: {item}"
                )

        # 可选：计算全局统计量
        if self.use_global_stats:
            all_spectra = []
            for i, item in enumerate(self.total_data):
                smiles, ir_spectra = item[:2]
                ir_spectra = np.asarray(ir_spectra, dtype=np.float32)

                if ir_spectra.ndim != 1:
                    raise ValueError(
                        f"Sample {i} ir_spectra must be 1D, but got shape {ir_spectra.shape}"
                    )

                all_spectra.append(ir_spectra)

            try:
                all_spectra = np.stack(all_spectra, axis=0)
            except Exception as e:
                raise ValueError(
                    "Failed to stack all spectra. "
                    "Please check whether all ir_spectra have the same length. "
                    f"Original error: {e}"
                )

            self.global_mean = float(np.mean(all_spectra))
            self.global_std = float(np.std(all_spectra))
            if self.global_std < 1e-8:
                self.global_std = 1.0
        else:
            self.global_mean = None
            self.global_std = None



    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        item = self.total_data[idx]

        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(
                f"Data item at index {idx} is invalid. "
                f"Expected (smiles, ir_spectra), got: {type(item)}"
            )

        smiles, ir_spectra = item
        spectrum_raw = np.asarray(ir_spectra, dtype=np.float32).copy()

        if spectrum_raw.ndim != 1:
            raise ValueError(
                f"ir_spectra at index {idx} must be 1D, but got shape {spectrum_raw.shape}"
            )

        # 标准化
        if self.normalize:
            if self.use_global_stats:
                mean = self.global_mean
                std = self.global_std
            else:
                mean = float(np.mean(spectrum_raw))
                std = float(np.std(spectrum_raw))
                if std < 1e-8:
                    std = 1.0

            spectrum_norm = (spectrum_raw - mean) / (std + 1e-6)
        else:
            spectrum_norm = spectrum_raw.copy()
            mean = float(np.mean(spectrum_raw))
            std = float(np.std(spectrum_raw))
            if std < 1e-8:
                std = 1.0

        spectrum_tensor = torch.tensor(spectrum_norm, dtype=torch.float32)

        # 训练模式：只返回 tensor
        if not self.return_dict:
            return spectrum_tensor

        # 评估模式：返回完整信息
        return {
            "spectrum": spectrum_tensor,
            "spectrum_raw": torch.tensor(spectrum_raw, dtype=torch.float32),
            "smiles": smiles,
            "idx": idx,
            "mean": mean,
            "std": std,
        }


def eval_collate_fn(batch):
    """
    评估时使用的 collate_fn
    """
    return {
        "spectrum": torch.stack([b["spectrum"] for b in batch]),
        "spectrum_raw": torch.stack([b["spectrum_raw"] for b in batch]),
        "smiles": [b["smiles"] for b in batch],
        "idx": [b["idx"] for b in batch],
        "mean": [b["mean"] for b in batch],
        "std": [b["std"] for b in batch],
    }