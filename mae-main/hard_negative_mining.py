"""
hard_negative_mining.py — Hard Negative Mining 分析模块

目标（分析实验，非训练 loss 改造）：
  1. 对验证集每个样本，找到最相似但不同的负样本
  2. 输出 Top-K hard negatives
  3. 比较全验证集 vs hard negative 子集的指标差异

相似度度量：
  - 主度量：Pearson correlation（捕获整体形状相似性）
  - 辅助度量：Cosine similarity

定义：
  Hard Negative(x_i) = argmax_{j≠i} sim(x_i^true, x_j^true)
  即真实光谱与 x_i 最相似的其他样本。

科学声明：
  - 相似度在标准化光谱空间计算
  - Top-K = 5（默认）
  - Hard negative 子集 = 所有样本 Top-1 hard negative 的去重集合
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import csv
import json


@dataclass
class HardNegativeInfo:
    """单个 hard negative 的信息"""
    query_id: int
    query_smiles: str
    neg_id: int
    neg_smiles: str
    rank: int                  # 1-based
    pearson_sim: float
    cosine_sim: float


def compute_similarity_matrix(
    spectra: np.ndarray,
    metric: str = "pearson",
) -> np.ndarray:
    """
    计算验证集所有样本两两之间的相似度矩阵。

    参数:
        spectra: (N, L) 标准化光谱矩阵
        metric: "pearson" 或 "cosine"

    返回:
        sim_matrix: (N, N) 相似度矩阵, 对角线设为 -inf
    """
    N = spectra.shape[0]
    sim_matrix = np.full((N, N), -np.inf)

    if metric == "cosine":
        # 向量化计算 cosine similarity
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = spectra / norms
        sim_matrix = normalized @ normalized.T
        np.fill_diagonal(sim_matrix, -np.inf)
    elif metric == "pearson":
        # Pearson correlation
        # 先中心化
        centered = spectra - spectra.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = centered / norms
        sim_matrix = normalized @ normalized.T
        np.fill_diagonal(sim_matrix, -np.inf)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return sim_matrix


def find_hard_negatives(
    spectra: np.ndarray,
    sample_ids: List[int],
    smiles_list: List[str],
    top_k: int = 5,
) -> Tuple[List[List[HardNegativeInfo]], np.ndarray, np.ndarray]:
    """
    为每个验证样本找 Top-K hard negatives。

    参数:
        spectra: (N, L) 标准化光谱
        sample_ids: 样本 ID 列表
        smiles_list: smiles 列表
        top_k: 每个样本保留的 hard negative 数量

    返回:
        all_hard_negs: List[List[HardNegativeInfo]], 每个样本的 Top-K
        pearson_sim_matrix: (N, N)
        cosine_sim_matrix: (N, N)
    """
    N = spectra.shape[0]
    pearson_sim = compute_similarity_matrix(spectra, "pearson")
    cosine_sim = compute_similarity_matrix(spectra, "cosine")

    all_hard_negs = []

    for i in range(N):
        # 按 Pearson similarity 降序排列
        sorted_indices = np.argsort(-pearson_sim[i])
        hard_negs = []
        for rank, j in enumerate(sorted_indices[:top_k], 1):
            if j == i:
                continue
            hn = HardNegativeInfo(
                query_id=sample_ids[i],
                query_smiles=smiles_list[i],
                neg_id=sample_ids[j],
                neg_smiles=smiles_list[j],
                rank=rank,
                pearson_sim=float(pearson_sim[i, j]),
                cosine_sim=float(cosine_sim[i, j]),
            )
            hard_negs.append(hn)
            if len(hard_negs) >= top_k:
                break
        all_hard_negs.append(hard_negs)

    return all_hard_negs, pearson_sim, cosine_sim


def get_hard_negative_subset_indices(
    all_hard_negs: List[List[HardNegativeInfo]],
    sample_ids: List[int],
) -> List[int]:
    """
    获取 hard negative 子集 = 所有样本的 Top-1 hard negative 去重后的 local index 集合。

    返回:
        local_indices: List[int] — 在验证集中的局部索引
    """
    id_to_local = {sid: i for i, sid in enumerate(sample_ids)}
    hn_ids = set()
    for hn_list in all_hard_negs:
        if hn_list:
            hn_ids.add(hn_list[0].neg_id)
    # 加上对应的 query samples
    for hn_list in all_hard_negs:
        if hn_list:
            hn_ids.add(hn_list[0].query_id)

    local_indices = sorted([id_to_local[sid] for sid in hn_ids if sid in id_to_local])
    return local_indices


def compare_subsets(
    all_metrics: list,
    hn_local_indices: List[int],
) -> dict:
    """
    比较全验证集与 hard negative 子集的指标差异。

    参数:
        all_metrics: List[SampleMetrics], 全验证集
        hn_local_indices: hard negative 子集在 all_metrics 中的索引

    返回:
        comparison dict
    """
    fields = [
        "full_mae", "full_pearson",
        "full_peak_pos_error", "full_fwhm_error", "full_intensity_ratio_error",
    ]

    hn_metrics = [all_metrics[i] for i in hn_local_indices if i < len(all_metrics)]

    comparison = {}
    for field in fields:
        all_vals = [getattr(m, field) for m in all_metrics if getattr(m, field) >= 0]
        hn_vals = [getattr(m, field) for m in hn_metrics if getattr(m, field) >= 0]

        comparison[field] = {
            "full_set": {
                "mean": float(np.mean(all_vals)) if all_vals else None,
                "std": float(np.std(all_vals)) if all_vals else None,
                "median": float(np.median(all_vals)) if all_vals else None,
                "n": len(all_vals),
            },
            "hard_neg_subset": {
                "mean": float(np.mean(hn_vals)) if hn_vals else None,
                "std": float(np.std(hn_vals)) if hn_vals else None,
                "median": float(np.median(hn_vals)) if hn_vals else None,
                "n": len(hn_vals),
            },
        }

    return comparison


def save_hard_negatives_csv(
    all_hard_negs: List[List[HardNegativeInfo]],
    output_path: str,
):
    """保存 hard negatives 到 CSV"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "query_smiles", "neg_id", "neg_smiles",
            "rank", "pearson_sim", "cosine_sim",
        ])
        for hn_list in all_hard_negs:
            for hn in hn_list:
                writer.writerow([
                    hn.query_id, hn.query_smiles,
                    hn.neg_id, hn.neg_smiles,
                    hn.rank, f"{hn.pearson_sim:.6f}", f"{hn.cosine_sim:.6f}",
                ])
