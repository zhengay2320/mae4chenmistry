"""
spectral_metrics.py — 光谱评估指标计算模块

包含两个评估口径：
  1. 主口径 (full)：原始真实光谱 x_true vs 完整重建光谱 x_recon
  2. 补充口径 (masked)：仅在 masked 区域上评估

指标：
  - MAE (Mean Absolute Error)
  - Pearson correlation coefficient
  - FWHM error (平均匹配峰对的半峰全宽误差)
  - Peak position error (平均匹配峰对的位置误差)
  - Peak intensity ratio error (平均匹配峰对的强度比误差)

所有基础指标 (MAE, Pearson) 在标准化光谱空间计算。
峰指标通过 peak_analysis 模块计算。
"""

import numpy as np
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import Optional

from peak_analysis import (
    analyze_peaks,
    PeakAnalysisResult,
    SpectrumPreprocessor,
    PeakDetector,
)


@dataclass
class SampleMetrics:
    """单个样本的完整评估指标"""

    # 样本标识
    sample_id: int
    smiles: str

    # ---- 主口径 (Full Spectrum) ----
    full_mae: float
    full_pearson: float
    full_peak_pos_error: float       # 平均峰位误差 (采样点数)
    full_fwhm_error: float           # 平均 FWHM 误差 (采样点数)
    full_intensity_ratio_error: float  # 平均强度比误差

    # ---- 补充口径 (Masked Region Only) ----
    masked_mae: float
    masked_pearson: float

    # ---- 峰统计 (主口径) ----
    n_true_peaks: int
    n_recon_peaks: int
    n_matched_peaks: int
    n_missed_peaks: int
    n_spurious_peaks: int


def compute_mae(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """计算 Mean Absolute Error"""
    return float(np.mean(np.abs(x_true - x_recon)))


def compute_pearson(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """
    计算 Pearson 相关系数。
    如果方差为 0（恒定信号），返回 0.0。
    """
    if np.std(x_true) < 1e-10 or np.std(x_recon) < 1e-10:
        return 0.0
    r, _ = pearsonr(x_true, x_recon)
    return float(r)


def reconstruct_full_spectrum(
    x_true: np.ndarray,
    pred_masked: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 1,
) -> np.ndarray:
    """
    将模型预测的 masked patch 填回未遮挡区域，得到完整重建光谱。

    参数:
        x_true: 原始真实光谱 (L,)
        pred_masked: 模型对 masked patch 的预测
                     如果 patch_size=1，形状为 (n_masked,)
                     如果 patch_size>1，形状为 (n_masked_patches, patch_size)
        mask: 遮挡掩码
              如果 patch_size=1，形状为 (L,)，1 = masked, 0 = visible
              如果 patch_size>1，形状为 (n_patches,)，1 = masked, 0 = visible
        patch_size: 每个 patch 的大小

    返回:
        x_recon: 完整重建光谱 (L,)
                 masked 位置填入预测值，其余位置保留真实值
    """
    x_recon = x_true.copy()

    if patch_size == 1:
        # 逐点 mask
        masked_indices = np.where(mask > 0.5)[0]
        if len(masked_indices) != len(pred_masked):
            # 如果 pred_masked 长度与 masked 数量不一致，尝试截断或扩展
            n = min(len(masked_indices), len(pred_masked))
            x_recon[masked_indices[:n]] = pred_masked[:n]
        else:
            x_recon[masked_indices] = pred_masked
    else:
        # patch-level mask
        n_patches = len(mask)
        masked_patch_idx = np.where(mask > 0.5)[0]
        pred_idx = 0
        for pi in masked_patch_idx:
            start = pi * patch_size
            end = min(start + patch_size, len(x_recon))
            length = end - start
            if pred_idx < len(pred_masked):
                x_recon[start:end] = pred_masked[pred_idx][:length]
                pred_idx += 1

    return x_recon


def evaluate_sample(
    x_true: np.ndarray,
    x_recon: np.ndarray,
    mask: np.ndarray,
    sample_id: int,
    smiles: str,
    patch_size: int = 1,
    preprocessor: Optional[SpectrumPreprocessor] = None,
    detector: Optional[PeakDetector] = None,
    max_match_distance: int = 20,
) -> SampleMetrics:
    """
    对单个样本计算所有评估指标。

    参数:
        x_true: 原始真实光谱 (标准化空间, 1D array)
        x_recon: 完整重建光谱 (标准化空间, 1D array)
                 （已通过 reconstruct_full_spectrum 构建）
        mask: 遮挡掩码 (与 x_true 等长，或 patch-level)
        sample_id: 样本索引
        smiles: 分子标识
        patch_size: patch 大小
        preprocessor: 峰分析预处理器
        detector: 峰检测器
        max_match_distance: 峰匹配最大距离

    返回:
        SampleMetrics
    """

    # ===================== 主口径 (Full Spectrum) =====================
    full_mae = compute_mae(x_true, x_recon)
    full_pearson = compute_pearson(x_true, x_recon)

    # 峰分析
    peak_result: PeakAnalysisResult = analyze_peaks(
        x_true, x_recon, preprocessor, detector, max_match_distance
    )

    # ===================== 补充口径 (Masked Region) =====================
    if patch_size == 1:
        masked_indices = np.where(mask > 0.5)[0]
    else:
        masked_indices = []
        n_patches = len(mask)
        for pi in range(n_patches):
            if mask[pi] > 0.5:
                start = pi * patch_size
                end = min(start + patch_size, len(x_true))
                masked_indices.extend(range(start, end))
        masked_indices = np.array(masked_indices)

    if len(masked_indices) > 1:
        masked_mae = compute_mae(x_true[masked_indices], x_recon[masked_indices])
        masked_pearson = compute_pearson(x_true[masked_indices], x_recon[masked_indices])
    else:
        masked_mae = -1.0
        masked_pearson = -1.0

    return SampleMetrics(
        sample_id=sample_id,
        smiles=smiles,
        full_mae=full_mae,
        full_pearson=full_pearson,
        full_peak_pos_error=peak_result.mean_peak_pos_error,
        full_fwhm_error=peak_result.mean_fwhm_error,
        full_intensity_ratio_error=peak_result.mean_intensity_ratio_error,
        masked_mae=masked_mae,
        masked_pearson=masked_pearson,
        n_true_peaks=peak_result.n_true,
        n_recon_peaks=peak_result.n_recon,
        n_matched_peaks=peak_result.n_matched,
        n_missed_peaks=peak_result.n_missed,
        n_spurious_peaks=peak_result.n_spurious,
    )


def aggregate_metrics(metrics_list: list) -> dict:
    """
    对所有样本指标做聚合统计。

    返回字典包含:
      - 每项指标的 mean, std, median, min, max
      - worst-5 样本 (按 full_mae 排序)
    """
    import json

    field_names = [
        "full_mae", "full_pearson",
        "full_peak_pos_error", "full_fwhm_error", "full_intensity_ratio_error",
        "masked_mae", "masked_pearson",
    ]

    summary = {}

    for field in field_names:
        values = [getattr(m, field) for m in metrics_list if getattr(m, field) >= 0]
        if len(values) == 0:
            summary[field] = {
                "mean": None, "std": None, "median": None,
                "min": None, "max": None, "n_valid": 0,
            }
            continue
        arr = np.array(values)
        summary[field] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_valid": len(values),
        }

    # Worst-5 samples by full_mae
    sorted_by_mae = sorted(metrics_list, key=lambda m: -m.full_mae)
    worst_5 = [
        {"sample_id": m.sample_id, "smiles": m.smiles, "full_mae": m.full_mae}
        for m in sorted_by_mae[:5]
    ]
    summary["worst_5_by_mae"] = worst_5

    # 峰统计
    total_true = sum(m.n_true_peaks for m in metrics_list)
    total_recon = sum(m.n_recon_peaks for m in metrics_list)
    total_matched = sum(m.n_matched_peaks for m in metrics_list)
    total_missed = sum(m.n_missed_peaks for m in metrics_list)
    total_spurious = sum(m.n_spurious_peaks for m in metrics_list)
    summary["peak_stats"] = {
        "total_true_peaks": total_true,
        "total_recon_peaks": total_recon,
        "total_matched": total_matched,
        "total_missed": total_missed,
        "total_spurious": total_spurious,
        "match_rate": total_matched / max(total_true, 1),
        "precision": total_matched / max(total_recon, 1),
    }

    return summary
