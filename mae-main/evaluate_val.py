"""
evaluate_val.py — 验证集评估与 Hard Negative Mining 主脚本

用法:
    python evaluate_val.py \
        --data_path /path/to/data \
        --checkpoint /path/to/model.pth \
        --output_dir ./eval_results \
        --mask_ratio 0.75 \
        --patch_size 16 \
        --top_k 5 \
        --device cuda

功能:
  1. 加载验证集和模型权重
  2. 对每个验证样本生成重建光谱
  3. 计算主口径和补充口径指标
  4. 执行 hard negative mining 分析
  5. 输出 sample_metrics.csv, summary.json, hard_negatives.csv, 可视化图

依赖:
  - spectral_metrics.py
  - peak_analysis.py
  - hard_negative_mining.py
  - datasets/ir_spectra_dataset_ext.py
  - models/mae_model.py (原有模型定义)
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- 本地模块 ----
from spectral_metrics import (
    evaluate_sample,
    reconstruct_full_spectrum,
    aggregate_metrics,
    SampleMetrics,
)
from peak_analysis import (
    SpectrumPreprocessor,
    PeakDetector,
    analyze_peaks,
)
from hard_negative_mining import (
    find_hard_negatives,
    get_hard_negative_subset_indices,
    compare_subsets,
    save_hard_negatives_csv,
)

# ---- 数据集 ----
# 使用扩展版数据集
sys.path.insert(0, os.path.dirname(__file__))
from datasets.ir_spectra_dataset_ext import IRSpectraDataset, eval_collate_fn


# ============================================================
# 模型加载 — 请根据实际 mae_model.py 定义调整
# ============================================================

def load_model(checkpoint_path: str, device: str, **model_kwargs):
    """
    加载 MAE 模型。

    请根据你的 mae_model.py 中的类名和参数调整。
    典型接口：
        model = MaskedAutoencoderViT1D(
            seq_len=..., patch_size=..., embed_dim=..., ...
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    """
    # try:
    #     from models.mae_model import MaskedAutoencoder1D
    #     model_cls = MaskedAutoencoder1D
    # except ImportError:
    #     try:
    #         from models.mae_model import MAE1D
    #         model_cls = MAE1D
    #     except ImportError:
    #         try:
    #             from models.mae_model import MaskedAutoencoderViT1D
    #             model_cls = MaskedAutoencoderViT1D
    #         except ImportError:
    #             raise ImportError(
    #                 "无法从 models.mae_model 中导入模型类。"
    #                 "请确认模型类名称，并在此函数中修改 import。"
    #             )

    # model = model_cls(**model_kwargs)
    from models import mae_model
    model = mae_model.mae_vit_base_patch9_dec512d8b()

    state_dict = torch.load(checkpoint_path, map_location=device)
    # 处理可能被包在 "model" 或 "state_dict" key 里的情况
    if isinstance(state_dict, dict):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def run_model_inference(model, batch_spectrum, mask_ratio, device):
    """
    执行模型前向推理，获取重建结果和 mask。

    请根据你的模型 forward 接口调整。
    典型 MAE forward 返回:
        loss, pred, mask = model(x, mask_ratio=mask_ratio)

    参数:
        model: MAE 模型
        batch_spectrum: (B, L) 或 (B, 1, L) 标准化光谱
        mask_ratio: 遮挡比例
        device: 设备

    返回:
        pred: (B, L) 预测值（所有位置，仅 masked 区域有意义）
        mask: (B, n_patches) 或 (B, L) 遮挡掩码
    """
    x = batch_spectrum.to(device)

    # 当前 mae_model.py 的 patch_embed.forward() 期望输入为 (B, L)
    if x.dim() == 3 and x.shape[1] == 1:
        x_input = x.squeeze(1)  # (B, 1, L) -> (B, L)
    elif x.dim() == 2:
        x_input = x  # already (B, L)
    else:
        raise ValueError(f"Unexpected batch_spectrum shape: {x.shape}, expected (B, L) or (B, 1, L)")

    with torch.no_grad():
        # 典型 MAE forward 签名：
        #   loss, pred, mask = model(x, mask_ratio)
        # 或
        #   output = model(x, mask_ratio)  # output.pred, output.mask
        result = model(x_input, mask_ratio=mask_ratio)

        if isinstance(result, tuple):
            if len(result) == 3:
                loss, pred, mask = result
            elif len(result) == 2:
                pred, mask = result
            else:
                raise ValueError(f"模型返回了 {len(result)} 个值，期望 2 或 3 个")
        elif hasattr(result, "pred"):
            pred = result.pred
            mask = result.mask
        else:
            raise ValueError("无法解析模型输出，请检查 forward 返回格式")

    # 确保 pred 和 mask 是正确的维度
    if pred.dim() == 3:
        pred = pred.squeeze(1)  # (B, L)
    if mask.dim() == 3:
        mask = mask.squeeze(1)

    return pred.cpu().numpy(), mask.cpu().numpy()


# ============================================================
# 可视化
# ============================================================

def plot_spectrum_comparison(
    x_true: np.ndarray,
    x_recon: np.ndarray,
    mask: np.ndarray,
    sample_id: int,
    smiles: str,
    metrics: SampleMetrics,
    output_path: str,
    patch_size: int = 1,
    preprocessor=None,
    detector=None,
):
    """
    绘制原谱 vs 重建谱对比图，标注峰位、半峰宽、mask 区域。
    """
    if preprocessor is None:
        preprocessor = SpectrumPreprocessor()
    if detector is None:
        detector = PeakDetector()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    L = len(x_true)
    x_axis = np.arange(L)

    # ---- 主图：真实谱 vs 重建谱 ----
    ax = axes[0]

    # 标注 mask 区域（浅红色背景）
    if patch_size == 1:
        mask_expanded = mask[:L] if len(mask) >= L else np.zeros(L)
    else:
        mask_expanded = np.zeros(L)
        n_patches = len(mask)
        for pi in range(n_patches):
            if mask[pi] > 0.5:
                start = pi * patch_size
                end = min(start + patch_size, L)
                mask_expanded[start:end] = 1.0

    # 绘制 mask 区域背景
    in_mask = False
    mask_start = 0
    for i in range(L):
        if mask_expanded[i] > 0.5 and not in_mask:
            mask_start = i
            in_mask = True
        elif mask_expanded[i] < 0.5 and in_mask:
            ax.axvspan(mask_start, i, alpha=0.15, color="red", label="" if mask_start > 0 else "Masked region")
            in_mask = False
    if in_mask:
        ax.axvspan(mask_start, L, alpha=0.15, color="red")

    ax.plot(x_axis, x_true, color="#2c3e50", linewidth=1.2, label="True spectrum", alpha=0.9)
    ax.plot(x_axis, x_recon, color="#e74c3c", linewidth=1.0, label="Reconstructed", alpha=0.8, linestyle="--")

    # 标注峰位
    peak_result = analyze_peaks(x_true, x_recon, preprocessor, detector)

    for tp in peak_result.true_peaks:
        ax.axvline(tp.position, color="#3498db", alpha=0.4, linewidth=0.8, linestyle=":")
        ax.plot(tp.position, x_true[tp.position], "v", color="#3498db", markersize=6)

    for rp in peak_result.recon_peaks:
        ax.plot(rp.position, x_recon[rp.position], "^", color="#e74c3c", markersize=5, alpha=0.7)

    # 标注 FWHM（对匹配的峰对）
    for mp in peak_result.matched_pairs:
        tp = mp.true_peak
        if tp.fwhm > 0:
            half_h = (tp.height + (tp.height - tp.prominence)) / 2.0
            ax.hlines(half_h, tp.left_base, tp.right_base,
                      color="#3498db", alpha=0.5, linewidth=1.5, linestyle="-")

    ax.set_ylabel("Intensity (standardized)")
    title_str = f"Sample {sample_id}"
    if smiles and smiles != f"sample_{sample_id}":
        title_str += f" | {smiles[:40]}"
    title_str += f" | MAE={metrics.full_mae:.4f} | Pearson={metrics.full_pearson:.4f}"
    ax.set_title(title_str, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # ---- 残差图 ----
    ax2 = axes[1]
    residual = x_recon - x_true
    ax2.fill_between(x_axis, residual, 0, alpha=0.4,
                     color=np.where(mask_expanded > 0.5, "#e74c3c", "#95a5a6"))
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Residual")
    ax2.set_ylim(-max(abs(residual.min()), abs(residual.max()), 0.1) * 1.2,
                  max(abs(residual.min()), abs(residual.max()), 0.1) * 1.2)

    # ---- Mask 可视化 ----
    ax3 = axes[2]
    ax3.imshow(mask_expanded.reshape(1, -1), aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax3.set_yticks([])
    ax3.set_xlabel("Spectral point index")
    ax3.set_ylabel("Mask")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_distribution(all_metrics: list, output_dir: str):
    """绘制各指标的分布直方图"""
    fields = {
        "full_mae": "MAE (Full Spectrum)",
        "full_pearson": "Pearson Correlation (Full)",
        "full_peak_pos_error": "Peak Position Error",
        "full_fwhm_error": "FWHM Error",
        "full_intensity_ratio_error": "Intensity Ratio Error",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (field, title) in enumerate(fields.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        values = [getattr(m, field) for m in all_metrics if getattr(m, field) >= 0]
        if values:
            ax.hist(values, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
            ax.axvline(np.mean(values), color="#e74c3c", linestyle="--",
                       label=f"Mean={np.mean(values):.4f}")
            ax.axvline(np.median(values), color="#2ecc71", linestyle=":",
                       label=f"Median={np.median(values):.4f}")
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # 隐藏多余的 subplot
    for idx in range(len(fields), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Validation Set Metrics Distribution", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_hard_negative_pairs(
    pair_data: list,
    output_path: str,
    n_pairs: int = 5,
):
    """
    绘制 hardest pairs 的对比图。

    pair_data: List[dict] with keys:
        query_true, neg_true, query_id, neg_id, pearson_sim
    """
    n = min(n_pairs, len(pair_data))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        d = pair_data[i]
        L = len(d["query_true"])
        ax.plot(range(L), d["query_true"], label=f"Query #{d['query_id']}", alpha=0.8)
        ax.plot(range(L), d["neg_true"], label=f"Hard Neg #{d['neg_id']}", alpha=0.8, linestyle="--")
        ax.set_title(f"Pearson sim = {d['pearson_sim']:.4f}", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylabel("Intensity")

    axes[-1].set_xlabel("Spectral point index")
    plt.suptitle(f"Top-{n} Hardest Negative Pairs (by Pearson similarity)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 主流程
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MAE 1D IR Spectra Validation Evaluation")
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据文件路径 (CSV 或目录)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="输出目录")
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                        help="遮挡比例")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch 大小")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批量大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Hard negative Top-K")
    parser.add_argument("--max_match_distance", type=int, default=20,
                        help="峰匹配最大距离")
    parser.add_argument("--n_worst_vis", type=int, default=10,
                        help="可视化 worst-case 样本数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="数据划分随机种子")

    # 模型参数 — 请根据实际 mae_model.py 构造函数调整
    parser.add_argument("--seq_len", type=int, default=1600,
                        help="光谱序列长度")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="嵌入维度")
    parser.add_argument("--depth", type=int, default=6,
                        help="Transformer 层数")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="注意力头数")
    parser.add_argument("--decoder_embed_dim", type=int, default=128,
                        help="解码器嵌入维度")
    parser.add_argument("--decoder_depth", type=int, default=4,
                        help="解码器层数")
    parser.add_argument("--decoder_num_heads", type=int, default=4,
                        help="解码器注意力头数")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 创建输出目录 ----
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[INFO] Using device: {device}")

    # ---- 加载验证集 ----
    print("[INFO] Loading validation dataset...")
    val_dataset = IRSpectraDataset(
        data_path=args.data_path,
        normalize=True,
        # split="val",
        # seed=args.seed,
        return_dict=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=0,
    )
    print(f"[INFO] Validation set size: {len(val_dataset)}")

    # ---- 加载模型 ----
    print("[INFO] Loading model...")
    model_kwargs = {
        "seq_len": args.seq_len,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "decoder_embed_dim": args.decoder_embed_dim,
        "decoder_depth": args.decoder_depth,
        "decoder_num_heads": args.decoder_num_heads,
    }
    # 注意：如果你的模型类参数不同，请修改 model_kwargs
    model = load_model(args.checkpoint, device, **model_kwargs)
    print("[INFO] Model loaded.")

    # ---- 初始化峰分析器 ----
    preprocessor = SpectrumPreprocessor(smooth_window=11, smooth_polyorder=3)
    detector = PeakDetector(
        prominence_min=0.05,
        distance_min=10,
        height_min=0.02,
        width_min=2,
        fwhm_max_ratio=0.25,
    )

    # ============================================================
    # Phase 1: 逐样本评估
    # ============================================================
    print("[INFO] Phase 1: Per-sample evaluation...")
    all_metrics = []
    all_x_true = []      # 收集标准化光谱（用于 hard negative mining）
    all_sample_ids = []
    all_smiles = []
    all_x_recon = []
    all_masks = []

    for batch_idx, batch in enumerate(val_loader):
        spectrum = batch["spectrum"]   # (B, L) 标准化
        smiles_list = batch["smiles"]
        idx_list = batch["idx"]

        # 模型推理
        pred, mask = run_model_inference(model, spectrum, args.mask_ratio, device)

        B = spectrum.shape[0]
        for i in range(B):
            x_true = spectrum[i].numpy()
            pred_i = pred[i]
            mask_i = mask[i]
            sample_id = idx_list[i]
            smi = smiles_list[i]

            # 构建完整重建光谱
            x_recon = reconstruct_full_spectrum(
                x_true, pred_i, mask_i, patch_size=args.patch_size
            )

            # 计算指标
            sample_metrics = evaluate_sample(
                x_true=x_true,
                x_recon=x_recon,
                mask=mask_i,
                sample_id=sample_id,
                smiles=smi,
                patch_size=args.patch_size,
                preprocessor=preprocessor,
                detector=detector,
                max_match_distance=args.max_match_distance,
            )

            all_metrics.append(sample_metrics)
            all_x_true.append(x_true)
            all_x_recon.append(x_recon)
            all_masks.append(mask_i)
            all_sample_ids.append(sample_id)
            all_smiles.append(smi)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx+1)*args.batch_size}/{len(val_dataset)} samples")

    print(f"[INFO] Evaluation complete. {len(all_metrics)} samples processed.")

    # ---- 保存 sample_metrics.csv ----
    csv_path = os.path.join(args.output_dir, "sample_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "smiles",
            "full_mae", "full_pearson",
            "full_peak_pos_error", "full_fwhm_error", "full_intensity_ratio_error",
            "masked_mae", "masked_pearson",
            "n_true_peaks", "n_recon_peaks", "n_matched", "n_missed", "n_spurious",
        ])
        for m in all_metrics:
            writer.writerow([
                m.sample_id, m.smiles,
                f"{m.full_mae:.6f}", f"{m.full_pearson:.6f}",
                f"{m.full_peak_pos_error:.4f}", f"{m.full_fwhm_error:.4f}",
                f"{m.full_intensity_ratio_error:.6f}",
                f"{m.masked_mae:.6f}", f"{m.masked_pearson:.6f}",
                m.n_true_peaks, m.n_recon_peaks,
                m.n_matched_peaks, m.n_missed_peaks, m.n_spurious_peaks,
            ])
    print(f"[INFO] Saved: {csv_path}")

    # ---- 聚合统计 ----
    summary = aggregate_metrics(all_metrics)

    # ============================================================
    # Phase 2: Hard Negative Mining
    # ============================================================
    print("[INFO] Phase 2: Hard negative mining...")
    spectra_matrix = np.array(all_x_true)  # (N, L)

    all_hard_negs, pearson_sim, cosine_sim = find_hard_negatives(
        spectra=spectra_matrix,
        sample_ids=all_sample_ids,
        smiles_list=all_smiles,
        top_k=args.top_k,
    )

    # 保存 hard negatives CSV
    hn_csv_path = os.path.join(args.output_dir, "hard_negatives.csv")
    save_hard_negatives_csv(all_hard_negs, hn_csv_path)
    print(f"[INFO] Saved: {hn_csv_path}")

    # Hard negative 子集分析
    hn_local_indices = get_hard_negative_subset_indices(all_hard_negs, all_sample_ids)
    hn_comparison = compare_subsets(all_metrics, hn_local_indices)
    summary["hard_negative_comparison"] = hn_comparison
    summary["hard_negative_subset_size"] = len(hn_local_indices)

    # ---- 保存 summary.json ----
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[INFO] Saved: {summary_path}")

    # ============================================================
    # Phase 3: 可视化
    # ============================================================
    print("[INFO] Phase 3: Generating visualizations...")

    # 3.1 指标分布图
    plot_metrics_distribution(all_metrics, fig_dir)
    print(f"  Saved: metrics_distribution.png")

    # 3.2 Worst-case 样本可视化
    sorted_by_mae = sorted(range(len(all_metrics)), key=lambda i: -all_metrics[i].full_mae)
    n_vis = min(args.n_worst_vis, len(sorted_by_mae))
    for rank, local_idx in enumerate(sorted_by_mae[:n_vis]):
        m = all_metrics[local_idx]
        fig_path = os.path.join(fig_dir, f"worst_{rank+1}_sample_{m.sample_id}.png")
        plot_spectrum_comparison(
            x_true=all_x_true[local_idx],
            x_recon=all_x_recon[local_idx],
            mask=all_masks[local_idx],
            sample_id=m.sample_id,
            smiles=m.smiles,
            metrics=m,
            output_path=fig_path,
            patch_size=args.patch_size,
            preprocessor=preprocessor,
            detector=detector,
        )
        print(f"  Saved: worst_{rank+1}_sample_{m.sample_id}.png")

    # 3.3 Hard negative pair 可视化
    pair_data = []
    for i, hn_list in enumerate(all_hard_negs):
        if hn_list:
            hn = hn_list[0]  # Top-1
            # 找到 neg 在 all_x_true 中的 local index
            neg_local = None
            for j, sid in enumerate(all_sample_ids):
                if sid == hn.neg_id:
                    neg_local = j
                    break
            if neg_local is not None:
                pair_data.append({
                    "query_true": all_x_true[i],
                    "neg_true": all_x_true[neg_local],
                    "query_id": hn.query_id,
                    "neg_id": hn.neg_id,
                    "pearson_sim": hn.pearson_sim,
                })

    # 按 Pearson similarity 降序（最相似的 pair）
    pair_data.sort(key=lambda d: -d["pearson_sim"])
    hn_fig_path = os.path.join(fig_dir, "hardest_negative_pairs.png")
    plot_hard_negative_pairs(pair_data, hn_fig_path, n_pairs=5)
    print(f"  Saved: hardest_negative_pairs.png")

    # ============================================================
    # 最终报告输出
    # ============================================================
    print("\n" + "=" * 60)
    print("  VALIDATION EVALUATION SUMMARY")
    print("=" * 60)

    for field in ["full_mae", "full_pearson", "full_peak_pos_error",
                   "full_fwhm_error", "full_intensity_ratio_error"]:
        stats = summary.get(field, {})
        if stats.get("mean") is not None:
            print(f"  {field:35s}: "
                  f"mean={stats['mean']:.4f} ± {stats['std']:.4f}, "
                  f"median={stats['median']:.4f}, "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}], "
                  f"n={stats['n_valid']}")

    print(f"\n  Peak Statistics:")
    ps = summary.get("peak_stats", {})
    print(f"    Total true peaks:  {ps.get('total_true_peaks', 'N/A')}")
    print(f"    Total recon peaks: {ps.get('total_recon_peaks', 'N/A')}")
    print(f"    Match rate:        {ps.get('match_rate', 'N/A'):.4f}")
    print(f"    Precision:         {ps.get('precision', 'N/A'):.4f}")

    print(f"\n  Hard Negative Subset Size: {len(hn_local_indices)}")
    print(f"\n  Worst-5 samples by MAE:")
    for w in summary.get("worst_5_by_mae", []):
        print(f"    ID={w['sample_id']}, MAE={w['full_mae']:.4f}, SMILES={w['smiles'][:30]}")

    print(f"\n  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
