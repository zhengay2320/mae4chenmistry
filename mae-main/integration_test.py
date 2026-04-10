"""
integration_test.py — 端到端集成测试（无需 torch 和模型权重）

使用合成光谱数据验证整个评估流水线：
  - 峰分析
  - 光谱指标计算
  - Hard negative mining
  - CSV / JSON 输出
  - 可视化图生成
"""

import sys
import os
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from peak_analysis import SpectrumPreprocessor, PeakDetector, analyze_peaks
from spectral_metrics import (
    compute_mae, compute_pearson,
    reconstruct_full_spectrum, evaluate_sample, aggregate_metrics, SampleMetrics,
)
from hard_negative_mining import (
    find_hard_negatives, get_hard_negative_subset_indices,
    compare_subsets, save_hard_negatives_csv,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_synthetic_spectrum(L: int, n_peaks: int = 5, noise: float = 0.05, seed=None):
    """生成带有高斯峰的合成红外光谱"""
    rng = np.random.RandomState(seed)
    x = np.zeros(L, dtype=np.float32)
    peak_params = []
    for _ in range(n_peaks):
        center = rng.randint(20, L - 20)
        amplitude = rng.uniform(0.5, 3.0)
        width = rng.uniform(3, 12)
        x += amplitude * np.exp(-0.5 * ((np.arange(L) - center) / width) ** 2)
        peak_params.append((center, amplitude, width))
    x += rng.randn(L).astype(np.float32) * noise
    return x, peak_params


def generate_perturbed_spectrum(x_true, peak_params, L, noise=0.08, seed=None):
    """根据真实光谱的峰参数生成略有偏移的重建光谱"""
    rng = np.random.RandomState(seed)
    x_recon = np.zeros(L, dtype=np.float32)
    for center, amplitude, width in peak_params:
        c_shift = center + rng.randint(-3, 4)
        a_shift = amplitude * rng.uniform(0.85, 1.15)
        w_shift = width * rng.uniform(0.9, 1.1)
        x_recon += a_shift * np.exp(-0.5 * ((np.arange(L) - c_shift) / w_shift) ** 2)
    x_recon += rng.randn(L).astype(np.float32) * noise
    return x_recon


def main():
    output_dir = "/tmp/eval_integration_test"
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    N = 30   # 样本数
    L = 400  # 光谱长度

    print("=" * 60)
    print("  Integration Test — Synthetic Data")
    print("=" * 60)

    # ======== 1. 生成合成数据 ========
    print("\n[1/5] Generating synthetic data...")
    all_x_true = []
    all_x_recon = []
    all_masks = []
    all_sample_ids = []
    all_smiles = []

    for i in range(N):
        x_true, peak_params = generate_synthetic_spectrum(L, n_peaks=4, seed=i)
        x_recon = generate_perturbed_spectrum(x_true, peak_params, L, seed=i + 1000)

        # 生成 patch-level mask (patch_size=1 for simplicity)
        mask = np.zeros(L, dtype=np.float32)
        rng = np.random.RandomState(i + 2000)
        masked_idx = rng.choice(L, size=int(L * 0.75), replace=False)
        mask[masked_idx] = 1.0

        # 重建光谱：masked 区域用 x_recon, 其余用 x_true
        x_full_recon = reconstruct_full_spectrum(x_true, x_recon[mask > 0.5], mask, patch_size=1)

        all_x_true.append(x_true)
        all_x_recon.append(x_full_recon)
        all_masks.append(mask)
        all_sample_ids.append(i)
        all_smiles.append(f"C{'C' * (i % 10)}O")

    print(f"  Generated {N} samples, L={L}")

    # ======== 2. 逐样本评估 ========
    print("\n[2/5] Per-sample evaluation...")
    preprocessor = SpectrumPreprocessor(smooth_window=11, smooth_polyorder=3)
    detector = PeakDetector(prominence_min=0.05, distance_min=8, height_min=0.02, width_min=2)

    all_metrics = []
    for i in range(N):
        m = evaluate_sample(
            x_true=all_x_true[i],
            x_recon=all_x_recon[i],
            mask=all_masks[i],
            sample_id=all_sample_ids[i],
            smiles=all_smiles[i],
            patch_size=1,
            preprocessor=preprocessor,
            detector=detector,
            max_match_distance=20,
        )
        all_metrics.append(m)

    # Save CSV
    csv_path = os.path.join(output_dir, "sample_metrics.csv")
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
    print(f"  Saved: {csv_path}")

    # Aggregate
    summary = aggregate_metrics(all_metrics)

    # ======== 3. Hard Negative Mining ========
    print("\n[3/5] Hard negative mining...")
    spectra_matrix = np.array(all_x_true)

    all_hard_negs, pearson_sim, cosine_sim = find_hard_negatives(
        spectra=spectra_matrix,
        sample_ids=all_sample_ids,
        smiles_list=all_smiles,
        top_k=5,
    )

    hn_csv_path = os.path.join(output_dir, "hard_negatives.csv")
    save_hard_negatives_csv(all_hard_negs, hn_csv_path)
    print(f"  Saved: {hn_csv_path}")

    hn_local_indices = get_hard_negative_subset_indices(all_hard_negs, all_sample_ids)
    hn_comparison = compare_subsets(all_metrics, hn_local_indices)
    summary["hard_negative_comparison"] = hn_comparison
    summary["hard_negative_subset_size"] = len(hn_local_indices)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {summary_path}")

    # ======== 4. 可视化 ========
    print("\n[4/5] Generating visualizations...")

    # 4.1 Metrics distribution
    fields = {
        "full_mae": "MAE (Full Spectrum)",
        "full_pearson": "Pearson Correlation",
        "full_peak_pos_error": "Peak Position Error",
        "full_fwhm_error": "FWHM Error",
        "full_intensity_ratio_error": "Intensity Ratio Error",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    for idx, (field, title) in enumerate(fields.items()):
        ax = axes_flat[idx]
        values = [getattr(m, field) for m in all_metrics if getattr(m, field) >= 0]
        if values:
            ax.hist(values, bins=15, color="#3498db", alpha=0.7, edgecolor="white")
            ax.axvline(np.mean(values), color="#e74c3c", linestyle="--",
                       label=f"Mean={np.mean(values):.4f}")
            ax.axvline(np.median(values), color="#2ecc71", linestyle=":",
                       label=f"Median={np.median(values):.4f}")
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=9)
    for idx in range(len(fields), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.suptitle("Validation Metrics Distribution (Synthetic)", fontsize=12)
    plt.tight_layout()
    dist_path = os.path.join(fig_dir, "metrics_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {dist_path}")

    # 4.2 Worst-case spectrum comparison
    sorted_by_mae = sorted(range(N), key=lambda i: -all_metrics[i].full_mae)
    for rank in range(min(3, N)):
        idx = sorted_by_mae[rank]
        m = all_metrics[idx]
        x_true = all_x_true[idx]
        x_recon = all_x_recon[idx]
        mask = all_masks[idx]

        fig, axes2 = plt.subplots(3, 1, figsize=(14, 10),
                                   gridspec_kw={"height_ratios": [3, 1, 1]})
        ax_main = axes2[0]

        # Mask region highlighting
        in_mask = False
        for j in range(L):
            if mask[j] > 0.5 and not in_mask:
                start = j
                in_mask = True
            elif mask[j] < 0.5 and in_mask:
                ax_main.axvspan(start, j, alpha=0.12, color="red")
                in_mask = False
        if in_mask:
            ax_main.axvspan(start, L, alpha=0.12, color="red")

        ax_main.plot(x_true, color="#2c3e50", linewidth=1.2, label="True", alpha=0.9)
        ax_main.plot(x_recon, color="#e74c3c", linewidth=1.0, label="Reconstructed",
                     alpha=0.8, linestyle="--")

        # Mark peaks
        peak_result = analyze_peaks(x_true, x_recon, preprocessor, detector)
        for tp in peak_result.true_peaks:
            ax_main.plot(tp.position, x_true[tp.position], "v", color="#3498db", markersize=6)
        for rp in peak_result.recon_peaks:
            ax_main.plot(rp.position, x_recon[rp.position], "^", color="#e74c3c",
                         markersize=5, alpha=0.7)
        # FWHM bars
        for mp in peak_result.matched_pairs:
            tp = mp.true_peak
            if tp.fwhm > 0 and tp.left_base >= 0:
                half_h = (tp.height + (tp.height - tp.prominence)) / 2
                ax_main.hlines(half_h, tp.left_base, tp.right_base,
                               color="#3498db", alpha=0.5, linewidth=1.5)

        ax_main.set_title(
            f"Worst-{rank+1} | Sample {m.sample_id} | "
            f"MAE={m.full_mae:.4f} | Pearson={m.full_pearson:.4f} | "
            f"Peaks: {m.n_matched_peaks} matched, {m.n_missed_peaks} missed",
            fontsize=9,
        )
        ax_main.legend(fontsize=8)
        ax_main.set_ylabel("Intensity (std)")

        # Residual
        residual = x_recon - x_true
        axes2[1].fill_between(range(L), residual, 0, alpha=0.4, color="#e74c3c")
        axes2[1].axhline(0, color="black", linewidth=0.5)
        axes2[1].set_ylabel("Residual")

        # Mask bar
        axes2[2].imshow(mask.reshape(1, -1), aspect="auto", cmap="Reds", vmin=0, vmax=1)
        axes2[2].set_yticks([])
        axes2[2].set_xlabel("Spectral point index")

        plt.tight_layout()
        worst_path = os.path.join(fig_dir, f"worst_{rank+1}_sample_{m.sample_id}.png")
        plt.savefig(worst_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {worst_path}")

    # 4.3 Hard negative pairs
    pair_data = []
    for i, hn_list in enumerate(all_hard_negs):
        if hn_list:
            hn = hn_list[0]
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
    pair_data.sort(key=lambda d: -d["pearson_sim"])

    n_pairs = min(5, len(pair_data))
    if n_pairs > 0:
        fig, axes3 = plt.subplots(n_pairs, 1, figsize=(14, 3 * n_pairs))
        if n_pairs == 1:
            axes3 = [axes3]
        for pi in range(n_pairs):
            d = pair_data[pi]
            axes3[pi].plot(d["query_true"], label=f"Query #{d['query_id']}", alpha=0.8)
            axes3[pi].plot(d["neg_true"], label=f"Hard Neg #{d['neg_id']}",
                           alpha=0.8, linestyle="--")
            axes3[pi].set_title(f"Pearson sim = {d['pearson_sim']:.4f}", fontsize=9)
            axes3[pi].legend(fontsize=8)
        axes3[-1].set_xlabel("Spectral point index")
        plt.suptitle("Hardest Negative Pairs", fontsize=11)
        plt.tight_layout()
        hn_fig_path = os.path.join(fig_dir, "hardest_negative_pairs.png")
        plt.savefig(hn_fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {hn_fig_path}")

    # ======== 5. Report ========
    print("\n[5/5] Final report:")
    print("=" * 60)
    print("  VALIDATION EVALUATION SUMMARY (Synthetic)")
    print("=" * 60)
    for field in ["full_mae", "full_pearson", "full_peak_pos_error",
                   "full_fwhm_error", "full_intensity_ratio_error"]:
        stats = summary.get(field, {})
        if stats.get("mean") is not None:
            print(f"  {field:35s}: "
                  f"mean={stats['mean']:.4f} +/- {stats['std']:.4f}, "
                  f"median={stats['median']:.4f}, "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}], "
                  f"n={stats['n_valid']}")

    ps = summary.get("peak_stats", {})
    print(f"\n  Peak Statistics:")
    print(f"    Total true peaks:  {ps.get('total_true_peaks', 'N/A')}")
    print(f"    Total recon peaks: {ps.get('total_recon_peaks', 'N/A')}")
    print(f"    Match rate:        {ps.get('match_rate', 0):.4f}")
    print(f"    Precision:         {ps.get('precision', 0):.4f}")

    print(f"\n  Hard Negative Subset Size: {summary.get('hard_negative_subset_size', 'N/A')}")

    hn_comp = summary.get("hard_negative_comparison", {})
    if hn_comp:
        print(f"\n  Full-set vs Hard-neg subset comparison:")
        for field in ["full_mae", "full_pearson"]:
            fc = hn_comp.get(field, {})
            full = fc.get("full_set", {})
            hn = fc.get("hard_neg_subset", {})
            print(f"    {field}: full_mean={full.get('mean', 'N/A'):.4f}, "
                  f"hn_mean={hn.get('mean', 'N/A'):.4f}")

    print(f"\n  Worst-5 by MAE:")
    for w in summary.get("worst_5_by_mae", []):
        print(f"    ID={w['sample_id']}, MAE={w['full_mae']:.4f}")

    print(f"\n  All outputs in: {output_dir}")
    print("=" * 60)
    print("\nIntegration test PASSED.")


if __name__ == "__main__":
    main()
