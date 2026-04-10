"""
peak_analysis.py — 光谱峰分析模块

功能：
  1. 光谱预处理（平滑、可选基线校正、归一化）
  2. 峰检测（含参数化过滤规则）
  3. 峰匹配（真实峰 vs 重建峰，最近邻贪心）
  4. FWHM 计算
  5. Peak intensity ratio 计算

所有峰指标均在标准化光谱空间计算，位置和宽度以采样点索引为单位。

科学性声明：
  - 峰检测在平滑 + min-max 归一化后的光谱上执行，以保证稳定性
  - FWHM 在平滑后的标准化光谱上用线性插值计算
  - Peak intensity 使用标准化光谱上的峰高（减去局部基线 = prominence）
  - 异常峰过滤规则在函数文档中明确说明
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class PeakInfo:
    """单个峰的完整信息"""
    position: int           # 峰位索引
    height: float           # 峰高（标准化空间）
    prominence: float       # 显著性
    fwhm: float             # 半峰全宽（采样点数），-1 表示计算失败
    left_base: float        # 左半高位置
    right_base: float       # 右半高位置


@dataclass
class PeakMatchResult:
    """一对匹配峰的结果"""
    true_peak: PeakInfo
    recon_peak: PeakInfo
    position_error: float   # |pos_true - pos_recon|
    fwhm_error: float       # |fwhm_true - fwhm_recon|, -1 if either invalid
    intensity_ratio_error: float  # |I_recon/I_true - 1|, -1 if I_true too small


@dataclass
class PeakAnalysisResult:
    """单个样本的峰分析汇总"""
    true_peaks: List[PeakInfo]
    recon_peaks: List[PeakInfo]
    matched_pairs: List[PeakMatchResult]
    missed_peaks: List[PeakInfo]      # 真实峰未匹配到重建峰
    spurious_peaks: List[PeakInfo]    # 重建峰无对应真实峰
    # 聚合指标
    mean_peak_pos_error: float
    mean_fwhm_error: float
    mean_intensity_ratio_error: float
    n_true: int
    n_recon: int
    n_matched: int
    n_missed: int
    n_spurious: int


# ============================================================
# 预处理
# ============================================================

class SpectrumPreprocessor:
    """
    光谱预处理流水线。

    步骤：
      1. Savitzky-Golay 平滑（消除高频噪声）
      2. (可选) Baseline correction
      3. (可选) Min-max 归一化到 [0, 1]（仅用于峰检测阈值）
    """

    def __init__(
        self,
        smooth_window: int = 11,
        smooth_polyorder: int = 3,
        do_baseline: bool = False,
        do_minmax: bool = True,
    ):
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder
        self.do_baseline = do_baseline
        self.do_minmax = do_minmax

    def smooth(self, spectrum: np.ndarray) -> np.ndarray:
        """Savitzky-Golay 平滑"""
        if len(spectrum) < self.smooth_window:
            return spectrum.copy()
        return savgol_filter(spectrum, self.smooth_window, self.smooth_polyorder)

    def baseline_correction_als(
        self, spectrum: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10
    ) -> np.ndarray:
        """
        Asymmetric Least Squares baseline estimation (Eilers & Boelens, 2005).
        返回 baseline-corrected 光谱。
        """
        from scipy.sparse import diags, csc_matrix
        from scipy.sparse.linalg import spsolve

        L = len(spectrum)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = csc_matrix(D)
        w = np.ones(L)
        for _ in range(n_iter):
            W = diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.T)
            z = spsolve(Z, w * spectrum)
            w = p * (spectrum > z) + (1 - p) * (spectrum <= z)
        return spectrum - z

    def minmax_normalize(self, spectrum: np.ndarray) -> np.ndarray:
        """Min-max 归一化到 [0, 1]"""
        vmin, vmax = spectrum.min(), spectrum.max()
        if vmax - vmin < 1e-10:
            return np.zeros_like(spectrum)
        return (spectrum - vmin) / (vmax - vmin)

    def preprocess(self, spectrum: np.ndarray) -> np.ndarray:
        """完整预处理流水线"""
        s = self.smooth(spectrum)
        if self.do_baseline:
            s = self.baseline_correction_als(s)
        if self.do_minmax:
            s = self.minmax_normalize(s)
        return s

    def preprocess_for_detection(self, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回两个版本：
          - smoothed: 平滑后（标准化空间），用于 FWHM 和强度计算
          - normalized: 平滑 + minmax 归一化，用于峰检测
        """
        smoothed = self.smooth(spectrum)
        if self.do_baseline:
            smoothed = self.baseline_correction_als(smoothed)
        normalized = self.minmax_normalize(smoothed)
        return smoothed, normalized


# ============================================================
# 峰检测
# ============================================================

class PeakDetector:
    """
    基于 scipy.signal.find_peaks 的峰检测器。

    参数说明：
      - prominence_min: 峰在归一化光谱上的最小显著性 (0~1 scale)
      - distance_min: 相邻峰最小间距（采样点数）
      - height_min: 峰的最小绝对高度（归一化空间）
      - width_min: 峰的最小宽度（采样点数）
      - fwhm_max_ratio: FWHM > 光谱长度 * ratio 视为异常峰
    """

    def __init__(
        self,
        prominence_min: float = 0.05,
        distance_min: int = 10,
        height_min: float = 0.02,
        width_min: int = 2,
        fwhm_max_ratio: float = 0.25,
    ):
        self.prominence_min = prominence_min
        self.distance_min = distance_min
        self.height_min = height_min
        self.width_min = width_min
        self.fwhm_max_ratio = fwhm_max_ratio

    def detect(
        self,
        spectrum_smoothed: np.ndarray,
        spectrum_normalized: np.ndarray,
    ) -> List[PeakInfo]:
        """
        在归一化光谱上检测峰，然后在平滑光谱上计算 FWHM 和强度。

        参数:
            spectrum_smoothed: 平滑后的标准化光谱（用于 FWHM/强度）
            spectrum_normalized: 平滑 + minmax 归一化的光谱（用于检测阈值）

        返回:
            List[PeakInfo]
        """
        # 在归一化光谱上检测
        peaks_idx, properties = find_peaks(
            spectrum_normalized,
            prominence=self.prominence_min,
            distance=self.distance_min,
            height=self.height_min,
            width=self.width_min,
        )

        if len(peaks_idx) == 0:
            return []

        # 在平滑光谱（标准化空间）上计算 FWHM
        L = len(spectrum_smoothed)
        max_fwhm = L * self.fwhm_max_ratio

        peak_infos = []
        for i, pos in enumerate(peaks_idx):
            height = float(spectrum_smoothed[pos])
            prom = float(properties["prominences"][i])

            # 计算 FWHM（在平滑标准化光谱上）
            fwhm_val, left_b, right_b = self._compute_fwhm(spectrum_smoothed, pos)

            # 异常 FWHM 过滤
            if fwhm_val > max_fwhm:
                fwhm_val = -1.0

            peak_infos.append(PeakInfo(
                position=int(pos),
                height=height,
                prominence=prom,
                fwhm=fwhm_val,
                left_base=left_b,
                right_base=right_b,
            ))

        return peak_infos

    def _compute_fwhm(
        self, spectrum: np.ndarray, peak_pos: int
    ) -> Tuple[float, float, float]:
        """
        在光谱上对指定峰位计算 FWHM。
        使用线性插值精确定位半高点。

        返回: (fwhm, left_half_pos, right_half_pos)
               如果计算失败返回 (-1, -1, -1)
        """
        L = len(spectrum)
        peak_val = spectrum[peak_pos]

        # 寻找峰两侧的局部最小值作为基线参考
        # 向左找局部最小
        left_min_val = peak_val
        for j in range(peak_pos - 1, -1, -1):
            if spectrum[j] < left_min_val:
                left_min_val = spectrum[j]
            if spectrum[j] > spectrum[j + 1] and j < peak_pos - 2:
                break

        # 向右找局部最小
        right_min_val = peak_val
        for j in range(peak_pos + 1, L):
            if spectrum[j] < right_min_val:
                right_min_val = spectrum[j]
            if spectrum[j] > spectrum[j - 1] and j > peak_pos + 2:
                break

        base_val = (left_min_val + right_min_val) / 2.0
        half_height = (peak_val + base_val) / 2.0

        # 向左搜索半高点
        left_pos = -1.0
        for j in range(peak_pos, 0, -1):
            if spectrum[j - 1] <= half_height <= spectrum[j]:
                # 线性插值
                denom = spectrum[j] - spectrum[j - 1]
                if abs(denom) < 1e-12:
                    left_pos = float(j)
                else:
                    left_pos = j - 1 + (half_height - spectrum[j - 1]) / denom
                break

        # 向右搜索半高点
        right_pos = -1.0
        for j in range(peak_pos, L - 1):
            if spectrum[j + 1] <= half_height <= spectrum[j]:
                denom = spectrum[j] - spectrum[j + 1]
                if abs(denom) < 1e-12:
                    right_pos = float(j)
                else:
                    right_pos = j + (spectrum[j] - half_height) / denom
                break

        if left_pos < 0 or right_pos < 0:
            return (-1.0, -1.0, -1.0)

        fwhm = right_pos - left_pos
        return (fwhm, left_pos, right_pos)


# ============================================================
# 峰匹配
# ============================================================

def match_peaks(
    true_peaks: List[PeakInfo],
    recon_peaks: List[PeakInfo],
    max_match_distance: int = 20,
) -> Tuple[List[PeakMatchResult], List[PeakInfo], List[PeakInfo]]:
    """
    最近邻贪心峰匹配。

    规则：
      1. 以 true_peaks 为参考集
      2. 对每个真实峰，在未匹配的重建峰中找位置最近的
      3. 若距离 > max_match_distance，则标记为 missed
      4. 剩余未匹配的重建峰标记为 spurious

    参数:
        true_peaks: 真实光谱峰列表
        recon_peaks: 重建光谱峰列表
        max_match_distance: 最大匹配距离（采样点数）

    返回:
        (matched_pairs, missed_peaks, spurious_peaks)
    """
    matched_pairs = []
    missed_peaks = []
    used_recon = set()

    # 按真实峰的 prominence 从大到小排序（优先匹配显著峰）
    sorted_true = sorted(enumerate(true_peaks), key=lambda x: -x[1].prominence)

    for orig_idx, tp in sorted_true:
        best_dist = float("inf")
        best_rp_idx = -1

        for ri, rp in enumerate(recon_peaks):
            if ri in used_recon:
                continue
            dist = abs(tp.position - rp.position)
            if dist < best_dist:
                best_dist = dist
                best_rp_idx = ri

        if best_rp_idx >= 0 and best_dist <= max_match_distance:
            rp = recon_peaks[best_rp_idx]
            used_recon.add(best_rp_idx)

            # 计算 Peak position error
            pos_err = abs(tp.position - rp.position)

            # 计算 FWHM error
            if tp.fwhm > 0 and rp.fwhm > 0:
                fwhm_err = abs(tp.fwhm - rp.fwhm)
            else:
                fwhm_err = -1.0

            # 计算 Intensity ratio error
            #   |I_recon / I_true - 1|
            #   当 I_true 接近 0 时不计算
            INTENSITY_FLOOR = 0.01
            if abs(tp.prominence) > INTENSITY_FLOOR:
                ratio_err = abs(rp.prominence / tp.prominence - 1.0)
            else:
                ratio_err = -1.0

            matched_pairs.append(PeakMatchResult(
                true_peak=tp,
                recon_peak=rp,
                position_error=pos_err,
                fwhm_error=fwhm_err,
                intensity_ratio_error=ratio_err,
            ))
        else:
            missed_peaks.append(tp)

    spurious_peaks = [rp for ri, rp in enumerate(recon_peaks) if ri not in used_recon]

    return matched_pairs, missed_peaks, spurious_peaks


# ============================================================
# 单样本峰分析入口
# ============================================================

def analyze_peaks(
    x_true: np.ndarray,
    x_recon: np.ndarray,
    preprocessor: Optional[SpectrumPreprocessor] = None,
    detector: Optional[PeakDetector] = None,
    max_match_distance: int = 20,
) -> PeakAnalysisResult:
    """
    对一对 (x_true, x_recon) 执行完整峰分析。

    参数:
        x_true: 真实光谱 (标准化空间, 1D array)
        x_recon: 重建光谱 (标准化空间, 1D array)
        preprocessor: 预处理器 (默认使用默认参数)
        detector: 峰检测器 (默认使用默认参数)
        max_match_distance: 峰匹配最大距离

    返回:
        PeakAnalysisResult
    """
    if preprocessor is None:
        preprocessor = SpectrumPreprocessor()
    if detector is None:
        detector = PeakDetector()

    # 预处理
    true_smooth, true_norm = preprocessor.preprocess_for_detection(x_true)
    recon_smooth, recon_norm = preprocessor.preprocess_for_detection(x_recon)

    # 峰检测
    true_peaks = detector.detect(true_smooth, true_norm)
    recon_peaks = detector.detect(recon_smooth, recon_norm)

    # 峰匹配
    matched, missed, spurious = match_peaks(true_peaks, recon_peaks, max_match_distance)

    # 聚合指标
    valid_pos_errors = [m.position_error for m in matched]
    valid_fwhm_errors = [m.fwhm_error for m in matched if m.fwhm_error >= 0]
    valid_ratio_errors = [m.intensity_ratio_error for m in matched if m.intensity_ratio_error >= 0]

    mean_pos = float(np.mean(valid_pos_errors)) if valid_pos_errors else -1.0
    mean_fwhm = float(np.mean(valid_fwhm_errors)) if valid_fwhm_errors else -1.0
    mean_ratio = float(np.mean(valid_ratio_errors)) if valid_ratio_errors else -1.0

    return PeakAnalysisResult(
        true_peaks=true_peaks,
        recon_peaks=recon_peaks,
        matched_pairs=matched,
        missed_peaks=missed,
        spurious_peaks=spurious,
        mean_peak_pos_error=mean_pos,
        mean_fwhm_error=mean_fwhm,
        mean_intensity_ratio_error=mean_ratio,
        n_true=len(true_peaks),
        n_recon=len(recon_peaks),
        n_matched=len(matched),
        n_missed=len(missed),
        n_spurious=len(spurious),
    )
