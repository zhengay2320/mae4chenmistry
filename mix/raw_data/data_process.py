import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 配置区：按需修改
# =========================
INPUT_DIR = Path("multimodal_spectroscopic_dataset")   # parquet 数据集目录
OUTPUT_DIR = Path("ir_json_output")                    # 新的输出目录
SAVE_PRETTY_JSON = False                               # 是否格式化输出 JSON（True 更易读，False 文件更小）


# =========================
# 工具函数
# =========================
def to_jsonable(obj):
    """
    将 numpy / pandas 对象递归转换为可写入 JSON 的原生 Python 类型
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    elif isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    else:
        return obj


def write_json(data, out_path: Path, pretty: bool = False):
    """
    写 JSON 文件
    """
    with open(out_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False)


# =========================
# 主逻辑
# =========================
def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(INPUT_DIR.glob("aligned_chunk_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"没有找到 aligned_chunk_*.parquet 文件: {INPUT_DIR.resolve()}")

    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")

    # -------------------------
    # 全局统计量初始化
    # -------------------------
    global_min = np.inf
    global_max = -np.inf
    total_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0

    valid_molecule_count = 0      # 成功参与 IR 统计的分子数量
    skipped_molecule_count = 0    # 没有可用 IR 的分子数量

    # -------------------------
    # 逐文件处理
    # -------------------------
    for parquet_path in parquet_files:
        print(f"\n正在处理: {parquet_path.name}")

        try:
            # 只读取需要的两列，减少内存占用
            df = pd.read_parquet(parquet_path, columns=["smiles", "ir_spectra"])
        except Exception as e:
            print(f"读取失败: {parquet_path.name}, 错误: {e}")
            continue

        records = []

        for _, row in df.iterrows():
            smiles = row["smiles"]
            ir_data = row["ir_spectra"]

            # 保存 JSON 记录
            record = {
                "smiles": smiles,
                "ir_spectra": to_jsonable(ir_data)
            }
            records.append(record)

            # -------------------------
            # 统计 IR 全局数值信息
            # -------------------------
            try:
                arr = np.asarray(ir_data, dtype=np.float64).reshape(-1)

                # 过滤 NaN / inf
                arr = arr[np.isfinite(arr)]

                if arr.size == 0:
                    skipped_molecule_count += 1
                    continue

                valid_molecule_count += 1

                # 更新最小值、最大值
                arr_min = arr.min()
                arr_max = arr.max()
                if arr_min < global_min:
                    global_min = arr_min
                if arr_max > global_max:
                    global_max = arr_max

                # 更新总数、总和、平方和
                total_count += arr.size
                total_sum += arr.sum()
                total_sum_sq += np.square(arr).sum()

            except Exception:
                skipped_molecule_count += 1
                continue

        # 每个 parquet 输出一个同名 json
        out_path = OUTPUT_DIR / f"{parquet_path.stem}.json"
        write_json(records, out_path, pretty=SAVE_PRETTY_JSON)

        print(f"已保存: {out_path.name} | 分子数: {len(records)}")

    # -------------------------
    # 汇总全局统计量
    # -------------------------
    if total_count == 0:
        stats = {
            "message": "没有统计到任何有效的 ir_spectra 数值",
            "valid_molecule_count": valid_molecule_count,
            "skipped_molecule_count": skipped_molecule_count,
            "total_value_count": 0
        }
    else:
        mean = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean ** 2)

        # 数值误差保护
        variance = max(0.0, variance)

        stats = {
            "valid_molecule_count": int(valid_molecule_count),
            "skipped_molecule_count": int(skipped_molecule_count),
            "total_value_count": int(total_count),
            "global_min": float(global_min),
            "global_max": float(global_max),
            "global_mean": float(mean),
            "global_variance": float(variance)
        }

    stats_path = OUTPUT_DIR / "ir_global_stats.json"
    write_json(stats, stats_path, pretty=True)

    print("\n全部处理完成")
    print(f"IR 全局统计已保存: {stats_path.resolve()}")
    print("\n统计结果如下:")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
