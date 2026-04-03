import os
import re
import time
import logging
from pathlib import Path

# =========================
# 配置区
# =========================
CHECKPOINT_DIR = "/home/rspip/zay/mae/mae-main/output_dir"   # 改成你的目录
MAX_KEEP = 10
CHECK_INTERVAL_SECONDS = 600   # 10分钟

# 只匹配 checkpoint-数字.pth
CKPT_PATTERN = re.compile(r"^checkpoint-(\d+)\.pth$")

# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("checkpoint_cleaner.log")
    ]
)
logger = logging.getLogger(__name__)


def get_checkpoint_files(checkpoint_dir: Path):
    """
    扫描目录中所有符合 checkpoint-数字.pth 的文件。
    返回:
        List[Tuple[Path, int]]
        [(文件路径, epoch编号), ...]
    """
    checkpoint_files = []

    for entry in checkpoint_dir.iterdir():
        if not entry.is_file():
            continue

        match = CKPT_PATTERN.match(entry.name)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((entry, epoch_num))

    return checkpoint_files


def cleanup_old_checkpoints(checkpoint_dir: Path, max_keep: int = 10):
    """
    如果 checkpoint 数量超过 max_keep，则按 epoch 从小到大删除最旧的。
    """
    checkpoint_files = get_checkpoint_files(checkpoint_dir)

    if not checkpoint_files:
        logger.info("未发现任何符合规则的 checkpoint 文件。")
        return

    # 按 epoch 从小到大排序
    checkpoint_files.sort(key=lambda x: x[1])

    logger.info(
        f"当前共发现 {len(checkpoint_files)} 个 checkpoint 文件: "
        f"{[f'{p.name}(epoch={e})' for p, e in checkpoint_files]}"
    )

    if len(checkpoint_files) <= max_keep:
        logger.info(f"checkpoint 数量未超过上限 {max_keep}，无需删除。")
        return

    num_to_delete = len(checkpoint_files) - max_keep
    files_to_delete = checkpoint_files[:num_to_delete]

    for file_path, epoch_num in files_to_delete:
        try:
            os.remove(file_path)
            logger.info(f"已删除 checkpoint: {file_path.name} (epoch={epoch_num})")
        except Exception as e:
            logger.error(f"删除失败: {file_path.name} (epoch={epoch_num}), error={e}")

    remaining_files = get_checkpoint_files(checkpoint_dir)
    remaining_files.sort(key=lambda x: x[1])

    logger.info(
        f"清理完成，当前保留 {len(remaining_files)} 个 checkpoint 文件: "
        f"{[f'{p.name}(epoch={e})' for p, e in remaining_files]}"
    )


def main():
    checkpoint_dir = Path(CHECKPOINT_DIR)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"目录不存在: {checkpoint_dir}")

    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"不是目录: {checkpoint_dir}")

    logger.info(f"开始监控目录: {checkpoint_dir}")
    logger.info(
        f"每 {CHECK_INTERVAL_SECONDS} 秒检查一次，只保留最新 {MAX_KEEP} 个 checkpoint，"
        f"删除规则：按 epoch 编号从小到大删除。"
    )

    while True:
        try:
            cleanup_old_checkpoints(checkpoint_dir, MAX_KEEP)
        except Exception as e:
            logger.error(f"检查过程中出现异常: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
