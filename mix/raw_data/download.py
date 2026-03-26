import os
import subprocess
import argparse
import shutil
import sys


def run_command(command, shell=True):
    """封装子进程调用，模拟 set -e 的效果"""
    try:
        subprocess.run(command, shell=shell, check=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {e}")
        sys.exit(1)


def main():
    # 1. 参数解析 (替代 getopts)
    parser = argparse.ArgumentParser(description="下载并解压多模态遥感数据集")
    parser.add_argument("-o", "--output_folder",default=r'E:\MAE_data\mix\raw_data\data', help="输出根目录")
    args = parser.parse_args()

    output_folder = args.output_folder
    download_dir = os.path.join(output_folder, "raw_data")
    zip_filename = "mm_dataset.zip"
    zip_path = os.path.join(download_dir, zip_filename)
    url = "https://zenodo.org/records/14770232/files/multimodal_spectroscopic_dataset.zip?download=1"

    # 2. 检查 aria2c 是否安装
    if not shutil.which("aria2c"):
        print("Error: aria2c could not be found.")
        print("Please install it using: conda install -c conda-forge aria2")
        sys.exit(1)

    # 3. 准备目录与环境变量
    os.makedirs(download_dir, exist_ok=True)
    os.environ["UNZIP_DISABLE_ZIPBOMB_DETECTION"] = "TRUE"

    # 4. 使用 aria2c 下载 (16线程)
    print(f"🚀 正在下载数据集至 {download_dir} (16 threads)...")
    aria_cmd = (
        f"aria2c -x 16 -s 16 -k 1M -c "
        f"-d '{download_dir}' -o '{zip_filename}' '{url}'"
    )
    run_command(aria_cmd)

    # 5. 解压文件 (替代 unzip)
    print("📦 正在解压存档...")
    unzip_cmd = f"unzip -q '{zip_path}' -d '{download_dir}/'"

    try:
        run_command(unzip_cmd)
    except SystemExit:
        if os.path.exists(zip_path):
            print("⚠️ 解压失败，正在删除损坏的存档...")
            os.remove(zip_path)
        sys.exit(1)

    print("✅ 任务完成!")


if __name__ == "__main__":
    main()