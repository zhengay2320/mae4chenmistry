#!/bin/bash
set -euo pipefail

output_folder=""

while getopts "o:" opt; do
  case $opt in
    o) output_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

if [[ -z "${output_folder}" ]]; then
  echo "Usage: $0 -o <output_folder>" >&2
  exit 1
fi

# ==========================================
if ! command -v aria2c &> /dev/null; then
    echo "Error: aria2c could not be found."
    echo "Please install it using: conda install -c conda-forge aria2"
    exit 1
fi
# ==========================================

export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

mkdir -p "${output_folder}/raw_data"

# 定义文件名和下载目录
download_dir="${output_folder}/raw_data"
zip_filename="mm_dataset.zip"
zip_path="${download_dir}/${zip_filename}"
url="https://zenodo.org/records/14770232/files/multimodal_spectroscopic_dataset.zip?download=1"

echo "Downloading Multimodal Dataset using aria2c (16 threads)..."

# ==========================================
# 使用 aria2c 进行多线程下载
# -x 16: 16个连接数
# -s 16: 16个分片
# -c: 断点续传
# -d: 下载目录
# -o: 输出文件名
aria2c -x 16 -s 16 -k 1M -c -d "${download_dir}" -o "${zip_filename}" "${url}"
# ==========================================

echo "Extracting archive"
if ! unzip -q "${zip_path}" -d "${output_folder}/raw_data/"; then
  echo "Unzip failed, deleting corrupted archive" >&2
  exit 1
fi

echo "Done!"