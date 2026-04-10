import pickle
import os
import json
import numpy as np
def preprocess_and_save(input_path1, output_path):
    """
    对原始数据进行预处理，并保存到硬盘
    """
    # TODO: 根据你的原始数据修改这里的加载逻辑
    file_list = [file_name for file_name in os.listdir(input_path1)
                 if file_name.endswith('.json') and os.path.isfile(os.path.join(input_path1, file_name))]
    total_data = []
    for file_name in file_list:
        file_path = os.path.join(input_path1, file_name)
        print(f'正在处理{file_path}')
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                for entry in json_data:
                    smiles = entry['smiles']
                    ir_spectra = np.array(entry['ir_spectra'], dtype=np.float32)
                    total_data.append((smiles, ir_spectra))
        except Exception as e:
            print(f"无法加载文件 {file_name}: {e}")
    #
    # file_list2 = [file_name for file_name in os.listdir(input_path2)
    #              if file_name.endswith('.json') and os.path.isfile(os.path.join(input_path2, file_name))]
    # for file_name in file_list2:
    #     file_path = os.path.join(input_path2, file_name)
    #     print(f'正在处理{file_path}')
    #     try:
    #         with open(file_path, 'r') as f:
    #             json_data = json.load(f)
    #             for entry in json_data:
    #                 smiles = entry['smiles']
    #                 ir_spectra = np.array(entry['ir_spectra'], dtype=np.float32)
    #                 total_data.append((smiles, ir_spectra))
    #     except Exception as e:
    #         print(f"无法加载文件 {file_name}: {e}")

    # 保存到硬盘
    print(f"Saving processed data to {output_path} ...")
    with open(output_path, 'wb') as f:
        pickle.dump(total_data, f)
    print("Done.")


if __name__ == "__main__":
    preprocess_and_save(
        "/home/ubuntu/data/zhenganyang/MAE_data/mix/raw_data/ir_json_dataset/val",
        "/home/ubuntu/data/zhenganyang/MAE_data/mix/raw_data/ir_json_dataset/val.pkl")
