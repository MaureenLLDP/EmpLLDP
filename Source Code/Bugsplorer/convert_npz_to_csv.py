import numpy as np
import pandas as pd
import argparse

def convert_npz_to_csv(npz_file_path, csv_file_path):
    # 读取.npz文件
    data = np.load(npz_file_path)

    # 提取数据
    true_prob = data['true_prob']
    labels = data['labels']

    # 将数据转换为Pandas DataFrame
    df = pd.DataFrame({
        'true_prob': true_prob,
        'labels': labels
    })

    # 保存为.csv文件
    df.to_csv(csv_file_path, index=False)

    print(f"Data has been saved to {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npz file to .csv file.")
    parser.add_argument("npz_file", help="Path to the input .npz file")
    parser.add_argument("csv_file", help="Path to the output .csv file")
    args = parser.parse_args()

    convert_npz_to_csv(args.npz_file, args.csv_file)