#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def count_npz_rows(npz_path):
    """
    统计.npz文件中所有数组的行数（第一个维度）
    返回格式: {文件名: {数组名: 行数}}
    """
    try:
        with np.load(npz_path) as data:
            return {
                npz_path.name: {
                    arr_name: arr.shape[0] if arr.ndim >= 1 else 1
                    for arr_name, arr in data.items()
                }
            }
    except Exception as e:
        return {npz_path.name: {"error": str(e)}}

def batch_count_npz(directory):
    """
    批量统计目录下所有.npz文件的行数
    """
    npz_files = Path(directory).glob("*.npz")
    results = {}
    
    for npz_file in npz_files:
        results.update(count_npz_rows(npz_file))
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="统计.npz文件的行数")
    parser.add_argument("path", help=".npz文件或目录路径")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file() and path.suffix == ".npz":
        results = count_npz_rows(path)
    elif path.is_dir():
        results = batch_count_npz(path)
    else:
        raise ValueError("输入路径必须是.npz文件或包含.npz的目录")

    # 打印结果
    print("\n.npz文件行数统计:")
    for filename, arr_info in results.items():
        print(f"\n文件: {filename}")
        for arr_name, rows in arr_info.items():
            print(f"  {arr_name}: {rows}行")

    # 保存结果到CSV
    output_csv = "npz_row_counts.csv"
    with open(output_csv, "w") as f:
        f.write("filename,array_name,rows\n")
        for filename, arr_info in results.items():
            for arr_name, rows in arr_info.items():
                f.write(f"{filename},{arr_name},{rows}\n")
    print(f"\n结果已保存到: {output_csv}")