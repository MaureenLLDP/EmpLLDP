import sys
import os
import pandas as pd
import time
import numpy as np
import warnings
from itertools import chain
from tqdm import tqdm

# --- 路径修复 ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.append(project_root)

from script.my_util import *

warnings.filterwarnings("ignore")

# --- 路径定义 (使用os.path.join更稳健) ---
script_dir = os.path.dirname(current_script_path)
save_path = os.path.join(script_dir, 'data')
used_file_data_path = os.path.join(project_root, 'used_file_data')
source_code_path = os.path.join(project_root, 'sourcecode')


def readData(path, file):
    full_path = os.path.join(path, file)
    try:
        return pd.read_csv(full_path, encoding='latin', keep_default_na=False)
    except FileNotFoundError:
        print(f"警告: 文件未找到 {full_path}, 跳过。")
        return None

def toTUDA_optimized(line_data: pd.DataFrame, project, release, method='lineflow'):
    save_pre = os.path.join(save_path, project, release, 'raw')
    os.makedirs(save_pre, exist_ok=True)

    files_list = line_data['filename'].unique()
    doc2vec_dict, pdg_dict, edge_label_dict = {}, {}, {}
    
    print("步骤1: 预加载文件...")
    for file_name in tqdm(files_list, desc="Pre-loading files"):
        try:
            folder_base = os.path.join(source_code_path, project.lower(), release, file_name.replace('.java', ''))
            word2vec_path = f"{folder_base}_{method}.doc2vec"
            doc2vec_dict[file_name] = np.loadtxt(word2vec_path, delimiter=',')
            
            pdg_path = os.path.join(source_code_path, project, release, file_name.replace('.java', ''))
            pdg_dict[file_name] = np.atleast_2d(np.loadtxt(pdg_path + '_pdg.txt'))
            
            # 修正：确保即使只有一个数字也能读为1D数组
            edge_label_dict[file_name] = np.atleast_1d(np.loadtxt(pdg_path + '_edge_label.txt'))

        except Exception as e:
            print(f"\n警告: 加载 {file_name} 的依赖文件失败: {e}。将从处理中排除此文件。")
            if file_name in doc2vec_dict: del doc2vec_dict[file_name]
            if file_name in pdg_dict: del pdg_dict[file_name]
            if file_name in edge_label_dict: del edge_label_dict[file_name]
            continue
    
    valid_files = list(doc2vec_dict.keys())
    if not valid_files:
        print("没有成功加载任何文件，处理终止。")
        return
    line_data = line_data[line_data['filename'].isin(valid_files)]

    node_attributes_list, DS_A_list, edge_labels_list = [], [], []
    graph_labels, graph_indicators, node_labels, node_types_all, node_ids = [], [], [], [], []
    
    node_offset = 0
    graph_idx = 1

    print("\n步骤2: 处理数据...")
    for file_name, file_df in tqdm(line_data.groupby('filename', sort=False), desc="Processing files"):
        num_nodes_in_file = len(file_df)
        is_buggy_file = file_df['file-label'].any()
        graph_labels.append(1 if is_buggy_file else 0)
        vector = doc2vec_dict.get(file_name)
        if vector is not None:
            node_attributes_list.append(vector)
        graph_indicators.extend([graph_idx] * num_nodes_in_file)
        graph_idx += 1
        node_labels.extend(file_df['line-label'].astype(int).tolist())
        node_types_all.extend(file_df['node_label'].astype(int).tolist())
        node_ids.extend([f"{release}::{file_name}::{int(line)}" for line in file_df['line_number']])
        pdg = pdg_dict.get(file_name)
        if pdg is not None and pdg.size > 0:
            DS_A_list.append(pdg + node_offset)
        edge_label = edge_label_dict.get(file_name)
        if edge_label is not None and edge_label.size > 0:
            edge_labels_list.append(edge_label)
        node_offset += num_nodes_in_file

    print("\n步骤3: 合并数据...")
    node_attributes = np.vstack(node_attributes_list) if node_attributes_list else np.array([])
    DS_A = np.vstack(DS_A_list) if DS_A_list else np.array([])
    edge_labels = np.concatenate(edge_labels_list) if edge_labels_list else np.array([])
    if node_attributes.size > 0:
        node_attributes = np.hstack([node_attributes, np.array(node_types_all).reshape(-1, 1)])

    print('\n--- 统计信息 ---')
    print(f' node_ids：{len(node_ids)}')
    print(f' graph_labels：{len(graph_labels)}')
    print(f' node_labels：{len(node_labels)}')
    print(f' node_attributes：{node_attributes.shape}')
    print(f' graph_indicators：{len(graph_indicators)}')
    print(f' edge_labels：{edge_labels.shape}')
    print(f' DS_A：{DS_A.shape}')
    print('------------------')

    np.savetxt(os.path.join(save_pre, f'{release}_graph_labels.txt'), graph_labels, fmt='%d')
    np.savetxt(os.path.join(save_pre, f'{release}_node_labels.txt'), node_labels, fmt='%d')
    np.savetxt(os.path.join(save_pre, f'{release}_node_attributes.txt'), node_attributes, fmt='%.8f', delimiter=',')
    np.savetxt(os.path.join(save_pre, f'{release}_graph_indicator.txt'), graph_indicators, fmt='%d')
    np.savetxt(os.path.join(save_pre, f'{release}_A.txt'), DS_A, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(save_pre, f'{release}_edge_labels.txt'), edge_labels, fmt='%d')

    with open(os.path.join(save_pre, f'{release}_node_ids.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(node_ids))
    print(f"数据已保存至: {save_pre}")

def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        for release in all_releases[project]:
            start_time = time.time()
            line_data = readData(used_file_data_path, release + '.csv')
            if line_data is not None and not line_data.empty:
                toTUDA_optimized(line_data=line_data, project=project, release=release, method='lineflow')
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Project: {project}, Release: {release}, Time: {elapsed_time:.2f} seconds")
            print('-' * 50, release, 'done', '-' * 50)
    
    if count > 0:
        average_time = total_time / count
        print(f"Average Time: {average_time:.2f} seconds")

if __name__ == '__main__':
    main()