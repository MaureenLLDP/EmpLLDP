# TUDataset_raw.py 顶部添加以下代码
import sys
import os

# 获取当前脚本的绝对路径（如 D:\科研\LineFlowDP-main\script\TUDataset_raw.py）
current_script_path = os.path.abspath(__file__)
# 推导项目根目录路径（LineFlowDP-main/）
project_root = os.path.dirname(os.path.dirname(current_script_path))  # 向上两级目录

# 将项目根目录添加到 Python 模块搜索路径
sys.path.append(project_root)

import time
import numpy as np
import os
import warnings
from itertools import chain
from script.my_util import *

warnings.filterwarnings("ignore")

save_path = './data/'
source_code_path = '../sourcecode/'
used_file_data_path = '../used_file_data/'


def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin', keep_default_na=False)
    return data


def toTUDA(line_data: pd.DataFrame, project, release, method='lineflow'):
    save_pre = save_path + project + '/' + release + '/' + 'raw/'
    if not os.path.exists(save_pre):
        os.makedirs(save_pre)

    file_data = line_data.drop_duplicates('filename', keep='first')
    files_list = list(file_data['filename'])

    doc2vec_dict = {}
    pdg_dict = {}
    edge_label_dict = {}

    for file_name in files_list:
        folder_base = source_code_path + project.lower() + '/' + release + '/' + file_name.replace('.java', '')
        if method == 'lineflow':
            word2vec_path = folder_base + '_lineflow.doc2vec'
        elif method == 'linenoflow':
            word2vec_path = folder_base + '_linenoflow.doc2vec'
        elif method == 'noflow':
            word2vec_path = folder_base + '_noflow.doc2vec'
        doc2vec_dict[file_name] = np.loadtxt(word2vec_path, delimiter=',')
        # pdg/edge_label
        pdg_path = source_code_path + project + '/' + release + '/' + file_name.replace('.java', '')
        pdg_dict[file_name] = np.loadtxt(pdg_path + '_pdg.txt')
        edge_label_dict[file_name] = np.loadtxt(pdg_path + '_edge_label.txt')

    graph_labels = []
    graph_indicators = []
    graph_indicators_i = 1
    node_attributes = []
    edge_labels = []

    node_labels = []  # line-level labels
    node_types_all = []  # node types

    node_ids = []

    DS_A = []
    max = 0
    flag_frist_dfg_is_null = False

    for file_data_index in file_data.index:
        file_name = file_data.loc[file_data_index, 'filename']
        current_file_data = line_data.loc[line_data['filename'] == file_name]

        node_ids.extend([f"{release}::{file_name}::{int(line)}"
                         for line in current_file_data['line_number']])

        # graph_labels
        if not file_data.loc[file_data_index, 'file-label']:
            graph_labels.append(0)
        else:
            graph_labels.append(1)

        # node_attributes
        vector = doc2vec_dict[file_name]
        if file_data_index == 0:
            node_attributes = vector
        else:
            node_attributes = np.vstack((node_attributes, vector))

        temp = [int(graph_indicators_i) for i in range(len(current_file_data))]
        graph_indicators.append(temp)
        graph_indicators_i += 1

        line_labels = list(current_file_data['line-label'].astype(int))
        node_labels.append(line_labels)

        node_types = list(current_file_data['node_label'].astype(int))
        node_types_all.extend(node_types)

        # DS_A
        dfg_line = pdg_dict[file_name] + max
        max = max + len(current_file_data)
        if file_data_index == 0:
            if len(dfg_line) == 0:
                flag_frist_dfg_is_null = True
                DS_A = [0, 0]
            else:
                DS_A = dfg_line
        else:
            try:
                DS_A = np.vstack((DS_A, dfg_line))
            except:
                DS_A = DS_A

        edge_label = edge_label_dict[file_name]
        if file_data_index == 0:
            edge_labels = edge_label
        else:
            edge_labels = np.hstack((edge_labels, edge_label))

    graph_indicators = list(chain.from_iterable(graph_indicators))

    if flag_frist_dfg_is_null:
        DS_A = DS_A[1:]

    node_labels_flat = list(chain.from_iterable(node_labels))
    node_attributes = np.hstack([node_attributes, np.array(node_types_all).reshape(-1, 1)])

    print(' node_ids：', len(node_ids))
    print(' graph_labels：', len(graph_labels))
    print(' node_labels：', len(node_labels_flat))
    print(' node_attributes：', len(node_attributes))
    print(' graph_indicators：', len(graph_indicators))
    print(' edge_labels：', len(edge_labels))
    print(' DS_A：', len(DS_A))

    np.savetxt(save_pre + release + '_graph_labels.txt', graph_labels, fmt='%d')
    np.savetxt(save_pre + release + '_node_labels.txt', node_labels_flat, fmt='%d')
    np.savetxt(save_pre + release + '_node_attributes.txt', node_attributes, fmt='%.8f', delimiter=',')
    np.savetxt(save_pre + release + '_graph_indicator.txt', graph_indicators, fmt='%d')
    np.savetxt(save_pre + release + '_A.txt', DS_A, fmt='%d', delimiter=',')
    np.savetxt(save_pre + release + '_edge_labels.txt', edge_labels, fmt='%d', delimiter=',')

    with open(save_pre + release + '_node_ids.txt', 'w', encoding='utf-8') as f:
        for node_id in node_ids:
            f.write(node_id + '\n')


def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        for release in all_releases[project]:
            start_time = time.time()
            line_data = readData(used_file_data_path, release + '.csv')
            toTUDA(line_data=line_data, project=project, release=release, method='lineflow')
            # toTUDA(line_data=line_data, project=project, release=release, method='linenoflow')
            # toTUDA(line_data=line_data, project=project, release=release, method='noflow')
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Project: {project}, Release: {release}, Time: {elapsed_time} seconds")
            print('-' * 50, release, 'done', '-' * 50)
    average_time = total_time / count
    print(f"Average Time: {average_time} seconds")


if __name__ == '__main__':
    main()
