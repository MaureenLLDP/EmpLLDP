import pandas as pd
import os, re
import numpy as np
from my_util import *

data_root_dir = 'datasets/original/'
save_dir = "datasets/preprocessed_data/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_lvl_dir = data_root_dir + 'File-level/'
line_lvl_dir = data_root_dir + 'Line-level/'


def is_comment_line(code_line, comments_list):
    code_line = code_line.strip()

    if len(code_line) == 0:
        return False
    elif code_line.startswith('//'):
        return True
    elif code_line in comments_list:
        return True

    return False


def is_empty_line(code_line):
    '''
        input
            code_line (string)
        output
            boolean value
    '''

    if len(code_line.strip()) == 0:
        return True

    return False


def create_code_df(code_str, filename):

    df = pd.DataFrame()

    code_lines = code_str.splitlines()

    preprocess_code_lines = []
    is_comments = []
    is_blank_line = []

    comments = re.findall(r'(/\*[\s\S]*?\*/)', code_str, re.DOTALL)
    comments_str = '\n'.join(comments)
    comments_list = comments_str.split('\n')

    for l in code_lines:
        l = l.strip()
        is_comment = is_comment_line(l, comments_list)
        is_comments.append(is_comment)

        is_blank_line.append(is_empty_line(l))
        preprocess_code_lines.append(l)

    if 'test' in filename:
        is_test = True
    else:
        is_test = False

    df['filename'] = [filename] * len(code_lines)
    df['is_test_file'] = [is_test] * len(code_lines)
    df['code_line'] = preprocess_code_lines
    df['line_number'] = np.arange(1, len(code_lines) + 1)
    df['is_comment'] = is_comments
    df['is_blank'] = is_blank_line

    return df


def reindex_edge_index(edge_index, graph_indicator, graph_id):
    """
    重新索引 edge_index，使其变为局部索引
    """
    node_mask = (graph_indicator == graph_id)  # 该图的所有节点
    node_indices = torch.nonzero(node_mask).squeeze()  # 找到这些节点的索引
    num_nodes = node_indices.shape[0]

    if num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long)  # 返回空 edge_index

    # 创建全局索引到局部索引的映射
    reindex_map = {int(node_indices[i]): i for i in range(len(node_indices))}

    # 过滤只属于该图的边
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]  # 仅保留属于当前图的边

    if edge_index.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.long)  # 返回空 edge_index

    # 重新映射 edge_index
    edge_index = torch.tensor([[reindex_map[int(src)], reindex_map[int(dst)]]
                                for src, dst in edge_index.t().tolist()], dtype=torch.long).t()

    return edge_index

def preprocess_data(proj_name):
    cur_all_rel = all_releases[proj_name]

    for rel in cur_all_rel:
        print(file_lvl_dir +rel)
        file_level_data = pd.read_csv(file_lvl_dir + rel + '_ground-truth-files_dataset.csv', encoding='latin')
        line_level_data = pd.read_csv(line_lvl_dir + rel + '_defective_lines_dataset.csv', encoding='latin')

        file_level_data = file_level_data.fillna('')

        buggy_files = list(line_level_data['File'].unique())

        preprocessed_df_list = []

        for idx, row in file_level_data.iterrows():

            filename = row['File']

            if '.java' not in filename:
                continue

            code = row['SRC']
            label = row['Bug']

            code_df = create_code_df(code, filename)
            code_df['file-label'] = [label] * len(code_df)
            code_df['line-label'] = [False] * len(code_df)

            if filename in buggy_files:
                buggy_lines = list(line_level_data[line_level_data['File'] == filename]['Line_number'])
                code_df['line-label'] = code_df['line_number'].isin(buggy_lines)

            if len(code_df) > 0:
                preprocessed_df_list.append(code_df)

        all_df = pd.concat(preprocessed_df_list)
        all_df.to_csv(save_dir + rel + ".csv", index=False)
        print('finish release {}'.format(rel))


for proj in list(all_releases.keys()):
    preprocess_data(proj)
