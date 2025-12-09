import time
import numpy as np
import os
import pandas as pd
import pygraphviz as pgv
from tqdm import tqdm
from my_util import *
import re

# preprocessed_file_path = '../datasets/preprocessed_data/'
# save_used_file_path = '../datasets/used_file_data/'
# source_code_path = '../sourcecode/'
preprocessed_file_path = 'datasets/preprocessed_data/'
saved_file_path = '../used_file_data/'
source_code_path = '../sourcecode/'


def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin', keep_default_na=False)
    return data


def get_line_num(label):
    """
     <123> or <123... 456> format extraction line numbers
    """
    result = []
    matches = re.findall(r'<(\d+(?:\.\.\.\d+)?)>', label)
    for m in matches:
        if '...' in m:
            start, end = map(int, m.split('...'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(m))
    return result


def get_all_pdg(project, release):
    data_all = readData(path=preprocessed_file_path, file=release + '.csv')
    data = data_all.drop_duplicates('filename', keep='first')

    grouped = data_all.groupby('filename')
    invalid_filenames = []
    for filename, group in grouped:
        java_path = f'{source_code_path}/{project}/{release}/{filename}'
        pdg_dot_path = java_path.replace('.java', '_pdg.dot')
        if not os.path.exists(pdg_dot_path):
            invalid_filenames.append(filename)

    for index in tqdm(data.index, desc=f"{release} files"):
        file_name = str(data.loc[index, 'filename'])
        # print(f'Processing file:{file_name}')
        java_path = str(source_code_path + project + '/' + release + '/' + file_name)
        save_file_name_prefix = java_path.replace('.java', '')

        os.makedirs(save_used_file_path, exist_ok=True)

        if ('.java' not in file_name) or (file_name in invalid_filenames):
            continue

        node_lines = {}
        edges = []

        dotfile = str(file_name).replace('.java', '_pdg.dot')

        dotfile_path = f'{source_code_path}/{project}/{release}/{dotfile}'
        g = None
        try:
            g = pgv.AGraph(dotfile_path, encoding='utf-8')
        except:
            g = pgv.AGraph(dotfile_path, encoding='ansi')

        for n in g.nodes():
            attrs = n.attr
            try:
                label = attrs.get('label', '')
            except:
                label = '<str>'
            shape = attrs.get('shape', '')
            fillcolor = attrs.get('fillcolor', '')
            if fillcolor == 'aquamarine':
                continue
            if label and '<' in label and '>' in label:
                lines = get_line_num(label)
                node_lines[n.get_name()] = lines
                if shape == 'box':
                    node_label = 1
                elif shape == 'ellipse':
                    node_label = 2
                elif shape == 'diamond':
                    node_label = 3
                else:
                    node_label = 4
                for line in lines:
                    data_all.loc[
                        (data_all['filename'] == file_name) &
                        (data_all['line_number'] == line), 'node_label'
                    ] = node_label

        for e in g.edges():
            style = e.attr.get('style', '')
            if style == 'dotted':
                edge_label_flag = 1
            elif style == 'solid':
                edge_label_flag = 2
            elif style == 'bold':
                edge_label_flag = 3
            else:
                edge_label_flag = 4
            edge_source = e[0]
            edge_target = e[1]
            edges.append([edge_source, edge_target, edge_label_flag])

        source = []
        target = []
        edge_label = []
        for edge in edges:
            if (edge[0] in node_lines.keys()) & (edge[1] in node_lines.keys()):
                for i in node_lines.get(edge[0]):
                    for j in node_lines.get(edge[1]):
                        source.append(i)
                        target.append(j)
                        edge_label.append(edge[2])
        if len(source) > 0:
            pdg = np.vstack((source, target)).T
            np.savetxt(save_file_name_prefix + '_pdg.txt', pdg, fmt='%d')
            np.savetxt(save_file_name_prefix + '_edge_label.txt', edge_label, fmt='%d')
        else:
            open(save_file_name_prefix + '_pdg.txt', 'w').close()
            open(save_file_name_prefix + '_edge_label.txt', 'w').close()

    data_all['node_label'] = data_all['node_label'].fillna(4)

    # 删除并保存
    if invalid_filenames:
        print(f"Deleted {len(invalid_filenames)} files：")
        for fname in invalid_filenames:
            print(f"  Have Deleted：{fname}")
        data_all = data_all[~data_all['filename'].isin(invalid_filenames)].copy()
        data_all.to_csv(os.path.join(save_used_file_path, f'{release}.csv'), index=False)
    else:
        data_all.to_csv(os.path.join(save_used_file_path, f'{release}.csv'), index=False)


def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        for release in all_releases[project]:
            start_time = time.time()
            get_all_pdg(project=project, release=release)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Project: {project}, Release: {release}, Time: {elapsed_time} seconds")
    average_time = total_time / count
    print(f"Average Time: {average_time} seconds")


if __name__ == '__main__':
    main()
