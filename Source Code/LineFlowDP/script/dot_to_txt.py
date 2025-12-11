import time
import numpy as np
import os
import pandas as pd
import pygraphviz as pgv
from tqdm import tqdm
from my_util import *
import re
import subprocess

preprocessed_file_path = 'datasets/preprocessed_data/'
saved_file_path = '../used_file_data/'
source_code_path = '../sourcecode/'
graph_tool_path = '../PropertyGraph-main/out/artifacts/PropertyGraph_jar/PropertyGraph.jar'


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


def regenerate_missing_pdg(java_path):
    """
    尝试重新生成缺失的 PDG 文件
    """
    try:
        get_pdg_command = f'java -jar {graph_tool_path} -d {java_path} -p 2>&1'
        pdg_result = subprocess.run(get_pdg_command, shell=True, capture_output=True, text=True, timeout=60)
        
        if pdg_result.returncode == 0:
            java_dir = os.path.dirname(java_path)
            pdg_dir = os.path.join(java_dir, 'PDG')
            
            if os.path.exists(pdg_dir) and os.listdir(pdg_dir):
                import shutil
                files = os.listdir(pdg_dir)
                # 移动所有生成的文件
                for file in files:
                    src_file = os.path.join(pdg_dir, file)
                    dst_file = os.path.join(java_dir, file)
                    shutil.move(src_file, dst_file)
                
                # 删除空目录
                if os.path.exists(pdg_dir):
                    os.rmdir(pdg_dir)
                return True
        return False
    except Exception as e:
        print(f"Error regenerating PDG: {e}")
        return False


def get_all_pdg(project, release, auto_regenerate=True):
    data_all = readData(path=preprocessed_file_path, file=release + '.csv')
    data = data_all.drop_duplicates('filename', keep='first')

    grouped = data_all.groupby('filename')
    invalid_filenames = []
    missing_pdg_files = []
    
    # 预先检查所有文件
    print(f"Checking PDG files for {release}...")
    for filename, group in grouped:
        java_path = f'{source_code_path}/{project}/{release}/{filename}'
        pdg_dot_path = java_path.replace('.java', '_pdg.dot')
        
        if not os.path.exists(pdg_dot_path):
            missing_pdg_files.append((filename, java_path, pdg_dot_path))
    
    if missing_pdg_files:
        print(f"Found {len(missing_pdg_files)} missing PDG files")
        
        if auto_regenerate:
            print("Attempting to regenerate missing PDG files...")
            regenerated = 0
            for filename, java_path, pdg_dot_path in tqdm(missing_pdg_files, desc="Regenerating PDG"):
                if os.path.exists(java_path):
                    if regenerate_missing_pdg(java_path):
                        if os.path.exists(pdg_dot_path):
                            regenerated += 1
                        else:
                            invalid_filenames.append(filename)
                    else:
                        invalid_filenames.append(filename)
                else:
                    print(f"Java file not found: {java_path}")
                    invalid_filenames.append(filename)
            
            print(f"Successfully regenerated {regenerated}/{len(missing_pdg_files)} PDG files")
        else:
            # 如果不自动重新生成，直接标记为无效
            invalid_filenames.extend([item[0] for item in missing_pdg_files])
    
    # 开始处理文件
    for index in tqdm(data.index, desc=f"{release} processing files"):
        file_name = str(data.loc[index, 'filename'])
        java_path = str(source_code_path + project + '/' + release + '/' + file_name)
        save_file_name_prefix = java_path.replace('.java', '')

        os.makedirs(saved_file_path, exist_ok=True)

        if ('.java' not in file_name) or (file_name in invalid_filenames):
            continue

        node_lines = {}
        edges = []

        dotfile = str(file_name).replace('.java', '_pdg.dot')
        dotfile_path = f'{source_code_path}/{project}/{release}/{dotfile}'
        
        # 再次确认文件存在
        if not os.path.exists(dotfile_path):
            print(f"PDG file still missing: {dotfile_path}")
            if file_name not in invalid_filenames:
                invalid_filenames.append(file_name)
            continue
        
        g = None
        try:
            g = pgv.AGraph(dotfile_path, encoding='utf-8')
        except:
            try:
                g = pgv.AGraph(dotfile_path, encoding='ansi')
            except Exception as e:
                print(f"Error loading graph for {file_name}: {e}")
                if file_name not in invalid_filenames:
                    invalid_filenames.append(file_name)
                continue

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

    # 删除无效文件并保存
    if invalid_filenames:
        print(f"Removing {len(invalid_filenames)} invalid files from dataset")
        for fname in invalid_filenames[:10]:  # 只打印前10个
            print(f"  Removed: {fname}")
        if len(invalid_filenames) > 10:
            print(f"  ... and {len(invalid_filenames) - 10} more")
        
        data_all = data_all[~data_all['filename'].isin(invalid_filenames)].copy()
    
    data_all.to_csv(os.path.join(saved_file_path, f'{release}.csv'), index=False)
    print(f"Saved processed data to {saved_file_path}{release}.csv")


def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        for release in all_releases[project]:
            print(f"\n{'='*60}")
            print(f"Processing: {project}/{release}")
            print(f"{'='*60}")
            
            start_time = time.time()
            get_all_pdg(project=project, release=release, auto_regenerate=True)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Time: {elapsed_time:.2f} seconds\n")
    
    average_time = total_time / count
    print(f"\n{'='*60}")
    print(f"All processing completed!")
    print(f"Average Time: {average_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()