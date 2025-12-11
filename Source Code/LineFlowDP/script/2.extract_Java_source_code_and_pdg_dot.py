import shutil

from tqdm import tqdm

from my_util import *
import os
import pandas as pd
import subprocess

used_file_path = '../used_file_data/'
source_code_path = '../sourcecode/'

save_dir = "datasets/preprocessed_data/"
sourcecode_dir = "../sourcecode/"
graph_tool_path = '../PropertyGraph-main/out/artifacts/PropertyGraph_jar/PropertyGraph.jar'
# java -jar PropertyGraph.jar -d test/src -p -c -a


os.makedirs(sourcecode_dir, exist_ok=True)


def preprocess_code_line(code_line):
    '''
        input
            code_line (string)
        Replace strings and characters with <str> and <char>.
    '''

    code_line = re.sub("\".*?\"", "<str>", code_line)
    code_line = re.sub("\'.*?\'", "<char>", code_line)

    code_line = code_line.strip()

    return code_line


def get_sourcecode(proj_name):
    """
        input: project
        process: sourcecode
    """
    cur_all_rel = all_releases[proj_name]

    for rel in cur_all_rel:
        df_rel = pd.read_csv(os.path.join(save_dir, f'{rel}.csv'), na_filter=False)

        df_rel['code_line'] = df_rel['code_line'].apply(preprocess_code_line)

        grouped = df_rel.groupby('filename')
        for filename, group in tqdm(grouped, desc=f"{rel} get java files"):
            code = '\n'.join(map(str, group['code_line']))
            output_path = os.path.join(f'{sourcecode_dir}/{proj_name}/{rel}', filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)

        print(f'{rel} done')


def get_PDG_dot(proj_name):
    """
        input: project
        get: PDG of each java file
    """

    cur_all_rel = all_releases[proj_name]

    for rel in cur_all_rel:
        df_rel = pd.read_csv(os.path.join(save_dir, f'{rel}.csv'), na_filter=False)
        grouped = df_rel.groupby('filename')
        for filename, group in tqdm(grouped, desc=f"{rel} get pdg files"):

            java_path = f'{sourcecode_dir}/{proj_name}/{rel}/{filename}'
            get_pdg_command = f'java -jar {graph_tool_path} -d {java_path} -p'
            pdg_result = subprocess.run(get_pdg_command, shell=True, capture_output=True, text=True)
            if pdg_result.returncode == 0:
                java_dir = os.path.dirname(java_path)
                pdg_dir = os.path.join(java_dir, 'PDG')
                move_files(pdg_dir, java_dir)
                os.rmdir(pdg_dir)
            else:
                pass


def move_files(src_dir, dst_dir):
    files = os.listdir(src_dir)

    src_file = os.path.join(src_dir, files[0])
    dst_file = os.path.join(dst_dir, files[0])

    shutil.move(src_file, dst_file)


if __name__ == '__main__':
    for proj in list(all_releases.keys()):
        get_sourcecode(proj)
        get_PDG_dot(proj)
