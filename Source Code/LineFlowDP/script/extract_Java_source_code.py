import shutil
import json
from tqdm import tqdm
from my_util import *
import os
import pandas as pd
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import tempfile
import time

used_file_path = '../used_file_data/'
source_code_path = '../sourcecode/'
save_dir = "datasets/preprocessed_data/"
sourcecode_dir = "../sourcecode/"
graph_tool_path = '../PropertyGraph-main/out/artifacts/PropertyGraph_jar/PropertyGraph.jar'

os.makedirs(sourcecode_dir, exist_ok=True)

def preprocess_code_line(code_line):
    code_line = re.sub("\".*?\"", "<str>", code_line)
    code_line = re.sub("\'.*?\'", "<char>", code_line)
    code_line = code_line.strip()
    return code_line

def get_sourcecode(proj_name):
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


def process_single_file_fixed(filename, proj_name, rel, graph_tool_path, sourcecode_dir):
    """
    修复并发问题：使用临时目录避免进程间冲突
    """
    # 标准化路径，避免双斜杠
    java_path = os.path.normpath(f'{sourcecode_dir}/{proj_name}/{rel}/{filename}')
    
    if not os.path.exists(java_path):
        return {'filename': filename, 'success': False, 'error': 'Java file not found'}
    
    # 创建唯一的临时目录，避免并发冲突
    with tempfile.TemporaryDirectory() as temp_dir:
        # 复制 Java 文件到临时目录
        temp_java_path = os.path.join(temp_dir, os.path.basename(filename))
        shutil.copy2(java_path, temp_java_path)
        
        # 在临时目录中生成 PDG
        get_pdg_command = f'java -jar {graph_tool_path} -d "{temp_java_path}" -p 2>&1'
        
        try:
            pdg_result = subprocess.run(
                get_pdg_command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=60
            )
            
            if pdg_result.returncode == 0:
                # 查找生成的 PDG 文件（在 PDG 子目录中）
                temp_pdg_dir = os.path.join(temp_dir, 'PDG')
                
                if os.path.exists(temp_pdg_dir):
                    pdg_files = os.listdir(temp_pdg_dir)
                    if pdg_files:
                        # 移动 PDG 文件到原始 Java 文件所在目录
                        java_dir = os.path.dirname(java_path)
                        
                        for pdg_file in pdg_files:
                            src_pdg = os.path.join(temp_pdg_dir, pdg_file)
                            dst_pdg = os.path.join(java_dir, pdg_file)
                            
                            # 如果目标文件已存在，先删除（可能是之前失败的残留）
                            if os.path.exists(dst_pdg):
                                os.remove(dst_pdg)
                            
                            shutil.move(src_pdg, dst_pdg)
                        
                        return {'filename': filename, 'success': True, 'error': None}
                    else:
                        return {'filename': filename, 'success': False, 'error': 'PDG directory empty'}
                else:
                    # PropertyGraph 可能直接生成在临时目录根目录
                    all_files = os.listdir(temp_dir)
                    pdg_files = [f for f in all_files if f.endswith('_pdg.dot')]
                    
                    if pdg_files:
                        java_dir = os.path.dirname(java_path)
                        for pdg_file in pdg_files:
                            src_pdg = os.path.join(temp_dir, pdg_file)
                            dst_pdg = os.path.join(java_dir, pdg_file)
                            
                            if os.path.exists(dst_pdg):
                                os.remove(dst_pdg)
                            
                            shutil.move(src_pdg, dst_pdg)
                        
                        return {'filename': filename, 'success': True, 'error': None}
                    else:
                        return {'filename': filename, 'success': False, 'error': 'No PDG files generated'}
            else:
                error_msg = pdg_result.stderr[:200] if pdg_result.stderr else 'Unknown error'
                return {'filename': filename, 'success': False, 'error': error_msg}
        
        except subprocess.TimeoutExpired:
            return {'filename': filename, 'success': False, 'error': 'Timeout (>60s)'}
        except Exception as e:
            return {'filename': filename, 'success': False, 'error': str(e)}


def get_PDG_dot_parallel(proj_name, max_workers=48):
    """并行生成 PDG 文件（修复版）"""
    cur_all_rel = all_releases[proj_name]
    all_failed_files = {}
    
    for rel in cur_all_rel:
        failed_files = []
        df_rel = pd.read_csv(os.path.join(save_dir, f'{rel}.csv'), na_filter=False)
        grouped = df_rel.groupby('filename')
        
        filenames = list(grouped.groups.keys())
        total_files = len(filenames)
        
        print(f"\n{rel}: Processing {total_files} files with {max_workers} workers")
        
        process_func = partial(
            process_single_file_fixed,
            proj_name=proj_name,
            rel=rel,
            graph_tool_path=graph_tool_path,
            sourcecode_dir=sourcecode_dir
        )
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_func, filename): filename 
                for filename in filenames
            }
            
            with tqdm(total=total_files, desc=f"{rel} get pdg files") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()
                    if not result['success']:
                        failed_files.append(result['filename'])
                        if len(failed_files) <= 5:
                            print(f"\nWarning: {result['filename']} - {result['error']}")
                    pbar.update(1)
        
        if failed_files:
            all_failed_files[rel] = failed_files
            print(f"\n{rel}: {len(failed_files)}/{total_files} files failed to generate PDG ({len(failed_files)/total_files*100:.1f}%)")
            
            failed_log_path = f'{sourcecode_dir}/{proj_name}/{rel}_failed_pdg.json'
            with open(failed_log_path, 'w') as f:
                json.dump(failed_files, f, indent=2)
            print(f"Failed files logged to: {failed_log_path}")
        else:
            print(f"\n{rel}: All {total_files} files processed successfully! ✓")
    
    return all_failed_files


if __name__ == '__main__':
    import multiprocessing
    
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected {cpu_count} CPU cores")
    
    # 智能配置
    if cpu_count >= 100:
        max_workers = 48
        print("⚡ Super server detected! Using 48 workers (optimized for JVM + I/O)")
    elif cpu_count >= 50:
        max_workers = 32
    elif cpu_count >= 20:
        max_workers = 20
    elif cpu_count >= 10:
        max_workers = 12
    else:
        max_workers = max(cpu_count - 2, 4)
    
    print(f"🚀 Using {max_workers} parallel workers\n")
    print(f"✨ Fixed: Using temporary directories to avoid concurrent conflicts\n")
    
    all_failed = {}
    for proj in list(all_releases.keys()):
        print(f"\n{'='*50}")
        print(f"Processing project: {proj}")
        print(f"{'='*50}")
        
        get_sourcecode(proj)
        failed = get_PDG_dot_parallel(proj, max_workers=max_workers)
        
        if failed:
            all_failed[proj] = failed
    
    if all_failed:
        with open(f'{sourcecode_dir}/all_failed_pdg.json', 'w') as f:
            json.dump(all_failed, f, indent=2)
        print(f"\n{'='*50}")
        print("Summary of all failed PDG generations:")
        total_failed = 0
        total_files = 0
        for proj, releases in all_failed.items():
            for rel, files in releases.items():
                total_failed += len(files)
                df = pd.read_csv(os.path.join(save_dir, f'{rel}.csv'), na_filter=False)
                total_files += len(df['filename'].unique())
                print(f"{proj}/{rel}: {len(files)} files failed")
        
        print(f"\nOverall failure rate: {total_failed}/{total_files} ({total_failed/total_files*100:.1f}%)")
    
    print(f"\n{'='*50}")
    print("All done! ✓")
    print(f"{'='*50}")