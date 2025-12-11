"""
自动化测试所有项目的脚本
"""
import os
import subprocess
import shutil

# 定义所有项目及其版本
ALL_PROJECTS = {
    "activemq": ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
    "camel": ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
    "derby": ["derby-10.2.1.6", "derby-10.3.1.4", "derby-10.5.1.1"],
    "groovy": ["groovy-1_5_7", "groovy-1_6_BETA_1", "groovy-1_6_BETA_2"],
    "hbase": ["hbase-0.94.0", "hbase-0.95.0", "hbase-0.95.2"],
    "hive": ["hive-0.9.0", "hive-0.10.0", "hive-0.12.0"],
    "jruby": ["jruby-1.1", "jruby-1.4.0", "jruby-1.5.0", "jruby-1.7.0.preview1"],
    "lucene": ["lucene-2.3.0", "lucene-2.9.0", "lucene-3.0.0", "lucene-3.1"],
    "wicket": ["wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"]
}

# 预处理脚本模板
PREPROCESS_TEMPLATE = """import os
import re
import numpy as np
import pandas as pd

current_script_dir = os.path.dirname(os.path.abspath(__file__))
data_root_dir = os.path.join(current_script_dir, "../dataset/linedp/")
save_dir = os.path.join(current_script_dir, "../dataset/linedp/preprocessed_data/")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_lvl_dir = os.path.join(data_root_dir, "File-level")
line_lvl_dir = os.path.join(data_root_dir, "Line-level")

char_to_remove = ["+", "-", "*", "/", "=", "++", "--", "\\\\", "<str>", "<char>", "|", "&", "!"]

all_releases = {PROJECT_DICT}

def preprocess_data(proj_name):
    cur_all_rel = all_releases[proj_name]
    for rel in cur_all_rel:
        file_level_data = pd.read_csv(
            os.path.join(file_lvl_dir, rel + "_ground-truth-files_dataset.csv"), encoding="latin"
        )
        line_level_data = pd.read_csv(
            os.path.join(line_lvl_dir, rel + "_defective_lines_dataset.csv"), encoding="latin"
        )
        file_level_data = file_level_data.fillna("")
        buggy_files = list(line_level_data["File"].unique())
        preprocessed_df_list = []
        
        for idx, row in file_level_data.iterrows():
            filename = row["File"]
            if ".java" not in filename:
                continue
            code = row["SRC"]
            label = row["Bug"]
            code_df = create_code_df(code, filename)
            code_df["file-label"] = [label] * len(code_df)
            code_df["line-label"] = [False] * len(code_df)
            if filename in buggy_files:
                buggy_lines = list(line_level_data[line_level_data["File"] == filename]["Line_number"])
                code_df["line-label"] = code_df["line_number"].isin(buggy_lines)
            if len(code_df) > 0:
                preprocessed_df_list.append(code_df)
        
        all_df = pd.concat(preprocessed_df_list)
        all_df.to_csv(os.path.join(save_dir, rel + ".csv"), index=False)
        print("finish release {}".format(rel))

def make_splits():
    train_releases = [releases[0] for releases in all_releases.values()]
    val_releases = [releases[1] for releases in all_releases.values()]
    test_releases = [release for releases in all_releases.values() for release in releases[2:]]
    
    releases_of_split = {
        "train": train_releases,
        "val": val_releases,
        "test": test_releases
    }
    
    for split, releases in releases_of_split.items():
        all_df = pd.concat([pd.read_csv(os.path.join(save_dir, release + ".csv")) for release in releases])
        all_df.to_parquet(os.path.join(save_dir, split + ".parquet.gzip"), index=False)
        print("finish split {} with {} rows from {}".format(split, len(all_df), releases))

def create_code_df(code_str, filename):
    df = pd.DataFrame()
    code_lines = code_str.splitlines()
    preprocess_code_lines = []
    is_comments = []
    is_blank_line = []
    comments = re.findall(r"(/\\*[\\s\\S]*?\\*/)", code_str, re.DOTALL)
    comments_str = "\\n".join(comments)
    comments_list = comments_str.split("\\n")
    
    for code_line in code_lines:
        code_line = code_line.strip()
        is_comment = is_comment_line(code_line, comments_list)
        is_comments.append(is_comment)
        if not is_comment:
            code_line = preprocess_code_line(code_line)
        is_blank_line.append(is_empty_line(code_line))
        preprocess_code_lines.append(code_line)
    
    is_test = "test" in filename
    df["filename"] = [filename] * len(code_lines)
    df["is_test_file"] = [is_test] * len(code_lines)
    df["code_line"] = preprocess_code_lines
    df["line_number"] = np.arange(1, len(code_lines) + 1)
    df["is_comment"] = is_comments
    df["is_blank"] = is_blank_line
    return df

def is_comment_line(code_line, comments_list):
    code_line = code_line.strip()
    if len(code_line) == 0:
        return False
    elif code_line.startswith("//"):
        return True
    elif code_line in comments_list:
        return True
    return False

def preprocess_code_line(code_line):
    code_line = re.sub("''", "'", code_line)
    code_line = re.sub('".*?"', "<str>", code_line)
    code_line = re.sub("'.*?'", "<char>", code_line)
    code_line = re.sub(r"\\b\\d+\\b", "", code_line)
    code_line = re.sub("\\\\[.*?]", "", code_line)
    code_line = re.sub("[.,:;{}()]", " ", code_line)
    for char in char_to_remove:
        code_line = code_line.replace(char, " ")
    return code_line.strip()

def is_empty_line(code_line):
    return len(code_line.strip()) == 0

if __name__ == '__main__':
    for proj in list(all_releases.keys()):
        preprocess_data(proj)
    make_splits()
"""


def generate_preprocess_script(project_name, versions):
    """生成特定项目的预处理脚本"""
    project_dict = f'{{"{project_name}": {versions}}}'
    script_content = PREPROCESS_TEMPLATE.replace("{PROJECT_DICT}", project_dict)
    
    with open("script/preprocess_data_temp.py", "w") as f:
        f.write(script_content)


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def process_project(project_name, versions):
    """处理单个项目"""
    print(f"\n{'#'*60}")
    print(f"# Processing: {project_name}")
    print(f"{'#'*60}")
    
    # 1. 生成并运行预处理脚本
    generate_preprocess_script(project_name, versions)
    if not run_command(
        "python script/preprocess_data_temp.py",
        f"Step 1: Preprocessing {project_name} data"
    ):
        print(f"ERROR: Failed to preprocess {project_name}")
        return False
    
    # 2. 清除缓存
    cache_dir = f"cache/roberta-linedp-{project_name}-version-512-16"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")
    
    # 3. 检查模型是否存在
    model_path = f"checkpoints/roberta-linedp-{project_name}-version"
    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}, skipping test")
        return False
    
    # 4. 运行测试
    test_cmd = f"""python -m src.bug_prediction.BugPredictionTester \\
--model_type=roberta \\
--config_name=huggingface/CodeBERTa-small-v1 \\
--tokenizer_name=huggingface/CodeBERTa-small-v1 \\
--model_name={model_path} \\
--encoder_type=line \\
--dataset_path=dataset/linedp/preprocessed_data \\
--batch_size=16 \\
--cache_dir=cache/roberta-linedp-{project_name}-version \\
--output_path=outputs/roberta-linedp-{project_name}-version"""
    
    if not run_command(test_cmd, f"Step 2: Testing {project_name}"):
        print(f"ERROR: Failed to test {project_name}")
        return False
    
    print(f"\n✓ Successfully completed {project_name}")
    return True


def main():
    print("="*60)
    print("Automated Cross-Version Testing for All Projects")
    print("="*60)
    
    # 删除旧的CSV结果
    csv_path = "results/cross_project_20250921.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Removed old results: {csv_path}\n")
    
    # 处理所有项目
    results = {}
    for project_name, versions in ALL_PROJECTS.items():
        success = process_project(project_name, versions)
        results[project_name] = success
    
    # 清理临时文件
    temp_script = "script/preprocess_data_temp.py"
    if os.path.exists(temp_script):
        os.remove(temp_script)
    
    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for project, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{project:15s} : {status}")
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Detailed outputs in: outputs/")


if __name__ == "__main__":
    main()