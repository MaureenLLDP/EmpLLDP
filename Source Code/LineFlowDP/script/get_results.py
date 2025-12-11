import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
import os
from my_util import all_test_releases, all_projs 

def calculate_line_level_metrics(df):
    """
    计算行级指标（只对预测为缺陷的文件）
    """
    df = df.copy()
    df['risk_score'] = df['katz'] + df['degree'] + df['closeness']
    
    # 只保留预测为缺陷的文件中的行
    df_defective_files = df[df['file_pred_label'] == 1].copy()
    
    if len(df_defective_files) == 0:
        return None
    
    # 按risk_score降序排序
    df_sorted = df_defective_files.sort_values('risk_score', ascending=False).reset_index(drop=True)
    
    metrics = {}
    
    # 1. Recall@Top20%LOC
    total_loc = len(df_sorted)
    top_20_loc = int(total_loc * 0.2)
    total_defective_lines = df_sorted['line_label'].sum()
    
    if total_defective_lines > 0:
        top_20_defective = df_sorted.head(top_20_loc)['line_label'].sum()
        metrics['Recall@20%LOC'] = top_20_defective / total_defective_lines
    else:
        metrics['Recall@20%LOC'] = 0.0
    
    # 2. Effort@Top20%Recall
    if total_defective_lines > 0:
        target_defects = total_defective_lines * 0.2
        cumsum = df_sorted['line_label'].cumsum()
        indices = np.where(cumsum >= target_defects)[0]
        if len(indices) > 0:
            effort_loc = indices[0] + 1
            metrics['Effort@20%Recall'] = effort_loc / total_loc
        else:
            metrics['Effort@20%Recall'] = 1.0
    else:
        metrics['Effort@20%Recall'] = 0.0
    
    # 3. IFA (Initial False Alarms)
    ifa_list = []
    for filename in df_defective_files['filename'].unique():
        file_lines = df_defective_files[df_defective_files['filename'] == filename].copy()
        file_lines = file_lines.sort_values('risk_score', ascending=False).reset_index(drop=True)
        
        defect_positions = file_lines[file_lines['line_label'] == 1].index.tolist()
        if len(defect_positions) > 0:
            first_defect_pos = defect_positions[0]
            ifa_list.append(first_defect_pos)
    
    metrics['IFA'] = np.mean(ifa_list) if len(ifa_list) > 0 else 0.0
    
    # 4. Line-level AUC
    if len(df_sorted['line_label'].unique()) > 1:
        try:
            metrics['Line_AUC'] = roc_auc_score(df_sorted['line_label'], df_sorted['risk_score'])
        except:
            metrics['Line_AUC'] = np.nan
    else:
        metrics['Line_AUC'] = np.nan
    
    return metrics

def calculate_file_level_metrics(df):
    """
    计算文件级指标
    """
    # 获取每个文件的标签和预测（每个文件只取一次）
    file_df = df.groupby('filename').first().reset_index()
    
    y_true = file_df['file_label'].values
    y_pred = file_df['file_pred_label'].values
    y_prob = file_df['file_pred_prob'].values
    
    metrics = {}
    
    # Accuracy
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    
    # Balanced Accuracy
    metrics['Balanced_Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # MCC
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    # AUC
    if len(np.unique(y_true)) > 1:
        try:
            metrics['File_AUC'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['File_AUC'] = np.nan
    else:
        metrics['File_AUC'] = np.nan
    
    return metrics

def main():
    results_dir = './results/'
    output_dir = './metrics/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储结果
    line_version_results = []
    file_version_results = []
    
    # 遍历所有项目和测试版本
    for project in all_projs:
        test_releases = all_test_releases.get(project, [])
        
        for release in test_releases:
            csv_path = os.path.join(results_dir, f'{release}.csv')
            
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping...")
                continue
            
            # 读取数据
            df = pd.read_csv(csv_path)
            
            # 计算行级指标
            line_metrics = calculate_line_level_metrics(df)
            if line_metrics is not None:
                line_result = {
                    'project': project,
                    'release': release,
                    **line_metrics
                }
                line_version_results.append(line_result)
                print(f"[Line-level] {release}: "
                      f"Recall@20%={line_metrics['Recall@20%LOC']:.4f}, "
                      f"Effort@20%={line_metrics['Effort@20%Recall']:.4f}, "
                      f"IFA={line_metrics['IFA']:.2f}, "
                      f"AUC={line_metrics['Line_AUC']:.4f}")
            
            # 计算文件级指标
            file_metrics = calculate_file_level_metrics(df)
            file_result = {
                'project': project,
                'release': release,
                **file_metrics
            }
            file_version_results.append(file_result)
            print(f"[File-level] {release}: "
                  f"Acc={file_metrics['Accuracy']:.4f}, "
                  f"BA={file_metrics['Balanced_Accuracy']:.4f}, "
                  f"MCC={file_metrics['MCC']:.4f}, "
                  f"AUC={file_metrics['File_AUC']:.4f}\n")
    
    # ============ 保存行级指标 ============
    # 1. 版本级行级指标
    line_version_df = pd.DataFrame(line_version_results)
    line_version_output = os.path.join(output_dir, 'line_level_metrics_by_version.csv')
    line_version_df.to_csv(line_version_output, index=False)
    print(f"✓ Line-level version metrics saved to: {line_version_output}")
    
    # 2. 项目级行级指标
    line_project_results = []
    for project in all_projs:
        project_data = line_version_df[line_version_df['project'] == project]
        if len(project_data) > 0:
            line_project_metrics = {
                'project': project,
                'Recall@20%LOC': project_data['Recall@20%LOC'].mean(),
                'Effort@20%Recall': project_data['Effort@20%Recall'].mean(),
                'IFA': project_data['IFA'].mean(),
                'Line_AUC': project_data['Line_AUC'].mean()
            }
            line_project_results.append(line_project_metrics)
    
    line_project_df = pd.DataFrame(line_project_results)
    line_project_output = os.path.join(output_dir, 'line_level_metrics_by_project.csv')
    line_project_df.to_csv(line_project_output, index=False)
    print(f"✓ Line-level project metrics saved to: {line_project_output}")
    
    # ============ 保存文件级指标 ============
    # 3. 版本级文件级指标
    file_version_df = pd.DataFrame(file_version_results)
    file_version_output = os.path.join(output_dir, 'file_level_metrics_by_version.csv')
    file_version_df.to_csv(file_version_output, index=False)
    print(f"✓ File-level version metrics saved to: {file_version_output}")
    
    # 4. 项目级文件级指标
    file_project_results = []
    for project in all_projs:
        project_data = file_version_df[file_version_df['project'] == project]
        if len(project_data) > 0:
            file_project_metrics = {
                'project': project,
                'Accuracy': project_data['Accuracy'].mean(),
                'Balanced_Accuracy': project_data['Balanced_Accuracy'].mean(),
                'MCC': project_data['MCC'].mean(),
                'File_AUC': project_data['File_AUC'].mean()
            }
            file_project_results.append(file_project_metrics)
    
    file_project_df = pd.DataFrame(file_project_results)
    file_project_output = os.path.join(output_dir, 'file_level_metrics_by_project.csv')
    file_project_df.to_csv(file_project_output, index=False)
    print(f"✓ File-level project metrics saved to: {file_project_output}")
    
    # ============ 打印汇总 ============
    print("\n" + "="*100)
    print("LINE-LEVEL METRICS BY PROJECT")
    print("="*100)
    print(line_project_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    print("\n" + "="*100)
    print("FILE-LEVEL METRICS BY PROJECT")
    print("="*100)
    print(file_project_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("="*100)

if __name__ == '__main__':
    main()