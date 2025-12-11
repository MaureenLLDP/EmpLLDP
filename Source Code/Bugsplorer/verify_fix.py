"""
验证修复后IFA指标的正确性
运行此脚本对比修复前后的结果
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path


def load_and_check_npz(filepath: str):
    """加载npz文件并检查数据质量"""
    data = np.load(filepath)
    true_prob = data['true_prob']
    labels = data['labels']
    
    print(f"\n{'='*60}")
    print(f"文件: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # 基本统计
    print(f"\n数据统计:")
    print(f"  总样本数: {len(labels)}")
    print(f"  缺陷行数: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.2f}%)")
    print(f"  正常行数: {len(labels) - np.sum(labels)}")
    
    # 预测分布
    print(f"\n预测概率分布:")
    print(f"  最小值: {true_prob.min():.6f}")
    print(f"  最大值: {true_prob.max():.6f}")
    print(f"  平均值: {true_prob.mean():.6f}")
    print(f"  中位数: {np.median(true_prob):.6f}")
    
    # 检查对应关系
    sorted_idx = np.argsort(true_prob)[::-1]
    top_k_positions = [10, 50, 100, 200, 500]
    
    print(f"\n前K个预测中的缺陷检出率:")
    for k in top_k_positions:
        if k <= len(labels):
            top_k_idx = sorted_idx[:k]
            defects_in_top_k = np.sum(labels[top_k_idx])
            print(f"  Top-{k:4d}: {defects_in_top_k:4d} / {k:4d} = {defects_in_top_k/k*100:5.2f}%")
    
    # 计算IFA
    ifa = calculate_ifa(labels, true_prob)
    print(f"\n初始误报数 (IFA): {ifa}")
    
    # 计算其他关键指标
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, true_prob)
    
    recall20 = recall_at_top20_percent_loc(labels, true_prob)
    effort20 = effort_at_top20_percent_recall(labels, true_prob)
    
    print(f"\n关键指标:")
    print(f"  AUC:            {auc:.6f}")
    print(f"  Recall@20%LOC:  {recall20:.6f}")
    print(f"  Effort@20%Rec:  {effort20:.6f}")
    print(f"  IFA:            {ifa}")
    
    # 异常检测
    print(f"\n异常检测:")
    if ifa > 500:
        print(f"  ⚠️  IFA过高 (>{ifa})，可能存在标签-预测错位问题")
    else:
        print(f"  ✓  IFA正常 ({ifa})")
    
    if auc > 0.9 and ifa > 500:
        print(f"  ⚠️  AUC很高但IFA很高，强烈怀疑存在数据对齐问题")
    
    return {
        'filename': os.path.basename(filepath),
        'total_lines': len(labels),
        'defect_lines': np.sum(labels),
        'auc': auc,
        'recall20': recall20,
        'effort20': effort20,
        'ifa': ifa
    }


def calculate_ifa(y_true: np.ndarray, y_score: np.ndarray):
    """计算初始误报数"""
    sorted_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_idx]
    
    # 找到第一个缺陷的位置
    defect_positions = np.where(y_true_sorted == 1)[0]
    if len(defect_positions) == 0:
        return len(y_true_sorted)  # 没有缺陷，全是误报
    
    return int(defect_positions[0])


def recall_at_top20_percent_loc(y_true: np.ndarray, y_score: np.ndarray):
    """Recall@20%LOC"""
    twenty_percent = int(len(y_true) * 0.20)
    y_score_sorted_idx = np.argsort(y_score)[::-1]
    top_twenty_percent_index = y_score_sorted_idx[:twenty_percent]
    top_twenty_percent = y_true[top_twenty_percent_index]
    result = np.sum(top_twenty_percent) / np.sum(y_true)
    return result


def effort_at_top20_percent_recall(y_true: np.ndarray, y_score: np.ndarray):
    """Effort@20%Recall"""
    twenty_percent_defect = int(sum(y_true) * 0.20)
    y_score_sorted_idx = np.argsort(y_score)[::-1]
    
    defect_count = 0
    for i, j in enumerate(y_score_sorted_idx, 1):
        if y_true[j]:
            defect_count += 1
        if defect_count == twenty_percent_defect:
            return i / len(y_true)
    
    return 1.0


def compare_directories(old_dir: str, new_dir: str):
    """对比修复前后两个目录的结果"""
    print("\n" + "="*80)
    print("对比修复前后的结果")
    print("="*80)
    
    old_files = {f for f in os.listdir(old_dir) if f.endswith('.npz')}
    new_files = {f for f in os.listdir(new_dir) if f.endswith('.npz')}
    common_files = old_files & new_files
    
    print(f"\n找到 {len(common_files)} 个共同文件")
    
    results = []
    for filename in sorted(common_files):
        old_path = os.path.join(old_dir, filename)
        new_path = os.path.join(new_dir, filename)
        
        old_data = np.load(old_path)
        new_data = np.load(new_path)
        
        old_ifa = calculate_ifa(old_data['labels'], old_data['true_prob'])
        new_ifa = calculate_ifa(new_data['labels'], new_data['true_prob'])
        
        improvement = old_ifa - new_ifa
        results.append({
            'filename': filename,
            'old_ifa': old_ifa,
            'new_ifa': new_ifa,
            'improvement': improvement
        })
    
    df = pd.DataFrame(results)
    print(f"\n{'文件名':<40} {'修复前IFA':>12} {'修复后IFA':>12} {'改善量':>12}")
    print("-" * 80)
    for _, row in df.iterrows():
        status = "✓" if row['improvement'] > 0 else "✗"
        print(f"{status} {row['filename']:<38} {row['old_ifa']:>12.0f} {row['new_ifa']:>12.0f} {row['improvement']:>12.0f}")
    
    print("\n统计:")
    print(f"  平均IFA改善: {df['improvement'].mean():.1f}")
    print(f"  改善的文件数: {(df['improvement'] > 0).sum()} / {len(df)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  1. 检查单个文件: python verify_fix.py <npz_file_path>")
        print("  2. 对比两个目录: python verify_fix.py <old_dir> <new_dir>")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # 单文件检查模式
        filepath = sys.argv[1]
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在 {filepath}")
            sys.exit(1)
        load_and_check_npz(filepath)
        
    elif len(sys.argv) == 3:
        # 目录对比模式
        old_dir = sys.argv[1]
        new_dir = sys.argv[2]
        
        if not os.path.isdir(old_dir):
            print(f"错误: 目录不存在 {old_dir}")
            sys.exit(1)
        if not os.path.isdir(new_dir):
            print(f"错误: 目录不存在 {new_dir}")
            sys.exit(1)
            
        compare_directories(old_dir, new_dir)