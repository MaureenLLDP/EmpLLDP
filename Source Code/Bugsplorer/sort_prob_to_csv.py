import numpy as np
import pandas as pd

def process_npz_to_csv(input_file, output_file):
    """
    读取.npz文件，按true_prob排序并保存为CSV
    
    参数:
        input_file: 输入的.npz文件路径
        output_file: 输出的.csv文件路径
    """
    # 1. 加载.npz文件
    data = np.load(input_file)
    
    # 2. 提取true_prob和labels
    true_prob = data['true_prob']
    labels = data['labels']
    
    # 3. 检查数据形状是否一致
    if len(true_prob) != len(labels):
        raise ValueError("true_prob和labels的长度不一致！")
    
    # 4. 按true_prob降序排序（概率高的排前面）
    sorted_indices = np.argsort(true_prob)[::-1]  # 获取排序后的索引
    sorted_true_prob = true_prob[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # 5. 创建DataFrame
    df = pd.DataFrame({
        'true_prob': sorted_true_prob,
        'label': sorted_labels,
        'rank': np.arange(1, len(sorted_true_prob)+1)  # 添加排名列
    })
    
    # 6. 保存为CSV
    df.to_csv(output_file, index=False)
    print(f"已成功保存排序结果到: {output_file}")

# 使用示例
if __name__ == "__main__":
    input_path = "outputs/roberta-linedp-activemq-regular.npz"  # 替换为你的.npz文件路径
    output_path = "sorted_results.csv"  # 输出的CSV文件名
    
    process_npz_to_csv(input_path, output_path)