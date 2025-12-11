import torch
import sys
from torch.utils.data import TensorDataset

def verify_dataset_structure(file_path):
    """
    加载一个 PyTorch 的 .pt 数据集文件，并检查其内部结构。
    """
    try:
        print(f"--- 正在加载数据集文件: {file_path} ---")
        dataset = torch.load(file_path)
        print("✅ 文件加载成功!")

        if not isinstance(dataset, TensorDataset) or len(dataset) == 0:
            print("❌ 错误: 加载的文件不是一个有效或非空的数据集 (TensorDataset)。")
            return

        print(f"\n数据集中样本总数: {len(dataset)}")

        # 检查第一个样本的结构
        first_sample = dataset[0]
        num_tensors = len(first_sample)

        print(f"\n>>> 在第一个样本中发现了 {num_tensors} 个张量。 <<<")
        print("------------------------------------------")

        for i, tensor in enumerate(first_sample):
            print(f"张量 #{i+1}:")
            print(f"  - 数据类型 (dtype): {tensor.dtype}")
            print(f"  - 形状 (shape): {tensor.shape}")

        print("------------------------------------------")
        if num_tensors == 4:
            print("\n[结论] 👉 这个数据集确实是【新格式】（包含 global_index 和 file_id）。")
        elif num_tensors == 2 or num_tensors == 3:
            print("\n[结论] 👉 这个数据集仍然是【旧格式】。")
        else:
            print("\n[结论] 👉 数据集格式未知，不符合预期。")

    except Exception as e:
        print(f"\n❌ 在尝试读取文件时发生错误: {e}")
        print("请再三确认文件路径是否正确，以及该文件是否确实是一个 PyTorch TensorDataset 文件。")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python verify_dataset.py <你的 .pt 文件路径>")
        print("例如: python verify_dataset.py cache/roberta-linedp-activemq-version/train.pt")
    else:
        verify_dataset_structure(sys.argv[1])