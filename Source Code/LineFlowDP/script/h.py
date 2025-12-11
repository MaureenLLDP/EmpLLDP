import os
import torch
from MyHeteroDataset import MyHeteroDataset
from torch_geometric.loader import DataLoader

def check_dataset_info(proj='activemq', release='activemq-5.0.0', data_root='./data/'):
    """检查数据集的标签格式和基本信息"""
    
    print(f"{'='*60}")
    print(f"检查数据集: {proj} - {release}")
    print(f"{'='*60}\n")
    
    # 加载数据集
    dataset = MyHeteroDataset(
        root=os.path.join(data_root, proj), 
        name=release, 
        use_node_attr=True, 
        force_reload=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 检查第一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n第一个样本信息:")
        print(f"  - 数据类型: {type(sample)}")
        print(f"  - 标签 (y): {sample.y}")
        print(f"  - 标签类型: {sample.y.dtype}")
        print(f"  - 标签形状: {sample.y.shape}")
        print(f"  - 标签值: {sample.y.item() if sample.y.numel() == 1 else sample.y.tolist()}")
    
    # 统计所有标签
    all_labels = []
    for data in dataset:
        label = data.y.item() if data.y.numel() == 1 else data.y.tolist()
        all_labels.append(label)
    
    print(f"\n标签统计:")
    print(f"  - 唯一标签值: {sorted(set(all_labels))}")
    print(f"  - 标签数量分布:")
    from collections import Counter
    label_counts = Counter(all_labels)
    for label, count in sorted(label_counts.items()):
        percentage = count / len(all_labels) * 100
        print(f"    标签 {label}: {count} 个 ({percentage:.2f}%)")
    
    # 检查batch后的情况
    print(f"\n检查DataLoader batch:")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    
    print(f"  - Batch类型: {type(batch)}")
    print(f"  - Batch.y: {batch.y}")
    print(f"  - Batch.y 类型: {batch.y.dtype}")
    print(f"  - Batch.y 形状: {batch.y.shape}")
    print(f"  - Batch.y 值: {batch.y.tolist()}")
    
    # 检查节点特征
    print(f"\n节点特征信息:")
    print(f"  - 特征形状: {batch['node'].x.shape}")
    print(f"  - 特征维度: {batch['node'].x.shape[1]}")
    
    # 检查边信息
    print(f"\n边信息:")
    print(f"  - Edge index 形状: {batch['node', 'edge', 'node'].edge_index.shape}")
    print(f"  - Edge type 形状: {batch['node', 'edge', 'node'].edge_type.shape}")
    print(f"  - 唯一的 edge types: {torch.unique(batch['node', 'edge', 'node'].edge_type).tolist()}")
    
    print(f"\n{'='*60}")
    print("检查完成！")
    print(f"{'='*60}")
    
    return dataset, all_labels


if __name__ == '__main__':
    # 你可以修改这些参数来检查不同的数据集
    dataset, labels = check_dataset_info(
        proj='activemq',
        release='activemq-5.0.0',
        data_root='./data/'
    )
    
    # 额外建议
    print("\n根据检查结果的建议:")
    unique_labels = sorted(set(labels))
    if len(unique_labels) == 2 and unique_labels == [0, 1]:
        print("✓ 标签是二分类 (0/1)，适合使用 BCEWithLogitsLoss")
        print("✓ 需要将模型输出改为 out_channels=1")
        print("✓ 需要将标签转换为 float 类型")
    elif len(unique_labels) == 2:
        print(f"✓ 标签是二分类，但值为 {unique_labels}")
        print("  需要将标签映射到 0/1")
    else:
        print(f"✗ 标签有 {len(unique_labels)} 个类别: {unique_labels}")
        print("  这不是二分类问题，不适合用 BCELoss")