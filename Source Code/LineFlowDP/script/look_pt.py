import torch

DATA_PATH = "./data/activemq/activemq-5.0.0/processed/data.pt"
data_list = torch.load(DATA_PATH)

print(f"✅ 成功加载 {len(data_list)} 个图数据")

# 遍历所有图，检查 edge_index 是否超出范围
for idx, data in enumerate(data_list):
    num_nodes = data.x.shape[0]  # 该图的节点数

    if data.edge_index.numel() == 0:
        print(f"⚠️ 警告：图 {idx} 没有边！跳过 min/max 计算")
        continue  # 直接跳过该图

    max_edge_index = data.edge_index.max().item()
    min_edge_index = data.edge_index.min().item()

    print(f"🔍 图 {idx}: 节点数={num_nodes}, edge_index 范围=[{min_edge_index}, {max_edge_index}]")

    # 检查是否有超出范围的索引
    if max_edge_index >= num_nodes or min_edge_index < 0:
        print(f"❌ 警告：图 {idx} 的 edge_index 存在无效索引！")
