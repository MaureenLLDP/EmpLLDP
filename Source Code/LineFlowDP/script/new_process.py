import os
import numpy as np
import torch
from torch_geometric.data import Data

# 配置区 ==============================================
DATA_ROOT = './data/'  # 数据根目录
REQUIRED_FILES = [  # 必须包含的文件后缀
    '_A.txt',
    '_edge_labels.txt',
    '_graph_indicator.txt',
    '_graph_labels.txt',
    '_node_attributes.txt',
    '_node_labels.txt'
]


# =====================================================

def validate_file_structure(project, version):
    """严格校验文件结构"""
    raw_path = os.path.join(DATA_ROOT, project, version, 'raw')
    missing = [f"{version}{suffix}" for suffix in REQUIRED_FILES if not os.path.exists(os.path.join(raw_path, f"{version}{suffix}"))]

    if missing:
        print(f"❌ 缺失文件 ({len(missing)} 个): {missing}")
        return False
    return True


def load_data_file(project, version, suffix, dtype=torch.float, delimiter=','):
    """安全加载 .txt 文件"""
    path = os.path.join(DATA_ROOT, project, version, 'raw', f"{version}{suffix}")

    try:
        array = np.loadtxt(path, delimiter=delimiter, ndmin=2)
        # 关键修复：将1-based索引转为0-based
        if suffix in ['_A.txt', '_graph_indicator.txt']:
            array -= 1  # 节点编号从0开始
        return torch.tensor(array, dtype=dtype)
    except Exception as e:
        raise RuntimeError(f"加载 {os.path.basename(path)} 失败: {e}")


def reindex_edge_index(edge_index, graph_indicator, graph_id):
    """重新索引 edge_index，使其变为局部索引"""
    node_mask = (graph_indicator == graph_id)  # 该图的所有节点
    node_indices = torch.nonzero(node_mask).squeeze()
    num_nodes = node_indices.shape[0]

    if num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long)  # 无边，返回空 edge_index

    # 创建全局索引到局部索引的映射
    reindex_map = {int(node_indices[i]): i for i in range(len(node_indices))}

    # 过滤边，只保留当前图的节点
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]  # 仅保留属于当前图的边

    if edge_index.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.long)  # 无边，返回空 edge_index

    # 重新映射 edge_index
    edge_index = torch.tensor([[reindex_map[int(src)], reindex_map[int(dst)]]
                               for src, dst in edge_index.t().tolist()], dtype=torch.long).t()

    return edge_index


def process_project(project, versions):
    """处理单个项目的所有版本"""
    print(f"\n🔍 开始处理项目: {project}")

    total = 0
    for ver in versions:
        print(f"\n🔄 处理版本: {ver}")

        # 校验文件结构
        if not validate_file_structure(project, ver):
            continue

        try:
            # 加载所有必要文件
            edge_index = load_data_file(project, ver, '_A.txt', torch.long, ',').t().contiguous()
            edge_labels = load_data_file(project, ver, '_edge_labels.txt', torch.long)
            graph_indicator = load_data_file(project, ver, '_graph_indicator.txt', torch.long).squeeze()
            graph_labels = load_data_file(project, ver, '_graph_labels.txt', torch.long)
            node_attrs = load_data_file(project, ver, '_node_attributes.txt')
            node_labels = load_data_file(project, ver, '_node_labels.txt', torch.long)

            # 处理图数据
            data_list = []
            for graph_id in torch.unique(graph_indicator):
                node_mask = (graph_indicator == graph_id)
                edge_index_graph = reindex_edge_index(edge_index, graph_indicator, graph_id)

                # 过滤 edge_attr，仅保留当前图的边特征
                edge_attr_graph = edge_labels[edge_index_graph[0]] if edge_labels is not None else None

                data = Data(
                    x=node_attrs[node_mask],
                    y=graph_labels[graph_id],
                    edge_index=edge_index_graph,
                    edge_attr=edge_attr_graph,
                    node_label=node_labels[node_mask]
                )
                data_list.append(data)

            if edge_index_graph.numel() == 0:
                print(f"⚠️ 图 {graph_id.item()} 没有边！跳过 min/max 计算")
            else:
                print(f"✅ 图 {graph_id.item()} 处理完成: 节点={node_attrs[node_mask].shape[0]}, "
                      f"边={edge_index_graph.shape[1]}, edge_index 范围=[{edge_index_graph.min().item()}, {edge_index_graph.max().item()}]")

            # 保存结果
            save_dir = os.path.join(DATA_ROOT, project, ver, 'processed')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(data_list, os.path.join(save_dir, 'data.pt'))
            print(f"✅ 成功保存 {len(data_list)} 个图数据")
            total += len(data_list)

        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")

    return total


if __name__ == "__main__":
    # 项目配置（版本参数使用完整格式）
    all_releases = {
        "activemq": ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
        "camel": ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
        "derby": ["derby-10.2.1.6", "derby-10.3.1.4", "derby-10.5.1.1"]
    }

    total = 0
    for project, versions in all_releases.items():
        total += process_project(project, versions)

    print(f"\n🏁 处理完成！共处理 {total} 个图")
