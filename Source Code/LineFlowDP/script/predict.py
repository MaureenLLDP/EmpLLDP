import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.utils import to_networkx, k_hop_subgraph
from tqdm import tqdm
from math import sqrt

from train_rgcn import RGCN, all_data


EPS = 1e-15  # 避免数值计算错误


class GNNExplainer(torch.nn.Module):
    def __init__(self, model, epochs=100, lr=0.01, num_hops=None, return_type='log_prob', log=True):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.num_hops = num_hops
        self.return_type = return_type
        self.log = log

    def set_masks(self, x, edge_index, edge_attr):
        """ 初始化节点特征和边掩码 """
        F, E = x.size(1), edge_index.size(1)
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * 0.1)

        for module in self.model.modules():
            if isinstance(module, RGCNConv):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def clear_masks(self):
        """ 清除掩码，恢复模型状态 """
        for module in self.model.modules():
            if isinstance(module, RGCNConv):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_mask = None
        self.edge_mask = None

    def loss(self, log_logits, pred_label):
        """ 计算损失，优化掩码 """
        loss = -log_logits[0, pred_label[0]]
        m = self.edge_mask.sigmoid()
        loss += 0.005 * m.sum()  # 控制边掩码大小
        loss += 1.0 * (-m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)).mean()  # 熵正则化
        return loss

    def explain_graph(self, data):
        """ 解释整个图，返回节点重要性和边重要性 """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        self.model.eval()
        self.clear_masks()

        with torch.no_grad():
            out = self.model(x, edge_index, edge_attr.argmax(dim=1), data.batch)
            log_logits = F.log_softmax(out, dim=1)
            pred_label = log_logits.argmax(dim=-1)

        self.set_masks(x, edge_index, edge_attr)
        self.to(x.device)
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.epochs, desc="Explaining graph")

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            out = self.model(h, edge_index, edge_attr.argmax(dim=1), data.batch)
            log_logits = F.log_softmax(out, dim=1)
            loss = self.loss(log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        return self.node_feat_mask.detach().sigmoid(), self.edge_mask.detach().sigmoid()


# ----------------- 计算代码行风险评分 -----------------
def compute_sna_risk(data, edge_mask):
    """ 计算 SNA 指标：度中心性、Katz 中心性、接近中心性 """
    G = to_networkx(data, to_undirected=True)

    # 计算 Degree Centrality
    degree_centrality = nx.degree_centrality(G)

    # 计算 Katz Centrality
    try:
        katz_centrality = nx.katz_centrality_numpy(G, alpha=0.01, beta=1.0)
    except nx.NetworkXError:
        katz_centrality = {n: 0 for n in G.nodes}

    # 计算 Closeness Centrality
    closeness_centrality = nx.closeness_centrality(G)

    # 归一化
    max_degree = max(degree_centrality.values()) if degree_centrality else 1
    max_katz = max(katz_centrality.values()) if katz_centrality else 1
    max_closeness = max(closeness_centrality.values()) if closeness_centrality else 1

    risk_scores = {}
    for node in G.nodes:
        risk_scores[node] = (
                (degree_centrality[node] / max_degree) +
                (katz_centrality[node] / max_katz) +
                (closeness_centrality[node] / max_closeness)
        )

    return risk_scores


# ----------------- 运行完整的分析流程 -----------------
def analyze_defective_file(model, data):
    """ 运行完整流程：使用 GNNExplainer 解释预测结果，并计算 SNA 风险 """
    explainer = GNNExplainer(model)
    node_mask, edge_mask = explainer.explain_graph(data)

    # 计算 SNA 风险
    risk_scores = compute_sna_risk(data, edge_mask)

    # 排序，输出风险最高的代码行
    sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n🔍 代码行风险排名（从高到低）：")
    for node, score in sorted_risks[:10]:  # 只显示前 10 行
        print(f"行 {node}: 风险评分 {score:.4f}")

    return sorted_risks


def recall_at_top_20_loc(risk_scores, defective_lines, total_lines):
    """
    计算 Recall@Top20%LOC
    :param risk_scores: {行号: 风险评分}，按评分降序排列
    :param defective_lines: 真实缺陷行的集合 {行号1, 行号2, ...}
    :param total_lines: 总代码行数
    :return: Recall@Top20%LOC
    """
    top_20_loc = int(total_lines * 0.2)  # 计算前 20% 代码行数
    sorted_lines = sorted(risk_scores, key=risk_scores.get, reverse=True)  # 按风险降序排序
    selected_lines = set(sorted_lines[:top_20_loc])  # 取前 20% 行

    found_defective = len(selected_lines & defective_lines)  # 计算找到的缺陷行数
    total_defective = len(defective_lines)  # 真实的缺陷行数

    return found_defective / total_defective if total_defective > 0 else 0.0


def effort_at_top_20_recall(risk_scores, defective_lines, total_lines):
    """
    计算 Effort@Top20%Recall
    :param risk_scores: {行号: 风险评分}，按评分降序排列
    :param defective_lines: 真实缺陷行的集合 {行号1, 行号2, ...}
    :param total_lines: 总代码行数
    :return: Effort@Top20%Recall
    """
    total_defective = len(defective_lines)
    top_20_defective = int(total_defective * 0.2)  # 需要找到的缺陷行数
    sorted_lines = sorted(risk_scores, key=risk_scores.get, reverse=True)  # 按风险降序排序

    found_defective = 0
    effort_LOC = 0

    for line in sorted_lines:
        effort_LOC += 1
        if line in defective_lines:
            found_defective += 1
        if found_defective >= top_20_defective:
            break

    return effort_LOC / total_lines if total_lines > 0 else 0.0


def initial_false_alarms(risk_scores, defective_lines):
    """
    计算 IFA（Initial False Alarms）
    :param risk_scores: {行号: 风险评分}，按评分降序排列
    :param defective_lines: 真实缺陷行的集合 {行号1, 行号2, ...}
    :return: IFA 值
    """
    sorted_lines = sorted(risk_scores, key=risk_scores.get, reverse=True)  # 按风险降序排序

    false_alarms = 0
    for line in sorted_lines:
        if line in defective_lines:
            break  # 发现第一个缺陷行，停止计数
        false_alarms += 1

    return false_alarms

def top_k_accuracy(risk_scores, defective_lines, k):
    """
    计算 Top-k Accuracy
    :param risk_scores: {行号: 风险评分}，按评分降序排列
    :param defective_lines: 真实缺陷行的集合 {行号1, 行号2, ...}
    :param k: 选取前 k 行
    :return: 1（包含缺陷）或 0（不包含缺陷）
    """
    sorted_lines = sorted(risk_scores, key=risk_scores.get, reverse=True)[:k]  # 取前 k 行
    return 1 if any(line in defective_lines for line in sorted_lines) else 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------- 示例用法 -----------------
if __name__ == "__main__":
    # 确保这些参数与训练时一致
    num_classes = len(torch.unique(torch.stack([data.y for data in all_data])))
    num_relations = all_data[0].edge_attr.size(1) if all_data[0].edge_attr.dim() > 1 else 1
    num_features = all_data[0].x.shape[1]

    # 重新实例化 RGCN 并加载参数
    model = RGCN(num_features, 128, num_classes, num_relations)
    model.load_state_dict(torch.load("rgcn_model.pth"))
    model.to(device)
    model.eval()  # 进入评估模式

    data = torch.load("./data/activemq/activemq-5.0.0/processed/data.pt")  # 加载数据
    num_defective = sum(1 for d in data if d.y.item() == 1)
    num_non_defective = sum(1 for d in data if d.y.item() == 0)

    if isinstance(data, list):
        print(f"📊 发现 {len(data)} 个图数据，逐个分析...")
        for i, graph in enumerate(data[:5]):  # 遍历前 5 个图
            print(f"\n🔍 分析第 {i + 1} 个图...")
            sorted_risks =analyze_defective_file(model, graph)
    else:
        sorted_risks =analyze_defective_file(model, data)
    risk_scores = {node: score for node, score in sorted_risks}
    for graph in data:  # 遍历 data 列表中的每个 PDG
        defective_lines = {i for i, label in enumerate(graph.node_label) if label == 1}
        total_lines = len(graph.node_label)

        print(f"📌 代码总行数: {total_lines}")
        print(f"📌 真实缺陷行数: {len(defective_lines)}, 缺陷行列表: {defective_lines}")

    recall = recall_at_top_20_loc(risk_scores, defective_lines, total_lines)
    effort = effort_at_top_20_recall(risk_scores, defective_lines, total_lines)
    ifa = initial_false_alarms(risk_scores, defective_lines)
    top_k_acc = top_k_accuracy(risk_scores, defective_lines, k=10)

    print(f"Recall@Top20%LOC: {recall:.4f}")
    print(f"Effort@Top20%Recall: {effort:.4f}")
    print(f"IFA: {ifa}")
    print(f"Top-k Accuracy: {top_k_acc}")










