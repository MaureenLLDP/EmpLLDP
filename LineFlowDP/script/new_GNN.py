import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from tqdm import tqdm
import os
from train_rgcn import RGCN  # ⬅️ 确保你的 `train_rgcn.py` 里定义了 RGCN 类
import argparse

EPS = 1e-15  # 避免数值计算错误


class GNNExplainer(torch.nn.Module):
    def __init__(self, model, epochs=100, lr=0.01, log=True):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log

        # 正则化参数
        self.coeffs = {
            'edge_size': 0.005,
            'node_feat_size': 1.0,
            'edge_ent': 1.0,
            'node_feat_ent': 0.1,
        }

    def set_masks(self, x, edge_index):
        """ 初始化节点特征和边掩码 """
        N, F = x.shape  # 节点数, 特征维度
        E = edge_index.size(1)  # 边数

        self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * 0.1)
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * 0.1)

        # 设置 GNN 解释模式
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def clear_masks(self):
        """ 清除掩码 """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_mask = None
        self.edge_mask = None

    def loss(self, log_logits, pred_label):
        """ 计算损失 """
        loss = -log_logits[0, pred_label[0]]  # 计算图级损失
        loss += self.coeffs['edge_size'] * self.edge_mask.sigmoid().sum()
        loss += self.coeffs['node_feat_size'] * self.node_feat_mask.sigmoid().sum()
        return loss

    def explain_graph(self, data):
        """ 解释整个图，返回节点重要性和边重要性 """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        self.model.eval()
        self.clear_masks()

        # 处理 edge_attr 作为 edge_type
        edge_type = edge_attr.argmax(dim=1) if edge_attr.dim() > 1 else edge_attr

        with torch.no_grad():
            out = self.model(x, edge_index, edge_type, data.batch)
            log_logits = F.log_softmax(out, dim=1)
            pred_label = log_logits.argmax(dim=-1)

        self.set_masks(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.epochs, desc="Explaining graph")

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            out = self.model(h, edge_index, edge_type, data.batch)
            log_logits = F.log_softmax(out, dim=1)
            loss = self.loss(log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        return self.node_feat_mask.detach().sigmoid(), self.edge_mask.detach().sigmoid()

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



# ----------------- 加载模型和数据 -----------------
def load_model(model_path, num_features, num_classes, num_relations, device):
    """ 加载训练好的 RGCN 模型 """
    model = RGCN(num_features, 128, num_classes, num_relations).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_data(data_path):
    """ 加载数据集 """
    data_list = torch.load(data_path)
    return data_list

def analyze_defective_file(model, data):
    """ 运行完整流程：使用 GNNExplainer 解释预测结果，并计算 SNA 风险 """
    explainer = GNNExplainer(model)
    node_mask, edge_mask = explainer.explain_graph(data)

    # 计算风险分数
    risk_scores = {i: node_mask[i].mean().item() for i in range(len(node_mask))}

    # 获取真实缺陷行
    defective_lines = {i for i, label in enumerate(data.node_label) if label == 1}
    total_lines = len(data.node_label)

    # 计算评估指标
    recall = recall_at_top_20_loc(risk_scores, defective_lines, total_lines)
    effort = effort_at_top_20_recall(risk_scores, defective_lines, total_lines)
    ifa = initial_false_alarms(risk_scores, defective_lines)
    top_k_acc = top_k_accuracy(risk_scores, defective_lines, k=10)

    # 输出排名
    sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n🔍 代码行风险排名（前 10 个）：")
    for node, score in sorted_risks[:10]:
        print(f"行 {node}: 风险评分 {score:.4f}")

    # 输出评估结果
    print(f"\n📊 Recall@Top20%LOC: {recall:.4f}")
    print(f"📊 Effort@Top20%Recall: {effort:.4f}")
    print(f"📊 Initial False Alarms: {ifa}")
    print(f"📊 Top-10 Accuracy: {top_k_acc}")

    return sorted_risks

# ----------------- 主函数 -----------------
def main():
    DATA_PATH = "./data/activemq/activemq-5.0.0/processed/data.pt"
    MODEL_PATH = "rgcn_model.pth"
    parser = argparse.ArgumentParser(description="Run GNNExplainer on RGCN model.")
    parser.add_argument("--data", type=str, default="./data/activemq/activemq-5.0.0/processed/data.pt", help="Path to data.pt")
    parser.add_argument("--model", type=str, default="rgcn_model.pth", help="Path to trained model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    data_list = load_data(DATA_PATH)
    print(f"✅ 加载数据成功，发现 {len(data_list)} 个图")

    # 选择第一个 PDG 进行分析
    data = data_list[0].to(device)

    # 获取模型参数
    num_features = data.x.shape[1]
    num_classes = 2  # 假设二分类
    num_relations = data.edge_attr.size(1) if data.edge_attr.dim() > 1 else 1

    # 加载模型
    model = load_model(MODEL_PATH, num_features, num_classes, num_relations, device)
    print("✅ 模型加载成功！")

    # 遍历所有 PDG 并分析
    # for idx, data in enumerate(data_list):
    #     data = data.to(device)
    #     print(f"\n 分析第 {idx + 1} 个图...")
    #     analyze_defective_file(model, data)
    data = data_list[2].to(device)
    analyze_defective_file(model, data)
    defective_lines = {i for i, label in enumerate(data.y) if label == 1}
    print(f"📌 `y` 唯一值: {torch.unique(data.y) if hasattr(data, 'y') else '无 y'}")
    print(
        f"📌 `graph_labels` 唯一值: {torch.unique(data.graph_labels) if hasattr(data, 'graph_labels') else '无 graph_labels'}")

    print(f"📌 真实缺陷行: {defective_lines}")
    print(f"📌 `node_label` 唯一值: {torch.unique(data.node_label)}")


if __name__ == "__main__":
    main()
