import json
import os
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from torch_geometric.loader import DataLoader
from MyHeteroDataset import MyHeteroDataset
from gnn_models.gnn_models import MyRGCN, MyGCN
import warnings

from GNN_Explainer import GNNExplainer

warnings.filterwarnings('ignore')

result_path = './results/'
os.makedirs(result_path, exist_ok=True)


# def test(model, loader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_probs = []
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             out = model(
#                 data["node"].x,
#                 data["node", "edge", "node"].edge_index,
#                 data["node", "edge", "node"].edge_type,
#                 data["node"].batch
#             )
#             prob = out.softmax(dim=-1)
#             pred = prob.argmax(dim=-1)
#             all_preds.append(pred.cpu())
#             all_labels.append(data.y.cpu())
#             all_probs.append(prob.cpu())
#     preds = torch.cat(all_preds)
#     probs = torch.cat(all_probs).numpy()
#     labels = torch.cat(all_labels)
#     acc = (preds == labels).float().mean().item()
#     bal_acc = balanced_accuracy_score(labels, preds)
#     mcc = matthews_corrcoef(labels, preds)
#     try:
#         auc = roc_auc_score(labels, probs[:, 1])
#     except Exception as e:
#         auc = float('nan')
#         print(f"Warning: AUC could not be computed: {e}")

#     print(
#         f"{'Test Accuracy:':2s} {acc:.4f} | {'Balanced Acc:':2s} {bal_acc:.4f} | {'MCC:':2s} {mcc:.4f} | {'AUC:':2s} {auc:.4f}")

#     return preds, labels, probs
def test(model, loader, device, threshold=0.5):
    """适配 BCE 输出 (out_channels=1)"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(
                data["node"].x,
                data["node", "edge", "node"].edge_index,
                data["node", "edge", "node"].edge_type,
                data["node"].batch
            )
            
            # 👇 BCE 输出处理
            prob = out.squeeze(-1)# (batch_size, 1) -> (batch_size,)
            pred = (prob > threshold).long()  # 二值化预测
            
            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())
            all_probs.append(prob.cpu())
    
    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs).numpy()  # 形状: (N,)
    labels = torch.cat(all_labels)
    
    acc = (preds == labels).float().mean().item()
    bal_acc = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    
    try:
        auc = roc_auc_score(labels, probs)  # 👈 直接用概率值
    except Exception as e:
        auc = float('nan')
        print(f"Warning: AUC could not be computed: {e}")

    print(f"Test Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | MCC: {mcc:.4f} | AUC: {auc:.4f}")
    
    return preds, labels, probs


def norm(x):
    x = np.array(list(x.values()))
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--test_release', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='RGCN', choices=['RGCN', 'GCN'])
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--in_channels', default=100, type=int, help='Input feature dimension')
    parser.add_argument('--hidden_channels', default=256, type=int)
    parser.add_argument('--out_channels', default=2, type=int, help='num of classes')
    parser.add_argument('--num_relations', default=4, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--threshold', default=0.5, type=float, help='BCE threshold')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--explain_ratio', default=1.0, type=float, help='Ratio of edges to explain in the subgraph')

    args = parser.parse_args()

    with open(os.path.join(args.checkpoint_dir, f"{args.project}_{args.backbone}_config.json")) as f:
        config = json.load(f)
    args.in_channels = config["in_channels"]
    args.hidden_channels = config["hidden_channels"]
    args.out_channels = config["out_channels"]
    args.num_relations = config["num_relations"]
    args.num_layers = config["num_layers"]
    args.dropout = config["dropout"]

    data_dir = './data/' + args.project
    dataset = MyHeteroDataset(root=data_dir, name=args.test_release, use_node_attr=True)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.backbone == 'RGCN':
        model = MyRGCN(
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_relations=args.num_relations,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = MyGCN(
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    model_path = os.path.join(args.checkpoint_dir, f"{args.project}.pkl")
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model = model.to(args.device)

    print(f"Testing {args.project}, release={args.test_release}, backbone={args.backbone} ...")
    #preds, labels, probs = test(model, test_loader, args.device)
    preds, labels, probs = test(model, test_loader, args.device, args.threshold)

    explainer = GNNExplainer(model, epochs=200)
    centrality_records = []

    for idx in range(len(dataset)):
        data = dataset[int(idx)].to(args.device)
        #file_pred_prob = probs[int(idx), 1]
        file_pred_prob = probs[int(idx)]

        file_pred_label = int(preds[int(idx)].item())
      
        if file_pred_label == 1:
            node_feat_mask, edge_mask = explainer.explain_graph(data)
            n_edges = edge_mask.shape[0]
            if n_edges == 0:
                subgraph_node_ids = set()
                katz_norm = dict()
                deg_norm = dict()
                clo_norm = dict()
            else:
                explain_ratio = args.explain_ratio
                topk = min(n_edges, max(1, int(explain_ratio * n_edges)))
                top_edges = edge_mask.topk(topk)[1].cpu().numpy()
                edge_index = data["node", "edge", "node"].edge_index[:, top_edges]
                G = nx.Graph()
                G.add_edges_from(edge_index.t().cpu().numpy())
                subgraph_node_ids = set(G.nodes())
                file_line_map = {nid: data["node"].node_id[nid] for nid in range(data["node"].x.size(0))}

                if len(subgraph_node_ids) > 0:
                    if len(G) <= 1:
                        print("Graph too small for Katz centrality.")
                        katz = 0
                    else:
                        katz = nx.katz_centrality_numpy(G)
                    degree = nx.degree_centrality(G)
                    closeness = nx.closeness_centrality(G)
                    katz_node_order = list(subgraph_node_ids)
                    if len(subgraph_node_ids) == 1:
                        katz_norm = {nid: 1.0 for nid in subgraph_node_ids}
                        deg_norm = {nid: 1.0 for nid in subgraph_node_ids}
                        clo_norm = {nid: 1.0 for nid in subgraph_node_ids}
                    else:
                        katz_scores = np.array([katz[nid] for nid in katz_node_order])
                        deg_scores = np.array([degree[nid] for nid in katz_node_order])
                        clo_scores = np.array([closeness[nid] for nid in katz_node_order])
                        katz_norm_vals = (katz_scores - katz_scores.min()) / (
                                katz_scores.max() - katz_scores.min() + 1e-9)
                        deg_norm_vals = (deg_scores - deg_scores.min()) / (deg_scores.max() - deg_scores.min() + 1e-9)
                        clo_norm_vals = (clo_scores - clo_scores.min()) / (clo_scores.max() - clo_scores.min() + 1e-9)
                        katz_norm = {nid: float(katz_norm_vals[j]) for j, nid in enumerate(katz_node_order)}
                        deg_norm = {nid: float(deg_norm_vals[j]) for j, nid in enumerate(katz_node_order)}
                        clo_norm = {nid: float(clo_norm_vals[j]) for j, nid in enumerate(katz_node_order)}
                else:
                    katz_norm = dict()
                    deg_norm = dict()
                    clo_norm = dict()
        else:

            subgraph_node_ids = set()
            katz_norm = dict()
            deg_norm = dict()
            clo_norm = dict()

        for nid in range(data["node"].x.size(0)):
            file_line = data["node"].node_id[nid]
            parts = file_line.split("::", 2)
            if len(parts) == 3:
                release, filename, line_number = parts
            elif len(parts) == 2:
                release, filename = parts
                line_number = ""
            else:
                release = parts[0]
                filename = ""
                line_number = ""
            is_key = int(nid in subgraph_node_ids)

            file_label = int(data.y.item()) if hasattr(data, "y") else None

            line_label = int(data["node"].y[nid].item()) if "y" in data["node"] else None
            centrality_records.append({
                "release": release,
                "graph_idx": int(idx),
                "filename": filename,
                "line_number": line_number,
                "is_key_subgraph": is_key,
                "katz": float(katz_norm.get(nid, 0.0)),
                "degree": float(deg_norm.get(nid, 0.0)),
                "closeness": float(clo_norm.get(nid, 0.0)),
                "file_pred_prob": float(file_pred_prob),
                "file_pred_label": int(file_pred_label),
                "file_label": file_label,
                "line_label": line_label
            })

    pd.DataFrame(centrality_records).to_csv(f"{result_path}/{args.test_release}.csv", index=False)


if __name__ == "__main__":
    # try:
    #     # 尝试执行主程序
    #     main()
    # finally:
    #     # 无论 main() 是否成功，这个块里的代码都将被执行
    #     print("准备执行关机命令...")
    #     os.system("shutdown -h now")
    main()