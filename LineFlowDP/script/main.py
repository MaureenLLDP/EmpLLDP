import json
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight
from torch_geometric.loader import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
from MyHeteroDataset import MyHeteroDataset
from gnn_models.gnn_models import MyRGCN, MyGCN
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser(description="Train RGCN on Software Defect Dataset")
    parser.add_argument('--project', default='activemq', type=str, help='project name')
    parser.add_argument('--train_release', default='activemq-5.0.0', type=str)
    parser.add_argument('--valid_release', default='activemq-5.1.0', type=str)
    parser.add_argument('--test_release', default='activemq-5.2.0', type=str)
    parser.add_argument('--data_root', default='./data/', type=str)
    parser.add_argument('--checkpoint_dir', default='./checkpoints/', type=str)
    parser.add_argument('--in_channels', default=100, type=int, help='Input feature dimension')
    parser.add_argument('--hidden_channels', default=256, type=int)
    parser.add_argument('--out_channels', default=1, type=int, help='Output dimension (1 for BCE)')  # 改为1
    parser.add_argument('--backbone', default='RGCN', type=str, help='The backbone of GNN,[RGCN, GCN]')
    parser.add_argument('--num_relations', default=4, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epochs', default=300, type=int)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--optimizer', default='AdamW', choices=['Adam', 'AdamW'])
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor by which the learning rate will be reduced. new_lr = lr * factor. (default: 0.5)')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum learning rate allowed during training. (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=25,
                        help='Number of epochs with no improvement after which learning rate will be reduced. (default: 25)')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use for training (default: cuda)')
    parser.add_argument('--use_balanced_loss', action='store_true', default=True,
                        help='Whether to use class-balanced loss (default: True)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--threshold', default=0.5, type=float, help='Prediction threshold for binary classification')
    return parser.parse_args()


def load_dataset(proj, release, data_root):
    dataset = MyHeteroDataset(root=os.path.join(data_root, proj), name=release, use_node_attr=True, force_reload=True)
    return dataset


def train_one_epoch(model, optimizer, criterion, loader, device, class_weights=None):
    """训练一个epoch，支持样本级加权"""
    model.train()
    total_loss = 0.
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        out = model(
            batch["node"].x,
            batch["node", "edge", "node"].edge_index,
            batch["node", "edge", "node"].edge_type,
            batch["node"].batch
        )
        
        # 计算基础loss（不带权重）
        loss = criterion(out.squeeze(), batch.y.float())
        
        # 如果使用加权，为每个样本分配对应类别的权重
        if class_weights is not None:
            # 为batch中每个样本分配权重
            weights = torch.tensor(
                [class_weights[label.item()] for label in batch.y],
                dtype=torch.float32,
                device=device
            )
            # 应用权重并求平均
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def eval_loss(model, criterion, loader, device):
    """验证/测试时不使用权重"""
    model.eval()
    total_loss = 0.
    
    for batch in loader:
        batch = batch.to(device)
        
        out = model(
            batch["node"].x,
            batch["node", "edge", "node"].edge_index,
            batch["node", "edge", "node"].edge_type,
            batch["node"].batch
        )
        
        # 不加权重
        loss = criterion(out.squeeze(), batch.y.float())
        total_loss += loss.mean().item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """评估模型性能"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    for batch in loader:
        batch = batch.to(device)
        
        out = model(
            batch["node"].x,
            batch["node", "edge", "node"].edge_index,
            batch["node", "edge", "node"].edge_type,
            batch["node"].batch
        )
        
        # 概率值
        prob = out.squeeze().cpu().numpy()
        y_prob.extend(prob)
        
        # 二值化预测
        pred = (out.squeeze() > threshold).long().cpu().numpy()
        y_pred.extend(pred)
        
        # 真实标签
        y_true.extend(batch.y.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'y_true': y_true,
        'y_pred': y_pred
    }


def main():
    args = get_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f'Train release: {args.train_release}, Valid release: {args.valid_release}, Test release: {args.test_release}')
    
    # 加载数据集
    train_dataset = load_dataset(args.project, args.train_release, args.data_root)
    valid_dataset = load_dataset(args.project, args.valid_release, args.data_root)
    test_dataset = load_dataset(args.project, args.test_release, args.data_root)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    if args.backbone.lower() == 'rgcn':
        model = MyRGCN(
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_relations=args.num_relations,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    elif args.backbone.lower() == 'gcn':
        model = MyGCN(
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    
    print(f'Model:\n{model}')
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer.lower() == 'adam' \
        else torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 计算类别权重
    class_weights = None
    if args.use_balanced_loss:
        labels = [data.y.item() for data in train_dataset]
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        class_weights = weights  # [weight_class0, weight_class1]
        
        print(f"\nClass weights (balanced):")
        print(f"  Class 0 (Normal): {class_weights[0]:.4f}")
        print(f"  Class 1 (Defect): {class_weights[1]:.4f}")
        print(f"  Weight ratio (Defect/Normal): {class_weights[1]/class_weights[0]:.4f}\n")
    
    # 损失函数（不带reduction，方便后续手动加权）
    train_criterion = nn.BCELoss(reduction='none')
    val_criterion = nn.BCELoss(reduction='none')
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor,
        patience=args.lr_patience, verbose=True, min_lr=args.lr_min
    )
    
    # 训练循环
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_epoch = 0
    
    pbar = tqdm(range(1, args.max_epochs + 1), desc="Training", ncols=120)
    for epoch in pbar:
        # 训练（使用加权）
        train_loss = train_one_epoch(model, optimizer, train_criterion, train_loader, device, class_weights)
        
        # 验证（不使用加权）
        val_loss = eval_loss(model, val_criterion, valid_loader, device)
        
        pbar.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'val_loss': f"{val_loss:.4f}",
        })
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{args.project}.pkl"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                pbar.close()
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
                break
    
    if epochs_no_improve < args.patience:
        print(f"Training finished! Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
    
    # 保存模型配置
    model_config = {
        'in_channels': args.in_channels,
        'hidden_channels': args.hidden_channels,
        "out_channels": args.out_channels,
        "num_relations": args.num_relations,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    with open(os.path.join(args.checkpoint_dir, f"{args.project}_{args.backbone}_config.json"), "w") as f:
        json.dump(model_config, f)
    
    # 加载最佳模型进行测试
    print("\nLoading best checkpoint for final test...")
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"{args.project}.pkl")))
    
    # 测试集评估
    test_results = evaluate(model, test_loader, device, threshold=args.threshold)
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Accuracy  : {test_results['accuracy']:.4f}")
    print(f"  F1-score  : {test_results['f1']:.4f}")
    print(f"  Precision : {test_results['precision']:.4f}")
    print(f"  Recall    : {test_results['recall']:.4f}")
    print(f"{'='*60}")
    
    # 打印混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_results['y_true'], test_results['y_pred'])
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Defect")
    print(f"Actual Normal   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Defect   {cm[1,0]:4d}    {cm[1,1]:4d}")


if __name__ == '__main__':
    main()