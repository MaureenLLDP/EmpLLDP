import argparse
from asyncore import write
import copy
from platform import release
from numpy import append
import torch
import json
import os
import pandas as pd
from torch.utils.data import DataLoader
from model.bilstm import FusedBilstm
from model.cnn import TextCNN
from gensim.models import Word2Vec
from dataset import BPE, TokenVocab, BiLSTMDataset 
from train.train import FusedTrainer 

features = ['IfStatement', 'SimpleName', 'InfixExpression', 'Assignment', 'ExpressionStatement', 'Block',
            'MethodInvocation']

feature = 'MethodInvocation'

projects = ['activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket']
RQ_path = "../LSTM/RQ1/"
word2vec_path = "../LSTM/word2vec/"
# RQ_path = "./LSTM/RQ1/"
# word2vec_path = "./LSTM/word2vec/"
if not os.path.exists(RQ_path):
    os.makedirs(RQ_path)

all_releases = {
    'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
    'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
    'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
    'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
    'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'],
    'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
    'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
    'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
    'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']
}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-td", "--train_dataset", type=str, help="train set")
    parser.add_argument("-vd", "--valid_dataset", type=str, default=None, help="validation set")
    parser.add_argument("-v", "--vocab_path", type=str, help="vocab path")
    parser.add_argument("-pm", "--pretrain_model", type=str, help="vocab path")
    parser.add_argument("-o", "--output_path", type=str, help="model save path")
    parser.add_argument("-fs", "--feed_forward_hidden", type=int, default=3072,
                        help="hidden size of feed-forward network")
    parser.add_argument("-aj", "--attention_json", type=str, help="attention_json")
    parser.add_argument("-hs", "--hidden", type=int, default=768, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=12, help="number of transformer layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=12, help="number of attention heads")
    parser.add_argument("-p", "--path_num", type=int, default=100, help="a AST's maximum path num")
    parser.add_argument("-n", "--node_num", type=int, default=20, help="a path's maximum node num")
    parser.add_argument("-c", "--code_len", type=int, default=200, help="maximum code len")
    parser.add_argument("-al", "--alpha", type=float, default=0.9, help="loss weight")
    parser.add_argument("-b", "--batch_size", type=int, default=2048, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker num")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--use_fusion", type=bool, default=True, help="whether to use feature fusion")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab_path = "../LSTM/vocaburary/vocab"
    #vocab_path = "./LSTM/vocaburary/vocab"

    AUC = []
    recall_20_LOC = []
    effort_20_recall = []
    IFA = []

    project_summary_list = []

    data_path = "../Datasets/preprocessed_data3/"
    #data_path = "./Datasets/preprocessed_data3/"

    for project in projects:

        word2vec_file_dir = os.path.join(word2vec_path, project + '-' + str(500) + 'dim.bin')
        word2vec = Word2Vec.load(word2vec_file_dir)

        files = all_releases[project]
        train_files = [files[0]] 
        valid_files = [files[1]]  
        test_files = files[2:] 
        
        scaler_save_path = os.path.join(RQ_path, f"{project}_scaler.pkl")

        train_dataset = BiLSTMDataset(
            word2vec, data_path, train_files, 50, 
            Train=True, use_manual_features=args.use_fusion,  # 训练时也使用人工特征
            scaler_save_path=scaler_save_path
        )
        train_data_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True, shuffle=True
        )

        valid_dataset = BiLSTMDataset(
            word2vec, data_path, valid_files, 50, 
            Train=False, use_manual_features=args.use_fusion,  # 验证时也使用人工特征
            scaler_save_path=scaler_save_path
        )
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True
        )

        test_dataset = BiLSTMDataset(
            word2vec, data_path, test_files, 50, 
            Train=False, use_manual_features=args.use_fusion,  # 根据融合参数决定是否加载人工特征
            scaler_save_path=scaler_save_path
        )
        test_data_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True
        )


        cuda_condition = torch.cuda.is_available() and args.with_cuda
        device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        Model = FusedBilstm(
            embedsize=500, 
            hid_size=64, 
            num_layers=2, 
            device=device,
            manual_feature_dim=32,
            use_fusion=args.use_fusion 
        )


        print("Creating Trainer")
        trainer = FusedTrainer(
            Model, args.alpha, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
            device=device, log_freq=args.log_freq
        )

        print("Training Start")
        with open(RQ_path + "fused_lstm_result.txt", "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"项目: {project}\n")
            f.write(f"融合模式: {'启用' if args.use_fusion else '禁用'}\n")
            f.write(f"{'='*50}\n")

        best_valid_auc = 0.0
        patience = 5
        counter = 0
        best_model_path = RQ_path + f"{project}_best_fused_model"
        best_epoch = 0

        original_test_data = trainer.test_data

        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

            train_loss = trainer.train(epoch)

            trainer.test_data = valid_data_loader
            valid_result = trainer.test(epoch)
            
            trainer.test_data = original_test_data

            current_valid_auc = valid_result['auc']

            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Valid AUC: {current_valid_auc:.4f}")

            with open(RQ_path + "fused_lstm_result.txt", "a") as f:
                f.write(
                    f"Epoch {epoch + 1:2d} - "
                    f"TrainLoss: {train_loss:.4f} - "
                    f"ValidAUC: {valid_result['auc']:.4f} - "
                    f"ValidPrecision: {valid_result['precision']:.4f} - "
                    f"ValidRecall: {valid_result['recall']:.4f} - "
                    f"ValidF1: {valid_result['f1']:.4f}\n"
                )

            if current_valid_auc > best_valid_auc:
                best_valid_auc = current_valid_auc
                best_epoch = epoch
                counter = 0
                trainer.save(epoch, best_model_path)
            else:
                counter += 1
                break


        best_model_file = best_model_path + f".ep{best_epoch}"
        trainer.model = torch.load(best_model_file, map_location=trainer.device)
        trainer.test_data = test_data_loader
        
        final_result = trainer.test(best_epoch)

        print(f"  AUC: {final_result['auc']:.4f}")
        print(f"  Precision: {final_result['precision']:.4f}")
        print(f"  Recall: {final_result['recall']:.4f}")
        print(f"  F1: {final_result['f1']:.4f}")
        print(f"  Recall@20%LOC: {final_result['recall@20%LOC']:.4f}")
        print(f"  Effort@20%Recall: {final_result['effort@20%Recall']:.4f}")
        print(f"  IFA: {final_result['IFA_line']:.4f}")

        AUC.append(final_result["auc"])
        recall_20_LOC.append(final_result["recall@20%LOC"])
        effort_20_recall.append(final_result["effort@20%Recall"])
        IFA.append(final_result["IFA_line"])

        project_summary_list.append({
            'Project': project,
            'AUC': final_result["auc"],
            'Precision': final_result["precision"],
            'Recall': final_result["recall"],
            'F1': final_result["f1"],
            'Balanced_Acc': final_result["balance_acc"],
            'MCC': final_result["mcc"],
            'Recall@20%LOC': final_result["recall@20%LOC"],
            'Effort@20%Recall': final_result["effort@20%Recall"],
            'IFA': final_result["IFA_line"]
        })

        with open(RQ_path + "fused_lstm_result.txt", "a") as f:
            f.write(f"FINAL_TEST - ")
            f.write(f"AUC: {final_result['auc']:.4f} - ")
            f.write(f"Precision: {final_result['precision']:.4f} - ")
            f.write(f"Recall: {final_result['recall']:.4f} - ")
            f.write(f"F1: {final_result['f1']:.4f} - ")
            f.write(f"BalancedAcc: {final_result['balance_acc']:.4f} - ")
            f.write(f"MCC: {final_result['mcc']:.4f} - ")
            f.write(f"Recall@20%LOC: {final_result['recall@20%LOC']:.4f} - ")
            f.write(f"Effort@20%Recall: {final_result['effort@20%Recall']:.4f} - ")
            f.write(f"IFA: {final_result['IFA_line']:.4f}\n\n")

        import glob
        model_pattern = best_model_path + ".ep*"
        all_model_files = glob.glob(model_pattern)
        best_model_file = best_model_path + f".ep{best_epoch}"

        for model_file in all_model_files:
            if model_file != best_model_file:
                try:
                    os.remove(model_file)

                except OSError as e:
                    print(f"fail {model_file}: {e}")

        print(f"best: {best_model_file}")

    import numpy as np
    fusion_type = "融合特征" if args.use_fusion else "深度特征"
    
    with open(RQ_path + "fused_summary_result.txt", "w") as f_sum:
        f_sum.write(f"====== {fusion_type}结果汇总 ======\n")
        f_sum.write(f"项目总数: {len(projects)}\n")
        f_sum.write(f"平均AUC: {np.mean(AUC):.4f} ± {np.std(AUC):.4f}\n")
        f_sum.write(f"平均Recall@20%LOC: {np.mean(recall_20_LOC):.4f} ± {np.std(recall_20_LOC):.4f}\n")
        f_sum.write(f"平均Effort@20%Recall: {np.mean(effort_20_recall):.4f} ± {np.std(effort_20_recall):.4f}\n")
        f_sum.write(f"平均IFA: {np.mean(IFA):.4f} ± {np.std(IFA):.4f}\n")
        f_sum.write(f"\n最佳结果:\n")
        f_sum.write(f"最佳AUC: {np.max(AUC):.4f}\n")
        f_sum.write(f"最佳Recall@20%LOC: {np.max(recall_20_LOC):.4f}\n")
        f_sum.write(f"最佳Effort@20%Recall: {np.min(effort_20_recall):.4f}\n")
        f_sum.write(f"最佳IFA: {np.min(IFA):.4f}\n")

    summary_df = pd.DataFrame(project_summary_list)
    summary_csv_path = os.path.join(RQ_path, "fused_project_summary_metrics.csv")
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    
if __name__ == "__main__":
    # try:
    train()
    # finally:
        # os.system("shutdown -h now")