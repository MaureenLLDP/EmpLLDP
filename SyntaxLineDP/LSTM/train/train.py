from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, matthews_corrcoef, balanced_accuracy_score
import os

class FusedTrainer:
    def __init__(self, model, alpha, train_dataloader: DataLoader,
                 test_dataloader: DataLoader = None, lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 warmup_steps=10000, device=None, log_freq: int = 10, predict_threshold: float = 0.5):
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        
        self.weight_dict = {}
        self._compute_class_weights()
        
        self.criterion = nn.BCELoss()
        
        self.log_freq = log_freq
        self.predict_threshold = predict_threshold

        print(f"Using device: {self.device}")
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.all_predictions_prob_for_csv = []
        self.all_true_labels_for_csv = []
        self.all_filenames_for_csv = []
        self.all_line_numbers_for_csv = []

    def _compute_class_weights(self):
        if self.train_data is None:
            self.weight_dict = {'clean': 1.0, 'defect': 1.0}
            return
        
        all_labels = []
        for batch in self.train_data:
            labels = batch["is_defect"].cpu().numpy()
            all_labels.extend(labels)
        
        sample_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(all_labels), 
            y=all_labels
        )

        self.weight_dict['defect'] = np.max(sample_weights)
        self.weight_dict['clean'] = np.min(sample_weights)
        
        print(f"Class weights - Clean: {self.weight_dict['clean']:.3f}, Defect: {self.weight_dict['defect']:.3f}")

    def get_loss_weight(self, labels):
        label_list = labels.cpu().numpy().squeeze().tolist()
        weight_list = []
        
        for lab in label_list:
            if lab == 0:
                weight_list.append(self.weight_dict['clean'])
            else:
                weight_list.append(self.weight_dict['defect'])
        
        weight_tensor = torch.tensor(weight_list, dtype=torch.float32).to(self.device)
        return weight_tensor

    def train(self, epoch):
        self.model.train()
        avg_loss, metrics = self.iteration(epoch, self.train_data, train=True)
        
        print(f"\n--- Epoch {epoch} Training Metrics ---")
        print(f"  Avg Loss: {avg_loss:.5f}")
        print(f"  TP: {metrics['TP']:.0f} FP: {metrics['FP']:.0f} FN: {metrics['FN']:.0f} TN: {metrics['TN']:.0f}")
        print(f"  Precision: {metrics['precision']:.4f} Recall: {metrics['recall']:.4f} F1: {metrics['f1']:.4f}")
        print(f"  Balanced Acc: {metrics['balance_acc']:.4f} AUC: {metrics['auc']:.4f} MCC: {metrics['mcc']:.4f}")
        
        return avg_loss

    def test(self, epoch):
        self.model.eval()
        
        self.all_predictions_prob_for_csv = []
        self.all_true_labels_for_csv = []
        self.all_filenames_for_csv = []
        self.all_line_numbers_for_csv = []

        with torch.no_grad():
            avg_loss, metrics = self.iteration(epoch, self.test_data, train=False)
        
        print(f"\n--- Epoch {epoch} Test Metrics ---")
        print(f"  Avg Loss: {avg_loss:.5f}")
        print(f"  TP: {metrics['TP']:.0f} FP: {metrics['FP']:.0f} FN: {metrics['FN']:.0f} TN: {metrics['TN']:.0f}")
        print(f"  Precision: {metrics['precision']:.4f} Recall: {metrics['recall']:.4f} F1: {metrics['f1']:.4f}")
        print(f"  Balanced Acc: {metrics['balance_acc']:.4f} AUC: {metrics['auc']:.4f} MCC: {metrics['mcc']:.4f}")
        print(f"  Recall@20%LOC: {metrics['recall@20%LOC']:.4f} Effort@20%Recall: {metrics['effort@20%Recall']:.4f} IFA: {metrics['IFA_line']:.4f}")
        
        self._save_predictions_to_csv(epoch)
        
        return metrics

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code, epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        
        total_TP = 0.0
        total_FN = 0.0
        total_FP = 0.0
        total_TN = 0.0
        
        all_line_labels_epoch = []
        all_line_preds_prob_epoch = []
        all_predict_labels_binary_epoch = []

        if not train:
            filenames_for_domain_metrics = []
            file_labels_for_domain_metrics = []
            line_labels_for_domain_metrics = []
            line_preds_prob_for_domain_metrics = []

        for i, data in data_iter:
            if i == 0:  
                print(f"\n--- {str_code} Data batch {i} ---")
                print(f"ast_input batch shape: {data['ast_input'].shape}")
                print(f"is_defect batch shape: {data['is_defect'].shape}")
                if "line_number" in data: 
                    print(f"filename: {data['filename'][0]}, line_number: {data['line_number'][0]}")
                if "manual_features" in data:
                    print(f"manual_features batch shape: {data['manual_features'].shape}")
               
            torch.cuda.empty_cache()
            
            ast_input_on_device = data["ast_input"].to(self.device)
            is_defect_on_device = data["is_defect"].to(self.device)
             
            targets = self.model.forward(ast_input_on_device)
          
            if train:
                weight_tensor = self.get_loss_weight(is_defect_on_device)
                self.criterion.weight = weight_tensor
            else:
                self.criterion.weight = None
            
            loss = self.criterion(targets.squeeze(dim=-1), is_defect_on_device.float())

            if train:
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()

            current_batch_predict_probs = targets.squeeze(dim=-1).cpu().detach().numpy().tolist()
            current_batch_true_labels = is_defect_on_device.cpu().numpy().tolist()
            current_batch_binary_preds = [(1 if p > self.predict_threshold else 0) for p in current_batch_predict_probs]

            all_line_labels_epoch.extend(current_batch_true_labels)
            all_line_preds_prob_epoch.extend(current_batch_predict_probs)
            all_predict_labels_binary_epoch.extend(current_batch_binary_preds)
            
            for label, predict in zip(current_batch_true_labels, current_batch_binary_preds):
                if label == 1 and predict == 1:
                    total_TP += 1
                elif label == 0 and predict == 1:
                    total_FP += 1
                elif label == 1 and predict == 0:
                    total_FN += 1
                elif label == 0 and predict == 0:
                    total_TN += 1

            avg_loss += loss.item()
            
            if not train:
                self.all_filenames_for_csv.extend(data["filename"])
                self.all_line_numbers_for_csv.extend(data["line_number"].cpu().numpy().tolist())
                self.all_true_labels_for_csv.extend(current_batch_true_labels)
                self.all_predictions_prob_for_csv.extend(current_batch_predict_probs)

                filenames_for_domain_metrics.extend(data["filename"])
                file_labels_for_domain_metrics.extend(data["file_label"].cpu().numpy().tolist())
                line_labels_for_domain_metrics.extend(current_batch_true_labels)
                line_preds_prob_for_domain_metrics.extend(current_batch_predict_probs)

        avg_loss /= len(data_loader)

        metrics = {}
        metrics['TP'] = total_TP
        metrics['FP'] = total_FP
        metrics['FN'] = total_FN
        metrics['TN'] = total_TN

        metrics['precision'] = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        metrics['recall'] = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        metrics['f1'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0

        if len(set(all_line_labels_epoch)) > 1: 
            metrics['balance_acc'] = balanced_accuracy_score(all_line_labels_epoch, all_predict_labels_binary_epoch)
            try: 
                metrics['auc'] = roc_auc_score(all_line_labels_epoch, all_line_preds_prob_epoch)
            except ValueError as e: 
                print(f"AUC calculation error: {e}")
                metrics['auc'] = 0.0
            
            try:
                metrics['mcc'] = matthews_corrcoef(all_line_labels_epoch, all_predict_labels_binary_epoch)
            except Exception: 
                metrics['mcc'] = 0.0
        else: 
            metrics['balance_acc'] = 0.0
            metrics['auc'] = 0.0
            metrics['mcc'] = 0.0

        if not train:
            df_for_domain_metrics = pd.DataFrame({
                "filename": filenames_for_domain_metrics,
                "file_label": file_labels_for_domain_metrics,
                "line_label": line_labels_for_domain_metrics,
                "line_pred": line_preds_prob_for_domain_metrics,
            })
            
            df_sorted = df_for_domain_metrics.sort_values(by='line_pred', ascending=False).reset_index(drop=True)
            sorted_line_label = list(df_sorted["line_label"])
            metrics = self._compute_domain_metrics(metrics, sorted_line_label, len(sorted_line_label))
            
            metrics['IFA_line'] = self._compute_ifa(df_sorted)
            
        else: 
            metrics['recall@20%LOC'] = 0.0
            metrics['effort@20%Recall'] = 0.0
            metrics['IFA_line'] = 0.0

        return avg_loss, metrics

    def _compute_domain_metrics(self, metrics, sorted_line_label, num_total_lines, prefix=''):
        # Recall@20%LOC
        num_top_lines = int(num_total_lines * 0.2)
        if num_total_lines > 0:
            defects_in_top_20 = sum(sorted_line_label[:num_top_lines])
            total_defects = sum(sorted_line_label)
            metrics[prefix + 'recall@20%LOC'] = defects_in_top_20 / total_defects if total_defects > 0 else 0.0
        else:
            metrics[prefix + 'recall@20%LOC'] = 0.0

        # Effort@20%Recall
        total_defect = sum(sorted_line_label)
        if total_defect > 0:
            top20_defect_count = int(total_defect * 0.2)
            TP_effort = 0
            count_lines_to_reach_20_recall = 0
            for i in range(len(sorted_line_label)):
                if sorted_line_label[i] == 1:
                    TP_effort += 1
                if TP_effort >= top20_defect_count:
                    count_lines_to_reach_20_recall = i + 1
                    break
            metrics[prefix + 'effort@20%Recall'] = count_lines_to_reach_20_recall / num_total_lines if num_total_lines > 0 else 0.0
        else:
            metrics[prefix + 'effort@20%Recall'] = 0.0

        return metrics

    def _compute_ifa(self, df_sorted):
        IFA_values = []  
        df_buggy_files = df_sorted[df_sorted['file_label'] == True].copy()
        if not df_buggy_files.empty:
            for file, df_file in df_buggy_files.groupby("filename"):
                labels = list(df_file['line_label'])
                for index, label in enumerate(labels):
                    if label == 1:
                        IFA_values.append(index + 1)
                        break
            return np.mean(IFA_values) if IFA_values else 0.0
        else:
            return 0.0

    def _save_predictions_to_csv(self, epoch):
        output_dir = "prediction_results"
        os.makedirs(output_dir, exist_ok=True)

        if not (len(self.all_predictions_prob_for_csv) == len(self.all_true_labels_for_csv) == 
                len(self.all_filenames_for_csv) == len(self.all_line_numbers_for_csv)):
            print(f"Warning: Data collection lists have inconsistent lengths in epoch {epoch}. Skipping CSV save.")
            return
        
        predicted_classes_for_csv = (np.array(self.all_predictions_prob_for_csv) >= self.predict_threshold).astype(int)
        
        data_to_save = {
            'filename': self.all_filenames_for_csv,
            'line_number': self.all_line_numbers_for_csv,
            'prediction_probability': self.all_predictions_prob_for_csv,
            'true_label': self.all_true_labels_for_csv,
            'predicted_class': predicted_classes_for_csv
        }
        df = pd.DataFrame(data_to_save)
        
        output_filename = os.path.join(output_dir, f"epoch_{epoch}_predictions.csv")
        df.to_csv(output_filename, index=False)
        print(f"Predictions saved to {output_filename}")

    def save(self, epoch, file_path="output/trained.model"):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model, output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


    def load_manual_features(self, manual_features_path, test_files):
        pass