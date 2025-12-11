import random
from typing import Iterable

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.AbstractBaseLogger import AbstractBaseLogger
from src.bug_prediction.BugPredictionArgs import model_class_of, BugPredictionArgs
from src.bug_prediction.BugPredictionModel import BugPredictionModel
from src.bug_prediction.Scorer import Scorer, Score


class ModelLoader(AbstractBaseLogger):
    def __init__(self, name: str, args: BugPredictionArgs):
        super().__init__(name)
        self.args = args
        # --- Device ---
        if self.args.no_gpu:
            self.device = torch.device("cpu")
            print("Using CPU")
        else:
            assert torch.cuda.is_available(), "CUDA not available but no_gpu=False"
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        self.info(f"Args: {self.args}")
        self.info(f"{self.device=}")

        self._seed()

    def _load_models(self, pad_token_id: int, is_checkpoint=False):
        model_classes = model_class_of[self.args.model_type]
        self.config = model_classes.config.from_pretrained(
            self.args.config_name or self.args.model_name
        )
        self.info(
            f"Loading {self.args.model_type} model from {self.args.model_name} "
            f"{is_checkpoint=}",
        )

        self.model = BugPredictionModel(
            self.args.model_name,
            self.config,
            self.args.encoder_type,
            is_checkpoint,
            pad_token_id,
            self.args.model_type,
            self.args.max_line_length,
            self.args.max_file_length,
            class_weight=torch.tensor(
                [1, 1 if self.args.class_weight is None else self.args.class_weight],
                device=self.device,
                dtype=torch.float32,
            ),
        )

        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)
        self.model = self.model.to(self.device)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model_size = sum([np.prod(p.size()) for p in model_parameters])
        self.info(
            f"Finished loading model {self.args.model_type} "
            f"with size {int(model_size // 1e6)}M trainable params"
        )

    def evaluate(
        self,
        data_loader: DataLoader,
        metric_names: Iterable[str],
        output_file: str = None,
    ) -> Score:
        """
        Modified to handle handcrafted features and global indices for correct metric calculation.
        """
        self.info(f"Num examples: {len(data_loader.dataset)}")
        self.info(f"Num batches: {len(data_loader)}")
        self.info(f"Metrics: {metric_names}")

        total_loss = 0.0
        logit_batches, label_batches, global_idx_batches, file_id_batches = [], [], [], []
        self.model.eval()

        eval_progress = tqdm(
            data_loader, desc="Evaluating", position=1, smoothing=0.001
        )
        for _, batch in enumerate(eval_progress):
            if len(batch) == 5:
                inputs, labels, global_indices, file_ids, handcrafted = batch
            elif len(batch) == 4:
                inputs, labels, global_indices, file_ids = batch
                handcrafted = None
            elif len(batch) == 3:
                inputs, labels, global_indices = batch
                file_ids = None
                handcrafted = None
            else:
                inputs, labels = batch
                global_indices = None
                file_ids = None
                handcrafted = None

            with torch.no_grad():
                loss, logit = self.model(
                    inputs.to(self.device),
                    labels.float().to(self.device),
                    handcrafted_tensor=handcrafted.to(self.device) if handcrafted is not None else None
                )
                total_loss += loss.mean().item()
                logit_batches.append(logit.cpu().numpy())
                label_batches.append(labels.cpu().numpy())
                
                if global_indices is not None:
                    global_idx_batches.append(global_indices.cpu().numpy())
                if file_ids is not None:
                    file_id_batches.append(file_ids.cpu().numpy())

        logits = np.concatenate(logit_batches, axis=0)
        labels = np.concatenate(label_batches, axis=0)
        loss = total_loss / len(data_loader)

        if global_idx_batches:
            global_indices = np.concatenate(global_idx_batches, axis=0)
        else:
            global_indices = None
            
        if file_id_batches:
            file_ids = np.concatenate(file_id_batches, axis=0)
        else:
            file_ids = None

        labels_flat = labels.reshape(-1)
        if len(logits.shape) == 3:
            true_prob_flat = logits[:, :, 1].reshape(-1)
        elif len(logits.shape) == 2:
            true_prob_flat = logits.reshape(-1)
        else:
            raise ValueError(f"Invalid logits.shape={logits.shape}")

        if global_indices is not None:
            global_indices_flat = global_indices.reshape(-1)
            valid_mask = global_indices_flat >= 0
            labels_flat = labels_flat[valid_mask]
            true_prob_flat = true_prob_flat[valid_mask]
            global_indices_flat = global_indices_flat[valid_mask]
            
            if file_ids is not None:
                file_ids_flat = file_ids.reshape(-1)[valid_mask]
            else:
                file_ids_flat = None

            sort_indices = np.argsort(global_indices_flat)
            labels_flat = labels_flat[sort_indices]
            true_prob_flat = true_prob_flat[sort_indices]
            if file_ids_flat is not None:
                file_ids_flat = file_ids_flat[sort_indices]
            
            self.info(f"Reordered predictions by global indices. Valid samples: {len(labels_flat)}")
        else:
            file_ids_flat = None

        if output_file is not None:
            save_dict = {'true_prob': true_prob_flat, 'labels': labels_flat}
            if file_ids_flat is not None:
                save_dict['file_ids'] = file_ids_flat
            np.savez(output_file, **save_dict)

        return Scorer.compute_score(metric_names, labels_flat, true_prob_flat, loss, file_ids=file_ids_flat)

    def _predict_from_file_classifier(self, data_loader: DataLoader):
        """
        File-level prediction (if needed)
        """
        self.info(f"Num examples: {len(data_loader.dataset)}")
        self.info(f"Num batches: {len(data_loader)}")

        total_loss = 0.0
        logit_list, label_list = [], []
        self.model.eval()

        eval_progress = tqdm(
            data_loader, desc="Predicting (file-level)", position=1, smoothing=0.001
        )
        
        for batch in eval_progress:

            inputs, labels = batch[0], batch[1]

            with torch.no_grad():
                loss, logit = self.model(
                    inputs.to(self.device),
                    labels.float().to(self.device),
                )
                total_loss += loss.mean().item()
                logit_list.append(logit.cpu().numpy())
                label_list.append(labels.cpu().numpy())

        logits = np.concatenate(logit_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        loss = total_loss / len(data_loader)

        if len(logits.shape) == 2 and logits.shape[1] == 2:
            true_prob = logits[:, 1]
        else:
            true_prob = logits

        self.info(f"File-level prediction complete: {len(labels)} files")
        
        return true_prob, labels, loss

    def _seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)