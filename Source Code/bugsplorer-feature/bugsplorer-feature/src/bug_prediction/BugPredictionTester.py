import time
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.bug_prediction.BugPredictionArgs import BugPredictionArgs, model_class_of
from src.bug_prediction.DatasetManager import DatasetManager
from src.bug_prediction.ModelLoader import ModelLoader
from src.bug_prediction.Scorer import Score, Scorer

K = 1500


class IndicesValues(NamedTuple):
    indices: torch.Tensor
    values: torch.Tensor


class BugPredictionTester(ModelLoader):
    __slots__ = ('csv_path',)

    def __init__(self):
        args = BugPredictionArgs().parse_args()
        assert args.output_path is not None
        super().__init__(self.get_log_file_name(), args)

        self.csv_path = Path("results/cross_version_20251210.csv")
        self._init_csv_file()

        dataset_manager = DatasetManager(
            model_class_of[self.args.model_type].tokenizer,
            self.args.tokenizer_name,
            self.args.dataset_path,
            "line",
            ("test",),
            self.args.cache_dir,
            info=self.info,
            max_line_len=self.args.max_line_length,
            max_file_len=self.args.max_file_length,
        )

        (test_dataset,) = dataset_manager.load_dataset()
        self.test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=self.args.batch_size,
        )

        self._load_models(dataset_manager.tokenizer.pad_token_id, is_checkpoint=True)

    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'Project',
                    'Recall20LOC',
                    'Effort20Recall',
                    'AUC_line',
                    'IFA_line',
                    'IFA_file_level' 
                ]
                writer.writerow(header)

    def _write_to_csv(self, test_score, elapsed_seconds):
        """Write test results to CSV file"""
        output_name = Path(self.args.output_path).name
        project_name = output_name.split('-')[2]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                project_name,
                f"{test_score.recall_at_top20_percent_loc:.15f}",
                f"{test_score.effort_at_top20_percent_recall:.15f}",
                f"{test_score.roc_auc_score:.15f}",
                test_score.initial_false_alarm,
                f"{test_score.initial_false_alarm_file_level:.6f}"  # 新增
            ]
            writer.writerow(row)

    def test(self):
        start_time = time.perf_counter()
        self.info(f"Metrics: {Score.get_scores_names()}")
        
        if self.args.encoder_type == "file":
            true_prob, labels, loss = self._predict_from_file_classifier(
                self.test_dataloader,
            )
            np.savez(
                self.args.output_path,
                true_prob=true_prob,
                labels=labels,
            )
            test_score = Scorer.compute_score(
                Score.get_scores_names(),
                labels,
                true_prob,
                loss=loss,
            )
            self.info(f"Test Results:\n{test_score}")
        elif self.args.encoder_type == "line":
            self.info("Regular Evaluation:")
            test_score = self.evaluate(
                self.test_dataloader,
                Score.get_scores_names(),
                output_file=f"{self.args.output_path}-regular",
            )
            self.info(f"Test Results:\n{test_score}")
        else:
            raise ValueError(f"Unknown encoder type: {self.args.encoder_type}")

        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        self._write_to_csv(test_score, elapsed_seconds)

        self.info(f"Testing completed in {timedelta(seconds=elapsed_seconds)}")
        self.info(f"Results saved to {self.csv_path}")


def keep_k(s: IndicesValues, k: int):
    return IndicesValues(
        indices=s.indices[:k],
        values=s.values[:k],
    )


if __name__ == "__main__":
    BugPredictionTester().test()