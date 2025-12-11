from functools import cache
from typing import Iterable, NamedTuple
import inspect
import numpy as np
from sklearn import metrics


def recall_at_top20_percent_loc(y_true: np.ndarray, y_score: np.ndarray):
    twenty_percent = int(len(y_true) * 0.20)
    y_score_sorted_idx = np.argsort(y_score)[::-1]
    top_twenty_percent_index = y_score_sorted_idx[:twenty_percent]
    top_twenty_percent = y_true[top_twenty_percent_index]
    result = np.sum(top_twenty_percent) / np.sum(y_true)
    return result


def effort_at_top20_percent_recall(y_true: np.ndarray, y_score: np.ndarray):
    twenty_percent_defect = int(sum(y_true) * 0.20)
    y_score_sorted_idx = np.argsort(y_score)[::-1]

    defect_count = 0
    for i, j in enumerate(y_score_sorted_idx, 1):
        if y_true[j]:
            defect_count += 1
        if defect_count == twenty_percent_defect:
            return i / len(y_true)

    return 1


def initial_false_alarm(y_true: np.ndarray, y_score: np.ndarray):
    """
    LLIFA
    """
    sorted_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_idx]
    
    defect_positions = np.where(y_true_sorted == 1)[0]
    if len(defect_positions) == 0:
        return len(y_true_sorted)
    
    return int(defect_positions[0])


def initial_false_alarm_file_level(y_true: np.ndarray, y_score: np.ndarray, file_ids: np.ndarray):
    """
    FLIFA
    
    """
    unique_files = np.unique(file_ids)
    ifa_list = []
    
    for file_id in unique_files:
        file_mask = file_ids == file_id
        file_labels = y_true[file_mask]
        file_scores = y_score[file_mask]
        
        if file_labels.sum() == 0:
            continue

        sorted_idx = np.argsort(file_scores)[::-1]
        sorted_labels = file_labels[sorted_idx]

        defect_positions = np.where(sorted_labels == 1)[0]
        if len(defect_positions) > 0:
            ifa_list.append(defect_positions[0])

    if len(ifa_list) == 0:
        return 0.0
    
    return np.mean(ifa_list)


custom_metrics = {
    "recall_at_top20_percent_loc": recall_at_top20_percent_loc,
    "effort_at_top20_percent_recall": effort_at_top20_percent_recall,
    "initial_false_alarm": initial_false_alarm,
    "initial_false_alarm_file_level": initial_false_alarm_file_level,
}


class Score(NamedTuple):
    loss: float
    accuracy_score: float = -1.0
    balanced_accuracy_score: float = -1.0
    roc_auc_score: float = -1.0
    matthews_corrcoef: float = -1.0
    precision_score: float = -1.0
    recall_score: float = -1.0
    f1_score: float = -1.0
    recall_at_top20_percent_loc: float = -1.0
    effort_at_top20_percent_recall: float = -1.0
    initial_false_alarm: float = -1.0
    initial_false_alarm_file_level: float = -1.0 

    @staticmethod
    @cache
    def get_scores_names():
        return set(score for score in Score._fields if score != "loss")

    def __repr__(self):
        values = ",\n".join(
            f"    {score_name}={score_value}"
            for score_name, score_value in self._asdict().items()
        )
        return f"Score(\n{values}\n)"


class Scorer:
    @staticmethod
    def compute_score(
        metric_names: Iterable[str],
        labels: np.ndarray,
        true_prob: np.ndarray,
        loss=-1.0,
        file_ids: np.ndarray = None,
    ):
        threshold = Scorer.get_threshold_from_roc_curve(labels, true_prob)
        print("threshold:", threshold)
        predictions = true_prob > threshold

        scores = {"loss": loss}
        for score_name in metric_names:
            assert score_name in Score.get_scores_names(), score_name

            if hasattr(metrics, score_name):
                scorer = getattr(metrics, score_name)
            else:
                scorer = custom_metrics[score_name]

            args = {"y_true": labels}
            sig = inspect.signature(scorer)
            params = list(sig.parameters.keys())

            if "y_pred" in params:
                args["y_pred"] = predictions
            elif "y_score" in params:
                args["y_score"] = true_prob
            elif "y_prob" in params:
                args["y_prob"] = true_prob

            if score_name == "initial_false_alarm_file_level":
                if file_ids is None:
                    print(f"Warning: file_ids not provided for {score_name}, skipping")
                    scores[score_name] = -1.0
                    continue
                args["file_ids"] = file_ids

            try:
                score = scorer(**args)
                scores[score_name] = score
            except Exception as e:
                print(f"Error calculating {score_name}: {str(e)}")
                scores[score_name] = -1.0

        return Score(**scores)

    @staticmethod
    def get_threshold_from_roc_curve(labels, true_prob):
        """
        returns the threshold that maximizes balanced accuracy score
        """
        fpr, tpr, thresholds = metrics.roc_curve(labels, true_prob)
        tpr_fpr = tpr - fpr
        return thresholds[np.argmax(tpr_fpr)]