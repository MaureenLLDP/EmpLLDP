import os
import numpy as np
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from Scorer import Scorer, Score

def load_npz_file(filepath: str) -> dict:
    data = np.load(filepath)
    filename = os.path.basename(filepath)
    project_name = filename.split('-')[2]
    
    return {
        'true_prob': data['true_prob'],
        'labels': data['labels'],
        'filename': os.path.basename(filepath),
        'project': project_name
    }

def evaluate_single_file(data: dict) -> dict:
    score = Scorer.compute_score(
        Score.get_scores_names(),
        data['labels'],
        data['true_prob']
    )
    return {
        'filename': data['filename'],
        'project': data['project'],
        **score._asdict()
    }

def process_directory(input_dir: str) -> List[dict]:
    results = []
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    for filename in tqdm(npz_files, desc="Processing files"):
        filepath = os.path.join(input_dir, filename)
        data = load_npz_file(filepath)
        results.append(evaluate_single_file(data))
    
    return results

def save_results(results: List[dict], output_csv: str):
    df = pd.DataFrame(results)
    
    column_mapping = {
        'project': 'Project',
        'roc_auc_score': 'AUC',
        'balanced_accuracy_score': 'BA',
        'recall_at_top20_percent_loc': 'Recall20LOC',
        'effort_at_top20_percent_recall': 'Effort20Recall',
        'ifa_absolute': 'IFA',
        'initial_false_alarm': 'IFA%',
        'accuracy_score': 'ACC'
    }
    
    df = df[column_mapping.keys()].rename(columns=column_mapping)
    df.to_csv(output_csv, index=False, float_format='%.15f')

if __name__ == "__main__":
    
    input_dir = "/root/autodl-tmp/outputs/cross_project"
    output_csv = "/root/autodl-tmp/results/cross_project.csv"

    all_results = process_directory(input_dir)
    save_results(all_results, output_csv)