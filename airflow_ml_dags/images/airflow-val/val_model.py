import os
import json
from typing import Tuple, Dict

import joblib
import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score


def download_val_data(path_to_data: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(os.path.join(path_to_data, 'val_data.csv'))
    target = pd.read_csv(os.path.join(path_to_data, 'val_target.csv'))
    return data, target


def model_score(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    """Function counting classification score. """
    return {
        'f1_score': f1_score(target, predicts),
        'roc_auc_score': roc_auc_score(target, predicts),
    }


def export_score(scores: dict, path: str) -> str:
    with open(path, 'w') as fio:
        json.dump(scores, fio)
    return path


@click.command('val')
@click.option('--input-dir')
@click.option('--model-dir')
@click.option('--score-dir')
def val_model(input_dir: str, model_dir: str, score_dir: str):
    """Function builts model and train it. """
    data, target = download_val_data(input_dir)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as fio:
        model = joblib.load(fio)
    predicts = model.predict(data)
    score = model_score(predicts, target)
    os.makedirs(score_dir, exist_ok=True)
    export_score(score, os.path.join(score_dir, 'model_score.json'))


if __name__ == '__main__':
    val_model()
