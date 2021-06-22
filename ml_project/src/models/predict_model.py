import json
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

from src.config.model_config import ModelParams


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Function predicts from given features. Combines transform and predict. """
    predicts = model.predict(features)
    return predicts


def model_score(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    """Function counting classification score. """
    return {
        'f1_score': f1_score(target, predicts),
        'roc_auc_score': roc_auc_score(target, predicts),
    }


def export_score(scores: dict, path: str) -> str:
    scores['time'] = time.ctime()
    with open(path, 'w') as fio:
        json.dump(scores, fio)
    return path
