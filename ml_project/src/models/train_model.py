from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
import joblib

from src.config.model_config import ModelParams

SklearnClfModel = Union[LogisticRegression, RandomForestClassifier]


def train_model(features: pd.DataFrame, target: pd.Series,
                model_params: ModelParams) -> SklearnClfModel:
    """Function builts model and train it. """
    if model_params.model_type == 'LogisticRegression':
        model = LogisticRegression(penalty='l1', solver='liblinear')
    elif model_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=model_params.n_estimators)
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def create_model_pipeline(
        model: SklearnClfModel, transformer: ColumnTransformer
) -> Pipeline:
    """Function creates pipeline from column transformer and model. """
    return Pipeline([('feature_part', transformer), ('model_part', model)])


def predict_model(
        model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
        predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "r2_score": r2_score(target, predicts),
        "rmse": mean_squared_error(target, predicts, squared=False),
        "mae": mean_absolute_error(target, predicts),
    }



def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
