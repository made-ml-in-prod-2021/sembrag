from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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


def export_model(model: object, output: str) -> str:
    """Function exports model pipeline to serialized file. """
    with open(output, 'wb') as output_file:
        joblib.dump(model, output_file)
    return output
