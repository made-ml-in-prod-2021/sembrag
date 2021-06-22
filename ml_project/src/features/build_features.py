from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.config.feature_config import FeatureParams


def get_target(data: pd.DataFrame, params: FeatureParams) -> Tuple[pd.Series, pd.DataFrame]:
    """Function gets target from DataFrame and drops it from data. """
    target = data[params.target_col]
    data = data.drop(params.target_col, axis=1)
    return target, data


def build_num_pipeline(params: FeatureParams) -> Pipeline:
    """Numerical pipeline according to parameters. """
    transformers_list = [
        ('imputer', SimpleImputer(missing_values=np.nan, strategy=params.num_impute_strategy)),
    ]
    if params.scale_features:
        transformers_list.append(('scaler', StandardScaler()))
    num_pipeline = Pipeline(transformers_list)
    return num_pipeline


def build_cat_pipeline(params: FeatureParams) -> Pipeline:
    """Categorical pipeline according to parameters. """
    cat_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(missing_values=np.nan, strategy=params.cat_impute_strategy)),
            ('OHE', OneHotEncoder(sparse=False)),
        ]
    )
    return cat_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    """Transformer for all data. """
    transformers_list = [
        ('numerical', build_num_pipeline(params), params.numerical_features),
    ]
    if len(params.categorical_features) > 0:
        transformers_list.append(
            ('categorical', build_cat_pipeline(params), params.categorical_features))
    transformer = ColumnTransformer(transformers_list)
    return transformer


def process_features(data: pd.DataFrame, transformer: ColumnTransformer,
                     params: FeatureParams) -> pd.DataFrame:
    """Function that processes data with transformer. """
    col_names = data.columns
    if 'categorical' in transformer.named_transformers_:
        # Get names of created by OHE columns and replace names of initial categorial features
        ohe_columns = transformer.named_transformers_['categorical'].named_steps[
            'OHE'].get_feature_names()
        col_names = col_names.drop(params.categorical_features)
        col_names = col_names.to_list()
        col_names.extend(ohe_columns.tolist())
    try:
        transformed_data = pd.DataFrame(transformer.transform(data), columns=col_names)
    except:
        from pdb import set_trace as brk; brk()
    return transformed_data


def drop_features(data: pd.DataFrame, features_to_drop: List[str]) -> pd.DataFrame:
    """Function to drop features from DataFrame. """
    return data.drop(features_to_drop, axis=1)
