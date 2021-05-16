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
    data.drop(params.target_col)
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
            ('OHE', OneHotEncoder()),
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
            ('categorial', build_cat_pipeline(params), params.categorical_features))
    transformer = ColumnTransformer(transformers_list)


def process_features(data: pd.DataFrame, transformer: ColumnTransformer,
                     params: FeatureParams) -> pd.DataFrame:
    """Function that processes data with transformer. """
    transformed_data = pd.DataFrame(transformer.transform())
    return transformed_data
