import json

import joblib
import pytest
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.features.build_features import build_cat_pipeline, build_num_pipeline, build_transformer, \
    get_target, process_features, drop_features
from src.models.train_model import train_model, create_model_pipeline, export_model
from src.models.predict_model import predict_model, model_score
from src.data.make_dataset import split_test_train
from src.full_model_pipeline import run_model_pipeline
from src.config.config import TrainingPipelineParams
from src.config.feature_config import FeatureParams
from src.config.model_config import ModelParams
from src.config.data_config import SplittingParams, DataFileParams

TEST_DATA_SHAPE = (1000, 6)
TEST_DATA_CAT_COL_QTY = 4
CATEGORIES = ['abc', 'def', 'ghj', 'klm', 'nop']
CSV_SEP = ';'
TEST_SIZE = 0.25


@pytest.fixture
def raw_num_data() -> pd.DataFrame:
    # Fake data columns and target names
    fake = Faker()
    colnames = fake.words(nb=TEST_DATA_SHAPE[1] + 1, unique=True)
    target_name = colnames.pop()
    # Prepare random data
    random_data = np.random.randn(*TEST_DATA_SHAPE)
    df = pd.DataFrame(random_data, columns=colnames)
    # Create target with a simple regression
    target = pd.Series([0] * TEST_DATA_SHAPE[0])
    target.name = target_name
    for column in df.columns:
        target += np.random.randint(-50, 150) * df[column]
    target_class = [0 if target_ < 0 else 1 for target_ in target]
    df = df.join(pd.Series(target_class, name=target_name))
    return df


def test_full_model_pipeline(raw_num_data: pd.DataFrame, tmpdir: pytest.fixture):
    cat_columns = []
    num_columns = raw_num_data.columns[: -1]
    target_name = raw_num_data.columns[-1]
    data_path = tmpdir.join('test_data.csv')
    raw_num_data.to_csv(data_path, sep=',', index=False)
    data_params = DataFileParams(path=data_path, sep=',')
    feat_params = FeatureParams(numerical_features=num_columns, target_col=target_name,
                                categorical_features=cat_columns)
    split_params = SplittingParams(test_size=TEST_SIZE)
    model_params = ModelParams(model_type='RandomForestClassifier', n_estimators=100)
    output_model = tmpdir.join('model.pkl')
    output_score = tmpdir.join('score.json')
    params = TrainingPipelineParams(
        data_file_params=data_params,
        split_params=split_params,
        feature_params=feat_params,
        model_params=model_params,
        output_model_path=output_model,
        metric_path=output_score
    )
    run_model_pipeline(params)
    with open(output_score, 'r') as fio:
        model_scores = json.load(fio)
    with open(output_model, 'rb') as fio:
        model_from_file = joblib.load(fio)
    assert all(('feature_part', 'model_part' in model_from_file.named_steps)), (
        f'Expected find feature_part and model_part, but found {model_pipeline.named_steps}'
    )
    assert 'f1_score' in model_scores, (
        f'Expected f1_score predictions but not found '
    )
    assert 'roc_auc_score' in model_scores, (
        f'Expected roc_auc_score predictions but not found '
    )
    assert 0 < model_scores['f1_score'] <= 1, (
        f'Expected value between "0" and "1" but found {model_scores["f1_score"]}'
    )
    assert 0 < model_scores['roc_auc_score'] <= 1, (
        f'Expected value between "0" and "1" but found {model_scores["roc_auc_score"]}'
    )