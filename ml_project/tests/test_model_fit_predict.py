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
from src.models.train_model import train_model, create_model_pipeline, predict_model, model_score, \
    export_model
from src.data.make_dataset import split_test_train
from src.config.feature_config import FeatureParams
from src.config.model_config import ModelParams
from src.config.data_config import SplittingParams

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


@pytest.fixture
def raw_cat_data() -> pd.DataFrame:
    # Fake data columns and target names
    fake = Faker()
    colnames = fake.words(nb=TEST_DATA_SHAPE[1] + 1, unique=True)
    target_name = colnames.pop()
    # Prepare random numerical data
    num_data = np.random.randn(TEST_DATA_SHAPE[0], TEST_DATA_SHAPE[1] - TEST_DATA_CAT_COL_QTY)
    df = pd.DataFrame(num_data, columns=colnames[:TEST_DATA_SHAPE[1] - TEST_DATA_CAT_COL_QTY])
    # Create target with a simple regression
    target = pd.Series([0] * TEST_DATA_SHAPE[0])
    target.name = target_name
    for column in df.columns:
        target += np.random.randint(-50, 150) * df[column]
    # Prepare and add random categorical data
    cat_features = [fake.words(nb=TEST_DATA_SHAPE[0], ext_word_list=CATEGORIES) for _ in
                    range(TEST_DATA_CAT_COL_QTY)]
    cat_df = pd.DataFrame(cat_features)
    cat_df = cat_df.T
    cat_df.columns = colnames[TEST_DATA_SHAPE[1] - TEST_DATA_CAT_COL_QTY:]
    for column in cat_df.columns:
        target += np.random.randint(-5, 5) * cat_df[column].apply(CATEGORIES.index)
    df = df.join(cat_df)
    # Add target
    target_class = [0 if target_ < 0 else 1 for target_ in target]
    df = df.join(pd.Series(target_class, name=target_name))
    return df


def test_train_model_logreg(raw_num_data: pd.DataFrame):
    params = FeatureParams(numerical_features=[], target_col=raw_num_data.columns[-1])
    target, features = get_target(raw_num_data, params)
    model_params = ModelParams(model_type='LogisticRegression', random_state=3)
    model = train_model(features, target, model_params)
    assert type(model) == LogisticRegression, (
        f'Expected LogisticRegression type but got {type(model)}'
    )


def test_train_model_randomforest(raw_num_data: pd.DataFrame):
    params = FeatureParams(numerical_features=[], target_col=raw_num_data.columns[-1])
    target, features = get_target(raw_num_data, params)
    model_params = ModelParams(model_type='RandomForestClassifier', n_estimators=100)
    model = train_model(features, target, model_params)
    assert type(model) == RandomForestClassifier, (
        f'Expected RandomForestClassifier type but got {type(model)}'
    )
    assert model.n_estimators == model_params.n_estimators, (
        f'Expected {model_params.n_estimators} estimators but got {model.n_estimators}'
    )


def test_create_model_pipeline(raw_num_data: pd.DataFrame):
    params = FeatureParams(numerical_features=[], target_col=raw_num_data.columns[-1])
    target, features = get_target(raw_num_data, params)
    model_params = ModelParams(model_type='RandomForestClassifier', n_estimators=100)
    model = train_model(features, target, model_params)
    cat_columns = []
    num_columns = raw_num_data.columns[: -1]
    target_name = raw_num_data.columns[-1]
    params = FeatureParams(numerical_features=num_columns, target_col=target_name,
                           categorical_features=cat_columns)
    transformer = build_transformer(params)
    model_pipeline = create_model_pipeline(model, transformer)
    model_pipeline.fit(features, target)
    assert all(('feature_part', 'model_part' in model_pipeline.named_steps)), (
        f'Expected find feature_part and model_part, but found {model_pipeline.named_steps}'
    )


def test_predict_model(raw_num_data: pd.DataFrame):
    cat_columns = []
    num_columns = raw_num_data.columns[: -1]
    target_name = raw_num_data.columns[-1]
    feat_params = FeatureParams(numerical_features=num_columns, target_col=target_name,
                                categorical_features=cat_columns)
    split_params = SplittingParams(test_size=TEST_SIZE)
    model_params = ModelParams(model_type='RandomForestClassifier', n_estimators=100)
    train, test = split_test_train(raw_num_data, split_params)
    train_target, train_features = get_target(train, feat_params)
    val_target, val_features = get_target(test, feat_params)
    transformer = build_transformer(feat_params)
    transformer.fit(train_features)
    features = process_features(train_features, transformer, feat_params)
    model = train_model(features, train_target, model_params)
    model_pipeline = create_model_pipeline(model, transformer)
    val_features = process_features(val_features, transformer, feat_params)
    predicts = predict_model(model_pipeline, val_features)
    assert len(predicts) == len(val_target), (
        f'Expected {len(val_target)} predictions but got {len(predicts)}'
    )
    assert set(predicts) == set((0, 1)), (
        f'Expected only "0" and "1" but found {set(predicts)}'
    )


def test_evaluate_model(raw_num_data: pd.DataFrame):
    cat_columns = []
    num_columns = raw_num_data.columns[: -1]
    target_name = raw_num_data.columns[-1]
    feat_params = FeatureParams(numerical_features=num_columns, target_col=target_name,
                                categorical_features=cat_columns)
    split_params = SplittingParams(test_size=TEST_SIZE)
    model_params = ModelParams(model_type='RandomForestClassifier', n_estimators=100)
    train, test = split_test_train(raw_num_data, split_params)
    train_target, train_features = get_target(train, feat_params)
    val_target, val_features = get_target(test, feat_params)
    transformer = build_transformer(feat_params)
    transformer.fit(train_features)
    features = process_features(train_features, transformer, feat_params)
    model = train_model(features, train_target, model_params)
    model_pipeline = create_model_pipeline(model, transformer)
    val_features = process_features(val_features, transformer, feat_params)
    predicts = predict_model(model_pipeline, val_features)
    model_scores = model_score(predicts, val_target)
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


def test_export_model(raw_num_data: pd.DataFrame, tmpdir):
    params = FeatureParams(numerical_features=[], target_col=raw_num_data.columns[-1])
    target, features = get_target(raw_num_data, params)
    model_params = ModelParams(model_type='RandomForestClassifier', n_estimators=100)
    model = train_model(features, target, model_params)
    cat_columns = []
    num_columns = raw_num_data.columns[: -1]
    target_name = raw_num_data.columns[-1]
    params = FeatureParams(numerical_features=num_columns, target_col=target_name,
                           categorical_features=cat_columns)
    transformer = build_transformer(params)
    model_pipeline = create_model_pipeline(model, transformer)
    model_pipeline.fit(features, target)
    output_model_path = tmpdir.join('model.pkl')
    print(output_model_path)
    export_model(model_pipeline, output_model_path)
    with open(output_model_path, 'rb') as fio:
        model_from_file = joblib.load(fio)
    assert all(('feature_part', 'model_part' in model_from_file.named_steps)), (
        f'Expected find feature_part and model_part, but found {model_pipeline.named_steps}'
    )