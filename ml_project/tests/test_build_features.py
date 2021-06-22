import pytest
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.features.build_features import build_cat_pipeline, build_num_pipeline, build_transformer, \
    get_target, process_features, drop_features
from src.config.feature_config import FeatureParams


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
    df = df.join(target)
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
    df = df.join(cat_df)
    # Add target
    df = df.join(target)
    return df


def test_get_target(raw_num_data: pd.DataFrame):
    raw_data = raw_num_data.copy()
    target_name = raw_data.columns[-1]
    target_ = raw_data[target_name]
    data_ = raw_data.drop(target_name, axis=1)
    params = FeatureParams(numerical_features=[], target_col=target_name)
    target, data = get_target(raw_num_data, params)
    assert target_.shape == target.shape, (
        f'Expected {target_.shape} shape of target, but got {target.shape}'
    )
    assert data_.shape == data.shape, (
        f'Expected {data_.shape} shape of target, but got {data.shape}'
    )
    assert target_name == target.name, (
        f'Expected name {target_name}, but got {target.name}'
    )
    assert target_name not in data.columns, (
        'Found target in data.'
    )


def test_build_num_pipeline_without_scaling(raw_num_data: pd.DataFrame):
    col_names = list(raw_num_data.columns)
    target_name = col_names.pop()
    params = FeatureParams(numerical_features=col_names, target_col=target_name,
                           num_impute_strategy='mean', scale_features=False)
    pipeline = build_num_pipeline(params)
    assert type(pipeline) == Pipeline, (
        f'Expected type Pipeline type but found {type(pipeline)}'
    )
    assert 'imputer' in pipeline.named_steps, (
        f'Expected to find "imputer" in steps of Pipeline but did not succeed'
    )
    assert 'scaler' not in pipeline.named_steps, (
        f'Found "scaler" in steps of Pipeline but should not'
    )
    assert pipeline['imputer'].strategy == 'mean', (
        f'Expected "mean" as a strategy for imputer but found {pipeline["imputer"].strategy}'
    )


def test_build_num_pipeline_with_scaling(raw_num_data: pd.DataFrame):
    col_names = list(raw_num_data.columns)
    target_name = col_names.pop()
    params = FeatureParams(numerical_features=col_names, target_col=target_name,
                           scale_features=True)
    pipeline = build_num_pipeline(params)
    assert type(pipeline) == Pipeline, (
        f'Expected type Pipeline type but found {type(pipeline)}'
    )
    assert 'imputer' in pipeline.named_steps, (
        f'Expected to find "imputer" in steps of Pipeline but did not succeed'
    )
    assert 'scaler' in pipeline.named_steps, (
        f'Expected to find "scaler" in steps of Pipeline but did not succeed'
    )
    assert pipeline['imputer'].strategy == 'median', (
        f'Expected "mean" as a strategy for imputer but found {pipeline["imputer"].strategy}'
    )


def test_build_cat_pipeline(raw_cat_data: pd.DataFrame):
    cat_columns = list(raw_cat_data.columns)
    start_cat_col = TEST_DATA_SHAPE[1] - TEST_DATA_CAT_COL_QTY
    num_columns = cat_columns[:start_cat_col]
    cat_columns = cat_columns[start_cat_col: -1]
    target_name = cat_columns[-1]
    params = FeatureParams(numerical_features=num_columns, target_col=target_name,
                           categorical_features=cat_columns)
    pipeline = build_cat_pipeline(params)
    assert type(pipeline) == Pipeline, (
        f'Expected type Pipeline type but found {type(pipeline)}'
    )
    assert 'imputer' in pipeline.named_steps, (
        f'Expected to find "imputer" in steps of Pipeline but did not succeed'
    )
    assert 'OHE' in pipeline.named_steps, (
        f'Expected to find "OHE" in steps of Pipeline but did not succeed'
    )
    assert pipeline['imputer'].strategy == 'most_frequent', (
        f'Expected "mean" as a strategy for imputer but found {pipeline["imputer"].strategy}'
    )


def test_build_transformer_only_numerical(raw_cat_data: pd.DataFrame):
    cat_columns = list(raw_cat_data.columns)
    start_cat_col = TEST_DATA_SHAPE[1] - TEST_DATA_CAT_COL_QTY
    num_columns = cat_columns[:start_cat_col]
    cat_columns = cat_columns[start_cat_col: -1]
    target_name = cat_columns[-1]
    params = FeatureParams(numerical_features=num_columns, target_col=target_name,
                           categorical_features=cat_columns)
    transformer = build_transformer(params)
    trafos = transformer.transformers
    assert type(transformer) == ColumnTransformer, {
        f'Expected ColumnTransformer type but got {type(transformer)}'
    }
    assert trafos[0][0] == 'numerical', (
        f'Expected "numerical" as a first transformer but found {trafos[0][0]}'
    )
    assert trafos[1][0] == 'categorical', (
        f'Expected "categorical" as a first transformer but found {trafos[1][0]}'
    )


def test_process_features(raw_cat_data: pd.DataFrame):
    all_columns = list(raw_cat_data.columns)
    start_cat_col = TEST_DATA_SHAPE[1] - TEST_DATA_CAT_COL_QTY
    num_columns = all_columns[:start_cat_col]
    cat_columns = all_columns[start_cat_col: -1]
    target_name = all_columns[-1]
    diversity_of_cat_data = [len(set(raw_cat_data[col])) for col in cat_columns]
    expected_columns_qty_for_ohe = sum(diversity_of_cat_data) + len(num_columns)
    expected_transformed_shape = (raw_cat_data.shape[0], expected_columns_qty_for_ohe)
    params = FeatureParams(numerical_features=num_columns, target_col=target_name,
                           categorical_features=cat_columns)
    transformer = build_transformer(params)
    raw_cat_data.drop(target_name, axis=1, inplace=True)
    transformer.fit(raw_cat_data)
    transformed_data = process_features(raw_cat_data, transformer, params)
    assert type(transformed_data) == pd.DataFrame, (
        f'Expected Pandas Dataframe type but got {type(transformed_data)}'
    )
    assert transformed_data.shape == expected_transformed_shape, (
        f'Expected shape {expected_transformed_shape} of data, but got {transformed_data.shape}'
    )


def test_drop_features(raw_cat_data: pd.DataFrame):
    all_columns = list(raw_cat_data.columns)
    start_cat_col = TEST_DATA_SHAPE[1] - TEST_DATA_CAT_COL_QTY
    cat_columns = all_columns[start_cat_col: -1]
    column_to_drop = cat_columns[:TEST_DATA_CAT_COL_QTY // 2]
    cleaned_data = drop_features(raw_cat_data, column_to_drop)
    expected_cleaned_data_shape = (
        raw_cat_data.shape[0], raw_cat_data.shape[1] - len(column_to_drop))
    assert cleaned_data.shape == expected_cleaned_data_shape, (
        f'Expected {expected_cleaned_data_shape} shape of cleaned data, but got {cleaned_data.shape}'
    )
