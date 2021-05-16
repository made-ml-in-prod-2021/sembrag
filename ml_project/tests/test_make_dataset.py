import pytest
import numpy as np
import pandas as pd
from faker import Faker

from src.data.make_dataset import load_data_from_csv, split_test_train
from src.config.data_config import DataFileParams, SplittingParams

TEST_DATA_SHAPE = (1000, 15)
CSV_SEP = ';'


@pytest.fixture
def raw_data() -> pd.DataFrame:
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


def test_load_data_from_csv(tmpdir, raw_data: pd.DataFrame):
    csv_file_temp_path = tmpdir.join('data_file.csv')
    raw_data.to_csv(csv_file_temp_path, sep=CSV_SEP, index=False)
    load_params = DataFileParams(path=csv_file_temp_path, sep=CSV_SEP)
    data_from_csv = load_data_from_csv(load_params)
    assert raw_data.shape == data_from_csv.shape, (
        f'Expected {raw_data.shape} shape, but got {data_from_csv.shape}'
    )
    assert all(raw_data.columns == data_from_csv.columns), (
        f'Expected columns {raw_data.columns} \n but got columns {data_from_csv.columns}'
    )
    assert all(raw_data == data_from_csv), (f'Expected \n {raw_datadata} \n but got \n {data_from_csv}')