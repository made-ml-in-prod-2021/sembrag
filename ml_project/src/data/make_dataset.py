from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.data_config import DataFileParams, SplittingParams


def load_data_from_csv(params: DataFileParams) -> pd.DataFrame:
    """Function loading data from csv-file. """
    data = pd.read_csv(params.path, sep=params.sep)
    data.dropna(inplace=True)
    return data


def split_test_train(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function splitting data for train and test according to test_size parameter. """
    train, test = train_test_split(data, test_size=params.test_size, random_state=params.random_state)
    return train, test
