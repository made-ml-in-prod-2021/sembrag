import os
from typing import Tuple

import click
import numpy as np
import pandas as pd

TEST_DATA_SHAPE = (1000, 6)
COEFS = (-15, 1, 67, -37, 20, 5)  # Coefficients for generating data, some random numbers, does not matter


def raw_num_data() -> Tuple[pd.DataFrame, pd.Series]:
    # Prepare random data
    random_data = np.random.randn(*TEST_DATA_SHAPE)
    df = pd.DataFrame(random_data)
    # Create target with a simple regression
    target = pd.Series([0] * TEST_DATA_SHAPE[0])
    for i, column in enumerate(df.columns):
        target += COEFS[i] * df[column]
    target_class = pd.Series([0 if target_ < 0 else 1 for target_ in target])
    return df, target_class


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    data, target = raw_num_data()
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, 'data.csv'))
    target.to_csv(os.path.join(output_dir, 'target.csv'))


if __name__ == '__main__':
    download()