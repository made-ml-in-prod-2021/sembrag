import os
from typing import Tuple

import joblib
import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def download_preprocessed_data(path_to_data: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(os.path.join(path_to_data, 'data.csv'))
    target = pd.read_csv(os.path.join(path_to_data, 'target.csv'))
    return data, target


@click.command('train')
@click.option('--input-dir')
@click.option('--output-dir')
def train_model(input_dir: str, output_dir: str):
    """Function builts model and train it. """
    data, target = download_preprocessed_data(input_dir)
    model = RandomForestClassifier(n_estimators=150)
    model.fit(data, target)
    os.makedirs(output_dir, exist_ok=True)
    model_out_path = os.path.join(output_dir, 'model.pkl')
    with open(model_out_path, 'wb') as fio:
        joblib.dump(model, fio)


if __name__ == '__main__':
    train_model()