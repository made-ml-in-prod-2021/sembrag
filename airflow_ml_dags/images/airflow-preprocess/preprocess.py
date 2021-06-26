import os

import numpy as np
import pandas as pd
import click
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_num_pipeline() -> Pipeline:
    """Numerical pipeline according to parameters. """
    transformers_list = [
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scaler', StandardScaler())
    ]
    num_pipeline = Pipeline(transformers_list)
    return num_pipeline


@click.command('preprocess')
@click.option('--input-dir')
@click.option('--output-dir')
def preprocess(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    target = pd.read_csv(os.path.join(input_dir, 'target.csv'))
    data_train, data_test, target_train, target_test = train_test_split(data, target)
    data_process_pipeline = build_num_pipeline()
    processed_data_train = pd.DataFrame(data_process_pipeline.fit_transform(data_train))
    processed_val_data = pd.DataFrame(data_process_pipeline.transform(data_test))
    os.makedirs(output_dir, exist_ok=True)
    processed_data_train.to_csv(os.path.join(output_dir, 'data.csv'), index=False)
    target_train.to_csv(os.path.join(output_dir, 'target.csv'), index=False)
    processed_val_data.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
    target_test.to_csv(os.path.join(output_dir, 'val_target.csv'), index=False)


if __name__ == '__main__':
    preprocess()
