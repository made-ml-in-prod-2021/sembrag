import os

import joblib
import pandas as pd
import click


@click.command('predict')
@click.option('--input-dir')
@click.option('--output-dir')
@click.option('--model-dir')
def predict(input_dir: str, output_dir: str, model_dir: str):
    data = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as fio:
        model = joblib.load(fio)
    predicts = pd.Series(model.predict(data))
    os.makedirs(output_dir, exist_ok=True)
    predicts.to_csv(os.path.join(output_dir, 'predictions.csv'))

if __name__ == '__main__':
    predict()