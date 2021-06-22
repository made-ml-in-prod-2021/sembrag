import os

import numpy as np
import pandas as pd
import requests
import click

from src.online_inference.app import ModelOut

DEFAULT_PATH = os.path.join('model', 'test_for_predict.csv')


@click.command(name='predict')
@click.argument('model_ip')
@click.argument('path_to_data', default=DEFAULT_PATH)
def predict_command(model_ip: str, path_to_data: str):
    test_data = pd.read_csv(path_to_data, index_col=0)
    features = test_data.columns.to_list()
    indexes = test_data.index.to_list()
    request_data = test_data.values.tolist()
    for index, req_data in zip(indexes, request_data):
        response = requests.get(
            f'http://{model_ip}:8000/predict/',
            json={'data': [req_data], 'features': features, 'indexes': [index]},
        )
        print(response.status_code)
        print(response.json())


if __name__ == '__main__':
    predict_command()
