import numpy as np
import pandas as pd
import requests

from src.online_inference.app import ModelOut

if __name__ == '__main__':
    test_data = pd.read_csv('..\\..\\model\\test_for_predict.csv', index_col=0)
    features = test_data.columns.to_list()
    indexes = test_data.index.to_list()
    request_data = test_data.values.tolist()
    response = requests.get(
        'http://127.0.0.1:8000/predict/',
        json={'data': request_data, 'features': features, 'indexes': indexes},
    )
    print(response.status_code)
    print(response.json())