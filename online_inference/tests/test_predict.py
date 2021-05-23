import os
from unittest import mock
from typing import List

import pytest
from fastapi.testclient import TestClient
import pandas as pd

from src.online_inference.app import app, app_name



@pytest.fixture(autouse=True)
def mock_settings_env_vars():
    default_path_to_model = os.path.join('online_inference', 'model', 'clf_hw2.pkl')
    with mock.patch.dict(os.environ, {"PATH_TO_MODEL": default_path_to_model}):
        yield


def test_model_is_loaded():
    print(os.getcwd())
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.json(), (
            f'Model is not loaded.'
        )


def test_entry_point():
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.json() == f'{app_name} entry point', (
            f'Waited {app_name} as application name but got {response.json()}'
        )


def test_predict_end_point():
    with TestClient(app) as client:
        path_to_data = os.path.join('online_inference', 'model', 'test_for_predict.csv')
        test_data = pd.read_csv(path_to_data, index_col=0)
        features = test_data.columns.to_list()
        indexes = test_data.index.to_list()
        request_data = test_data.values.tolist()
        response = client.get(
            '/predict/',
            json={'data': request_data, 'features': features, 'indexes': indexes},
        )
        predicts = response.json()
        assert response.status_code == 200
        assert type(response.json()) == list
        assert all(['idx' in predict for predict in predicts])
        assert all(['class_pred' in predict for predict in predicts])
        assert all([0 <= predict['class_pred'] <= 1 for predict in predicts])


