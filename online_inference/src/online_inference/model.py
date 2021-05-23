import logging
from typing import List

import joblib
import pandas as pd
from pydantic import BaseModel, conlist
from src.online_inference.DataClasses_ import Data, ModelOut

LOG_FILE = 'model.log'
APP_NAME = 'MODEL'


class Model:
    """Class for working with model loaded from file. """

    def __init__(self, path: str):
        """Init function loading model from pickle file. """
        self.model = joblib.load(path)

    def make_predict(self, data: List, features: List[str], indexes: List) -> List[ModelOut]:
        """Method getting data with features and indexes and giving back predictions of model. """
        data_ = pd.DataFrame(data, columns=features, index=indexes)
        idxs = [int(idx) for idx in indexes]
        predicts = self.model.predict(data_)
        return [ModelOut(idx=int(idx), class_pred=int(predicted_class)) for idx, predicted_class in
                zip(idxs, predicts)]


