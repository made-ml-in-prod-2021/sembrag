from typing import List, Union, Optional

from pydantic import BaseModel, conlist

class Data(BaseModel):
    data: List[List]
    features: List[str]
    indexes: List


class ModelOut(BaseModel):
    idx: int
    class_pred: int