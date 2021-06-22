from typing import Optional
from dataclasses import dataclass

from .data_config import DataFileParams
from .data_config import SplittingParams
from .feature_config import FeatureParams
from .model_config import ModelParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    data_file_params: DataFileParams
    split_params: SplittingParams
    feature_params: FeatureParams
    model_params: ModelParams
    output_model_path: str
    metric_path: str


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
