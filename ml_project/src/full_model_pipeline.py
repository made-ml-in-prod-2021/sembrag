import sys
import logging

import click

from src.config.config import TrainingPipelineParams, read_training_pipeline_params
from src.data.make_dataset import load_data_from_csv, split_test_train
from src.features.build_features import get_target, build_transformer, process_features, \
    drop_features
from src.models.train_model import train_model, create_model_pipeline, export_model
from src.models.predict_model import predict_model, model_score, export_score

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_model_pipeline(params: TrainingPipelineParams) -> str:
    logger.info(f'Start train pipeline with params {params}')
    data = load_data_from_csv(params.data_file_params)
    logger.info(f'Loaded raw data shape is {data.shape}')
    data = drop_features(data, params.feature_params.features_to_drop)
    logger.info(f'Data shape after feature drop is {data.shape}')
    train, test = split_test_train(data, params.split_params)
    logger.info(f'Train dataset is {train.shape} and test (validation) dataset is {test.shape}')
    train_target, train_features = get_target(train, params.feature_params)
    test_target, test_features = get_target(test, params.feature_params)
    logger.info(f'Splitted train and test datasets for target and features')
    transformer = build_transformer(params.feature_params)
    transformer.fit(train_features)
    logger.info('Transformer has been built and fitted')
    train_features = process_features(train_features, transformer, params.feature_params)
    logger.info(f'Train_features dataset after transformation shape is {train_features.shape}')
    model = train_model(train_features, train_target, params.model_params)
    model_pipeline = create_model_pipeline(model, transformer)
    logger.info(
        f'Model has been built and added to pipeline. Model type is {params.model_params.model_type}')
    val_predicts = predict_model(model_pipeline, test_features)
    scores = model_score(val_predicts, test_target)
    logger.info(f'Model has been validated with score {scores}')
    export_score(scores, params.metric_path)
    logger.info(f'Scores have been exported to the {params.metric_path}')
    export_model(model_pipeline, params.output_model_path)
    logger.info(f'Model has been exported to the {params.output_model_path}')


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    run_model_pipeline(params)


if __name__ == '__main__':
    train_pipeline_command()
