import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago

DATA_FOLDER = '/home/smf/hw3/data'

default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def _wait_for_file(path: str) -> bool:
    return os.path.exists(path)


with DAG(
        '01_train_model',
        default_args=default_args,
        schedule_interval='@weekly',
        start_date=days_ago(20),
) as dag:
    wait = PythonSensor(
        task_id='wait_for_file',
        python_callable=_wait_for_file,
        op_kwargs={'path': '/opt/airflow/data' + '/raw/{{ ds }}/data.csv'},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    preprocess = DockerOperator(
        image='airflow-preprocess',
        command='--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}',
        task_id='docker-airflow-preprocess',
        do_xcom_push=False,
        volumes=[f'{DATA_FOLDER}:/data']
    )

    train = DockerOperator(
        image='airflow-train',
        command='--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}',
        task_id='docker-airflow-train',
        do_xcom_push=False,
        volumes=[f'{DATA_FOLDER}:/data']
    )

    validate = DockerOperator(
        image='airflow-val',
        command='--input-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }} --score-dir /data/models/{{ ds }}',
        task_id='docker-airflow-val',
        do_xcom_push=False,
        volumes=[f'{DATA_FOLDER}:/data']
    )

    wait >> preprocess >> train >> validate
