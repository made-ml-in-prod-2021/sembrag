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
        '02_predict',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=days_ago(20),
) as dag:

    wait_data = PythonSensor(
        task_id='wait_for_file',
        python_callable=_wait_for_file,
        op_kwargs={'path': DATA_FOLDER + '/raw/{{ ds }}/data.csv'},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode='poke',
    )
    predict = DockerOperator(
        image='airflow-predict',
        command='--input-dir /data/processed/{{ ds }} --output-dir /data/predictions/{{ ds }} --model-dir /data/models/{{ ds }}',
        task_id='docker-airflow-predict',
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=[f'{DATA_FOLDER}:/data']
    )

    wait_data >> predict