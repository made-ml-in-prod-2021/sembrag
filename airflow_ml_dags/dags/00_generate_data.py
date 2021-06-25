from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

DATA_FOLDER = '/home/smf/PycharmProjects/sembrag_hw3/airflow_ml_dags/data'

default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        '00_generate_data',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=days_ago(20),
) as dag:
    generate = DockerOperator(
        image='airflow-generate',
        command='/data/raw/{{ ds }}',
        task_id='docker-airflow-generate',
        do_xcom_push=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        volumes=[f'{DATA_FOLDER}:/data']
    )
