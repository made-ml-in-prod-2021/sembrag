import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable

DATA_FOLDER = '/home/smf/hw3/data'

default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def _wait_for_file(path: str) -> bool:
    print('!!!!!!!!#########', path)
    return os.path.exists(path)


used_model_path = Variable.get('used_model_path')
used_model_path_with_file = os.path.join(used_model_path, 'model.pkl')
model_dir = used_model_path.lstrip('/opt/airflow')

with DAG(
        '02_predict',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=days_ago(20),
) as dag:
    wait_model = PythonSensor(
        task_id='wait_for_model',
        python_callable=_wait_for_file,
        op_kwargs={'path': used_model_path_with_file},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode='poke',
    )
    wait_data = PythonSensor(
        task_id='wait_for_data',
        python_callable=_wait_for_file,
        op_kwargs={'path': '/opt/airflow/data' + '/raw/{{ ds }}/data.csv'},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode='poke',
    )
    predict = DockerOperator(
        image='airflow-predict',
        command='--input-dir /data/processed/{{ ds }} --output-dir /data/predictions/{{ ds }} --model-dir ' + model_dir,
        task_id='docker-airflow-predict',
        do_xcom_push=False,
        volumes=[f'{DATA_FOLDER}:/data']
    )

    wait_model >> wait_data >> predict
