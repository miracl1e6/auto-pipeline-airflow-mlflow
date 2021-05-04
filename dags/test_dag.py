from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "miracl6",
    "start_date": days_ago(1),  # запуск день назад
    "retries": 5,  # запуск таска до 5 раз, если ошибка
    #"retry_delay": datetime.timedelta(minutes=5),  # дельта запуска при повторе 5 минут
    "task_concurency": 1  # одновременно только 1 таск
}

piplines = {'train': {"schedule": "1 * * * *"},  # At 20:39 on Saturday MSK
            "predict": {"schedule": "2 * * * *"}}  # At 23:48 every day - 3 hours diff

def init_dag(dag, task_id):
    with dag:
        t1 = BashOperator(
            task_id=f"{task_id}",
            bash_command=f'python3 /Users/miracl6/airflow-mlflow-tutorial/{task_id}.py')
    return dag

for task_id, params in piplines.items():
    # DAG - ациклический граф
    dag = DAG(task_id,
              schedule_interval=params['schedule'],
              max_active_runs=1,
              default_args=default_args
              )
    init_dag(dag, task_id)
    globals()[task_id] = dag
