# Автоматизаци пайплана ML при помощи Airflow и Mlflow

## VENV

python -m venv myyvenv

Порядок действий для первичной установки без postgresql (SQLite)

## Mlflow

1) Создаем папку и виртуальное окружение mkdir airflow-mlflow-tutorial python -m venv myvenv source myvenv/bin/activate

2) Установка sqlite pip install pysqlite3

3) Установка 

mlflow pip install 

mlflow mkdir mlflow 

export MLFLOW_REGISTRY_URI=mlflow

Полезная инфа: https://www.mlflow.org/docs/latest/tracking.html#tracking-ui

4) Запуск сервера:
   mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///${MLFLOW_REGISTRY_URI}/mlflow.db --default-artifact-root ${MLFLOW_REGISTRY_URI}

5) Если хотите что то перезапустить и убить процессы 

ps -A | grep gunicorn 

kill -9 `ps aux | grep mlflow | awk '{print $2}'`

## Airflow

1) Создаем отдельную директорию mkdir airflow

2) Установка airflow pip install apache-airflow==2.0.1
   --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.0.1/constraints-3.7.txt"
   export AIRFLOW_HOME=.

3) Инициализаци БД airflow db init Также в файле конфигурации airflow.cfg прописать:
   [webserver]
   rbac = True

а также:
load_examples = False

4) Создание пользователя airflow users create --username miracl6 --firstname miracl6 --lastname miracl6 --role Admin
   --email ***@***.com

5) Запуск Airflow 
 
**!!перед этим, когда запускаете уже в другой например день:**

source "ваш venv"
export AIRFLOW_HOME=.


airflow webserver -p 8080 

airflow scheduler

6) Если хотите что то перезапустить и убить процессы 

ps -A | grep gunicorn

kill -9 `ps aux | grep airflow | awk '{print $2}'`



kill -9 `ps aux | grep mlflow | awk '{print $2}'`

