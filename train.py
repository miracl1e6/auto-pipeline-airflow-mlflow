import logging
import os

import yaml
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

import mlflow
import src.cluster_train as cl
import src.get_comments as gc
import src.preprocessing_text as pt

config_path = os.path.join('/Users/miracl6/airflow-mlflow-tutorial/config/params_all.yaml')
config = yaml.safe_load(open(config_path))['train']
os.chdir(config['dir_folder'])
SEED = config['SEED']

logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def get_version_model(config_name, client):
    """
    Получение последней версии модели из MLFlow
    """
    dict_push = {}
    for count, value in enumerate(client.search_model_versions(f"name='{config_name}'")):
        # Все версии модели
        dict_push[count] = value
    return dict(list(dict_push.items())[-1][1])['version']


def main():
    """
    Получение тематик из текста и сохранение модели
    """
    # Выгрузка топ комменариев
    comments = gc.get_all_comments(**config['comments'])

    comments_clean = pt.get_clean_text(comments, stopwords.words(config['stopwords']))
    tfidf = TfidfVectorizer(**config['tf_model']).fit(comments_clean)

    # Матрица векторизов комментов и модель
    X_matrix = pt.vectorize_text(comments_clean, tfidf)

    # Разделение текста на тематики при помощи кластеризации, выбор наилучшего числа кластеров
    cluster_labels = cl.get_clusters(X_matrix,
                                     random_state=SEED,
                                     **config['clustering'])

    # Обучение линейной модели на поиска сформированных тематик
    X_train, X_test, y_train, y_test = train_test_split(X_matrix,
                                                        cluster_labels,
                                                        **config['cross_val'],
                                                        random_state=SEED)
    clf_lr = LogisticRegression(**config['model'])

    # MLFlow трэкинг
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(config['name_experiment'])
    with mlflow.start_run():
        clf_lr.fit(X_train, y_train)

        # Логирование модели и параметров
        mlflow.log_param('f1',
                         cl.get_f1_score(y_test, clf_lr.predict(X_test), set(cluster_labels)))
        mlflow.log_param('accuracy',
                         accuracy_score(y_test, clf_lr.predict(X_test)))
        mlflow.log_param('precision',
                         cl.get_precision_score(y_test, clf_lr.predict(X_test), set(cluster_labels)))
        mlflow.sklearn.log_model(tfidf,
                                 artifact_path="vector",
                                 registered_model_name=f"{config['model_vec']}")
        mlflow.sklearn.log_model(clf_lr,
                                 artifact_path='model_lr',
                                 registered_model_name=f"{config['model_lr']}")
        mlflow.log_artifact(local_path='./train.py',
                            artifact_path='code')
        mlflow.end_run()

    # Получение последней версии модели и сохраннение в файлы
    client = MlflowClient()
    last_version_lr = get_version_model(config['model_lr'], client)
    last_version_vec = get_version_model(config['model_vec'], client)

    yaml_file = yaml.safe_load(open(config_path))
    yaml_file['predict']["version_lr"] = int(last_version_lr)
    yaml_file['predict']["version_vec"] = int(last_version_vec)

    with open(config_path, 'w') as fp:
        yaml.dump(yaml_file, fp, encoding='UTF-8', allow_unicode=True)


if __name__ == "__main__":
    main()
