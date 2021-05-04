from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score, silhouette_score, precision_score


def get_clusters(data, count_max_clusters, random_state, affinity,
                 silhouette_metric):
    """
    Подбор наилучшего числа кластеров, возвращает полученные кластера тематик
    """
    cluster_labels = {}
    silhouette_mean = []

    for i in range(2, count_max_clusters, 1):
        clf = SpectralClustering(n_clusters=i,
                                 affinity=affinity,
                                 random_state=random_state)
        clf.fit(data)
        cluster_labels[i] = clf.labels_
        silhouette_mean.append(
            silhouette_score(data, clf.labels_, metric=silhouette_metric))
    n_clusters = silhouette_mean.index(max(silhouette_mean)) + 2
    return cluster_labels[n_clusters]


def get_f1_score(y_test, y_pred, unique_cluster_labels):
    """
    Возращает результат обучения классификатора по тематикам
    """
    return f1_score(
        y_test, y_pred,
        average='macro') \
        if len(unique_cluster_labels) > 2 \
        else f1_score(y_test, y_pred)

def get_precision_score(y_test, y_pred, unique_cluster_labels):
    """
    Возращает результат обучения классификатора по тематикам
    """
    return precision_score(
        y_test, y_pred,
        average='macro') \
        if len(unique_cluster_labels) > 2 \
        else precision_score(y_test, y_pred)

