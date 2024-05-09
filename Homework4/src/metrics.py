import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score


def get_adjusted_random_index(labels_true: np.array, labels_pred: np.array):
    assert len(labels_true.shape) == 1
    assert len(labels_pred.shape) == 1
    return adjusted_rand_score(labels_true, labels_pred)


def get_silhouette_width(features: np.array, clusters: np.array):
    assert len(clusters.shape) == 1
    return silhouette_score(features, clusters)
