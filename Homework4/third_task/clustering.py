import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering, KMeans


def dbscan_clustering_obesity(features: np.array) -> np.array:
    features = features
    prediction = DBSCAN(eps=0.1, min_samples=20).fit_predict(features)
    if -1 in prediction:
        print("Found outliers")

    print(f"{len(np.unique(prediction))} different classed found")
    return prediction


def hierarchical_clustering_obesity(features: np.array, n_clusters: int) -> np.array:
    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)


def k_means_clustering_obesity(features: np.array, n_clusters: int) -> np.array:
    return KMeans(n_clusters=n_clusters, random_state=100).fit_predict(features)
