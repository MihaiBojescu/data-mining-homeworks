import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from src.metrics import get_silhouette_width
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def do_hierarchical_clustering_with_index(features: np.array, max_nr_of_clusters: int, title: str) -> tuple[int, float]:
    silhouette_score_map = dict[int, float]()
    for nr in range(2, max_nr_of_clusters + 1):
        clusters = AgglomerativeClustering(n_clusters=nr).fit_predict(features)
        silhouette_score_map[nr] = get_silhouette_width(features, clusters)

    plot_index_per_cluster(silhouette_score_map, title)
    return max(list(silhouette_score_map.items()), key=lambda item: item[1])


def do_k_means_clustering_with_index(features: np.array, max_nr_of_clusters: int, title: str) -> tuple[int, float]:
    silhouette_score_map = dict[int, float]()
    for nr in range(2, max_nr_of_clusters + 1):
        clusters = KMeans(n_clusters=nr).fit_predict(features)
        silhouette_score_map[nr] = get_silhouette_width(features, clusters)

    plot_index_per_cluster(silhouette_score_map, title)
    return max(list(silhouette_score_map.items()), key=lambda item: item[1])


def plot_index_per_cluster(silhouette_score_map: dict[int, float], title: str):
    keys, values = zip(*list(silhouette_score_map.items()))
    sns.lineplot(data=pd.DataFrame({"nr_of_clusters": keys, "silhouette_width": values}),
                 x="nr_of_clusters", y="silhouette_width")

    plt.title(title)
    plt.show()

