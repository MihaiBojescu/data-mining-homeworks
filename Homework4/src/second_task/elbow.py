import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def plot_dendrogram(model, title: str, **kwargs):

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

    plt.title(f"{title} |Dendrogram|")
    plt.show()


def plot_hierarchical_distances(model, plot_first_n: int, title: str):
    cluster_distances = np.flip(model.distances_)[:plot_first_n].tolist()
    cluster_idxes = range(1, len(cluster_distances) + 1)
    sns.lineplot(data=pd.DataFrame({"hierarchical_cluster_cut": cluster_idxes, "cluster_distances": cluster_distances}),
                 x="hierarchical_cluster_cut", y="cluster_distances")

    plt.title(f"{title} |Cluster distances|")
    plt.show()


def do_hierarchical_clustering(features: np.array, plot_first_n: int, title: str):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(features)
    plot_dendrogram(model, title, truncate_mode="level", p=3)
    plot_hierarchical_distances(model, plot_first_n, title)


#  Inertia = Σ(distance(point, centroid)^2)
#  suma patratelor distantelor de la punctele din cluder la centroid <=> Varianta intracluster

def do_k_means_clustering(features: np.array, max_nr_of_clusters: int, title: str):

    variances = dict[int, float]()

    for nr in range(1, max_nr_of_clusters + 1):
        model = KMeans(n_clusters=nr).fit(features)
        variances[nr] = model.inertia_

    keys, values = zip(*list(variances.items()))
    sns.lineplot(data=pd.DataFrame({"nr_of_clusters": keys, "intra_cluster_covariance": values}),
                 x="nr_of_clusters", y="intra_cluster_covariance")

    plt.title(title)
    plt.show()

