from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import typing as t
import numpy as np
import matplotlib.pyplot as plt
from src.metrics import get_adjusted_random_index


@dataclass
class Dataset:
    title: str
    data: np.ndarray
    nr_of_clusters: int


@dataclass
class Cluster:
    title: str
    y: np.ndarray
    y_hat: np.ndarray

    @property
    def full_title(self) -> str:
        adjusted_random_index = get_adjusted_random_index(
            self.y.reshape(-1), self.y_hat.reshape(-1)
        )
        return f"{self.title}. ARI: {adjusted_random_index:.5f}"


def run_k_means(x: list[int], nr_of_clusters: int):
    return KMeans(
        n_clusters=nr_of_clusters,
        max_iter=400,
    ).fit_predict(X=x)


def run_em_gmm(x: list[int], nr_of_clusters: int):
    return GaussianMixture(
        n_components=nr_of_clusters,
        max_iter=100,
    ).fit_predict(X=x)


def run_hierarchical_clustering(
    x: list[int],
    linkage: t.Union[
        t.Literal["ward"],
        t.Literal["complete"],
        t.Literal["average"],
        t.Literal["single"],
    ],
    nr_of_clusters: int,
):
    return AgglomerativeClustering(
        n_clusters=nr_of_clusters, metric="euclidean", linkage=linkage
    ).fit_predict(X=x)


def run_dbscan(x: list[int], eps: float = 0.5, min_samples: int = 5):
    """
    Note:
       Density-based scanning does not accept a number of clusters in advance.
       Rather, it infers the number of clusters.
    """
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X=x)


def build_cluster_plots(
    x: np.array,
    all_clusters: list[Cluster],
    title: str,
    columns=2,
):
    fig, axs = plt.subplots(len(all_clusters) // columns + 1, columns, constrained_layout=True, )

    fig.set_figheight((len(all_clusters) // columns + 1) * 3)
    fig.set_figwidth(columns * 3)
    fig.suptitle(title)

    for i, cluster in enumerate(all_clusters):
        axs[i // columns][i % columns].scatter(x[:, 0], x[:, 1], c=cluster.y_hat)
        axs[i // columns][i % columns].set_title(cluster.full_title)
        axs[i // columns][i % columns].set_xlabel("X")
        axs[i // columns][i % columns].set_ylabel("Y")

    fig.show()
