from typing import Optional

from src.data_loader import (
    dataset_2d_10c,
    dataset_iris,
    dataset_long,
    dataset_order2_3clust,
    dataset_smile,
    dataset_square,
)
from src.data_utils import normalize
from src.first_task.cluster import (
    Cluster,
    Dataset,
    build_cluster_plots,
    run_dbscan,
    run_em_gmm,
    run_hierarchical_clustering,
    run_k_means,
)
from src.first_task.som import k_means, plot_torus, plot_u_matrix, train_som
from src.metrics import get_adjusted_random_index
from src.visualize import plot_data


def main():
    cluster_on_datasets()
    som_on_iris()
    som_on_iris((15, 15))


def cluster_on_datasets():
    datasets = [
        Dataset(title="2d 10c", data=dataset_2d_10c(), nr_of_clusters=10),
        Dataset(title="Long", data=dataset_long(), nr_of_clusters=2),
        Dataset(
            title="Order 2, 3 clusters", data=dataset_order2_3clust(), nr_of_clusters=3
        ),
        Dataset(title="Smile", data=dataset_smile(), nr_of_clusters=4),
        Dataset(title="Square", data=dataset_square(), nr_of_clusters=4),
    ]

    for dataset in datasets:
        features, labels = dataset.data
        features = normalize(features)

        clusters_k_means = run_k_means(
            x=features, nr_of_clusters=dataset.nr_of_clusters
        )
        clusters_em_gmm = run_em_gmm(x=features, nr_of_clusters=dataset.nr_of_clusters)
        clusters_hierarchical_single_linkage = run_hierarchical_clustering(
            x=features, linkage="single", nr_of_clusters=dataset.nr_of_clusters
        )
        clusters_hierarchical_average_linkage = run_hierarchical_clustering(
            x=features, linkage="average", nr_of_clusters=dataset.nr_of_clusters
        )
        clusters_hierarchical_complete_linkage = run_hierarchical_clustering(
            x=features, linkage="complete", nr_of_clusters=dataset.nr_of_clusters
        )
        clusters_hierarchical_ward_linkage = run_hierarchical_clustering(
            x=features, linkage="ward", nr_of_clusters=dataset.nr_of_clusters
        )
        clusters_dbscan = run_dbscan(x=features)

        build_cluster_plots(
            x=features,
            all_clusters=[
                Cluster(title="K-Means", y=labels, y_hat=clusters_k_means),
                Cluster(
                    title="EM/GMM",
                    y=labels,
                    y_hat=clusters_em_gmm,
                ),
                Cluster(
                    title="Hierarchical, \nsingle linkage",
                    y=labels,
                    y_hat=clusters_hierarchical_single_linkage,
                ),
                Cluster(
                    title="Hierarchical, \naverage linkage",
                    y=labels,
                    y_hat=clusters_hierarchical_average_linkage,
                ),
                Cluster(
                    title="Hierarchical, \ncomplete linkage",
                    y=labels,
                    y_hat=clusters_hierarchical_complete_linkage,
                ),
                Cluster(
                    title=f"Hierarchical, \nward linkage",
                    y=labels,
                    y_hat=clusters_hierarchical_ward_linkage,
                ),
                Cluster(
                    title=f"Density-based \nscanning",
                    y=labels,
                    y_hat=clusters_dbscan,
                ),
            ],
            title=f"Clusters for {dataset.title}, {dataset.nr_of_clusters} clusters",
        )


def som_on_iris(map_size: Optional[tuple[int, int]] = None):
    features, labels = dataset_iris()
    features = normalize(features)
    som = train_som(features, map_size)
    plot_u_matrix(som, title="U Matrix")
    plot_u_matrix(som, title="U Matrix with Hits", hits=True)
    clusters = k_means(som, 3)

    print(
        f"Adjusted Random Index: "
        f"{get_adjusted_random_index(labels.reshape(-1), clusters.reshape(-1))}"
    )

    plot_data(
        features,
        clusters,
        f"Clusters from K-Means with K=3 on SOM neuron weights{map_size if map_size else ''}",
    )


if __name__ == "__main__":
    main()
