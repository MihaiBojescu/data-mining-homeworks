from Homework4.data_loader import dataset_iris, dataset_2d_10c
from Homework4.data_utils import normalize
from Homework4.second_task.elbow import do_hierarchical_clustering, do_k_means_clustering
from Homework4.second_task.index import do_k_means_clustering_with_index, do_hierarchical_clustering_with_index


def elbow():
    features_iris, labels_iris = dataset_iris()
    features_iris = normalize(features_iris)
    do_hierarchical_clustering(features_iris, 10, "Iris with Ward metric")

    features_2d_10c, labels_2d_10c = dataset_2d_10c()
    features_2d_10c = normalize(features_2d_10c)
    do_hierarchical_clustering(features_2d_10c, 15, "2d_10c with Ward metric")

    do_k_means_clustering(features_iris, 10, "Iris |K-Means| intra cluster variance")
    do_k_means_clustering(features_2d_10c, 15, "2d_10c |K-Means| intra cluster variance")


def silhouette_width():
    features_iris, labels_iris = dataset_iris()
    features_iris = normalize(features_iris)
    nr_of_clusters, silhouette_width_val = do_hierarchical_clustering_with_index(features_iris, 10, "Iris |Hierarchical - Ward| - silhouette_width")
    print(f"Iris |Hierarchical - Ward|, the best silhouette_width: {silhouette_width_val} on {nr_of_clusters} clusters")

    features_2d_10c, labels_2d_10c = dataset_2d_10c()
    features_2d_10c = normalize(features_2d_10c)
    nr_of_clusters, silhouette_width_val = do_hierarchical_clustering_with_index(features_2d_10c, 20, "2d_10c |Hierarchical - Ward| - silhouette_width")
    print(f"2d_10c |Hierarchical - Ward|, the best silhouette_width: {silhouette_width_val} on {nr_of_clusters} clusters")

    nr_of_clusters, silhouette_width_val = do_k_means_clustering_with_index(features_iris, 10, "Iris |K-Means| - silhouette_width")
    print(f"Iris |K-Means|, the best silhouette_width: {silhouette_width_val} on {nr_of_clusters} clusters")
    nr_of_clusters, silhouette_width_val = do_k_means_clustering_with_index(features_2d_10c, 20, "2d_10c |K-Means| - silhouette_width")
    print(f"2d_10c |K-Means|, the best silhouette_width: {silhouette_width_val} on {nr_of_clusters} clusters")


if __name__ == "__main__":
    #elbow()
    silhouette_width()
