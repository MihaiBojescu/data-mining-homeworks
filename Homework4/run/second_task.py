from Homework4.data_loader import dataset_iris, dataset_2d_10c
from Homework4.data_utils import normalize
from Homework4.second_task.elbow import do_hierarchical_clustering, do_k_means_clustering


if __name__ == "__main__":
    features_iris, labels_iris = dataset_iris()
    do_hierarchical_clustering(features_iris, 10, "Iris with Ward metric")

    features_2d_10c, labels_2d_10c = dataset_2d_10c()
    do_hierarchical_clustering(features_2d_10c, 15, "2d_10c with Ward metric")

    do_k_means_clustering(features_iris, 10, "Iris |K-Means| intra cluster variance")
    do_k_means_clustering(features_2d_10c, 15, "2d_10c |K-Means| intra cluster variance")
