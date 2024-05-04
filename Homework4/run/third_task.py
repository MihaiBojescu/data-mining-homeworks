import numpy as np
from Homework4.data_loader import dataset_obesity
from Homework4.second_task.elbow import do_hierarchical_clustering, do_k_means_clustering
from Homework4.third_task.clustering import dbscan_clustering_obesity, hierarchical_clustering_obesity, k_means_clustering_obesity
from Homework4.visualize import plot_data, plot_contingency_matrix
from Homework4.data_utils import normalize
from Homework4.metrics import get_adjusted_random_index


def obesity_tasks():
    features, labels = dataset_obesity()
    #features = normalize(features)
    print(f"Nr. of different classes {len(np.unique(labels))}")
    do_hierarchical_clustering(features, 10, "Obesity hierarchical clustering with Ward metric")
    # predicted_labels = dbscan_clustering_obesity(features)
    # predicted_labels = hierarchical_clustering_obesity(features, 7)
    predicted_labels = k_means_clustering_obesity(features, 7)
    plot_data(features[:, :2], predicted_labels, "Obesity clustering with hierarchical clustering with Ward metric")
    plot_data(features[:, :2], labels.reshape(-1), "Obesity true labels")

    print(f"Adjusted Random Index: {get_adjusted_random_index(labels.reshape(-1), predicted_labels.reshape(-1))}")

    plot_contingency_matrix(labels.reshape(-1), predicted_labels.reshape(-1))


if __name__ == "__main__":
    obesity_tasks()
