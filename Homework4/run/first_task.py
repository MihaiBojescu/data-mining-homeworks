from Homework4.data_loader import dataset_iris
from Homework4.data_utils import normalize
from Homework4.first_task.som import train_som, plot_u_matrix, plot_torus, k_means
from Homework4.metrics import get_adjusted_random_index
from Homework4.visualize import plot_data
from typing import Optional


def som_on_iris(map_size: Optional[tuple[int, int]] = None):
    features, labels = dataset_iris()
    features = normalize(features)
    som = train_som(features, map_size)
    plot_u_matrix(som, title="U Matrix")
    plot_u_matrix(som, title="U Matrix with Hits", hits=True)
    clusters = k_means(som, 3)

    print(f"Adjusted Random Index: "
          f"{get_adjusted_random_index(labels.reshape(-1), clusters.reshape(-1))}")

    plot_data(features, clusters, f"Clusters from K-Means with K=3 on SOM neuron weights{map_size if map_size else ''}")


if __name__ == "__main__":
    som_on_iris()
    som_on_iris((15, 15))
