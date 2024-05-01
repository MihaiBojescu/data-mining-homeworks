from Homework4.data_loader import dataset_iris
from Homework4.data_utils import normalize
from Homework4.first_task.som import train_som, plot_u_matrix, plot_torus, k_means
from Homework4.metrics import get_adjusted_random_index
from Homework4.visualize import plot_data


def som_on_iris():
    features, labels = dataset_iris()
    som = train_som(normalize(features))
    plot_u_matrix(som, title="U Matrix")
    plot_u_matrix(som, title="U Matrix with Hits", hits=True)
    clusters = k_means(som, 3)

    print(f"Adjusted Random Index: "
          f"{get_adjusted_random_index(labels.reshape(-1), clusters.reshape(-1))}")

    plot_data(features, clusters, "Clusters from K-Means with K=3 on SOM neuron weights")


if __name__ == "__main__":
    som_on_iris()
