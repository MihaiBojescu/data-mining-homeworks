from visualize import plot_data
from data_loader import dataset_smile, dataset_long, dataset_square, dataset_iris, dataset_2d_10c, dataset_order2_3clust
from data_utils import normalize
from first_task.som import train_som, plot_u_matrix_with_bmu


def main_plot():
    #features, labels = dataset_smile()
    #features, labels = dataset_long()
    #features, labels = dataset_square()
    features, labels = dataset_iris()
    #features, labels = dataset_2d_10c()
    #features, labels = dataset_order2_3clust()
    plot_data(normalize(features), labels.reshape(-1))
    plot_data(features, labels.reshape(-1))


def main_test_som():
    features, labels = dataset_2d_10c()
    som = train_som(features)
    plot_u_matrix_with_bmu(som)
    #print(som.results_dataframe)
    # plot_data(normalize(features), labels.reshape(-1))
    # plot_data(features, labels.reshape(-1))


if __name__ == "__main__":
    #main_plot()
    main_test_som()


