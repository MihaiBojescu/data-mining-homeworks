from src.data_loader import dataset_wine
from src.run.first_task import run_univariate, run_multivariate


def main():
    features, labels = dataset_wine()
    columns = [
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline",
    ]
    autoencoder_model_state_dict_path = "second_task_model.pt"

    univariate_outliers = run_univariate(features=features, labels=labels)
    multivariate_outliers = run_multivariate(
        features=features,
        labels=labels,
        columns=columns,
        autoencoder_model_state_dict_path=autoencoder_model_state_dict_path,
    )


if __name__ == "__main__":
    main()
