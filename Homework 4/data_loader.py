import os
import pandas as pd
import numpy as np


def load_dataset(path: str, delimiter: str) -> tuple[np.array, np.array]:
    with open(path, "r") as fd:
        dataset = pd.read_csv(fd, delimiter=delimiter).to_numpy()
        shape = dataset.shape
        assert shape[1] > 2
        return dataset[:, :-1], dataset[:, -1:]


def dataset_2d_10c() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(__file__, "./data/2d-10c.dat"), " ")


def dataset_iris() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(__file__, "./data/iris.csv"), ",")


def dataset_long() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(__file__, "./data/long.dat"), " ")


def dataset_order2_3clust() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(__file__, "./data/order2-3clust.csv"), ",")


def dataset_smile() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(__file__, "./data/smile.csv"), ",")


def dataset_square() -> tuple[np.array, np.array]:
    return load_dataset(os.path.join(__file__, "./data/square.dat"), " ")
