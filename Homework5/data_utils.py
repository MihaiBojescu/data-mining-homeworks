import numpy as np
from sklearn import preprocessing


def normalize(data: np.array) -> np.array:
    return preprocessing.StandardScaler().fit_transform(data)
