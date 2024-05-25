import typing as t
import numpy as np


def get_outliers_mean_k_sd(data: np.ndarray[t.Literal["N"], float], k: int = 3):
    data = np.array(data).copy()
    data.sort()

    mean = np.mean(data)
    sd = np.std(data)

    lower_bound = mean - k * sd
    upper_bound = mean + k * sd

    return data[(data < lower_bound) | (data > upper_bound)]
