import typing as t
import numpy as np


def get_outliers_k_iqr(
    data: np.ndarray[t.Literal["N"], float],
    q1: float = 25,
    q3: float = 75,
    k_iqr: float = 1.5,
):
    data = np.array(data).copy()
    data.sort()

    q1_value = np.percentile(data, q=q1)
    q3_value = np.percentile(data, q=q3)
    iqr_value = q3_value - q1_value

    lower_bound = q1_value - k_iqr * iqr_value
    upper_bound = q3_value + k_iqr * iqr_value

    return data[(data < lower_bound) | (data > upper_bound)]
