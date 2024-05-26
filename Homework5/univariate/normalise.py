import typing as t
import numpy as np

def normalise(data: np.ndarray[t.Literal["N"], float]):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)