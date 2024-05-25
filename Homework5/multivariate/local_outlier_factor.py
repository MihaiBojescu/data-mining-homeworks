import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


class LocalOutlierFactorOutlierDetector:
    def __init__(self, contamination: float | str = "auto"):
        self.__contamination = contamination

    def build(self, features: np.array):
        pass

    def predict(self, features: np.array) -> pd.DataFrame:

        model = LocalOutlierFactor(contamination=self.__contamination)
        outliers = model.fit_predict(features)
        outlier_scores = model.negative_outlier_factor_

        outlier_scores = (outlier_scores.max() - outlier_scores) / (outlier_scores.max() - outlier_scores.min())

        dataframe = {
            "outlier_scores": outlier_scores,
            "outlier": outliers == -1
        }
        return pd.DataFrame(dataframe)
