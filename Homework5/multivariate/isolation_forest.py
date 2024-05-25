import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


class IsolationForestOutlierDetector:
    def __init__(self, contamination: float | str = "auto"):
        self.__model = IsolationForest(contamination=contamination, random_state=24)

    def build(self, features: np.array):
        self.__model = self.__model.fit(features)

    def predict(self, features: np.array) -> pd.DataFrame:

        outlier_scores = self.__model.decision_function(features)
        outlier_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(outlier_scores.reshape(-1, 1))
        outlier_scores = 1 - outlier_scores
        outlier_scores = outlier_scores.reshape(-1)

        dataframe = {
            "outlier_scores": outlier_scores,
            "outlier": self.__model.predict(features) == -1
        }
        return pd.DataFrame(dataframe)
