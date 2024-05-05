import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestOutlierDetector:
    def __init__(self, contamination: float | str = "auto"):
        self.__model = IsolationForest(contamination=contamination)

    def build(self, features: np.array):
        self.__model = self.__model.fit(features)

    def predict(self, features: np.array) -> pd.DataFrame:
        dataframe = {
            "outlier_scores": self.__model.decision_function(features),
            "outlier": self.__model.predict(features) == -1
        }
        return pd.DataFrame(dataframe)
