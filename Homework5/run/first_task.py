from Homework5.multivariate.isolation_forest import IsolationForestOutlierDetector
from Homework5.multivariate.autoencoder import AutoencoderOutlierDetector
from Homework5.data_loader import dataset_obesity
from Homework5.data_utils import normalize
from Homework5.visualize import pair_plot
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def run_isolation_forest(features: np.array, features_to_train: np.array):
    isolation_forest_algorithm = IsolationForestOutlierDetector(contamination=0.04)
    isolation_forest_algorithm.build(features_to_train)
    outliers = isolation_forest_algorithm.predict(features_to_train)
    # The closer the point is to the root of the decission tree, the more easily it was to be separated from the rest
    # Thus, it is more likely to be an outlier, negative scores indicate outliers, and negative numbers indicate detected outliers

    outlier_scores = outliers["outlier_scores"].to_numpy()
    outlier_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(outlier_scores.reshape(-1, 1))
    outlier_scores = 1 - outlier_scores
    outlier_scores = outlier_scores.reshape(-1)

    pair_plot(features, outlier_scores, ["Weight", "Height", "Age", "FAF", "Male"])


def run_autoencoder(features: np.array, features_to_train: np.array):
    # This is returning some strange inconsistent results
    # TODO: Investigate

    autoencoder_algorithm = AutoencoderOutlierDetector()
    autoencoder_algorithm.build(features_to_train)
    outliers = autoencoder_algorithm.predict(features_to_train)

    outlier_scores = outliers["outlier_scores"].to_numpy()
    outlier_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(outlier_scores.reshape(-1, 1))
    outlier_scores = outlier_scores.reshape(-1)

    pair_plot(features, outlier_scores, ["Weight", "Height", "Age", "FAF", "Male"])


def run_multivariate():
    features, labels = dataset_obesity()
    features_to_train = normalize(features.to_numpy())
    run_isolation_forest(features, features_to_train)
    #run_autoencoder(features, features_to_train)

    print("Da")


if __name__ == "__main__":
    run_multivariate()
