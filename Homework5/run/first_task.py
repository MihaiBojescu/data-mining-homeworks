import matplotlib.pyplot as plt

from Homework5.univariate.mean_k_sd import get_outliers_mean_k_sd
from Homework5.univariate.k_iqr import get_outliers_k_iqr
from Homework5.multivariate.isolation_forest import IsolationForestOutlierDetector
from Homework5.multivariate.autoencoder import AutoencoderOutlierDetector
from Homework5.multivariate.local_outlier_factor import LocalOutlierFactorOutlierDetector
from Homework5.data_loader import dataset_obesity
from Homework5.data_utils import normalize
from Homework5.visualize import pair_plot_6_bins, pair_plot_2_bins
from Homework5.combine_predictions import combine_predictions
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np



def run_univariate():
    features, labels = dataset_obesity()
    features_to_train = normalize(features.to_numpy())

    for feature in features:
        outliers_mean_k_sd = get_outliers_mean_k_sd(data=features[feature], k=3)
        inliers_mean_k_sd = features[feature][~np.in1d(features[feature], outliers_mean_k_sd)]
        outliers_k_iqr = get_outliers_k_iqr(data=features[feature], k_iqr=1.5)
        inliers_k_iqr = features[feature][~np.in1d(features[feature], outliers_k_iqr)]

def __log_smooth_distribution(outlier_scores: np.array, deg: int):
    outlier_scores = MinMaxScaler(feature_range=(1, deg)).fit_transform(outlier_scores.reshape(-1, 1))
    outlier_scores = np.log2(outlier_scores)
    outlier_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(outlier_scores.reshape(-1, 1))
    return outlier_scores.reshape(-1)


def run_isolation_forest(features: np.array, features_to_train: np.array) -> tuple[np.array, float]:
    isolation_forest_algorithm = IsolationForestOutlierDetector(contamination=0.05)
    isolation_forest_algorithm.build(features_to_train)
    outliers = isolation_forest_algorithm.predict(features_to_train)
    # The closer the point is to the root of the decission tree, the more easily it was to be separated from the rest
    # Thus, it is more likely to be an outlier, negative scores indicate outliers, and negative numbers indicate detected outliers

    outlier_scores = outliers["outlier_scores"].to_numpy()
    outlier_score_threshold = np.sort(outlier_scores)[-int(len(outlier_scores) * 0.03)]

    pair_plot_6_bins(features, outlier_scores, ["Weight", "Height", "Age", "FAF", "Male"], "Isolation Forest Adjusted Scores")

    pair_plot_2_bins(features, (outlier_scores > outlier_score_threshold).astype(float), ["Weight", "Height", "Age", "FAF", "Male"], "Isolation Forest Outliers")

    return outlier_scores, outlier_score_threshold


def run_autoencoder(features: np.array, features_to_train: np.array) -> tuple[np.array, float]:

    autoencoder_algorithm = AutoencoderOutlierDetector()
    autoencoder_algorithm.build(features_to_train)
    outliers = autoencoder_algorithm.predict(features_to_train)

    outlier_scores = outliers["outlier_scores"].to_numpy()
    outlier_score_threshold = np.sort(outlier_scores)[-int(len(outlier_scores) * 0.04)]

    smooth_outlier_scores = __log_smooth_distribution(outlier_scores, 8)

    pair_plot_6_bins(features, smooth_outlier_scores, ["Weight", "Height", "Age", "FAF", "Male"], "Autoencoder Adjusted Scores")

    pair_plot_2_bins(features, (outlier_scores > outlier_score_threshold).astype(float), ["Weight", "Height", "Age", "FAF", "Male"], "Autoencoder Outliers")

    return outlier_scores, outlier_score_threshold


def run_local_outlier_factor(features: np.array, features_to_train: np.array) -> tuple[np.array, float]:

    local_outlier_factor_algorithm = LocalOutlierFactorOutlierDetector(contamination=0.05)
    local_outlier_factor_algorithm.build(features_to_train)
    outliers = local_outlier_factor_algorithm.predict(features_to_train)

    outlier_scores = outliers["outlier_scores"].to_numpy()
    outlier_score_threshold = np.sort(outlier_scores)[-int(len(outlier_scores) * 0.02)]

    smooth_outlier_scores = __log_smooth_distribution(outlier_scores, 64)

    pair_plot_6_bins(features, smooth_outlier_scores, ["Weight", "Height", "Age", "FAF", "Male"], "Local Outlier Factor Adjusted Scores")

    pair_plot_2_bins(features, (outlier_scores > outlier_score_threshold).astype(float), ["Weight", "Height", "Age", "FAF", "Male"], "Local Outlier Factor Outliers")

    return outlier_scores, outlier_score_threshold


def run_multivariate():
    features, labels = dataset_obesity()
    features_to_train = normalize(features.to_numpy())

    isolation_forest_results, isolation_forest_threshold = run_isolation_forest(features, features_to_train)
    autoencoder_results, autoencoder_threshold = run_autoencoder(features, features_to_train)
    local_outlier_factor_results, local_outlier_factor_threshold = run_local_outlier_factor(features, features_to_train)

    combined_results = combine_predictions(
        [isolation_forest_results > isolation_forest_threshold,
         autoencoder_results > autoencoder_threshold,
         local_outlier_factor_results > local_outlier_factor_threshold
         ], 2)

    pair_plot_2_bins(features, combined_results,
                     ["Weight", "Height", "Age", "FAF", "Male"], "Combined results outliers")

    thresholds_metrics = {
        "values": [isolation_forest_threshold, autoencoder_threshold, local_outlier_factor_threshold],
        "names": ["Isolation forest threshold (Scaled)", "Autoencoder threshold (Scaled)", "Local outlier factor threshold (Scaled)"]
    }

    ax = sns.barplot(thresholds_metrics, x="names", y="values")
    ax.set_title("Thresholds")

    plt.xticks(rotation=45)
    ax.figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_univariate()
    run_multivariate()
