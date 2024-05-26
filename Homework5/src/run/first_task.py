import matplotlib.pyplot as plt
import pandas as pd

from src.univariate.mean_k_sd import get_outliers_mean_k_sd
from src.univariate.k_iqr import get_outliers_k_iqr
from src.univariate.plot import plot_univariate
from src.univariate.normalise import normalise
from src.multivariate.isolation_forest import IsolationForestOutlierDetector
from src.multivariate.autoencoder import AutoencoderOutlierDetector
from src.multivariate.local_outlier_factor import (
    LocalOutlierFactorOutlierDetector,
)
from src.data_loader import dataset_obesity
from src.data_utils import normalize
from src.visualize import pair_plot_6_bins, pair_plot_2_bins
from src.combine_predictions import combine_predictions
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np


def main():
    features, labels = dataset_obesity()
    columns = ["Weight", "Height", "Age", "FAF", "Male"]
    autoencoder_model_state_dict_path = "first_task_model.pt"

    run_univariate(features=features, labels=labels)
    run_multivariate(
        features=features,
        labels=labels,
        columns=columns,
        autoencoder_model_state_dict_path=autoencoder_model_state_dict_path,
    )


def run_univariate(features: pd.DataFrame, labels: np.array):
    normalised_features = [normalise(features[feature]) for feature in features]
    outliers_results = {}

    for i, feature in enumerate(features):
        outliers_mean_2_sd = get_outliers_mean_k_sd(data=features[feature], k=2)
        outliers_mean_2_sd_normalised = get_outliers_mean_k_sd(
            data=normalised_features[i], k=2
        )
        outliers_mean_3_sd = get_outliers_mean_k_sd(data=features[feature], k=3)
        outliers_mean_3_sd_normalised = get_outliers_mean_k_sd(
            data=normalised_features[i], k=3
        )
        outliers_1_5_iqr = get_outliers_k_iqr(data=features[feature], k_iqr=1.5)
        outliers_1_5_iqr_normalised = get_outliers_k_iqr(
            data=normalised_features[i], k_iqr=1.5
        )

        plot_univariate(
            title=f"{feature} outliers: Mean +/- 2*sd",
            data=features[feature],
            outliers=outliers_mean_2_sd,
        )
        plot_univariate(
            title=f"{feature} outliers (normalised): Mean +/- 2*sd",
            data=normalised_features[i],
            outliers=outliers_mean_2_sd_normalised,
        )
        plot_univariate(
            title=f"{feature} outliers: Mean +/- 3*sd",
            data=features[feature],
            outliers=outliers_mean_3_sd,
        )
        plot_univariate(
            title=f"{feature} outliers (normalised): Mean +/- 3*sd",
            data=normalised_features[i],
            outliers=outliers_mean_3_sd_normalised,
        )
        plot_univariate(
            title=f"{feature} outliers: 1.5 IQR",
            data=features[feature],
            outliers=outliers_1_5_iqr,
        )
        plot_univariate(
            title=f"{feature} outliers (normalised): 1.5 IQR",
            data=normalised_features[i],
            outliers=outliers_1_5_iqr_normalised,
        )

        outliers_results[feature] = {
            "outliers_mean_2_sd": outliers_mean_2_sd,
            "outliers_mean_2_sd_normalised": outliers_mean_2_sd_normalised,
            "outliers_mean_3_sd": outliers_mean_3_sd,
            "outliers_mean_3_sd_normalised": outliers_mean_3_sd_normalised,
            "outliers_1_5_iqr": outliers_1_5_iqr,
            "outliers_1_5_iqr_normalised": outliers_1_5_iqr_normalised,
        }

    return outliers_results


def __log_smooth_distribution(outlier_scores: np.array, deg: int):
    outlier_scores = MinMaxScaler(feature_range=(1, deg)).fit_transform(
        outlier_scores.reshape(-1, 1)
    )
    outlier_scores = np.log2(outlier_scores)
    outlier_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        outlier_scores.reshape(-1, 1)
    )
    return outlier_scores.reshape(-1)


def run_isolation_forest(
    features: np.array, features_to_train: np.array, columns: list[str]
) -> tuple[np.array, float]:
    isolation_forest_algorithm = IsolationForestOutlierDetector(contamination=0.05)
    isolation_forest_algorithm.build(features_to_train)
    outliers = isolation_forest_algorithm.predict(features_to_train)
    # The closer the point is to the root of the decission tree, the more easily it was to be separated from the rest
    # Thus, it is more likely to be an outlier, negative scores indicate outliers, and negative numbers indicate detected outliers

    outlier_scores = outliers["outlier_scores"].to_numpy()
    outlier_score_threshold = np.sort(outlier_scores)[-int(len(outlier_scores) * 0.03)]

    pair_plot_6_bins(
        features,
        outlier_scores,
        columns,
        "Isolation Forest Adjusted Scores",
    )

    pair_plot_2_bins(
        features,
        (outlier_scores > outlier_score_threshold).astype(float),
        columns,
        "Isolation Forest Outliers",
    )

    return outliers, outlier_scores, outlier_score_threshold


def run_autoencoder(
    features: np.array,
    features_to_train: np.array,
    columns: list[str],
    model_state_dict_path: str,
) -> tuple[np.array, float]:

    autoencoder_algorithm = AutoencoderOutlierDetector()
    autoencoder_algorithm.build(features_to_train, model_state_dict_path)
    outliers = autoencoder_algorithm.predict(features_to_train)

    outlier_scores = outliers["outlier_scores"].to_numpy()
    outlier_score_threshold = np.sort(outlier_scores)[-int(len(outlier_scores) * 0.04)]

    smooth_outlier_scores = __log_smooth_distribution(outlier_scores, 8)

    pair_plot_6_bins(
        features,
        smooth_outlier_scores,
        columns,
        "Autoencoder Adjusted Scores",
    )

    pair_plot_2_bins(
        features,
        (outlier_scores > outlier_score_threshold).astype(float),
        columns,
        "Autoencoder Outliers",
    )

    return outliers, outlier_scores, outlier_score_threshold


def run_local_outlier_factor(
    features: np.array, features_to_train: np.array, columns: list[str]
) -> tuple[np.array, float]:

    local_outlier_factor_algorithm = LocalOutlierFactorOutlierDetector(
        contamination=0.05
    )
    local_outlier_factor_algorithm.build(features_to_train)
    outliers = local_outlier_factor_algorithm.predict(features_to_train)

    outlier_scores = outliers["outlier_scores"].to_numpy()
    outlier_score_threshold = np.sort(outlier_scores)[-int(len(outlier_scores) * 0.02)]

    smooth_outlier_scores = __log_smooth_distribution(outlier_scores, 64)

    pair_plot_6_bins(
        features,
        smooth_outlier_scores,
        columns,
        "Local Outlier Factor Adjusted Scores",
    )

    pair_plot_2_bins(
        features,
        (outlier_scores > outlier_score_threshold).astype(float),
        columns,
        "Local Outlier Factor Outliers",
    )

    return outliers, outlier_scores, outlier_score_threshold


def run_multivariate(
    features: pd.DataFrame,
    labels: np.array,
    columns: list[str],
    autoencoder_model_state_dict_path: str,
):
    features_to_train = normalize(features.to_numpy())

    isolation_forest_outliers, isolation_forest_results, isolation_forest_threshold = (
        run_isolation_forest(features, features_to_train, columns)
    )
    autoencoder_outliers, autoencoder_results, autoencoder_threshold = run_autoencoder(
        features, features_to_train, columns, autoencoder_model_state_dict_path
    )
    (
        local_outlier_factor_outliers,
        local_outlier_factor_results,
        local_outlier_factor_threshold,
    ) = run_local_outlier_factor(features, features_to_train, columns)

    combined_results = combine_predictions(
        [
            isolation_forest_results > isolation_forest_threshold,
            autoencoder_results > autoencoder_threshold,
            local_outlier_factor_results > local_outlier_factor_threshold,
        ],
        2,
    )

    pair_plot_2_bins(
        features,
        combined_results,
        columns,
        "Combined results outliers",
    )

    thresholds_metrics = {
        "values": [
            isolation_forest_threshold,
            autoencoder_threshold,
            local_outlier_factor_threshold,
        ],
        "names": [
            "Isolation forest threshold (Scaled)",
            "Autoencoder threshold (Scaled)",
            "Local outlier factor threshold (Scaled)",
        ],
    }

    ax = sns.barplot(thresholds_metrics, x="names", y="values")
    ax.set_title("Thresholds")

    plt.xticks(rotation=45)
    ax.figure.tight_layout()
    plt.show()

    return {
        "isolation_forest_outliers": isolation_forest_outliers,
        "autoencoder_outliers": autoencoder_outliers,
        "local_outlier_factor_outliers": local_outlier_factor_outliers,
    }


if __name__ == "__main__":
    main()
