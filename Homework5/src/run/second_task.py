from dataclasses import dataclass
import pandas as pd
import typing as t
import numpy as np
from src.data_loader import dataset_wine
from src.run.first_task import run_univariate, run_multivariate


def main():
    #
    # Note for devs:
    #
    #   Unselected features are commented out, in case
    #   they might be needed for later
    #
    features, labels = dataset_wine()
    columns = [
        # "Alcohol",
        # "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        # "Total phenols",
        # "Flavanoids",
        # "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        # "Hue",
        # "OD280/OD315 of diluted wines",
        "Proline",
    ]
    autoencoder_model_state_dict_path = "second_task_model.pt"

    univariate_outliers = run_univariate(features=features, labels=labels)
    multivariate_outliers = run_multivariate(
        features=features,
        labels=labels,
        columns=columns,
        autoencoder_model_state_dict_path=autoencoder_model_state_dict_path,
    )

    confusion_results_isolation_forest = build_confusion_results(
        multivariate_outliers["isolation_forest_outliers"], labels
    )
    confusion_results_autoencoder = build_confusion_results(
        multivariate_outliers["autoencoder_outliers"], labels
    )
    confusion_results_local_outlier_factor = build_confusion_results(
        multivariate_outliers["local_outlier_factor_outliers"],
        labels,
    )

    isolation_forest_precision = precision(confusion_results_isolation_forest)
    isolation_forest_recall = recall(confusion_results_isolation_forest)
    autoencoder_precision = precision(confusion_results_autoencoder)
    autoencoder_recall = recall(confusion_results_autoencoder)
    local_outlier_factor_precision = precision(confusion_results_local_outlier_factor)
    local_outlier_factor_recall = recall(confusion_results_local_outlier_factor)

    print(f"Isolation forest precision: {isolation_forest_precision:.2f}")
    print(f"Isolation forest recall: {isolation_forest_recall:.2f}")
    print(f"Autoencoder precision: {autoencoder_precision:.2f}")
    print(f"Autoencoder recall: {autoencoder_recall:.2f}")
    print(f"Local outlier factor precision: {local_outlier_factor_precision:.2f}")
    print(f"Local outlier factor recall: {local_outlier_factor_recall:.2f}")


@dataclass
class ConfusionResults:
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


def build_confusion_results(
    y: np.ndarray[t.Literal["N", 1], int], y_hat: np.ndarray[t.Literal["N", 1], int]
):
    confusion_results = ConfusionResults()

    for y_entry, y_hat_entry in zip(y, y_hat):
        #
        # Note for devs:
        #
        #   1 = outlier
        #   0 = inlier
        #
        #   Thus, true positives are instances when y_entry == 0, and y_hat_entry == 0.
        #   The other values can be inferred.
        #
        confusion_result_entry = (y_entry[0] == 0, y_hat_entry[0] == 0)

        if confusion_result_entry == (True, True):
            confusion_results.true_positives += 1
        elif confusion_result_entry == (False, False):
            confusion_results.true_negatives += 1
        elif confusion_result_entry == (False, True):
            confusion_results.false_positives += 1
        elif confusion_result_entry == (True, False):
            confusion_results.false_negatives += 1

    return confusion_results


def precision(confusion_results: ConfusionResults):
    return confusion_results.true_positives / (
        confusion_results.true_positives + confusion_results.false_positives
    )


def recall(confusion_results: ConfusionResults):
    return confusion_results.true_positives / (
        confusion_results.true_positives + confusion_results.false_negatives
    )


if __name__ == "__main__":
    main()
