import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def pca_reduction(features: np.array):
    return PCA(n_components=2).fit_transform(features)


def plot_data(features: np.array, labels: np.array, title: str):
    assert len(labels.shape) == 1
    assert features.shape[1] >= 2

    if features.shape[1] > 2:
        print("Applying PCA to reduce plot_data features dimensionality")
        features = pca_reduction(features)

    features_df = pd.DataFrame(
        {"x_component": features[:, 0].tolist(),
         "y_component": features[:, 1].tolist(),
         "label": labels.tolist()}
    )

    sns.scatterplot(features_df, x="x_component", y="y_component", hue="label").set_title(title)
    plt.show()
    

def plot_contingency_matrix(ground_truth: np.array, predicted: np.array):
    cm = contingency_matrix(ground_truth, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()



