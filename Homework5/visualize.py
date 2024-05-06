import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def pair_plot(data_frame: pd.DataFrame, outliers: np.array, columns: list[str]):
    outliers = np.digitize(outliers, np.array([0.0, 0.18, 0.36, 0.54, 0.72, 1]))

    palette = sns.color_palette("coolwarm")
    data_frame["OutlierScore"] = outliers.tolist()
    sns.pairplot(data_frame, vars=columns, hue="OutlierScore", palette=palette)
    plt.show()
