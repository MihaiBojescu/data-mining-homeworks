import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def __pair_plot(data_frame: pd.DataFrame, outliers: np.array, columns: list[str], pair_plot_title: str, digitizer: np.array):
    outliers = np.digitize(outliers, digitizer)

    assert len(digitizer) == 6 or len(digitizer) == 2

    if len(digitizer) == 6:
        palette = sns.color_palette("coolwarm")
    else:
        palette = ["#6788ee", "#e4725b"]

    data_frame["OutlierScore"] = outliers.tolist()
    pp = sns.pairplot(data_frame, vars=columns, hue="OutlierScore", palette=palette)
    pp.fig.suptitle(pair_plot_title)
    plt.show()


def pair_plot_6_bins(data_frame: pd.DataFrame, outliers: np.array, columns: list[str], pair_plot_title: str):
    __pair_plot(data_frame, outliers, columns, pair_plot_title, np.array([0.0, 0.18, 0.36, 0.54, 0.72, 1]))


def pair_plot_2_bins(data_frame: pd.DataFrame, outliers: np.array, columns: list[str], pair_plot_title: str):
    __pair_plot(data_frame, outliers, columns, pair_plot_title, np.array([0.0, 1]))
