import typing as t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_univariate(
    title: str,
    data: np.ndarray[t.Literal["N"], float],
    outliers: np.ndarray[t.Literal["N"], float],
):
    inliers = data[~np.in1d(data, outliers)]
    sturges_bins = np.ceil(1 + np.log2(len(data))).astype(int)

    fig = plt.figure()
    fig.suptitle(title)
    fig.tight_layout()

    gs = gridspec.GridSpec(3, 1)


    axs1 = plt.subplot(gs[0, 0])
    axs1.set_title("Boxplot")
    axs1.boxplot(x=data, vert=False)

    axs2 = plt.subplot(gs[1, 0])
    axs2.set_title("Histogram: inliers vs outliers")
    axs2.hist(x=inliers, color="blue")
    axs2.hist(x=outliers, color="red")

    axs3 = plt.subplot(gs[2, 0])
    axs3.set_title("Plot on axis: inliers vs outliers")
    axs3.scatter(x=inliers, y=[1 for _ in range(len(inliers))], c="blue")
    axs3.scatter(x=outliers, y=[1 for _ in range(len(outliers))], c="red")

    fig.show()

    plt.waitforbuttonpress()
