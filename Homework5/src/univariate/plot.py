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

    fig = plt.figure()
    fig.suptitle(title)
    fig.set_figheight(12)
    fig.set_figwidth(8)

    gs = gridspec.GridSpec(5, 2, figure=fig, wspace=0.5, hspace=0.5)

    axs1 = plt.subplot(gs[0, :])
    axs1.set_title("Boxplot")
    axs1.boxplot(x=data, vert=False)

    axs2 = plt.subplot(gs[1, :])
    axs2.set_title("Histogram: inliers vs outliers")
    axs2.hist(x=inliers, color="blue")
    axs2.hist(x=outliers, color="red")

    axs3 = plt.subplot(gs[2, :])
    axs3.set_title("Plot on axis: inliers vs outliers")
    axs3.scatter(x=inliers, y=[1 for _ in range(len(inliers))], c="blue")
    axs3.scatter(x=outliers, y=[1 for _ in range(len(outliers))], c="red")

    axs4 = plt.subplot(gs[3:, 0])
    axs4.set_title("Inliers vs outliers pie plot")
    axs4.pie(
        x=[
            len(inliers) / len(data),
            len(outliers) / len(data),
        ],
        labels=[
            f"Inliers ({len(inliers) / len(data) * 100:.2f}%)",
            f"Outliers ({len(outliers) / len(data) * 100:.2f}%)",
        ],
        colors=["blue", "red"],
    )

    axs5 = plt.subplot(gs[3:, 1])
    axs5.set_title("Inliers + outliers = total")
    axs5.text(0.5, 0.35, f"{len(inliers)}\n+ {len(outliers)}\n= {len(data)}", fontsize=23, ha='center')

    fig.show()
