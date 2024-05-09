import os
import intrasom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from intrasom.visualization import PlotFactory
from intrasom.clustering import ClusterFactory
from pathlib import Path
from typing import Optional

PARENT_DIR = str(os.path.join(Path(__file__).parent.absolute()))


def train_som(data: np.array, map_size: Optional[tuple[int, int]] = None) -> intrasom.intrasom.SOM:
    if map_size is None:
        map_size_part = int(np.sqrt(5 * np.sqrt(data.shape[0])))
        map_size_part = map_size_part + (map_size_part % 2)
        map_size = (map_size_part, map_size_part)

    current_som = intrasom.SOMFactory.build(data,
                                            mapsize=map_size,
                                            mapshape='toroid',
                                            lattice='hexa',
                                            normalization='var',
                                            initialization="pca",
                                            neighborhood='gaussian',
                                            training='batch')

    current_som.train(train_len_factor=2, previous_epoch=True, save=False)
    return current_som


def plot_u_matrix(som: intrasom.intrasom.SOM, title: str, hits: bool = False):
    plot = PlotFactory(som)
    plot.plot_umatrix(figsize=(13, 2.5),
                      hits=hits,
                      title=title,
                      legend_title="Distance",
                      file_name="umatrix_hits",
                      file_path=os.path.join(PARENT_DIR, "dump_images"))

    load_and_print_image("umatrix_hits.jpg")


def plot_torus(som: intrasom.intrasom.SOM, title: str):
    plot = PlotFactory(som)
    plot.plot_torus(hits=True, inner_out_prop=0.25, red_factor=2)


def load_and_print_image(image_name: str):
    imp_path = os.path.join(PARENT_DIR, "dump_images", image_name)
    img = mpimg.imread(imp_path)
    plt.imshow(img)
    plt.show()
    os.remove(imp_path)


def k_means(som: intrasom.intrasom.SOM, nr_of_clusters: int) -> np.array:
    clustering = ClusterFactory(som)
    clusters = clustering.kmeans(k=nr_of_clusters)
    clustering.results_cluster(clusters)

    clustering.plot_kmeans(figsize=(12, 5),
                           clusters=clusters,
                           umatrix=True,
                           colormap="gist_rainbow",
                           alfa_clust=0.5,
                           legend_text_size=7,
                           cluster_outline=True,
                           plot_labels=True,
                           clusterout_maxtext_size=12,
                           save=True,
                           file_name="k_means",
                           file_path=os.path.join(PARENT_DIR, "dump_images"))

    load_and_print_image("k_means.jpg")

    result = clustering.results_cluster(clusters).to_numpy()
    return result[:, -1]
