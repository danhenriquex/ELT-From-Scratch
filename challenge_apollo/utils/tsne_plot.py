import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_tsne_plots(final_emb, perplexities, output_dir="challenge_apollo/results"):
    """
    Generates and saves t-SNE plots for different perplexity values.

    Args:
      final_emb (dict): A dictionary where keys are syndrome labels and values are lists of embeddings.
      perplexities (list): A list of perplexity values to use for t-SNE.
      output_dir (str): Directory to save the PNG files (default: 'tsne_plots').
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    syndromes = []
    embeddings = []

    for syndrome, embedding_list in final_emb.items():
        syndromes.extend([syndrome] * len(embedding_list))
        embeddings.extend(embedding_list)

    embeddings = np.array(embeddings)

    syndrome_colors = {
        syndrome: plt.cm.jet(i / len(final_emb))
        for i, syndrome in enumerate(final_emb.keys())
    }

    for perplexity in perplexities:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))

        for i, syndrome in enumerate(syndromes):
            plt.scatter(
                embeddings_2d[i, 0],
                embeddings_2d[i, 1],
                color=syndrome_colors[syndrome],
                label=syndrome,
                alpha=0.7,
            )

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                label=syndrome,
                markersize=8,
                linestyle="",
            )
            for syndrome, color in syndrome_colors.items()
        ]
        plt.legend(handles=handles, loc="best")

        plt.title(f"t-SNE Visualization (Perplexity={perplexity})")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        output_file = f"{output_dir}/tsne_perplexity_{perplexity}.png"
        plt.savefig(output_file)
        plt.close()

        print(f"Saved t-SNE plot for perplexity {perplexity} as {output_file}")


def plot_tsne_2d(data):
    syndromes = []
    embeddings = []

    for syndrome, embedding_list in data.items():
        syndromes.extend([syndrome] * len(embedding_list))
        embeddings.extend(embedding_list)

    embeddings = np.array(embeddings)

    n_samples, n_features = embeddings.shape
    n_components = min(n_samples, n_features)

    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)

    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=0)
    embeddings_2d = tsne_2d.fit_transform(embeddings_reduced)

    syndrome_colors = {
        syndrome: plt.cm.jet(i / len(data)) for i, syndrome in enumerate(data.keys())
    }

    plt.figure(figsize=(8, 6))
    for i, syndrome in enumerate(syndromes):
        plt.scatter(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            color=syndrome_colors[syndrome],
            label=syndrome,
            alpha=0.7,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=syndrome,
            markersize=8,
            linestyle="",
        )
        for syndrome, color in syndrome_colors.items()
    ]
    plt.legend(handles=handles)

    plt.title("t-SNE Visualization of Syndromes (2D)")
    plt.savefig(
        "challenge_apollo/results/tsne_plot_2d.png", format="png", bbox_inches="tight"
    )


def plot_tsne_3d(data):
    syndromes = []
    embeddings = []

    for syndrome, embedding_list in data.items():
        syndromes.extend([syndrome] * len(embedding_list))
        embeddings.extend(embedding_list)

    embeddings = np.array(embeddings)

    n_samples, n_features = embeddings.shape
    n_components = min(n_samples, n_features)

    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)

    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=0)
    embeddings_3d = tsne_3d.fit_transform(embeddings_reduced)

    syndrome_colors = {
        syndrome: plt.cm.jet(i / len(data)) for i, syndrome in enumerate(data.keys())
    }

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, syndrome in enumerate(syndromes):
        ax.scatter(
            embeddings_3d[i, 0],
            embeddings_3d[i, 1],
            embeddings_3d[i, 2],
            color=syndrome_colors[syndrome],
            label=syndrome,
            alpha=0.7,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=syndrome,
            markersize=8,
            linestyle="",
        )
        for syndrome, color in syndrome_colors.items()
    ]
    ax.legend(handles=handles)

    plt.title("t-SNE Visualization of Syndromes (3D)")

    plt.savefig(
        "challenge_apollo/results/tsne_plot_3d.png", format="png", bbox_inches="tight"
    )
