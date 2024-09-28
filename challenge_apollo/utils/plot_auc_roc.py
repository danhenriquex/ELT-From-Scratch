import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_roc_auc(cosine_metrics, euclidean_metrics, k_values):
    """
    Plot the ROC AUC curve for the given cosine and euclidean metrics.

    Args:
        cosine_metrics (dict): The dictionary of cosine metrics.
        euclidean_metrics (dict): The dictionary of euclidean metrics.
        k_values (list): The list of k values to consider.

    Returns:
        None
    """

    plt.figure(figsize=(10, 8))

    mean_fpr = np.linspace(0, 1, 100)

    for k in k_values:
        mean_tpr_cosine = np.zeros_like(mean_fpr)
        for fpr_cosine, tpr_cosine in cosine_metrics[k]["fpr_tpr"]:
            mean_tpr_cosine += np.interp(mean_fpr, fpr_cosine, tpr_cosine)
        mean_tpr_cosine /= len(cosine_metrics[k]["fpr_tpr"])
        roc_auc_cosine = auc(mean_fpr, mean_tpr_cosine)

        mean_tpr_euclidean = np.zeros_like(mean_fpr)
        for fpr_euclidean, tpr_euclidean in euclidean_metrics[k]["fpr_tpr"]:
            mean_tpr_euclidean += np.interp(mean_fpr, fpr_euclidean, tpr_euclidean)
        mean_tpr_euclidean /= len(euclidean_metrics[k]["fpr_tpr"])
        roc_auc_euclidean = auc(mean_fpr, mean_tpr_euclidean)

        plt.plot(
            mean_fpr,
            mean_tpr_cosine,
            label=f"Cosine ROC AUC (k={k}) = {roc_auc_cosine:.2f}",
        )
        plt.plot(
            mean_fpr,
            mean_tpr_euclidean,
            label=f"Euclidean ROC AUC (k={k}) = {roc_auc_euclidean:.2f}",
            linestyle="--",
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Comparison: Cosine vs Euclidean KNN")
    plt.legend(loc="lower right")
    plt.grid()

    # Save the plot as a PNG file
    plt.savefig("challenge_apollo/results/roc_auc_comparison.png")
