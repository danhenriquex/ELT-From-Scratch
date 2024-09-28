import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize


def compute_knn_metrics(embeddings_value, syndromes_id, k_values):
    """
    Compute the KNN metrics for the given embeddings and syndromes.

    Args:
        embeddings_value (numpy.ndarray): The array of embeddings.
        syndromes_id (numpy.ndarray): The array of syndrome labels.
        k_values (list): The list of k values to consider.

    Returns:
        tuple: A tuple containing two dictionaries:
                - cosine_metrics: The dictionary of cosine metrics.
                - euclidean_metrics: The dictionary of euclidean metrics.
    """

    le = LabelEncoder()
    syndromes_encoded = le.fit_transform(syndromes_id)

    syndromes_binarized = label_binarize(
        syndromes_encoded, classes=np.unique(syndromes_encoded)
    )

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    cosine_metrics = {}
    euclidean_metrics = {}

    for k in k_values:
        cosine_auc = []
        euclidean_auc = []
        cosine_acc = []
        euclidean_acc = []
        cosine_fpr_tprs = []
        euclidean_fpr_tprs = []

        for train_index, test_index in kf.split(embeddings_value):
            X_train, X_test = (
                embeddings_value[train_index],
                embeddings_value[test_index],
            )
            y_train, y_test = (
                syndromes_encoded[train_index],
                syndromes_encoded[test_index],
            )
            y_test_binarized = syndromes_binarized[test_index]

            # Cosine KNN Classification
            knn_cosine = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn_cosine.fit(X_train, y_train)
            cosine_acc.append(knn_cosine.score(X_test, y_test))

            # Cosine ROC AUC
            y_pred_cosine = knn_cosine.predict_proba(X_test)
            cosine_auc.append(
                roc_auc_score(y_test_binarized, y_pred_cosine, multi_class="ovr")
            )

            # Compute FPR, TPR for ROC Curve
            fpr_cosine, tpr_cosine, _ = roc_curve(
                y_test_binarized.ravel(), y_pred_cosine.ravel()
            )
            cosine_fpr_tprs.append((fpr_cosine, tpr_cosine))

            # Euclidean KNN Classification
            knn_euclidean = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            knn_euclidean.fit(X_train, y_train)
            euclidean_acc.append(knn_euclidean.score(X_test, y_test))

            # Euclidean ROC AUC
            y_pred_euclidean = knn_euclidean.predict_proba(X_test)
            euclidean_auc.append(
                roc_auc_score(y_test_binarized, y_pred_euclidean, multi_class="ovr")
            )

            # Compute FPR, TPR for ROC Curve
            fpr_euclidean, tpr_euclidean, _ = roc_curve(
                y_test_binarized.ravel(), y_pred_euclidean.ravel()
            )
            euclidean_fpr_tprs.append((fpr_euclidean, tpr_euclidean))

        cosine_metrics[k] = {
            "accuracy": np.mean(cosine_acc),
            "roc_auc": np.mean(cosine_auc),
            "fpr_tpr": cosine_fpr_tprs,
        }
        euclidean_metrics[k] = {
            "accuracy": np.mean(euclidean_acc),
            "roc_auc": np.mean(euclidean_auc),
            "fpr_tpr": euclidean_fpr_tprs,
        }

    return cosine_metrics, euclidean_metrics


def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_labels, multi_class="ovr")
    return {"accuracy": accuracy, "AUC": auc}
