import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def perform_cross_validation(embeddings_value, syndromes_id, k_neighbors=5):
    """
    Perform cross-validation on the given embeddings and syndromes.

    Args:
        embeddings_value (numpy.ndarray): The array of embeddings.
        syndromes_id (numpy.ndarray): The array of syndrome labels.
        k_neighbors (int): The number of neighbors to consider.

    Returns:
        tuple: A tuple containing two lists:
                - cosine_accuracies: The list of cosine accuracies.
                - euclidean_accuracies: The list of euclidean accuracies.
    """

    le = LabelEncoder()
    syndromes_id_encoded = le.fit_transform(syndromes_id)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    cosine_accuracies = []
    euclidean_accuracies = []

    k = k_neighbors

    for train_index, test_index in kf.split(embeddings_value):
        X_train, X_test = embeddings_value[train_index], embeddings_value[test_index]
        y_train, y_test = (
            syndromes_id_encoded[train_index],
            syndromes_id_encoded[test_index],
        )

        knn_cosine = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn_cosine.fit(X_train, y_train)
        cosine_accuracy = knn_cosine.score(X_test, y_test)
        cosine_accuracies.append(cosine_accuracy)

        knn_euclidean = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn_euclidean.fit(X_train, y_train)
        euclidean_accuracy = knn_euclidean.score(X_test, y_test)
        euclidean_accuracies.append(euclidean_accuracy)

    return cosine_accuracies, euclidean_accuracies
