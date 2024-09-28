import numpy as np
from models.knn_classifier import knn_classify
from sklearn.neighbors import KNeighborsClassifier


def test_knn_classify():
    train_data = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
    train_labels = np.array([0, 0, 1, 1])

    test_data = np.array([[1.5, 1]])

    k = 2
    expected_output = np.array([0])

    result = knn_classify(train_data, train_labels, test_data, k=k, metric="cosine")

    assert np.array_equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"

    result_euclidean = knn_classify(
        train_data, train_labels, test_data, k=k, metric="euclidean"
    )
    assert np.array_equal(
        result_euclidean, expected_output
    ), f"Expected {expected_output}, but got {result_euclidean}"
