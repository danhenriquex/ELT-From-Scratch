from sklearn.neighbors import KNeighborsClassifier


def knn_classify(train_data, train_labels, test_data, k=5, metric="cosine"):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(train_data, train_labels)
    return knn.predict(test_data)
