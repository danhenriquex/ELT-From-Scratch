import numpy as np
from scipy.spatial.distance import cosine, euclidean


def compute_cosine_distance(vector, gallery_vectors):
    return np.array([cosine(vector, g_vector) for g_vector in gallery_vectors])


def compute_euclidean_distance(vector, gallery_vectors):
    return np.array([euclidean(vector, g_vector) for g_vector in gallery_vectors])
