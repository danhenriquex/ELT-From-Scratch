import numpy as np


def prepare_data(embeddings):
    """
    Prepares the data by associating image embeddings with syndromes.
    Args:
        embeddings (dict): A dictionary with syndromes as keys and embedding vectors as values.
    Returns:
        dict: Processed final_emb dictionary with syndrome keys and associated image embeddings.
    """

    final_emb = {}

    for syndrome in embeddings.keys():
        img_emb_list = []
        for key in embeddings[syndrome].keys():
            for images in embeddings[syndrome][key]:
                subject_emb_list = []
                for img_emb in embeddings[syndrome][key][images]:
                    subject_emb_list.append(img_emb)
                img_emb_list.append(subject_emb_list)

        final_emb[syndrome] = img_emb_list

    return final_emb


def prepare_embeddings_and_labels(final_emb):
    """
    Prepares embeddings and labels from the given final_emb dictionary.

    Args:
        final_emb (dict): A dictionary where keys are syndrome labels and values are lists of embeddings.

    Returns:
        tuple: A tuple containing two numpy arrays:
           - embeddings_value: The array of embeddings.
           - syndromes_id: The array of syndrome labels.
    """
    embeddings_value = []
    syndromes_id = []

    for syndrome, vectors in final_emb.items():
        for vector in vectors:
            embeddings_value.append(vector)
            syndromes_id.append(syndrome)

    embeddings_value = np.array(embeddings_value)
    syndromes_id = np.array(syndromes_id)

    return embeddings_value, syndromes_id
