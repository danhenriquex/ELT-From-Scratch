import pickle

import hydra
from data.pre_processing import prepare_data, prepare_embeddings_and_labels
from models.cross_validation import perform_cross_validation
from models.metrics import compute_knn_metrics
from omegaconf import DictConfig
from utils.plot_auc_roc import plot_roc_auc
from utils.save_to_pdf import save_table_to_pdf
from utils.tsne_plot import save_tsne_plots


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    with open("challenge_apollo/data/mini_gm_public_v0.1.p", "rb") as f:
        embeddings = pickle.load(f)

    k_values = cfg.model.knn.k_neighbors
    perplexities = cfg.manifold.tsne.perplexity

    final_emb = prepare_data(embeddings)

    embeddings_value, syndromes_id = prepare_embeddings_and_labels(final_emb)

    cosine_accuracies, euclidean_accuracies = perform_cross_validation(
        embeddings_value, syndromes_id, k_neighbors=k_values[3]
    )

    print("Cosine Metrics: ", cosine_accuracies)
    print("Euclidean Metrics: ", euclidean_accuracies)

    cosine_metrics, euclidean_metrics = compute_knn_metrics(
        embeddings_value, syndromes_id, k_values
    )

    plot_roc_auc(cosine_metrics, euclidean_metrics, k_values)

    save_tsne_plots(final_emb, perplexities=perplexities)

    save_table_to_pdf(cosine_metrics, euclidean_metrics, k_values)


if __name__ == "__main__":
    main()
