import numpy as np
from sklearn.cluster import HDBSCAN
from umap import UMAP


np.random.seed(42)


def cluster_embeddings(embeddings, min_cluster_size: int = 3, reduced_dimensions: int = 5):
    """Cluster embedding vectors and return one label per input vector.

    Notes:
    - Returns `-1` labels for small/noisy inputs that cannot form a stable cluster.
    - Uses UMAP first (optional) to denoise high-dimensional vectors before HDBSCAN.
    """
    if len(embeddings) == 0:
        return np.array([])

    if len(embeddings) < min_cluster_size:
        return np.array([-1] * len(embeddings))

    embedding_array = np.asarray(embeddings)

    if reduced_dimensions > 0:
        # Keep UMAP parameters safe for very small batches.
        n_neighbors = max(2, min(15, len(embedding_array) - 1))
        reducer = UMAP(
            n_neighbors=n_neighbors,
            n_components=reduced_dimensions,
            metric="cosine",
            random_state=42,
        )
        embedding_array = reducer.fit_transform(embedding_array)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )

    return clusterer.fit_predict(embedding_array)


def group_windows_by_label(window_data: list, labels) -> dict:
    """Group transcript windows by their cluster label."""
    clusters = {}

    for window, label in zip(window_data, labels):
        cluster_id = str(label)
        clusters.setdefault(cluster_id, {"windows": []})
        clusters[cluster_id]["windows"].append(window)

    return clusters
