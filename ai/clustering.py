import logging

import numpy as np
from sklearn.cluster import HDBSCAN
from umap import UMAP


np.random.seed(42)
logger = logging.getLogger(__name__)


def cluster_embeddings(embeddings, min_cluster_size: int = 3, reduced_dimensions: int = 5):
    """Cluster embedding vectors and return one label per input vector.

    Notes:
    - Returns `-1` labels for small/noisy inputs that cannot form a stable cluster.
    - Uses UMAP first (optional) to denoise high-dimensional vectors before HDBSCAN.
    """
    total_count = len(embeddings)
    if total_count == 0:
        return np.array([])

    if total_count < min_cluster_size:
        return np.array([-1] * total_count)

    # Start pessimistic: anything malformed/unstable stays uncategorized (-1).
    labels = np.array([-1] * total_count)

    valid_indices = []
    valid_vectors = []
    expected_dim = None
    for index, embedding in enumerate(embeddings):
        vector = np.asarray(embedding, dtype=float).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            continue
        if expected_dim is None:
            expected_dim = vector.size
        if vector.size != expected_dim:
            # Mixed vector dimensions are treated as invalid for this clustering pass.
            continue
        valid_indices.append(index)
        valid_vectors.append(vector)

    if len(valid_vectors) < min_cluster_size:
        return labels

    embedding_array = np.vstack(valid_vectors)

    if reduced_dimensions > 0:
        # UMAP can fail on tiny/disconnected manifolds; use it only when input is
        # large enough and fall back to raw vectors on any reduction error.
        use_umap = embedding_array.shape[0] >= max(4, min_cluster_size + 1)
        if use_umap:
            n_neighbors = max(2, min(15, embedding_array.shape[0] - 1))
            n_components = max(2, min(reduced_dimensions, embedding_array.shape[1]))
            reducer = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric="cosine",
                random_state=42,
            )
            try:
                reduced = reducer.fit_transform(embedding_array)
                # Some edge-cases return empty/reduced-invalid arrays; keep raw vectors then.
                if np.asarray(reduced).size > 0 and reduced.shape[0] == embedding_array.shape[0]:
                    embedding_array = reduced
            except Exception as exc:  # noqa: BLE001
                logger.warning("UMAP reduction failed; falling back to raw embeddings. Error: %s", exc)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    try:
        predicted = clusterer.fit_predict(embedding_array)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HDBSCAN failed; returning noise labels. Error: %s", exc)
        return labels

    for position, label in zip(valid_indices, predicted):
        labels[position] = int(label)
    return labels


def group_windows_by_label(window_data: list, labels) -> dict:
    """Group transcript windows by their cluster label."""
    clusters = {}

    for window, label in zip(window_data, labels):
        cluster_id = str(label)
        clusters.setdefault(cluster_id, {"windows": []})
        clusters[cluster_id]["windows"].append(window)

    return clusters
