import json
import numpy as np
from datetime import datetime

from sklearn.cluster import HDBSCAN
from umap import UMAP

import ai.database as db
from ai import paths
from ai.language_model import openai_client
from ai.whisper import transcribe


np.random.seed(42)


def clustering(embeddings, min_cluster_size=3, reduced_dimensions=5):
    """
    Cluster embeddings with optional UMAP reduction first.
    Returns one label per embedding.
    """
    if len(embeddings) == 0:
        return np.array([])

    # If there are too few points, just mark all as noise
    if len(embeddings) < min_cluster_size:
        return np.array([-1] * len(embeddings))

    if reduced_dimensions > 0:
        umap = UMAP(
            n_neighbors=2,
            n_components=reduced_dimensions,
            metric="cosine",
            random_state=42,
        )
        reduced_embeddings = umap.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings

    print("Clustering...")

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )

    return clusterer.fit_predict(reduced_embeddings)


def group_windows(window_data, labels):
    """
    Group window objects by cluster label.
    """
    clusters = {}

    for window, label in zip(window_data, labels):
        key = str(label)

        clusters.setdefault(key, {
            "windows": [],
        })

        clusters[key]["windows"].append(window)

    return clusters


def generate_categories(clusters, llm_client):
    countdown = 0
    categories = {}
    category_names = []

    print("Generating category names and summaries...")

    for label, win in clusters.items():
        if label == "-1":
            win["summary"] = "Uncategorized windows."
            categories["Uncategorized"] = win
            continue

        countdown += 1
        print(f"{countdown}/{len(clusters)}")

        window_texts = [w["text"] for w in win["windows"]]

        category_name = llm_client.get_category(window_texts, category_names)
        category_names.append(category_name)

        categories[category_name] = win
        win["summary"] = llm_client.get_summary(window_texts)

    return categories


def process_file(audio_file, author, category):
    """
    Main pipeline:
      1) Load existing DB
      2) Add metadata
      3) Transcribe audio into text windows
      4) Save transcription immediately
      5) Create embeddings
      6) Merge old windows from same top-level category
      7) Cluster
      8) Name/summarize clusters
      9) Save full DB + categories.json
    """
    database = db.load(paths.DATABASE)

    metadata = {
        "audio_file": audio_file,
        "author": author,
        "category": category,
        "date": datetime.today().strftime("%Y-%m-%d"),
    }

    database = db.add_metadata(database, metadata["category"], metadata)

    windows = transcribe(metadata["audio_file"], database, metadata)

    # Save right after transcription so it is not lost if later steps fail
    db.save(database, paths.DATABASE)

    if not windows:
        print("No transcription windows produced.")
        return

    llm_client = openai_client()

    new_embeddings = llm_client.get_embeddings(windows)

    window_data = [
        {
            "text": text,
            "embedding": embedding.tolist(),
            "source_file": metadata["audio_file"],
            "author": metadata["author"],
            "date": metadata["date"],
        }
        for text, embedding in zip(windows, new_embeddings)
    ]

    # Add previous windows from this same top-level category
    for existing_window_list in db.get_existing_windows(database, metadata["category"]):
        window_data += existing_window_list

    embeddings = [w["embedding"] for w in window_data]

    labels = clustering(embeddings)

    clusters = group_windows(window_data, labels)
    print("Clustering done.")

    categories = generate_categories(clusters, llm_client)

    database.setdefault("data", {})
    database["data"][metadata["category"]] = categories

    print("Saving database...")
    db.save(database, paths.DATABASE)

    # Human-readable output without embeddings
    categories_no_embeddings = json.loads(json.dumps(categories))
    for cluster in categories_no_embeddings.values():
        for window in cluster["windows"]:
            window.pop("embedding", None)

    with open("./data/categories.json", "w", encoding="utf-8") as f:
        json.dump(categories_no_embeddings, f, ensure_ascii=False, indent=2)

    print("Done!")