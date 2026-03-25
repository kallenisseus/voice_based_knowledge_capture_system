"""End-to-end processing pipeline for newly uploaded audio.

Why this file exists:
- It is the orchestration layer for business flow.
- It calls specialized modules (`transcription`, `clustering`,
  `language_models`, `repository`) but does not own low-level details.

When changing behavior, start here first, then drill into the specific
module responsible for each step.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np
from django.utils import timezone

from ai.clustering import cluster_embeddings, group_windows_by_label
from ai.language_models import KnowledgeModelClient
from ai.repository import (
    add_segments_to_cluster,
    create_cluster,
    create_or_update_upload,
    get_category_clusters,
    get_category_segment_payloads,
    get_cluster_member_count,
    get_cluster_texts,
    remove_empty_category_clusters,
    remove_upload,
    replace_category_clusters,
    replace_upload_segments,
    set_upload_processing_error,
    update_cluster_profile,
    update_upload_transcription,
)
from ai.transcription import transcribe_audio_file


logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.60


@dataclass
class UploadedAudio:
    
    audio_file: str
    author: str
    category: str
    recorded_on: date

    @classmethod
    def create(cls, audio_file: str, author: str, category: str):
        return cls(
            audio_file=audio_file,
            author=author,
            category=category,
            recorded_on=timezone.localdate(),
        )


def process_uploaded_file(audio_file: str, author: str, category: str):
    """Main entrypoint used by the UI after a successful file upload."""
    uploaded_audio = UploadedAudio.create(audio_file, author, category)
    upload = create_or_update_upload(
        audio_file=uploaded_audio.audio_file,
        author=uploaded_audio.author,
        category=uploaded_audio.category,
        recorded_on=uploaded_audio.recorded_on,
    )

    try:
        transcription = transcribe_audio_file(uploaded_audio.audio_file, uploaded_audio.category)
        update_upload_transcription(upload, transcription)

        transcription_segments = transcription.get("segments", [])
        cleaned_segments = [segment for segment in transcription_segments if segment.get("text", "").strip()]
        created_segments = replace_upload_segments(upload, cleaned_segments)

        if not cleaned_segments:
            rebuild_category(uploaded_audio.category)
            return upload

        model_client = KnowledgeModelClient()
        new_embeddings = model_client.embed_texts([segment["text"] for segment in cleaned_segments])
        assign_new_segments_to_clusters(
            category=uploaded_audio.category,
            segments=created_segments,
            embeddings=new_embeddings,
            model_client=model_client,
        )
        return upload
    except Exception as exc:
        set_upload_processing_error(upload, str(exc))
        raise


def assign_new_segments_to_clusters(
    *,
    category: str,
    segments,
    embeddings,
    model_client: KnowledgeModelClient,
) -> None:
    """Assign newly created transcript segments into existing/new clusters."""
    if not segments:
        return

    clusters = get_category_clusters(category)
    named_clusters = [cluster for cluster in clusters if cluster.name != "Uncategorized"]
    existing_names = {cluster.name for cluster in clusters}

    pair_count = min(len(segments), len(embeddings))
    # Zip only the overlapping slice so mismatched lengths never crash assignment.
    segment_embedding_pairs = list(zip(segments[:pair_count], embeddings[:pair_count]))
    # Segments without vectors are preserved and routed to Uncategorized.
    segments_without_vectors = list(segments[pair_count:])
    # Segments with vectors that failed similarity against existing named clusters.
    unmatched_pairs = []
    assignments = defaultdict(list)

    for segment, embedding in segment_embedding_pairs:
        best_cluster = None
        best_score = -1.0

        for cluster in named_clusters:
            if not cluster.embedding:
                continue
            score = cosine_similarity(embedding, np.asarray(cluster.embedding, dtype=float))
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster and best_score >= SIMILARITY_THRESHOLD:
            assignments[best_cluster.id].append((segment, embedding))
        else:
            unmatched_pairs.append((segment, embedding))

    if unmatched_pairs:
        # Second-pass clustering for unmatched items: some can form new topics.
        unmatched_groups, uncategorized_pairs = group_unmatched_segments(unmatched_pairs)

        for group in unmatched_groups:
            group_texts = [segment.text for segment, _ in group]
            proposed_name = build_cluster_name(group_texts, existing_names, model_client)
            cluster = create_cluster(category, proposed_name)
            existing_names.add(cluster.name)
            named_clusters.append(cluster)
            assignments[cluster.id].extend(group)

        uncategorized = None
        if uncategorized_pairs or segments_without_vectors:
            uncategorized = next((cluster for cluster in clusters if cluster.name == "Uncategorized"), None)
            if not uncategorized:
                uncategorized = create_cluster(category, "Uncategorized")
                clusters.append(uncategorized)
        if uncategorized and uncategorized_pairs:
            assignments[uncategorized.id].extend(uncategorized_pairs)
        if uncategorized and segments_without_vectors:
            assignments[uncategorized.id].extend((segment, None) for segment in segments_without_vectors)
    elif segments_without_vectors:
        uncategorized = next((cluster for cluster in clusters if cluster.name == "Uncategorized"), None)
        if not uncategorized:
            uncategorized = create_cluster(category, "Uncategorized")
            clusters.append(uncategorized)
        assignments[uncategorized.id].extend((segment, None) for segment in segments_without_vectors)

    touched_clusters = {
        # Refresh only clusters that actually received new segments in this run.
        cluster.id: cluster
        for cluster in get_category_clusters(category)
        if cluster.id in assignments
    }

    for cluster_id, pair_list in assignments.items():
        cluster = touched_clusters.get(cluster_id)
        if not cluster:
            continue

        new_segments = [segment for segment, _ in pair_list]
        new_vectors = [np.asarray(vector, dtype=float) for _, vector in pair_list if vector is not None]
        add_segments_to_cluster(cluster, new_segments)

        old_count = cluster.member_count
        merged_embedding = merge_embedding(cluster.embedding, old_count, new_vectors)
        member_count = get_cluster_member_count(cluster)
        texts = get_cluster_texts(cluster)

        summary_payload = summarize_cluster_payload(
            texts=texts,
            model_client=model_client,
            existing_summary=cluster.summary,
        )

        update_cluster_profile(
            cluster,
            summary=summary_payload["summary"],
            summary_sections=summary_payload["summary_sections"],
            embedding=merged_embedding,
            member_count=member_count,
        )


def rebuild_category(category: str, model_client: KnowledgeModelClient | None = None) -> None:
    """Fully recompute clusters/summaries for one category from persisted segments."""
    window_data = get_category_segment_payloads(category)
    if not window_data:
        replace_category_clusters(category, {})
        return

    model_client = model_client or KnowledgeModelClient()
    embeddings = model_client.embed_texts([window["text"] for window in window_data])
    embedding_list = [embedding.tolist() for embedding in embeddings]

    enriched_windows = []
    for window, vector in zip(window_data, embedding_list):
        enriched_window = dict(window)
        enriched_window["embedding"] = vector
        enriched_windows.append(enriched_window)

    labels = cluster_embeddings(embedding_list)
    clustered_windows = group_windows_by_label(enriched_windows, labels)
    categories = generate_category_summaries(clustered_windows, model_client)
    replace_category_clusters(category, categories)


def delete_uploaded_audio(upload_id: int) -> tuple[str, str | None]:
    """Delete upload assets and rebuild the affected category in one workflow."""
    from uibase.models import AudioUpload

    upload = AudioUpload.objects.get(pk=upload_id)
    file_name, category = remove_upload(upload)

    warning = None
    try:
        rebuild_category(category)
    except Exception as exc:
        logger.exception("Failed to rebuild category '%s' after deletion.", category)
        remove_empty_category_clusters(category)
        warning = f"Deleted {file_name}, but reclustering failed: {exc}"

    return file_name, warning


def generate_category_summaries(clusters: dict, model_client: KnowledgeModelClient) -> dict:
    """Create cluster names + summaries for freshly grouped windows."""
    category_names = []
    categories = {}

    for cluster_id, cluster in clusters.items():
        cluster_windows = cluster.get("windows", [])
        cluster_texts = [window["text"] for window in cluster_windows]
        cluster_vectors = [np.asarray(window.get("embedding") or [], dtype=float) for window in cluster_windows]
        cluster_vectors = [vector for vector in cluster_vectors if vector.size > 0]
        centroid = normalize_vector(np.mean(cluster_vectors, axis=0)) if cluster_vectors else []

        if cluster_id == "-1":
            name = "Uncategorized"
        else:
            name = build_cluster_name(cluster_texts, set(category_names), model_client)

        summary_payload = summarize_cluster_payload(
            texts=cluster_texts,
            model_client=model_client,
            existing_summary="",
        )

        category_names.append(name)
        categories[name] = {
            "summary": summary_payload["summary"],
            "summary_sections": summary_payload["summary_sections"],
            "embedding": centroid,
            "windows": [strip_embedding(window) for window in cluster_windows],
        }

    return categories


def group_unmatched_segments(pairs):
    """Run a secondary clustering pass for new segments that matched nothing."""
    if not pairs:
        return [], []

    vectors = [np.asarray(vector, dtype=float).tolist() for _, vector in pairs]
    labels = cluster_embeddings(vectors, min_cluster_size=2, reduced_dimensions=5)
    grouped = defaultdict(list)
    uncategorized = []

    for (segment, vector), label in zip(pairs, labels):
        if label == -1:
            uncategorized.append((segment, vector))
            continue
        grouped[int(label)].append((segment, vector))

    return list(grouped.values()), uncategorized


def build_cluster_name(texts: list[str], existing_names: set[str], model_client: KnowledgeModelClient) -> str:
    """Ask the model for a short, unique cluster name with safe fallback."""
    if not texts:
        return make_unique_name("Cluster", existing_names)

    try:
        candidate = model_client.name_category(texts, sorted(existing_names))
    except Exception as exc:
        logger.warning("Falling back to a local cluster name. Error: %s", exc)
        candidate = "Cluster"

    return make_unique_name(candidate, existing_names)


def make_unique_name(candidate: str, existing_names: set[str]) -> str:
    """Guarantee uniqueness of cluster names by adding numeric suffixes."""
    base = (candidate or "Cluster").strip() or "Cluster"
    if base not in existing_names:
        return base

    suffix = 2
    while f"{base} {suffix}" in existing_names:
        suffix += 1
    return f"{base} {suffix}"


def summarize_cluster_payload(
    *,
    texts: list[str],
    model_client: KnowledgeModelClient,
    existing_summary: str,
) -> dict:
    """Build structured summary payload, falling back locally on model failure."""
    if not texts:
        return {
            "summary": existing_summary,
            "summary_sections": [],
        }

    try:
        payload = model_client.summarize_cluster_structured(
            texts=texts,
            existing_summary=existing_summary,
        )
        if payload.get("summary"):
            return payload
    except Exception as exc:
        logger.warning("Falling back to local summary structure. Error: %s", exc)

    return fallback_structured_summary(texts)


def _truncate_clean_text(text: str, limit: int) -> str:
    """Normalize whitespace and enforce a character cap."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def fallback_structured_summary(texts: list[str]) -> dict:
    """Deterministic summary fallback so UI stays useful without LLM output."""
    cleaned_texts = [_truncate_clean_text(text, 300) for text in texts if (text or "").strip()]
    combined = " ".join(cleaned_texts).strip()
    overview = _truncate_clean_text(combined, 1150) or "No summary available."

    step_summaries = []
    for index, snippet in enumerate(cleaned_texts[:6], start=1):
        sentence = snippet
        if sentence and sentence[-1] not in ".!?":
            sentence = f"{sentence}."
        step_summaries.append(f"Step {index}: {sentence}")

    process_flow = " ".join(step_summaries) or "No process highlights available."
    key_observations = " ".join(step_summaries[2:]) or process_flow
    failure_points = (
        "Failure risk usually increases when preparation, sequencing, or verification gets skipped. "
        "Recheck assumptions between major operations, confirm each transition point, and validate expected behavior "
        "before moving to the next action."
    )
    validation = (
        "Complete the work by verifying component condition, fastener security, and functional response under real usage. "
        "Capture what changed, what remains uncertain, and what should be rechecked on the next service event."
    )
    long_form_summary = _truncate_clean_text(
        f"{overview} {process_flow} {failure_points} {validation}",
        1700,
    )

    sections = [
        {"title": "Overview", "body": overview},
        {
            "title": "Process Highlights",
            "body": process_flow,
        },
        {
            "title": "Execution Sequence",
            "body": key_observations,
        },
        {
            "title": "Common Failure Points",
            "body": failure_points,
        },
        {
            "title": "Validation Checklist",
            "body": validation,
        },
    ]

    return {
        "summary": long_form_summary or overview,
        "summary_sections": sections,
    }


def strip_embedding(window: dict) -> dict:
    """Remove per-window embedding vectors before persisting cluster payload."""
    payload = dict(window)
    payload.pop("embedding", None)
    return payload


def cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity with defensive handling for empty vectors."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return -1.0

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return -1.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def merge_embedding(existing_embedding: list, existing_count: int, new_vectors: list[np.ndarray]) -> list[float]:
    """Incrementally merge previous cluster centroid with newly added vectors."""
    if not new_vectors and existing_embedding:
        return normalize_vector(np.asarray(existing_embedding, dtype=float))
    if not new_vectors:
        return []

    new_matrix = np.vstack(new_vectors)
    new_mean = np.mean(new_matrix, axis=0)

    if existing_embedding and existing_count > 0:
        existing_vector = np.asarray(existing_embedding, dtype=float)
        weighted = (existing_vector * float(existing_count)) + (new_mean * float(len(new_vectors)))
        return normalize_vector(weighted)

    return normalize_vector(new_mean)


def normalize_vector(vector) -> list[float]:
    """Return L2-normalized vector as plain list for JSONField storage."""
    arr = np.asarray(vector, dtype=float)
    if arr.size == 0:
        return []
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()
