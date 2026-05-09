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
    flag_cluster_as_stale,
    get_category_context,
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
DEFAULT_MACHINE_TYPE = "Unassigned"
PROMOTIONAL_PHRASES = (
    "today we're going to look at",
    "today we are going to look at",
    "today we're taking a look at",
    "this is so simple anyone could do",
    "these are the things that you're going to need",
    "subscribe",
    "follow us",
    "social media",
    "thank you for watching",
    "thanks for watching",
    "thumbs up",
    "comment section",
    "comments section",
    "like the video",
    "latest video",
    "link in the description",
    "leave a link in the description",
)


@dataclass
class UploadedAudio:

    audio_file: str
    author: str
    machine_name: str
    machine_type: str
    subcategory_paths: list[list[str]]
    hierarchy_path: list[str]
    extra_tags: list[str]
    recorded_on: date

    @property
    def category(self) -> str:
        """Pipeline cluster key: keep one machine per category bucket."""
        return self.machine_name

    @classmethod
    def create(
        cls,
        audio_file: str,
        author: str,
        machine_name: str,
        machine_type: str = DEFAULT_MACHINE_TYPE,
        subcategory_paths=None,
        hierarchy_path=None,
        extra_tags=None,
    ):
        normalized_subcategory_paths = _normalize_subcategory_paths(subcategory_paths)
        primary_hierarchy_path = _select_primary_subcategory_path(
            normalized_subcategory_paths,
            fallback=hierarchy_path,
        )
        return cls(
            audio_file=audio_file,
            author=author,
            machine_name=(machine_name or "").strip(),
            machine_type=(machine_type or "").strip() or DEFAULT_MACHINE_TYPE,
            subcategory_paths=normalized_subcategory_paths,
            hierarchy_path=primary_hierarchy_path,
            extra_tags=_normalize_string_list(extra_tags),
            recorded_on=timezone.localdate(),
        )


def process_uploaded_file(
    audio_file: str,
    author: str,
    machine_name: str,
    machine_type: str = DEFAULT_MACHINE_TYPE,
    subcategory_paths=None,
    hierarchy_path=None,
    extra_tags=None,
):
    """Main entrypoint used by the UI after a successful file upload."""
    uploaded_audio = UploadedAudio.create(
        audio_file=audio_file,
        author=author,
        machine_name=machine_name,
        machine_type=machine_type,
        subcategory_paths=subcategory_paths,
        hierarchy_path=hierarchy_path,
        extra_tags=extra_tags,
    )
    upload = create_or_update_upload(
        audio_file=uploaded_audio.audio_file,
        author=uploaded_audio.author,
        machine_name=uploaded_audio.machine_name,
        machine_type=uploaded_audio.machine_type,
        subcategory_paths=uploaded_audio.subcategory_paths,
        hierarchy_path=uploaded_audio.hierarchy_path,
        extra_tags=uploaded_audio.extra_tags,
        category=uploaded_audio.category,
        recorded_on=uploaded_audio.recorded_on,
    )

    try:
        transcription = transcribe_audio_file(
            audio_file_name=uploaded_audio.audio_file,
            machine_name=uploaded_audio.machine_name,
            machine_type=uploaded_audio.machine_type,
            subcategory_paths=uploaded_audio.subcategory_paths,
            hierarchy_path=uploaded_audio.hierarchy_path,
            extra_tags=uploaded_audio.extra_tags,
        )
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
    category_context = get_category_context(category)

    pair_count = min(len(segments), len(embeddings))
    # Zip only the overlapping slice so mismatched lengths never crash assignment.
    segment_embedding_pairs = list(zip(segments[:pair_count], embeddings[:pair_count]))
    # Segments without vectors are preserved and routed to Uncategorized.
    segments_without_vectors = list(segments[pair_count:])
    # Low-signal snippets (for example channel outro/promo text) are kept in
    # storage but routed away from technical clusters.
    forced_uncategorized_pairs = []
    # Segments with vectors that failed similarity against existing named clusters.
    unmatched_pairs = []
    assignments = defaultdict(list)

    for segment, embedding in segment_embedding_pairs:
        if is_promotional_text(segment.text):
            forced_uncategorized_pairs.append((segment, embedding))
            continue

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

    uncategorized_pairs = list(forced_uncategorized_pairs)
    if unmatched_pairs:
        # Second-pass clustering for unmatched items: some can form new topics.
        unmatched_groups, discovered_uncategorized_pairs = group_unmatched_segments(unmatched_pairs)
        uncategorized_pairs.extend(discovered_uncategorized_pairs)

        for group in unmatched_groups:
            group_texts = [segment.text for segment, _ in group]
            if not has_meaningful_cluster_content(group_texts):
                uncategorized_pairs.extend(group)
                continue
            proposed_name = build_cluster_name(group_texts, existing_names, model_client, context=category_context)
            cluster = create_cluster(category, proposed_name)
            existing_names.add(cluster.name)
            named_clusters.append(cluster)
            assignments[cluster.id].extend(group)

    if uncategorized_pairs or segments_without_vectors:
        uncategorized = next((cluster for cluster in clusters if cluster.name == "Uncategorized"), None)
        if not uncategorized:
            uncategorized = create_cluster(category, "Uncategorized")
            clusters.append(uncategorized)
        if uncategorized_pairs:
            assignments[uncategorized.id].extend(uncategorized_pairs)
        if segments_without_vectors:
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
        new_texts = [str(segment.text or "").strip() for segment in new_segments if str(segment.text or "").strip()]
        new_vectors = [np.asarray(vector, dtype=float) for _, vector in pair_list if vector is not None]
        add_segments_to_cluster(cluster, new_segments)

        old_count = cluster.member_count
        merged_embedding = merge_embedding(cluster.embedding, old_count, new_vectors)
        member_count = get_cluster_member_count(cluster)
        has_existing_summary = bool((cluster.summary or "").strip() or (cluster.summary_sections or []))
        if has_existing_summary:
            summary_mode = "incremental"
            texts = new_texts or get_cluster_texts(cluster, limit=80)
        else:
            summary_mode = "full"
            texts = get_cluster_texts(cluster)

        summary_payload = summarize_cluster_payload(
            texts=texts,
            model_client=model_client,
            existing_summary=cluster.summary,
            context=category_context,
            mode=summary_mode,
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
    categories = generate_category_summaries(
        clustered_windows,
        model_client,
        context=get_category_context(category),
    )
    replace_category_clusters(category, categories)


def refresh_category_after_deletion(category: str, model_client: KnowledgeModelClient | None = None) -> None:
    """Fast post-delete refresh: keep clusters coherent without full reclustering."""
    clusters = get_category_clusters(category)
    if not clusters:
        return

    category_context = get_category_context(category)
    model_client = model_client or KnowledgeModelClient()

    for cluster in clusters:
        remaining_members = [
            membership
            for membership in cluster.memberships.all()
            if membership.segment and (membership.segment.text or "").strip()
        ]
        if not remaining_members:
            cluster.delete()
            continue

        texts = [membership.segment.text for membership in remaining_members]
        embeddings = model_client.embed_texts(texts)
        centroid = normalize_vector(np.mean(embeddings, axis=0)) if embeddings.size else []

        # Keep delete fast by using deterministic local summarization (no LLM roundtrip).
        summary_payload = fallback_structured_summary(texts, context=category_context)
        update_cluster_profile(
            cluster,
            summary=summary_payload["summary"],
            summary_sections=summary_payload["summary_sections"],
            embedding=centroid,
            member_count=len(remaining_members),
        )

    remove_empty_category_clusters(category)


def delete_uploaded_audio_batch(upload_ids: list[int], keep_summaries: bool = True) -> dict:
    """Delete many uploads and either mark summaries stale or recompute them."""
    from uibase.models import AudioUpload, CategoryCluster, ClusterSegment

    cleaned_ids = []
    seen_ids = set()
    for value in upload_ids or []:
        try:
            upload_id = int(value)
        except (TypeError, ValueError):
            continue
        if upload_id <= 0 or upload_id in seen_ids:
            continue
        seen_ids.add(upload_id)
        cleaned_ids.append(upload_id)

    uploads = list(AudioUpload.objects.filter(id__in=cleaned_ids).order_by("-created_at", "-id"))
    if not uploads:
        return {
            "deleted_count": 0,
            "deleted_files": [],
            "affected_categories": [],
            "dropped_categories": [],
            "stale_clusters": 0,
            "refreshed_categories": [],
            "warning": None,
            "keep_summaries": bool(keep_summaries),
        }

    deleted_files = []
    affected_categories = set()
    cluster_deleted_files = defaultdict(set)

    memberships = (
        ClusterSegment.objects.select_related("cluster", "segment__upload")
        .filter(segment__upload_id__in=[upload.id for upload in uploads])
    )
    for membership in memberships:
        upload = membership.segment.upload
        if upload and upload.stored_name:
            cluster_deleted_files[membership.cluster_id].add(upload.stored_name)

    for upload in uploads:
        file_name, category = remove_upload(upload)
        deleted_files.append(file_name)
        affected_categories.add(category)

    dropped_categories = []
    refreshed_categories = []
    stale_clusters = 0

    try:
        for category in sorted(affected_categories):
            has_remaining_uploads = AudioUpload.objects.filter(category=category).exists()
            if not has_remaining_uploads:
                replace_category_clusters(category, {})
                dropped_categories.append(category)
                continue

            if keep_summaries:
                remove_empty_category_clusters(category)
            else:
                refresh_category_after_deletion(category)
                refreshed_categories.append(category)

        if keep_summaries:
            for cluster_id, files in cluster_deleted_files.items():
                cluster = CategoryCluster.objects.filter(pk=cluster_id).first()
                if not cluster:
                    continue

                current_count = get_cluster_member_count(cluster)
                if current_count <= 0:
                    cluster.delete()
                    continue

                if cluster.member_count != current_count:
                    cluster.member_count = current_count
                    cluster.save(update_fields=["member_count"])

                flag_cluster_as_stale(cluster, sorted(files))
                stale_clusters += 1
    except Exception as exc:
        logger.exception("Post-delete update failed after removing uploads.")
        for category in affected_categories:
            remove_empty_category_clusters(category)
        return {
            "deleted_count": len(deleted_files),
            "deleted_files": deleted_files,
            "affected_categories": sorted(affected_categories),
            "dropped_categories": sorted(dropped_categories),
            "stale_clusters": stale_clusters,
            "refreshed_categories": sorted(refreshed_categories),
            "warning": f"Deleted files, but post-delete update failed: {exc}",
            "keep_summaries": bool(keep_summaries),
        }

    return {
        "deleted_count": len(deleted_files),
        "deleted_files": deleted_files,
        "affected_categories": sorted(affected_categories),
        "dropped_categories": sorted(dropped_categories),
        "stale_clusters": stale_clusters,
        "refreshed_categories": sorted(refreshed_categories),
        "warning": None,
        "keep_summaries": bool(keep_summaries),
    }


def delete_uploaded_audio(upload_id: int, keep_summaries: bool = True) -> tuple[str, str | None]:
    """Delete one upload and return the deleted filename + optional warning."""
    payload = delete_uploaded_audio_batch([upload_id], keep_summaries=keep_summaries)
    deleted_files = payload.get("deleted_files") or []
    file_name = deleted_files[0] if deleted_files else f"upload#{upload_id}"
    return file_name, payload.get("warning")


def redo_cluster_summary(cluster_id: int) -> dict:
    """Regenerate one cluster summary/embedding from its current remaining members."""
    from uibase.models import CategoryCluster

    cluster = CategoryCluster.objects.filter(pk=cluster_id).first()
    if not cluster:
        return {"updated": 0, "deleted": 0, "category": "", "cluster_name": ""}

    texts = get_cluster_texts(cluster, limit=320)
    member_count = get_cluster_member_count(cluster)
    if member_count <= 0 or not texts:
        cluster.delete()
        return {"updated": 0, "deleted": 1, "category": cluster.category, "cluster_name": cluster.name}

    model_client = KnowledgeModelClient()
    embeddings = model_client.embed_texts(texts)
    merged_embedding = normalize_vector(np.mean(embeddings, axis=0)) if embeddings.size else []
    summary_payload = summarize_cluster_payload(
        texts=texts,
        model_client=model_client,
        existing_summary="",
        context=get_category_context(cluster.category),
        mode="full",
    )
    update_cluster_profile(
        cluster,
        summary=summary_payload["summary"],
        summary_sections=summary_payload["summary_sections"],
        embedding=merged_embedding,
        member_count=member_count,
        mark_fresh=True,
    )
    return {"updated": 1, "deleted": 0, "category": cluster.category, "cluster_name": cluster.name}


def redo_category_summaries(category: str) -> dict:
    """Regenerate summaries for all clusters under one machine/category bucket."""
    clusters = get_category_clusters(category)
    if not clusters:
        return {"updated": 0, "deleted": 0, "category": category}

    updated = 0
    deleted = 0
    model_client = KnowledgeModelClient()
    category_context = get_category_context(category)

    for cluster in clusters:
        texts = get_cluster_texts(cluster, limit=320)
        member_count = get_cluster_member_count(cluster)
        if member_count <= 0 or not texts:
            cluster.delete()
            deleted += 1
            continue

        embeddings = model_client.embed_texts(texts)
        merged_embedding = normalize_vector(np.mean(embeddings, axis=0)) if embeddings.size else []
        summary_payload = summarize_cluster_payload(
            texts=texts,
            model_client=model_client,
            existing_summary="",
            context=category_context,
            mode="full",
        )
        update_cluster_profile(
            cluster,
            summary=summary_payload["summary"],
            summary_sections=summary_payload["summary_sections"],
            embedding=merged_embedding,
            member_count=member_count,
            mark_fresh=True,
        )
        updated += 1

    remove_empty_category_clusters(category)
    return {"updated": updated, "deleted": deleted, "category": category}


def remove_machine_type_category(machine_type: str) -> dict:
    """Remove one overcategory by deleting all uploads and related data in it."""
    from uibase.models import AudioUpload

    type_value = (machine_type or "").strip()
    if not type_value:
        return {"deleted_count": 0, "deleted_files": [], "machine_type": "", "warning": "Missing category."}

    upload_ids = list(AudioUpload.objects.filter(machine_type=type_value).values_list("id", flat=True))
    if not upload_ids:
        return {
            "deleted_count": 0,
            "deleted_files": [],
            "machine_type": type_value,
            "warning": None,
        }

    payload = delete_uploaded_audio_batch(upload_ids, keep_summaries=True)
    payload["machine_type"] = type_value
    return payload


def rename_machine_type(machine_type: str, new_machine_type: str) -> dict:
    """Rename one top-level type label across uploads."""
    from uibase.models import AudioUpload

    old_value = (machine_type or "").strip()
    new_value = (new_machine_type or "").strip()
    if not old_value:
        return {"updated": 0, "old": old_value, "new": new_value, "warning": "Missing current category."}
    if not new_value:
        return {"updated": 0, "old": old_value, "new": new_value, "warning": "Missing new category name."}
    if old_value.casefold() == new_value.casefold():
        return {"updated": 0, "old": old_value, "new": new_value, "warning": "New category name is the same as current."}

    updated = AudioUpload.objects.filter(machine_type=old_value).update(machine_type=new_value)
    return {"updated": int(updated), "old": old_value, "new": new_value, "warning": None}


def remove_cluster(cluster_id: int) -> dict:
    """Remove one undercategory cluster while keeping source uploads."""
    from uibase.models import CategoryCluster

    cluster = CategoryCluster.objects.prefetch_related("memberships__segment__upload").filter(pk=cluster_id).first()
    if not cluster:
        return {
            "deleted": 0,
            "cluster_id": cluster_id,
            "cluster_name": "",
            "category": "",
            "member_count": 0,
            "files": [],
        }

    member_count = get_cluster_member_count(cluster)
    files = sorted(
        {
            membership.segment.upload.stored_name
            for membership in cluster.memberships.all()
            if membership.segment and membership.segment.upload
        }
    )
    payload = {
        "deleted": 1,
        "cluster_id": cluster.id,
        "cluster_name": cluster.name,
        "category": cluster.category,
        "member_count": member_count,
        "files": files,
    }
    cluster.delete()
    remove_empty_category_clusters(payload["category"])
    return payload


def rename_cluster(cluster_id: int, new_name: str) -> dict:
    """Rename one undercategory cluster with category-level uniqueness."""
    from uibase.models import CategoryCluster

    cluster = CategoryCluster.objects.filter(pk=cluster_id).first()
    desired_name = (new_name or "").strip()
    if not cluster:
        return {"updated": 0, "cluster_id": cluster_id, "old_name": "", "new_name": desired_name, "category": ""}
    if not desired_name:
        return {
            "updated": 0,
            "cluster_id": cluster.id,
            "old_name": cluster.name,
            "new_name": desired_name,
            "category": cluster.category,
            "warning": "Missing new undercategory name.",
        }

    existing_names = set(
        CategoryCluster.objects.filter(category=cluster.category).exclude(pk=cluster.pk).values_list("name", flat=True)
    )
    normalized_name = make_unique_name(desired_name, existing_names)
    if normalized_name == cluster.name:
        return {
            "updated": 0,
            "cluster_id": cluster.id,
            "old_name": cluster.name,
            "new_name": normalized_name,
            "category": cluster.category,
            "warning": "Undercategory already has that name.",
        }

    old_name = cluster.name
    cluster.name = normalized_name
    cluster.save(update_fields=["name"])
    return {
        "updated": 1,
        "cluster_id": cluster.id,
        "old_name": old_name,
        "new_name": normalized_name,
        "category": cluster.category,
        "warning": None,
    }


def _resolve_target_cluster_for_move(
    *,
    category: str,
    target_cluster_id: int | None = None,
    new_cluster_name: str | None = None,
    force_uncategorized: bool = False,
):
    """Resolve or create the destination cluster for segment/cluster move operations."""
    from uibase.models import CategoryCluster

    if force_uncategorized:
        target = CategoryCluster.objects.filter(category=category, name="Uncategorized").first()
        if not target:
            target = create_cluster(category, "Uncategorized")
        return target

    if target_cluster_id:
        target = CategoryCluster.objects.filter(pk=target_cluster_id, category=category).first()
        if target:
            return target

    desired_name = (new_cluster_name or "").strip()
    if desired_name:
        existing_names = set(CategoryCluster.objects.filter(category=category).values_list("name", flat=True))
        normalized_name = make_unique_name(desired_name, existing_names)
        existing = CategoryCluster.objects.filter(category=category, name=normalized_name).first()
        if existing:
            return existing
        return create_cluster(category, normalized_name)

    return None


def move_segment_to_cluster(
    *,
    segment_id: int,
    target_cluster_id: int | None = None,
    new_cluster_name: str | None = None,
    force_uncategorized: bool = False,
) -> dict:
    """Move one snippet/segment to another undercategory within the same machine."""
    from uibase.models import ClusterSegment, TranscriptSegment

    segment = TranscriptSegment.objects.select_related("upload").filter(pk=segment_id).first()
    if not segment:
        return {"updated": 0, "warning": "Snippet not found.", "segment_id": segment_id}

    category = segment.upload.category
    source_cluster_ids = list(
        ClusterSegment.objects.filter(segment_id=segment.id, cluster__category=category).values_list("cluster_id", flat=True)
    )
    target_cluster = _resolve_target_cluster_for_move(
        category=category,
        target_cluster_id=target_cluster_id,
        new_cluster_name=new_cluster_name,
        force_uncategorized=force_uncategorized,
    )
    if not target_cluster:
        return {
            "updated": 0,
            "warning": "No target undercategory selected.",
            "segment_id": segment.id,
            "category": category,
        }

    ClusterSegment.objects.filter(segment_id=segment.id, cluster__category=category).delete()
    add_segments_to_cluster(target_cluster, [segment])

    touched_cluster_ids = sorted(set([*source_cluster_ids, target_cluster.id]))
    updated_clusters = 0
    deleted_clusters = 0
    for cluster_id in touched_cluster_ids:
        payload = redo_cluster_summary(cluster_id)
        updated_clusters += int(payload.get("updated") or 0)
        deleted_clusters += int(payload.get("deleted") or 0)

    remove_empty_category_clusters(category)
    return {
        "updated": 1,
        "segment_id": segment.id,
        "target_cluster_id": target_cluster.id,
        "target_cluster_name": target_cluster.name,
        "category": category,
        "updated_clusters": updated_clusters,
        "deleted_clusters": deleted_clusters,
        "warning": None,
    }


def merge_cluster_into_target(
    *,
    cluster_id: int,
    target_cluster_id: int | None = None,
    new_cluster_name: str | None = None,
    force_uncategorized: bool = False,
) -> dict:
    """Move all snippets from one undercategory into another and remove source cluster."""
    from uibase.models import CategoryCluster

    source = (
        CategoryCluster.objects.prefetch_related("memberships__segment")
        .filter(pk=cluster_id)
        .first()
    )
    if not source:
        return {"updated": 0, "warning": "Source undercategory not found.", "cluster_id": cluster_id}

    target = _resolve_target_cluster_for_move(
        category=source.category,
        target_cluster_id=target_cluster_id,
        new_cluster_name=new_cluster_name,
        force_uncategorized=force_uncategorized,
    )
    if not target:
        return {"updated": 0, "warning": "No merge target selected.", "cluster_id": source.id, "category": source.category}
    if target.id == source.id:
        return {"updated": 0, "warning": "Source and target undercategory are the same.", "cluster_id": source.id, "category": source.category}

    segments = [membership.segment for membership in source.memberships.all() if membership.segment]
    add_segments_to_cluster(target, segments)
    source_name = source.name
    source.delete()

    payload = redo_cluster_summary(target.id)
    remove_empty_category_clusters(target.category)
    return {
        "updated": int(payload.get("updated") or 0),
        "deleted": 1,
        "source_name": source_name,
        "target_name": payload.get("cluster_name") or target.name,
        "category": target.category,
        "moved_segments": len(segments),
        "warning": None,
    }


def delete_segment_from_knowledge(segment_id: int) -> dict:
    """Delete one snippet from storage and refresh affected undercategories."""
    from uibase.models import ClusterSegment, TranscriptSegment

    segment = TranscriptSegment.objects.select_related("upload").filter(pk=segment_id).first()
    if not segment:
        return {"deleted": 0, "warning": "Snippet not found.", "segment_id": segment_id}

    category = segment.upload.category
    source_cluster_ids = list(
        ClusterSegment.objects.filter(segment_id=segment.id, cluster__category=category).values_list("cluster_id", flat=True)
    )
    text_preview = _truncate_clean_text(segment.text, 160)
    segment.delete()

    updated_clusters = 0
    deleted_clusters = 0
    for cluster_id in sorted(set(source_cluster_ids)):
        payload = redo_cluster_summary(cluster_id)
        updated_clusters += int(payload.get("updated") or 0)
        deleted_clusters += int(payload.get("deleted") or 0)

    remove_empty_category_clusters(category)
    return {
        "deleted": 1,
        "segment_id": segment_id,
        "category": category,
        "updated_clusters": updated_clusters,
        "deleted_clusters": deleted_clusters,
        "text_preview": text_preview,
        "warning": None,
    }


def generate_category_summaries(
    clusters: dict,
    model_client: KnowledgeModelClient,
    context: dict | None = None,
) -> dict:
    """Create cluster names + summaries for freshly grouped windows."""
    category_names = []
    categories = {}
    uncategorized_windows = []

    for cluster_id, cluster in clusters.items():
        cluster_windows = cluster.get("windows", [])
        if not cluster_windows:
            continue

        cluster_texts = [window["text"] for window in cluster_windows]
        if cluster_id != "-1" and not has_meaningful_cluster_content(cluster_texts):
            uncategorized_windows.extend(cluster_windows)
            continue

        cluster_vectors = [np.asarray(window.get("embedding") or [], dtype=float) for window in cluster_windows]
        cluster_vectors = [vector for vector in cluster_vectors if vector.size > 0]
        centroid = normalize_vector(np.mean(cluster_vectors, axis=0)) if cluster_vectors else []

        if cluster_id == "-1":
            name = "Uncategorized"
            uncategorized_windows.extend(cluster_windows)
            continue
        else:
            name = build_cluster_name(cluster_texts, set(category_names), model_client, context=context)

        summary_payload = summarize_cluster_payload(
            texts=cluster_texts,
            model_client=model_client,
            existing_summary="",
            context=context,
            mode="full",
        )

        category_names.append(name)
        categories[name] = {
            "summary": summary_payload["summary"],
            "summary_sections": summary_payload["summary_sections"],
            "embedding": centroid,
            "windows": [strip_embedding(window) for window in cluster_windows],
        }

    if uncategorized_windows:
        uncategorized_texts = [window.get("text", "") for window in uncategorized_windows]
        uncategorized_vectors = [np.asarray(window.get("embedding") or [], dtype=float) for window in uncategorized_windows]
        uncategorized_vectors = [vector for vector in uncategorized_vectors if vector.size > 0]
        uncategorized_centroid = (
            normalize_vector(np.mean(uncategorized_vectors, axis=0))
            if uncategorized_vectors
            else []
        )
        uncategorized_summary = summarize_cluster_payload(
            texts=uncategorized_texts,
            model_client=model_client,
            existing_summary="",
            context=context,
            mode="full",
        )
        categories["Uncategorized"] = {
            "summary": uncategorized_summary["summary"],
            "summary_sections": uncategorized_summary["summary_sections"],
            "embedding": uncategorized_centroid,
            "windows": [strip_embedding(window) for window in uncategorized_windows],
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


def build_cluster_name(
    texts: list[str],
    existing_names: set[str],
    model_client: KnowledgeModelClient,
    *,
    context: dict | None = None,
) -> str:
    """Ask the model for a short, unique cluster name with safe fallback."""
    candidate_texts = select_grounded_texts(texts)
    if not candidate_texts:
        return make_unique_name("Cluster", existing_names)

    try:
        candidate = model_client.name_category(candidate_texts, sorted(existing_names), context=context)
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
    context: dict | None = None,
    mode: str = "full",
) -> dict:
    """Build structured summary payload, falling back locally on model failure."""
    normalized_mode = str(mode or "full").strip().lower()
    if normalized_mode not in {"full", "incremental"}:
        normalized_mode = "full"

    grounded_texts = select_grounded_texts(texts)
    if not grounded_texts:
        return {
            "summary": existing_summary,
            "summary_sections": [],
        }

    try:
        payload = model_client.summarize_cluster_structured(
            texts=grounded_texts,
            existing_summary=existing_summary,
            context=context,
            mode=normalized_mode,
        )
        if payload.get("summary"):
            return payload
    except Exception as exc:
        logger.warning("Falling back to local summary structure. Error: %s", exc)

    if normalized_mode == "incremental":
        return fallback_incremental_summary(
            new_texts=grounded_texts,
            existing_summary=existing_summary,
        )

    return fallback_structured_summary(grounded_texts, context=context)


def _truncate_clean_text(text: str, limit: int) -> str:
    """Normalize whitespace and enforce a character cap."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def fallback_structured_summary(texts: list[str], context: dict | None = None) -> dict:
    """Deterministic grounded fallback that never invents facts."""
    cleaned_texts = select_grounded_texts(texts)
    if not cleaned_texts:
        return {
            "summary": "No snippet content available for summary.",
            "summary_sections": [],
        }

    summary_sentences = []
    for snippet in cleaned_texts[:4]:
        sentence = _truncate_clean_text(snippet, 360)
        if sentence and sentence[-1] not in ".!?":
            sentence = f"{sentence}."
        summary_sentences.append(sentence)

    summary = _truncate_clean_text(" ".join(summary_sentences), 1400) or "No summary available."
    sections = []
    for index, snippet in enumerate(cleaned_texts[:5], start=1):
        sections.append(
            {
                "title": f"Snippet {index}",
                "body": _truncate_clean_text(snippet, 560),
            }
        )

    return {
        "summary": summary,
        "summary_sections": sections,
    }


def fallback_incremental_summary(*, new_texts: list[str], existing_summary: str) -> dict:
    """Cheap local incremental update used during normal ingestion."""
    cleaned_new = select_grounded_texts(new_texts)
    base = _truncate_clean_text(existing_summary or "", 1400)

    if not cleaned_new:
        return {
            "summary": base or "No summary available.",
            "summary_sections": [{"title": "No New Update", "body": "No new snippet evidence was available."}]
            if base
            else [],
        }

    update_sentences = []
    for snippet in cleaned_new[:3]:
        sentence = _truncate_clean_text(snippet, 320)
        if sentence and sentence[-1] not in ".!?":
            sentence = f"{sentence}."
        update_sentences.append(sentence)

    update_block = _truncate_clean_text(" ".join(update_sentences), 900)
    if base:
        merged_summary = _truncate_clean_text(f"{base} Update: {update_block}", 1600)
    else:
        merged_summary = update_block

    return {
        "summary": merged_summary or base or "No summary available.",
        "summary_sections": [
            {
                "title": "Latest Update",
                "body": update_block or "No new update.",
            }
        ],
    }


def strip_embedding(window: dict) -> dict:
    """Remove per-window embedding vectors before persisting cluster payload."""
    payload = dict(window)
    payload.pop("embedding", None)
    return payload


def _normalize_string_list(values) -> list[str]:
    """Normalize optional string arrays for taxonomy fields."""
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    cleaned = []
    seen = set()
    for value in values:
        item = str(value or "").strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def is_promotional_text(text: str) -> bool:
    """Detect channel-engagement snippets that should not drive technical clusters."""
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return True
    return any(phrase in normalized for phrase in PROMOTIONAL_PHRASES)


def select_grounded_texts(texts: list[str]) -> list[str]:
    """Prepare summary/name inputs using only unique non-empty snippet content."""
    cleaned = []
    seen = set()
    for text in texts or []:
        normalized_text = " ".join((text or "").split())
        if not normalized_text:
            continue
        dedupe_key = normalized_text.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(normalized_text)

    if not cleaned:
        return []

    non_promotional = [text for text in cleaned if not is_promotional_text(text)]
    return non_promotional or cleaned


def has_meaningful_cluster_content(texts: list[str]) -> bool:
    """Guard against creating dedicated clusters from pure promo/outro text."""
    for text in texts or []:
        if not is_promotional_text(text):
            return True
    return False


def _normalize_subcategory_paths(values) -> list[list[str]]:
    """Normalize optional nested subcategory paths into a stable list format."""
    if not values:
        return []
    if isinstance(values, str):
        values = [values]

    normalized = []
    seen = set()
    for raw_path in values:
        if isinstance(raw_path, str):
            parts = [part.strip() for part in raw_path.split(">")]
        elif isinstance(raw_path, (list, tuple)):
            parts = [str(part or "").strip() for part in raw_path]
        else:
            continue

        path = [part for part in parts if part]
        if not path:
            continue
        key = tuple(part.casefold() for part in path)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(path)
    return normalized


def _select_primary_subcategory_path(paths: list[list[str]], fallback=None) -> list[str]:
    """Pick one path for legacy single-path fields while keeping full path data."""
    normalized = _normalize_subcategory_paths(paths)
    if normalized:
        return normalized[0]
    return _normalize_string_list(fallback)


def format_taxonomy_context(context: dict | None) -> str:
    """Create a compact taxonomy sentence used in AI prompts/fallback summaries."""
    if not context:
        return ""
    machine_type = str(context.get("machine_type") or "").strip()
    machine_name = str(context.get("machine_name") or "").strip()
    subcategory_paths = _normalize_subcategory_paths(context.get("subcategory_paths"))
    hierarchy_path = _normalize_string_list(context.get("hierarchy_path"))
    extra_tags = _normalize_string_list(context.get("extra_tags"))

    parts = []
    if machine_type:
        parts.append(f"Type: {machine_type}")
    if machine_name:
        parts.append(f"Machine: {machine_name}")
    if subcategory_paths:
        path_labels = [(" > ".join(path)) for path in subcategory_paths if path]
        if path_labels:
            parts.append("Subcategories: " + "; ".join(path_labels[:8]))
    if hierarchy_path:
        parts.append("Hierarchy: " + " > ".join(hierarchy_path))
    if extra_tags:
        parts.append("Tags: " + ", ".join(extra_tags))
    return " | ".join(parts)


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
