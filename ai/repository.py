"""Database access helpers for the audio knowledge-base domain.

This module is intentionally focused on persistence concerns only:
- creating/updating upload and transcript rows
- reading payloads needed by clustering/summarization
- creating/updating cluster membership and summary records

Keeping all ORM queries here lets the pipeline module stay focused on
business flow, while this file remains the single place to adjust data
shape and query performance.
"""

from collections import defaultdict

from django.db import transaction
from django.db.models import Max, Prefetch

from ai import paths
from uibase.models import AudioUpload, CategoryCluster, ClusterSegment, TranscriptSegment


def _date_as_string(value) -> str:
    """Normalize date-like values before exposing them to templates/API payloads."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def create_or_update_upload(
    *,
    audio_file: str,
    author: str,
    category: str,
    recorded_on,
    original_name: str | None = None,
) -> AudioUpload:
    """Upsert one uploaded file record keyed by stored filename."""
    upload, _ = AudioUpload.objects.update_or_create(
        stored_name=audio_file,
        defaults={
            "original_name": original_name or audio_file,
            "author": author,
            "category": category,
            "recorded_on": recorded_on,
        },
    )
    return upload


def update_upload_transcription(upload: AudioUpload, transcription: dict) -> AudioUpload:
    """Persist the top-level transcript metadata produced by transcription.py."""
    upload.transcription_text = transcription.get("text", "") or ""
    upload.language = transcription.get("language") or ""
    upload.duration = transcription.get("duration") or 0.0
    upload.processing_error = ""
    upload.save(
        update_fields=[
            "transcription_text",
            "language",
            "duration",
            "processing_error",
        ]
    )
    return upload


def set_upload_processing_error(upload: AudioUpload, error: str) -> AudioUpload:
    """Store the latest processing error so the UI can show failure state."""
    upload.processing_error = error
    upload.save(update_fields=["processing_error"])
    return upload


def replace_upload_segments(
    upload: AudioUpload,
    segments: list[dict],
    embeddings=None,
) -> list[TranscriptSegment]:
    """Replace all transcript segments for one upload in a single batch."""
    upload.segments.all().delete()

    # Space-saving mode: snippet vectors are not persisted in SQLite.
    objects = []
    for position, segment in enumerate(segments):
        objects.append(
            TranscriptSegment(
                upload=upload,
                position=position,
                start=segment.get("start"),
                end=segment.get("end"),
                text=segment.get("text", ""),
                embedding=[],
            )
        )

    return TranscriptSegment.objects.bulk_create(objects)


def get_category_segment_payloads(category: str) -> list[dict]:
    """Return cluster-ready segment payloads for one category."""
    queryset = (
        TranscriptSegment.objects.select_related("upload")
        .filter(upload__category=category)
        .order_by("upload__created_at", "upload_id", "position", "id")
    )

    payloads = []
    for segment in queryset:
        payloads.append(
            {
                "segment_id": segment.id,
                "text": segment.text,
                "source_file": segment.upload.stored_name,
                "author": segment.upload.author,
                "date": _date_as_string(segment.upload.recorded_on),
                "start": segment.start,
                "end": segment.end,
            }
        )
    return payloads


def get_category_clusters(category: str) -> list[CategoryCluster]:
    """Load clusters with memberships pre-fetched for fast pipeline access."""
    return list(
        CategoryCluster.objects.filter(category=category)
        .prefetch_related(
            Prefetch(
                "memberships",
                queryset=ClusterSegment.objects.select_related("segment", "segment__upload").order_by(
                    "position",
                    "segment__position",
                    "id",
                ),
            )
        )
        .order_by("position", "id")
    )


def create_cluster(category: str, name: str) -> CategoryCluster:
    """Create a new cluster at the end of the category ordering."""
    max_position = (
        CategoryCluster.objects.filter(category=category).aggregate(max_pos=Max("position")).get("max_pos") or -1
    )
    return CategoryCluster.objects.create(
        category=category,
        name=name,
        position=max_position + 1,
    )


def add_segments_to_cluster(cluster: CategoryCluster, segments: list[TranscriptSegment]) -> None:
    """Attach transcript segments to a cluster while preserving insertion order."""
    if not segments:
        return

    max_position = (
        ClusterSegment.objects.filter(cluster=cluster).aggregate(max_pos=Max("position")).get("max_pos") or -1
    )
    memberships = []
    for offset, segment in enumerate(segments, start=1):
        memberships.append(
            ClusterSegment(
                cluster=cluster,
                segment=segment,
                position=max_position + offset,
            )
        )
    ClusterSegment.objects.bulk_create(memberships, ignore_conflicts=True)


def update_cluster_profile(
    cluster: CategoryCluster,
    *,
    summary: str,
    summary_sections: list[dict],
    embedding: list[float],
    member_count: int,
) -> CategoryCluster:
    """Persist derived cluster fields after assignment and summarization."""
    cluster.summary = summary
    cluster.summary_sections = summary_sections
    cluster.embedding = embedding
    cluster.member_count = member_count
    cluster.save(
        update_fields=[
            "summary",
            "summary_sections",
            "embedding",
            "member_count",
        ]
    )
    return cluster


def get_cluster_texts(cluster: CategoryCluster, limit: int = 220) -> list[str]:
    """Fetch the latest cluster texts, oldest-to-newest, capped by limit."""
    queryset = (
        TranscriptSegment.objects.filter(cluster_memberships__cluster=cluster)
        .order_by("-cluster_memberships__position")
        .values_list("text", flat=True)[:limit]
    )
    texts = list(queryset)
    texts.reverse()
    return texts


def get_cluster_member_count(cluster: CategoryCluster) -> int:
    """Return current segment count for a cluster."""
    return ClusterSegment.objects.filter(cluster=cluster).count()


def replace_category_clusters(category: str, categories: dict) -> None:
    """Replace all cluster rows for a category inside one transaction."""
    with transaction.atomic():
        CategoryCluster.objects.filter(category=category).delete()

        for cluster_position, (name, cluster_data) in enumerate(categories.items()):
            cluster = CategoryCluster.objects.create(
                category=category,
                name=name,
                summary=cluster_data.get("summary", "") or "",
                summary_sections=cluster_data.get("summary_sections") or [],
                embedding=cluster_data.get("embedding") or [],
                member_count=len(cluster_data.get("windows", [])),
                position=cluster_position,
            )

            memberships = []
            for window_position, window in enumerate(cluster_data.get("windows", [])):
                segment_id = window.get("segment_id")
                if not segment_id:
                    continue
                memberships.append(
                    ClusterSegment(
                        cluster=cluster,
                        segment_id=segment_id,
                        position=window_position,
                    )
                )

            ClusterSegment.objects.bulk_create(memberships)


def remove_upload(upload: AudioUpload) -> tuple[str, str]:
    """Delete an upload row and its on-disk audio file."""
    stored_name = upload.stored_name
    category = upload.category
    file_path = paths.AUDIO_DIR / stored_name

    upload.delete()
    file_path.unlink(missing_ok=True)

    return stored_name, category


def remove_empty_category_clusters(category: str) -> None:
    """Clean up clusters with no memberships after partial rebuild failures."""
    CategoryCluster.objects.filter(category=category, memberships__isnull=True).delete()


def load_dashboard_data() -> dict:
    """Build the complete dashboard payload expected by the UI layer."""
    uploads = []
    metadata = defaultdict(dict)
    prefetched_uploads = AudioUpload.objects.prefetch_related(
        Prefetch("segments", queryset=TranscriptSegment.objects.order_by("position", "id"))
    )

    for upload in prefetched_uploads:
        segments = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in upload.segments.all()
        ]
        metadata[upload.category][upload.stored_name] = {
            "upload_id": upload.id,
            "audio_file": upload.stored_name,
            "author": upload.author,
            "category": upload.category,
            "date": _date_as_string(upload.recorded_on),
            "transcription": {
                "text": upload.transcription_text,
                "segments": segments,
                "language": upload.language or None,
                "duration": upload.duration,
            },
        }
        uploads.append(
            {
                "id": upload.id,
                "audio_file": upload.stored_name,
                "author": upload.author,
                "category": upload.category,
                "date": _date_as_string(upload.recorded_on),
                "processing_error": upload.processing_error,
            }
        )

    data = defaultdict(dict)
    prefetched_clusters = (
        CategoryCluster.objects.filter(memberships__isnull=False)
        .distinct()
        .prefetch_related(
            Prefetch(
                "memberships",
                queryset=ClusterSegment.objects.select_related("segment__upload").order_by(
                    "position",
                    "segment__position",
                    "id",
                ),
            )
        )
        .order_by("category", "position", "id")
    )

    for cluster in prefetched_clusters:
        windows = []
        for membership in cluster.memberships.all():
            segment = membership.segment
            upload = segment.upload
            windows.append(
                {
                    "upload_id": upload.id,
                    "text": segment.text,
                    "source_file": upload.stored_name,
                    "author": upload.author,
                    "date": _date_as_string(upload.recorded_on),
                    "start": segment.start,
                    "end": segment.end,
                }
            )

        if not windows:
            continue

        data[cluster.category][cluster.name] = {
            "summary": cluster.summary,
            "summary_sections": cluster.summary_sections or [],
            "windows": windows,
        }

    return {
        "data": {category: dict(clusters) for category, clusters in data.items()},
        "metadata": {category: dict(files) for category, files in metadata.items()},
        "uploads": uploads,
    }
