"""Database access helpers for the audio knowledge-base domain.

This module is intentionally focused on persistence concerns only:
- creating/updating upload and transcript rows
- reading payloads needed by clustering/summarization
- creating/updating cluster membership and summary records

Keeping all ORM queries here lets the pipeline module stay focused on
business flow, while this file remains the single place to adjust data
shape and query performance.
"""

import json
from collections import defaultdict

from django.db import transaction
from django.db.models import Max, Prefetch

from ai import paths
from uibase.models import (
    AudioUpload,
    CategoryCluster,
    ClusterSegment,
    MachineStyle,
    MachineTypeStyle,
    TranscriptSegment,
)


MACHINE_TYPE_OVERRIDES = {
    "car_brakes": "Car",
    "car brakes": "Car",
}


def _date_as_string(value) -> str:
    """Normalize date-like values before exposing them to templates/API payloads."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def _normalize_hex_color(value: str, fallback: str = "#3A78F2") -> str:
    """Normalize user-provided hex colors to #RRGGBB."""
    text = str(value or "").strip()
    if not text:
        return fallback
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return fallback
    try:
        int(text, 16)
    except ValueError:
        return fallback
    return f"#{text.upper()}"


def _normalize_string_list(values) -> list[str]:
    """Return a cleaned list of non-empty strings with duplicates removed in order."""
    if not values:
        return []

    if isinstance(values, str):
        values = [values]

    cleaned = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _normalize_subcategory_paths(values) -> list[list[str]]:
    """Normalize optional subcategory path input into a list of clean paths."""
    if not values:
        return []

    # Accept JSON-encoded values from hidden form fields when needed.
    if isinstance(values, str):
        raw_text = values.strip()
        if not raw_text:
            return []
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = [raw_text]
        values = parsed

    if not isinstance(values, (list, tuple)):
        return []

    normalized_paths = []
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
        normalized_paths.append(path)

    return normalized_paths


def _merge_subcategory_paths(existing_paths, incoming_paths) -> list[list[str]]:
    """Merge two subcategory path collections while preserving first-seen order."""
    merged = []
    seen = set()

    for collection in (existing_paths or [], incoming_paths or []):
        for path in _normalize_subcategory_paths(collection):
            key = tuple(part.casefold() for part in path)
            if key in seen:
                continue
            seen.add(key)
            merged.append(path)

    return merged


def _subcategory_paths_to_labels(paths: list[list[str]]) -> list[str]:
    """Render subcategory paths as UI-friendly labels."""
    return [" > ".join(path) for path in paths if path]


def _select_primary_subcategory_path(paths: list[list[str]], fallback=None) -> list[str]:
    """Pick the most stable path to use where a single path is needed."""
    normalized = _normalize_subcategory_paths(paths)
    if normalized:
        return normalized[0]
    fallback_normalized = _normalize_string_list(fallback)
    return fallback_normalized


def _resolve_machine_type(machine_type: str | None, machine_name: str | None) -> str:
    """Normalize machine type and auto-assign known machines to stable top-level types."""
    machine_name_value = (machine_name or "").strip()
    machine_type_value = (machine_type or "").strip()
    if machine_type_value and machine_type_value.casefold() != "unassigned":
        return machine_type_value

    override = MACHINE_TYPE_OVERRIDES.get(machine_name_value.casefold())
    if override:
        return override

    return machine_type_value or "Unassigned"


def create_or_update_upload(
    *,
    audio_file: str,
    author: str,
    machine_name: str,
    machine_type: str = "Unassigned",
    subcategory_paths=None,
    hierarchy_path=None,
    extra_tags=None,
    category: str | None = None,
    recorded_on,
    original_name: str | None = None,
) -> AudioUpload:
    """Upsert one uploaded file record keyed by stored filename."""
    machine_name_value = (machine_name or "").strip()
    machine_type_value = _resolve_machine_type(machine_type, machine_name_value)
    category_value = (category or machine_name_value or "Unassigned").strip()
    normalized_subcategory_paths = _normalize_subcategory_paths(subcategory_paths)
    primary_hierarchy_path = _select_primary_subcategory_path(
        normalized_subcategory_paths,
        fallback=hierarchy_path,
    )

    upload, _ = AudioUpload.objects.update_or_create(
        stored_name=audio_file,
        defaults={
            "original_name": original_name or audio_file,
            "author": author,
            "machine_name": machine_name_value or category_value,
            "machine_type": machine_type_value,
            "subcategory_paths": normalized_subcategory_paths,
            "hierarchy_path": primary_hierarchy_path,
            "extra_tags": _normalize_string_list(extra_tags),
            "category": category_value,
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
        subcategory_paths = _normalize_subcategory_paths(segment.upload.subcategory_paths)
        hierarchy_path = _select_primary_subcategory_path(
            subcategory_paths,
            fallback=segment.upload.hierarchy_path,
        )
        payloads.append(
            {
                "segment_id": segment.id,
                "text": segment.text,
                "source_file": segment.upload.stored_name,
                "author": segment.upload.author,
                "date": _date_as_string(segment.upload.recorded_on),
                "machine_name": segment.upload.machine_name or segment.upload.category,
                "machine_type": _resolve_machine_type(
                    segment.upload.machine_type,
                    segment.upload.machine_name or segment.upload.category,
                ),
                "subcategory_paths": subcategory_paths,
                "subcategory_labels": _subcategory_paths_to_labels(subcategory_paths),
                "hierarchy_path": hierarchy_path,
                "extra_tags": _normalize_string_list(segment.upload.extra_tags),
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
    mark_fresh: bool = True,
) -> CategoryCluster:
    """Persist derived cluster fields after assignment and summarization."""
    cluster.summary = summary
    cluster.summary_sections = summary_sections
    cluster.embedding = embedding
    cluster.member_count = member_count
    if mark_fresh:
        cluster.needs_resummary = False
        cluster.stale_deleted_count = 0
        cluster.stale_deleted_files = []

    update_fields = [
        "summary",
        "summary_sections",
        "embedding",
        "member_count",
    ]
    if mark_fresh:
        update_fields.extend(
            [
                "needs_resummary",
                "stale_deleted_count",
                "stale_deleted_files",
            ]
        )
    cluster.save(
        update_fields=update_fields
    )
    return cluster


def flag_cluster_as_stale(cluster: CategoryCluster, deleted_files: list[str]) -> CategoryCluster:
    """Mark one cluster as needing re-summary after upload deletions."""
    existing_files = _normalize_string_list(cluster.stale_deleted_files)
    merged_files = _normalize_string_list([*existing_files, *deleted_files])

    cluster.needs_resummary = True
    cluster.stale_deleted_files = merged_files
    cluster.stale_deleted_count = len(merged_files)
    cluster.save(
        update_fields=[
            "needs_resummary",
            "stale_deleted_files",
            "stale_deleted_count",
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


def get_category_context(category: str) -> dict:
    """Return machine/taxonomy context for one clustering category key."""
    upload = AudioUpload.objects.filter(category=category).order_by("-created_at", "-id").first()
    if not upload:
        category_value = (category or "").strip() or "Unknown Machine"
        return {
            "machine_name": category_value,
            "machine_type": "Unassigned",
            "subcategory_paths": [],
            "subcategory_labels": [],
            "hierarchy_path": [],
            "extra_tags": [],
            "category_key": category_value,
        }

    machine_name = (upload.machine_name or upload.category or "").strip() or "Unknown Machine"
    subcategory_paths = _normalize_subcategory_paths(upload.subcategory_paths)
    hierarchy_path = _select_primary_subcategory_path(
        subcategory_paths,
        fallback=upload.hierarchy_path,
    )
    return {
        "machine_name": machine_name,
        "machine_type": _resolve_machine_type(upload.machine_type, machine_name),
        "subcategory_paths": subcategory_paths,
        "subcategory_labels": _subcategory_paths_to_labels(subcategory_paths),
        "hierarchy_path": hierarchy_path,
        "extra_tags": _normalize_string_list(upload.extra_tags),
        "category_key": upload.category,
    }


def load_dashboard_data() -> dict:
    """Build the complete dashboard payload expected by the UI layer."""
    uploads = []
    metadata = defaultdict(dict)
    machine_context_by_name = {}
    machine_type_colors = {
        style.machine_type: _normalize_hex_color(style.color_hex)
        for style in MachineTypeStyle.objects.all()
    }
    machine_colors = {
        style.machine_name: _normalize_hex_color(style.color_hex)
        for style in MachineStyle.objects.all()
    }
    prefetched_uploads = AudioUpload.objects.prefetch_related(
        Prefetch("segments", queryset=TranscriptSegment.objects.order_by("position", "id"))
    )

    for upload in prefetched_uploads:
        machine_name = (upload.machine_name or upload.category or "").strip() or "Unknown Machine"
        machine_type = _resolve_machine_type(upload.machine_type, machine_name)
        subcategory_paths = _normalize_subcategory_paths(upload.subcategory_paths)
        hierarchy_path = _select_primary_subcategory_path(
            subcategory_paths,
            fallback=upload.hierarchy_path,
        )
        subcategory_labels = _subcategory_paths_to_labels(subcategory_paths)
        extra_tags = _normalize_string_list(upload.extra_tags)
        machine_context = machine_context_by_name.setdefault(
            machine_name,
            {
                "machine_name": machine_name,
                "machine_type": machine_type,
                "subcategory_paths": [],
                "subcategory_labels": [],
                "hierarchy_path": hierarchy_path,
                "extra_tags": extra_tags,
            },
        )
        machine_context["machine_type"] = machine_type
        machine_context["subcategory_paths"] = _merge_subcategory_paths(
            machine_context.get("subcategory_paths"),
            subcategory_paths,
        )
        machine_context["subcategory_labels"] = _subcategory_paths_to_labels(
            machine_context["subcategory_paths"]
        )
        machine_context["hierarchy_path"] = _select_primary_subcategory_path(
            machine_context["subcategory_paths"],
            fallback=machine_context.get("hierarchy_path"),
        )
        machine_context["extra_tags"] = _normalize_string_list(
            [*(machine_context.get("extra_tags") or []), *extra_tags]
        )

        segments = [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in upload.segments.all()
        ]
        metadata[machine_type][upload.stored_name] = {
            "upload_id": upload.id,
            "audio_file": upload.stored_name,
            "author": upload.author,
            "category": upload.category,
            "machine_name": machine_name,
            "machine_type": machine_type,
            "subcategory_paths": subcategory_paths,
            "subcategory_labels": subcategory_labels,
            "hierarchy_path": hierarchy_path,
            "extra_tags": extra_tags,
            "type_color": machine_type_colors.get(machine_type, "#3A78F2"),
            "machine_color": machine_colors.get(machine_name, machine_type_colors.get(machine_type, "#3A78F2")),
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
                "machine_name": machine_name,
                "machine_type": machine_type,
                "subcategory_paths": subcategory_paths,
                "subcategory_labels": subcategory_labels,
                "hierarchy_path": hierarchy_path,
                "extra_tags": extra_tags,
                "type_color": machine_type_colors.get(machine_type, "#3A78F2"),
                "machine_color": machine_colors.get(machine_name, machine_type_colors.get(machine_type, "#3A78F2")),
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
        machine_name = (cluster.category or "").strip() or "Unknown Machine"
        machine_context = machine_context_by_name.get(
            machine_name,
            {
                "machine_name": machine_name,
                "machine_type": _resolve_machine_type(None, machine_name),
                "subcategory_paths": [],
                "subcategory_labels": [],
                "hierarchy_path": [],
                "extra_tags": [],
            },
        )
        machine_type = _resolve_machine_type(machine_context.get("machine_type"), machine_name)
        machine_paths = _normalize_subcategory_paths(machine_context.get("subcategory_paths"))
        path_counts = defaultdict(int)
        topic_name = (cluster.name or "Uncategorized").strip() or "Uncategorized"

        windows = []
        for membership in cluster.memberships.all():
            segment = membership.segment
            upload = segment.upload
            upload_machine_name = (upload.machine_name or upload.category or "").strip() or machine_name
            upload_subcategory_paths = _normalize_subcategory_paths(upload.subcategory_paths)
            upload_hierarchy_path = _select_primary_subcategory_path(
                upload_subcategory_paths,
                fallback=upload.hierarchy_path,
            )
            if not upload_subcategory_paths and upload_hierarchy_path:
                upload_subcategory_paths = [upload_hierarchy_path]
            for path in upload_subcategory_paths:
                path_counts[tuple(path)] += 1

            windows.append(
                {
                    "upload_id": upload.id,
                    "segment_id": segment.id,
                    "text": segment.text,
                    "source_file": upload.stored_name,
                    "author": upload.author,
                    "date": _date_as_string(upload.recorded_on),
                    "machine_name": upload_machine_name,
                    "machine_type": _resolve_machine_type(
                        upload.machine_type,
                        upload_machine_name,
                    ),
                    "subcategory_paths": upload_subcategory_paths,
                    "subcategory_labels": _subcategory_paths_to_labels(upload_subcategory_paths),
                    "hierarchy_path": upload_hierarchy_path,
                    "extra_tags": _normalize_string_list(upload.extra_tags),
                    "start": segment.start,
                    "end": segment.end,
                }
            )

        if not windows:
            continue

        if path_counts:
            ranked_paths = sorted(
                path_counts.items(),
                key=lambda item: (
                    -item[1],
                    len(item[0]),
                    tuple(part.casefold() for part in item[0]),
                ),
            )
            cluster_subcategory_paths = [list(path_key) for path_key, _ in ranked_paths]
            hierarchy_path = cluster_subcategory_paths[0]
        else:
            cluster_subcategory_paths = machine_paths
            hierarchy_path = _select_primary_subcategory_path(
                cluster_subcategory_paths,
                fallback=machine_context.get("hierarchy_path"),
            )

        machine_path_parts = [machine_name, *hierarchy_path]
        machine_path = " > ".join(part for part in machine_path_parts if part)
        subcategory_name = f"{machine_path} / {topic_name}" if machine_path else topic_name

        data[machine_type][subcategory_name] = {
            "cluster_id": cluster.id,
            "summary": cluster.summary,
            "summary_sections": cluster.summary_sections or [],
            "windows": windows,
            "machine_name": machine_name,
            "machine_type": machine_type,
            "subcategory_paths": cluster_subcategory_paths,
            "subcategory_labels": _subcategory_paths_to_labels(cluster_subcategory_paths),
            "hierarchy_path": hierarchy_path,
            "extra_tags": _normalize_string_list(machine_context.get("extra_tags")),
            "type_color": machine_type_colors.get(machine_type, "#3A78F2"),
            "machine_color": machine_colors.get(machine_name, machine_type_colors.get(machine_type, "#3A78F2")),
            "cluster_name": topic_name,
            "needs_resummary": bool(cluster.needs_resummary),
            "stale_deleted_count": int(cluster.stale_deleted_count or 0),
            "stale_deleted_files": _normalize_string_list(cluster.stale_deleted_files),
        }

    return {
        "data": {category: dict(clusters) for category, clusters in data.items()},
        "metadata": {category: dict(files) for category, files in metadata.items()},
        "uploads": uploads,
        "styles": {
            "machine_types": machine_type_colors,
            "machines": machine_colors,
        },
    }
