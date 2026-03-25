import json
from datetime import date
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from ai import paths
from ai.language_models import KnowledgeModelClient
from ai.pipeline import rebuild_category
from ai.repository import (
    create_or_update_upload,
    replace_upload_segments,
    update_upload_transcription,
)
from uibase.models import AudioUpload, CategoryCluster


def _load_json_file(source: Path) -> dict:
    return json.loads(source.read_text(encoding="utf-8"))


def _pick_source(explicit_source: str | None) -> Path:
    if explicit_source:
        candidate = Path(explicit_source)
        if not candidate.exists():
            raise CommandError(f"Source file does not exist: {candidate}")
        return candidate

    candidates = [
        paths.DATABASE_FILE,
        paths.PROJECT_ROOT / "data_no_embeddings.json",
    ]

    for candidate in candidates:
        if not candidate.exists() or candidate.stat().st_size == 0:
            continue

        try:
            _load_json_file(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    raise CommandError("No valid legacy JSON source was found.")


def _parse_recorded_on(value: str | None) -> date:
    if not value:
        return timezone.localdate()

    try:
        return date.fromisoformat(value)
    except ValueError:
        return timezone.localdate()


class Command(BaseCommand):
    help = "Imports legacy JSON metadata/transcripts into SQLite and rebuilds clusters."

    def add_arguments(self, parser):
        parser.add_argument(
            "--source",
            type=str,
            help="Optional path to a legacy JSON file.",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete existing SQL-backed uploads and clusters before importing.",
        )

    def handle(self, *args, **options):
        source = _pick_source(options.get("source"))
        legacy = _load_json_file(source)
        metadata = legacy.get("metadata") or {}

        if not metadata:
            raise CommandError(f"No metadata found in {source}")

        if options.get("clear"):
            self.stdout.write("Clearing existing SQLite data...")
            CategoryCluster.objects.all().delete()
            AudioUpload.objects.all().delete()

        model_client = KnowledgeModelClient()
        import_count = 0
        category_names = set()

        for files in metadata.values():
            for file_name, item in files.items():
                category = (item.get("category") or "").strip()
                if not category:
                    continue

                category_names.add(category)
                transcription = item.get("transcription") or {}
                segments = [
                    {
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "text": (segment.get("text") or "").strip(),
                    }
                    for segment in transcription.get("segments", [])
                    if (segment.get("text") or "").strip()
                ]

                upload = create_or_update_upload(
                    audio_file=file_name,
                    original_name=item.get("audio_file") or file_name,
                    author=(item.get("author") or "").strip(),
                    category=category,
                    recorded_on=_parse_recorded_on(item.get("date")),
                )
                update_upload_transcription(upload, transcription)

                embeddings = (
                    model_client.embed_texts([segment["text"] for segment in segments])
                    if segments
                    else []
                )
                replace_upload_segments(upload, segments, embeddings)
                import_count += 1

        for category in sorted(category_names):
            self.stdout.write(f"Rebuilding category '{category}'...")
            rebuild_category(category, model_client=model_client)

        self.stdout.write(
            self.style.SUCCESS(
                f"Imported {import_count} upload(s) from {source} into SQLite."
            )
        )
