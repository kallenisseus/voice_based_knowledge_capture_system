"""Whisper transcription utilities.

This module isolates speech-to-text concerns so the pipeline can treat
transcription as a single step and keep orchestration logic separate.
"""

from faster_whisper import WhisperModel

from ai import paths


MODEL_SIZE = "tiny"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

_model = None


def get_model():
    """Lazy-load a single Whisper model instance for reuse across uploads."""
    global _model

    if _model is None:
        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )

    return _model


def transcribe_audio_file(
    *,
    audio_file_name: str,
    machine_name: str,
    machine_type: str = "Unassigned",
    subcategory_paths=None,
    hierarchy_path=None,
    extra_tags=None,
) -> dict:
    """Transcribe one saved audio file and return normalized transcript payload."""
    audio_path = paths.AUDIO_DIR / audio_file_name

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return {
            "text": "",
            "segments": [],
            "language": None,
            "duration": 0.0,
        }

    model = get_model()
    print(f"Transcribing: {audio_path}")

    normalized_subcategory_paths = []
    for raw_path in (subcategory_paths or []):
        if isinstance(raw_path, str):
            parts = [part.strip() for part in raw_path.split(">")]
        elif isinstance(raw_path, (list, tuple)):
            parts = [str(value).strip() for value in raw_path]
        else:
            continue
        path = [part for part in parts if part]
        if path:
            normalized_subcategory_paths.append(path)

    hierarchy_path = [str(value).strip() for value in (hierarchy_path or []) if str(value).strip()]
    extra_tags = [str(value).strip() for value in (extra_tags or []) if str(value).strip()]
    context_parts = [
        f"machine: {machine_name or 'unknown'}",
        f"type: {machine_type or 'Unassigned'}",
    ]
    if normalized_subcategory_paths:
        context_parts.append(
            "subcategory paths: " + "; ".join(" > ".join(path) for path in normalized_subcategory_paths[:8])
        )
    if hierarchy_path:
        context_parts.append("hierarchy: " + " > ".join(hierarchy_path))
    if extra_tags:
        context_parts.append("tags: " + ", ".join(extra_tags))

    segments_generator, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        initial_prompt="This recording context is " + " | ".join(context_parts),
    )

    raw_segments = list(segments_generator)
    segments = []
    full_text_parts = []

    for segment in raw_segments:
        text = segment.text.strip()
        if not text:
            continue

        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
        })
        full_text_parts.append(text)

    combined_segments = combine_segments(segments, max_words=120)
    duration = raw_segments[-1].end if raw_segments else 0.0

    return {
        "text": " ".join(full_text_parts).strip(),
        "segments": combined_segments,
        "language": info.language,
        "duration": duration,
    }


def combine_segments(segments: list, max_words: int = 120) -> list:
    """Merge small Whisper segments into larger chunks for clustering/summaries."""
    if not segments:
        return []

    combined_segments = []
    current_chunk = ""
    chunk_start = None
    chunk_end = None

    for segment in segments:
        text = segment["text"].strip()
        current_words = len(current_chunk.split()) if current_chunk else 0
        new_words = len(text.split())

        if current_words + new_words > max_words:
            if current_chunk:
                combined_segments.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": current_chunk.strip(),
                })

            current_chunk = text
            chunk_start = segment["start"]
            chunk_end = segment["end"]
            continue

        if current_chunk:
            current_chunk += " " + text
        else:
            current_chunk = text
            chunk_start = segment["start"]

        chunk_end = segment["end"]

    if current_chunk:
        combined_segments.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": current_chunk.strip(),
        })

    return combined_segments
