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


def transcribe_audio_file(audio_file_name: str, machine_category: str) -> dict:
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

    segments_generator, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        initial_prompt=f"This is about: {machine_category}",
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
