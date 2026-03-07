from faster_whisper import WhisperModel
import os
from ai import paths


# -------------------- Model --------------------

MODEL_SIZE = "tiny"   # try "tiny", "base", "small", "medium"
DEVICE = "cpu"         # change to "cuda" if you have NVIDIA GPU set up
COMPUTE_TYPE = "int8"  # good default for CPU compute_type="float16" or int8_float16

_model = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
    return _model


# -------------------- Transcription --------------------

def transcribe_with_whisper(audio_path: str, machine_category: str) -> dict:
    """
    Returns text + segments (with timestamps) + language + duration.
    Uses faster-whisper locally.
    """
    model = get_model()
    print(f"Transcribing: {audio_path}")

    segments_generator, info = model.transcribe(
        audio_path,
        beam_size=5,
        initial_prompt=f"This is about: {machine_category}"
    )

    # faster-whisper returns a generator, so transcription really happens
    # when we iterate / convert to list
    raw_segments = list(segments_generator)

    segments = []
    full_text_parts = []

    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue

        segments.append({
            "start": seg.start,
            "end": seg.end,
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
    """
    Combine segments into chunks that don't exceed max_words.
    Returns a list of dicts with 'text', 'start', and 'end' timestamps.
    """
    if not segments:
        return []

    combined_segments = []
    current_chunk = ""
    chunk_start = None
    chunk_end = None

    for seg in segments:
        text = seg["text"].strip()
        word_count_current = len(current_chunk.split()) if current_chunk else 0
        word_count_new = len(text.split())

        if word_count_current + word_count_new > max_words:
            if current_chunk:
                combined_segments.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": current_chunk.strip()
                })

            current_chunk = text
            chunk_start = seg["start"]
            chunk_end = seg["end"]
        else:
            if current_chunk:
                current_chunk += " " + text
                chunk_end = seg["end"]
            else:
                current_chunk = text
                chunk_start = seg["start"]
                chunk_end = seg["end"]

    if current_chunk:
        combined_segments.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": current_chunk.strip()
        })

    return combined_segments


# -------------------- Main --------------------

def transcribe(audio_file_name, database, metadata):
    audio_file = os.path.join(paths.DATA_ROOT, "audio", audio_file_name)

    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        return []

    transcription_data = transcribe_with_whisper(audio_file, metadata["category"])
    metadata["transcription"] = transcription_data

    print("\n" + "=" * 60)
    print("Transcription complete!")

    return [s["text"] for s in transcription_data["segments"]]