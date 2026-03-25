# VoiceBased

VoiceBased is a Django app that ingests audio files, transcribes them, clusters transcript segments, and serves a searchable summary UI.

## Project Navigation

### Start Here (Most Common Entry Points)

- `uibase/views.py`
  Handles HTTP actions (upload, delete, render dashboard).
- `ai/pipeline.py`
  Orchestrates the end-to-end processing flow after an upload.
- `ai/repository.py`
  Owns all SQLite reads/writes for uploads, segments, clusters, and dashboard payloads.

### Processing Flow

`uibase/views.py -> ai/pipeline.py -> ai/transcription.py -> ai/language_models.py + ai/clustering.py -> ai/repository.py`

### Module Responsibilities

- `ai/transcription.py`
  Transcribes audio with Whisper and returns normalized transcript payloads.
- `ai/language_models.py`
  Creates embeddings and generates category names/summaries with OpenAI.
- `ai/clustering.py`
  Groups embeddings using UMAP + HDBSCAN.
- `uibase/templates/uibase/UI.html`
  Main dashboard structure and browser-side interaction logic.
- `uibase/static/uibase/css/UI.css`
  Full dashboard styling (modern/fluid layout + navigator panel).

## Data Model (SQLite)

- `AudioUpload`
  One row per uploaded file.
- `TranscriptSegment`
  Timestamped transcript chunks linked to one upload.
- `CategoryCluster`
  Summary + centroid for a semantic group inside a category.
- `ClusterSegment`
  Join table linking transcript segments to clusters.

## Vector Strategy

- Per-segment embeddings are generated at processing time but are not persisted in SQLite.
- Each new segment embedding is compared against each existing cluster centroid embedding.
- If similarity is high enough, the segment is assigned to that cluster; otherwise, unmatched segments go through a second clustering pass and may end up in `Uncategorized`.
- After assignment, the cluster centroid embedding is updated with a weighted merge so future uploads can be matched efficiently.

## AI Processing Build (Step by Step)

1. Upload arrives in `uibase/views.py` (`dashboard` POST flow).
2. `ai/pipeline.py::process_uploaded_file` creates/updates the upload row.
3. `ai/transcription.py` generates transcript text + timestamped segments.
4. Segments are stored in SQLite (`TranscriptSegment`), but without persisted per-segment vectors.
5. `ai/language_models.py` generates fresh embeddings for the new segment texts in-memory.
6. `ai/pipeline.py::assign_new_segments_to_clusters` compares each new segment vector to existing cluster centroid vectors.
7. Matched segments are appended to existing clusters; unmatched segments are grouped into new topic clusters or `Uncategorized`.
8. For touched clusters, summaries are regenerated and centroid vectors are updated and persisted.
9. UI payload is rebuilt through `ai/repository.py::load_dashboard_data`.

## Snippet-To-Transcript UX

- Clicking a snippet in the left panel:
  - Opens parent category/subcategory if needed.
  - Loads and seeks audio.
  - Renders a focused transcript window on the right side.
- Transcript windowing is adaptive:
  - shows `3-6` transcript rows based on viewport height,
  - keeps the selected snippet near the top when possible,
  - falls back near the end when fewer rows remain.
- Panel scrolling is local to the right transcript container (not full-page jump).

## Configuration Notes

- Required:
  - `OPENAI_API_KEY` in `.env`
- Optional:
  - `OPENAI_CHAT_MODEL` (defaults to `gpt-4o-mini`)
- If you get `401 invalid_api_key`, verify:
  - correct key value in `.env`,
  - no leading/trailing spaces,
  - environment reload/restart after change.

## Notes for Contributors

- Legacy compatibility wrapper modules were removed to keep the codebase easier to scan.
- Use the new modules directly (`pipeline.py`, `repository.py`, `transcription.py`, `language_models.py`).
- When debugging behavior, begin in `ai/pipeline.py`, then follow the step-specific module it calls.
