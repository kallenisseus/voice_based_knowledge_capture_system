# AI Workflow

The `ai/` package is organized by responsibility so each module has one clear job.

## Processing Order

`pipeline.py -> transcription.py -> language_models.py -> clustering.py -> repository.py`

## Module Guide

1. `pipeline.py`
   Orchestrates upload processing, cluster assignment, and summary refresh.
2. `transcription.py`
   Converts audio into timestamped transcript segments.
3. `language_models.py`
   Generates embeddings and LLM-based naming/summaries.
4. `clustering.py`
   Performs UMAP dimensionality reduction and HDBSCAN clustering.
5. `repository.py`
   Handles all SQLite persistence and dashboard payload assembly.
6. `paths.py`
   Shared path constants (including audio storage path).

## Embedding + Matching Behavior

- New transcript segments are embedded during processing.
- Those vectors are compared to stored cluster centroid vectors, if any segments similarity is above the threshold (SIMILARITY_THRESHOLD = 0.60) it gets assigned to a existing cluster, the best cluster is picked.
- Segment-level vectors are intentionally not stored long-term (space-saving mode).
- Cluster centroid vectors are persisted and updated after each assignment so matching quality improves over time.
- Low-signal promo/intro/outro snippets are forced to `Uncategorized` instead of creating technical clusters.

## Grounded Summary Behavior

- New summary evidence always comes from snippet text.
- Prompt rules explicitly forbid adding facts not present in snippets.
- Default ingestion uses incremental mode: existing summary is a baseline and only new snippet evidence is added.
- Manual redo endpoints use full mode: regenerate from current cluster snippets.
- If model output fails validation, local fallback summary stays extractive and grounded to snippet text (no generic invented safety/failure sections).

## Taxonomy Context

- Uploads now carry machine taxonomy metadata:
  - `machine_name` (required)
  - `machine_type` (fallbacks to `Unassigned`)
  - `extra_tags` (optional list)
- New upload UI is tags-first; legacy `subcategory_paths`/`hierarchy_path` fields remain supported for existing data.
- The clustering bucket remains machine-specific (`AudioUpload.category` key), while UI grouping is type-first.
- Summarization and cluster naming prompts include machine/type/subcategory/hierarchy/tag context to keep AI output anchored to the right asset context.
