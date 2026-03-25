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
