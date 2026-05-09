# Agent Guide

This folder defines reusable specialist briefs for this repository.

They are not plugins and they are not permanent background workers. They are
project-specific role descriptions that we can use when we want to delegate a
task to a focused Codex sub-agent.

## Why These Agents Exist

This project naturally splits into four work areas:

- `AI`
  - Transcription, embeddings, clustering, summaries, grounding, token usage.
- `UI`
  - Django templates, CSS, browser-side interaction, responsive behavior.
- `Security`
  - Upload safety, deletion safety, secret handling, validation, review.
- `Structure`
  - Architecture, module boundaries, naming, docs, comments, maintainability.

Using these roles keeps tasks tighter, lowers context switching, and makes it
easier to review changes.

## Agent List

- `ai-agent.md`
  Focused on model behavior, clustering, summary quality, and AI cost/quality
  tradeoffs.
- `ui-agent.md`
  Focused on dashboard UX, tree navigation, readability, polish, and frontend
  behavior.
- `security-agent.md`
  Focused on review and hardening across uploads, secrets, file handling, and
  request validation.
- `structure-agent.md`
  Focused on project organization, maintainability, documentation, and cleanup.

## Recommended Usage

Use these as specialist roles when the task is clearly scoped.

- Use the `AI agent` when the problem is about snippet quality, cluster
  assignment, summaries, embeddings, or prompt behavior.
- Use the `UI agent` when the problem is about layout, interaction, taxonomy
  navigation, accessibility, readability, or responsive issues.
- Use the `Security agent` when you want a review, risk analysis, or a hardening
  pass.
- Use the `Structure agent` when the project feels harder to understand or
  change safely.

## Good Delegation Pattern

Give the agent:

- one clear objective
- file ownership or likely file scope
- expected output
- constraints

Example:

```text
Review the AI clustering flow for low-signal snippets.
Focus on ai/pipeline.py, ai/language_models.py, and ai/clustering.py.
Find why promo/intro text still leaks into real summaries.
Propose or implement the smallest safe fix and report changed files.
```

## File Ownership Guidance

To reduce overlap when multiple agents work in parallel, prefer these default
write scopes:

- `AI agent`
  - `ai/`
  - `uibase/views.py` only when upload/processing context changes require it
- `UI agent`
  - `uibase/templates/`
  - `uibase/static/`
  - `uibase/views.py` only for UI payload shaping
- `Security agent`
  - review-first across `uibase/`, `ai/`, `umenergy/settings.py`
- `Structure agent`
  - repo-wide docs, comments, naming, service boundaries, cleanup work

## Important Rule

The `Security agent` should usually act as a reviewer first, not a broad code
editor. That keeps risk work careful and easier to verify.

## Notes

- These briefs are intentionally repo-specific.
- If the system changes a lot, update these files so the agent roles stay
  accurate.
- If one role starts to repeat the same workflow many times, that is a good sign
  we should create a dedicated skill next.
