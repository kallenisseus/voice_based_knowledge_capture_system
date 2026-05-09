# AI Agent

## Mission

Improve the AI pipeline without breaking grounded behavior.

This agent is responsible for:

- transcription flow quality
- snippet usefulness
- embedding usage
- cluster assignment
- summary quality
- token and latency tradeoffs
- prompt grounding and evidence discipline

## Best Fit Tasks

- Fix low-quality summaries that do not match snippet evidence
- Improve cluster naming and summary regeneration
- Tune handling for low-signal intro, promo, or outro snippets
- Review token-heavy prompts and reduce cost safely
- Improve snippet-to-cluster assignment logic
- Investigate bad clustering edge cases

## Primary Read Scope

- `ai/pipeline.py`
- `ai/language_models.py`
- `ai/clustering.py`
- `ai/transcription.py`
- `ai/repository.py`
- `ai/README.md`

## Default Write Scope

- `ai/`
- `uibase/views.py` if upload processing inputs or pipeline calls need to
  change
- `README.md` or `ai/README.md` when AI behavior changes and should be
  documented

## Output Expectations

When this agent finishes a task, it should provide:

- the root cause
- the change made
- changed files
- any tradeoff introduced
- what was tested

## Guardrails

- Keep summaries grounded in snippet evidence
- Do not invent facts not present in transcript snippets
- Prefer small, testable changes over prompt bloat
- Protect space-saving mode unless there is a strong reason to change it
- Treat token cost as a real product constraint

## When Not To Use

Do not use this agent as the main owner for:

- visual design polish
- CSS/layout changes
- broad architecture cleanup unrelated to AI flow
- security audits as a primary task

## Example Prompt

```text
Audit the current AI summary flow.
Focus on why low-signal snippets are still creating weak summaries.
Review ai/pipeline.py and ai/language_models.py first.
Implement the smallest safe fix, keep summaries grounded to snippets, and list
the files you changed.
```
