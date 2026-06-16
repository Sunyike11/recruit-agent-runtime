# Architecture

Recruit-Graph uses a unified RuntimeEntry path for CLI, `main.py`, and future API entrypoints.

```text
RuntimeEntryHarness
  -> RecruitGraphFactory
      -> Skill Production Graph (default)
      -> Legacy Graph (explicit baseline / fallback)
```

## Skill Production Graph

The default production graph is SkillExecutor-based:

```text
Planner Skill
-> Retriever Skill
-> CandidateProfilePreview v2
-> Matcher Skill
-> ClaimVerify Skill
-> optional Refiner Skill loop
```

Each skill is invoked through the shared `SkillExecutor`, which emits summary-only runtime events. The graph does not call Planner/Retriever/Matcher/Refiner agents directly outside their skill adapters.

## Runtime Layer

The runtime manages:

- session_id
- task_id
- thread_id
- task lifecycle
- graph/node/skill event timeline
- SQLite metadata persistence

Events are designed to avoid full JD, resume text, prompt, LLM response, reasoning, and secret values.

## Legacy Baseline

The Legacy Graph is still available through:

```bash
python scripts/run_recruit_runtime.py --graph-mode legacy --jd "..."
```

It is used for:

- compatibility validation
- A/B comparison
- hard-failure rollback

It is no longer the default graph mode.

## Local Storage

Private runtime artifacts are intentionally excluded from Git:

- `data/`
- `chroma_db/`
- `storage/`
- `evaluation_indexes/`
- `evaluation_results/`
