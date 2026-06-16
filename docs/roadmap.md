# Roadmap

## Current

- CLI-based local recruiting runtime.
- Skill Production Graph as default.
- Legacy fallback and comparison path.
- Local PDF resume ingestion into Chroma.
- SQLite task/event persistence.
- Synthetic/anonymized evaluation dataset and metrics.
- Observation-only claim verification after Matcher reports.

## Coming Soon

### Real Candidate MCP Server

Read-only tools:

- `search_candidates`
- `get_candidate_profile`
- `get_resume_evidence`

### FastAPI Backend

Planned backend capabilities:

- resume upload
- async matching tasks
- task status polling
- runtime timeline inspection

### Observability

Planned operational views:

- task success/failure
- skill duration
- fallback usage
- retrieval and matcher quality metrics

## Not Claimed Yet

The current public repository does not claim:

- multi-tenant production service
- web resume upload
- Redis/Celery/PostgreSQL deployment
- full dashboard observability
- production MCP server
