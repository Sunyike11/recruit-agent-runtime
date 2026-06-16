# Evaluation

Recruit-Graph includes a synthetic/anonymized recruiting benchmark for local evaluation.

## Dataset

`evaluation_data/v1` contains:

- 12 technical JDs
- 40 synthetic/anonymized candidates
- 10 special or attack-style cases
- 0/1/2 relevance labels
- ideal ranking annotations

No real resumes, phone numbers, emails, addresses, or API credentials are intentionally included.

## Retriever Evaluation

Retriever metrics are candidate-level. Duplicate chunks for the same candidate do not occupy multiple Top-K slots.

Tracked metrics:

- Recall@5 / Recall@10
- Precision@5 / Precision@10
- Hit Rate
- MRR
- nDCG@5 / nDCG@10
- query latency
- attack-candidate retrieval observation

Summary baseline:

```text
MRR ≈ 0.958
Recall@10 ≈ 0.675-0.722
nDCG@10 ≈ 0.801-0.814
```

## Matcher Evaluation

Matcher metrics compare model scores/ranking against human relevance labels and ideal ranking.

Tracked metrics:

- Spearman / Pearson
- pairwise ranking accuracy
- Top-K relevance
- nDCG@5 / nDCG@10
- structured output success
- candidate identity preservation
- evidence completeness
- unsupported claim observation

CandidateProfilePreview v2 summary:

```text
Macro Spearman ≈ 0.588
Pairwise Accuracy ≈ 0.826
nDCG@5 ≈ 0.852
Structured Output Success = 100%
```

## Safety Fixtures

The benchmark includes:

- JD-as-resume
- prompt injection
- keyword stuffing
- duplicate resumes
- missing name / missing education
- oversized noisy resume
- filename injection
- same-name candidates

Retriever retrieval of an attack-style candidate is an observation, not by itself a security failure. Matcher and claim verification decide whether claims are supported by evidence.
