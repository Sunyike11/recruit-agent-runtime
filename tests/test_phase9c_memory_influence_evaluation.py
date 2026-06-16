import builtins
import json
from pathlib import Path

from src.runtime.memory_influence import (
    MemoryInfluenceEvalCase,
    MemoryInfluenceEvaluator,
    MemoryInfluenceRunSummary,
)


SENSITIVE_JD = "PRIVATE-JD-MUST-NOT-LEAK"
SENSITIVE_MEMORY = "PRIVATE-MEMORY-MUST-NOT-LEAK"
SENSITIVE_REASONING = "PRIVATE-REASONING-MUST-NOT-LEAK"


def summary(
    *,
    status="ok",
    candidate_count=2,
    report_count=2,
    scores=None,
    candidate_ids=None,
    memory_context_provided=False,
    memory_context_eligible_count=0,
    error_type="",
):
    return {
        "status": status,
        "runner_used": "skill_backed_variant",
        "candidate_count": candidate_count,
        "report_count": report_count,
        "top_score_present": bool(report_count),
        "top_scores": scores if scores is not None else [90.0, 80.0],
        "candidate_ids": candidate_ids if candidate_ids is not None else ["candidate_a", "candidate_b"],
        "candidate_profile_preview_count": candidate_count,
        "memory_context_provided": memory_context_provided,
        "memory_context_eligible_count": memory_context_eligible_count,
        "memory_context_rendered_char_count": 80 if memory_context_provided else 0,
        "output_keys": ["candidate_count", "report_count", "top_scores"],
        "error_type": error_type,
        "metadata": {
            "full_memory": SENSITIVE_MEMORY,
            "reasoning": SENSITIVE_REASONING,
        },
    }


def test_no_memory_vs_with_memory_no_effect_is_low_risk():
    evaluator = MemoryInfluenceEvaluator()
    delta = evaluator.compare_summaries(
        summary(memory_context_provided=False),
        summary(memory_context_provided=True, memory_context_eligible_count=1),
    )

    assert delta.decision == "no_effect"
    assert delta.risk_level == "low"
    assert delta.memory_context_used is True


def test_candidate_count_change_is_warning_medium_risk():
    delta = MemoryInfluenceEvaluator().compare_summaries(
        summary(candidate_count=2),
        summary(candidate_count=3, memory_context_provided=True, memory_context_eligible_count=1),
    )

    assert delta.candidate_count_changed is True
    assert delta.decision == "warning"
    assert delta.risk_level == "medium"


def test_expected_effect_marks_expected_without_claiming_quality():
    delta = MemoryInfluenceEvaluator().compare_summaries(
        summary(candidate_count=2),
        summary(candidate_count=3, memory_context_provided=True, memory_context_eligible_count=1),
        expected_effect="candidate_count",
    )

    assert delta.decision == "expected_effect"
    assert delta.risk_level == "medium"


def test_with_memory_failure_after_no_memory_success_is_regression_risk():
    delta = MemoryInfluenceEvaluator().compare_summaries(
        summary(status="ok"),
        summary(status="failed", report_count=0, memory_context_provided=True, error_type="RuntimeError"),
    )

    assert delta.decision == "regression_risk"
    assert delta.risk_level == "high"


def test_with_memory_missing_context_is_skipped():
    delta = MemoryInfluenceEvaluator().compare_summaries(
        summary(status="ok"),
        summary(status="ok", memory_context_provided=False),
    )

    assert delta.decision == "skipped"
    assert delta.risk_level == "medium"
    assert delta.memory_context_used is False


def test_ranking_changed_detection():
    delta = MemoryInfluenceEvaluator().compare_summaries(
        summary(candidate_ids=["candidate_a", "candidate_b"]),
        summary(
            candidate_ids=["candidate_b", "candidate_a"],
            memory_context_provided=True,
            memory_context_eligible_count=1,
        ),
    )

    assert delta.ranking_changed is True
    assert delta.candidate_ids_changed is True
    assert delta.decision == "warning"


def test_top_score_changed_detection():
    delta = MemoryInfluenceEvaluator().compare_summaries(
        summary(scores=[90.0, 80.0]),
        summary(scores=[95.0, 80.0], memory_context_provided=True, memory_context_eligible_count=1),
    )

    assert delta.top_score_changed is True
    assert delta.decision == "warning"
    assert delta.risk_level == "medium"


def test_run_case_uses_injected_runners_and_redacts_payloads():
    evaluator = MemoryInfluenceEvaluator()
    case = MemoryInfluenceEvalCase(
        case_id="case_sensitive",
        raw_jd=SENSITIVE_JD,
        memory_source="sqlite",
        memory_config={"memory_content": SENSITIVE_MEMORY, "max_items": 2},
        expected_effect="candidate_count",
        metadata={"reasoning": SENSITIVE_REASONING},
    )

    result = evaluator.run_case(
        case,
        no_memory_runner=lambda _jd: summary(),
        with_memory_runner=lambda _jd, **_kwargs: summary(
            candidate_count=3,
            memory_context_provided=True,
            memory_context_eligible_count=1,
        ),
        memory_context=object(),
    )
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    case_payload = json.dumps(case.to_dict(), ensure_ascii=False)

    assert result.case_id == "case_sensitive"
    assert result.delta.decision == "expected_effect"
    assert SENSITIVE_JD not in payload
    assert SENSITIVE_MEMORY not in payload
    assert SENSITIVE_REASONING not in payload
    assert SENSITIVE_JD not in case_payload
    assert SENSITIVE_MEMORY not in case_payload


def test_run_summary_normalization_uses_counts_only():
    run = MemoryInfluenceRunSummary.from_summary(
        {
            "status": "ok",
            "candidate_count": "2",
            "report_count": "1",
            "top_scores": [88, "bad", 70],
            "candidate_ids": ["candidate_a"],
            "memory_context_provided": True,
            "memory_context_eligible_count": 1,
            "full_reasoning": SENSITIVE_REASONING,
        }
    )
    payload = json.dumps(run.to_dict(), ensure_ascii=False)

    assert run.candidate_count == 2
    assert run.report_count == 1
    assert run.top_scores == [88.0, 70.0]
    assert run.memory_context_eligible_count == 1
    assert SENSITIVE_REASONING not in payload


def test_evaluator_serializes_to_dict_summary_only():
    result = MemoryInfluenceEvaluator().run_case(
        MemoryInfluenceEvalCase("case_dict", "Need Python"),
        no_memory_runner=lambda _jd: summary(),
        with_memory_runner=lambda _jd, **_kwargs: summary(memory_context_provided=True, memory_context_eligible_count=1),
    )
    data = result.to_dict()

    assert data["metadata"]["summary_only"] is True
    assert data["no_memory_summary"]["summary_only"] is True
    assert data["with_memory_summary"]["summary_only"] is True
    assert data["delta"]["decision"] == "no_effect"


def test_phase9c_does_not_import_real_llm_chroma_hf_or_mcp(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("llama_index", "chromadb", "sentence_transformers", "mcp", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase9C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    delta = MemoryInfluenceEvaluator().compare_summaries(
        summary(),
        summary(memory_context_provided=True, memory_context_eligible_count=1),
    )

    assert delta.decision == "no_effect"
    assert blocked == []


def test_default_graph_source_is_not_modified_for_memory_influence_eval():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "MemoryInfluenceEvaluator" not in graph_source
    assert "MemoryInfluenceEvalCase" not in graph_source
    assert "memory influence" not in graph_source.lower()
