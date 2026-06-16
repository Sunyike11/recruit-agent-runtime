import builtins
from pathlib import Path

from src.evaluation import CorrelationReport, EvalRecord, EvalReport, EvalResult
from src.reflection import (
    InMemoryReflectionStore,
    ReflectionRecord,
    ReflectionSourceType,
    ReflectionStatus,
    reflection_from_correlation_report,
    reflection_from_eval_record,
    reflection_from_eval_report,
)


SECRET_TEXT = "FULL-RESUME-TEXT-AND-API-SECRET-MUST-NOT-BECOME-REFLECTION"


def test_reflection_record_can_create():
    record = ReflectionRecord(
        source_type=ReflectionSourceType.MANUAL.value,
        source_id="manual_1",
        target_type="task",
        target_id="task_1",
        status=ReflectionStatus.WARNING.value,
        summary="Manual summary only.",
        findings=["Check a deterministic signal."],
        recommended_actions=["Review evidence"],
        evidence_refs=["event_summary_1"],
        tags=["manual"],
    ).validate()

    assert record.target_id == "task_1"
    assert record.status == "warning"
    assert record.reflection_id.startswith("reflection_")


def test_reflection_record_round_trips_through_dict():
    record = ReflectionRecord(
        source_type=ReflectionSourceType.EVAL_RECORD.value,
        source_id="eval_1",
        target_type="runtime_timeline",
        target_id="task_1",
        status=ReflectionStatus.SUCCESS.value,
        summary="Passed deterministic checks.",
        metadata={"summary_only": True},
    )

    restored = ReflectionRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.created_at == record.created_at


def test_in_memory_reflection_store_can_save_get_and_delete():
    store = InMemoryReflectionStore()
    record = ReflectionRecord(summary="Store me.")

    store.save_reflection(record)

    assert store.get_reflection(record.reflection_id) is record
    assert store.delete_reflection(record.reflection_id) is True
    assert store.delete_reflection(record.reflection_id) is False


def test_store_filters_by_target_source_and_tag():
    store = InMemoryReflectionStore()
    task_record = ReflectionRecord(
        source_type=ReflectionSourceType.EVAL_RECORD.value,
        target_type="task",
        target_id="task_1",
        tags=["evaluation", "warning"],
    )
    workflow_record = ReflectionRecord(
        source_type=ReflectionSourceType.CORRELATION_REPORT.value,
        target_type="tool_workflow",
        target_id="workflow_1",
        tags=["correlation"],
    )
    store.save_reflection(task_record)
    store.save_reflection(workflow_record)

    assert store.list_reflections(target_type="task") == [task_record]
    assert store.list_reflections(target_id="workflow_1") == [workflow_record]
    assert store.list_reflections(source_type="eval_record") == [task_record]
    assert store.list_reflections(tag="correlation") == [workflow_record]


def test_reflection_from_all_passed_eval_report_is_success():
    report = EvalReport.from_results(
        [
            EvalResult("case_1", "task", True, 1.0),
            EvalResult("case_2", "tool_workflow", True, 1.0),
        ]
    )

    reflection = reflection_from_eval_report(report, target_id="batch_1", target_type="task")

    assert reflection.status == ReflectionStatus.SUCCESS.value
    assert reflection.target_id == "batch_1"
    assert reflection.recommended_actions == []
    assert "2/2 cases passed" in reflection.summary


def test_reflection_from_failed_eval_report_is_warning_with_action():
    report = EvalReport.from_results(
        [
            EvalResult("case_ok", "task", True, 1.0),
            EvalResult("case_failed", "task", False, 0.0),
        ]
    )

    reflection = reflection_from_eval_report(report, target_id="batch_failed", target_type="task")

    assert reflection.status == ReflectionStatus.WARNING.value
    assert "Review failed evaluation cases" in reflection.recommended_actions
    assert "case_failed" in reflection.evidence_refs
    assert any("case_failed" in finding for finding in reflection.findings)


def test_reflection_from_eval_record_preserves_safe_reference_not_report_payload():
    eval_record = EvalRecord(
        eval_id="eval_safe",
        case_id="case_failed",
        target_type="runtime_timeline",
        target_id="task_safe",
        passed=False,
        score=0.25,
        report_json={"complete_payload": SECRET_TEXT},
        metadata={"secret": SECRET_TEXT},
    )

    reflection = reflection_from_eval_record(eval_record)
    serialized = str(reflection.to_dict())

    assert reflection.source_id == "eval_safe"
    assert reflection.target_id == "task_safe"
    assert reflection.status == ReflectionStatus.WARNING.value
    assert SECRET_TEXT not in serialized


def test_reflection_from_correlation_report_generates_policy_actions():
    correlation = CorrelationReport(
        target_id="task_policy",
        evaluation_passed=False,
        average_score=0.5,
        failed_cases=["failed_tool_case"],
        event_counts={"tool_denied": 1, "tool_sandbox_denied": 1},
        tool_denied_count=1,
        tool_sandbox_denied_count=1,
        tool_approval_required_count=1,
    )

    reflection = reflection_from_correlation_report(correlation)

    assert reflection.status == ReflectionStatus.FAILURE.value
    assert "Inspect runtime timeline and audit report" in reflection.recommended_actions
    assert "Review tool permission policy" in reflection.recommended_actions
    assert "Review sandbox profile" in reflection.recommended_actions
    assert "Review tool approval requirements" in reflection.recommended_actions


def test_correlation_with_policy_signal_but_passed_evaluation_is_warning():
    correlation = CorrelationReport(
        target_id="task_warning",
        evaluation_passed=True,
        average_score=1.0,
        tool_denied_count=1,
    )

    reflection = reflection_from_correlation_report(correlation)

    assert reflection.status == ReflectionStatus.WARNING.value
    assert reflection.recommended_actions == ["Review tool permission policy"]


def test_reflection_derivation_does_not_copy_sensitive_eval_or_correlation_content():
    report = EvalReport.from_results(
        [
            EvalResult(
                "case_sensitive",
                "task",
                False,
                0.0,
                checks=[{"actual": SECRET_TEXT, "passed": False}],
                error=SECRET_TEXT,
                metadata={"raw": SECRET_TEXT},
            )
        ],
        metadata={"payload": SECRET_TEXT},
    )
    correlation = CorrelationReport(
        target_id="task_sensitive",
        evaluation_passed=False,
        average_score=0.0,
        failed_cases=["case_sensitive"],
        errors=[SECRET_TEXT],
        summary=SECRET_TEXT,
        metadata={"payload": SECRET_TEXT},
    )

    reflected_report = reflection_from_eval_report(report, target_id="task_sensitive", target_type="task")
    reflected_correlation = reflection_from_correlation_report(correlation)

    assert SECRET_TEXT not in str(reflected_report.to_dict())
    assert SECRET_TEXT not in str(reflected_correlation.to_dict())


def test_phase5e_reflection_layer_does_not_import_or_write_memory_store():
    derivation_source = Path("src/reflection/derivation.py").read_text(encoding="utf-8")
    reflection_store_source = Path("src/reflection/store.py").read_text(encoding="utf-8")
    memory_store_source = Path("src/memory/store.py").read_text(encoding="utf-8")

    assert "MemorySQLiteStore" not in derivation_source
    assert "MemoryRecord" not in derivation_source
    assert "src.memory" not in reflection_store_source
    assert "ReflectionRecord" not in memory_store_source


def test_phase5e_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5E test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    reflection = reflection_from_eval_report(
        EvalReport.from_results([EvalResult("local", "task", True, 1.0)]),
        target_id="task_local",
        target_type="task",
    )

    assert reflection.status == "success"
    assert blocked == []


def test_phase5e_does_not_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "ReflectionRecord" not in graph_source
    assert "src.reflection" not in graph_source
    assert "reflection_from_" not in graph_source
