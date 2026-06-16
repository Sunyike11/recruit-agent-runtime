import builtins
import json
from pathlib import Path

from src.evaluation import (
    EvalCase,
    EvalRecord,
    EvalReport,
    EvalResult,
    EvalRunner,
    export_eval_records_json,
    export_eval_records_text,
    export_eval_report_json,
    export_eval_report_text,
    project_runtime_timeline,
    project_task_timeline,
)
from src.runtime import Event, SQLiteRuntimeStore, SessionManager, TaskManager


SECRET_TEXT = "PRIVATE-RESUME-FULL-TEXT-DO-NOT-EXPORT"


def timeline_case(input_data, checks):
    return EvalCase(
        case_id="projected_timeline",
        target_type="runtime_timeline",
        input_data=input_data,
        checks=checks,
    )


def test_project_runtime_timeline_counts_task_skill_and_tool_events():
    projection = project_runtime_timeline(
        [
            Event("1", "task_created", task_id="task_1"),
            Event("2", "task_started", task_id="task_1"),
            Event("3", "skill_started", task_id="task_1"),
            Event("4", "skill_completed", task_id="task_1"),
            Event("5", "tool_started", task_id="task_1"),
            Event("6", "tool_completed", task_id="task_1"),
            Event("7", "task_completed", task_id="task_1"),
        ]
    )

    assert projection.target_id == "task_1"
    assert projection.status == "completed"
    assert projection.task_event_counts["task_completed"] == 1
    assert projection.skill_event_counts["skill_completed"] == 1
    assert projection.tool_event_counts["tool_completed"] == 1
    assert projection.event_counts["task_started"] == 1


def test_project_runtime_timeline_recognizes_failures_and_error_summaries():
    projection = project_runtime_timeline(
        [
            {"event_type": "tool_failed", "task_id": "task_failed", "payload": {"error": "tool timeout"}},
            {"event_type": "task_failed", "task_id": "task_failed", "payload": {"error": "task stopped"}},
        ]
    )

    assert projection.status == "failed"
    assert projection.errors == ["tool timeout", "task stopped"]
    assert projection.event_counts["tool_failed"] == 1


def test_project_runtime_timeline_gracefully_ignores_missing_or_unknown_fields():
    projection = project_runtime_timeline(
        [
            {},
            {"event_type": "unknown_event", "payload": "not-a-dict"},
            {"event_type": "skill_failed", "payload": None},
        ],
        target_id="task_sparse",
    )

    assert projection.target_id == "task_sparse"
    assert projection.status == "failed"
    assert projection.errors == []
    assert projection.skill_event_counts == {"skill_failed": 1}
    assert projection.metadata["ignored_event_count"] == 2


def test_projection_to_dict_can_be_evaluated_for_status_and_required_keys():
    projection = project_runtime_timeline([{"event_type": "task_completed"}], target_id="task_eval")
    result = EvalRunner().run_case(
        timeline_case(
            projection.to_dict(),
            [
                {"type": "status_is", "value": "completed"},
                {"type": "required_keys_present", "keys": ["event_counts", "events", "status"]},
            ],
        )
    )

    assert result.passed is True


def test_eval_runner_event_type_present_check_accepts_projection():
    projection = project_runtime_timeline([{"event_type": "tool_sandbox_denied"}])
    result = EvalRunner().run_case(
        timeline_case(
            projection.to_dict(),
            [{"type": "event_type_present", "event_type": "tool_sandbox_denied"}],
        )
    )

    assert result.passed is True


def test_eval_runner_event_type_count_at_least_check_accepts_projection():
    projection = project_runtime_timeline(
        [
            {"event_type": "skill_completed"},
            {"event_type": "skill_completed"},
        ]
    )
    result = EvalRunner().run_case(
        timeline_case(
            projection.to_dict(),
            [{"type": "event_type_count_at_least", "event_type": "skill_completed", "value": 2}],
        )
    )

    assert result.passed is True
    assert result.checks[0]["actual"] == 2


def test_all_tool_policy_and_approval_events_are_projected():
    projection = project_runtime_timeline(
        [
            {"event_type": "tool_denied"},
            {"event_type": "tool_approval_required"},
            {"event_type": "tool_approval_granted"},
            {"event_type": "tool_approval_rejected"},
            {"event_type": "tool_sandbox_denied"},
        ]
    )

    assert projection.tool_event_counts == {
        "tool_denied": 1,
        "tool_approval_required": 1,
        "tool_approval_granted": 1,
        "tool_approval_rejected": 1,
        "tool_sandbox_denied": 1,
    }


def test_export_eval_report_json_is_parseable_and_summary_only():
    report = EvalReport.from_results(
        [
            EvalResult(
                case_id="sensitive_report",
                target_type="runtime_timeline",
                passed=False,
                score=0.0,
                checks=[{"name": "equals", "passed": False, "actual": SECRET_TEXT}],
                metadata={"raw_text": SECRET_TEXT},
            )
        ],
        metadata={"secret": SECRET_TEXT},
    )

    exported = export_eval_report_json(report)
    payload = json.loads(exported)

    assert payload["results"][0]["case_id"] == "sensitive_report"
    assert payload["summary_only"] is True
    assert SECRET_TEXT not in exported
    assert "actual" not in payload["results"][0]["checks"][0]


def test_export_eval_report_text_is_readable_and_hides_sensitive_payload():
    report = EvalReport.from_results(
        [
            EvalResult(
                case_id="text_report",
                target_type="tool_workflow",
                passed=True,
                score=1.0,
                checks=[{"name": "contains", "passed": True, "actual": SECRET_TEXT}],
            )
        ]
    )

    exported = export_eval_report_text(report)

    assert "Evaluation Report" in exported
    assert "text_report" in exported
    assert SECRET_TEXT not in exported


def test_export_eval_records_json_and_text_omit_full_report_payload():
    record = EvalRecord(
        eval_id="eval_safe",
        case_id="case_safe",
        target_type="task",
        target_id="task_1",
        passed=True,
        score=1.0,
        report_json={"raw_payload": SECRET_TEXT},
        metadata={"secret": SECRET_TEXT},
    )

    json_export = export_eval_records_json([record])
    text_export = export_eval_records_text([record])
    payload = json.loads(json_export)

    assert payload["records"][0]["report_present"] is True
    assert payload["records"][0]["case_id"] == "case_safe"
    assert "case_safe" in text_export
    assert SECRET_TEXT not in json_export
    assert SECRET_TEXT not in text_export


def test_project_task_timeline_reads_events_from_sqlite_runtime_store(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "phase5c.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "5C"})
    task = TaskManager(store).create_task(session.session_id, jd_text="summary only", thread_id="thread-5c")
    store.append_event("skill_started", session_id=session.session_id, task_id=task.task_id)
    store.append_event("skill_completed", session_id=session.session_id, task_id=task.task_id)
    store.append_event("task_completed", session_id=session.session_id, task_id=task.task_id)

    projection = project_task_timeline(store, task.task_id)

    assert projection.target_id == task.task_id
    assert projection.status == "completed"
    assert projection.task_event_counts["task_created"] == 1
    assert projection.skill_event_counts["skill_completed"] == 1
    assert projection.metadata["source"] == "runtime_store"


def test_phase5c_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5C test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    projection = project_runtime_timeline([{"event_type": "task_completed"}])
    report = EvalRunner().run_cases(
        [timeline_case(projection.to_dict(), [{"type": "status_is", "value": "completed"}])]
    )

    assert report.passed_cases == 1
    assert blocked == []


def test_phase5c_does_not_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "EvaluationTargetProjection" not in graph_source
    assert "project_runtime_timeline" not in graph_source
    assert "export_eval_report_json" not in graph_source
