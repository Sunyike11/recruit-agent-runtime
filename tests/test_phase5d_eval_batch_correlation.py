import builtins
import json
from pathlib import Path

from src.evaluation import (
    BatchEvalRunner,
    EvalCase,
    EvalReport,
    EvalResult,
    EvaluationCatalog,
    RuntimeAuditCorrelation,
    export_correlation_report_json,
    export_correlation_report_text,
    project_runtime_timeline,
)
from src.runtime import Event
from src.tools import ToolAuditReporter


SECRET_TEXT = "FULL-PRIVATE-RESUME-AND-API-KEY-MUST-NOT-EXPORT"


def make_catalog():
    return EvaluationCatalog.from_dict(
        {
            "metadata": {"fixture_version": "phase5d", "summary_only": True},
            "cases": [
                {
                    "case_id": "runtime_ok",
                    "target_type": "runtime_timeline",
                    "checks": [{"type": "event_type_present", "event_type": "task_completed"}],
                },
                {
                    "case_id": "skill_ok",
                    "target_type": "skill_workflow",
                    "checks": [{"type": "status_is", "value": "completed"}],
                },
                {
                    "case_id": "tool_ok",
                    "target_type": "tool_workflow",
                    "checks": [{"type": "min_count", "path": "steps", "value": 1}],
                },
            ],
        }
    )


def target_resolver(eval_case):
    targets = {
        "runtime_timeline": project_runtime_timeline(
            [{"event_type": "task_completed"}],
            target_id="task_batch",
        ).to_dict(),
        "skill_workflow": {"status": "completed"},
        "tool_workflow": {"steps": [{"tool_name": "echo_tool", "success": True}]},
    }
    return targets.get(eval_case.target_type)


def test_batch_eval_runner_runs_multiple_catalog_cases():
    report = BatchEvalRunner().run_catalog(make_catalog(), target_resolver=target_resolver)

    assert report.total_cases == 3
    assert report.passed_cases == 3
    assert report.average_score == 1.0
    assert report.metadata["runner"] == "deterministic_batch_eval"


def test_target_resolver_can_supply_each_target_type():
    resolved = []

    def resolver(eval_case):
        resolved.append(eval_case.target_type)
        return target_resolver(eval_case)

    BatchEvalRunner().run_catalog(make_catalog(), target_resolver=resolver)

    assert resolved == ["runtime_timeline", "skill_workflow", "tool_workflow"]


def test_batch_run_can_generate_eval_records():
    report, records = BatchEvalRunner().run_catalog_with_records(
        make_catalog(),
        target_resolver=target_resolver,
        target_id_resolver=lambda eval_case: f"target_{eval_case.target_type}",
    )

    assert report.passed_cases == 3
    assert len(records) == 3
    assert records[0].target_id == "target_runtime_timeline"
    assert all(record.metadata["batch_run"] is True for record in records)


def test_missing_resolved_target_creates_clear_failed_result():
    catalog = EvaluationCatalog.from_dict(
        {
            "cases": [
                {
                    "case_id": "missing_target",
                    "target_type": "runtime_timeline",
                    "checks": [{"type": "event_type_present", "event_type": "task_completed"}],
                }
            ]
        }
    )

    report = BatchEvalRunner().run_catalog(catalog, target_resolver=lambda eval_case: None)

    assert report.failed_cases == 1
    assert "target_resolver returned no target" in report.results[0].error
    assert report.results[0].metadata["target_resolution_failed"] is True


def test_runtime_audit_correlation_links_eval_report_and_projection():
    evaluation = EvalReport.from_results([EvalResult("runtime_ok", "runtime_timeline", True, 1.0)])
    projection = project_runtime_timeline(
        [
            {"event_type": "task_completed", "task_id": "task_correlated"},
            {"event_type": "skill_completed", "task_id": "task_correlated"},
            {"event_type": "tool_completed", "task_id": "task_correlated"},
        ]
    )

    report = RuntimeAuditCorrelation.correlate(evaluation, projection)

    assert report.target_id == "task_correlated"
    assert report.evaluation_passed is True
    assert report.event_counts["skill_completed"] == 1
    assert report.tool_event_counts["tool_completed"] == 1
    assert "evaluation=passed" in report.summary


def test_runtime_audit_correlation_accepts_eval_records():
    _, records = BatchEvalRunner().run_catalog_with_records(
        make_catalog(),
        target_resolver=target_resolver,
    )
    projection = project_runtime_timeline([{"event_type": "task_completed"}], target_id="task_records")

    report = RuntimeAuditCorrelation.correlate(records, projection)

    assert report.evaluation_passed is True
    assert report.average_score == 1.0
    assert report.metadata["evaluation_source"] == "eval_records"


def test_runtime_audit_correlation_uses_tool_audit_counts():
    runtime_events = [
        Event("1", "tool_denied", task_id="task_tools", payload={"tool_name": "write_fake"}),
        Event("2", "tool_sandbox_denied", task_id="task_tools", payload={"tool_name": "network_fake"}),
        Event("3", "tool_approval_required", task_id="task_tools", payload={"tool_name": "external_fake"}),
    ]
    projection = project_runtime_timeline(runtime_events)
    tool_report = ToolAuditReporter.from_events(runtime_events)
    evaluation = EvalReport.from_results([EvalResult("tool_policy", "runtime_timeline", True, 1.0)])

    report = RuntimeAuditCorrelation.correlate(evaluation, projection, tool_audit_report=tool_report)

    assert report.tool_denied_count == 1
    assert report.tool_sandbox_denied_count == 1
    assert report.tool_approval_required_count == 1
    assert report.metadata["tool_audit_correlated"] is True


def test_failed_eval_cases_are_listed_in_correlation_report():
    evaluation = EvalReport.from_results(
        [
            EvalResult("pass_case", "task", True, 1.0),
            EvalResult("failed_case", "tool_workflow", False, 0.0),
        ]
    )

    report = RuntimeAuditCorrelation.correlate(
        evaluation,
        project_runtime_timeline([], target_id="target_failed"),
    )

    assert report.evaluation_passed is False
    assert report.failed_cases == ["failed_case"]
    assert report.average_score == 0.5


def test_correlation_export_is_summary_only_and_does_not_expose_payload_or_error_text():
    projection = project_runtime_timeline(
        [
            {
                "event_type": "tool_failed",
                "task_id": "task_secret",
                "payload": {"input_summary": SECRET_TEXT, "error": SECRET_TEXT},
            }
        ]
    )
    evaluation = EvalReport.from_results(
        [EvalResult("secret_case", "runtime_timeline", False, 0.0, metadata={"secret": SECRET_TEXT})]
    )

    report = RuntimeAuditCorrelation.correlate(evaluation, projection)
    json_export = export_correlation_report_json(report)
    text_export = export_correlation_report_text(report)
    data = json.loads(json_export)

    assert data["metadata"]["summary_only"] is True
    assert data["errors"] == ["1 projected runtime error(s) recorded"]
    assert SECRET_TEXT not in json_export
    assert SECRET_TEXT not in text_export


def test_correlation_report_to_dict_contains_summary_counts_only():
    projection = project_runtime_timeline(
        [{"event_type": "tool_denied"}, {"event_type": "tool_denied"}],
        target_id="task_counts",
    )
    report = RuntimeAuditCorrelation.correlate(
        EvalReport.from_results([EvalResult("case", "task", True, 1.0)]),
        projection,
    )

    data = report.to_dict()

    assert data["target_id"] == "task_counts"
    assert data["tool_denied_count"] == 2
    assert "events" not in data


def test_phase5d_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5D test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    report = BatchEvalRunner().run_catalog(make_catalog(), target_resolver=target_resolver)
    correlated = RuntimeAuditCorrelation.correlate(
        report,
        project_runtime_timeline([{"event_type": "task_completed"}], target_id="task_local"),
    )

    assert correlated.evaluation_passed is True
    assert blocked == []


def test_phase5d_does_not_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "BatchEvalRunner" not in graph_source
    assert "RuntimeAuditCorrelation" not in graph_source
    assert "CorrelationReport" not in graph_source
