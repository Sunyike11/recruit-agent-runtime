import builtins
from pathlib import Path

from src.evaluation import EvalCase, EvalReport, EvalResult, EvalRunner
from src.runtime.models import Event
from src.skills.workflow import SkillWorkflowResult
from src.tools.workflow import ToolWorkflowResult


def case(target_type="task", input_data=None, checks=None, expected=None):
    return EvalCase(
        case_id=f"{target_type}_case",
        target_type=target_type,
        input_data=input_data,
        expected=dict(expected or {}),
        checks=list(checks or []),
        metadata={"phase": "5A"},
    )


def test_eval_case_can_create_and_load_fixture_dict():
    eval_case = EvalCase.from_dict(
        {
            "case_id": "case_1",
            "target_type": "task",
            "input_data": {"status": "completed"},
            "checks": [{"type": "status_is", "value": "completed"}],
            "tags": ["smoke"],
        }
    )

    assert eval_case.case_id == "case_1"
    assert eval_case.target_type == "task"
    assert eval_case.tags == ["smoke"]


def test_eval_result_can_create():
    result = EvalResult(case_id="case_1", target_type="task", passed=True, score=1.0)

    assert result.passed is True
    assert result.score == 1.0
    assert result.created_at is not None


def test_eval_report_can_aggregate_results():
    report = EvalReport.from_results(
        [
            EvalResult("passed", "task", True, 1.0),
            EvalResult("failed", "task", False, 0.5),
        ]
    )

    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1
    assert report.average_score == 0.75


def test_eval_runner_can_run_single_case_with_evaluator():
    eval_case = case(
        input_data={"raw": "input"},
        checks=[{"type": "status_is", "value": "completed"}],
    )

    result = EvalRunner().run_case(eval_case, evaluator=lambda data: {"status": "completed", **data})

    assert result.passed is True
    assert result.score == 1.0


def test_eval_runner_can_run_multiple_cases():
    runner = EvalRunner()
    report = runner.run_cases(
        [
            case(input_data={"status": "completed"}, checks=[{"type": "status_is", "value": "completed"}]),
            case(input_data={"status": "failed"}, checks=[{"type": "status_is", "value": "completed"}]),
        ]
    )

    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1


def test_expected_dict_provides_default_equals_checks():
    result = EvalRunner().run_case(case(input_data={"status": "completed"}, expected={"status": "completed"}))

    assert result.passed is True
    assert result.checks[0]["name"] == "equals"


def test_required_keys_present_check():
    result = EvalRunner().run_case(
        case(
            input_data={"status": "completed", "task_id": "task_1"},
            checks=[{"type": "required_keys_present", "keys": ["status", "task_id"]}],
        )
    )

    assert result.passed is True


def test_min_count_check():
    result = EvalRunner().run_case(
        case(
            target_type="skill_workflow",
            input_data={"match_reports": [{"score": 88}]},
            checks=[{"type": "min_count", "path": "match_reports", "value": 1}],
        )
    )

    assert result.passed is True
    assert result.checks[0]["actual"] == 1


def test_max_count_check():
    result = EvalRunner().run_case(
        case(
            input_data={"errors": []},
            checks=[{"type": "max_count", "path": "errors", "value": 0}],
        )
    )

    assert result.passed is True


def test_equals_check():
    result = EvalRunner().run_case(
        case(input_data={"candidate_count": 2}, checks=[{"type": "equals", "path": "candidate_count", "value": 2}])
    )

    assert result.passed is True


def test_contains_check():
    result = EvalRunner().run_case(
        case(
            input_data={"state_keys": ["job_requirement", "match_reports"]},
            checks=[{"type": "contains", "path": "state_keys", "value": "match_reports"}],
        )
    )

    assert result.passed is True


def test_status_is_check():
    result = EvalRunner().run_case(
        case(input_data={"status": "partial"}, checks=[{"type": "status_is", "value": "partial"}])
    )

    assert result.passed is True


def test_event_type_present_check():
    result = EvalRunner().run_case(
        case(
            target_type="runtime_timeline",
            input_data=[{"event_type": "task_completed"}],
            checks=[{"type": "event_type_present", "event_type": "task_completed"}],
        )
    )

    assert result.passed is True


def test_event_type_count_at_least_check():
    result = EvalRunner().run_case(
        case(
            target_type="runtime_timeline",
            input_data=[
                {"event_type": "skill_completed"},
                {"event_type": "skill_completed"},
            ],
            checks=[{"type": "event_type_count_at_least", "event_type": "skill_completed", "value": 2}],
        )
    )

    assert result.passed is True
    assert result.checks[0]["actual"] == 2


def test_event_type_absent_supports_safe_negative_event_expectation():
    result = EvalRunner().run_case(
        case(
            target_type="runtime_timeline",
            input_data=[{"event_type": "tool_completed"}],
            checks=[{"type": "event_type_absent", "event_type": "tool_denied"}],
        )
    )

    assert result.passed is True


def test_runtime_timeline_event_objects_can_be_evaluated():
    timeline = [
        Event(event_id="1", event_type="task_completed", task_id="task_1"),
        Event(event_id="2", event_type="skill_completed", task_id="task_1"),
        Event(event_id="3", event_type="skill_completed", task_id="task_1"),
    ]
    result = EvalRunner().run_case(
        case(
            target_type="runtime_timeline",
            input_data=timeline,
            checks=[
                {"type": "event_type_present", "event_type": "task_completed"},
                {"type": "event_type_count_at_least", "event_type": "skill_completed", "value": 2},
                {"type": "event_type_absent", "event_type": "tool_denied"},
            ],
        )
    )

    assert result.passed is True
    assert result.score == 1.0


def test_skill_workflow_fake_result_can_be_evaluated():
    workflow_result = SkillWorkflowResult(
        status="completed",
        success=True,
        retrieved_candidates=[{"candidate_id": "candidate_1"}],
        match_reports=[{"total_score": 88.0}],
    )
    result = EvalRunner().run_case(
        case(
            target_type="skill_workflow",
            input_data=workflow_result,
            checks=[
                {"type": "status_is", "value": "completed"},
                {"type": "min_count", "path": "retrieved_candidates", "value": 1},
                {"type": "min_count", "path": "match_reports", "value": 1},
            ],
        )
    )

    assert result.passed is True


def test_tool_workflow_fake_result_can_be_evaluated():
    workflow_result = ToolWorkflowResult(
        status="completed",
        success=True,
        outputs={"parsed": {"type": "dict", "keys": ["keywords"], "size": 1}},
        steps=[{"tool_name": "resume_text_parse_fake", "success": True}],
    )
    result = EvalRunner().run_case(
        case(
            target_type="tool_workflow",
            input_data=workflow_result,
            checks=[
                {"type": "status_is", "value": "completed"},
                {"type": "required_keys_present", "path": "outputs", "keys": ["parsed"]},
                {"type": "min_count", "path": "steps", "value": 1},
            ],
        )
    )

    assert result.passed is True


def test_failed_check_returns_false_and_clear_check_details():
    result = EvalRunner().run_case(
        case(input_data={"status": "failed"}, checks=[{"type": "status_is", "value": "completed"}])
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.checks[0] == {
        "name": "status_is",
        "passed": False,
        "expected": "completed",
        "actual": "failed",
        "error": "",
    }


def test_phase5a_has_no_real_llm_mcp_or_external_tool_dependency(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = EvalRunner().run_case(
        case(input_data={"status": "completed"}, checks=[{"type": "status_is", "value": "completed"}])
    )

    assert result.passed is True
    assert blocked == []


def test_phase5a_does_not_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "EvalRunner" not in graph_source
    assert "src.evaluation" not in graph_source
