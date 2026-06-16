import builtins
from pathlib import Path

import pytest

from src.evaluation import (
    EvalCase,
    EvalRecord,
    EvalReport,
    EvalResult,
    EvalRunner,
    EvaluationCatalog,
    EvaluationCatalogError,
    InMemoryEvalRecordStore,
    create_eval_record,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation_cases.json"


def simple_case(case_id="task_ok", target_type="task", tags=None):
    return {
        "case_id": case_id,
        "target_type": target_type,
        "input_data": {"status": "completed"},
        "checks": [{"type": "status_is", "value": "completed"}],
        "tags": list(tags or []),
        "metadata": {"summary_only": True},
    }


def test_evaluation_catalog_can_be_created_from_dict():
    catalog = EvaluationCatalog.from_dict(
        {"cases": [simple_case("one"), simple_case("two", tags=["smoke"])]}
    )

    assert [eval_case.case_id for eval_case in catalog.list_cases()] == ["one", "two"]
    assert catalog.get_case("two").tags == ["smoke"]


def test_evaluation_catalog_loads_json_fixture():
    catalog = EvaluationCatalog.from_json_file(FIXTURE_PATH)

    assert [eval_case.case_id for eval_case in catalog.list_cases()] == [
        "runtime_timeline_completed",
        "skill_workflow_match",
        "tool_workflow_completed",
    ]
    assert catalog.metadata["fixture_version"] == "v1"


def test_catalog_filters_by_tag_and_target_type():
    catalog = EvaluationCatalog.from_json_file(FIXTURE_PATH)

    assert len(catalog.filter_by_tag("smoke")) == 3
    assert [eval_case.case_id for eval_case in catalog.filter_by_target_type("skill_workflow")] == [
        "skill_workflow_match"
    ]


def test_invalid_eval_case_raises_clear_catalog_error():
    with pytest.raises(EvaluationCatalogError, match="invalid evaluation case"):
        EvaluationCatalog.from_dict({"cases": [simple_case(case_id="", target_type="task")]})

    with pytest.raises(EvaluationCatalogError, match="duplicate evaluation case_id"):
        EvaluationCatalog.from_dict({"cases": [simple_case(), simple_case()]})


def test_eval_result_and_report_serialize_round_trip():
    result = EvalResult(
        case_id="serialized",
        target_type="task",
        passed=True,
        score=1.0,
        checks=[{"name": "status_is", "passed": True}],
        metadata={"summary_only": True},
    )
    report = EvalReport.from_results([result], metadata={"fixture_version": "v1"})

    restored = EvalReport.from_dict(report.to_dict())

    assert restored.total_cases == 1
    assert restored.results[0].case_id == "serialized"
    assert restored.results[0].checks == [{"name": "status_is", "passed": True}]
    assert restored.metadata["fixture_version"] == "v1"


def test_eval_record_can_create_and_serialize():
    record = EvalRecord(
        eval_id="eval_fixed",
        case_id="task_ok",
        target_type="task",
        target_id="task_1",
        passed=True,
        score=1.0,
        report_json={"passed": True},
        metadata={"summary_only": True},
    )

    restored = EvalRecord.from_dict(record.to_dict())

    assert restored.eval_id == "eval_fixed"
    assert restored.report_json == {"passed": True}
    assert restored.created_at == record.created_at


def test_in_memory_eval_record_store_saves_gets_and_filters():
    store = InMemoryEvalRecordStore()
    runtime_record = EvalRecord(case_id="runtime", target_type="runtime_timeline", target_id="task_1")
    skill_record = EvalRecord(case_id="skill", target_type="skill_workflow", target_id="workflow_1")
    store.save_record(runtime_record)
    store.save_record(skill_record)

    assert store.get_record(runtime_record.eval_id) is runtime_record
    assert store.list_records(target_type="runtime_timeline") == [runtime_record]
    assert store.list_records(target_id="workflow_1") == [skill_record]
    assert store.list_records(case_id="skill") == [skill_record]


def test_create_eval_record_from_eval_result():
    eval_case = EvalCase.from_dict(simple_case())
    result = EvalRunner().run_case(eval_case)

    record = create_eval_record(eval_case, result, target_id="task_1", metadata={"run": "local"})

    assert record.passed is True
    assert record.score == 1.0
    assert record.target_id == "task_1"
    assert record.report_json["case_id"] == "task_ok"
    assert record.metadata == {"record_source": "eval_result", "run": "local"}


def test_create_eval_record_from_eval_report():
    eval_case = EvalCase.from_dict(simple_case(case_id="report_case"))
    report = EvalRunner().run_cases([eval_case])

    record = create_eval_record(eval_case, report, target_id="batch_1")

    assert record.passed is True
    assert record.report_json["total_cases"] == 1
    assert record.metadata["record_source"] == "eval_report"


def test_runtime_timeline_fixture_runs_and_creates_record():
    eval_case = EvaluationCatalog.from_json_file(FIXTURE_PATH).get_case("runtime_timeline_completed")
    result = EvalRunner().run_case(eval_case)
    store = InMemoryEvalRecordStore()
    record = store.save_record(create_eval_record(eval_case, result, target_id="task_fixture_1"))

    assert result.passed is True
    assert record.target_type == "runtime_timeline"
    assert store.list_records(target_id="task_fixture_1") == [record]


def test_skill_workflow_fixture_runs_and_creates_record():
    eval_case = EvaluationCatalog.from_json_file(FIXTURE_PATH).get_case("skill_workflow_match")
    result = EvalRunner().run_case(eval_case)
    record = create_eval_record(eval_case, result, target_id="skill_workflow_fixture_1")

    assert result.passed is True
    assert record.target_type == "skill_workflow"
    assert record.report_json["score"] == 1.0


def test_tool_workflow_fixture_runs_and_creates_record():
    eval_case = EvaluationCatalog.from_json_file(FIXTURE_PATH).get_case("tool_workflow_completed")
    result = EvalRunner().run_case(eval_case)
    record = create_eval_record(eval_case, result, target_id="tool_workflow_fixture_1")

    assert result.passed is True
    assert record.target_type == "tool_workflow"
    assert record.report_json["passed"] is True


def test_phase5b_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    catalog = EvaluationCatalog.from_json_file(FIXTURE_PATH)
    report = EvalRunner().run_cases(catalog.list_cases())

    assert report.passed_cases == 3
    assert blocked == []


def test_phase5b_does_not_modify_production_graph():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "EvaluationCatalog" not in graph_source
    assert "EvalRecord" not in graph_source
    assert "src.evaluation" not in graph_source
