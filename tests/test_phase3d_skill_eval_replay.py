import builtins
import sys

from src.skills import (
    BaseSkill,
    EchoSkill,
    SkillEvalCase,
    SkillEvalReport,
    SkillEvalResult,
    SkillEvalRunner,
    SkillExecutionContext,
    SkillExecutor,
    SkillRegistry,
    SkillSpec,
    replay_case_from_fixture,
    replay_case_from_skill_event_payload,
)


def block_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked retrieval import in Phase3D test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class FailingEvalSkill(BaseSkill):
    spec = SkillSpec(name="failing_eval_demo", version="v1")

    def run(self, input_data, context=None):
        raise RuntimeError("expected eval failure")


def make_registry(*skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def test_skill_eval_case_can_create():
    eval_case = SkillEvalCase(
        case_id="case_1",
        skill_name="echo",
        skill_version="v1",
        input_data={"message": "hello"},
        expected_output={"message": "hello"},
        expected_success=True,
        tags=["demo"],
        metadata={"phase": "3D"},
    )

    assert eval_case.case_id == "case_1"
    assert eval_case.expected_output == {"message": "hello"}
    assert eval_case.tags == ["demo"]


def test_skill_eval_result_can_create():
    result = SkillEvalResult(
        case_id="case_1",
        skill_name="echo",
        skill_version="v1",
        success=True,
        passed=True,
        output={"ok": True},
    )

    assert result.passed is True
    assert result.output == {"ok": True}


def test_skill_eval_runner_can_run_echo_skill_case():
    runner = SkillEvalRunner(make_registry(EchoSkill()))
    eval_case = SkillEvalCase(
        case_id="echo_case",
        skill_name="echo",
        input_data={"message": "hello"},
        expected_output={"message": "hello"},
    )

    result = runner.run_case(eval_case)

    assert result.success is True
    assert result.passed is True
    assert result.output == {"message": "hello"}


def test_expected_output_key_value_match_passes():
    runner = SkillEvalRunner(make_registry(EchoSkill()))
    eval_case = SkillEvalCase(
        case_id="subset_case",
        skill_name="echo",
        input_data={"message": "hello", "extra": True},
        expected_output={"message": "hello"},
    )

    result = runner.run_case(eval_case)

    assert result.passed is True
    assert result.checks[-1]["name"] == "expected_output_contains"


def test_expected_output_mismatch_fails():
    runner = SkillEvalRunner(make_registry(EchoSkill()))
    eval_case = SkillEvalCase(
        case_id="mismatch_case",
        skill_name="echo",
        input_data={"message": "hello"},
        expected_output={"message": "goodbye"},
    )

    result = runner.run_case(eval_case)

    assert result.success is True
    assert result.passed is False
    assert result.checks[-1]["mismatches"]["message"]["actual"] == "hello"


def test_expected_failure_case_can_pass_evaluation():
    runner = SkillEvalRunner(make_registry(FailingEvalSkill()))
    eval_case = SkillEvalCase(
        case_id="failure_case",
        skill_name="failing_eval_demo",
        input_data={"value": 1},
        expected_success=False,
    )

    result = runner.run_case(eval_case)

    assert result.success is False
    assert result.passed is True
    assert result.error == "expected eval failure"


def test_run_cases_generates_skill_eval_report():
    runner = SkillEvalRunner(make_registry(EchoSkill()))
    cases = [
        SkillEvalCase(
            case_id="pass_case",
            skill_name="echo",
            input_data={"message": "hello"},
            expected_output={"message": "hello"},
        ),
        SkillEvalCase(
            case_id="fail_case",
            skill_name="echo",
            input_data={"message": "hello"},
            expected_output={"message": "nope"},
        ),
    ]

    report = runner.run_cases(cases)

    assert isinstance(report, SkillEvalReport)
    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1


def test_fixture_replay_can_full_replay_deterministic_skill():
    runner = SkillEvalRunner(make_registry(EchoSkill()))
    fixture = {
        "case_id": "fixture_case",
        "skill_name": "echo",
        "input_data": {"message": "from fixture"},
        "expected_output": {"message": "from fixture"},
        "metadata": {"fixture_version": "v1"},
    }

    result = replay_case_from_fixture(fixture, runner)

    assert result.passed is True
    assert result.metadata["replay_mode"] == "fixture_full_replay"
    assert result.metadata["full_replay"] is True


def test_event_summary_replay_is_marked_limited_not_full_replay():
    payload = {
        "execution_id": "skill_execution_1",
        "skill_name": "echo",
        "skill_version": "v1",
        "status": "completed",
        "input_summary": {"type": "dict", "keys": ["message"], "size": 1},
        "output_summary": {"type": "dict", "keys": ["message"], "size": 1},
        "error": "",
    }

    audit_record = replay_case_from_skill_event_payload(payload)

    assert audit_record["replay_mode"] == "event_summary_limited_replay"
    assert audit_record["full_replay"] is False
    assert audit_record["can_execute_skill"] is False
    assert "summaries only" in audit_record["reason"]


def test_skill_eval_runner_can_use_skill_executor():
    registry = make_registry(EchoSkill())
    executor = SkillExecutor(registry)
    runner = SkillEvalRunner(executor)
    context = SkillExecutionContext(task_id="task_eval", session_id="session_eval")
    eval_case = SkillEvalCase(
        case_id="executor_case",
        skill_name="echo",
        input_data={"message": "via executor"},
        expected_output={"message": "via executor"},
    )

    result = runner.run_case(eval_case, context=context)

    assert result.passed is True
    assert result.output == {"message": "via executor"}


def test_phase3d_does_not_import_real_retrieval_modules(monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    runner = SkillEvalRunner(make_registry(EchoSkill()))

    result = runner.run_case(
        SkillEvalCase(
            case_id="safe_case",
            skill_name="echo",
            input_data={"safe": True},
            expected_output={"safe": True},
        )
    )

    assert result.passed is True
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
