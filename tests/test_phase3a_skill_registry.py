import builtins
import sys

import pytest

from src.memory import MemoryContext
from src.skills import (
    BaseSkill,
    CandidateMatchStubSkill,
    EchoSkill,
    KeywordExtractSkill,
    SkillAlreadyRegisteredError,
    SkillExecutionContext,
    SkillRegistry,
    SkillResult,
    SkillSpec,
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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase3A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class EchoSkillV2(EchoSkill):
    spec = SkillSpec(
        name="echo",
        version="v2",
        description="Second deterministic echo skill.",
    )


class FailingSkill(BaseSkill):
    spec = SkillSpec(name="failing_demo", version="v1", description="Always fails.")

    def run(self, input_data, context=None):
        raise RuntimeError("expected skill failure")


def test_skill_spec_can_create():
    spec = SkillSpec(
        name="demo",
        version="v1",
        description="Demo skill",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        required_tools=["local_parser"],
        required_memory_types=["semantic"],
        tags=["demo"],
        metadata={"owner": "test"},
    )

    assert spec.name == "demo"
    assert spec.version == "v1"
    assert spec.required_memory_types == ["semantic"]


def test_skill_result_can_create():
    result = SkillResult(
        skill_name="demo",
        version="v1",
        success=True,
        output={"ok": True},
        metadata={"phase": "3A"},
    )

    assert result.skill_name == "demo"
    assert result.output == {"ok": True}


def test_echo_skill_can_run_successfully():
    result = EchoSkill().execute({"message": "hello"})

    assert result.success is True
    assert result.skill_name == "echo"
    assert result.version == "v1"
    assert result.output == {"message": "hello"}


def test_skill_registry_can_register_get_and_list():
    registry = SkillRegistry()
    skill = EchoSkill()

    registry.register(skill)

    assert registry.get("echo") is skill
    assert registry.list_skills() == [skill]


def test_same_name_different_versions_can_register():
    registry = SkillRegistry()
    v1 = EchoSkill()
    v2 = EchoSkillV2()

    registry.register(v1)
    registry.register(v2)

    assert registry.get("echo", version="v1") is v1
    assert registry.get("echo", version="v2") is v2


def test_duplicate_name_version_raises_by_default():
    registry = SkillRegistry()
    registry.register(EchoSkill())

    with pytest.raises(SkillAlreadyRegisteredError):
        registry.register(EchoSkill())


def test_get_specific_version_is_correct():
    registry = SkillRegistry()
    v1 = EchoSkill()
    v2 = EchoSkillV2()
    registry.register(v1)
    registry.register(v2)

    assert registry.get("echo", version="v1") is v1


def test_get_latest_version_returns_most_recently_registered_version():
    registry = SkillRegistry()
    v1 = EchoSkill()
    v2 = EchoSkillV2()
    registry.register(v1)
    registry.register(v2)

    assert registry.get("echo") is v2


def test_skill_run_receives_execution_context():
    context = SkillExecutionContext(
        task_id="task_1",
        session_id="session_1",
        thread_id="thread_1",
        metadata={"caller": "test"},
    )

    result = EchoSkill().execute({"message": "hello"}, context=context)

    assert result.success is True
    assert result.metadata["task_id"] == "task_1"
    assert result.metadata["session_id"] == "session_1"
    assert result.metadata["thread_id"] == "thread_1"


def test_skill_run_can_receive_memory_context_without_using_it_for_decisions():
    memory_context = MemoryContext()
    context = SkillExecutionContext(memory_context=memory_context)

    result = EchoSkill().execute({"message": "memory is optional"}, context=context)

    assert result.success is True
    assert result.output == {"message": "memory is optional"}
    assert result.metadata["has_memory_context"] is True


def test_skill_failure_is_wrapped_as_failed_result():
    result = FailingSkill().execute({"value": 1})

    assert result.success is False
    assert result.skill_name == "failing_demo"
    assert result.error == "expected skill failure"


def test_keyword_extract_skill_is_deterministic():
    result = KeywordExtractSkill().execute({"text": "Python and LangGraph recruiting workflow"})

    assert result.success is True
    assert result.output == {"keywords": ["Python", "LangGraph"]}


def test_candidate_match_stub_skill_is_deterministic():
    result = CandidateMatchStubSkill().execute(
        {
            "required_skills": ["Python", "PyTorch", "LangGraph"],
            "candidate_skills": ["Python", "LangGraph"],
        }
    )

    assert result.success is True
    assert result.output == {"matched_skills": ["LangGraph", "Python"], "score": 66.67}


def test_phase3a_does_not_import_real_retrieval_modules(monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    registry = SkillRegistry()
    registry.register(EchoSkill())
    result = registry.get("echo").execute({"safe": True})

    assert result.success is True
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
