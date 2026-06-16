import builtins
import sys

import pytest

from src.skills import (
    BaseSkill,
    CandidateMatchStubSkill,
    EchoSkill,
    KeywordExtractSkill,
    SkillManifest,
    SkillNodeAdapter,
    SkillRegistry,
    SkillSpec,
    load_skill_manifest_from_dict,
    validate_skill_manifest,
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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase3B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class FailingSkill(BaseSkill):
    spec = SkillSpec(name="failing_adapter_demo", version="v1")

    def run(self, input_data, context=None):
        raise RuntimeError("adapter skill failed")


class FakeSkillWorkflow:
    def __init__(self, nodes):
        self.nodes = nodes

    def run(self, state):
        current = state.copy()
        events = []
        for node_name, node in self.nodes:
            update = node(current)
            current.update(update)
            events.append({node_name: update})
        return current, events


def test_skill_manifest_can_load_from_dict():
    manifest = load_skill_manifest_from_dict(
        {
            "name": "demo_skill",
            "version": "v1",
            "description": "Demo manifest",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "required_tools": ["local_parser"],
            "required_memory_types": ["semantic"],
            "tags": ["demo"],
            "metadata": {"phase": "3B"},
        }
    )

    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "demo_skill"
    assert manifest.required_tools == ["local_parser"]


def test_manifest_can_validate_and_convert_to_skill_spec():
    data = {"name": "demo_skill", "version": "v1", "tags": ["demo"]}

    assert validate_skill_manifest(data) is True
    spec = load_skill_manifest_from_dict(data).to_skill_spec()

    assert spec.name == "demo_skill"
    assert spec.version == "v1"
    assert spec.tags == ["demo"]


def test_manifest_requires_name_and_version():
    with pytest.raises(ValueError):
        validate_skill_manifest({"name": "missing_version"})


def test_skill_node_adapter_extracts_input_from_state():
    captured = {}

    def input_mapper(state):
        captured["state"] = state
        return {"message": state["raw_text"]}

    adapter = SkillNodeAdapter(
        registry=SkillRegistry(),
        skill_name="echo",
        input_mapper=input_mapper,
        output_mapper=lambda result, state: {},
    )

    adapter.input_mapper({"raw_text": "hello"})

    assert captured["state"] == {"raw_text": "hello"}


def test_adapter_calls_echo_skill_and_writes_state_update():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="echo",
        input_mapper=lambda state: {"echoed": state["raw_text"]},
        output_mapper=lambda result, state: {"echo_output": result.output},
    )

    update = adapter({"raw_text": "hello"})

    assert update["echo_output"] == {"echoed": "hello"}
    assert update["skill_execution_metadata"]["skill_name"] == "echo"


def test_adapter_calls_keyword_extract_skill_and_writes_keywords():
    registry = SkillRegistry()
    registry.register(KeywordExtractSkill())
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="keyword_extract_stub",
        input_mapper=lambda state: {"text": state["jd_text"]},
        output_mapper=lambda result, state: {
            "extracted_jd": {"tech_stack": result.output["keywords"]},
            "next_action": "retrieve_candidates",
        },
    )

    update = adapter({"jd_text": "Python LangGraph matching"})

    assert update["extracted_jd"] == {"tech_stack": ["Python", "LangGraph"]}
    assert update["next_action"] == "retrieve_candidates"


def test_adapter_calls_candidate_match_stub_and_writes_fake_match_result():
    registry = SkillRegistry()
    registry.register(CandidateMatchStubSkill())
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="candidate_match_stub",
        input_mapper=lambda state: {
            "required_skills": state["required_skills"],
            "candidate_skills": state["candidate"]["skills"],
        },
        output_mapper=lambda result, state: {
            "final_reports": [
                {
                    "candidate_id": state["candidate"]["candidate_id"],
                    "matched_skills": result.output["matched_skills"],
                    "total_score": result.output["score"],
                }
            ],
            "next_action": "end",
        },
    )

    update = adapter(
        {
            "required_skills": ["Python", "PyTorch", "LangGraph"],
            "candidate": {"candidate_id": "candidate_1", "skills": ["Python", "LangGraph"]},
        }
    )

    assert update["final_reports"][0]["candidate_id"] == "candidate_1"
    assert update["final_reports"][0]["total_score"] == 66.67
    assert update["next_action"] == "end"


def test_adapter_failure_returns_error_update():
    registry = SkillRegistry()
    registry.register(FailingSkill())
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="failing_adapter_demo",
        input_mapper=lambda state: {"value": state["value"]},
        output_mapper=lambda result, state: {"handled_success": result.success},
    )

    update = adapter({"value": 1})

    assert update["handled_success"] is False
    assert update["skill_error"] == "adapter skill failed"
    assert update["skill_execution_metadata"]["success"] is False
    assert update["skill_execution_metadata"]["error"] == "adapter skill failed"


def test_adapter_generates_skill_execution_metadata():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="echo",
        input_mapper=lambda state: {"message": state["message"]},
        output_mapper=lambda result, state: {"echo_output": result.output},
    )

    update = adapter({"message": "hello"})
    metadata = update["skill_execution_metadata"]

    assert metadata == {
        "skill_name": "echo",
        "version": "v1",
        "success": True,
        "error": "",
        "input_keys": ["message"],
        "output_keys": ["echo_output"],
    }


def test_fake_skill_backed_node_can_execute_in_fake_workflow():
    registry = SkillRegistry()
    registry.register(KeywordExtractSkill())
    registry.register(CandidateMatchStubSkill())
    keyword_node = SkillNodeAdapter(
        registry=registry,
        skill_name="keyword_extract_stub",
        input_mapper=lambda state: {"text": state["jd_text"]},
        output_mapper=lambda result, state: {
            "required_skills": result.output["keywords"],
            "extracted_jd": {"tech_stack": result.output["keywords"]},
        },
    )
    match_node = SkillNodeAdapter(
        registry=registry,
        skill_name="candidate_match_stub",
        input_mapper=lambda state: {
            "required_skills": state["required_skills"],
            "candidate_skills": state["candidate"]["skills"],
        },
        output_mapper=lambda result, state: {
            "final_reports": [{"candidate_id": state["candidate"]["candidate_id"], "total_score": result.output["score"]}],
            "next_action": "end",
        },
    )
    workflow = FakeSkillWorkflow([("keyword_node", keyword_node), ("match_node", match_node)])

    final_state, events = workflow.run(
        {
            "jd_text": "Need Python and LangGraph",
            "candidate": {"candidate_id": "candidate_1", "skills": ["Python", "LangGraph"]},
        }
    )

    assert final_state["extracted_jd"] == {"tech_stack": ["Python", "LangGraph"]}
    assert final_state["final_reports"][0]["total_score"] == 100.0
    assert events[-1]["match_node"]["next_action"] == "end"


def test_phase3b_does_not_import_real_retrieval_modules(monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="echo",
        input_mapper=lambda state: {"safe": state["safe"]},
        output_mapper=lambda result, state: {"echo_output": result.output},
    )

    update = adapter({"safe": True})

    assert update["echo_output"] == {"safe": True}
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
