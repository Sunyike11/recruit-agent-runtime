import builtins
import importlib
import sys

import pytest

from src.skills import (
    BaseSkill,
    SkillResult,
    SkillSpec,
    SkillWrapperContract,
    SkillWrapperContractCase,
    SkillWrapperContractError,
    build_default_wrapper_contract_cases,
    validate_skill_wrapper_contract,
)


def block_real_agent_and_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.planner",
            "src.agents.refiner",
            "src.agents.matcher",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real agent/retrieval import in Phase3H test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class MissingInputSchemaSkill(BaseSkill):
    spec = SkillSpec(
        name="planner_extract",
        version="v1",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {"job_requirement": {"type": "object"}}},
    )

    def run(self, input_data, context=None):
        return SkillResult(skill_name=self.spec.name, version=self.spec.version, success=True, output={})


class MissingOutputSchemaSkill(BaseSkill):
    spec = SkillSpec(
        name="query_refine",
        version="v1",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        output_schema={"type": "object", "properties": {}},
    )

    def run(self, input_data, context=None):
        return SkillResult(skill_name=self.spec.name, version=self.spec.version, success=True, output={})


def contract_case_by_name(skill_name):
    cases = {case.contract.skill_name: case for case in build_default_wrapper_contract_cases()}
    return cases[skill_name]


def test_default_wrapper_contract_cases_include_all_three_wrappers():
    cases = build_default_wrapper_contract_cases()

    assert [case.contract.skill_name for case in cases] == [
        "planner_extract",
        "query_refine",
        "resume_retrieve",
        "candidate_match",
    ]
    assert all(case.contract.supports_injected_callable is True for case in cases)
    assert all(case.contract.should_not_import_real_agent_on_module_import is True for case in cases)


def test_planner_extract_skill_passes_contract_validation():
    assert validate_skill_wrapper_contract(contract_case_by_name("planner_extract")) is True


def test_query_refine_skill_passes_contract_validation():
    assert validate_skill_wrapper_contract(contract_case_by_name("query_refine")) is True


def test_candidate_match_skill_passes_contract_validation():
    assert validate_skill_wrapper_contract(contract_case_by_name("candidate_match")) is True


def test_contract_validator_detects_missing_required_input_schema():
    valid_case = contract_case_by_name("planner_extract")
    bad_case = SkillWrapperContractCase(
        contract=valid_case.contract,
        skill_factory=lambda: MissingInputSchemaSkill(),
        input_data=valid_case.input_data,
        expected_output=valid_case.expected_output,
        failure_input_data=valid_case.failure_input_data,
        fake_state=valid_case.fake_state,
        input_mapper=valid_case.input_mapper,
        output_mapper=valid_case.output_mapper,
        expected_state_output_keys=valid_case.expected_state_output_keys,
    )

    with pytest.raises(SkillWrapperContractError, match="Input schema missing"):
        validate_skill_wrapper_contract(bad_case)


def test_contract_validator_detects_missing_required_output_schema():
    valid_case = contract_case_by_name("query_refine")
    bad_case = SkillWrapperContractCase(
        contract=valid_case.contract,
        skill_factory=lambda: MissingOutputSchemaSkill(),
        input_data=valid_case.input_data,
        expected_output=valid_case.expected_output,
        failure_input_data=valid_case.failure_input_data,
        fake_state=valid_case.fake_state,
        input_mapper=valid_case.input_mapper,
        output_mapper=valid_case.output_mapper,
        expected_state_output_keys=valid_case.expected_state_output_keys,
    )

    with pytest.raises(SkillWrapperContractError, match="Output schema missing"):
        validate_skill_wrapper_contract(bad_case)


def test_all_wrappers_can_register_to_skill_registry_through_contract():
    for case in build_default_wrapper_contract_cases():
        assert validate_skill_wrapper_contract(case) is True


def test_all_wrappers_can_execute_through_skill_executor_contract():
    for case in build_default_wrapper_contract_cases():
        assert validate_skill_wrapper_contract(case) is True


def test_all_wrappers_can_eval_through_skill_eval_runner_contract():
    for case in build_default_wrapper_contract_cases():
        assert validate_skill_wrapper_contract(case) is True


def test_all_wrappers_can_call_through_skill_node_adapter_contract():
    for case in build_default_wrapper_contract_cases():
        assert validate_skill_wrapper_contract(case) is True


def test_importing_agent_adapters_does_not_import_real_agents_or_retrieval_modules(monkeypatch):
    block_real_agent_and_retrieval_imports(monkeypatch)
    sys.modules.pop("src.skills.agent_adapters", None)
    sys.modules.pop("src.agents.planner", None)
    sys.modules.pop("src.agents.refiner", None)
    sys.modules.pop("src.agents.matcher", None)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    module = importlib.import_module("src.skills.agent_adapters")

    assert hasattr(module, "PlannerExtractSkill")
    assert hasattr(module, "QueryRefineSkill")
    assert hasattr(module, "RetrieverSkill")
    assert hasattr(module, "CandidateMatchSkill")
    assert "src.agents.planner" not in sys.modules
    assert "src.agents.refiner" not in sys.modules
    assert "src.agents.matcher" not in sys.modules
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules


def test_phase3h_does_not_modify_real_graph():
    with open("src/core/graph.py", "r", encoding="utf-8") as graph_file:
        graph_source = graph_file.read()

    assert "SkillRegistry" not in graph_source
    assert "planner_extract" not in graph_source
    assert "query_refine" not in graph_source
    assert "resume_retrieve" not in graph_source
    assert "candidate_match" not in graph_source
