from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from src.domain.models import CandidateProfile, JobRequirement, ResumeDocument
from src.skills.agent_adapters import (
    CandidateMatchSkill,
    PlannerExtractSkill,
    QueryRefineSkill,
    RetrieverSkill,
    normalize_planner_output_to_job_requirement,
)
from src.skills.eval import SkillEvalCase, SkillEvalRunner
from src.skills.execution import SkillExecutor
from src.skills.models import SkillResult
from src.skills.node_adapter import SkillNodeAdapter
from src.skills.registry import SkillRegistry


class SkillWrapperContractError(AssertionError):
    pass


@dataclass
class SkillWrapperContract:
    skill_name: str
    required_input_keys: List[str]
    required_output_keys: List[str]
    supports_injected_callable: bool = True
    should_not_import_real_agent_on_module_import: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillWrapperContractCase:
    contract: SkillWrapperContract
    skill_factory: Callable[[], Any]
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    failure_input_data: Dict[str, Any]
    fake_state: Dict[str, Any]
    input_mapper: Callable[[dict], dict]
    output_mapper: Callable[[SkillResult, dict], dict]
    expected_state_output_keys: List[str] = field(default_factory=list)


def validate_skill_wrapper_contract(case: SkillWrapperContractCase) -> bool:
    skill = case.skill_factory()
    _validate_spec(skill, case.contract)

    registry = SkillRegistry()
    registry.register(skill)
    if registry.get(case.contract.skill_name) is not skill:
        raise SkillWrapperContractError(f"SkillRegistry did not return {case.contract.skill_name}")

    executor = SkillExecutor(registry)
    result = executor.execute(case.contract.skill_name, case.input_data)
    _validate_success_result(result, case.contract)

    eval_result = SkillEvalRunner(executor).run_case(
        SkillEvalCase(
            case_id=f"{case.contract.skill_name}_contract_case",
            skill_name=case.contract.skill_name,
            input_data=case.input_data,
            expected_output=case.expected_output,
        )
    )
    if not eval_result.passed:
        raise SkillWrapperContractError(f"SkillEvalRunner contract case failed: {eval_result.checks}")

    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name=case.contract.skill_name,
        input_mapper=case.input_mapper,
        output_mapper=case.output_mapper,
        skill_executor=executor,
    )
    state_update = adapter(case.fake_state)
    for key in case.expected_state_output_keys:
        if key not in state_update:
            raise SkillWrapperContractError(f"SkillNodeAdapter output missing key: {key}")
    metadata = state_update.get("skill_execution_metadata", {})
    if metadata.get("skill_name") != case.contract.skill_name or metadata.get("success") is not True:
        raise SkillWrapperContractError("SkillNodeAdapter did not produce successful skill metadata")

    failure_result = executor.execute(case.contract.skill_name, case.failure_input_data)
    if failure_result.success is not False:
        raise SkillWrapperContractError("Failure path did not return SkillResult(success=False)")

    return True


def build_default_wrapper_contract_cases() -> List[SkillWrapperContractCase]:
    job_requirement = JobRequirement(
        job_id="job_contract_1",
        raw_text="Need Python LangGraph agent engineer",
        title="Agent Engineer",
        required_skills=["Python", "LangGraph"],
        preferred_skills=["RAG"],
        education="Bachelor",
        experience_years=3,
        location="Remote",
    ).to_dict()
    candidate_profile = CandidateProfile(
        candidate_id="candidate_contract_1",
        name="Alice",
        skills=["Python", "LangGraph", "RAG"],
        education="Bachelor",
        experience=["Built agent workflows"],
        projects=["Recruit matching agent"],
    ).to_dict()
    resume_document = ResumeDocument(
        resume_id="resume_contract_1",
        candidate_id="candidate_contract_1",
        source_path="fixtures/resume_contract_1.txt",
        raw_text="Alice has Python and LangGraph experience.",
        chunks=["Alice has Python and LangGraph experience."],
    ).to_dict()
    normalized_job_requirement = normalize_planner_output_to_job_requirement(
        {
            "job_requirement": job_requirement,
            "extracted_keywords": ["Python", "LangGraph"],
        }
    )["job_requirement"]

    return [
        SkillWrapperContractCase(
            contract=SkillWrapperContract(
                skill_name="planner_extract",
                required_input_keys=["raw_text"],
                required_output_keys=["job_requirement"],
            ),
            skill_factory=lambda: PlannerExtractSkill(
                extract_callable=lambda input_data, context: {
                    "job_requirement": job_requirement,
                    "extracted_keywords": ["Python", "LangGraph"],
                }
            ),
            input_data={"raw_text": "Need Python LangGraph agent engineer"},
            expected_output={"job_requirement": normalized_job_requirement},
            failure_input_data={"metadata": {"missing": "raw_text"}},
            fake_state={"jd_text": "Need Python LangGraph agent engineer"},
            input_mapper=lambda state: {"raw_text": state["jd_text"]},
            output_mapper=lambda result, state: {
                "job_requirement": result.output["job_requirement"],
                "extracted_keywords": result.output.get("extracted_keywords", []),
            },
            expected_state_output_keys=["job_requirement", "extracted_keywords"],
        ),
        SkillWrapperContractCase(
            contract=SkillWrapperContract(
                skill_name="query_refine",
                required_input_keys=["query"],
                required_output_keys=["refined_query"],
            ),
            skill_factory=lambda: QueryRefineSkill(
                refine_callable=lambda input_data, context: {
                    "refined_query": f"{input_data['query']} LangGraph",
                    "reason": "contract fixture broadened query",
                }
            ),
            input_data={"query": "Python agent"},
            expected_output={"refined_query": "Python agent LangGraph"},
            failure_input_data={"context": "missing query"},
            fake_state={"extracted_jd": {"search_query": "Python agent"}, "refinement_advice": "broaden"},
            input_mapper=lambda state: {
                "query": state["extracted_jd"]["search_query"],
                "context": state["refinement_advice"],
            },
            output_mapper=lambda result, state: {"refined_query": result.output["refined_query"]},
            expected_state_output_keys=["refined_query"],
        ),
        SkillWrapperContractCase(
            contract=SkillWrapperContract(
                skill_name="resume_retrieve",
                required_input_keys=["query"],
                required_output_keys=["evidence"],
            ),
            skill_factory=lambda: RetrieverSkill(
                retrieve_callable=lambda input_data, context: {
                    "candidates": [candidate_profile],
                    "resume_documents": [resume_document],
                    "evidence": ["Python and LangGraph evidence"],
                }
            ),
            input_data={"query": "Python LangGraph", "top_k": 1},
            expected_output={"evidence": ["Python and LangGraph evidence"]},
            failure_input_data={"metadata": {"missing": "query_or_job_requirement"}},
            fake_state={"search_query": "Python LangGraph"},
            input_mapper=lambda state: {"query": state["search_query"], "top_k": 1},
            output_mapper=lambda result, state: {
                "retrieved_evidence": result.output["evidence"],
                "candidate_pool": result.output.get("candidates", []),
            },
            expected_state_output_keys=["retrieved_evidence", "candidate_pool"],
        ),
        SkillWrapperContractCase(
            contract=SkillWrapperContract(
                skill_name="candidate_match",
                required_input_keys=["job_requirement", "candidate_profile"],
                required_output_keys=["total_score", "match_report"],
            ),
            skill_factory=lambda: CandidateMatchSkill(
                match_callable=lambda input_data, context: {
                    "total_score": 88,
                    "recommendation": "strong_match",
                }
            ),
            input_data={
                "job_requirement": job_requirement,
                "candidate_profile": candidate_profile,
                "evidence": ["Python and LangGraph overlap"],
            },
            expected_output={"total_score": 88.0, "recommendation": "strong_match"},
            failure_input_data={"job_requirement": job_requirement},
            fake_state={"job_requirement": job_requirement, "candidate_profile": candidate_profile},
            input_mapper=lambda state: {
                "job_requirement": state["job_requirement"],
                "candidate_profile": state["candidate_profile"],
            },
            output_mapper=lambda result, state: {
                "match_report": result.output["match_report"],
                "total_score": result.output["total_score"],
            },
            expected_state_output_keys=["match_report", "total_score"],
        ),
    ]


def _validate_spec(skill, contract: SkillWrapperContract):
    if skill.spec.name != contract.skill_name:
        raise SkillWrapperContractError(f"Expected skill name {contract.skill_name}, got {skill.spec.name}")
    if not skill.spec.version:
        raise SkillWrapperContractError(f"Skill {contract.skill_name} must have a non-empty version")

    input_properties = skill.spec.input_schema.get("properties", {})
    output_properties = skill.spec.output_schema.get("properties", {})
    for key in contract.required_input_keys:
        if key not in input_properties:
            raise SkillWrapperContractError(f"Input schema missing required key: {key}")
    for key in contract.required_output_keys:
        if key not in output_properties:
            raise SkillWrapperContractError(f"Output schema missing required key: {key}")


def _validate_success_result(result: SkillResult, contract: SkillWrapperContract):
    if result.success is not True:
        raise SkillWrapperContractError(f"SkillExecutor failed for {contract.skill_name}: {result.error}")
    if not isinstance(result.output, dict):
        raise SkillWrapperContractError(f"Skill {contract.skill_name} output must be a dict")
    for key in contract.required_output_keys:
        if key not in result.output:
            raise SkillWrapperContractError(f"Skill output missing required key: {key}")
