import builtins
import importlib
import sys

from src.domain.models import CandidateProfile, JobRequirement, ResumeDocument
from src.runtime import SQLiteRuntimeStore, SessionManager, TaskManager
from src.skills import (
    RetrieverSkill,
    SkillEvalCase,
    SkillEvalRunner,
    SkillExecutionContext,
    SkillExecutionRecorder,
    SkillExecutor,
    SkillNodeAdapter,
    SkillRegistry,
    validate_skill_wrapper_contract,
)
from src.skills.contracts import build_default_wrapper_contract_cases


def block_real_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes) or name == "HuggingFaceEmbedding":
            raise ModuleNotFoundError(f"blocked retrieval import in Phase3I test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def make_registry(*skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def make_job_requirement():
    return JobRequirement(
        job_id="job_1",
        raw_text="Need Python LangGraph engineer",
        title="Agent Engineer",
        required_skills=["Python", "LangGraph"],
    ).to_dict()


def make_candidate(candidate_id="candidate_1"):
    return CandidateProfile(
        candidate_id=candidate_id,
        name="Alice",
        skills=["Python", "LangGraph"],
        education="Bachelor",
        experience=["Built agent workflows"],
    ).to_dict()


def make_resume(resume_id="resume_1"):
    return ResumeDocument(
        resume_id=resume_id,
        candidate_id="candidate_1",
        source_path="fixtures/resume_1.txt",
        raw_text="Alice built Python LangGraph systems.",
        chunks=["Alice built Python LangGraph systems."],
    ).to_dict()


def make_runtime(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    session = SessionManager(store).create_session(metadata={"phase": "3I"})
    task = TaskManager(store).create_task(session.session_id, jd_text="招聘JD", thread_id="thread-retriever-skill")
    context = SkillExecutionContext(
        task_id=task.task_id,
        session_id=session.session_id,
        thread_id=task.thread_id,
        metadata={"source": "phase3i-test"},
    )
    return store, task, context


def event_types(store, task_id):
    return [event.event_type for event in store.list_events_by_task(task_id)]


def retriever_contract_case():
    return {
        case.contract.skill_name: case
        for case in build_default_wrapper_contract_cases()
    }["resume_retrieve"]


def test_retriever_skill_with_fake_callable_executes_successfully():
    skill = RetrieverSkill(
        retrieve_callable=lambda input_data, context: {
            "candidates": [make_candidate()],
            "resume_documents": [make_resume()],
            "evidence": ["Alice has Python and LangGraph experience"],
        }
    )

    result = skill.execute({"query": "Python LangGraph", "job_requirement": make_job_requirement()})

    assert result.success is True
    assert result.output["candidates"][0]["candidate_id"] == "candidate_1"
    assert result.output["resume_documents"][0]["resume_id"] == "resume_1"
    assert result.output["evidence"] == ["Alice has Python and LangGraph experience"]


def test_retriever_skill_dict_with_candidates_succeeds():
    result = RetrieverSkill(
        retrieve_callable=lambda input_data, context: {"candidates": [make_candidate()]}
    ).execute({"query": "Python"})

    assert result.success is True
    assert result.output["candidates"][0]["name"] == "Alice"


def test_retriever_skill_dict_with_resume_documents_succeeds():
    result = RetrieverSkill(
        retrieve_callable=lambda input_data, context: {"resume_documents": [make_resume()]}
    ).execute({"query": "Python"})

    assert result.success is True
    assert result.output["resume_documents"][0]["source_path"] == "fixtures/resume_1.txt"


def test_retriever_skill_dict_with_evidence_succeeds():
    result = RetrieverSkill(
        retrieve_callable=lambda input_data, context: {"evidence": ["resume chunk evidence"]}
    ).execute({"job_requirement": make_job_requirement()})

    assert result.success is True
    assert result.output["evidence"] == ["resume chunk evidence"]


def test_retriever_skill_list_output_is_converted_to_evidence():
    result = RetrieverSkill(
        retrieve_callable=lambda input_data, context: ["evidence one", "evidence two"]
    ).execute({"query": "Python"})

    assert result.success is True
    assert result.output == {"evidence": ["evidence one", "evidence two"]}


def test_retriever_skill_top_k_truncates_candidates_and_evidence():
    result = RetrieverSkill(
        retrieve_callable=lambda input_data, context: {
            "candidates": [make_candidate("candidate_1"), make_candidate("candidate_2")],
            "evidence": ["one", "two", "three"],
        }
    ).execute({"query": "Python", "top_k": 1})

    assert result.success is True
    assert len(result.output["candidates"]) == 1
    assert result.output["candidates"][0]["candidate_id"] == "candidate_1"
    assert result.output["evidence"] == ["one"]


def test_retriever_skill_empty_dict_or_no_valid_fields_fails():
    result = RetrieverSkill(retrieve_callable=lambda input_data, context: {}).execute({"query": "Python"})

    assert result.success is False
    assert "candidates, resume_documents, or evidence" in result.error


def test_retriever_skill_invalid_list_field_type_fails():
    result = RetrieverSkill(
        retrieve_callable=lambda input_data, context: {"candidates": {"candidate_id": "bad"}}
    ).execute({"query": "Python"})

    assert result.success is False
    assert "candidates" in result.error


def test_retriever_skill_callable_exception_is_wrapped_as_failed_result():
    def fail(input_data, context):
        raise RuntimeError("retrieve failed")

    result = RetrieverSkill(retrieve_callable=fail).execute({"query": "Python"})

    assert result.success is False
    assert result.error == "retrieve failed"


def test_retriever_skill_can_register_to_skill_registry():
    skill = RetrieverSkill(retrieve_callable=lambda input_data, context: {"evidence": ["safe"]})
    registry = make_registry(skill)

    assert registry.get("resume_retrieve") is skill


def test_retriever_skill_executor_records_skill_events(tmp_path):
    store, task, context = make_runtime(tmp_path)
    registry = make_registry(RetrieverSkill(retrieve_callable=lambda input_data, context: {"evidence": ["safe"]}))
    executor = SkillExecutor(registry, recorder=SkillExecutionRecorder(store))

    result = executor.execute("resume_retrieve", {"query": "Python"}, context=context)

    assert result.success is True
    assert result.output["evidence"] == ["safe"]
    assert event_types(store, task.task_id) == [
        "task_created",
        "skill_started",
        "skill_completed",
    ]
    assert store.list_events_by_task(task.task_id)[-1].payload["skill_name"] == "resume_retrieve"


def test_retriever_skill_eval_runner_can_run_fixture_case():
    registry = make_registry(RetrieverSkill(retrieve_callable=lambda input_data, context: {"evidence": ["safe"]}))
    runner = SkillEvalRunner(registry)

    result = runner.run_case(
        SkillEvalCase(
            case_id="resume_retrieve_case",
            skill_name="resume_retrieve",
            input_data={"query": "Python"},
            expected_output={"evidence": ["safe"]},
        )
    )

    assert result.passed is True


def test_retriever_skill_node_adapter_maps_fake_state():
    registry = make_registry(
        RetrieverSkill(
            retrieve_callable=lambda input_data, context: {
                "evidence": [f"evidence for {input_data['query']}"],
                "candidates": [make_candidate()],
            }
        )
    )
    adapter = SkillNodeAdapter(
        registry=registry,
        skill_name="resume_retrieve",
        input_mapper=lambda state: {
            "query": state["extracted_jd"]["search_query"],
            "job_requirement": state["job_requirement"],
            "top_k": 1,
        },
        output_mapper=lambda result, state: {
            "retrieved_evidence": result.output["evidence"],
            "candidate_pool": result.output.get("candidates", []),
        },
    )

    update = adapter(
        {
            "extracted_jd": {"search_query": "Python"},
            "job_requirement": make_job_requirement(),
        }
    )

    assert update["retrieved_evidence"] == ["evidence for Python"]
    assert update["candidate_pool"][0]["candidate_id"] == "candidate_1"
    assert update["skill_execution_metadata"]["skill_name"] == "resume_retrieve"


def test_retriever_skill_passes_skill_wrapper_contract():
    assert validate_skill_wrapper_contract(retriever_contract_case()) is True


def test_importing_agent_adapters_does_not_import_real_retrieval_modules(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    sys.modules.pop("src.skills.agent_adapters", None)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    sys.modules.pop("llama_index", None)
    sys.modules.pop("chromadb", None)

    module = importlib.import_module("src.skills.agent_adapters")
    skill = module.RetrieverSkill(retrieve_callable=lambda input_data, context: {"evidence": ["safe"]})
    result = skill.execute({"query": "safe"})

    assert result.success is True
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
    assert "llama_index" not in sys.modules
    assert "chromadb" not in sys.modules


def test_phase3i_does_not_modify_real_graph():
    with open("src/core/graph.py", "r", encoding="utf-8") as graph_file:
        graph_source = graph_file.read()

    assert "SkillRegistry" not in graph_source
    assert "resume_retrieve" not in graph_source
