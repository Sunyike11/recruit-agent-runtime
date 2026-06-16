import builtins
from pathlib import Path

from src.domain.models import CandidateProfile, JobRequirement
from src.memory import (
    InMemoryMemoryGovernanceStore,
    MemoryContextEligibilityPolicy,
    MemoryRecord,
    MemoryType,
)
from src.skills import (
    CandidateMatchSkill,
    PlannerExtractSkill,
    RecruitmentSkillWorkflow,
    RetrieverSkill,
    SkillExecutionContext,
    SkillExecutor,
    SkillRegistry,
    build_shadow_workflow_memory_context,
    create_skill_execution_context_with_memory,
)


SENSITIVE_TEXT = "PRIVATE-RESUME-CONTENT-MUST-NOT-REACH-SHADOW-CONTEXT"


def promoted_memory(content="prefer LangGraph for agent workflows", **kwargs):
    metadata = {
        "promoted_from_reflection": True,
        "dry_run": False,
        "source_reflection_id": "reflection_1",
        "source_candidate_id": "candidate_1",
        "approved_by": "reviewer_1",
        "reviewer": "reviewer_1",
        "summary_only": True,
    }
    metadata.update(kwargs.pop("metadata", {}))
    return MemoryRecord(
        memory_type=kwargs.pop("memory_type", MemoryType.PROCEDURAL.value),
        content=content,
        importance=kwargs.pop("importance", 0.6),
        tags=kwargs.pop("tags", ["reflection_candidate", "procedure"]),
        metadata=metadata,
        **kwargs,
    )


def fake_planner(input_data, context):
    return {
        "job_requirement": JobRequirement(
            job_id="job_memory_shadow",
            raw_text=input_data["raw_text"],
            title="Agent Engineer",
            required_skills=["Python", "LangGraph"],
        ).to_dict()
    }


def fake_retriever_observes_memory(input_data, context):
    preview = (
        context.memory_context.format_for_prompt()
        if context and context.memory_context is not None
        else ""
    )
    seen = "prefer LangGraph" in preview
    return {
        "candidates": [
            CandidateProfile(
                candidate_id="candidate_memory_shadow",
                name="Alice",
                skills=["Python", "LangGraph"],
            ).to_dict()
        ],
        "evidence": ["deterministic evidence"],
        "metadata": {"memory_context_seen": seen},
    }


def fake_matcher(input_data, context):
    candidate = input_data["candidate_profile"]
    return {
        "total_score": 100.0,
        "recommendation": "strong_match",
        "match_report": {
            "job_id": input_data["job_requirement"]["job_id"],
            "candidate_id": candidate["candidate_id"],
            "total_score": 100.0,
        },
    }


def make_workflow():
    registry = SkillRegistry()
    registry.register(PlannerExtractSkill(extract_callable=fake_planner))
    registry.register(RetrieverSkill(retrieve_callable=fake_retriever_observes_memory))
    registry.register(CandidateMatchSkill(match_callable=fake_matcher))
    return RecruitmentSkillWorkflow(SkillExecutor(registry))


def test_eligible_memory_preview_can_be_built_for_shadow_workflow():
    context = build_shadow_workflow_memory_context([promoted_memory()])

    assert "prefer LangGraph" in context.format_for_prompt()
    assert "Promoted Memory Context Preview:" in context.format_for_prompt()
    assert context.is_empty() is False


def test_revoked_memory_is_not_in_shadow_preview():
    memory = promoted_memory(content="revoked memory text")
    store = InMemoryMemoryGovernanceStore()
    store.revoke_memory(memory.memory_id, "Withdrawn.", "reviewer_1")

    context = build_shadow_workflow_memory_context([memory], governance_store=store)

    assert "revoked memory text" not in context.format_for_prompt()
    assert context.is_empty() is True


def test_expired_and_superseded_memory_are_not_in_shadow_preview():
    expired = promoted_memory(content="expired memory text")
    superseded = promoted_memory(content="superseded memory text")
    store = InMemoryMemoryGovernanceStore()
    store.expire_memory(expired.memory_id, "Expired.", "reviewer_1")
    store.mark_superseded(superseded.memory_id, "replacement_memory", "Updated.", "reviewer_1")

    context = build_shadow_workflow_memory_context([expired, superseded], governance_store=store)

    assert "expired memory text" not in context.format_for_prompt()
    assert "superseded memory text" not in context.format_for_prompt()


def test_sensitive_memory_is_not_in_shadow_preview():
    context = build_shadow_workflow_memory_context(
        [promoted_memory(content=SENSITIVE_TEXT, metadata={"sensitive": True})]
    )

    assert SENSITIVE_TEXT not in context.format_for_prompt()


def test_memory_preview_can_be_attached_to_skill_execution_context():
    base = SkillExecutionContext(task_id="task_1", metadata={"source": "test"})
    preview = build_shadow_workflow_memory_context([promoted_memory()])

    context = create_skill_execution_context_with_memory(base, preview)

    assert context.task_id == "task_1"
    assert context.memory_context is preview
    assert context.metadata["source"] == "test"
    assert context.metadata["memory_context_mode"] == "shadow_preview"
    assert base.memory_context is None


def test_shadow_workflow_accepts_context_with_memory_preview():
    preview = build_shadow_workflow_memory_context([promoted_memory()])
    context = create_skill_execution_context_with_memory(memory_preview=preview)

    result = make_workflow().run("Need Python LangGraph engineer", context=context)

    assert result.success is True
    assert result.status == "completed"


def test_fake_skill_callable_can_observe_memory_context():
    preview = build_shadow_workflow_memory_context([promoted_memory()])
    context = create_skill_execution_context_with_memory(memory_preview=preview)

    result = make_workflow().run("Need Python LangGraph engineer", context=context)
    retriever_result = result.skill_results[1]

    assert retriever_result.skill_name == "resume_retrieve"
    assert retriever_result.output["metadata"]["memory_context_seen"] is True


def test_revoked_memory_is_not_observed_by_fake_skill():
    memory = promoted_memory()
    store = InMemoryMemoryGovernanceStore()
    store.revoke_memory(memory.memory_id, "Withdrawn.", "reviewer_1")
    preview = build_shadow_workflow_memory_context([memory], governance_store=store)
    context = create_skill_execution_context_with_memory(memory_preview=preview)

    result = make_workflow().run("Need Python LangGraph engineer", context=context)

    assert result.skill_results[1].output["metadata"]["memory_context_seen"] is False


def test_shadow_adapter_does_not_modify_production_graph_or_memory_builder():
    adapter_source = Path("src/skills/memory_context_adapter.py").read_text(encoding="utf-8")
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")
    builder_source = Path("src/memory/context.py").read_text(encoding="utf-8")

    assert "build_eligible_memory_context_preview" in adapter_source
    assert "RecruitmentSkillWorkflow" not in graph_source
    assert "build_shadow_workflow_memory_context" not in graph_source
    assert "ShadowWorkflowMemoryContext" not in graph_source
    assert "ShadowWorkflowMemoryContext" not in builder_source


def test_shadow_adapter_rejects_ambiguous_policy_configuration():
    store = InMemoryMemoryGovernanceStore()

    try:
        build_shadow_workflow_memory_context(
            [promoted_memory()],
            eligibility_policy=MemoryContextEligibilityPolicy(),
            governance_store=store,
        )
    except ValueError as exc:
        assert "not both" in str(exc)
    else:
        raise AssertionError("expected ambiguous policy configuration to fail")


def test_phase5j_does_not_depend_on_real_llm_mcp_or_external_tools(monkeypatch):
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("mcp", "llama_index", "chromadb", "src.agents", "src.services.retriever")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked import in Phase5J test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    preview = build_shadow_workflow_memory_context([promoted_memory()])
    context = create_skill_execution_context_with_memory(memory_preview=preview)
    result = make_workflow().run("Need Python LangGraph engineer", context=context)

    assert result.success is True
    assert blocked == []
