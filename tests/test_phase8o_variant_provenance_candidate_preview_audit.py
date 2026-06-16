import inspect
import json
from pathlib import Path

from src.runtime.entry import RuntimeEntryConfig, RuntimeEntryHarness
from src.runtime.variant_runner import (
    audit_candidate_profile_previews,
    build_real_retriever_callable,
    build_real_skill_wrapper_variant_runner,
    build_variant_provenance_summary,
    project_retrieval_documents_to_candidate_previews,
)


FULL_CHUNK = "FULL PHASE8O RESUME CHUNK SHOULD NOT LEAK Python RAG LangGraph"


class FakeRetriever:
    index = object()

    def search(self, _query, k=3):
        return [
            {
                "text": FULL_CHUNK,
                "metadata": {"file_name": "phase8o.pdf", "source": "/secret/phase8o.pdf"},
                "score": 0.91,
            }
        ][:k]


def fallback_planner(_input, _context=None):
    return {
        "job_requirement": {
            "required_skills": ["Python", "RAG", "LangGraph"],
            "metadata": {
                "search_query": "Python RAG LangGraph",
                "source": "deterministic_planner_fallback",
                "planner_fallback_used": True,
                "planner_fallback_type": "deterministic",
                "real_planner_invoked": True,
                "real_planner_failed": True,
                "fallback_not_real_planner_success": True,
                "summary_only": True,
            },
        },
        "metadata": {
            "planner_fallback_used": True,
            "planner_fallback_type": "deterministic",
            "real_planner_invoked": True,
            "real_planner_failed": True,
            "fallback_not_real_planner_success": True,
            "summary_only": True,
        },
    }


def fake_matcher(input_data, _context=None):
    candidate = input_data["candidate_profile"]
    return {
        "total_score": 92,
        "recommendation": "strong_match",
        "match_report": {
            "candidate_id": candidate["candidate_id"],
            "total_score": 92,
            "metadata": {
                "source": "FakeMatcher",
                "candidate_profile_preview": candidate["metadata"]["candidate_profile_preview"],
            },
        },
        "metadata": {"source": "FakeMatcher"},
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def test_candidate_preview_audit_counts_summary_fields_only():
    previews = project_retrieval_documents_to_candidate_previews(
        {
            "resume_documents": [
                {
                    "rank": 1,
                    "text_length": len(FULL_CHUNK),
                    "metadata_keys": ["file_name", "source"],
                    "score_present": True,
                    "file_name": "phase8o.pdf",
                    "skills": ["Python", "RAG", "LangGraph"],
                }
            ],
            "evidence": [
                {
                    "rank": 1,
                    "text_length": len(FULL_CHUNK),
                    "metadata_keys": ["file_name", "source"],
                    "score_present": True,
                }
            ],
        }
    )
    audit = audit_candidate_profile_previews(previews)
    serialized = json.dumps({"previews": previews, "audit": audit}, ensure_ascii=False)

    assert audit["candidate_profile_preview_count"] == 1
    assert audit["candidate_id_present"] == 1
    assert audit["candidate_name_present"] == 1
    assert audit["skills_count"] == 3
    assert audit["evidence_summary_present"] == 1
    assert audit["source_document_id_present"] == 1
    assert audit["summary_only"] is True
    assert FULL_CHUNK not in serialized


def test_real_wrapper_variant_summary_contains_provenance_and_preview_audit():
    retriever = build_real_retriever_callable(retriever_factory=FakeRetriever)
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fallback_planner,
        retrieve_callable=retriever,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
        enable_candidate_preview_projection=True,
    )

    summary = runner("招聘 Python RAG LangGraph 工程师")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "ok"
    assert summary["candidate_count"] == 1
    assert summary["candidate_profile_preview_count"] == 1
    assert summary["report_count"] == 1
    assert summary["top_score_present"] is True
    assert summary["planner_source"] == "deterministic_planner_fallback"
    assert summary["retriever_source"] == "document_chunk_retrieval"
    assert summary["matcher_source"] == "FakeMatcher"
    assert summary["real_planner_invoked"] is True
    assert summary["planner_fallback_used"] is True
    assert summary["fallback_not_real_planner_success"] is True
    assert summary["retriever_factory_source"] == "injected_retriever_factory"
    assert summary["candidate_preview_source"] == "document_chunk_projection"
    assert summary["matcher_input_source"] == "candidate_profile_preview"
    assert summary["candidate_preview_audit"]["candidate_id_present"] == 1
    assert summary["candidate_preview_audit"]["skills_count"] == 3
    assert FULL_CHUNK not in serialized


def test_runtime_entry_preserves_provenance_fields_summary_only():
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fallback_planner,
        retrieve_callable=build_real_retriever_callable(retriever_factory=FakeRetriever),
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
        enable_candidate_preview_projection=True,
    )
    result = RuntimeEntryHarness().run(
        "招聘 Python RAG LangGraph 工程师",
        default_runner=lambda _jd: {"status": "ok"},
        variant_runner=runner,
        config=RuntimeEntryConfig(use_skill_backed_variant=True),
    )

    output = result.output_summary
    assert result.status == "ok"
    assert result.runner_used == "skill_backed_variant"
    assert output["planner_source"] == "deterministic_planner_fallback"
    assert output["retriever_source"] == "document_chunk_retrieval"
    assert output["matcher_source"] == "FakeMatcher"
    assert output["candidate_preview_audit"]["source_document_id_present"] == 1
    assert output["provenance_summary_only"] is True


def test_provenance_helper_is_public_and_summary_only():
    source = inspect.getsource(build_variant_provenance_summary)

    assert "full" not in source.lower()
    assert "resume_text" not in source
    assert "llm_response" not in source


def test_default_graph_behavior_is_not_modified():
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "build_variant_provenance_summary" not in graph_source
    assert "audit_candidate_profile_previews" not in graph_source
    assert "CandidatePreviewRecruitmentSkillWorkflow" not in graph_source


def test_no_memory_or_real_dependency_imports_in_audit_helpers():
    source = inspect.getsource(audit_candidate_profile_previews)
    projection_source = inspect.getsource(project_retrieval_documents_to_candidate_previews)

    assert "MemorySQLiteStore" not in source + projection_source
    assert "src.memory" not in source + projection_source
    assert "llama_index" not in source + projection_source
    assert "chromadb" not in source + projection_source
