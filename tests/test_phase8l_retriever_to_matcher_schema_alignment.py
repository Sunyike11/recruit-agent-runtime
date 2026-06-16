import inspect
import json

from src.runtime.variant_runner import (
    build_real_skill_wrapper_variant_runner,
    project_retrieval_documents_to_candidate_previews,
)


FULL_CHUNK_TEXT = "FULL RESUME CHUNK SHOULD NOT LEAK Python RAG LangGraph"


def retriever_document_output(_input, _context=None):
    return {
        "resume_documents": [
            {
                "rank": 1,
                "text_length": len(FULL_CHUNK_TEXT),
                "metadata_keys": ["file_name", "source"],
                "score_present": True,
                "file_name": "candidate-a.pdf",
                "skills": ["Python", "RAG", "LangGraph"],
            }
        ],
        "evidence": [
            {
                "rank": 1,
                "text_length": len(FULL_CHUNK_TEXT),
                "metadata_keys": ["file_name", "source"],
                "score_present": True,
                "skills": ["Python", "RAG", "LangGraph"],
            }
        ],
        "metadata": {
            "source": "document_chunk_retrieval",
            "candidate_profile_level": False,
            "summary_only": True,
        },
    }


def fallback_planner(_input, _context=None):
    return {
        "job_requirement": {
            "required_skills": ["Python", "RAG", "LangGraph"],
            "tech_stack": ["Python", "RAG", "LangGraph"],
            "search_query": "Python RAG LangGraph",
            "metadata": {
                "planner_fallback_used": True,
                "planner_fallback_type": "deterministic",
                "real_planner_invoked": True,
                "real_planner_failed": True,
                "summary_only": True,
            },
        },
        "metadata": {
            "planner_fallback_used": True,
            "planner_fallback_type": "deterministic",
            "real_planner_invoked": True,
            "real_planner_failed": True,
            "summary_only": True,
        },
    }


def matcher_capture(captured):
    def match(input_data, _context=None):
        candidate = dict(input_data["candidate_profile"])
        captured.append(candidate)
        return {
            "total_score": 88,
            "recommendation": "strong_match",
            "match_report": {
                "candidate_id": candidate["candidate_id"],
                "total_score": 88,
                "candidate_profile_preview": candidate["metadata"]["candidate_profile_preview"],
            },
        }

    return match


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def test_project_retrieval_documents_to_candidate_previews_summary_only():
    previews = project_retrieval_documents_to_candidate_previews(retriever_document_output({}, None))
    serialized = json.dumps(previews, ensure_ascii=False)

    assert len(previews) == 1
    preview = previews[0]
    assert preview["candidate_id"].startswith("candidate_preview_")
    assert preview["source_document_id"] == "candidate-a.pdf"
    assert preview["skills"] == ["Python", "RAG", "LangGraph"]
    assert preview["metadata"]["candidate_profile_preview"] is True
    assert preview["metadata"]["source"] == "document_chunk_projection"
    assert preview["evidence_summary"]["text_length"] == len(FULL_CHUNK_TEXT)
    assert FULL_CHUNK_TEXT not in serialized


def test_candidate_preview_id_is_stable():
    first = project_retrieval_documents_to_candidate_previews(retriever_document_output({}, None))[0]
    second = project_retrieval_documents_to_candidate_previews(retriever_document_output({}, None))[0]

    assert first["candidate_id"] == second["candidate_id"]


def test_keyword_extraction_is_limited_to_safe_skill_list():
    previews = project_retrieval_documents_to_candidate_previews(
        {
            "resume_documents": [
                {
                    "rank": 1,
                    "text_length": 100,
                    "metadata_keys": [],
                    "score_present": True,
                    "skills": ["Python", "SecretSkill", "RAG"],
                }
            ],
            "evidence": [],
        }
    )

    assert previews[0]["skills"] == ["Python", "RAG"]


def test_variant_runner_projects_documents_to_matcher_candidate_preview():
    captured = []
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fallback_planner,
        retrieve_callable=retriever_document_output,
        match_callable=matcher_capture(captured),
        refine_callable=fake_refiner,
        enable_candidate_preview_projection=True,
    )

    summary = runner("Need Python RAG LangGraph engineer")

    assert summary["status"] == "ok"
    assert summary["candidate_count"] == 1
    assert summary["candidate_profile_preview_count"] == 1
    assert summary["match_count"] == 1
    assert captured[0]["metadata"]["candidate_profile_preview"] is True
    assert captured[0]["metadata"]["source"] == "document_chunk_projection"


def test_explicit_planner_fallback_markers_are_preserved():
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fallback_planner,
        retrieve_callable=retriever_document_output,
        match_callable=lambda input_data, _context=None: {
            "total_score": 80,
            "recommendation": "possible_match",
            "match_report": {
                "candidate_id": input_data["candidate_profile"]["candidate_id"],
                "total_score": 80,
            },
        },
        refine_callable=fake_refiner,
        enable_candidate_preview_projection=True,
    )

    summary = runner("Need Python RAG")

    assert summary["planner_fallback_used"] is True
    assert summary["planner_fallback_type"] == "deterministic"
    assert summary["real_planner_invoked"] is True
    assert summary["real_planner_failed"] is True
    assert summary["fallback_not_real_planner_success"] is True


def test_matcher_failure_diagnostics_do_not_leak_full_resume_text():
    def failing_matcher(_input, _context=None):
        raise RuntimeError(FULL_CHUNK_TEXT)

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fallback_planner,
        retrieve_callable=retriever_document_output,
        match_callable=failing_matcher,
        refine_callable=fake_refiner,
        enable_candidate_preview_projection=True,
    )

    summary = runner("Need Python RAG")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "matcher_wrapper_failed"
    assert summary["candidate_profile_preview_count"] == 1
    assert FULL_CHUNK_TEXT not in serialized


def test_default_graph_behavior_is_not_modified():
    import src.core.graph as graph

    assert "project_retrieval_documents_to_candidate_previews" not in graph.create_recruit_graph.__code__.co_names
    assert "SkillRegistry" not in graph.create_recruit_graph.__code__.co_names


def test_memory_is_not_read_or_written():
    source = inspect.getsource(project_retrieval_documents_to_candidate_previews)

    assert "MemorySQLiteStore" not in source
    assert "src.memory" not in source
