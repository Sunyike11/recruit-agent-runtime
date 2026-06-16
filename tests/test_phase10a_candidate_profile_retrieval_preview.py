import inspect
import json

from src.runtime import variant_runner
from src.runtime.candidate_preview import (
    CandidatePreviewBuildConfig,
    build_candidate_preview_quality_audit,
    build_candidate_profile_previews_from_retrieval_results,
    candidate_profile_preview_to_matcher_input,
    group_retrieval_chunks_by_candidate,
)
from src.runtime.variant_runner import build_real_skill_wrapper_variant_runner


FULL_CHUNK = (
    "FULL PRIVATE RESUME TEXT SHOULD NOT LEAK. 张三本科计算机，参与 RAG Agent 检索匹配平台项目，"
    "使用 Python LangGraph Chroma Docker 部署自动化评估系统，有工程实践。"
)


def fake_planner(_input, _context=None):
    return {
        "job_requirement": {
            "required_skills": ["Python", "RAG", "LangGraph"],
            "tech_stack": ["Python", "RAG", "LangGraph"],
            "metadata": {
                "search_query": "Python RAG LangGraph Agent",
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


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def fake_matcher(captured):
    def match(input_data, _context=None):
        candidate = dict(input_data["candidate_profile"])
        captured.append(candidate)
        return {
            "total_score": 86,
            "match_report": {
                "candidate_id": candidate["candidate_id"],
                "total_score": 86,
                "metadata": {
                    "source": "FakeMatcher",
                    "candidate_profile_preview": candidate["metadata"]["candidate_profile_preview"],
                },
            },
            "metadata": {"source": "FakeMatcher"},
        }

    return match


def test_chunks_grouped_by_candidate_id():
    grouped = group_retrieval_chunks_by_candidate(
        [
            {"text": "Python RAG", "metadata": {"candidate_id": "cand-a", "file_name": "a.pdf"}},
            {"text": "LangGraph Agent", "metadata": {"candidate_id": "cand-a", "file_name": "a.pdf"}},
        ]
    )

    assert list(grouped.keys()) == ["cand-a"]
    assert len(grouped["cand-a"]) == 2


def test_chunks_grouped_by_document_id_file_name_fallback():
    result = build_candidate_profile_previews_from_retrieval_results(
        [
            {"text": "Python RAG", "metadata": {"document_id": "doc-1", "file_name": "李四简历.pdf"}},
            {"text": "Agent 平台", "metadata": {"document_id": "doc-1", "file_name": "李四简历.pdf"}},
        ]
    )

    assert result.candidate_profile_preview_count == 1
    preview = result.previews[0]
    assert preview.source_document_id == "doc-1"
    assert preview.candidate_name == "李四"
    assert preview.evidence_chunk_count == 2


def test_candidate_name_from_metadata_and_file_name():
    metadata_result = build_candidate_profile_previews_from_retrieval_results(
        [{"text": "Python", "metadata": {"candidate_id": "cand-meta", "candidate_name": "王五", "file_name": "x.pdf"}}]
    )
    file_result = build_candidate_profile_previews_from_retrieval_results(
        [{"text": "Python", "metadata": {"file_name": "赵六简历.pdf"}}]
    )

    assert metadata_result.previews[0].candidate_name == "王五"
    assert file_result.previews[0].candidate_name == "赵六"


def test_missing_name_produces_quality_flag():
    result = build_candidate_profile_previews_from_retrieval_results(
        [{"text": "Python RAG", "metadata": {"document_id": "doc-no-name"}}]
    )

    assert "candidate_name_missing" in result.previews[0].preview_quality_flags


def test_deterministic_keyword_extraction_and_matched_terms():
    result = build_candidate_profile_previews_from_retrieval_results(
        [
            {
                "text": FULL_CHUNK,
                "metadata": {"candidate_id": "cand-keywords", "candidate_name": "张三", "file_name": "zhangsan.pdf"},
            }
        ],
        raw_jd="招聘 Python RAG LangGraph 工程师",
    )
    preview = result.previews[0]

    assert {"Python", "RAG", "LangGraph", "Chroma", "Docker"}.issubset(set(preview.skills))
    assert "项目" in preview.project_keywords
    assert "本科" in preview.education_keywords
    assert "工程" in preview.experience_keywords
    assert "Python" in preview.matched_query_terms
    assert "RAG" in preview.matched_query_terms


def test_evidence_summary_truncated_and_no_raw_full_chunk_text():
    result = build_candidate_profile_previews_from_retrieval_results(
        [{"text": FULL_CHUNK * 4, "metadata": {"candidate_name": "张三", "file_name": "zhangsan.pdf"}}],
        config=CandidatePreviewBuildConfig(max_evidence_chars=80),
    )
    preview_dict = result.previews[0].to_dict()
    serialized = json.dumps(preview_dict, ensure_ascii=False)

    assert len(preview_dict["evidence_summary"]) <= 80
    assert "summary_truncated" in preview_dict["preview_quality_flags"]
    assert FULL_CHUNK * 4 not in serialized


def test_quality_audit_counts():
    result = build_candidate_profile_previews_from_retrieval_results(
        [{"text": FULL_CHUNK, "metadata": {"candidate_name": "张三", "file_name": "zhangsan.pdf"}}]
    )
    audit = build_candidate_preview_quality_audit(result.previews)

    assert audit["candidate_profile_preview_count"] == 1
    assert audit["candidate_name_present_count"] == 1
    assert audit["skills_present_count"] == 1
    assert audit["project_keywords_present_count"] == 1
    assert audit["evidence_summary_present_count"] == 1


def test_matcher_input_compatibility_summary_only():
    result = build_candidate_profile_previews_from_retrieval_results(
        [{"text": FULL_CHUNK, "metadata": {"candidate_name": "张三", "file_name": "zhangsan.pdf"}}]
    )
    matcher_input = candidate_profile_preview_to_matcher_input(result.previews[0])
    serialized = json.dumps(matcher_input, ensure_ascii=False)

    assert matcher_input["candidate_profile_preview"] is True
    assert matcher_input["metadata"]["candidate_profile_preview"] is True
    assert matcher_input["metadata"]["source"] == "document_chunk_projection"
    assert isinstance(matcher_input["evidence_summary"], dict)
    assert matcher_input["summary_only"] is True
    assert FULL_CHUNK not in serialized


def test_enhanced_preview_integrates_with_variant_runner_summary():
    captured = []

    def retriever(_input, _context=None):
        return {
            "resume_documents": [
                {
                    "text": FULL_CHUNK,
                    "metadata": {"candidate_name": "张三", "file_name": "zhangsan.pdf"},
                    "score": 0.93,
                }
            ],
            "evidence": [],
            "metadata": {"source": "document_chunk_retrieval", "summary_only": True},
        }

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=retriever,
        match_callable=fake_matcher(captured),
        refine_callable=fake_refiner,
        enable_candidate_preview_projection=True,
    )

    summary = runner("招聘 Python RAG LangGraph 工程师")

    assert summary["status"] == "ok"
    assert summary["enhanced_candidate_preview_used"] is True
    assert summary["candidate_preview_fallback_used"] is False
    assert summary["candidate_preview_grouped_document_count"] == 1
    assert summary["candidate_name_present_count"] == 1
    assert summary["skills_present_count"] == 1
    assert summary["project_keywords_present_count"] == 1
    assert summary["evidence_summary_present_count"] == 1
    assert captured[0]["project_keywords"]


def test_fallback_to_legacy_preview_on_enhanced_builder_failure(monkeypatch):
    def broken_builder(*_args, **_kwargs):
        raise RuntimeError("builder exploded with private text")

    monkeypatch.setattr(variant_runner, "build_candidate_profile_previews_from_retrieval_results", broken_builder)

    def retriever(_input, _context=None):
        return {
            "resume_documents": [
                {
                    "rank": 1,
                    "text_length": len(FULL_CHUNK),
                    "metadata_keys": ["file_name"],
                    "file_name": "legacy.pdf",
                    "skills": ["Python", "RAG"],
                    "score_present": True,
                }
            ],
            "evidence": [],
            "metadata": {"source": "document_chunk_retrieval", "summary_only": True},
        }

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=retriever,
        match_callable=fake_matcher([]),
        refine_callable=fake_refiner,
        enable_candidate_preview_projection=True,
    )

    summary = runner("招聘 Python RAG 工程师")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "ok"
    assert summary["candidate_preview_fallback_used"] is True
    assert summary["enhanced_candidate_preview_used"] is False
    assert FULL_CHUNK not in serialized


def test_required_helpers_do_not_import_real_runtime_dependencies():
    source = inspect.getsource(build_candidate_profile_previews_from_retrieval_results)
    variant_source = inspect.getsource(variant_runner.project_retrieval_documents_to_candidate_previews)

    assert "llama_index" not in source + variant_source
    assert "chromadb" not in source + variant_source
    assert "HuggingFace" not in source + variant_source
    assert "MCP" not in source + variant_source


def test_default_graph_behavior_is_not_modified():
    import src.core.graph as graph

    assert "CandidateProfilePreview" not in graph.create_recruit_graph.__code__.co_names
    assert "build_candidate_profile_previews_from_retrieval_results" not in graph.create_recruit_graph.__code__.co_names
