import builtins
import json
import sys

from scripts.run_recruit_runtime import run_cli
from src.runtime.variant_runner import (
    adapt_resume_retriever_results,
    build_real_retriever_callable,
    build_real_skill_wrapper_variant_runner,
)


SENSITIVE_TEXT = "FULL-RESUME-CHUNK-SHOULD-NOT-LEAK"


def block_real_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real retriever dependency in Phase8F test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def fake_planner(input_data, _context=None):
    return {
        "job_requirement": {
            "job_id": "phase8f_job",
            "required_skills": ["Python", "RAG", "LangGraph"],
            "metadata": {"search_query": "Python RAG LangGraph"},
        },
        "extracted_keywords": ["Python", "RAG", "LangGraph"],
    }


def fake_matcher(input_data, _context=None):
    candidate_id = input_data["candidate_profile"].get("candidate_id", "candidate")
    return {
        "total_score": 88,
        "recommendation": "strong_match",
        "match_report": {"candidate_id": candidate_id, "total_score": 88},
    }


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"], "reason": "document retrieval needs profile projection later"}


def fake_search_runner(query, top_k):
    assert "Python" in query
    return [
        {
            "text": SENSITIVE_TEXT,
            "metadata": {
                "file_name": "alice_resume.pdf",
                "source": "/safe/path/alice_resume.pdf",
                "candidate_id": "candidate_1",
            },
            "score": 0.91,
        },
        {
            "text": "short safe text",
            "metadata": {"file_name": "bob_resume.pdf"},
            "score": None,
        },
    ][:top_k]


def test_adapter_converts_fake_search_results_to_retriever_skill_output():
    output = adapt_resume_retriever_results(fake_search_runner("Python", 2))
    serialized = json.dumps(output, ensure_ascii=False)

    assert output["candidates"] == []
    assert len(output["resume_documents"]) == 2
    assert len(output["evidence"]) == 2
    assert output["resume_documents"][0]["rank"] == 1
    assert output["resume_documents"][0]["text_length"] == len(SENSITIVE_TEXT)
    assert output["resume_documents"][0]["metadata_keys"] == ["candidate_id", "file_name", "source"]
    assert output["resume_documents"][0]["file_name"] == "alice_resume.pdf"
    assert output["resume_documents"][0]["source"] == "alice_resume.pdf"
    assert output["metadata"]["retriever_invoked"] is True
    assert output["metadata"]["candidate_profile_level"] is False
    assert SENSITIVE_TEXT not in serialized
    assert "/safe/path" not in serialized


def test_build_real_retriever_callable_uses_fake_search_runner_summary_only(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    callable_ = build_real_retriever_callable(search_runner=fake_search_runner)

    output = callable_(
        {
            "query": "Python RAG LangGraph",
            "top_k": 1,
        }
    )

    assert len(output["resume_documents"]) == 1
    assert output["metadata"]["source"] == "document_chunk_retrieval"
    assert output["metadata"]["summary_only"] is True
    assert "src.services.retriever" not in sys.modules


def test_real_skill_wrapper_variant_with_fake_real_retriever_callable_runs(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=build_real_retriever_callable(search_runner=fake_search_runner),
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner("招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师")

    assert summary["status"] == "ok"
    assert summary["real_skill_wrapper_mode"] is True
    assert summary["candidate_count"] == 0
    assert summary["retrieved_count"] == 4
    assert summary["match_count"] == 0
    assert summary["refined_query_present"] is True
    assert summary["metadata"]["runner_type"] == "real_skill_wrapper_variant"
    assert summary["metadata"]["production_graph_invoked"] is False


def test_no_retriever_callable_still_skips():
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=None,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner("招聘JD")

    assert summary["status"] == "skipped"
    assert summary["error_hint"] == "retriever_callable_required"


def test_cli_use_real_retriever_flag_with_fake_callable(monkeypatch, capsys):
    block_real_retrieval_imports(monkeypatch)

    exit_code = run_cli(
        [
            "--jd",
            "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师",
            "--use-skill-backed-variant",
            "--use-real-skill-wrappers",
            "--use-real-retriever",
            "--json",
        ],
        default_runner=lambda _raw_jd: {"status": "ok", "candidate_count": 1, "report_count": 1},
        real_retriever_callable=build_real_retriever_callable(search_runner=fake_search_runner),
        planner_extract_callable=fake_planner,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "skill_backed_variant"
    assert output["status"] == "ok"
    assert output["output_summary"]["status"] == "ok"
    assert output["output_summary"]["candidate_count"] == 0
    assert output["output_summary"]["top_score_present"] is False
    assert output["metadata"]["use_skill_backed_variant"] is True


def test_default_graph_path_unchanged_and_no_memory_access(monkeypatch, capsys):
    block_real_retrieval_imports(monkeypatch)
    calls = {"default": 0}

    def default_runner(_raw_jd):
        calls["default"] += 1
        return {"status": "ok", "candidate_count": 1, "report_count": 1}

    exit_code = run_cli(
        ["--jd", "招聘JD", "--graph-mode", "legacy", "--json"],
        default_runner=default_runner,
        real_retriever_callable=build_real_retriever_callable(search_runner=fake_search_runner),
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["runner_used"] == "default_graph"
    assert calls["default"] == 1
    assert "MemorySQLiteStore" not in json.dumps(output, ensure_ascii=False)
    assert "src.services.retriever" not in sys.modules


def test_retriever_search_failure_returns_sanitized_error_hint(monkeypatch):
    block_real_retrieval_imports(monkeypatch)

    def failing_search(_query, _top_k):
        raise RuntimeError(f"boom {SENSITIVE_TEXT}")

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=build_real_retriever_callable(search_runner=failing_search),
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )
    summary = runner("招聘JD")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "retriever_search_failed"
    assert SENSITIVE_TEXT not in serialized


def test_required_path_does_not_import_real_retriever_dependencies(monkeypatch):
    block_real_retrieval_imports(monkeypatch)

    output = build_real_retriever_callable(search_runner=fake_search_runner)({"query": "Python", "top_k": 1})

    assert output["metadata"]["retriever_invoked"] is True
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
