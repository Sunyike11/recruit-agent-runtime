import json
import os
from pathlib import Path

from src.runtime.variant_runner import (
    build_real_retriever_callable,
    build_real_skill_wrapper_variant_runner,
    build_resume_retriever_for_runtime,
    check_retriever_embedding_readiness,
)


SENSITIVE_TOKEN = "TEST_HF_TOKEN_SHOULD_NOT_LEAK"
SENSITIVE_TEXT = "FULL-RESUME-CHUNK-SHOULD-NOT-LEAK-PHASE8N Python RAG"


class FakeResumeRetriever:
    def __init__(self):
        self.index = object()

    def search(self, query, k=3):
        return [
            {
                "text": SENSITIVE_TEXT,
                "metadata": {"file_name": "phase8n.pdf", "source": "/secret/path/phase8n.pdf"},
                "score": 0.7,
            }
        ][:k]


def fake_import_module(name):
    if name in {
        "src.services.retriever",
        "llama_index.core",
        "llama_index.embeddings.huggingface",
        "chromadb",
    }:
        return object()
    raise ModuleNotFoundError(name)


def readiness_available(**_kwargs):
    return {
        "embedding_model_name": "BAAI/bge-small-zh-v1.5",
        "hf_token_status": "missing",
        "hf_home_set": True,
        "transformers_cache_set": False,
        "sentence_transformers_cache_set": False,
        "cache_probe_supported": True,
        "cache_likely_available": True,
        "network_required_unknown": True,
        "allow_hf_network_probe": False,
        "embedding_dependency_importable": True,
        "chroma_dependency_importable": True,
        "llama_index_dependency_importable": True,
        "summary_only": True,
    }


def readiness_unavailable(**_kwargs):
    data = readiness_available()
    data["cache_likely_available"] = False
    return data


def readiness_unknown(**_kwargs):
    data = readiness_available()
    data["cache_likely_available"] = "unknown"
    data["hf_home_set"] = False
    return data


def fake_planner(_input, _context=None):
    return {
        "job_requirement": {
            "required_skills": ["Python", "RAG"],
            "metadata": {
                "search_query": "Python RAG",
                "planner_fallback_used": True,
                "planner_fallback_type": "deterministic",
                "real_planner_failed": True,
                "fallback_not_real_planner_success": True,
            },
        }
    }


def fake_matcher(_input, _context=None):
    return {"total_score": 70, "match_report": {"total_score": 70}}


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def test_embedding_readiness_diagnostics_do_not_leak_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SENSITIVE_TOKEN)
    diagnostics = check_retriever_embedding_readiness(
        config={"embedding_model": "BAAI/bge-small-zh-v1.5"},
        import_module=fake_import_module,
    )
    serialized = json.dumps(diagnostics, ensure_ascii=False)

    assert diagnostics["hf_token_status"] == "set"
    assert diagnostics["embedding_model_name"] == "BAAI/bge-small-zh-v1.5"
    assert diagnostics["summary_only"] is True
    assert SENSITIVE_TOKEN not in serialized


def test_cache_available_allows_retriever_factory():
    called = {"factory": False}

    def factory():
        called["factory"] = True
        return FakeResumeRetriever()

    retriever, diagnostics = build_resume_retriever_for_runtime(
        retriever_factory=factory,
        require_embedding_cache_ready=True,
        embedding_readiness_checker=readiness_available,
    )

    assert isinstance(retriever, FakeResumeRetriever)
    assert called["factory"] is True
    assert diagnostics["retriever_init_stage"] == "ready"
    assert diagnostics["retriever_init_diagnostics"]["embedding_readiness"]["cache_likely_available"] is True


def test_cache_unavailable_with_skip_strategy_returns_cache_error():
    def factory():
        raise AssertionError("factory should not be called when cache readiness blocks")

    callable_ = build_real_retriever_callable(
        retriever_factory=factory,
        require_embedding_cache_ready=True,
        skip_if_embedding_cache_unavailable=True,
        embedding_readiness_checker=readiness_unavailable,
    )
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=callable_,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )
    summary = runner("招聘 Python RAG")

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "retriever_embedding_cache_unavailable"
    assert summary["retriever_init_stage"] == "embedding_readiness"
    assert summary["retriever_embedding_readiness"]["cache_likely_available"] is False


def test_cache_unknown_default_policy_continues_to_factory():
    called = {"factory": False}

    def factory():
        called["factory"] = True
        return FakeResumeRetriever()

    retriever, diagnostics = build_resume_retriever_for_runtime(
        retriever_factory=factory,
        require_embedding_cache_ready=False,
        embedding_readiness_checker=readiness_unknown,
    )

    assert isinstance(retriever, FakeResumeRetriever)
    assert called["factory"] is True
    assert diagnostics["retriever_init_diagnostics"]["embedding_readiness"]["cache_likely_available"] == "unknown"


def test_instantiate_failure_carries_embedding_diagnostics():
    def factory():
        raise RuntimeError("FULL EMBEDDING ERROR SHOULD NOT LEAK")

    callable_ = build_real_retriever_callable(
        retriever_factory=factory,
        require_embedding_cache_ready=False,
        embedding_readiness_checker=readiness_unknown,
    )
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=callable_,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )
    summary = runner("招聘 Python RAG")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["error_hint"] == "retriever_init_failed"
    assert summary["retriever_init_diagnostics"]["embedding_readiness"]["cache_likely_available"] == "unknown"
    assert "FULL EMBEDDING ERROR" not in serialized


def test_dependency_import_flags_can_be_faked():
    diagnostics = check_retriever_embedding_readiness(
        config={"embedding_model": "fake-model"},
        import_module=fake_import_module,
    )

    assert diagnostics["embedding_dependency_importable"] is True
    assert diagnostics["chroma_dependency_importable"] is True
    assert diagnostics["llama_index_dependency_importable"] is True


def test_planner_fallback_markers_are_preserved_when_cache_blocks():
    callable_ = build_real_retriever_callable(
        retriever_factory=FakeResumeRetriever,
        require_embedding_cache_ready=True,
        skip_if_embedding_cache_unavailable=True,
        embedding_readiness_checker=readiness_unavailable,
    )
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=callable_,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )
    summary = runner("招聘 Python RAG")

    assert summary["planner_fallback_used"] is True
    assert summary["fallback_not_real_planner_success"] is True
    assert summary["real_planner_failed"] is True


def test_memory_and_default_graph_are_not_modified():
    variant_source = Path("src/runtime/variant_runner.py").read_text(encoding="utf-8")
    graph_source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "MemorySQLiteStore" not in variant_source
    assert "check_retriever_embedding_readiness" not in graph_source
    assert os.environ.get("HF_TOKEN") != SENSITIVE_TOKEN
