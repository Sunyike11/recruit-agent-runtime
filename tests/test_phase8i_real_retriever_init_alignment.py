import builtins
import json
import sys

from src.runtime.variant_runner import (
    adapt_resume_retriever_results,
    build_real_retriever_callable,
    build_real_skill_wrapper_variant_runner,
    resolve_runtime_retriever_config,
)


SENSITIVE_TEXT = "FULL-RESUME-CHUNK-SHOULD-NOT-LEAK-PHASE8I"
SENSITIVE_ERROR = "FULL-RETRIEVER-INIT-ERROR-SHOULD-NOT-LEAK"


def block_real_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "llama_index.embeddings",
            "llama_index.vector_stores",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked real retriever dependency in Phase8I test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class FakeResumeRetriever:
    def __init__(self):
        self.index = object()

    def search(self, query, k=3):
        assert "Python" in query
        return [
            {
                "text": SENSITIVE_TEXT,
                "metadata": {"file_name": "alice.pdf", "source": "/private/path/alice.pdf"},
                "score": 0.91,
            }
        ][:k]


class FakeNoIndexRetriever:
    index = None

    def search(self, _query, k=3):
        return []


def fake_planner(_input, _context=None):
    return {
        "extracted_jd": {
            "tech_stack": ["Python", "RAG", "LangGraph"],
            "education": "",
            "must_have": [],
            "search_query": "Python RAG LangGraph",
        }
    }


def fake_matcher(_input, _context=None):
    return {"total_score": 80, "recommendation": "possible", "match_report": {"total_score": 80}}


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def test_resolve_runtime_retriever_config_returns_summary_only():
    config = resolve_runtime_retriever_config()
    serialized = json.dumps(config["summary"], ensure_ascii=False)

    assert config["summary"]["summary_only"] is True
    assert "chroma_db" not in serialized
    assert "BAAI" not in serialized
    assert "embedding_model" not in serialized


def test_real_retriever_callable_uses_fake_resume_retriever_factory():
    callable_ = build_real_retriever_callable(retriever_factory=FakeResumeRetriever)

    output = callable_({"query": "Python RAG LangGraph", "top_k": 1})
    serialized = json.dumps(output, ensure_ascii=False)

    assert output["candidates"] == []
    assert len(output["resume_documents"]) == 1
    assert output["resume_documents"][0]["text_length"] == len(SENSITIVE_TEXT)
    assert output["resume_documents"][0]["file_name"] == "alice.pdf"
    assert output["resume_documents"][0]["source"] == "alice.pdf"
    assert output["metadata"]["candidate_profile_level"] is False
    assert SENSITIVE_TEXT not in serialized
    assert "/private/path" not in serialized


def test_adapter_output_is_retriever_skill_compatible_summary_only():
    output = adapt_resume_retriever_results(
        [
            {
                "text": SENSITIVE_TEXT,
                "metadata": {"file_name": "alice.pdf", "source": "alice.pdf"},
                "score": 0.1,
            }
        ]
    )

    assert set(output.keys()) >= {"candidates", "resume_documents", "evidence", "metadata"}
    assert output["metadata"]["summary_only"] is True
    assert output["resume_documents"][0]["score_present"] is True


def test_retriever_init_failure_returns_stage_and_config_summary(monkeypatch):
    block_real_retrieval_imports(monkeypatch)

    def failing_factory():
        raise RuntimeError(SENSITIVE_ERROR)

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=build_real_retriever_callable(retriever_factory=failing_factory),
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )
    summary = runner("招聘 Python RAG LangGraph")
    serialized = json.dumps(summary, ensure_ascii=False)

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "retriever_init_failed"
    assert summary["retriever_init_stage"] == "instantiate_resume_retriever"
    assert summary["retriever_config_summary"]["summary_only"] is True
    assert SENSITIVE_ERROR not in serialized
    assert "chroma_db" not in serialized


def test_retriever_index_missing_is_index_load_failure(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    callable_ = build_real_retriever_callable(retriever_factory=FakeNoIndexRetriever)

    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=callable_,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )
    summary = runner("招聘 Python RAG LangGraph")

    assert summary["status"] == "failed"
    assert summary["error_hint"] == "retriever_init_failed"
    assert summary["retriever_init_stage"] == "index_load"


def test_no_retriever_callable_phase8e_behavior_is_unchanged():
    runner = build_real_skill_wrapper_variant_runner(
        planner_extract_callable=fake_planner,
        retrieve_callable=None,
        match_callable=fake_matcher,
        refine_callable=fake_refiner,
    )

    summary = runner("招聘 Python")

    assert summary["status"] == "skipped"
    assert summary["error_hint"] == "retriever_callable_required"


def test_default_graph_behavior_not_modified():
    with open("src/core/graph.py", "r", encoding="utf-8") as graph_file:
        source = graph_file.read()

    assert "build_real_retriever_callable" not in source
    assert "SkillRegistry" not in source


def test_required_path_does_not_import_real_chroma_or_llamaindex(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    build_real_retriever_callable(retriever_factory=FakeResumeRetriever)({"query": "Python", "top_k": 1})

    assert "chromadb" not in sys.modules
    assert "llama_index.core" not in sys.modules
    assert "MemorySQLiteStore" not in sys.modules
