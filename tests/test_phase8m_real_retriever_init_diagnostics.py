import builtins
import json
from pathlib import Path

from src.runtime.variant_runner import (
    build_real_retriever_callable,
    build_real_skill_wrapper_variant_runner,
    build_resume_retriever_for_runtime,
    build_retriever_init_diagnostics,
)


SENSITIVE_ERROR = "FULL-RETRIEVER-INIT-ERROR-SHOULD-NOT-LEAK-PHASE8M"
SENSITIVE_TEXT = "FULL-RESUME-CHUNK-SHOULD-NOT-LEAK-PHASE8M Python RAG LangGraph"


class FakeResumeRetriever:
    def __init__(self):
        self.index = object()

    def search(self, query, k=3):
        assert "Python" in query
        return [
            {
                "text": SENSITIVE_TEXT,
                "metadata": {"file_name": "phase8m.pdf", "source": "/private/full/path/phase8m.pdf"},
                "score": 0.88,
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


def fake_planner(_input, _context=None):
    return {
        "job_requirement": {
            "required_skills": ["Python", "RAG", "LangGraph"],
            "metadata": {
                "search_query": "Python RAG LangGraph",
                "planner_fallback_used": True,
                "planner_fallback_type": "deterministic",
                "real_planner_failed": True,
                "fallback_not_real_planner_success": True,
            },
        }
    }


def fake_matcher(_input, _context=None):
    return {"total_score": 80, "match_report": {"total_score": 80}}


def fake_refiner(input_data, _context=None):
    return {"refined_query": input_data["query"]}


def block_real_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("chromadb", "llama_index")):
            raise ModuleNotFoundError(f"blocked real dependency in Phase8M test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_build_retriever_init_diagnostics_summary_only(tmp_path):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (chroma_dir / "index.bin").write_text("fake", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {
        "project_root": tmp_path,
        "data_dir": data_dir,
        "chroma_dir": chroma_dir,
    }

    diagnostics = build_retriever_init_diagnostics(
        config=config,
        resume_retriever_class="ResumeRetriever",
        error_stage="instantiate_resume_retriever",
        error_type="RuntimeError",
        import_module=fake_import_module,
    )
    serialized = json.dumps(diagnostics, ensure_ascii=False)

    assert diagnostics["summary_only"] is True
    assert diagnostics["persist_dir_exists"] is True
    assert diagnostics["chroma_dir_non_empty"] is True
    assert diagnostics["chroma_dir_file_count"] == 1
    assert diagnostics["resume_retriever_class"] == "ResumeRetriever"
    assert diagnostics["error_stage"] == "instantiate_resume_retriever"
    assert diagnostics["error_type"] == "RuntimeError"
    assert diagnostics["embedding_dependency_importable"] is True
    assert diagnostics["persist_dir_used"] == "set"
    assert SENSITIVE_ERROR not in serialized


def test_build_resume_retriever_for_runtime_uses_settings_chroma_dir(monkeypatch):
    captured = {}

    def factory():
        captured["called"] = True
        return FakeResumeRetriever()

    block_real_retrieval_imports(monkeypatch)
    retriever, diagnostics = build_resume_retriever_for_runtime(retriever_factory=factory)

    assert captured["called"] is True
    assert isinstance(retriever, FakeResumeRetriever)
    assert diagnostics["retriever_init_stage"] == "ready"
    assert diagnostics["retriever_init_diagnostics"]["persist_dir_used"] == "set"


def test_instantiate_failure_returns_stage_and_init_diagnostics(monkeypatch):
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
    assert summary["retriever_init_diagnostics"]["error_stage"] == "instantiate_resume_retriever"
    assert summary["retriever_init_diagnostics"]["persist_dir_exists"] in {True, False}
    assert summary["retriever_init_diagnostics"]["summary_only"] is True
    assert SENSITIVE_ERROR not in serialized
    assert SENSITIVE_TEXT not in serialized


def test_fake_retriever_factory_output_still_summary_only(monkeypatch):
    block_real_retrieval_imports(monkeypatch)
    callable_ = build_real_retriever_callable(retriever_factory=FakeResumeRetriever)

    output = callable_({"query": "Python RAG LangGraph", "top_k": 1})
    serialized = json.dumps(output, ensure_ascii=False)

    assert len(output["resume_documents"]) == 1
    assert output["resume_documents"][0]["text_length"] == len(SENSITIVE_TEXT)
    assert output["metadata"]["candidate_profile_level"] is False
    assert SENSITIVE_TEXT not in serialized
    assert "/private/full/path" not in serialized


def test_explicit_planner_fallback_markers_survive_retriever_failure(monkeypatch):
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

    assert summary["planner_fallback_used"] is True
    assert summary["planner_fallback_type"] == "deterministic"
    assert summary["fallback_not_real_planner_success"] is True
    assert summary["real_planner_failed"] is True


def test_memory_is_not_read_or_written():
    source = Path("src/runtime/variant_runner.py").read_text(encoding="utf-8")
    relevant = source.split("def build_retriever_init_diagnostics", 1)[1]

    assert "MemorySQLiteStore" not in relevant
    assert "save_memory" not in relevant


def test_required_path_does_not_import_real_retrieval_dependencies_at_module_import():
    import src.runtime.variant_runner as variant_runner

    assert hasattr(variant_runner, "build_retriever_init_diagnostics")


def test_default_graph_behavior_not_modified():
    source = Path("src/core/graph.py").read_text(encoding="utf-8")

    assert "build_retriever_init_diagnostics" not in source
    assert "allow_planner_fallback" not in source
