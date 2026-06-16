import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_real_retriever_quality.py"
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "retriever_quality_cases.json"
SENSITIVE_TEXT = "FULL-RESUME-CHUNK-SECRET-MUST-NOT-ENTER-RETRIEVER-SMOKE"


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_real_retriever_quality", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeReadinessResult:
    def __init__(self, name, status):
        self.name = name
        self.status = status


def readiness_ok():
    return [
        FakeReadinessResult("resume_retriever_import", "OK"),
        FakeReadinessResult("chroma_db_exists", "OK"),
        FakeReadinessResult("resume_retriever_init", "OK"),
    ]


def readiness_missing():
    return [
        FakeReadinessResult("resume_retriever_import", "OK"),
        FakeReadinessResult("chroma_db_exists", "SKIP"),
    ]


def fake_retrieval_runner(query, top_k):
    return {
        "index_record_count": 7,
        "results": [
            {
                "text": "Alice summary intentionally short",
                "metadata": {
                    "candidate_id": "candidate_1",
                    "file_name": "alice_resume.pdf",
                    "source": "safe_source/alice_resume.pdf",
                    "role": "agent engineer",
                },
                "score": 0.82,
            },
            {
                "text": "Bob summary intentionally short",
                "metadata": {
                    "candidate_id": "candidate_2",
                    "file_name": "bob_resume.pdf",
                    "source": "safe_source/bob_resume.pdf",
                },
                "score": None,
            },
        ][:top_k],
    }


def test_optional_real_retriever_quality_script_help_executes():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert SCRIPT_PATH.exists()
    assert result.returncode == 0
    assert "--query" in result.stdout
    assert "--top-k" in result.stdout
    assert "--expect-candidate-id" in result.stdout


def test_missing_readiness_default_mode_skips_without_invoking_retrieval():
    module = load_module()
    invoked = []

    summary = module.run_smoke(
        query="Python RAG",
        readiness_runner=readiness_missing,
        retrieval_runner=lambda query, top_k: invoked.append("retrieval"),
    )

    assert summary["status"] == "skipped"
    assert summary["exit_code"] == 0
    assert summary["retriever_initialized"] is False
    assert summary["retrieval_invoked"] is False
    assert summary["readiness"]["missing"] == [{"name": "chroma_db_exists", "status": "SKIP"}]
    assert invoked == []


def test_missing_readiness_strict_mode_returns_nonzero():
    module = load_module()

    summary = module.run_smoke(
        strict=True,
        readiness_runner=readiness_missing,
        retrieval_runner=fake_retrieval_runner,
    )

    assert summary["status"] == "skipped"
    assert summary["exit_code"] != 0


def test_fake_retriever_runner_generates_summary_only_ok_result():
    module = load_module()

    summary = module.run_smoke(
        query="Python RAG LangGraph",
        top_k=2,
        readiness_runner=readiness_ok,
        retrieval_runner=fake_retrieval_runner,
    )

    assert summary["status"] == "ok"
    assert summary["retriever_initialized"] is True
    assert summary["retrieval_invoked"] is True
    assert summary["result_count"] == 2
    assert summary["production_graph_invoked"] is False
    assert summary["summary_only"] is True
    assert summary["score_present"] is True
    assert "candidate_1" in summary["candidate_ids"]
    assert "file_name" in summary["source_keys"]


def test_expected_candidate_id_found_and_missing_do_not_crash():
    module = load_module()

    found = module.run_smoke(
        readiness_runner=readiness_ok,
        retrieval_runner=fake_retrieval_runner,
        expected_candidate_id="candidate_1",
    )
    missing = module.run_smoke(
        readiness_runner=readiness_ok,
        retrieval_runner=fake_retrieval_runner,
        expected_candidate_id="candidate_missing",
    )

    assert found["expected_candidate_found"] is True
    assert missing["expected_candidate_found"] is False
    assert missing["status"] == "ok"


def test_expected_source_contains_is_checked_without_exposing_metadata_values():
    module = load_module()

    summary = module.run_smoke(
        readiness_runner=readiness_ok,
        retrieval_runner=fake_retrieval_runner,
        expected_source_contains="alice_resume",
    )
    serialized = json.dumps(summary)

    assert summary["expected_source_found"] is True
    assert "safe_source/alice_resume.pdf" not in serialized


def test_failed_retrieval_path_returns_sanitized_error_type_only():
    module = load_module()

    def fail(query, top_k):
        print(SENSITIVE_TEXT)
        raise RuntimeError(f"retrieval failed with {SENSITIVE_TEXT}")

    summary = module.run_smoke(
        query=SENSITIVE_TEXT,
        readiness_runner=readiness_ok,
        retrieval_runner=fail,
    )
    serialized = json.dumps(summary)

    assert summary["status"] == "failed"
    assert summary["error_type"] == "RuntimeError"
    assert SENSITIVE_TEXT not in serialized


def test_json_output_is_parseable_with_fake_injected_runner(capsys):
    module = load_module()

    exit_code = module.main(
        ["--json", "--query", "Python RAG", "--top-k", "1", "--expect-candidate-id", "candidate_1"],
        readiness_runner=readiness_ok,
        retrieval_runner=fake_retrieval_runner,
    )
    parsed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert parsed["status"] == "ok"
    assert parsed["result_count"] == 1
    assert parsed["expected_candidate_found"] is True
    assert "exit_code" not in parsed


def test_fixture_can_load_summary_only_retriever_quality_case():
    from src.integration.retriever_quality import RetrieverQualityCase

    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    case = RetrieverQualityCase.from_dict(cases[0])

    assert case.case_id == "python_rag_agent"
    assert case.query == "Python RAG LangGraph AI Agent"
    assert case.top_k == 3
    assert "smoke" in case.tags


def test_structural_path_does_not_import_real_retriever_or_external_dependencies(monkeypatch):
    module = load_module()
    real_import = builtins.__import__
    blocked = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("src.agents.retriever", "src.services.retriever", "llama_index", "chromadb")):
            blocked.append(name)
            raise ModuleNotFoundError(f"blocked dependency in Phase7A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    summary = module.run_smoke(
        readiness_runner=readiness_ok,
        retrieval_runner=fake_retrieval_runner,
    )

    assert summary["status"] == "ok"
    assert blocked == []


def test_optional_retriever_smoke_does_not_modify_or_invoke_production_graph():
    graph_source = (PROJECT_ROOT / "src" / "core" / "graph.py").read_text(encoding="utf-8")
    script_source = SCRIPT_PATH.read_text(encoding="utf-8")
    module = load_module()

    summary = module.run_smoke(
        readiness_runner=readiness_ok,
        retrieval_runner=fake_retrieval_runner,
    )

    assert summary["production_graph_invoked"] is False
    assert "smoke_real_retriever_quality" not in graph_source
    assert "create_recruit_graph" not in script_source
    assert "src.core.graph" not in script_source
    assert "PlannerAgent" not in script_source
    assert "MatcherAgent" not in script_source
    assert "RefinerAgent" not in script_source
