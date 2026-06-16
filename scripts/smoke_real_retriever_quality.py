#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_QUERY = "Python RAG LangGraph AI Agent"


def run_retriever_readiness_checks() -> List[Any]:
    from scripts.smoke_real_readiness import (
        CheckResult,
        OK,
        SKIP,
        check_config_loads,
        check_import,
        check_path_exists,
        check_python_import_path,
        check_resume_retriever_instantiation,
        get_settings_path,
        load_project_dotenv,
    )

    checks = [
        check_python_import_path(),
        load_project_dotenv(),
        check_config_loads(),
        check_import("retriever_agent_import", "src.agents.retriever", "RetrieverAgent"),
        check_import("resume_retriever_import", "src.services.retriever", "ResumeRetriever"),
        check_import("llama_index_core_import", "llama_index.core"),
        check_import("llama_index_huggingface_embedding_import", "llama_index.embeddings.huggingface", "HuggingFaceEmbedding"),
        check_import("llama_index_chroma_vector_store_import", "llama_index.vector_stores.chroma", "ChromaVectorStore"),
        check_import("chroma_dependency_import", "chromadb"),
        check_path_exists("data_dir_exists", lambda: get_settings_path("data_dir")),
        check_path_exists("chroma_db_exists", lambda: get_settings_path("chroma_dir"), require_non_empty=True),
        _check_chroma_index_count(),
        check_resume_retriever_instantiation(),
    ]
    # Treat an uninspectable but present index as SKIP so default mode remains diagnostic.
    return [
        CheckResult(check.name, check.status if check.status in {OK, SKIP, "FAIL"} else check.status, check.detail)
        for check in checks
    ]


def _check_chroma_index_count():
    from scripts.smoke_real_readiness import CheckResult, FAIL, OK, SKIP

    try:
        from src.config import get_settings
    except Exception as exc:
        return CheckResult("chroma_index_non_empty", FAIL, f"config unavailable: {type(exc).__name__}")

    try:
        import chromadb

        path = get_settings().chroma_dir
        if not path.exists() or not any(path.iterdir()):
            return CheckResult("chroma_index_non_empty", SKIP, "chroma path missing or empty")
        collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection("resumes")
        count = collection.count()
        if count > 0:
            return CheckResult("chroma_index_non_empty", OK, f"record_count={count}")
        return CheckResult("chroma_index_non_empty", SKIP, "record_count=0")
    except Exception as exc:
        return CheckResult("chroma_index_non_empty", SKIP, f"count unavailable: {type(exc).__name__}")


def readiness_results_to_summary(results: List[Any]) -> Dict[str, Any]:
    counts = {
        "OK": sum(1 for result in results if result.status == "OK"),
        "FAIL": sum(1 for result in results if result.status == "FAIL"),
        "SKIP": sum(1 for result in results if result.status == "SKIP"),
    }
    return {
        "counts": counts,
        "all_ok": counts["FAIL"] == 0 and counts["SKIP"] == 0,
        "missing": [
            {"name": result.name, "status": result.status}
            for result in results
            if result.status != "OK"
        ],
    }


def inspect_index_record_count() -> Optional[int]:
    try:
        from src.config import get_settings
        import chromadb

        path = get_settings().chroma_dir
        collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection("resumes")
        return int(collection.count())
    except Exception:
        return None


def run_resume_retriever_search(query: str, top_k: int) -> Dict[str, Any]:
    from src.config import get_settings
    from src.services.retriever import ResumeRetriever

    retriever = ResumeRetriever(persist_dir=str(get_settings().chroma_dir))
    results = retriever.search(query, k=top_k)
    return {
        "results": results,
        "index_record_count": inspect_index_record_count(),
    }


def build_skipped_summary(
    query: str,
    top_k: int,
    readiness: Dict[str, Any],
    strict: bool,
) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "query_length": len(query),
        "top_k": top_k,
        "retriever_initialized": False,
        "retrieval_invoked": False,
        "result_count": 0,
        "result_summary": [],
        "source_keys": [],
        "candidate_ids": [],
        "document_ids": [],
        "score_present": False,
        "expected_candidate_found": None,
        "expected_source_found": None,
        "index_record_count": None,
        "error_type": "",
        "readiness": readiness,
        "summary_only": True,
        "production_graph_invoked": False,
        "exit_code": 1 if strict else 0,
    }


def run_smoke(
    query: str = DEFAULT_QUERY,
    top_k: int = 3,
    strict: bool = False,
    expected_candidate_id: Optional[str] = None,
    expected_source_contains: Optional[str] = None,
    readiness_runner: Callable[[], List[Any]] = run_retriever_readiness_checks,
    retrieval_runner: Callable[[str, int], Dict[str, Any]] = run_resume_retriever_search,
) -> Dict[str, Any]:
    readiness = readiness_results_to_summary(readiness_runner())
    if not readiness["all_ok"]:
        return build_skipped_summary(query, top_k, readiness, strict)

    try:
        from src.integration.retriever_quality import summarize_retrieval_results

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            raw = retrieval_runner(query, top_k)
        results = list(raw.get("results") or [])
        observation = summarize_retrieval_results(
            results,
            query=query,
            top_k=top_k,
            expected_candidate_id=expected_candidate_id,
            expected_source_contains=expected_source_contains,
            index_record_count=raw.get("index_record_count"),
        )
        summary = observation.to_dict()
        summary.update(
            {
                "status": "ok",
                "retriever_initialized": True,
                "retrieval_invoked": True,
                "error_type": "",
                "summary_only": True,
                "production_graph_invoked": False,
                "exit_code": 0,
            }
        )
        return summary
    except Exception as exc:
        return {
            "status": "failed",
            "query_length": len(query),
            "top_k": top_k,
            "retriever_initialized": True,
            "retrieval_invoked": True,
            "result_count": 0,
            "result_summary": [],
            "source_keys": [],
            "candidate_ids": [],
            "document_ids": [],
            "score_present": False,
            "expected_candidate_found": None if expected_candidate_id is None else False,
            "expected_source_found": None if expected_source_contains is None else False,
            "index_record_count": None,
            "error_type": type(exc).__name__,
            "summary_only": True,
            "production_graph_invoked": False,
            "exit_code": 1 if strict else 0,
        }


def print_text_summary(summary: Dict[str, Any]) -> None:
    print(f"STATUS: {summary['status']}")
    print(f"Query length: {summary['query_length']}")
    print(f"Top K: {summary['top_k']}")
    print(f"Retriever initialized: {summary['retriever_initialized']}")
    print(f"Retrieval invoked: {summary['retrieval_invoked']}")
    print(f"Result count: {summary['result_count']}")
    print(f"Source keys: {', '.join(summary['source_keys'])}")
    print(f"Candidate IDs: {', '.join(summary['candidate_ids'])}")
    print(f"Document IDs: {', '.join(summary['document_ids'])}")
    print(f"Score present: {summary['score_present']}")
    if summary.get("expected_candidate_found") is not None:
        print(f"Expected candidate found: {summary['expected_candidate_found']}")
    if summary.get("expected_source_found") is not None:
        print(f"Expected source found: {summary['expected_source_found']}")
    if summary.get("index_record_count") is not None:
        print(f"Index record count: {summary['index_record_count']}")
    if summary["error_type"]:
        print(f"Error type: {summary['error_type']}")
    if summary["status"] == "skipped":
        for missing in summary.get("readiness", {}).get("missing", []):
            print(f"- {missing['status']} {missing['name']}")


def main(
    argv=None,
    readiness_runner: Callable[[], List[Any]] = run_retriever_readiness_checks,
    retrieval_runner: Callable[[str, int], Dict[str, Any]] = run_resume_retriever_search,
) -> int:
    parser = argparse.ArgumentParser(description="Optional real Retriever readiness and quality smoke")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="retrieval query for the optional smoke")
    parser.add_argument("--top-k", type=int, default=3, help="retrieval top_k")
    parser.add_argument("--json", action="store_true", help="emit summary-only JSON")
    parser.add_argument("--strict", action="store_true", help="return non-zero when skipped or failed")
    parser.add_argument("--expect-candidate-id", default=None, help="optional expected candidate id signal")
    parser.add_argument("--expect-source-contains", default=None, help="optional expected metadata source substring")
    args = parser.parse_args(argv)

    summary = run_smoke(
        query=args.query,
        top_k=args.top_k,
        strict=args.strict,
        expected_candidate_id=args.expect_candidate_id,
        expected_source_contains=args.expect_source_contains,
        readiness_runner=readiness_runner,
        retrieval_runner=retrieval_runner,
    )
    if args.json:
        safe = dict(summary)
        safe.pop("exit_code", None)
        print(json.dumps(safe, ensure_ascii=False, indent=2, default=str))
    else:
        print_text_summary(summary)
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
