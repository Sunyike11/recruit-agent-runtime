#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TINY_JD = "招聘一名熟悉 Python、LangGraph、RAG 和向量数据库的 AI Agent 工程师，要求有项目经验。"


def readiness_results_to_summary(results: List[Any]) -> Dict[str, Any]:
    counts = {
        "OK": sum(1 for result in results if result.status == "OK"),
        "FAIL": sum(1 for result in results if result.status == "FAIL"),
        "SKIP": sum(1 for result in results if result.status == "SKIP"),
    }
    missing = [
        {
            "name": result.name,
            "status": result.status,
            "detail": result.detail,
        }
        for result in results
        if result.status != "OK"
    ]
    return {
        "counts": counts,
        "all_ok": counts["FAIL"] == 0 and counts["SKIP"] == 0,
        "missing": missing,
    }


def run_readiness_checks():
    from scripts.smoke_real_readiness import run_checks

    return run_checks()


def run_production_workflow(jd_text: str, max_candidates: int = 3) -> Dict[str, Any]:
    from src.core.graph import create_recruit_graph
    from src.core.state import create_initial_state

    app = create_recruit_graph(interrupt_before=[])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = create_initial_state(jd_text)

    graph_events = list(app.stream(initial_state, config))
    graph_state = app.get_state(config)
    final_state = dict(graph_state.values)
    return summarize_workflow_state(
        jd_text=jd_text,
        final_state=final_state,
        graph_events=graph_events,
        max_candidates=max_candidates,
    )


def summarize_workflow_state(
    jd_text: str,
    final_state: Dict[str, Any],
    graph_events: Optional[List[Any]] = None,
    max_candidates: int = 3,
) -> Dict[str, Any]:
    candidate_pool = list(final_state.get("candidate_pool") or [])
    final_reports = list(final_state.get("final_reports") or [])
    extracted_jd = dict(final_state.get("extracted_jd") or {})
    top_scores = _top_scores(final_reports)
    return {
        "status": "ok",
        "jd_length": len(jd_text),
        "graph_invoked": True,
        "graph_event_count": len(graph_events or []),
        "final_state_keys": sorted(str(key) for key in final_state.keys()),
        "retrieved_count": len(candidate_pool),
        "candidate_count": len(candidate_pool),
        "candidate_summaries": _candidate_summaries(candidate_pool, limit=max_candidates),
        "match_count": len(final_reports),
        "top_scores": top_scores,
        "need_refine": final_state.get("next_action") == "refine",
        "refined_query": extracted_jd.get("search_query", ""),
        "error": "",
    }


def build_skipped_summary(jd_text: str, readiness_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "jd_length": len(jd_text),
        "graph_invoked": False,
        "final_state_keys": [],
        "retrieved_count": 0,
        "candidate_count": 0,
        "match_count": 0,
        "top_scores": [],
        "need_refine": False,
        "refined_query": "",
        "error": "readiness checks are not all OK",
        "readiness": readiness_summary,
    }


def build_failed_summary(jd_text: str, error: Exception) -> Dict[str, Any]:
    return {
        "status": "failed",
        "jd_length": len(jd_text),
        "graph_invoked": True,
        "final_state_keys": [],
        "retrieved_count": 0,
        "candidate_count": 0,
        "match_count": 0,
        "top_scores": [],
        "need_refine": False,
        "refined_query": "",
        "error": str(error),
    }


def run_smoke(
    jd_text: str = DEFAULT_TINY_JD,
    max_candidates: int = 3,
    strict: bool = False,
    no_readiness: bool = False,
    readiness_runner: Callable[[], List[Any]] = run_readiness_checks,
    workflow_runner: Callable[[str, int], Dict[str, Any]] = run_production_workflow,
) -> Dict[str, Any]:
    if not no_readiness:
        readiness_summary = readiness_results_to_summary(readiness_runner())
        if not readiness_summary["all_ok"]:
            summary = build_skipped_summary(jd_text, readiness_summary)
            summary["exit_code"] = 1 if strict else 0
            return summary

    try:
        summary = workflow_runner(jd_text, max_candidates)
        summary.setdefault("status", "ok")
        summary.setdefault("error", "")
        summary["exit_code"] = 0
        return summary
    except Exception as exc:
        summary = build_failed_summary(jd_text, exc)
        summary["exit_code"] = 1 if strict else 0
        return summary


def print_text_summary(summary: Dict[str, Any]):
    status = summary.get("status", "unknown")
    print(f"STATUS: {status}")
    if status == "skipped":
        print(f"SKIP: {summary.get('error', '')}")
        for missing in summary.get("readiness", {}).get("missing", []):
            print(f"- {missing['status']} {missing['name']}: {missing['detail']}")
        return

    if status == "failed":
        print(f"ERROR: {summary.get('error', '')}")
        return

    print(f"JD length: {summary.get('jd_length')}")
    print(f"Graph invoked: {summary.get('graph_invoked')}")
    print(f"Final state keys: {summary.get('final_state_keys')}")
    print(f"Candidate count: {summary.get('candidate_count')}")
    print(f"Match count: {summary.get('match_count')}")
    print(f"Top scores: {summary.get('top_scores')}")
    print(f"Need refine: {summary.get('need_refine')}")
    if summary.get("refined_query"):
        print(f"Refined query: {summary.get('refined_query')}")


def main(
    argv=None,
    readiness_runner: Callable[[], List[Any]] = run_readiness_checks,
    workflow_runner: Callable[[str, int], Dict[str, Any]] = run_production_workflow,
) -> int:
    parser = argparse.ArgumentParser(description="Optional real Recruit Agent workflow smoke test")
    parser.add_argument("--jd", default=DEFAULT_TINY_JD, help="JD text to send to the production graph")
    parser.add_argument("--strict", action="store_true", help="return non-zero on skipped or failed smoke run")
    parser.add_argument("--max-candidates", type=int, default=3, help="max candidate summaries to include")
    parser.add_argument("--json", action="store_true", help="print JSON summary")
    parser.add_argument("--no-readiness", action="store_true", help="skip readiness checks before running the graph")
    args = parser.parse_args(argv)

    if args.json:
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            summary = run_smoke(
                jd_text=args.jd,
                max_candidates=args.max_candidates,
                strict=args.strict,
                no_readiness=args.no_readiness,
                readiness_runner=readiness_runner,
                workflow_runner=workflow_runner,
            )
        captured = captured_stdout.getvalue().strip()
        if captured:
            summary["captured_stdout"] = captured[-1000:]
        captured_err = captured_stderr.getvalue().strip()
        if captured_err:
            summary["captured_stderr"] = captured_err[-1000:]
        print(json.dumps(_json_safe_summary(summary), ensure_ascii=False, indent=2, default=str))
    else:
        summary = run_smoke(
            jd_text=args.jd,
            max_candidates=args.max_candidates,
            strict=args.strict,
            no_readiness=args.no_readiness,
            readiness_runner=readiness_runner,
            workflow_runner=workflow_runner,
        )
        print_text_summary(summary)

    return int(summary.get("exit_code", 0))


def _candidate_summaries(candidate_pool: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    summaries = []
    for candidate in candidate_pool[: max(limit, 0)]:
        metadata = dict(candidate.get("metadata") or {})
        text = str(candidate.get("text") or "")
        summaries.append(
            {
                "metadata_keys": sorted(str(key) for key in metadata.keys()),
                "source": metadata.get("source") or metadata.get("file_name") or metadata.get("candidate_id") or "",
                "text_preview": _truncate(text.replace("\n", " "), 160),
            }
        )
    return summaries


def _top_scores(final_reports: List[Dict[str, Any]]) -> List[float]:
    scores = []
    for report in final_reports:
        if "total_score" not in report:
            continue
        try:
            scores.append(float(report["total_score"]))
        except (TypeError, ValueError):
            continue
    return sorted(scores, reverse=True)[:3]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _json_safe_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    safe = dict(summary)
    safe.pop("exit_code", None)
    return safe


if __name__ == "__main__":
    raise SystemExit(main())
