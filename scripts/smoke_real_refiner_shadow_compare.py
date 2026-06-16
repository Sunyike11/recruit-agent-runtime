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


DEFAULT_QUERY = "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师"
DEFAULT_CONTEXT = "broaden the search query while retaining required skills"


def run_refiner_readiness_checks() -> List[Any]:
    from scripts.smoke_real_readiness import (
        check_agent_instantiation,
        check_config_loads,
        check_env_var,
        check_import,
        check_python_import_path,
        load_project_dotenv,
    )

    dotenv_result = load_project_dotenv()
    return [
        check_python_import_path(),
        dotenv_result,
        check_config_loads(),
        check_import("refiner_agent_import", "src.agents.refiner", "RefinerAgent"),
        check_env_var("openai_api_key", "OPENAI_API_KEY", required=True),
        check_agent_instantiation("refiner_agent_init", "src.agents.refiner", "RefinerAgent"),
    ]


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


def run_real_refiner(query: str, context: str) -> Dict[str, Any]:
    from src.agents.refiner import RefinerAgent

    result = RefinerAgent()(
        {
            "extracted_jd": {"search_query": query},
            "refinement_advice": context,
            "human_feedback": "",
        }
    )
    return {"extracted_jd": dict(result.get("extracted_jd") or {})}


def run_deterministic_shadow_refiner(query: str, context: str) -> Dict[str, Any]:
    from src.skills.agent_adapters import QueryRefineSkill

    skill = QueryRefineSkill(
        refine_callable=lambda input_data, skill_context: {
            "refined_query": input_data["query"],
            "reason": "deterministic optional smoke baseline",
        }
    )
    result = skill.execute({"query": query, "context": context})
    if not result.success:
        raise RuntimeError("Deterministic shadow refiner failed.")
    return dict(result.output)


def build_skipped_summary(query: str, context: str, readiness: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "query_length": len(query),
        "context_length": len(context),
        "real_refiner_invoked": False,
        "shadow_invoked": False,
        "decision_status": "skipped",
        "risk_level": "medium",
        "refined_query_length": 0,
        "refined_query_preview": "",
        "error_type": "",
        "readiness": readiness,
        "summary_only": True,
        "production_graph_invoked": False,
        "exit_code": 1 if strict else 0,
    }


def run_smoke(
    query: str = DEFAULT_QUERY,
    context: str = DEFAULT_CONTEXT,
    strict: bool = False,
    readiness_runner: Callable[[], List[Any]] = run_refiner_readiness_checks,
    real_refiner_runner: Callable[[str, str], Dict[str, Any]] = run_real_refiner,
    shadow_runner: Callable[[str, str], Dict[str, Any]] = run_deterministic_shadow_refiner,
) -> Dict[str, Any]:
    readiness = readiness_results_to_summary(readiness_runner())
    if not readiness["all_ok"]:
        return build_skipped_summary(query, context, readiness, strict)

    from src.integration.node_shadow import SingleNodeShadowCompareCase, SingleNodeShadowCompareHarness

    case = SingleNodeShadowCompareCase(
        case_id="optional_real_refiner_shadow_smoke",
        node_name="query_refine",
        node_type="refiner",
        input_data={"query": query, "context": context},
        production_callable=lambda input_data: real_refiner_runner(
            input_data["query"], input_data["context"]
        ),
        shadow_callable=lambda input_data: shadow_runner(
            input_data["query"], input_data["context"]
        ),
        metadata={"mode": "optional_real_refiner_shadow_smoke"},
    )

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        result = SingleNodeShadowCompareHarness().run_case(case)

    error_type = (
        result.metadata.get("production_error_type")
        or result.metadata.get("shadow_error_type")
        or ""
    )
    refined_length = int(result.production_output_summary.get("refined_query_length", 0))
    failed = result.decision.status == "skipped" and bool(error_type)
    return {
        "status": "failed" if failed else "ok",
        "query_length": len(query),
        "context_length": len(context),
        "real_refiner_invoked": True,
        "shadow_invoked": True,
        "decision_status": result.decision.status,
        "risk_level": result.decision.risk_level,
        "refined_query_length": refined_length,
        "refined_query_preview": "<present; redacted>" if refined_length else "",
        "error_type": error_type,
        "summary_only": True,
        "production_graph_invoked": False,
        "exit_code": 1 if strict and failed else 0,
    }


def print_text_summary(summary: Dict[str, Any]) -> None:
    print(f"STATUS: {summary['status']}")
    print(f"Query length: {summary['query_length']}")
    print(f"Real refiner invoked: {summary['real_refiner_invoked']}")
    print(f"Shadow invoked: {summary['shadow_invoked']}")
    print(f"Decision: {summary['decision_status']}")
    print(f"Risk: {summary['risk_level']}")
    print(f"Refined query length: {summary['refined_query_length']}")
    if summary["refined_query_preview"]:
        print(f"Refined query preview: {summary['refined_query_preview']}")
    if summary["error_type"]:
        print(f"Error type: {summary['error_type']}")
    if summary["status"] == "skipped":
        for missing in summary.get("readiness", {}).get("missing", []):
            print(f"- {missing['status']} {missing['name']}")


def main(
    argv=None,
    readiness_runner: Callable[[], List[Any]] = run_refiner_readiness_checks,
    real_refiner_runner: Callable[[str, str], Dict[str, Any]] = run_real_refiner,
    shadow_runner: Callable[[str, str], Dict[str, Any]] = run_deterministic_shadow_refiner,
) -> int:
    parser = argparse.ArgumentParser(description="Optional real Refiner node shadow-compare smoke")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="query text for the optional Refiner smoke")
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="optional refinement advice/context text")
    parser.add_argument("--json", action="store_true", help="emit summary-only JSON")
    parser.add_argument("--strict", action="store_true", help="return non-zero when skipped or failed")
    args = parser.parse_args(argv)

    summary = run_smoke(
        query=args.query,
        context=args.context,
        strict=args.strict,
        readiness_runner=readiness_runner,
        real_refiner_runner=real_refiner_runner,
        shadow_runner=shadow_runner,
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
