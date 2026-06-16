#!/usr/bin/env python
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_JD = "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师"
REAL_REQUIRED_KEYS = ("tech_stack", "education", "must_have", "search_query")


def run_planner_readiness_checks() -> List[Any]:
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
        check_import("planner_agent_import", "src.agents.planner", "PlannerAgent"),
        check_env_var("openai_api_key", "OPENAI_API_KEY", required=True),
        check_agent_instantiation("planner_agent_init", "src.agents.planner", "PlannerAgent"),
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


def run_real_planner(jd_text: str) -> Dict[str, Any]:
    from langchain_core.messages import HumanMessage
    from src.agents.planner import PlannerAgent

    result = PlannerAgent()({"messages": [HumanMessage(content=jd_text)]})
    return {"extracted_jd": dict(result.get("extracted_jd") or {})}


def run_deterministic_shadow_planner(jd_text: str) -> Dict[str, Any]:
    from src.skills.agent_adapters import PlannerExtractSkill

    skill = PlannerExtractSkill(
        extract_callable=lambda input_data, skill_context: {
            "job_requirement": {
                "job_id": "optional_planner_smoke_job",
                "raw_text": input_data["raw_text"],
                "required_skills": ["Python", "RAG", "LangGraph"],
            },
            "extracted_keywords": ["Python", "RAG", "LangGraph"],
        }
    )
    result = skill.execute({"raw_text": jd_text})
    if not result.success:
        raise RuntimeError("Deterministic shadow planner failed.")
    return dict(result.output)


def real_planner_projection(output: Mapping[str, Any]) -> Dict[str, bool]:
    extracted = output.get("extracted_jd")
    projected: Dict[str, bool] = {}
    if isinstance(extracted, Mapping):
        projected["structured_requirement"] = True
        if all(key in extracted for key in REAL_REQUIRED_KEYS):
            projected["required_fields"] = True
        keywords = extracted.get("tech_stack")
        if isinstance(keywords, list) and bool(keywords):
            projected["keyword_signal"] = True
    return projected


def shadow_planner_projection(output: Mapping[str, Any]) -> Dict[str, bool]:
    job_requirement = output.get("job_requirement")
    projected: Dict[str, bool] = {}
    if isinstance(job_requirement, Mapping):
        projected["structured_requirement"] = True
        if "required_skills" in job_requirement and "extracted_keywords" in output:
            projected["required_fields"] = True
        keywords = output.get("extracted_keywords")
        if isinstance(keywords, list) and bool(keywords):
            projected["keyword_signal"] = True
    return projected


def extracted_summary(output: Mapping[str, Any]) -> Dict[str, Any]:
    extracted = output.get("extracted_jd")
    if not isinstance(extracted, Mapping):
        return {"extracted_keys": [], "keyword_count": 0}
    keywords = extracted.get("tech_stack")
    return {
        "extracted_keys": sorted(str(key) for key in extracted.keys()),
        "keyword_count": len(keywords) if isinstance(keywords, list) else 0,
    }


def build_skipped_summary(jd_text: str, readiness: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "jd_length": len(jd_text),
        "real_planner_invoked": False,
        "shadow_invoked": False,
        "decision_status": "skipped",
        "risk_level": "medium",
        "extracted_keys": [],
        "keyword_count": 0,
        "error_type": "",
        "readiness": readiness,
        "summary_only": True,
        "production_graph_invoked": False,
        "exit_code": 1 if strict else 0,
    }


def run_smoke(
    jd_text: str = DEFAULT_JD,
    strict: bool = False,
    readiness_runner: Callable[[], List[Any]] = run_planner_readiness_checks,
    real_planner_runner: Callable[[str], Dict[str, Any]] = run_real_planner,
    shadow_runner: Callable[[str], Dict[str, Any]] = run_deterministic_shadow_planner,
) -> Dict[str, Any]:
    readiness = readiness_results_to_summary(readiness_runner())
    if not readiness["all_ok"]:
        return build_skipped_summary(jd_text, readiness, strict)

    from src.integration.node_shadow import SingleNodeShadowCompareCase, SingleNodeShadowCompareHarness

    real_summary: Dict[str, Any] = {"extracted_keys": [], "keyword_count": 0}

    def invoke_real(input_data: Dict[str, Any]) -> Dict[str, bool]:
        raw_output = real_planner_runner(input_data["jd_text"])
        real_summary.update(extracted_summary(raw_output))
        return real_planner_projection(raw_output)

    def invoke_shadow(input_data: Dict[str, Any]) -> Dict[str, bool]:
        return shadow_planner_projection(shadow_runner(input_data["jd_text"]))

    case = SingleNodeShadowCompareCase(
        case_id="optional_real_planner_shadow_smoke",
        node_name="planner_extract",
        node_type="planner",
        input_data={"jd_text": jd_text},
        production_callable=invoke_real,
        shadow_callable=invoke_shadow,
        metadata={"mode": "optional_real_planner_shadow_smoke"},
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
    failed = result.decision.status == "skipped" and bool(error_type)
    return {
        "status": "failed" if failed else "ok",
        "jd_length": len(jd_text),
        "real_planner_invoked": True,
        "shadow_invoked": True,
        "decision_status": result.decision.status,
        "risk_level": result.decision.risk_level,
        "extracted_keys": list(real_summary["extracted_keys"]),
        "keyword_count": int(real_summary["keyword_count"]),
        "error_type": error_type,
        "summary_only": True,
        "production_graph_invoked": False,
        "exit_code": 1 if strict and failed else 0,
    }


def print_text_summary(summary: Dict[str, Any]) -> None:
    print(f"STATUS: {summary['status']}")
    print(f"JD length: {summary['jd_length']}")
    print(f"Real planner invoked: {summary['real_planner_invoked']}")
    print(f"Shadow invoked: {summary['shadow_invoked']}")
    print(f"Decision: {summary['decision_status']}")
    print(f"Risk: {summary['risk_level']}")
    print(f"Extracted keys: {', '.join(summary['extracted_keys'])}")
    print(f"Keyword count: {summary['keyword_count']}")
    if summary["error_type"]:
        print(f"Error type: {summary['error_type']}")
    if summary["status"] == "skipped":
        for missing in summary.get("readiness", {}).get("missing", []):
            print(f"- {missing['status']} {missing['name']}")


def main(
    argv=None,
    readiness_runner: Callable[[], List[Any]] = run_planner_readiness_checks,
    real_planner_runner: Callable[[str], Dict[str, Any]] = run_real_planner,
    shadow_runner: Callable[[str], Dict[str, Any]] = run_deterministic_shadow_planner,
) -> int:
    parser = argparse.ArgumentParser(description="Optional real Planner node shadow-compare smoke")
    parser.add_argument("--jd", default=DEFAULT_JD, help="JD text for the optional Planner smoke")
    parser.add_argument("--json", action="store_true", help="emit summary-only JSON")
    parser.add_argument("--strict", action="store_true", help="return non-zero when skipped or failed")
    args = parser.parse_args(argv)

    summary = run_smoke(
        jd_text=args.jd,
        strict=args.strict,
        readiness_runner=readiness_runner,
        real_planner_runner=real_planner_runner,
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
