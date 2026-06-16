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


DEFAULT_JD = "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师"
DEFAULT_CANDIDATE_PROFILE = {
    "candidate_id": "candidate_smoke_1",
    "name": "Optional Smoke Candidate",
    "skills": ["Python", "RAG", "LangGraph"],
    "education": "Bachelor",
    "experience": ["Built deterministic AI Agent workflow examples"],
    "projects": ["RAG-based recruitment matching prototype"],
}


def run_matcher_readiness_checks() -> List[Any]:
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
        check_import("matcher_agent_import", "src.agents.matcher", "MatcherAgent"),
        check_env_var("openai_api_key", "OPENAI_API_KEY", required=True),
        check_agent_instantiation("matcher_agent_init", "src.agents.matcher", "MatcherAgent"),
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


def load_candidate_profile(candidate_json: Optional[str]) -> Dict[str, Any]:
    if not candidate_json:
        return dict(DEFAULT_CANDIDATE_PROFILE)
    candidate_path = Path(candidate_json)
    data = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("candidate JSON must contain an object")
    return dict(data)


def build_job_requirement(jd_text: str) -> Dict[str, Any]:
    return {
        "job_id": "optional_matcher_smoke_job",
        "title": "AI Agent Engineer",
        "raw_text": jd_text,
        "required_skills": ["Python", "RAG", "LangGraph"],
    }


def candidate_id(candidate_profile: Dict[str, Any]) -> str:
    return str(candidate_profile.get("candidate_id") or "candidate_smoke_1")


def candidate_profile_to_text(candidate_profile: Dict[str, Any]) -> str:
    skills = ", ".join(str(item) for item in candidate_profile.get("skills", []))
    experience = "; ".join(str(item) for item in candidate_profile.get("experience", []))
    projects = "; ".join(str(item) for item in candidate_profile.get("projects", []))
    parts = [
        str(candidate_profile.get("name", "")),
        f"Skills: {skills}" if skills else "",
        f"Education: {candidate_profile.get('education', '')}" if candidate_profile.get("education") else "",
        f"Experience: {experience}" if experience else "",
        f"Projects: {projects}" if projects else "",
    ]
    return "\n".join(part for part in parts if part)


def run_real_matcher(job_requirement: Dict[str, Any], candidate_profile: Dict[str, Any]) -> Dict[str, Any]:
    from src.agents.matcher import MatcherAgent

    known_candidate_id = candidate_id(candidate_profile)
    result = MatcherAgent()(
        {
            "extracted_jd": job_requirement,
            "candidate_pool": [
                {
                    "text": candidate_profile_to_text(candidate_profile),
                    "metadata": {"candidate_id": known_candidate_id},
                }
            ],
            "loop_count": 0,
        }
    )
    reports = []
    for raw_report in result.get("final_reports", []):
        report = dict(raw_report)
        report.setdefault("candidate_id", known_candidate_id)
        reports.append(report)
    return {"final_reports": reports}


def run_deterministic_shadow_matcher(
    job_requirement: Dict[str, Any],
    candidate_profile: Dict[str, Any],
) -> Dict[str, Any]:
    from src.skills.agent_adapters import CandidateMatchSkill

    skill = CandidateMatchSkill(
        match_callable=lambda input_data, skill_context: {
            "total_score": 85,
            "recommendation": "strong_match",
        }
    )
    result = skill.execute(
        {
            "job_requirement": job_requirement,
            "candidate_profile": candidate_profile,
        }
    )
    if not result.success:
        raise RuntimeError("Deterministic shadow matcher failed.")
    return dict(result.output)


def build_skipped_summary(
    jd_text: str,
    candidate_profile: Dict[str, Any],
    readiness: Dict[str, Any],
    strict: bool,
    compare_exact_scores: bool,
) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "jd_length": len(jd_text),
        "candidate_id": candidate_id(candidate_profile),
        "real_matcher_invoked": False,
        "shadow_invoked": False,
        "decision_status": "skipped",
        "risk_level": "medium",
        "score_present": False,
        "report_keys": [],
        "compare_exact_scores": compare_exact_scores,
        "error_type": "",
        "readiness": readiness,
        "summary_only": True,
        "production_graph_invoked": False,
        "exit_code": 1 if strict else 0,
    }


def build_input_failure_summary(jd_text: str, strict: bool, error_type: str) -> Dict[str, Any]:
    return {
        "status": "failed",
        "jd_length": len(jd_text),
        "candidate_id": "",
        "real_matcher_invoked": False,
        "shadow_invoked": False,
        "decision_status": "skipped",
        "risk_level": "medium",
        "score_present": False,
        "report_keys": [],
        "compare_exact_scores": False,
        "error_type": error_type,
        "summary_only": True,
        "production_graph_invoked": False,
        "exit_code": 1 if strict else 0,
    }


def run_smoke(
    jd_text: str = DEFAULT_JD,
    candidate_profile: Optional[Dict[str, Any]] = None,
    compare_exact_scores: bool = False,
    strict: bool = False,
    readiness_runner: Callable[[], List[Any]] = run_matcher_readiness_checks,
    real_matcher_runner: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]] = run_real_matcher,
    shadow_runner: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]] = run_deterministic_shadow_matcher,
) -> Dict[str, Any]:
    candidate_profile = dict(candidate_profile or DEFAULT_CANDIDATE_PROFILE)
    readiness = readiness_results_to_summary(readiness_runner())
    if not readiness["all_ok"]:
        return build_skipped_summary(
            jd_text,
            candidate_profile,
            readiness,
            strict,
            compare_exact_scores,
        )

    from src.integration.node_shadow import SingleNodeShadowCompareCase, SingleNodeShadowCompareHarness

    job_requirement = build_job_requirement(jd_text)
    observed_report_keys: List[str] = []

    def invoke_real(input_data: Dict[str, Any]) -> Dict[str, Any]:
        output = real_matcher_runner(
            input_data["job_requirement"], input_data["candidate_profile"]
        )
        reports = output.get("final_reports", [])
        if reports and isinstance(reports[0], dict):
            observed_report_keys.extend(sorted(str(key) for key in reports[0].keys()))
        return output

    case = SingleNodeShadowCompareCase(
        case_id="optional_real_matcher_shadow_smoke",
        node_name="candidate_match",
        node_type="matcher",
        input_data={
            "job_requirement": job_requirement,
            "candidate_profile": candidate_profile,
        },
        production_callable=invoke_real,
        shadow_callable=lambda input_data: shadow_runner(
            input_data["job_requirement"], input_data["candidate_profile"]
        ),
        expected_alignment={"compare_exact_scores": compare_exact_scores},
        metadata={"mode": "optional_real_matcher_shadow_smoke"},
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
        "candidate_id": candidate_id(candidate_profile),
        "real_matcher_invoked": True,
        "shadow_invoked": True,
        "decision_status": result.decision.status,
        "risk_level": result.decision.risk_level,
        "score_present": bool(result.production_output_summary.get("score_present")),
        "report_keys": sorted(set(observed_report_keys)),
        "compare_exact_scores": compare_exact_scores,
        "error_type": error_type,
        "summary_only": True,
        "production_graph_invoked": False,
        "exit_code": 1 if strict and failed else 0,
    }


def print_text_summary(summary: Dict[str, Any]) -> None:
    print(f"STATUS: {summary['status']}")
    print(f"JD length: {summary['jd_length']}")
    print(f"Candidate ID: {summary['candidate_id']}")
    print(f"Real matcher invoked: {summary['real_matcher_invoked']}")
    print(f"Shadow invoked: {summary['shadow_invoked']}")
    print(f"Decision: {summary['decision_status']}")
    print(f"Risk: {summary['risk_level']}")
    print(f"Score present: {summary['score_present']}")
    print(f"Report keys: {', '.join(summary['report_keys'])}")
    if summary["error_type"]:
        print(f"Error type: {summary['error_type']}")
    if summary["status"] == "skipped":
        for missing in summary.get("readiness", {}).get("missing", []):
            print(f"- {missing['status']} {missing['name']}")


def main(
    argv=None,
    readiness_runner: Callable[[], List[Any]] = run_matcher_readiness_checks,
    real_matcher_runner: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]] = run_real_matcher,
    shadow_runner: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]] = run_deterministic_shadow_matcher,
) -> int:
    parser = argparse.ArgumentParser(description="Optional real Matcher node shadow-compare smoke")
    parser.add_argument("--jd", default=DEFAULT_JD, help="JD text for the optional Matcher smoke")
    parser.add_argument("--candidate-json", help="path to a local fake candidate profile JSON object")
    parser.add_argument("--compare-exact-scores", action="store_true", help="compare numeric matcher scores exactly")
    parser.add_argument("--json", action="store_true", help="emit summary-only JSON")
    parser.add_argument("--strict", action="store_true", help="return non-zero when skipped or failed")
    args = parser.parse_args(argv)

    try:
        candidate_profile = load_candidate_profile(args.candidate_json)
        summary = run_smoke(
            jd_text=args.jd,
            candidate_profile=candidate_profile,
            compare_exact_scores=args.compare_exact_scores,
            strict=args.strict,
            readiness_runner=readiness_runner,
            real_matcher_runner=real_matcher_runner,
            shadow_runner=shadow_runner,
        )
    except Exception as exc:
        summary = build_input_failure_summary(args.jd, args.strict, type(exc).__name__)

    if args.json:
        safe = dict(summary)
        safe.pop("exit_code", None)
        print(json.dumps(safe, ensure_ascii=False, indent=2, default=str))
    else:
        print_text_summary(summary)
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
