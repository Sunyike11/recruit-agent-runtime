import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.memory_influence import MemoryInfluenceEvalCase, MemoryInfluenceEvaluator  # noqa: E402
from src.runtime.memory_influence_export import (  # noqa: E402
    export_memory_influence_result_json,
    export_memory_influence_result_text,
)


DEFAULT_JD = "招聘熟悉 Python、RAG 和 LangGraph 的 AI Agent 工程师"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate summary-only memory influence using deterministic fake runners."
    )
    parser.add_argument("--jd", default=DEFAULT_JD, help="Job description text.")
    parser.add_argument("--case-id", default="memory_influence_demo", help="Evaluation case id.")
    parser.add_argument("--memory-source", choices=["none", "demo", "sqlite"], default="demo")
    parser.add_argument("--memory-db-path", default=None)
    parser.add_argument("--memory-max-items", type=int, default=5)
    parser.add_argument("--memory-max-chars", type=int, default=1200)
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument("--text", action="store_true", help="Emit text report.")
    parser.add_argument(
        "--use-runtime-variant",
        action="store_true",
        help="Reserved optional path; current CLI keeps fake runners by default.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on regression-risk result.")
    return parser.parse_args(argv)


def run_cli(argv=None):
    args = parse_args(argv)
    result = run_memory_influence_eval(args)
    if args.json:
        print(export_memory_influence_result_json(result))
    else:
        print(export_memory_influence_result_text(result))
    if args.strict and result.delta.decision == "regression_risk":
        return 1
    return 0


def run_memory_influence_eval(args):
    case = MemoryInfluenceEvalCase(
        case_id=args.case_id,
        raw_jd=args.jd,
        memory_source=args.memory_source,
        memory_config={
            "source": args.memory_source,
            "memory_db_path_present": bool(args.memory_db_path),
            "max_items": args.memory_max_items,
            "max_chars": args.memory_max_chars,
        },
        expected_effect="candidate_count" if args.memory_source in {"demo", "sqlite"} else None,
        metadata={
            "cli": True,
            "use_runtime_variant": bool(args.use_runtime_variant),
            "summary_only": True,
        },
    )
    evaluator = MemoryInfluenceEvaluator()
    if args.use_runtime_variant:
        return evaluator.run_case(
            case,
            no_memory_runner=_fake_no_memory_runner,
            with_memory_runner=_runtime_variant_unavailable_runner,
            memory_context=None,
        )
    return evaluator.run_case(
        case,
        no_memory_runner=_fake_no_memory_runner,
        with_memory_runner=_fake_with_memory_runner(args.memory_source),
        memory_context={"summary_only": True, "source": args.memory_source},
    )


def _fake_no_memory_runner(_raw_jd):
    return {
        "status": "ok",
        "runner_used": "fake_memory_influence_runner",
        "candidate_count": 1,
        "report_count": 1,
        "top_score_present": True,
        "top_scores": [70.0],
        "candidate_ids": ["candidate_demo_1"],
        "candidate_profile_preview_count": 1,
        "memory_context_provided": False,
        "memory_context_eligible_count": 0,
        "memory_context_rendered_char_count": 0,
        "output_keys": ["candidate_count", "report_count", "top_scores"],
        "summary_only": True,
    }


def _fake_with_memory_runner(memory_source):
    def run(_raw_jd, memory_context=None, metadata=None):
        if memory_source == "none":
            return {
                **_fake_no_memory_runner(_raw_jd),
                "memory_context_provided": False,
                "memory_context_eligible_count": 0,
            }
        return {
            "status": "ok",
            "runner_used": "fake_memory_influence_runner",
            "candidate_count": 2,
            "report_count": 1,
            "top_score_present": True,
            "top_scores": [76.0],
            "candidate_ids": ["candidate_demo_1", "candidate_demo_2"],
            "candidate_profile_preview_count": 2,
            "memory_context_provided": bool(memory_context is not None),
            "memory_context_eligible_count": 1,
            "memory_context_rendered_char_count": 80,
            "output_keys": ["candidate_count", "report_count", "top_scores"],
            "summary_only": True,
            "metadata": {
                "metadata_keys": sorted(str(key) for key in (metadata or {}).keys()),
                "summary_only": True,
            },
        }

    return run


def _runtime_variant_unavailable_runner(_raw_jd, **_kwargs):
    return {
        "status": "skipped",
        "runner_used": "runtime_variant_not_enabled",
        "candidate_count": 0,
        "report_count": 0,
        "top_score_present": False,
        "candidate_ids": [],
        "top_scores": [],
        "memory_context_provided": False,
        "memory_context_eligible_count": 0,
        "error_type": "",
        "output_keys": ["status", "runner_used"],
        "summary_only": True,
    }


def main(argv=None):
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
