import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.production_ab_baseline import (  # noqa: E402
    ProductionABBaselineConfig,
    ProductionABBaselineRunner,
)
from src.integration.production_skill_graph import (  # noqa: E402
    ProductionSkillGraphConfig,
    build_production_skill_graph_runner,
)
from src.runtime.entry import build_default_graph_runner, load_jd_text  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a summary-only legacy-vs-production-skill A/B baseline.")
    parser.add_argument("--jd", default=None, help="Job description text.")
    parser.add_argument("--jd-file", default=None, help="Path to UTF-8 JD text.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K overlap cutoff.")
    parser.add_argument("--allow-planner-fallback", action="store_true", help="Allow explicit Planner fallback in skill graph.")
    parser.add_argument("--db-path", default=None, help="Reserved runtime DB path marker for manual runs.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when rollback is recommended.")
    return parser.parse_args(argv)


def run_cli(argv=None, legacy_runner=None, skill_runner=None):
    args = parse_args(argv)
    raw_jd = load_jd_text(args.jd, args.jd_file)
    legacy = legacy_runner or build_default_graph_runner()
    skill = skill_runner or build_production_skill_graph_runner(
        ProductionSkillGraphConfig(
            enabled=True,
            allow_planner_fallback=bool(args.allow_planner_fallback),
            use_real_retriever=True,
            use_candidate_profile_preview=True,
            summary_only=True,
        )
    ).run
    result = ProductionABBaselineRunner(
        ProductionABBaselineConfig(
            enabled=True,
            allow_planner_fallback=bool(args.allow_planner_fallback),
            top_k=args.top_k,
            rollback_on_skill_failure=True,
            summary_only=True,
        )
    ).run(raw_jd, legacy_runner=legacy, skill_runner=skill)
    result.setdefault("metadata", {})
    result["metadata"].update(
        {
            "db_path_present": bool(args.db_path),
            "allow_planner_fallback": bool(args.allow_planner_fallback),
            "summary_only": True,
        }
    )
    _emit(result, as_json=args.json)
    return 1 if args.strict and result.get("rollback_recommended") else 0


def _emit(payload, *, as_json: bool):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    comparison = payload.get("comparison") or {}
    print("status=" + str(payload.get("status", "")))
    print("decision=" + str(comparison.get("decision", "")))
    print("risk_level=" + str(comparison.get("risk_level", payload.get("risk_level", ""))))
    print("rollback_recommended=" + str(payload.get("rollback_recommended", False)).lower())


if __name__ == "__main__":
    raise SystemExit(run_cli())
