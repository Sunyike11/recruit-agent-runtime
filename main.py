import argparse
import json

from dotenv import load_dotenv

from src.core.graph_factory import resolve_recruit_graph_factory_config
from src.integration.production_skill_graph import (
    ProductionSkillGraphConfig,
    build_production_skill_graph_runner,
)
from src.runtime.entry import RuntimeEntryConfig, RuntimeEntryHarness, build_default_graph_runner


DEFAULT_INTERACTIVE_JD = """
招聘岗位：AI开发
职责：agent开发。
要求：计算机相关专业硕士，精通PyTorch。
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Recruit Agent through RuntimeEntryHarness.")
    parser.add_argument("--jd", default=DEFAULT_INTERACTIVE_JD, help="Job description text.")
    parser.add_argument("--graph-mode", choices=["legacy", "skill"], default=None)
    parser.add_argument("--allow-planner-fallback", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    load_dotenv()
    args = parse_args(argv)
    graph_config = resolve_recruit_graph_factory_config(
        requested_graph_mode=args.graph_mode,
        allow_planner_fallback=bool(args.allow_planner_fallback),
    )
    production_runner = None
    if graph_config.mode.value == "skill":
        production_runner = build_production_skill_graph_runner(
            ProductionSkillGraphConfig(
                enabled=True,
                allow_planner_fallback=bool(args.allow_planner_fallback),
                summary_only=True,
            )
        ).run

    result = RuntimeEntryHarness().run(
        args.jd,
        default_runner=build_default_graph_runner(),
        production_skill_graph_runner=production_runner,
        config=RuntimeEntryConfig(
            graph_mode=graph_config.mode.value,
            allow_planner_fallback=bool(args.allow_planner_fallback),
            summary_only=True,
            metadata={"entrypoint": "main.py", "summary_only": True},
        ),
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"status={payload.get('status')}")
        print(f"runner_used={payload.get('runner_used')}")
        print(f"selected_graph_mode={payload.get('output_summary', {}).get('selected_graph_mode', '')}")
        print(f"candidate_count={payload.get('output_summary', {}).get('candidate_count', 0)}")
        print(f"report_count={payload.get('output_summary', {}).get('report_count', 0)}")
        print("summary_only=true")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
