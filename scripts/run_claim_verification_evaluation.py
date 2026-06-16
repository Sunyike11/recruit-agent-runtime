import argparse
import json
from pathlib import Path

from src.evaluation.claim_verification import run_claim_verification_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run summary-only claim verification smoke evaluation.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_claim_verification_smoke().to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json or not args.output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
