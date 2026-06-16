#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.matcher_export import export_matcher_eval_report_json, export_matcher_eval_report_text
from src.evaluation.matcher_runner import (
    INPUT_FULL_RESUME,
    INPUT_PREVIEW,
    INPUT_PREVIEW_V1,
    INPUT_PREVIEW_V2,
    INPUT_RETRIEVER_TOP_K_PREVIEW,
    INPUT_RETRIEVER_TOP_K_PREVIEW_V1,
    INPUT_RETRIEVER_TOP_K_PREVIEW_V2,
    MatcherEvaluationConfig,
    MatcherEvaluationRunner,
    build_real_candidate_match_callable,
)


def _parse_modes(value: str) -> List[str]:
    if value == "all":
        return [INPUT_FULL_RESUME, INPUT_PREVIEW, INPUT_RETRIEVER_TOP_K_PREVIEW]
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_project_dotenv() -> Dict[str, Any]:
    dotenv_path = Path(".env")
    loaded = False
    if dotenv_path.exists():
        try:
            from dotenv import load_dotenv

            loaded = bool(load_dotenv(dotenv_path=dotenv_path, override=False))
        except Exception:
            loaded = False
    return {
        "dotenv_present": dotenv_path.exists(),
        "dotenv_loaded": loaded,
        "openai_api_key": "set" if os.environ.get("OPENAI_API_KEY") else "missing",
        "openai_api_base": "set" if os.environ.get("OPENAI_API_BASE") else "missing",
        "hf_token": "set" if os.environ.get("HF_TOKEN") else "missing",
        "hf_home": "set" if os.environ.get("HF_HOME") else "missing",
        "sentence_transformers_home": "set" if os.environ.get("SENTENCE_TRANSFORMERS_HOME") else "missing",
        "summary_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run summary-only matcher ranking evaluation.")
    parser.add_argument("--dataset-dir", default="evaluation_data/v1")
    parser.add_argument("--retrieval-result", default="evaluation_results/phase13b/retrieval_eval_raw_jd.json")
    parser.add_argument("--input-mode", default=INPUT_FULL_RESUME)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--output-dir", default="evaluation_results/phase13c")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-failed-cache", action="store_true")
    args = parser.parse_args()
    env_summary = _load_project_dotenv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "matcher_eval_cache.json"
    report_path = output_dir / "matcher_eval_summary.json"

    config = MatcherEvaluationConfig(
        dataset_dir=args.dataset_dir,
        retrieval_result_path=args.retrieval_result,
        input_modes=_parse_modes(args.input_mode),
        job_ids=_parse_csv(args.job_id),
        max_jobs=args.max_jobs,
        max_candidates=args.max_candidates,
        reuse_failed_cache=args.reuse_failed_cache,
    )
    runner = MatcherEvaluationRunner(config)
    cache = _load_cache(cache_path)
    try:
        report = runner.run(
            build_real_candidate_match_callable(),
            cache=cache,
            force=args.force,
            reuse_failed_cache=args.reuse_failed_cache,
        )
        report_dict = report.to_dict()
        report_dict["environment"] = env_summary
        _write_cache(cache_path, cache)
        json_payload = json.dumps(report_dict, ensure_ascii=False, indent=2)
        report_path.write_text(json_payload + "\n", encoding="utf-8")
        print(json_payload if args.json else export_matcher_eval_report_text(report))
        return 0
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_hint": "matcher_evaluation_failed",
            "summary_only": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
