import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402


def parse_args(argv=None):
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build a local Chroma resume index from PDF resumes."
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(settings.data_dir),
        help="Directory containing local PDF resumes. Defaults to RECRUIT_AGENT_DATA_DIR or ./data.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(settings.chroma_dir),
        help="Chroma persistence directory. Defaults to RECRUIT_AGENT_CHROMA_DIR or ./chroma_db.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and count PDFs without loading embeddings or writing an index.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args(argv)


def _emit(payload, *, as_json: bool):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for key in ("status", "pdf_dir", "persist_dir", "pdf_count", "dry_run", "error_hint"):
        if key in payload:
            print(f"{key}={payload[key]}")


def run_cli(argv=None):
    args = parse_args(argv)
    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    persist_dir = Path(args.persist_dir).expanduser().resolve()
    pdf_count = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.is_dir() else 0
    payload = {
        "status": "ok",
        "pdf_dir": str(pdf_dir),
        "persist_dir": str(persist_dir),
        "pdf_count": pdf_count,
        "dry_run": bool(args.dry_run),
        "summary_only": True,
    }

    if not pdf_dir.is_dir():
        payload.update({"status": "failed", "error_hint": "pdf_dir_not_found"})
        _emit(payload, as_json=args.json)
        return 1

    if args.dry_run:
        _emit(payload, as_json=args.json)
        return 0

    from src.services.retriever import ResumeRetriever

    os.makedirs(persist_dir, exist_ok=True)
    retriever = ResumeRetriever(persist_dir=str(persist_dir))
    retriever.build_index_from_pdfs(str(pdf_dir))
    payload["index_built"] = True
    _emit(payload, as_json=args.json)
    return 0


def main(argv=None):
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
