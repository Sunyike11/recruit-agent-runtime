import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.inspect import RuntimeInspector  # noqa: E402
from src.runtime.sqlite_store import SQLiteRuntimeStore  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inspect summary-only runtime task timelines.")
    parser.add_argument("--db-path", required=True, help="SQLite runtime store path.")
    parser.add_argument("--task-id", default=None, help="Task ID to inspect.")
    parser.add_argument("--latest", action="store_true", help="Inspect the latest task in the store.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    parser.add_argument("--events", action="store_true", help="Include event timeline summaries.")
    return parser.parse_args(argv)


def run_cli(argv=None):
    args = parse_args(argv)
    store = SQLiteRuntimeStore(args.db_path)
    inspector = RuntimeInspector()
    try:
        if args.latest or not args.task_id:
            inspection = inspector.inspect_latest_task(store)
        else:
            inspection = inspector.inspect_task(args.task_id, store)
    except Exception as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "summary_only": True,
        }
        _emit(payload, as_json=args.json)
        return 1

    payload = inspection.to_dict()
    payload["status"] = "ok"
    if not args.events:
        payload["timeline_summary"] = []
    _emit(payload, as_json=args.json)
    return 0


def _emit(payload, *, as_json: bool):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"status={payload.get('status')}")
    print(f"task_id={payload.get('task_id', '')}")
    print(f"task_status={payload.get('task_status', '')}")
    print(f"runner_used={payload.get('runner_used', '')}")
    print(f"event_count={payload.get('event_count', 0)}")
    print(f"error_type={payload.get('error_type', '')}")
    print(f"error_hint={payload.get('error_hint', '')}")
    print("summary_only=true")


def main(argv=None):
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
