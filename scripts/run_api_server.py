import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the Recruit-Graph FastAPI Runtime Service MVP.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for local development.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
