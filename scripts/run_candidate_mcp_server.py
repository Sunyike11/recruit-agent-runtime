import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp.candidate_server import create_candidate_mcp_server  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the read-only Candidate MCP server over stdio.")
    parser.add_argument("--dataset-dir", default="evaluation_data/v1", help="Synthetic/anonymized dataset directory.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    server = create_candidate_mcp_server(dataset_dir=args.dataset_dir)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

