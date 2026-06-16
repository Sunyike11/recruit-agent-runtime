#!/usr/bin/env python
import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OK = "OK"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


def resolve_dotenv_path() -> Path:
    override_path = os.getenv("RECRUIT_AGENT_DOTENV_PATH")
    if override_path:
        return Path(override_path).resolve()
    return PROJECT_ROOT / ".env"


def load_project_dotenv(import_module: Callable[[str], object] = importlib.import_module) -> CheckResult:
    dotenv_path = resolve_dotenv_path()
    if not dotenv_path.exists():
        return CheckResult("dotenv_load", SKIP, f".env not found at {dotenv_path}")

    try:
        dotenv_module = import_module("dotenv")
        load_dotenv = getattr(dotenv_module, "load_dotenv")
    except Exception as exc:
        return CheckResult("dotenv_load", SKIP, f"python-dotenv is not installed or unavailable: {exc!r}")

    try:
        loaded = load_dotenv(dotenv_path=dotenv_path, override=False)
    except Exception as exc:
        return CheckResult("dotenv_load", SKIP, f"unable to load .env: {exc!r}")

    if loaded:
        return CheckResult("dotenv_load", OK, f"loaded {dotenv_path}")
    return CheckResult("dotenv_load", SKIP, f".env loaded no values from {dotenv_path}")


def check_python_import_path() -> CheckResult:
    if str(PROJECT_ROOT) in sys.path:
        return CheckResult("python_import_path", OK, str(PROJECT_ROOT))
    return CheckResult("python_import_path", FAIL, "project root is not on sys.path")


def check_config_loads() -> CheckResult:
    try:
        from src.config import get_settings

        settings = get_settings()
        return CheckResult(
            "config_loads",
            OK,
            f"data_dir={settings.data_dir}; chroma_dir={settings.chroma_dir}; llm_model={settings.llm_model}",
        )
    except Exception as exc:
        return CheckResult("config_loads", FAIL, repr(exc))


def check_import(name: str, module_name: str, attr_name: Optional[str] = None) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
        if attr_name is not None:
            getattr(module, attr_name)
        target = module_name if attr_name is None else f"{module_name}.{attr_name}"
        return CheckResult(name, OK, target)
    except Exception as exc:
        return CheckResult(name, FAIL, repr(exc))


def check_agent_instantiation(name: str, module_name: str, class_name: str) -> CheckResult:
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except Exception as exc:
        return CheckResult(name, FAIL, f"import failed: {exc!r}")

    try:
        cls()
        return CheckResult(name, OK, f"{class_name} initialized")
    except Exception as exc:
        return CheckResult(name, SKIP, f"{class_name} import OK but initialization skipped/failed: {exc!r}")


def check_resume_retriever_instantiation() -> CheckResult:
    try:
        from src.services.retriever import ResumeRetriever
    except Exception as exc:
        return CheckResult("resume_retriever_init", FAIL, f"import failed: {exc!r}")

    try:
        from src.config import get_settings

        settings = get_settings()
        if not settings.chroma_dir.exists() or not any(settings.chroma_dir.iterdir()):
            return CheckResult(
                "resume_retriever_init",
                SKIP,
                f"chroma index path missing or empty: {settings.chroma_dir}",
            )
        ResumeRetriever(persist_dir=str(settings.chroma_dir))
        return CheckResult("resume_retriever_init", OK, "ResumeRetriever initialized")
    except Exception as exc:
        return CheckResult("resume_retriever_init", SKIP, f"ResumeRetriever init skipped/failed: {exc!r}")


def check_path_exists(name: str, path_getter: Callable[[], Path], require_non_empty: bool = False) -> CheckResult:
    try:
        path = path_getter()
    except Exception as exc:
        return CheckResult(name, FAIL, f"cannot resolve path: {exc!r}")

    if not path.exists():
        return CheckResult(name, SKIP, f"missing: {path}")
    if require_non_empty and path.is_dir() and not any(path.iterdir()):
        return CheckResult(name, SKIP, f"empty: {path}")
    return CheckResult(name, OK, str(path))


def check_env_var(name: str, env_name: str, required: bool = True) -> CheckResult:
    value = os.getenv(env_name)
    if value:
        return CheckResult(name, OK, f"{env_name}=set")
    status = FAIL if required else SKIP
    return CheckResult(name, status, f"{env_name} is not set")


def get_settings_path(attr_name: str) -> Path:
    from src.config import get_settings

    return getattr(get_settings(), attr_name)


def run_checks() -> List[CheckResult]:
    dotenv_result = load_project_dotenv()
    return [
        check_python_import_path(),
        dotenv_result,
        check_config_loads(),
        check_import("planner_agent_import", "src.agents.planner", "PlannerAgent"),
        check_import("matcher_agent_import", "src.agents.matcher", "MatcherAgent"),
        check_import("refiner_agent_import", "src.agents.refiner", "RefinerAgent"),
        check_import("retriever_agent_import", "src.agents.retriever", "RetrieverAgent"),
        check_import("resume_retriever_import", "src.services.retriever", "ResumeRetriever"),
        check_import("llama_index_core_import", "llama_index.core"),
        check_import("llama_index_huggingface_embedding_import", "llama_index.embeddings.huggingface", "HuggingFaceEmbedding"),
        check_import("llama_index_chroma_vector_store_import", "llama_index.vector_stores.chroma", "ChromaVectorStore"),
        check_import("chroma_dependency_import", "chromadb"),
        check_path_exists("data_dir_exists", lambda: get_settings_path("data_dir")),
        check_path_exists("chroma_db_exists", lambda: get_settings_path("chroma_dir"), require_non_empty=True),
        check_env_var("openai_api_key", "OPENAI_API_KEY", required=True),
        check_env_var("openai_api_base", "OPENAI_API_BASE", required=False),
        check_agent_instantiation("planner_agent_init", "src.agents.planner", "PlannerAgent"),
        check_agent_instantiation("matcher_agent_init", "src.agents.matcher", "MatcherAgent"),
        check_agent_instantiation("refiner_agent_init", "src.agents.refiner", "RefinerAgent"),
        check_resume_retriever_instantiation(),
    ]


def print_report(results: List[CheckResult]):
    for result in results:
        detail = f" - {result.detail}" if result.detail else ""
        print(f"[{result.status}] {result.name}{detail}")

    counts = {
        OK: sum(1 for result in results if result.status == OK),
        FAIL: sum(1 for result in results if result.status == FAIL),
        SKIP: sum(1 for result in results if result.status == SKIP),
    }
    print(f"SUMMARY: OK={counts[OK]} FAIL={counts[FAIL]} SKIP={counts[SKIP]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recruit Agent real workflow readiness smoke check")
    parser.add_argument("--strict", action="store_true", help="return non-zero when any check is FAIL or SKIP")
    args = parser.parse_args(argv)

    results = run_checks()
    print_report(results)
    if args.strict and any(result.status in {FAIL, SKIP} for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
