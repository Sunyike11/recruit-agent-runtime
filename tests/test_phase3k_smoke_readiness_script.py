import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_real_readiness.py"
OPENAI_KEY_ENV = "OPENAI_API_KEY"


def isolated_env_without_real_dotenv(tmp_path):
    env = os.environ.copy()
    env.pop(OPENAI_KEY_ENV, None)
    env["RECRUIT_AGENT_DOTENV_PATH"] = str(tmp_path / "missing.env")
    return env


def test_smoke_real_readiness_script_can_be_imported():
    spec = importlib.util.spec_from_file_location("smoke_real_readiness", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert hasattr(module, "run_checks")
    assert hasattr(module, "main")
    assert module.OK == "OK"
    assert module.FAIL == "FAIL"
    assert module.SKIP == "SKIP"


def test_smoke_real_readiness_loads_openai_api_key_from_temp_env_without_leaking_value(tmp_path):
    dotenv = tmp_path / ".env"
    secret_value = "TEST_OPENAI_KEY_PHASE3K_TEMP_SECRET_VALUE"
    dotenv.write_text(f"{OPENAI_KEY_ENV}={secret_value}\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop(OPENAI_KEY_ENV, None)
    env["RECRUIT_AGENT_DOTENV_PATH"] = str(dotenv)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "[OK] dotenv_load" in result.stdout
    assert "[OK] openai_api_key - OPENAI_API_KEY=set" in result.stdout
    assert secret_value not in result.stdout


def test_smoke_real_readiness_dotenv_missing_dependency_is_graceful(tmp_path):
    spec = importlib.util.spec_from_file_location("smoke_real_readiness_missing_dotenv", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{OPENAI_KEY_ENV}=TEST_OPENAI_KEY_NOT_USED\n", encoding="utf-8")

    def missing_dotenv_import(name):
        raise ModuleNotFoundError("fake missing python-dotenv")

    old_value = os.environ.get("RECRUIT_AGENT_DOTENV_PATH")
    os.environ["RECRUIT_AGENT_DOTENV_PATH"] = str(dotenv)
    try:
        result = module.load_project_dotenv(import_module=missing_dotenv_import)
    finally:
        if old_value is None:
            os.environ.pop("RECRUIT_AGENT_DOTENV_PATH", None)
        else:
            os.environ["RECRUIT_AGENT_DOTENV_PATH"] = old_value

    assert result.status == module.SKIP
    assert "python-dotenv" in result.detail


def test_smoke_real_readiness_dotenv_loader_does_not_override_existing_environment(tmp_path):
    spec = importlib.util.spec_from_file_location("smoke_real_readiness_fake_dotenv", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{OPENAI_KEY_ENV}=TEST_OPENAI_KEY_FROM_FILE\n", encoding="utf-8")
    calls = {}

    def fake_load_dotenv(dotenv_path, override):
        calls["dotenv_path"] = dotenv_path
        calls["override"] = override
        return True

    fake_dotenv_module = types.SimpleNamespace(load_dotenv=fake_load_dotenv)

    def fake_import(name):
        assert name == "dotenv"
        return fake_dotenv_module

    old_value = os.environ.get("RECRUIT_AGENT_DOTENV_PATH")
    os.environ["RECRUIT_AGENT_DOTENV_PATH"] = str(dotenv)
    try:
        result = module.load_project_dotenv(import_module=fake_import)
    finally:
        if old_value is None:
            os.environ.pop("RECRUIT_AGENT_DOTENV_PATH", None)
        else:
            os.environ["RECRUIT_AGENT_DOTENV_PATH"] = old_value

    assert result.status == module.OK
    assert calls["dotenv_path"] == dotenv.resolve()
    assert calls["override"] is False


def test_smoke_real_readiness_default_mode_does_not_crash_when_dependencies_are_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=isolated_env_without_real_dotenv(tmp_path),
    )

    assert result.returncode == 0
    assert "SUMMARY:" in result.stdout
    assert any(marker in result.stdout for marker in ("[OK]", "[FAIL]", "[SKIP]"))


def test_smoke_real_readiness_output_contains_status_summary(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=isolated_env_without_real_dotenv(tmp_path),
    )

    assert "OK=" in result.stdout
    assert "FAIL=" in result.stdout
    assert "SKIP=" in result.stdout


def test_smoke_real_readiness_strict_mode_returns_nonzero_for_fake_missing_dependency(monkeypatch):
    monkeypatch.setenv("RECRUIT_AGENT_CHROMA_DIR", str(PROJECT_ROOT / "tests" / "fixtures" / "missing_chroma_db"))
    monkeypatch.setenv("RECRUIT_AGENT_DOTENV_PATH", str(PROJECT_ROOT / "tests" / "fixtures" / "missing.env"))
    monkeypatch.delenv(OPENAI_KEY_ENV, raising=False)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--strict"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SUMMARY:" in result.stdout


def test_phase3k_does_not_modify_production_graph():
    graph_source = (PROJECT_ROOT / "src" / "core" / "graph.py").read_text(encoding="utf-8")

    assert "smoke_real_readiness" not in graph_source
    assert "SkillRegistry" not in graph_source
    assert "RecruitmentSkillWorkflow" not in graph_source
