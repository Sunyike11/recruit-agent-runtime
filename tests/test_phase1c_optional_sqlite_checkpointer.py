import builtins
import sys

from src.runtime.checkpoint import create_optional_sqlite_checkpointer


def block_retrieval_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        blocked_prefixes = (
            "llama_index",
            "chromadb",
            "src.agents.retriever",
            "src.services.retriever",
        )
        if name.startswith(blocked_prefixes):
            raise ModuleNotFoundError(f"blocked retrieval import in Phase1C+ test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_create_optional_sqlite_checkpointer_returns_real_saver(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    saver = create_optional_sqlite_checkpointer(tmp_path / "checkpoints.sqlite3")

    assert saver is not None
    assert saver.__class__.__name__ == "SqliteSaver"
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
