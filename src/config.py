import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    retriever_top_k: int = 5
    matcher_loop_limit: int = 2


def get_settings() -> Settings:
    """Load project settings with environment variable overrides."""
    project_root = Path(os.getenv("RECRUIT_AGENT_PROJECT_ROOT", PROJECT_ROOT)).resolve()
    return Settings(
        project_root=project_root,
        data_dir=Path(os.getenv("RECRUIT_AGENT_DATA_DIR", project_root / "data")).resolve(),
        chroma_dir=Path(os.getenv("RECRUIT_AGENT_CHROMA_DIR", project_root / "chroma_db")).resolve(),
        llm_model=os.getenv("RECRUIT_AGENT_LLM_MODEL", "deepseek-chat"),
        llm_temperature=float(os.getenv("RECRUIT_AGENT_LLM_TEMPERATURE", "0")),
        embedding_model=os.getenv("RECRUIT_AGENT_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        retriever_top_k=int(os.getenv("RECRUIT_AGENT_RETRIEVER_TOP_K", "5")),
        matcher_loop_limit=int(os.getenv("RECRUIT_AGENT_MATCHER_LOOP_LIMIT", "2")),
    )
