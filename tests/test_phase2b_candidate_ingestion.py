import builtins
import sys

from src.domain import DomainSQLiteStore, ResumeIngestionPipeline, ingest_resume_text


RAW_RESUME_TEXT = """
姓名：孙一可
教育背景：上海大学 信息与信号处理 硕士
专业技能：Python, PyTorch, LangGraph, RAG, 3DGS, Diffusion
实习经历：金赛数字研究所 AI 算法实习生，负责 LLM 数据清洗与 Prompt 设计
项目经历：3DGS 数字人重建项目，研究 3D生成 与 AIGC 系统
"""


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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase2B test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_raw_resume_text_generates_resume_document():
    resume, candidate = ingest_resume_text(RAW_RESUME_TEXT, source_path="data/resume.pdf")

    assert resume.resume_id
    assert resume.candidate_id == candidate.candidate_id
    assert resume.source_path == "data/resume.pdf"
    assert resume.raw_text == RAW_RESUME_TEXT
    assert resume.chunks
    assert resume.metadata["parser"] == "deterministic"


def test_raw_resume_text_generates_candidate_profile():
    resume, candidate = ingest_resume_text(RAW_RESUME_TEXT)

    assert candidate.candidate_id
    assert candidate.source_resume_id == resume.resume_id
    assert candidate.name == "孙一可"
    assert "PyTorch" in candidate.skills
    assert "LangGraph" in candidate.skills
    assert "硕士" in candidate.education


def test_candidate_id_can_be_explicitly_passed():
    resume, candidate = ingest_resume_text(RAW_RESUME_TEXT, candidate_id="candidate_explicit")

    assert candidate.candidate_id == "candidate_explicit"
    assert resume.candidate_id == "candidate_explicit"


def test_candidate_id_is_generated_when_missing():
    resume, candidate = ingest_resume_text(RAW_RESUME_TEXT)

    assert candidate.candidate_id.startswith("candidate_")
    assert resume.candidate_id == candidate.candidate_id


def test_heuristic_parser_extracts_skills_education_experience_and_projects():
    _, candidate = ingest_resume_text(RAW_RESUME_TEXT)

    assert {"Python", "PyTorch", "LangGraph", "RAG", "3DGS", "Diffusion"}.issubset(set(candidate.skills))
    assert candidate.education == "教育背景：上海大学 信息与信号处理 硕士"
    assert any("实习" in item for item in candidate.experience)
    assert any("项目" in item for item in candidate.projects)


def test_ingestion_pipeline_saves_to_domain_store(tmp_path):
    store = DomainSQLiteStore(tmp_path / "domain.sqlite3")
    pipeline = ResumeIngestionPipeline(store=store)

    resume, candidate = pipeline.ingest_text(RAW_RESUME_TEXT, source_path="data/resume.pdf")

    assert store.get_resume_document(resume.resume_id) == resume
    assert store.get_candidate_profile(candidate.candidate_id) == candidate
    assert store.list_resume_documents() == [resume]
    assert store.list_candidate_profiles() == [candidate]


def test_domain_store_reinstantiation_keeps_ingested_resume_and_candidate(tmp_path):
    db_path = tmp_path / "domain.sqlite3"
    store = DomainSQLiteStore(db_path)
    pipeline = ResumeIngestionPipeline(store=store)
    resume, candidate = pipeline.ingest_text(RAW_RESUME_TEXT)

    reopened = DomainSQLiteStore(db_path)

    assert reopened.get_resume_document(resume.resume_id) == resume
    assert reopened.get_candidate_profile(candidate.candidate_id) == candidate


def test_phase2b_does_not_import_real_retrieval_modules(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)
    store = DomainSQLiteStore(tmp_path / "domain.sqlite3")
    pipeline = ResumeIngestionPipeline(store=store)

    resume, candidate = pipeline.ingest_text(RAW_RESUME_TEXT)

    assert store.get_resume_document(resume.resume_id) == resume
    assert store.get_candidate_profile(candidate.candidate_id) == candidate
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
