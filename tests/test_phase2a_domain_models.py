import builtins
import sys

from src.domain import (
    CandidateProfile,
    DomainSQLiteStore,
    HumanFeedback,
    JobRequirement,
    MatchReport,
    ResumeDocument,
    SearchAttempt,
)


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
            raise ModuleNotFoundError(f"blocked retrieval import in Phase2A test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def roundtrip(model):
    return model.__class__.from_dict(model.to_dict())


def test_domain_models_can_create_and_roundtrip():
    job = JobRequirement(
        raw_text="招聘数字人算法工程师",
        title="数字人算法工程师",
        required_skills=["PyTorch", "3D生成"],
        preferred_skills=["顶会论文"],
        education="硕士",
        experience_years=2,
        location="上海",
        metadata={"source": "test"},
    )
    candidate = CandidateProfile(
        name="候选人A",
        skills=["PyTorch"],
        education="硕士",
        experience=["AI实习"],
        projects=["3DGS数字人"],
        source_resume_id="resume_1",
    )
    resume = ResumeDocument(
        candidate_id=candidate.candidate_id,
        source_path="data/a.pdf",
        raw_text="简历全文",
        chunks=["chunk1", "chunk2"],
    )
    report = MatchReport(
        job_id=job.job_id,
        candidate_id=candidate.candidate_id,
        total_score=88,
        dimension_scores={"skills": 40, "project": 35},
        strengths=["项目相关"],
        weaknesses=["论文不足"],
        evidence=["3DGS项目"],
        recommendation="QUALIFIED",
    )
    attempt = SearchAttempt(
        job_id=job.job_id,
        query="PyTorch 3D生成",
        retrieved_candidate_ids=[candidate.candidate_id],
        retrieved_resume_ids=[resume.resume_id],
    )
    feedback = HumanFeedback(
        task_id="task_1",
        target_type="candidate",
        target_id=candidate.candidate_id,
        feedback_type="approve",
        payload={"approved": True},
    )

    for model in [job, candidate, resume, report, attempt, feedback]:
        restored = roundtrip(model)
        assert restored == model


def test_job_requirement_can_save_and_read(tmp_path):
    store = DomainSQLiteStore(tmp_path / "domain.sqlite3")
    job = JobRequirement(title="数字人", required_skills=["PyTorch"], metadata={"phase": "2A"})

    store.save_job_requirement(job)
    loaded = store.get_job_requirement(job.job_id)

    assert loaded == job


def test_candidate_profile_can_save_and_read(tmp_path):
    store = DomainSQLiteStore(tmp_path / "domain.sqlite3")
    candidate = CandidateProfile(name="候选人A", skills=["PyTorch"], education="硕士")

    store.save_candidate_profile(candidate)
    loaded = store.get_candidate_profile(candidate.candidate_id)

    assert loaded == candidate


def test_match_report_can_save_and_read(tmp_path):
    store = DomainSQLiteStore(tmp_path / "domain.sqlite3")
    report = MatchReport(
        job_id="job_1",
        candidate_id="candidate_1",
        total_score=90,
        recommendation="OUTSTANDING",
        evidence=["匹配证据"],
    )

    store.save_match_report(report)
    loaded = store.get_match_report(report.match_id)

    assert loaded == report


def test_resume_document_and_search_attempt_can_save_and_read(tmp_path):
    store = DomainSQLiteStore(tmp_path / "domain.sqlite3")
    resume = ResumeDocument(candidate_id="candidate_1", source_path="resume.pdf", chunks=["a", "b"])
    attempt = SearchAttempt(
        job_id="job_1",
        query="PyTorch",
        retrieved_candidate_ids=["candidate_1"],
        retrieved_resume_ids=[resume.resume_id],
    )

    store.save_resume_document(resume)
    store.save_search_attempt(attempt)

    assert store.get_resume_document(resume.resume_id) == resume
    assert store.get_search_attempt(attempt.search_id) == attempt


def test_domain_store_survives_reinstantiation(tmp_path):
    db_path = tmp_path / "domain.sqlite3"
    store = DomainSQLiteStore(db_path)
    job = JobRequirement(title="数字人", required_skills=["3D生成"])
    candidate = CandidateProfile(name="候选人A", skills=["3DGS"])
    report = MatchReport(job_id=job.job_id, candidate_id=candidate.candidate_id, total_score=82)
    store.save_job_requirement(job)
    store.save_candidate_profile(candidate)
    store.save_match_report(report)

    reopened = DomainSQLiteStore(db_path)

    assert reopened.get_job_requirement(job.job_id) == job
    assert reopened.get_candidate_profile(candidate.candidate_id) == candidate
    assert reopened.get_match_report(report.match_id) == report


def test_phase2a_does_not_import_real_retrieval_modules(tmp_path, monkeypatch):
    block_retrieval_imports(monkeypatch)
    sys.modules.pop("src.agents.retriever", None)
    sys.modules.pop("src.services.retriever", None)

    store = DomainSQLiteStore(tmp_path / "domain.sqlite3")
    candidate = CandidateProfile(name="候选人A")
    store.save_candidate_profile(candidate)

    assert store.get_candidate_profile(candidate.candidate_id) == candidate
    assert "src.agents.retriever" not in sys.modules
    assert "src.services.retriever" not in sys.modules
