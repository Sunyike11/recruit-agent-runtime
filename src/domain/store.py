import json
import sqlite3
from pathlib import Path
from typing import Type

from src.domain.models import (
    CandidateProfile,
    JobRequirement,
    MatchReport,
    ResumeDocument,
    SearchAttempt,
)


class DomainSQLiteStore:
    """SQLite store for Phase2A domain objects.

    Objects are stored as JSON blobs to keep Phase2A conservative. Field-level
    indexing can be added later when the ingestion and memory layers stabilize.
    """

    TABLES = {
        "job_requirements": ("job_id", JobRequirement),
        "candidate_profiles": ("candidate_id", CandidateProfile),
        "resume_documents": ("resume_id", ResumeDocument),
        "match_reports": ("match_id", MatchReport),
        "search_attempts": ("search_id", SearchAttempt),
    }

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            for table_name, (id_column, _) in self.TABLES.items():
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        {id_column} TEXT PRIMARY KEY,
                        data_json TEXT NOT NULL
                    )
                    """
                )

    def save_job_requirement(self, job: JobRequirement) -> JobRequirement:
        self._save("job_requirements", "job_id", job.job_id, job.to_dict())
        return job

    def get_job_requirement(self, job_id: str) -> JobRequirement:
        return self._get("job_requirements", "job_id", job_id, JobRequirement)

    def save_candidate_profile(self, candidate: CandidateProfile) -> CandidateProfile:
        self._save("candidate_profiles", "candidate_id", candidate.candidate_id, candidate.to_dict())
        return candidate

    def get_candidate_profile(self, candidate_id: str) -> CandidateProfile:
        return self._get("candidate_profiles", "candidate_id", candidate_id, CandidateProfile)

    def list_candidate_profiles(self):
        return self._list("candidate_profiles", CandidateProfile)

    def save_resume_document(self, resume: ResumeDocument) -> ResumeDocument:
        self._save("resume_documents", "resume_id", resume.resume_id, resume.to_dict())
        return resume

    def get_resume_document(self, resume_id: str) -> ResumeDocument:
        return self._get("resume_documents", "resume_id", resume_id, ResumeDocument)

    def list_resume_documents(self):
        return self._list("resume_documents", ResumeDocument)

    def save_match_report(self, report: MatchReport) -> MatchReport:
        self._save("match_reports", "match_id", report.match_id, report.to_dict())
        return report

    def get_match_report(self, match_id: str) -> MatchReport:
        return self._get("match_reports", "match_id", match_id, MatchReport)

    def save_search_attempt(self, attempt: SearchAttempt) -> SearchAttempt:
        self._save("search_attempts", "search_id", attempt.search_id, attempt.to_dict())
        return attempt

    def get_search_attempt(self, search_id: str) -> SearchAttempt:
        return self._get("search_attempts", "search_id", search_id, SearchAttempt)

    def _save(self, table_name: str, id_column: str, object_id: str, data: dict):
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {table_name} ({id_column}, data_json)
                VALUES (?, ?)
                ON CONFLICT({id_column}) DO UPDATE SET data_json = excluded.data_json
                """,
                (object_id, json.dumps(data, ensure_ascii=False)),
            )

    def _get(self, table_name: str, id_column: str, object_id: str, model_cls: Type):
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT data_json FROM {table_name} WHERE {id_column} = ?",
                (object_id,),
            ).fetchone()
        if row is None:
            raise KeyError(object_id)
        return model_cls.from_dict(json.loads(row["data_json"]))

    def _list(self, table_name: str, model_cls: Type):
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT data_json FROM {table_name} ORDER BY rowid"
            ).fetchall()
        return [model_cls.from_dict(json.loads(row["data_json"])) for row in rows]
