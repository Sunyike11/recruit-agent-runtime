import hashlib
import json
import mimetypes
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.runtime.candidate_preview import (
    build_candidate_profile_preview_v2,
    candidate_profile_preview_v2_to_matcher_input,
)


RESUME_VERSION_STATUSES = {
    "uploaded",
    "queued",
    "parsing",
    "extracting_evidence",
    "indexing",
    "ready",
    "failed",
    "cancelled",
}
ALLOWED_MEDIA_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
}
MAX_UPLOAD_BYTES = 2_000_000


def utc_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_public_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class CandidateRecord:
    candidate_id: str
    tenant_id: str
    external_ref: str = ""
    status: str = "active"
    active_resume_version_id: str = ""
    active_profile_version_id: str = ""
    created_at: str = field(default_factory=utc_text)
    updated_at: str = field(default_factory=utc_text)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ResumeVersionRecord:
    resume_version_id: str
    candidate_id: str
    tenant_id: str
    version_number: int
    content_hash: str
    original_filename_safe: str
    media_type: str
    file_size: int
    storage_key: str
    status: str = "uploaded"
    parser_version: str = "resume_parse@1.0.0"
    profile_version: str = "candidate_profile_preview_v2"
    index_version: str = "managed_candidate_index_v1"
    created_at: str = field(default_factory=utc_text)
    ready_at: str = ""
    supersedes_version_id: str = ""
    error_type: str = ""
    ingestion_task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_hash: bool = False) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["content_hash_prefix"] = self.content_hash[:12]
        if not include_hash:
            data.pop("content_hash", None)
        data["summary_only"] = True
        return data


@dataclass
class CandidateProfileVersionRecord:
    profile_version_id: str
    candidate_id: str
    resume_version_id: str
    tenant_id: str
    schema_version: str = "candidate_profile_preview_v2"
    profile: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_version_id": self.profile_version_id,
            "candidate_id": self.candidate_id,
            "resume_version_id": self.resume_version_id,
            "tenant_id": self.tenant_id,
            "schema_version": self.schema_version,
            "profile": dict(self.profile),
            "created_at": self.created_at,
            "summary_only": True,
        }


@dataclass
class ResumeEvidenceRecord:
    evidence_id: str
    candidate_id: str
    resume_version_id: str
    tenant_id: str
    field_name: str
    evidence_type: str
    safe_summary: str
    source_locator: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "candidate_id": self.candidate_id,
            "resume_version_id": self.resume_version_id,
            "tenant_id": self.tenant_id,
            "field_name": self.field_name,
            "evidence_type": self.evidence_type,
            "safe_summary": self.safe_summary,
            "summary": self.safe_summary,
            "source_locator": self.source_locator,
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
            "summary_only": True,
        }


class CandidateSQLiteStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    external_ref TEXT,
                    status TEXT NOT NULL,
                    active_resume_version_id TEXT,
                    active_profile_version_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_tenant ON candidates(tenant_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_versions (
                    resume_version_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    original_filename_safe TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    storage_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ready_at TEXT,
                    supersedes_version_id TEXT,
                    error_type TEXT,
                    ingestion_task_id TEXT,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(tenant_id, candidate_id, content_hash),
                    UNIQUE(tenant_id, candidate_id, version_number)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_resume_versions_candidate ON resume_versions(tenant_id, candidate_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_profile_versions (
                    profile_version_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    resume_version_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    resume_version_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    safe_summary TEXT NOT NULL,
                    source_locator TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_index (
                    tenant_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    resume_version_id TEXT NOT NULL,
                    profile_version_id TEXT NOT NULL,
                    document_json TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, candidate_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_idempotency (
                    tenant_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, idempotency_key)
                )
                """
            )

    def create_candidate(self, *, tenant_id: str, external_ref: str = "", metadata: Optional[Mapping[str, Any]] = None) -> CandidateRecord:
        candidate = CandidateRecord(
            candidate_id=new_public_id("candidate"),
            tenant_id=tenant_id,
            external_ref=str(external_ref or "")[:120],
            metadata=dict(metadata or {}),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidates (
                    candidate_id, tenant_id, external_ref, status,
                    active_resume_version_id, active_profile_version_id,
                    created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    candidate.tenant_id,
                    candidate.external_ref,
                    candidate.status,
                    candidate.active_resume_version_id,
                    candidate.active_profile_version_id,
                    candidate.created_at,
                    candidate.updated_at,
                    _json(candidate.metadata),
                ),
            )
        return candidate

    def get_candidate(self, *, tenant_id: str, candidate_id: str) -> CandidateRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE tenant_id = ? AND candidate_id = ?",
                (tenant_id, candidate_id),
            ).fetchone()
        if row is None:
            raise KeyError("candidate_not_found")
        return _candidate_from_row(row)

    def create_resume_version(
        self,
        *,
        tenant_id: str,
        candidate_id: str,
        content_hash: str,
        original_filename_safe: str,
        media_type: str,
        file_size: int,
        storage_key: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ResumeVersionRecord, bool]:
        self.get_candidate(tenant_id=tenant_id, candidate_id=candidate_id)
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM resume_versions
                WHERE tenant_id = ? AND candidate_id = ? AND content_hash = ?
                """,
                (tenant_id, candidate_id, content_hash),
            ).fetchone()
            if existing is not None:
                return _resume_version_from_row(existing), False
            latest = conn.execute(
                """
                SELECT resume_version_id, version_number FROM resume_versions
                WHERE tenant_id = ? AND candidate_id = ?
                ORDER BY version_number DESC LIMIT 1
                """,
                (tenant_id, candidate_id),
            ).fetchone()
            version_number = int(latest["version_number"]) + 1 if latest else 1
            supersedes = str(latest["resume_version_id"] or "") if latest else ""
            record = ResumeVersionRecord(
                resume_version_id=new_public_id("resume_version"),
                candidate_id=candidate_id,
                tenant_id=tenant_id,
                version_number=version_number,
                content_hash=content_hash,
                original_filename_safe=original_filename_safe,
                media_type=media_type,
                file_size=int(file_size),
                storage_key=storage_key,
                status="uploaded",
                supersedes_version_id=supersedes,
                metadata=dict(metadata or {}),
            )
            conn.execute(
                """
                INSERT INTO resume_versions (
                    resume_version_id, candidate_id, tenant_id, version_number,
                    content_hash, original_filename_safe, media_type, file_size,
                    storage_key, status, parser_version, profile_version,
                    index_version, created_at, ready_at, supersedes_version_id,
                    error_type, ingestion_task_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _resume_version_values(record),
            )
        return record, True

    def get_resume_version(self, *, tenant_id: str, candidate_id: str, resume_version_id: str) -> ResumeVersionRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM resume_versions
                WHERE tenant_id = ? AND candidate_id = ? AND resume_version_id = ?
                """,
                (tenant_id, candidate_id, resume_version_id),
            ).fetchone()
        if row is None:
            raise KeyError("resume_version_not_found")
        return _resume_version_from_row(row)

    def list_resume_versions(self, *, tenant_id: str, candidate_id: str, limit: int = 50, offset: int = 0) -> List[ResumeVersionRecord]:
        self.get_candidate(tenant_id=tenant_id, candidate_id=candidate_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM resume_versions
                WHERE tenant_id = ? AND candidate_id = ?
                ORDER BY version_number DESC LIMIT ? OFFSET ?
                """,
                (tenant_id, candidate_id, int(limit), int(offset)),
            ).fetchall()
        return [_resume_version_from_row(row) for row in rows]

    def update_resume_status(self, *, tenant_id: str, resume_version_id: str, status: str, error_type: str = "", ingestion_task_id: str = "") -> None:
        if status not in RESUME_VERSION_STATUSES:
            raise ValueError("invalid_resume_status")
        ready_at = utc_text() if status == "ready" else ""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE resume_versions
                SET status = ?, error_type = ?, ingestion_task_id = COALESCE(NULLIF(?, ''), ingestion_task_id),
                    ready_at = COALESCE(NULLIF(?, ''), ready_at)
                WHERE tenant_id = ? AND resume_version_id = ?
                """,
                (status, error_type, ingestion_task_id, ready_at, tenant_id, resume_version_id),
            )

    def save_profile_and_evidence(
        self,
        *,
        tenant_id: str,
        candidate_id: str,
        resume_version_id: str,
        profile: Mapping[str, Any],
        evidence: Sequence[ResumeEvidenceRecord],
    ) -> CandidateProfileVersionRecord:
        profile_record = CandidateProfileVersionRecord(
            profile_version_id=new_public_id("profile_version"),
            candidate_id=candidate_id,
            resume_version_id=resume_version_id,
            tenant_id=tenant_id,
            profile=dict(profile),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidate_profile_versions (
                    profile_version_id, candidate_id, resume_version_id,
                    tenant_id, schema_version, profile_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_record.profile_version_id,
                    candidate_id,
                    resume_version_id,
                    tenant_id,
                    profile_record.schema_version,
                    _json(profile_record.profile),
                    profile_record.created_at,
                ),
            )
            conn.execute("DELETE FROM resume_evidence WHERE tenant_id = ? AND resume_version_id = ?", (tenant_id, resume_version_id))
            for item in evidence:
                conn.execute(
                    """
                    INSERT INTO resume_evidence (
                        evidence_id, candidate_id, resume_version_id, tenant_id,
                        field_name, evidence_type, safe_summary, source_locator,
                        provenance_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.evidence_id,
                        item.candidate_id,
                        item.resume_version_id,
                        item.tenant_id,
                        item.field_name,
                        item.evidence_type,
                        item.safe_summary,
                        item.source_locator,
                        _json(item.provenance),
                        item.created_at,
                    ),
                )
        return profile_record

    def activate_version(
        self,
        *,
        tenant_id: str,
        candidate_id: str,
        resume_version_id: str,
        profile_version_id: str,
    ) -> None:
        now = utc_text()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE candidates
                SET active_resume_version_id = ?, active_profile_version_id = ?, updated_at = ?
                WHERE tenant_id = ? AND candidate_id = ?
                """,
                (resume_version_id, profile_version_id, now, tenant_id, candidate_id),
            )

    def get_profile(self, *, tenant_id: str, candidate_id: str, profile_version_id: str = "", resume_version_id: str = "") -> CandidateProfileVersionRecord:
        where = "tenant_id = ? AND candidate_id = ?"
        args: List[Any] = [tenant_id, candidate_id]
        if profile_version_id:
            where += " AND profile_version_id = ?"
            args.append(profile_version_id)
        elif resume_version_id:
            where += " AND resume_version_id = ?"
            args.append(resume_version_id)
        else:
            candidate = self.get_candidate(tenant_id=tenant_id, candidate_id=candidate_id)
            where += " AND profile_version_id = ?"
            args.append(candidate.active_profile_version_id)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM candidate_profile_versions WHERE {where} ORDER BY created_at DESC LIMIT 1",
                tuple(args),
            ).fetchone()
        if row is None:
            raise KeyError("profile_not_found")
        return _profile_from_row(row)

    def list_evidence(
        self,
        *,
        tenant_id: str,
        candidate_id: str,
        resume_version_id: str = "",
        evidence_ids: Optional[Sequence[str]] = None,
        field_names: Optional[Sequence[str]] = None,
        limit: int = 20,
    ) -> List[ResumeEvidenceRecord]:
        if not resume_version_id:
            resume_version_id = self.get_candidate(tenant_id=tenant_id, candidate_id=candidate_id).active_resume_version_id
        clauses = ["tenant_id = ?", "candidate_id = ?", "resume_version_id = ?"]
        args: List[Any] = [tenant_id, candidate_id, resume_version_id]
        ids = [str(item) for item in evidence_ids or [] if str(item).strip()]
        fields = [str(item) for item in field_names or [] if str(item).strip()]
        if ids:
            clauses.append(f"evidence_id IN ({','.join('?' for _ in ids)})")
            args.extend(ids)
        if fields:
            clauses.append(f"field_name IN ({','.join('?' for _ in fields)})")
            args.extend(fields)
        args.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM resume_evidence
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at, evidence_id LIMIT ?
                """,
                tuple(args),
            ).fetchall()
        return [_evidence_from_row(row) for row in rows]

    def upsert_index_document(
        self,
        *,
        tenant_id: str,
        candidate_id: str,
        resume_version_id: str,
        profile_version_id: str,
        document: Mapping[str, Any],
        search_text: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidate_index (
                    tenant_id, candidate_id, resume_version_id, profile_version_id,
                    document_json, search_text, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, candidate_id) DO UPDATE SET
                    resume_version_id = excluded.resume_version_id,
                    profile_version_id = excluded.profile_version_id,
                    document_json = excluded.document_json,
                    search_text = excluded.search_text,
                    updated_at = excluded.updated_at
                """,
                (tenant_id, candidate_id, resume_version_id, profile_version_id, _json(document), search_text, utc_text()),
            )

    def search_index(self, *, tenant_id: str, query: str, top_k: int, excluded_candidate_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        excluded = set(str(item) for item in excluded_candidate_ids or [])
        terms = _tokenize(query)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_index WHERE tenant_id = ? ORDER BY updated_at DESC",
                (tenant_id,),
            ).fetchall()
        scored = []
        for row in rows:
            candidate_id = str(row["candidate_id"])
            if candidate_id in excluded:
                continue
            document = json.loads(row["document_json"])
            text = str(row["search_text"] or "").lower()
            score = sum(1 for term in terms if term in text) + 0.01
            scored.append((score, candidate_id, row, document))
        scored.sort(key=lambda item: (-item[0], item[1]))
        output = []
        for rank, (score, candidate_id, row, document) in enumerate(scored[:top_k], start=1):
            output.append(
                {
                    "candidate_id": candidate_id,
                    "rank": rank,
                    "retrieval_score": round(float(score), 6),
                    "resume_version_id": row["resume_version_id"],
                    "profile_version_id": row["profile_version_id"],
                    "document": document,
                    "summary_only": True,
                }
            )
        return output

    def remember_idempotency(self, *, tenant_id: str, key: str, fingerprint: str, object_id: str, object_type: str) -> Tuple[bool, str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT fingerprint, object_id FROM candidate_idempotency WHERE tenant_id = ? AND idempotency_key = ?",
                (tenant_id, key),
            ).fetchone()
            if row:
                if row["fingerprint"] != fingerprint:
                    raise ValueError("idempotency_conflict")
                return False, str(row["object_id"])
            conn.execute(
                """
                INSERT INTO candidate_idempotency (tenant_id, idempotency_key, fingerprint, object_id, object_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (tenant_id, key, fingerprint, object_id, object_type, utc_text()),
            )
        return True, object_id


class ResumeBlobStore:
    def __init__(self, root_dir: str | Path = "storage/resume_blobs"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def storage_key(self, *, tenant_id: str, candidate_id: str, content_hash: str, extension: str) -> str:
        tenant_part = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:12]
        safe_candidate = re.sub(r"[^A-Za-z0-9_-]", "_", candidate_id)[:80]
        return f"{tenant_part}/{safe_candidate}/{content_hash[:24]}{extension}"

    def put_bytes(self, *, storage_key: str, data: bytes) -> None:
        path = self._path(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def open(self, storage_key: str, mode: str = "rb"):
        return self._path(storage_key).open(mode)

    def read_bytes(self, storage_key: str) -> bytes:
        return self._path(storage_key).read_bytes()

    def exists(self, storage_key: str) -> bool:
        return self._path(storage_key).exists()

    def delete_uncommitted(self, storage_key: str) -> None:
        path = self._path(storage_key)
        if path.exists():
            path.unlink()

    def _path(self, storage_key: str) -> Path:
        key = str(storage_key or "").replace("\\", "/")
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError("invalid_storage_key")
        path = (self.root_dir / key).resolve()
        if not str(path).startswith(str(self.root_dir.resolve())):
            raise ValueError("invalid_storage_key")
        return path


def safe_filename(filename: str) -> str:
    name = Path(str(filename or "resume")).name
    name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]", "_", name).strip("._")
    return name[:120] or "resume.txt"


def infer_media_type(filename: str, declared: str = "") -> Tuple[str, str]:
    suffix = Path(filename or "").suffix.lower()
    guessed = mimetypes.types_map.get(suffix, "")
    media = str(declared or guessed or "").split(";", 1)[0].strip().lower()
    if media == "text/plain" or suffix == ".txt":
        return "text/plain", ".txt"
    if media == "application/pdf" or suffix == ".pdf":
        return "application/pdf", ".pdf"
    if media == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
    raise ValueError("UnsupportedMediaType")


def compute_sha256(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def extract_text_from_resume_bytes(data: bytes, *, media_type: str, filename: str = "") -> str:
    if not data:
        raise ValueError("EmptyFile")
    # Deterministic MVP: accept UTF-8/plain-text-like synthetic PDF/DOCX fixtures.
    # Real binary PDF/DOCX extraction can be upgraded behind this interface.
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            text = ""
    text = _strip_binary_noise(text)
    if not text.strip():
        raise ValueError("TextExtractionUnavailable")
    return text


def build_evidence_from_preview(*, tenant_id: str, candidate_id: str, resume_version_id: str, preview: Mapping[str, Any]) -> List[ResumeEvidenceRecord]:
    items: List[ResumeEvidenceRecord] = []
    for field_name, values in [
        ("education", preview.get("education_evidence_summaries") or []),
        ("experience", preview.get("experience_evidence_summaries") or []),
    ]:
        for index, value in enumerate(values, start=1):
            items.append(_evidence(tenant_id, candidate_id, resume_version_id, field_name, index, value))
    for index, project in enumerate(preview.get("projects") or [], start=1):
        summary = project.get("evidence_summary") if isinstance(project, Mapping) else str(project)
        items.append(_evidence(tenant_id, candidate_id, resume_version_id, "projects", index, summary))
    for skill, values in (preview.get("skill_evidence") or {}).items():
        for index, value in enumerate(values, start=1):
            items.append(_evidence(tenant_id, candidate_id, resume_version_id, "skills", index, f"{skill}: {value}"))
    achievements = preview.get("achievements") or {}
    if isinstance(achievements, Mapping):
        for field_name, values in achievements.items():
            for index, value in enumerate(values or [], start=1):
                items.append(_evidence(tenant_id, candidate_id, resume_version_id, str(field_name), index, value))
    if not items:
        items.append(_evidence(tenant_id, candidate_id, resume_version_id, "summary", 1, "No structured evidence extracted"))
    return items


def profile_to_index_document(profile: Mapping[str, Any], evidence: Sequence[ResumeEvidenceRecord]) -> Tuple[Dict[str, Any], str]:
    matcher_input = candidate_profile_preview_v2_to_matcher_input(profile)
    evidence_ids = [item.evidence_id for item in evidence]
    document = {
        "candidate_id": profile.get("candidate_id"),
        "resume_version_id": profile.get("resume_version_id", ""),
        "profile_version_id": profile.get("profile_version_id", ""),
        "skills": list(profile.get("skills") or []),
        "education": str(matcher_input.get("education") or ""),
        "experience": list(matcher_input.get("experience") or []),
        "projects": list(matcher_input.get("projects") or []),
        "achievements": dict(profile.get("achievements") or {}),
        "evidence_ids": evidence_ids,
        "summary_only": True,
    }
    search_text = " ".join(
        [
            " ".join(document["skills"]),
            document["education"],
            " ".join(document["experience"]),
            " ".join(document["projects"]),
            " ".join(item.safe_summary for item in evidence),
        ]
    )
    return document, search_text


def _evidence(tenant_id: str, candidate_id: str, resume_version_id: str, field_name: str, index: int, value: Any) -> ResumeEvidenceRecord:
    summary = _safe_summary(value)
    return ResumeEvidenceRecord(
        evidence_id=f"evidence_{hashlib.sha256(f'{candidate_id}:{resume_version_id}:{field_name}:{index}:{summary}'.encode()).hexdigest()[:16]}",
        candidate_id=candidate_id,
        resume_version_id=resume_version_id,
        tenant_id=tenant_id,
        field_name=field_name,
        evidence_type=field_name,
        safe_summary=summary,
        source_locator=f"{field_name}:{index}",
        provenance={
            "candidate_id": candidate_id,
            "resume_version_id": resume_version_id,
            "source_field": field_name,
            "evidence_present": bool(summary),
            "summary_only": True,
        },
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _candidate_from_row(row) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=row["candidate_id"],
        tenant_id=row["tenant_id"],
        external_ref=row["external_ref"] or "",
        status=row["status"],
        active_resume_version_id=row["active_resume_version_id"] or "",
        active_profile_version_id=row["active_profile_version_id"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _resume_version_values(record: ResumeVersionRecord) -> Tuple[Any, ...]:
    return (
        record.resume_version_id,
        record.candidate_id,
        record.tenant_id,
        record.version_number,
        record.content_hash,
        record.original_filename_safe,
        record.media_type,
        record.file_size,
        record.storage_key,
        record.status,
        record.parser_version,
        record.profile_version,
        record.index_version,
        record.created_at,
        record.ready_at,
        record.supersedes_version_id,
        record.error_type,
        record.ingestion_task_id,
        _json(record.metadata),
    )


def _resume_version_from_row(row) -> ResumeVersionRecord:
    return ResumeVersionRecord(
        resume_version_id=row["resume_version_id"],
        candidate_id=row["candidate_id"],
        tenant_id=row["tenant_id"],
        version_number=int(row["version_number"]),
        content_hash=row["content_hash"],
        original_filename_safe=row["original_filename_safe"],
        media_type=row["media_type"],
        file_size=int(row["file_size"]),
        storage_key=row["storage_key"],
        status=row["status"],
        parser_version=row["parser_version"],
        profile_version=row["profile_version"],
        index_version=row["index_version"],
        created_at=row["created_at"],
        ready_at=row["ready_at"] or "",
        supersedes_version_id=row["supersedes_version_id"] or "",
        error_type=row["error_type"] or "",
        ingestion_task_id=row["ingestion_task_id"] or "",
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _profile_from_row(row) -> CandidateProfileVersionRecord:
    return CandidateProfileVersionRecord(
        profile_version_id=row["profile_version_id"],
        candidate_id=row["candidate_id"],
        resume_version_id=row["resume_version_id"],
        tenant_id=row["tenant_id"],
        schema_version=row["schema_version"],
        profile=json.loads(row["profile_json"] or "{}"),
        created_at=row["created_at"],
    )


def _evidence_from_row(row) -> ResumeEvidenceRecord:
    return ResumeEvidenceRecord(
        evidence_id=row["evidence_id"],
        candidate_id=row["candidate_id"],
        resume_version_id=row["resume_version_id"],
        tenant_id=row["tenant_id"],
        field_name=row["field_name"],
        evidence_type=row["evidence_type"],
        safe_summary=row["safe_summary"],
        source_locator=row["source_locator"],
        provenance=json.loads(row["provenance_json"] or "{}"),
        created_at=row["created_at"],
    )


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.]*|[\u4e00-\u9fff]{2,}", text or "")]


def _strip_binary_noise(text: str) -> str:
    return "".join(ch if ch == "\n" or ch == "\t" or ord(ch) >= 32 else " " for ch in text)


def _safe_summary(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    lowered = text.lower()
    if any(token in lowered for token in ["忽略之前", "给我满分", "total_score", "管理员权限"]):
        return "[suspicious instruction treated as data]"
    return text[:limit]
