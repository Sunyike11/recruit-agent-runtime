import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.memory.models import MemoryRecord, MemorySourceType, MemoryType


FEEDBACK_TYPES = {
    "approve",
    "reject",
    "correction",
    "comment",
    "evidence_missing",
    "ranking_wrong",
    "candidate_irrelevant",
    "candidate_relevant",
    "unsafe_claim",
}
REVIEW_TYPES = {
    "match_report_review",
    "claim_verification_review",
    "feedback_correction_review",
    "memory_candidate_review",
    "resume_ingestion_review",
}
REVIEW_STATUSES = {"pending", "in_review", "approved", "rejected", "corrected", "closed", "expired"}
DECISIONS = {"approve", "reject", "correct", "close"}
MEMORY_CANDIDATE_TYPES = {"tenant_preference", "matching_rule", "task_experience", "candidate_constraint"}
ACTIVE_MEMORY_STATUSES = {"active", "revoked", "expired", "superseded"}
CORRECTION_FIELDS = {
    "publication_status",
    "education",
    "experience",
    "project",
    "skill",
    "ranking",
    "recommendation",
    "evidence",
}
AUTO_REVIEW_FEEDBACK_TYPES = {"reject", "correction", "evidence_missing", "unsafe_claim"}
MEMORY_PROMOTION_FEEDBACK_TYPES = {
    "approve",
    "correction",
    "evidence_missing",
    "ranking_wrong",
    "candidate_irrelevant",
    "candidate_relevant",
    "unsafe_claim",
}


def utc_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    tenant_id: str
    task_id: str
    session_id: str = ""
    candidate_id: str = ""
    report_id: str = ""
    resume_version_id: str = ""
    profile_version_id: str = ""
    feedback_type: str = "comment"
    rating: Optional[int] = None
    comment_length: int = 0
    correction_payload: Dict[str, Any] = field(default_factory=dict)
    claim_ids: List[str] = field(default_factory=list)
    created_by: str = "api"
    created_at: str = field(default_factory=utc_text)
    source: str = "api"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "tenant_id": self.tenant_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "report_id": self.report_id,
            "resume_version_id": self.resume_version_id,
            "profile_version_id": self.profile_version_id,
            "feedback_type": self.feedback_type,
            "rating": self.rating,
            "comment_length": self.comment_length,
            "correction_fields": sorted(str(key) for key in self.correction_payload.keys()),
            "claim_ids": list(self.claim_ids),
            "created_by": self.created_by,
            "created_at": self.created_at,
            "source": self.source,
            "metadata": _safe_metadata(self.metadata),
            "summary_only": True,
        }


@dataclass
class ReviewItem:
    review_id: str
    tenant_id: str
    review_type: str
    task_id: str
    candidate_id: str = ""
    report_id: str = ""
    feedback_id: str = ""
    claim_verification_status: str = ""
    priority: str = "normal"
    status: str = "pending"
    reason_codes: List[str] = field(default_factory=list)
    assigned_to: str = ""
    created_at: str = field(default_factory=utc_text)
    decided_at: str = ""
    decision_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "tenant_id": self.tenant_id,
            "review_type": self.review_type,
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "report_id": self.report_id,
            "feedback_id": self.feedback_id,
            "claim_verification_status": self.claim_verification_status,
            "priority": self.priority,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "assigned_to": self.assigned_to,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "decision_id": self.decision_id,
            "metadata": _safe_metadata(self.metadata),
            "summary_only": True,
        }


@dataclass(frozen=True)
class HumanDecision:
    decision_id: str
    review_id: str
    tenant_id: str
    decision: str
    correction_overlay: Dict[str, Any] = field(default_factory=dict)
    reason_length: int = 0
    promote_to_memory: bool = False
    memory_candidate_type: str = ""
    expires_at: str = ""
    supersedes_memory_id: str = ""
    created_by: str = "api"
    created_at: str = field(default_factory=utc_text)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "review_id": self.review_id,
            "tenant_id": self.tenant_id,
            "decision": self.decision,
            "correction_fields": sorted(str(key) for key in self.correction_overlay.keys()),
            "reason_length": self.reason_length,
            "promote_to_memory": self.promote_to_memory,
            "memory_candidate_type": self.memory_candidate_type,
            "expires_at": self.expires_at,
            "supersedes_memory_id": self.supersedes_memory_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "summary_only": True,
        }


@dataclass
class MemoryCandidate:
    memory_candidate_id: str
    tenant_id: str
    memory_type: str
    safe_content: str
    source_feedback_id: str
    source_decision_id: str
    source_task_id: str
    candidate_id: str = ""
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    provenance: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending_review"
    expires_at: str = ""
    supersedes_memory_id: str = ""
    created_at: str = field(default_factory=utc_text)
    decided_at: str = ""
    memory_id: str = ""

    def to_summary(self) -> Dict[str, Any]:
        return {
            "memory_candidate_id": self.memory_candidate_id,
            "tenant_id": self.tenant_id,
            "memory_type": self.memory_type,
            "safe_content_summary": _safe_text(self.safe_content, 160),
            "source_feedback_id": self.source_feedback_id,
            "source_decision_id": self.source_decision_id,
            "source_task_id": self.source_task_id,
            "candidate_id": self.candidate_id,
            "tags": list(self.tags),
            "importance": self.importance,
            "provenance": _safe_metadata(self.provenance),
            "status": self.status,
            "expires_at": self.expires_at,
            "supersedes_memory_id": self.supersedes_memory_id,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "memory_id": self.memory_id,
            "summary_only": True,
        }


@dataclass
class GovernedMemory:
    memory_id: str
    tenant_id: str
    memory_type: str
    safe_content: str
    source_memory_candidate_id: str
    source_feedback_id: str
    source_decision_id: str
    source_task_id: str
    candidate_id: str = ""
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    version: int = 1
    status: str = "active"
    expires_at: str = ""
    supersedes_memory_id: str = ""
    revoked_at: str = ""
    created_at: str = field(default_factory=utc_text)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "tenant_id": self.tenant_id,
            "memory_type": self.memory_type,
            "safe_content_summary": _safe_text(self.safe_content, 160),
            "source_memory_candidate_id": self.source_memory_candidate_id,
            "source_feedback_id": self.source_feedback_id,
            "source_decision_id": self.source_decision_id,
            "source_task_id": self.source_task_id,
            "candidate_id": self.candidate_id,
            "tags": list(self.tags),
            "importance": self.importance,
            "version": self.version,
            "status": self.status,
            "expires_at": self.expires_at,
            "supersedes_memory_id": self.supersedes_memory_id,
            "revoked_at": self.revoked_at,
            "created_at": self.created_at,
            "metadata": _safe_metadata(self.metadata),
            "summary_only": True,
        }

    def to_memory_record(self) -> MemoryRecord:
        return MemoryRecord(
            memory_id=self.memory_id,
            memory_type=_record_memory_type(self.memory_type),
            source_type=MemorySourceType.HUMAN_FEEDBACK.value,
            source_id=self.source_feedback_id,
            content=self.safe_content,
            importance=self.importance,
            tags=list(self.tags),
            metadata={
                "tenant_id": self.tenant_id,
                "memory_type": self.memory_type,
                "version": self.version,
                "promoted_from_reflection": True,
                "dry_run": False,
                "source_reflection_id": self.source_decision_id,
                "source_candidate_id": self.candidate_id or self.source_task_id,
                "source_task_id": self.source_task_id,
                "source_memory_candidate_id": self.source_memory_candidate_id,
                "approved_by": "human_review",
                "reviewer": "human_review",
                "expires_at": self.expires_at,
                "status": self.status,
                "summary_only": True,
            },
        )


class ReviewMemoryStore:
    """SQLite-backed feedback, review, and governed memory loop store."""

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
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    session_id TEXT,
                    candidate_id TEXT,
                    report_id TEXT,
                    resume_version_id TEXT,
                    profile_version_id TEXT,
                    feedback_type TEXT NOT NULL,
                    rating INTEGER,
                    comment_length INTEGER NOT NULL,
                    correction_json TEXT NOT NULL,
                    claim_ids_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_task ON feedback_records(tenant_id, task_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    review_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    review_type TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    candidate_id TEXT,
                    report_id TEXT,
                    feedback_id TEXT,
                    claim_verification_status TEXT,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    assigned_to TEXT,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    decision_id TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_tenant_status ON review_items(tenant_id, status)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS human_decisions (
                    decision_id TEXT PRIMARY KEY,
                    review_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    correction_json TEXT NOT NULL,
                    reason_length INTEGER NOT NULL,
                    promote_to_memory INTEGER NOT NULL,
                    memory_candidate_type TEXT,
                    expires_at TEXT,
                    supersedes_memory_id TEXT,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(tenant_id, review_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_candidates (
                    memory_candidate_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    safe_content TEXT NOT NULL,
                    source_feedback_id TEXT NOT NULL,
                    source_decision_id TEXT NOT NULL,
                    source_task_id TEXT NOT NULL,
                    candidate_id TEXT,
                    tags_json TEXT NOT NULL,
                    importance REAL NOT NULL,
                    provenance_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expires_at TEXT,
                    supersedes_memory_id TEXT,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    memory_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS governed_memories (
                    memory_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    safe_content TEXT NOT NULL,
                    source_memory_candidate_id TEXT NOT NULL,
                    source_feedback_id TEXT NOT NULL,
                    source_decision_id TEXT NOT NULL,
                    source_task_id TEXT NOT NULL,
                    candidate_id TEXT,
                    tags_json TEXT NOT NULL,
                    importance REAL NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    expires_at TEXT,
                    supersedes_memory_id TEXT,
                    revoked_at TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tenant_status ON governed_memories(tenant_id, status)")

    def create_feedback(
        self,
        *,
        tenant_id: str,
        task_id: str,
        session_id: str = "",
        feedback_type: str,
        rating: Optional[int] = None,
        comment: str = "",
        correction: Optional[Mapping[str, Any]] = None,
        claim_ids: Optional[Sequence[str]] = None,
        candidate_id: str = "",
        report_id: str = "",
        resume_version_id: str = "",
        profile_version_id: str = "",
        created_by: str = "api",
        source: str = "api",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> FeedbackRecord:
        if feedback_type not in FEEDBACK_TYPES:
            raise ValueError("unsupported_feedback_type")
        safe_correction = validate_correction_payload(correction or {})
        record = FeedbackRecord(
            feedback_id=new_id("feedback"),
            tenant_id=tenant_id,
            task_id=task_id,
            session_id=session_id,
            candidate_id=candidate_id,
            report_id=report_id,
            resume_version_id=resume_version_id,
            profile_version_id=profile_version_id,
            feedback_type=feedback_type,
            rating=rating,
            comment_length=len(comment or ""),
            correction_payload=safe_correction,
            claim_ids=[str(item)[:120] for item in (claim_ids or [])],
            created_by=created_by,
            source=source,
            metadata=_safe_metadata(metadata or {}),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_records (
                    feedback_id, tenant_id, task_id, session_id, candidate_id,
                    report_id, resume_version_id, profile_version_id, feedback_type,
                    rating, comment_length, correction_json, claim_ids_json,
                    created_by, created_at, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _feedback_values(record),
            )
        return record

    def list_feedback(self, *, tenant_id: str, task_id: str, limit: int = 100) -> List[FeedbackRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM feedback_records
                WHERE tenant_id = ? AND task_id = ?
                ORDER BY created_at, rowid
                LIMIT ?
                """,
                (tenant_id, task_id, max(1, min(int(limit), 200))),
            ).fetchall()
        return [_feedback_from_row(row) for row in rows]

    def create_review_from_feedback(
        self,
        feedback: FeedbackRecord,
        *,
        request_review: bool = False,
        reason_codes: Optional[Sequence[str]] = None,
    ) -> Optional[ReviewItem]:
        should_create = request_review or feedback.feedback_type in AUTO_REVIEW_FEEDBACK_TYPES
        if not should_create:
            return None
        review_type = "feedback_correction_review" if feedback.feedback_type == "correction" else "match_report_review"
        priority = "high" if feedback.feedback_type in {"reject", "unsafe_claim"} else "normal"
        return self.create_review(
            tenant_id=feedback.tenant_id,
            review_type=review_type,
            task_id=feedback.task_id,
            candidate_id=feedback.candidate_id,
            report_id=feedback.report_id,
            feedback_id=feedback.feedback_id,
            priority=priority,
            reason_codes=list(reason_codes or [feedback.feedback_type]),
            metadata={"source": "feedback", "summary_only": True},
        )

    def create_review(
        self,
        *,
        tenant_id: str,
        review_type: str,
        task_id: str,
        candidate_id: str = "",
        report_id: str = "",
        feedback_id: str = "",
        claim_verification_status: str = "",
        priority: str = "normal",
        reason_codes: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ReviewItem:
        if review_type not in REVIEW_TYPES:
            raise ValueError("unsupported_review_type")
        item = ReviewItem(
            review_id=new_id("review"),
            tenant_id=tenant_id,
            review_type=review_type,
            task_id=task_id,
            candidate_id=candidate_id,
            report_id=report_id,
            feedback_id=feedback_id,
            claim_verification_status=claim_verification_status,
            priority=priority,
            reason_codes=[str(code)[:80] for code in (reason_codes or [])],
            metadata=_safe_metadata(metadata or {}),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_items (
                    review_id, tenant_id, review_type, task_id, candidate_id,
                    report_id, feedback_id, claim_verification_status, priority,
                    status, reason_codes_json, assigned_to, created_at, decided_at,
                    decision_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _review_values(item),
            )
        return item

    def get_review(self, *, tenant_id: str, review_id: str) -> ReviewItem:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_items WHERE tenant_id = ? AND review_id = ?",
                (tenant_id, review_id),
            ).fetchone()
        if row is None:
            raise KeyError("review_not_found")
        return _review_from_row(row)

    def list_reviews(
        self,
        *,
        tenant_id: str,
        status: str = "",
        review_type: str = "",
        task_id: str = "",
        candidate_id: str = "",
        limit: int = 100,
    ) -> List[ReviewItem]:
        query = "SELECT * FROM review_items WHERE tenant_id = ?"
        params: List[Any] = [tenant_id]
        for field_name, value in [
            ("status", status),
            ("review_type", review_type),
            ("task_id", task_id),
            ("candidate_id", candidate_id),
        ]:
            if value:
                query += f" AND {field_name} = ?"
                params.append(value)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_review_from_row(row) for row in rows]

    def decide_review(
        self,
        *,
        tenant_id: str,
        review_id: str,
        decision: str,
        correction: Optional[Mapping[str, Any]] = None,
        reason: str = "",
        promote_to_memory: bool = False,
        memory_candidate_type: str = "",
        expires_at: str = "",
        supersedes_memory_id: str = "",
        created_by: str = "api",
    ) -> Tuple[HumanDecision, ReviewItem, Optional[MemoryCandidate]]:
        if decision not in DECISIONS:
            raise ValueError("unsupported_decision")
        safe_correction = validate_correction_payload(correction or {})
        if promote_to_memory and memory_candidate_type not in MEMORY_CANDIDATE_TYPES:
            raise ValueError("unsupported_memory_candidate_type")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_items WHERE tenant_id = ? AND review_id = ?",
                (tenant_id, review_id),
            ).fetchone()
            if row is None:
                raise KeyError("review_not_found")
            review = _review_from_row(row)
            if review.status not in {"pending", "in_review"}:
                raise ValueError("review_already_terminal")
            terminal_status = {"approve": "approved", "reject": "rejected", "correct": "corrected", "close": "closed"}[decision]
            decision_record = HumanDecision(
                decision_id=new_id("decision"),
                review_id=review_id,
                tenant_id=tenant_id,
                decision=decision,
                correction_overlay=safe_correction,
                reason_length=len(reason or ""),
                promote_to_memory=promote_to_memory,
                memory_candidate_type=memory_candidate_type if promote_to_memory else "",
                expires_at=expires_at,
                supersedes_memory_id=supersedes_memory_id,
                created_by=created_by,
            )
            conn.execute(
                """
                INSERT INTO human_decisions (
                    decision_id, review_id, tenant_id, decision, correction_json,
                    reason_length, promote_to_memory, memory_candidate_type,
                    expires_at, supersedes_memory_id, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _decision_values(decision_record),
            )
            decided_at = utc_text()
            conn.execute(
                """
                UPDATE review_items
                SET status = ?, decided_at = ?, decision_id = ?
                WHERE tenant_id = ? AND review_id = ?
                """,
                (terminal_status, decided_at, decision_record.decision_id, tenant_id, review_id),
            )
            memory_candidate = None
            if promote_to_memory:
                feedback = _feedback_from_row(
                    conn.execute(
                        "SELECT * FROM feedback_records WHERE tenant_id = ? AND feedback_id = ?",
                        (tenant_id, review.feedback_id),
                    ).fetchone()
                )
                if not _feedback_can_promote(feedback):
                    raise ValueError("feedback_not_promotable_to_memory")
                safe_content = _memory_content_from_feedback(feedback, decision_record)
                memory_candidate = MemoryCandidate(
                    memory_candidate_id=new_id("memory_candidate"),
                    tenant_id=tenant_id,
                    memory_type=memory_candidate_type,
                    safe_content=safe_content,
                    source_feedback_id=feedback.feedback_id,
                    source_decision_id=decision_record.decision_id,
                    source_task_id=feedback.task_id,
                    candidate_id=feedback.candidate_id,
                    tags=[memory_candidate_type, feedback.feedback_type],
                    importance=0.6,
                    provenance={
                        "source": "human_decision",
                        "feedback_type": feedback.feedback_type,
                        "task_id": feedback.task_id,
                        "candidate_id": feedback.candidate_id,
                        "resume_version_id": feedback.resume_version_id,
                        "profile_version_id": feedback.profile_version_id,
                        "summary_only": True,
                    },
                    expires_at=expires_at,
                    supersedes_memory_id=supersedes_memory_id,
                )
                conn.execute(
                    """
                    INSERT INTO memory_candidates (
                        memory_candidate_id, tenant_id, memory_type, safe_content,
                        source_feedback_id, source_decision_id, source_task_id,
                        candidate_id, tags_json, importance, provenance_json,
                        status, expires_at, supersedes_memory_id, created_at,
                        decided_at, memory_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _memory_candidate_values(memory_candidate),
                )
            updated = _review_from_row(
                conn.execute(
                    "SELECT * FROM review_items WHERE tenant_id = ? AND review_id = ?",
                    (tenant_id, review_id),
                ).fetchone()
            )
        return decision_record, updated, memory_candidate

    def get_memory_candidate(self, *, tenant_id: str, memory_candidate_id: str) -> MemoryCandidate:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_candidates WHERE tenant_id = ? AND memory_candidate_id = ?",
                (tenant_id, memory_candidate_id),
            ).fetchone()
        if row is None:
            raise KeyError("memory_candidate_not_found")
        return _memory_candidate_from_row(row)

    def list_memory_candidates(self, *, tenant_id: str, status: str = "", limit: int = 100) -> List[MemoryCandidate]:
        query = "SELECT * FROM memory_candidates WHERE tenant_id = ?"
        params: List[Any] = [tenant_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_memory_candidate_from_row(row) for row in rows]

    def approve_memory_candidate(self, *, tenant_id: str, memory_candidate_id: str) -> GovernedMemory:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_candidates WHERE tenant_id = ? AND memory_candidate_id = ?",
                (tenant_id, memory_candidate_id),
            ).fetchone()
            if row is None:
                raise KeyError("memory_candidate_not_found")
            candidate = _memory_candidate_from_row(row)
            if candidate.status != "pending_review":
                raise ValueError("memory_candidate_not_pending")
            version = self._next_memory_version(conn, tenant_id)
            memory = GovernedMemory(
                memory_id=new_id("memory"),
                tenant_id=tenant_id,
                memory_type=candidate.memory_type,
                safe_content=candidate.safe_content,
                source_memory_candidate_id=candidate.memory_candidate_id,
                source_feedback_id=candidate.source_feedback_id,
                source_decision_id=candidate.source_decision_id,
                source_task_id=candidate.source_task_id,
                candidate_id=candidate.candidate_id,
                tags=list(candidate.tags),
                importance=candidate.importance,
                version=version,
                expires_at=candidate.expires_at,
                supersedes_memory_id=candidate.supersedes_memory_id,
                metadata={"summary_only": True, "governed_memory": True},
            )
            if memory.supersedes_memory_id:
                conn.execute(
                    """
                    UPDATE governed_memories
                    SET status = 'superseded'
                    WHERE tenant_id = ? AND memory_id = ?
                    """,
                    (tenant_id, memory.supersedes_memory_id),
                )
            conn.execute(
                """
                INSERT INTO governed_memories (
                    memory_id, tenant_id, memory_type, safe_content,
                    source_memory_candidate_id, source_feedback_id,
                    source_decision_id, source_task_id, candidate_id,
                    tags_json, importance, version, status, expires_at,
                    supersedes_memory_id, revoked_at, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _governed_memory_values(memory),
            )
            decided_at = utc_text()
            conn.execute(
                """
                UPDATE memory_candidates
                SET status = 'approved', decided_at = ?, memory_id = ?
                WHERE tenant_id = ? AND memory_candidate_id = ?
                """,
                (decided_at, memory.memory_id, tenant_id, memory_candidate_id),
            )
        return memory

    def reject_memory_candidate(self, *, tenant_id: str, memory_candidate_id: str) -> MemoryCandidate:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_candidates WHERE tenant_id = ? AND memory_candidate_id = ?",
                (tenant_id, memory_candidate_id),
            ).fetchone()
            if row is None:
                raise KeyError("memory_candidate_not_found")
            candidate = _memory_candidate_from_row(row)
            if candidate.status != "pending_review":
                raise ValueError("memory_candidate_not_pending")
            decided_at = utc_text()
            conn.execute(
                """
                UPDATE memory_candidates
                SET status = 'rejected', decided_at = ?
                WHERE tenant_id = ? AND memory_candidate_id = ?
                """,
                (decided_at, tenant_id, memory_candidate_id),
            )
        return self.get_memory_candidate(tenant_id=tenant_id, memory_candidate_id=memory_candidate_id)

    def get_memory(self, *, tenant_id: str, memory_id: str) -> GovernedMemory:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM governed_memories WHERE tenant_id = ? AND memory_id = ?",
                (tenant_id, memory_id),
            ).fetchone()
        if row is None:
            raise KeyError("memory_not_found")
        return _governed_memory_from_row(row)

    def list_memories(self, *, tenant_id: str, status: str = "", memory_type: str = "", limit: int = 100) -> List[GovernedMemory]:
        query = "SELECT * FROM governed_memories WHERE tenant_id = ?"
        params: List[Any] = [tenant_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_governed_memory_from_row(row) for row in rows]

    def list_active_memory_records(self, *, tenant_id: str) -> List[MemoryRecord]:
        now = utc_text()
        memories = [
            memory
            for memory in self.list_memories(tenant_id=tenant_id, status="active", limit=200)
            if not memory.expires_at or memory.expires_at > now
        ]
        return [memory.to_memory_record() for memory in memories]

    def revoke_memory(self, *, tenant_id: str, memory_id: str) -> GovernedMemory:
        self.get_memory(tenant_id=tenant_id, memory_id=memory_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE governed_memories
                SET status = 'revoked', revoked_at = ?
                WHERE tenant_id = ? AND memory_id = ?
                """,
                (utc_text(), tenant_id, memory_id),
            )
        return self.get_memory(tenant_id=tenant_id, memory_id=memory_id)

    def metrics_summary(self, *, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        where = "WHERE tenant_id = ?" if tenant_id else ""
        params = [tenant_id] if tenant_id else []
        with self._connect() as conn:
            feedback_rows = conn.execute(f"SELECT feedback_type, COUNT(*) AS c FROM feedback_records {where} GROUP BY feedback_type", params).fetchall()
            review_rows = conn.execute(f"SELECT status, COUNT(*) AS c FROM review_items {where} GROUP BY status", params).fetchall()
            decision_rows = conn.execute(f"SELECT decision, COUNT(*) AS c FROM human_decisions {where} GROUP BY decision", params).fetchall()
            memory_candidate_count = conn.execute(f"SELECT COUNT(*) AS c FROM memory_candidates {where}", params).fetchone()["c"]
            memory_rejection_count = conn.execute(
                f"SELECT COUNT(*) AS c FROM memory_candidates {where} {'AND' if where else 'WHERE'} status = 'rejected'",
                params,
            ).fetchone()["c"]
            activation_total = conn.execute(f"SELECT COUNT(*) AS c FROM governed_memories {where}", params).fetchone()["c"]
            activation_count = conn.execute(
                f"SELECT COUNT(*) AS c FROM governed_memories {where} {'AND' if where else 'WHERE'} status = 'active'",
                params,
            ).fetchone()["c"]
            revoked_count = conn.execute(
                f"SELECT COUNT(*) AS c FROM governed_memories {where} {'AND' if where else 'WHERE'} status = 'revoked'",
                params,
            ).fetchone()["c"]
        feedback_by_type = {str(row["feedback_type"]): int(row["c"]) for row in feedback_rows}
        reviews_by_status = {str(row["status"]): int(row["c"]) for row in review_rows}
        decisions_by_type = {str(row["decision"]): int(row["c"]) for row in decision_rows}
        total_reviews = sum(reviews_by_status.values())
        completed_decisions = sum(decisions_by_type.values())
        return {
            "feedback_submitted_count": sum(feedback_by_type.values()),
            "feedback_by_type": feedback_by_type,
            "review_created_count": total_reviews,
            "review_pending_count": int(reviews_by_status.get("pending", 0)),
            "review_decision_count": completed_decisions,
            "review_approval_rate": _rate(decisions_by_type.get("approve", 0), completed_decisions),
            "review_rejection_rate": _rate(decisions_by_type.get("reject", 0), completed_decisions),
            "review_correction_rate": _rate(decisions_by_type.get("correct", 0), completed_decisions),
            "human_intervention_rate": _rate(sum(feedback_by_type.values()) + completed_decisions, max(1, total_reviews + sum(feedback_by_type.values()))),
            "review_latency_ms": {"count": completed_decisions, "p50": 0.0, "p95": 0.0, "summary_only": True},
            "memory_candidate_count": int(memory_candidate_count),
            "memory_activation_count": int(activation_total),
            "memory_active_count": int(activation_count),
            "memory_rejection_count": int(memory_rejection_count),
            "memory_revocation_count": int(revoked_count),
            "summary_only": True,
        }

    @staticmethod
    def _next_memory_version(conn, tenant_id: str) -> int:
        row = conn.execute(
            "SELECT MAX(version) AS v FROM governed_memories WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchone()
        return int(row["v"] or 0) + 1


def validate_correction_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        field = str(key)
        if field not in CORRECTION_FIELDS:
            raise ValueError("correction_field_not_allowed")
        text = str(value)
        if len(text) > 500:
            raise ValueError("correction_value_too_long")
        safe[field] = text
    return safe


def should_create_review_for_claim_verification(status: str, critical_unsupported_count: int = 0) -> bool:
    return str(status) == "review_required" or int(critical_unsupported_count or 0) > 0


def _feedback_values(record: FeedbackRecord):
    return (
        record.feedback_id,
        record.tenant_id,
        record.task_id,
        record.session_id,
        record.candidate_id,
        record.report_id,
        record.resume_version_id,
        record.profile_version_id,
        record.feedback_type,
        record.rating,
        int(record.comment_length),
        _json(record.correction_payload),
        _json(record.claim_ids),
        record.created_by,
        record.created_at,
        record.source,
        _json(record.metadata),
    )


def _review_values(item: ReviewItem):
    return (
        item.review_id,
        item.tenant_id,
        item.review_type,
        item.task_id,
        item.candidate_id,
        item.report_id,
        item.feedback_id,
        item.claim_verification_status,
        item.priority,
        item.status,
        _json(item.reason_codes),
        item.assigned_to,
        item.created_at,
        item.decided_at,
        item.decision_id,
        _json(item.metadata),
    )


def _decision_values(item: HumanDecision):
    return (
        item.decision_id,
        item.review_id,
        item.tenant_id,
        item.decision,
        _json(item.correction_overlay),
        item.reason_length,
        1 if item.promote_to_memory else 0,
        item.memory_candidate_type,
        item.expires_at,
        item.supersedes_memory_id,
        item.created_by,
        item.created_at,
    )


def _memory_candidate_values(item: MemoryCandidate):
    return (
        item.memory_candidate_id,
        item.tenant_id,
        item.memory_type,
        item.safe_content,
        item.source_feedback_id,
        item.source_decision_id,
        item.source_task_id,
        item.candidate_id,
        _json(item.tags),
        item.importance,
        _json(item.provenance),
        item.status,
        item.expires_at,
        item.supersedes_memory_id,
        item.created_at,
        item.decided_at,
        item.memory_id,
    )


def _governed_memory_values(item: GovernedMemory):
    return (
        item.memory_id,
        item.tenant_id,
        item.memory_type,
        item.safe_content,
        item.source_memory_candidate_id,
        item.source_feedback_id,
        item.source_decision_id,
        item.source_task_id,
        item.candidate_id,
        _json(item.tags),
        item.importance,
        item.version,
        item.status,
        item.expires_at,
        item.supersedes_memory_id,
        item.revoked_at,
        item.created_at,
        _json(item.metadata),
    )


def _feedback_from_row(row) -> FeedbackRecord:
    if row is None:
        raise KeyError("feedback_not_found")
    return FeedbackRecord(
        feedback_id=row["feedback_id"],
        tenant_id=row["tenant_id"],
        task_id=row["task_id"],
        session_id=row["session_id"] or "",
        candidate_id=row["candidate_id"] or "",
        report_id=row["report_id"] or "",
        resume_version_id=row["resume_version_id"] or "",
        profile_version_id=row["profile_version_id"] or "",
        feedback_type=row["feedback_type"],
        rating=row["rating"],
        comment_length=int(row["comment_length"] or 0),
        correction_payload=_load_json(row["correction_json"], {}),
        claim_ids=_load_json(row["claim_ids_json"], []),
        created_by=row["created_by"] or "api",
        created_at=row["created_at"],
        source=row["source"] or "api",
        metadata=_load_json(row["metadata_json"], {}),
    )


def _review_from_row(row) -> ReviewItem:
    if row is None:
        raise KeyError("review_not_found")
    return ReviewItem(
        review_id=row["review_id"],
        tenant_id=row["tenant_id"],
        review_type=row["review_type"],
        task_id=row["task_id"],
        candidate_id=row["candidate_id"] or "",
        report_id=row["report_id"] or "",
        feedback_id=row["feedback_id"] or "",
        claim_verification_status=row["claim_verification_status"] or "",
        priority=row["priority"] or "normal",
        status=row["status"],
        reason_codes=_load_json(row["reason_codes_json"], []),
        assigned_to=row["assigned_to"] or "",
        created_at=row["created_at"],
        decided_at=row["decided_at"] or "",
        decision_id=row["decision_id"] or "",
        metadata=_load_json(row["metadata_json"], {}),
    )


def _memory_candidate_from_row(row) -> MemoryCandidate:
    if row is None:
        raise KeyError("memory_candidate_not_found")
    return MemoryCandidate(
        memory_candidate_id=row["memory_candidate_id"],
        tenant_id=row["tenant_id"],
        memory_type=row["memory_type"],
        safe_content=row["safe_content"],
        source_feedback_id=row["source_feedback_id"],
        source_decision_id=row["source_decision_id"],
        source_task_id=row["source_task_id"],
        candidate_id=row["candidate_id"] or "",
        tags=_load_json(row["tags_json"], []),
        importance=float(row["importance"] or 0.5),
        provenance=_load_json(row["provenance_json"], {}),
        status=row["status"],
        expires_at=row["expires_at"] or "",
        supersedes_memory_id=row["supersedes_memory_id"] or "",
        created_at=row["created_at"],
        decided_at=row["decided_at"] or "",
        memory_id=row["memory_id"] or "",
    )


def _governed_memory_from_row(row) -> GovernedMemory:
    if row is None:
        raise KeyError("memory_not_found")
    return GovernedMemory(
        memory_id=row["memory_id"],
        tenant_id=row["tenant_id"],
        memory_type=row["memory_type"],
        safe_content=row["safe_content"],
        source_memory_candidate_id=row["source_memory_candidate_id"],
        source_feedback_id=row["source_feedback_id"],
        source_decision_id=row["source_decision_id"],
        source_task_id=row["source_task_id"],
        candidate_id=row["candidate_id"] or "",
        tags=_load_json(row["tags_json"], []),
        importance=float(row["importance"] or 0.5),
        version=int(row["version"] or 1),
        status=row["status"],
        expires_at=row["expires_at"] or "",
        supersedes_memory_id=row["supersedes_memory_id"] or "",
        revoked_at=row["revoked_at"] or "",
        created_at=row["created_at"],
        metadata=_load_json(row["metadata_json"], {}),
    )


def _feedback_can_promote(feedback: FeedbackRecord) -> bool:
    return feedback.feedback_type in MEMORY_PROMOTION_FEEDBACK_TYPES and feedback.feedback_type != "comment"


def _memory_content_from_feedback(feedback: FeedbackRecord, decision: HumanDecision) -> str:
    if decision.correction_overlay:
        parts = [f"{key}: {value}" for key, value in sorted(decision.correction_overlay.items())]
        return _safe_text("；".join(parts), 500)
    if feedback.claim_ids:
        return _safe_text(f"Human reviewed claims: {', '.join(feedback.claim_ids[:5])}; feedback={feedback.feedback_type}", 500)
    return _safe_text(f"Human feedback type={feedback.feedback_type}; candidate={feedback.candidate_id or 'task'}", 500)


def _record_memory_type(memory_type: str) -> str:
    if memory_type in {"tenant_preference", "matching_rule"}:
        return MemoryType.PROCEDURAL.value
    if memory_type == "task_experience":
        return MemoryType.EPISODIC.value
    return MemoryType.SEMANTIC.value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _load_json(value: Optional[str], default):
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _safe_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in dict(metadata or {}).items():
        text = str(value)
        if any(secret in str(key).lower() for secret in ("key", "token", "secret", "authorization")):
            safe[str(key)] = "<redacted>"
        elif len(text) > 240:
            safe[str(key)] = f"<present; length={len(text)}>"
        elif "/" in text and ("storage" in text or "Users" in text):
            safe[str(key)] = "<path-redacted>"
        else:
            safe[str(key)] = value
    safe["summary_only"] = True
    return safe


def _safe_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    return text[: max(0, int(max_chars))]


def _rate(numerator: Any, denominator: Any) -> float:
    try:
        den = float(denominator or 0)
        if den <= 0:
            return 0.0
        return round(float(numerator or 0) / den, 6)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
