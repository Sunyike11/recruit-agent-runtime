from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.memory.eligibility import MemoryContextEligibilityPolicy
from src.memory.governance import MemoryGovernancePolicy
from src.memory.models import MemoryRecord, MemoryType
from src.memory.store import MemorySQLiteStore
from src.skills.memory_context_adapter import build_shadow_workflow_memory_context


@dataclass
class RuntimeMemoryContextConfig:
    enabled: bool = False
    memory_types: Optional[List[str]] = None
    source_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_importance: float = 0.0
    max_items: int = 5
    max_chars: int = 1200
    require_governance: bool = True
    summary_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeMemoryContextResult:
    enabled: bool
    provided: bool
    eligible_count: int = 0
    denied_count: int = 0
    requires_review_count: int = 0
    rendered_char_count: int = 0
    governance_applied: bool = False
    revoked_filtered_count: int = 0
    expired_filtered_count: int = 0
    superseded_filtered_count: int = 0
    memory_context_preview: Any = None
    preview_text: str = ""
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "provided": bool(self.provided),
            "memory_source": str(self.metadata.get("memory_source") or "none"),
            "memory_db_path_present": bool(self.metadata.get("memory_db_path_present", False)),
            "memory_store_loaded": bool(self.metadata.get("memory_store_loaded", False)),
            "memory_records_seen": int(self.metadata.get("memory_records_seen", 0) or 0),
            "eligible_count": int(self.eligible_count),
            "denied_count": int(self.denied_count),
            "requires_review_count": int(self.requires_review_count),
            "rendered_char_count": int(self.rendered_char_count),
            "governance_applied": bool(self.governance_applied),
            "revoked_filtered_count": int(self.revoked_filtered_count),
            "expired_filtered_count": int(self.expired_filtered_count),
            "superseded_filtered_count": int(self.superseded_filtered_count),
            "reason": str(self.reason or ""),
            "metadata": _safe_metadata(self.metadata),
            "summary_only": True,
        }


@dataclass
class RuntimeMemorySourceConfig:
    source: str = "none"
    memory_db_path: Optional[str] = None
    governance_db_path: Optional[str] = None
    max_items: int = 5
    max_chars: int = 1200
    memory_types: Optional[List[str]] = None
    source_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_importance: float = 0.0
    require_governance: bool = True
    summary_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context_config(self) -> RuntimeMemoryContextConfig:
        return RuntimeMemoryContextConfig(
            enabled=self.source in {"demo", "sqlite"},
            memory_types=self.memory_types,
            source_types=self.source_types,
            tags=self.tags,
            min_importance=self.min_importance,
            max_items=self.max_items,
            max_chars=self.max_chars,
            require_governance=self.require_governance,
            summary_only=self.summary_only,
            metadata={**dict(self.metadata), "memory_source": self.source},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimePersistentMemoryLoader:
    """Read-only memory source loader for explicit runtime variant paths."""

    def load_memory_records(self, config: RuntimeMemorySourceConfig) -> List[MemoryRecord]:
        if config.source == "none":
            return []
        if config.source == "demo":
            return build_demo_runtime_memory_records()
        if config.source == "sqlite":
            db_path = Path(config.memory_db_path or "")
            if not db_path.exists():
                return []
            store = MemorySQLiteStore(db_path)
            return list(store.list_memories())
        raise ValueError(f"unsupported runtime memory source: {config.source}")


def build_readonly_runtime_memory_context(
    source_config: Optional[RuntimeMemorySourceConfig] = None,
    *,
    governance_store: Optional[Any] = None,
    target_context: Optional[Mapping[str, Any]] = None,
    loader: Optional[RuntimePersistentMemoryLoader] = None,
) -> RuntimeMemoryContextResult:
    config = source_config or RuntimeMemorySourceConfig()
    source = str(config.source or "none")
    db_path_present = bool(config.memory_db_path and Path(config.memory_db_path).exists())

    if source == "none":
        return RuntimeMemoryContextResult(
            enabled=False,
            provided=False,
            reason="runtime memory source disabled",
            metadata={
                "summary_only": True,
                "read_only": True,
                "memory_source": "none",
                "memory_db_path_present": False,
                "memory_store_loaded": False,
                "memory_records_seen": 0,
            },
        )

    if source == "sqlite" and not db_path_present:
        return RuntimeMemoryContextResult(
            enabled=True,
            provided=False,
            reason="sqlite memory db unavailable",
            metadata={
                "summary_only": True,
                "read_only": True,
                "memory_source": "sqlite",
                "memory_db_path_present": False,
                "memory_store_loaded": False,
                "memory_records_seen": 0,
            },
        )

    active_loader = loader or RuntimePersistentMemoryLoader()
    records = active_loader.load_memory_records(config)
    result = build_runtime_memory_context(
        records,
        config=config.to_context_config(),
        governance_store=governance_store,
        target_context=target_context,
    )
    result.metadata.update(
        {
            "memory_source": source,
            "memory_db_path_present": db_path_present,
            "memory_store_loaded": source == "demo" or bool(source == "sqlite" and db_path_present),
            "memory_records_seen": len(records),
            "governance_db_path_present": bool(config.governance_db_path and Path(config.governance_db_path).exists()),
        }
    )
    return result


def build_runtime_memory_context(
    memory_records: Iterable[MemoryRecord],
    *,
    config: Optional[RuntimeMemoryContextConfig] = None,
    governance_store: Optional[Any] = None,
    target_context: Optional[Mapping[str, Any]] = None,
) -> RuntimeMemoryContextResult:
    runtime_config = config or RuntimeMemoryContextConfig()
    if not runtime_config.enabled:
        return RuntimeMemoryContextResult(
            enabled=False,
            provided=False,
            reason="runtime memory context disabled",
            metadata={"summary_only": True, "read_only": True},
        )

    records = _filter_records(list(memory_records or []), runtime_config)
    governance_policy = MemoryGovernancePolicy() if runtime_config.require_governance else None
    policy = MemoryContextEligibilityPolicy(
        governance_policy=governance_policy,
        governance_store=governance_store,
    )
    decisions = [policy.evaluate(record, target_context=target_context) for record in records]
    eligible_records = [record for record, decision in zip(records, decisions) if decision.eligible]
    memory_context = build_shadow_workflow_memory_context(
        records,
        eligibility_policy=policy,
        target_context=target_context,
        max_items=runtime_config.max_items,
        max_chars=runtime_config.max_chars,
    )
    preview_text = memory_context.format_for_prompt()
    provided = bool(eligible_records) and not memory_context.is_empty()
    governance_counts = _governance_counts(records, governance_store)
    return RuntimeMemoryContextResult(
        enabled=True,
        provided=provided,
        eligible_count=sum(1 for decision in decisions if decision.status == "eligible"),
        denied_count=sum(1 for decision in decisions if decision.status == "denied"),
        requires_review_count=sum(1 for decision in decisions if decision.status == "requires_review"),
        rendered_char_count=len(preview_text) if provided else 0,
        governance_applied=bool(runtime_config.require_governance),
        revoked_filtered_count=governance_counts["revoked"],
        expired_filtered_count=governance_counts["expired"],
        superseded_filtered_count=governance_counts["superseded"],
        memory_context_preview=memory_context if provided else None,
        preview_text=preview_text if provided else "",
        reason="runtime memory context built" if provided else "no eligible runtime memory context",
        metadata={
            "summary_only": True,
            "read_only": True,
            "demo_memory_context": bool(runtime_config.metadata.get("demo_memory_context", False)),
            "memory_source": str(runtime_config.metadata.get("memory_source") or "memory_records"),
            "memory_db_path_present": bool(runtime_config.metadata.get("memory_db_path_present", False)),
            "memory_store_loaded": bool(runtime_config.metadata.get("memory_store_loaded", False)),
            "memory_records_seen": int(runtime_config.metadata.get("memory_records_seen", len(records)) or 0),
            "input_count": len(records),
            "target_context_keys": sorted(str(key) for key in (target_context or {}).keys()),
        },
    )


def build_demo_runtime_memory_records():
    """Build local in-memory demo records; never writes a persistent memory store."""

    return [
        MemoryRecord(
            memory_type=MemoryType.PROCEDURAL.value,
            content="Prefer LangGraph and RAG experience when evaluating AI Agent engineering roles.",
            importance=0.6,
            tags=["runtime_demo", "agent"],
            metadata={
                "promoted_from_reflection": True,
                "dry_run": False,
                "source_reflection_id": "demo_reflection_1",
                "source_candidate_id": "demo_candidate_1",
                "approved_by": "demo_reviewer",
                "reviewer": "demo_reviewer",
                "summary_only": True,
            },
        )
    ]


def _filter_records(records: List[MemoryRecord], config: RuntimeMemoryContextConfig) -> List[MemoryRecord]:
    filtered = []
    for record in records:
        if config.memory_types is not None and record.memory_type not in set(config.memory_types):
            continue
        if config.source_types is not None and record.source_type not in set(config.source_types):
            continue
        if config.tags is not None and not set(config.tags).intersection(record.tags):
            continue
        if record.importance < float(config.min_importance or 0.0):
            continue
        filtered.append(record)
    return filtered


def _governance_counts(records: List[MemoryRecord], governance_store: Optional[Any]) -> Dict[str, int]:
    counts = {"revoked": 0, "expired": 0, "superseded": 0}
    for record in records:
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        if metadata.get("revoked") is True:
            counts["revoked"] += 1
            continue
        latest = governance_store.get_latest_record(record.memory_id) if governance_store else None
        if latest is not None and latest.status in counts:
            counts[latest.status] += 1
    return counts


def _safe_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "summary_only": True,
        "read_only": bool(metadata.get("read_only", True)),
        "demo_memory_context": bool(metadata.get("demo_memory_context", False)),
        "memory_source": str(metadata.get("memory_source") or "none"),
        "memory_db_path_present": bool(metadata.get("memory_db_path_present", False)),
        "memory_store_loaded": bool(metadata.get("memory_store_loaded", False)),
        "memory_records_seen": int(metadata.get("memory_records_seen", 0) or 0),
        "memory_ids_used": [str(item) for item in (metadata.get("memory_ids_used") or [])],
        "memory_versions_used": [int(item) for item in (metadata.get("memory_versions_used") or [])],
        "input_count": int(metadata.get("input_count", 0) or 0),
        "target_context_keys": sorted(str(key) for key in metadata.get("target_context_keys", [])),
    }
