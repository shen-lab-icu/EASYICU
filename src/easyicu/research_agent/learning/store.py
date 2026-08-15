"""Permissioned, digest-bound memory storage for the research agent.

Memory is data, not prompt authority.  This module deliberately separates
unreviewed run lessons from reviewed knowledge and promoted lessons.  Reading
from a namespace always requires an explicit access policy; canonical runs can
never read quarantine memory.
"""

from __future__ import annotations
from ..authority.filesystem import publish_write_once_bytes

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MEMORY_OBJECT_SCHEMA = "easyicu.memory_object/1"
MEMORY_PROMOTION_RECEIPT_SCHEMA = "easyicu.memory_promotion_receipt/1"

MemoryReviewStatus = Literal[
    "preference",
    "reviewed",
    "quarantined",
    "promoted",
    "runtime",
]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of a memory payload."""

    return hashlib.sha256(_canonical_json_bytes(dict(payload))).hexdigest()


class MemoryReviewAttestation(BaseModel):
    """Human or regression-suite approval bound to exact memory bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: str = Field(min_length=1, max_length=80)
    review_scope: str = Field(min_length=1, max_length=500)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_refs: tuple[str, ...] = ()


class MemoryObject(BaseModel):
    """One immutable memory object with explicit scientific authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.memory_object/1"] = MEMORY_OBJECT_SCHEMA
    namespace: str = Field(min_length=3, max_length=240)
    key: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")
    version: str = Field(pattern=r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
    payload: dict[str, Any]
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = Field(min_length=1, max_length=500)
    producer: str = Field(min_length=1, max_length=200)
    review_status: MemoryReviewStatus
    applicable_scope: tuple[str, ...] = ()
    invalidation: tuple[str, ...] = ()
    profile_ref: str | None = Field(default=None, max_length=200)
    created_at: str = Field(min_length=1, max_length=80)
    attestation: MemoryReviewAttestation | None = None

    @field_validator("namespace")
    @classmethod
    def _safe_namespace(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+(?:/[A-Za-z0-9_.:-]+)*", value):
            raise ValueError("memory namespace contains an unsafe path component")
        return value

    @model_validator(mode="after")
    def _validate_authority(self) -> "MemoryObject":
        if payload_sha256(self.payload) != self.payload_sha256:
            raise ValueError("memory payload digest mismatch")

        root = self.namespace.split("/", 1)[0]
        expected_status = {
            "preferences": "preference",
            "reviewed_knowledge": "reviewed",
            "run_lessons": "quarantined",
            "promoted_lessons": "promoted",
            "runtime": "runtime",
        }.get(root)
        if expected_status is None:
            raise ValueError("unknown memory namespace root")
        if self.review_status != expected_status:
            raise ValueError("memory review status does not match its namespace")
        if root == "run_lessons" and "/quarantine/" not in (self.namespace + "/"):
            raise ValueError("run-derived lessons must remain in quarantine")

        reviewed = self.review_status in {"reviewed", "promoted"}
        if reviewed:
            if self.attestation is None:
                raise ValueError("reviewed memory requires a review attestation")
            if self.attestation.payload_sha256 != self.payload_sha256:
                raise ValueError("memory attestation does not bind the payload")
            if not self.profile_ref:
                raise ValueError("reviewed memory requires a profile binding")
        elif self.attestation is not None:
            raise ValueError("unreviewed memory cannot carry an attestation")
        return self

    @classmethod
    def create(
        cls,
        *,
        namespace: str,
        key: str,
        version: str,
        payload: Mapping[str, Any],
        source: str,
        producer: str,
        review_status: MemoryReviewStatus,
        created_at: str,
        applicable_scope: Iterable[str] = (),
        invalidation: Iterable[str] = (),
        profile_ref: str | None = None,
        attestation: MemoryReviewAttestation | None = None,
    ) -> "MemoryObject":
        payload_dict = dict(payload)
        return cls(
            namespace=namespace,
            key=key,
            version=version,
            payload=payload_dict,
            payload_sha256=payload_sha256(payload_dict),
            source=source,
            producer=producer,
            review_status=review_status,
            applicable_scope=tuple(applicable_scope),
            invalidation=tuple(invalidation),
            profile_ref=profile_ref,
            created_at=created_at,
            attestation=attestation,
        )


class MemoryPromotionReceipt(BaseModel):
    """Proof that a quarantined lesson was explicitly promoted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.memory_promotion_receipt/1"] = (
        MEMORY_PROMOTION_RECEIPT_SCHEMA
    )
    source_namespace: str
    source_key: str
    source_version: str
    source_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_namespace: str
    target_key: str
    target_version: str
    profile_ref: str
    attestation: MemoryReviewAttestation


class MemoryStore(Protocol):
    def put(self, memory: MemoryObject) -> None: ...

    def get(self, namespace: str, key: str, version: str) -> MemoryObject | None: ...

    def list(self, namespace: str) -> tuple[MemoryObject, ...]: ...


class FileSystemMemoryStore:
    """Small write-once reference backend using canonical JSON files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _validate_coordinates(namespace: str, key: str, version: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+(?:/[A-Za-z0-9_.:-]+)*", namespace):
            raise ValueError("unsafe memory namespace")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", key):
            raise ValueError("unsafe memory key")
        if not re.fullmatch(r"[1-9][0-9]*\.[0-9]+\.[0-9]+", version):
            raise ValueError("unsafe memory version")

    def put(self, memory: MemoryObject) -> None:
        memory = MemoryObject.model_validate(memory.model_dump(mode="python"))
        self._validate_coordinates(memory.namespace, memory.key, memory.version)
        path = self.root.joinpath(
            *memory.namespace.split("/"), memory.key, f"{memory.version}.json"
        )
        payload = memory.model_dump_json(indent=2).encode("utf-8") + b"\n"
        publish_write_once_bytes(
            path,
            payload,
            temp_prefix=".memory-",
            conflict_error=FileExistsError,
            conflict_message="memory object is immutable and already exists",
        )

    def get(self, namespace: str, key: str, version: str) -> MemoryObject | None:
        self._validate_coordinates(namespace, key, version)
        path = self.root.joinpath(*namespace.split("/"), key, f"{version}.json")
        if not path.exists():
            return None
        return MemoryObject.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, namespace: str) -> tuple[MemoryObject, ...]:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+(?:/[A-Za-z0-9_.:-]+)*", namespace):
            raise ValueError("unsafe memory namespace")
        directory = self.root.joinpath(*namespace.split("/"))
        if not directory.exists():
            return ()
        return tuple(
            MemoryObject.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*/*.json"))
        )


class LangGraphMemoryStoreAdapter:
    """Optional adapter; EasyICU schemas remain authoritative over LangGraph."""

    def __init__(self, store: Any) -> None:
        self.store = store

    @staticmethod
    def _namespace(namespace: str) -> tuple[str, ...]:
        return ("easyicu", "memory", *namespace.split("/"))

    @staticmethod
    def _storage_key(key: str, version: str) -> str:
        return f"{key}@{version}"

    def put(self, memory: MemoryObject) -> None:
        memory = MemoryObject.model_validate(memory.model_dump(mode="python"))
        self.store.put(
            self._namespace(memory.namespace),
            self._storage_key(memory.key, memory.version),
            memory.model_dump(mode="json"),
        )

    def get(self, namespace: str, key: str, version: str) -> MemoryObject | None:
        item = self.store.get(
            self._namespace(namespace), self._storage_key(key, version)
        )
        if item is None:
            return None
        value = getattr(item, "value", item)
        return MemoryObject.model_validate(value)

    def list(self, namespace: str) -> tuple[MemoryObject, ...]:
        items = self.store.search(self._namespace(namespace))
        return tuple(
            MemoryObject.model_validate(getattr(item, "value", item)) for item in items
        )


class MemoryAccessPolicy(BaseModel):
    """Explicit memory read authority for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical: bool = True
    profile_ref: str | None = None
    allowed_namespaces: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical_excludes_quarantine(self) -> "MemoryAccessPolicy":
        if self.canonical:
            invalid = tuple(
                namespace
                for namespace in self.allowed_namespaces
                if not namespace.startswith(
                    ("reviewed_knowledge/", "promoted_lessons/")
                )
            )
            if invalid:
                raise ValueError(
                    "canonical runs may read reviewed/promoted memory only; "
                    f"disallowed namespaces={invalid!r}"
                )
        return self


def select_memory(
    store: MemoryStore,
    *,
    policy: MemoryAccessPolicy,
) -> tuple[MemoryObject, ...]:
    """Select only explicitly authorized, profile-compatible memory."""

    selected: list[MemoryObject] = []
    for namespace in policy.allowed_namespaces:
        for memory in store.list(namespace):
            if policy.canonical and memory.review_status not in {
                "reviewed",
                "promoted",
            }:
                continue
            if memory.review_status in {"reviewed", "promoted"}:
                if not policy.profile_ref or memory.profile_ref != policy.profile_ref:
                    continue
            selected.append(memory)
    return tuple(
        sorted(selected, key=lambda item: (item.namespace, item.key, item.version))
    )


def promote_quarantined_memory(
    store: MemoryStore,
    *,
    source: MemoryObject,
    target_version: str,
    profile_ref: str,
    attestation: MemoryReviewAttestation,
) -> tuple[MemoryObject, MemoryPromotionReceipt]:
    """Promote an exact quarantined payload after external approval."""

    if source.review_status != "quarantined":
        raise ValueError("only quarantined run lessons may be promoted")
    if attestation.payload_sha256 != source.payload_sha256:
        raise ValueError("promotion attestation does not bind the source payload")
    target_namespace = f"promoted_lessons/{target_version}"
    promoted = MemoryObject.create(
        namespace=target_namespace,
        key=source.key,
        version=target_version,
        payload=source.payload,
        source=f"{source.namespace}/{source.key}@{source.version}",
        producer="memory_promotion",
        review_status="promoted",
        created_at=attestation.reviewed_at,
        applicable_scope=source.applicable_scope,
        invalidation=source.invalidation,
        profile_ref=profile_ref,
        attestation=attestation,
    )
    store.put(promoted)
    receipt = MemoryPromotionReceipt(
        source_namespace=source.namespace,
        source_key=source.key,
        source_version=source.version,
        source_payload_sha256=source.payload_sha256,
        target_namespace=promoted.namespace,
        target_key=promoted.key,
        target_version=promoted.version,
        profile_ref=profile_ref,
        attestation=attestation,
    )
    return promoted, receipt


def quarantine_run_lesson(
    store: MemoryStore,
    *,
    run_id: str,
    project: str,
    payload: Mapping[str, Any],
    created_at: str,
    producer: str = "legacy_run_memory",
) -> MemoryObject:
    """Mirror a run-derived record into quarantine without prompt authority."""

    project_slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", project).strip("-")
    if not project_slug:
        project_slug = "project"
    run_slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", run_id).strip("-") or "run"
    digest = payload_sha256(payload)
    memory = MemoryObject.create(
        namespace=f"run_lessons/quarantine/{project_slug}",
        key=f"{run_slug}-{digest[:12]}",
        version="1.0.0",
        payload=payload,
        source=run_id,
        producer=producer,
        review_status="quarantined",
        created_at=created_at,
        applicable_scope=(project_slug,),
        invalidation=("code_version_change", "data_contract_change"),
    )
    store.put(memory)
    return memory
