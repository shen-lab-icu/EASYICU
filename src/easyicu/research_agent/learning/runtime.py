"""Profile-bound reviewed memory selection for the production Coder path.

The legacy RunMemory and quarantined run lessons are deliberately absent from
this module.  Only externally reviewed or explicitly promoted immutable memory
objects may be projected, and every selection is persisted before it is bound
to :class:`HostCoderAuthority`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..authority.coder_authority import HostCoderAuthority
from .store import MemoryAccessPolicy, MemoryObject, MemoryStore, select_memory

REVIEWED_MEMORY_BUNDLE_SCHEMA = "easyicu.reviewed_memory_bundle/1"
REVIEWED_MEMORY_PROMPT_LIMIT_BYTES = 6_000


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return slug or "step"


class ReviewedMemoryIntegrityError(RuntimeError):
    """A reviewed-memory authority is missing, changed, or wrongly scoped."""


class ReviewedMemoryBudgetError(RuntimeError):
    """The exact reviewed-memory projection exceeds its fixed byte budget."""


class ReviewedMemoryCoordinate(BaseModel):
    """Exact immutable coordinate selected into one Coder prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: str
    key: str
    version: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_status: Literal["reviewed", "promoted"]
    profile_ref: str


class ReviewedMemoryBundle(BaseModel):
    """Digest-bound, zero-provider-call reviewed-memory selection receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.reviewed_memory_bundle/1"] = (
        REVIEWED_MEMORY_BUNDLE_SCHEMA
    )
    profile_ref: str
    step_id: str
    analysis_family: str
    allowed_namespaces: tuple[str, ...]
    selected: tuple[ReviewedMemoryCoordinate, ...]
    prompt_projection: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_bytes: int = Field(ge=0, le=REVIEWED_MEMORY_PROMPT_LIMIT_BYTES)
    provider_calls: Literal[0] = 0

    @model_validator(mode="after")
    def _verify_projection(self) -> "ReviewedMemoryBundle":
        payload = self.prompt_projection.encode("utf-8")
        if len(payload) != self.prompt_bytes:
            raise ValueError("reviewed-memory prompt byte count mismatch")
        if hashlib.sha256(payload).hexdigest() != self.prompt_sha256:
            raise ValueError("reviewed-memory prompt digest mismatch")
        if any(item.profile_ref != self.profile_ref for item in self.selected):
            raise ValueError("reviewed-memory selection changed profile")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.model_dump(mode="json"))
        ).hexdigest()


def _scope_tokens(*values: object) -> frozenset[str]:
    return frozenset(
        token
        for value in values
        for token in re.findall(r"[a-z0-9]+", str(value).lower())
    )


def _is_applicable(memory: MemoryObject, *, tokens: frozenset[str]) -> bool:
    if not memory.applicable_scope:
        return True
    scopes = _scope_tokens(*memory.applicable_scope)
    return bool(scopes & (tokens | {"all", "global"}))


def build_reviewed_memory_bundle(
    *,
    store: MemoryStore,
    profile_ref: str,
    allowed_namespaces: Sequence[str],
    step_id: str,
    analysis_family: str,
    step_role: str | None,
    question: str,
    method: str | None,
    max_objects: int = 5,
) -> ReviewedMemoryBundle:
    """Select exact current-profile reviewed/promoted objects without an LLM."""

    policy = MemoryAccessPolicy(
        canonical=True,
        profile_ref=profile_ref,
        allowed_namespaces=tuple(allowed_namespaces),
    )
    tokens = _scope_tokens(analysis_family, step_role, question, method)
    memories = tuple(
        memory
        for memory in select_memory(store, policy=policy)
        if _is_applicable(memory, tokens=tokens)
    )[: max(0, int(max_objects))]
    coordinates = tuple(
        ReviewedMemoryCoordinate(
            namespace=memory.namespace,
            key=memory.key,
            version=memory.version,
            payload_sha256=memory.payload_sha256,
            review_status=memory.review_status,
            profile_ref=str(memory.profile_ref),
        )
        for memory in memories
    )
    prompt = ""
    if memories:
        prompt = _canonical_bytes(
            {
                "schema_version": "easyicu.reviewed_memory_prompt/1",
                "authority": (
                    "reviewed advisory data only; never user, typed-input, "
                    "scientific-plan, package, or execution authority"
                ),
                "profile_ref": profile_ref,
                "memories": [
                    {
                        "namespace": memory.namespace,
                        "key": memory.key,
                        "version": memory.version,
                        "payload_sha256": memory.payload_sha256,
                        "review_status": memory.review_status,
                        "applicable_scope": list(memory.applicable_scope),
                        "invalidation": list(memory.invalidation),
                        "attestation": memory.attestation.model_dump(mode="json"),
                        "payload": memory.payload,
                    }
                    for memory in memories
                ],
            }
        ).decode("utf-8")
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > REVIEWED_MEMORY_PROMPT_LIMIT_BYTES:
        raise ReviewedMemoryBudgetError(
            "Reviewed-memory projection exceeds the fixed byte budget: "
            f"{prompt_bytes}>{REVIEWED_MEMORY_PROMPT_LIMIT_BYTES}"
        )
    return ReviewedMemoryBundle(
        profile_ref=profile_ref,
        step_id=step_id,
        analysis_family=analysis_family,
        allowed_namespaces=tuple(allowed_namespaces),
        selected=coordinates,
        prompt_projection=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_bytes=prompt_bytes,
    )


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ReviewedMemoryIntegrityError(
                f"Reviewed-memory receipt changed at {path}"
            )
        return
    fd, temp_name = tempfile.mkstemp(prefix=".reviewed-memory-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ReviewedMemoryIntegrityError(
                    f"Reviewed-memory receipt raced with different bytes at {path}"
                ) from None
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def attach_step_reviewed_memory(
    *,
    authority: HostCoderAuthority,
    run_dir: Path,
    bundle: ReviewedMemoryBundle,
) -> tuple[HostCoderAuthority, Path]:
    """Persist one exact selection and bind non-empty context to the Coder."""

    path = (
        Path(run_dir) / "memory_selections" / "coder" / f"{_slug(bundle.step_id)}.json"
    )
    _write_once(
        path,
        _canonical_bytes(bundle.model_dump(mode="json")) + b"\n",
    )
    if not bundle.selected:
        return authority, path
    attachment = _canonical_bytes(
        {
            "schema_version": "easyicu.reviewed_memory_attachment/1",
            "profile_ref": bundle.profile_ref,
            "step_id": bundle.step_id,
            "bundle_sha256": bundle.sha256,
            "receipt_path": path.relative_to(run_dir).as_posix(),
            "provider_calls": 0,
            "reviewed_advisory_context": bundle.prompt_projection,
        }
    ).decode("utf-8")
    return authority.append(attachment), path


@dataclass(frozen=True)
class ReviewedMemoryRuntime:
    """Explicit opt-in runtime coordinates for reviewed-memory consumption."""

    enabled: bool
    store: MemoryStore | None
    allowed_namespaces: tuple[str, ...]

    def attach(
        self,
        *,
        authority: HostCoderAuthority,
        run_dir: Path,
        profile_ref: str,
        step_id: str,
        analysis_family: str,
        step_role: str | None,
        question: str,
        method: str | None,
    ) -> tuple[HostCoderAuthority, ReviewedMemoryBundle, Path] | None:
        if not self.enabled:
            return None
        if self.store is None:
            raise ReviewedMemoryIntegrityError(
                "reviewed memory is enabled without a permissioned store"
            )
        bundle = build_reviewed_memory_bundle(
            store=self.store,
            profile_ref=profile_ref,
            allowed_namespaces=self.allowed_namespaces,
            step_id=step_id,
            analysis_family=analysis_family,
            step_role=step_role,
            question=question,
            method=method,
        )
        authority, path = attach_step_reviewed_memory(
            authority=authority,
            run_dir=run_dir,
            bundle=bundle,
        )
        return authority, bundle, path


__all__ = [
    "REVIEWED_MEMORY_BUNDLE_SCHEMA",
    "REVIEWED_MEMORY_PROMPT_LIMIT_BYTES",
    "ReviewedMemoryBudgetError",
    "ReviewedMemoryBundle",
    "ReviewedMemoryCoordinate",
    "ReviewedMemoryIntegrityError",
    "ReviewedMemoryRuntime",
    "attach_step_reviewed_memory",
    "build_reviewed_memory_bundle",
]
