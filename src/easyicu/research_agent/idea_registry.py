"""Preregistration ledger for idea-mining candidate hypotheses.

The registry is the fishing-control boundary for future idea-mining workflows:
candidate hypotheses must be registered and explicitly accepted before any
downstream code may turn them into executable research contexts.  This module
does not fetch literature, freeze source documents, rank candidates, or run the
analysis pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SelectionStatus = Literal["proposed", "accepted", "rejected"]


class IdeaRegistryError(RuntimeError):
    """Base class for idea-registry failures."""


class CandidateAlreadyRegisteredError(IdeaRegistryError):
    """Raised when registration would silently overwrite a candidate id."""


class CandidateNotRegisteredError(IdeaRegistryError):
    """Raised when a selection or gate references an unknown candidate."""


class CandidateNotExecutableError(IdeaRegistryError):
    """Raised when a candidate is not accepted by the human gate."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonempty(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


class CandidateRegistryEntry(BaseModel):
    """One append-only registry event for a candidate hypothesis.

    ``source_snapshot_id`` is intentionally opaque in S2.  The real
    freeze/hash semantics are owned by the later literature-source snapshot
    stage; S2 only requires a non-empty provenance handle so selection decisions
    can be audited and replayed.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_family_id: str
    candidate_id: str
    source_snapshot_id: str = Field(
        description="Opaque provenance handle; S2 does not validate hash format."
    )
    selection_status: SelectionStatus = "proposed"
    selected_by: Optional[str] = None
    selection_timestamp: str = Field(default_factory=_utc_now_iso)
    selection_reason: str = ""

    @field_validator("hypothesis_family_id", "candidate_id", "source_snapshot_id")
    @classmethod
    def _require_nonempty_id(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "field")
        return _nonempty(value, field_name)

    @field_validator("selected_by")
    @classmethod
    def _normalise_selected_by(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("selection_reason", "selection_timestamp")
    @classmethod
    def _normalise_text(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _accepted_or_rejected_requires_human_selection(
        self,
    ) -> "CandidateRegistryEntry":
        if self.selection_status in {"accepted", "rejected"}:
            if not self.selected_by:
                raise ValueError("accepted/rejected candidates require selected_by")
            if not self.selection_reason:
                raise ValueError("accepted/rejected candidates require selection_reason")
        return self


class IdeaCandidateRegistry:
    """Append-only JSON ledger for preregistered idea-mining candidates."""

    schema_version = "easyicu.idea_candidate_registry/1"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: List[CandidateRegistryEntry] = []
        if self.path.exists():
            self._load()
        else:
            self.write()

    @property
    def records(self) -> tuple[CandidateRegistryEntry, ...]:
        return tuple(self._records)

    def register_candidate(
        self,
        entry: CandidateRegistryEntry,
    ) -> CandidateRegistryEntry:
        """Append a new proposed candidate to the ledger.

        Candidate ids are immutable registry keys.  A duplicate registration is
        an error even if the payload is identical, because silently replacing a
        preregistered choice set would destroy the multiple-testing denominator.
        """

        if entry.selection_status != "proposed":
            raise IdeaRegistryError(
                "register_candidate only accepts entries with selection_status='proposed'"
            )
        if self._latest_by_candidate(entry.candidate_id) is not None:
            raise CandidateAlreadyRegisteredError(
                f"candidate_id is already registered: {entry.candidate_id}"
            )
        self._append(entry)
        return entry

    def record_selection(
        self,
        candidate_id: str,
        status: Literal["accepted", "rejected"],
        *,
        by: str,
        reason: str,
    ) -> CandidateRegistryEntry:
        """Append a human selection decision for an existing candidate."""

        if status not in {"accepted", "rejected"}:
            raise ValueError("status must be 'accepted' or 'rejected'")
        candidate_id = _nonempty(candidate_id, "candidate_id")
        prior = self._latest_by_candidate(candidate_id)
        if prior is None:
            raise CandidateNotRegisteredError(
                f"candidate_id is not registered: {candidate_id}"
            )
        entry = CandidateRegistryEntry(
            hypothesis_family_id=prior.hypothesis_family_id,
            candidate_id=prior.candidate_id,
            source_snapshot_id=prior.source_snapshot_id,
            selection_status=status,
            selected_by=by,
            selection_reason=reason,
        )
        self._append(entry)
        return entry

    def assert_executable(self, candidate_id: str) -> bool:
        """Strict human gate: only the latest ``accepted`` status may execute."""

        candidate_id = _nonempty(candidate_id, "candidate_id")
        latest = self._latest_by_candidate(candidate_id)
        if latest is None:
            raise CandidateNotRegisteredError(
                f"candidate_id is not registered: {candidate_id}"
            )
        if latest.selection_status != "accepted":
            raise CandidateNotExecutableError(
                f"candidate_id={candidate_id!r} is not executable; "
                f"latest selection_status={latest.selection_status!r}"
            )
        return True

    def family_size(self, hypothesis_family_id: str) -> int:
        """Return the preregistered family denominator for multiple testing."""

        family = _nonempty(hypothesis_family_id, "hypothesis_family_id")
        first_family_by_candidate: Dict[str, str] = {}
        for record in self._records:
            first_family_by_candidate.setdefault(
                record.candidate_id, record.hypothesis_family_id
            )
        return sum(1 for value in first_family_by_candidate.values() if value == family)

    def latest_entry(self, candidate_id: str) -> CandidateRegistryEntry:
        latest = self._latest_by_candidate(_nonempty(candidate_id, "candidate_id"))
        if latest is None:
            raise CandidateNotRegisteredError(
                f"candidate_id is not registered: {candidate_id}"
            )
        return latest

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "entries": [record.model_dump(mode="json") for record in self._records],
        }

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _append(self, entry: CandidateRegistryEntry) -> None:
        self._records.append(entry)
        self.write()

    def _latest_by_candidate(
        self,
        candidate_id: str,
    ) -> Optional[CandidateRegistryEntry]:
        for record in reversed(self._records):
            if record.candidate_id == candidate_id:
                return record
        return None

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            self.write()
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise IdeaRegistryError(f"idea registry is not valid JSON: {self.path}") from exc
        if payload.get("schema_version") != self.schema_version:
            raise IdeaRegistryError(
                f"unsupported idea registry schema_version: {payload.get('schema_version')!r}"
            )
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            raise IdeaRegistryError("idea registry entries must be a list")
        self._records = [CandidateRegistryEntry.model_validate(item) for item in entries]


__all__ = [
    "CandidateAlreadyRegisteredError",
    "CandidateNotExecutableError",
    "CandidateNotRegisteredError",
    "CandidateRegistryEntry",
    "IdeaCandidateRegistry",
    "IdeaRegistryError",
    "SelectionStatus",
]
