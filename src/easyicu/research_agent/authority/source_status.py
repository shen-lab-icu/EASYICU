"""Host-owned authority for source availability and verified event absence."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schema import ResearchContext

SOURCE_STATUS_CONTRACT_SCHEMA = "easyicu.source_status_contract/1"


class SourceStatusContractError(ValueError):
    """Raised when host source-status authority is malformed or contradictory."""


class SourceStatusCounts(BaseModel):
    """Mutually exclusive row counts for one source-bound variable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observed: int = Field(ge=0)
    verified_absent: int = Field(ge=0)
    unmeasured: int = Field(ge=0)
    source_missing: int = Field(ge=0)
    contradictory: int = Field(ge=0)

    @property
    def total(self) -> int:
        return int(
            self.observed
            + self.verified_absent
            + self.unmeasured
            + self.source_missing
            + self.contradictory
        )


class SourceStatusContract(BaseModel):
    """Digest-bound host statement about one variable's row-level source states.

    The contract does not itself rewrite a value column.  It only authorizes a
    downstream host SDK to materialize ``verified_absent`` as zero.  Generated
    analysis code cannot issue this authority.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.source_status_contract/1"] = (
        SOURCE_STATUS_CONTRACT_SCHEMA
    )
    variable: str = Field(min_length=1)
    n_total: int = Field(ge=0)
    counts: SourceStatusCounts
    source_coverage: Literal["complete", "partial", "unavailable"]
    verified_absent_value: Literal[0] | None = None
    authority_kind: Literal[
        "export_manifest",
        "event_reconciliation",
        "measurement_audit",
    ]
    authority_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_columns: tuple[str, ...] = Field(min_length=1)
    row_status_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    row_status_column: str | None = Field(default=None, min_length=1)
    row_identity_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def _validate_closed_partition(self) -> "SourceStatusContract":
        if self.counts.total != self.n_total:
            raise ValueError("source-status counts do not partition n_total")
        if self.source_coverage == "complete" and self.counts.source_missing:
            raise ValueError(
                "complete source coverage cannot contain source_missing rows"
            )
        if self.counts.verified_absent and self.verified_absent_value != 0:
            raise ValueError("verified_absent rows require verified_absent_value=0")
        if not self.counts.verified_absent and self.verified_absent_value is not None:
            raise ValueError(
                "verified_absent_value is set without verified_absent rows"
            )
        if len(set(self.source_columns)) != len(self.source_columns):
            raise ValueError("source_columns contains duplicates")
        if any(not str(column).strip() for column in self.source_columns):
            raise ValueError("source_columns contains an empty name")
        nonobserved = self.n_total - self.counts.observed
        row_binding = (
            self.row_status_artifact_sha256,
            self.row_status_column,
            self.row_identity_sha256,
        )
        if nonobserved and any(value is None for value in row_binding):
            raise ValueError(
                "non-observed rows require a status artifact, status column, "
                "and row-identity digest"
            )
        if not nonobserved and any(value is not None for value in row_binding):
            raise ValueError(
                "row-status bindings are set although every row is observed"
            )
        if self.source_coverage == "unavailable" and (
            self.counts.observed or self.counts.verified_absent
        ):
            raise ValueError(
                "unavailable source coverage cannot contain observed or "
                "verified-absent rows"
            )
        return self


def source_status_contract_digest(contract: SourceStatusContract) -> str:
    """Return the canonical digest of one verified contract."""

    payload = json.dumps(
        contract.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_status_contract_from_context(
    context: ResearchContext,
    *,
    variable: str,
) -> SourceStatusContract | None:
    """Load exact host authority from cohort provenance, never from prose."""

    provenance = context.cohort.provenance
    raw_contracts = provenance.get("source_status_contracts")
    if raw_contracts is None:
        return None
    if not isinstance(raw_contracts, Mapping):
        raise SourceStatusContractError("source_status_contracts is not an object")
    raw = raw_contracts.get(variable)
    if raw is None:
        return None
    try:
        contract = SourceStatusContract.model_validate(raw)
    except Exception as exc:
        raise SourceStatusContractError(
            f"invalid source-status contract for {variable!r}: {exc}"
        ) from exc
    if contract.variable != variable:
        raise SourceStatusContractError(
            "source-status contract variable does not match its provenance key"
        )
    if contract.n_total != context.cohort.n_stays:
        raise SourceStatusContractError(
            "source-status contract n_total does not match the locked cohort"
        )
    return contract


__all__ = [
    "SOURCE_STATUS_CONTRACT_SCHEMA",
    "SourceStatusContract",
    "SourceStatusContractError",
    "SourceStatusCounts",
    "source_status_contract_digest",
    "source_status_contract_from_context",
]
