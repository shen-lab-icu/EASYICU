"""Host-only materialization of verified event absence.

The SDK accepts one immutable :class:`LoadedTypedInput`; callers cannot supply
an arbitrary path or DataFrame.  It verifies the source-status contract against
the exact artifact and ordered row identity, then fills zero only for rows
classified as ``verified_absent``.  Unknown or contradictory source states
remain fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .source_status import (
    SourceStatusContract,
    SourceStatusCounts,
    source_status_contract_digest,
)
from .typed_input_receipt import TypedInputRowIdentity, _frame_digest
from .typed_input_sdk import LoadedTypedInput

SOURCE_STATUS_MATERIALIZATION_SCHEMA = "easyicu.source_status_materialization/1"
_CONSTRUCTION_TOKEN = object()
_ALLOWED_STATES = frozenset(
    {"observed", "verified_absent", "unmeasured", "source_missing", "contradictory"}
)


class SourceStatusMaterializationError(RuntimeError):
    """Source-status authority could not safely materialize a value column."""


class SourceStatusMaterializationReceipt(BaseModel):
    """Digest-bound receipt for one host-owned absence materialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.source_status_materialization/1"] = (
        SOURCE_STATUS_MATERIALIZATION_SCHEMA
    )
    variable: str = Field(min_length=1)
    source_status_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_input_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    value_column: str = Field(min_length=1)
    status_column: str = Field(min_length=1)
    counts: SourceStatusCounts
    output_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_digest(self) -> "SourceStatusMaterializationReceipt":
        if source_status_materialization_receipt_sha256(self) != self.receipt_sha256:
            raise ValueError("source-status materialization receipt SHA-256 mismatch")
        return self


def source_status_materialization_receipt_sha256(
    receipt: SourceStatusMaterializationReceipt | dict[str, object],
) -> str:
    """Return the self-digest of one materialization receipt."""

    if isinstance(receipt, SourceStatusMaterializationReceipt):
        payload = receipt.model_dump(mode="json")
    else:
        payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class MaterializedSourceStatusInput:
    """Immutable Arrow payload paired with its host materialization receipt."""

    __slots__ = ("__payload", "__receipt", "__sealed")

    def __init__(
        self,
        *,
        payload: pa.Table,
        receipt: SourceStatusMaterializationReceipt,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise SourceStatusMaterializationError(
                "MaterializedSourceStatusInput may only be constructed by the host SDK"
            )
        if _frame_digest(payload.to_pandas()) != receipt.output_frame_sha256:
            raise SourceStatusMaterializationError(
                "materialized payload does not match its receipt"
            )
        object.__setattr__(self, "_MaterializedSourceStatusInput__payload", payload)
        object.__setattr__(self, "_MaterializedSourceStatusInput__receipt", receipt)
        object.__setattr__(self, "_MaterializedSourceStatusInput__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_MaterializedSourceStatusInput__sealed", False):
            raise AttributeError("MaterializedSourceStatusInput is immutable")
        object.__setattr__(self, name, value)

    @property
    def payload(self) -> pa.Table:
        return self.__payload

    @property
    def receipt(self) -> SourceStatusMaterializationReceipt:
        return self.__receipt

    def to_pandas(self) -> pd.DataFrame:
        return self.__payload.to_pandas()


def materialize_verified_absence(
    *,
    source_input: LoadedTypedInput,
    contract: SourceStatusContract,
    value_column: str,
) -> MaterializedSourceStatusInput:
    """Materialize zero only for rows with exact verified-absence authority."""

    receipt = source_input.receipt
    if receipt.artifact_sha256 != contract.row_status_artifact_sha256:
        raise SourceStatusMaterializationError(
            "source input artifact does not match the source-status contract"
        )
    if not isinstance(receipt.row_identity, TypedInputRowIdentity):
        raise SourceStatusMaterializationError(
            "source-status materialization requires ordered row identity"
        )
    if receipt.row_identity.sha256 != contract.row_identity_sha256:
        raise SourceStatusMaterializationError(
            "source input row identity does not match the source-status contract"
        )
    if value_column != contract.variable or value_column not in contract.source_columns:
        raise SourceStatusMaterializationError(
            "value column is not the contract-bound variable"
        )
    status_column = str(contract.row_status_column or "")
    frame = source_input.to_pandas()
    required_columns = set(contract.source_columns) | {status_column, value_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise SourceStatusMaterializationError(
            f"source-status artifact is missing required columns: {missing_columns}"
        )
    if len(frame) != contract.n_total:
        raise SourceStatusMaterializationError(
            "source-status artifact row count does not match the contract"
        )

    status = frame[status_column].astype("string")
    if status.isna().any():
        raise SourceStatusMaterializationError(
            "source-status column contains missing rows"
        )
    unknown = sorted(set(status.astype(str)) - _ALLOWED_STATES)
    if unknown:
        raise SourceStatusMaterializationError(
            f"source-status column contains unknown states: {unknown}"
        )
    actual_counts = SourceStatusCounts(
        **{name: int((status == name).sum()) for name in sorted(_ALLOWED_STATES)}
    )
    if actual_counts != contract.counts:
        raise SourceStatusMaterializationError(
            "row-level source states do not match the contract counts"
        )
    if actual_counts.contradictory:
        raise SourceStatusMaterializationError(
            "contradictory rows cannot be materialized"
        )

    values = frame[value_column].copy(deep=True)
    observed = status == "observed"
    verified_absent = status == "verified_absent"
    unresolved = status.isin(["unmeasured", "source_missing"])
    if values.loc[observed].isna().any():
        raise SourceStatusMaterializationError(
            "observed source-status rows contain missing values"
        )
    existing_absent = values.loc[verified_absent].dropna()
    if not existing_absent.eq(0).all():
        raise SourceStatusMaterializationError(
            "verified-absent rows contain a nonzero value"
        )
    if values.loc[unresolved].notna().any():
        raise SourceStatusMaterializationError(
            "unresolved source-status rows contain observed values"
        )
    values.loc[verified_absent] = 0
    frame[value_column] = values

    output_sha256 = _frame_digest(frame)
    receipt_payload: dict[str, object] = {
        "schema_version": SOURCE_STATUS_MATERIALIZATION_SCHEMA,
        "variable": contract.variable,
        "source_status_contract_sha256": source_status_contract_digest(contract),
        "source_input_receipt_sha256": receipt.receipt_sha256,
        "source_artifact_sha256": receipt.artifact_sha256,
        "row_identity_sha256": receipt.row_identity.sha256,
        "value_column": value_column,
        "status_column": status_column,
        "counts": actual_counts.model_dump(mode="json"),
        "output_frame_sha256": output_sha256,
    }
    receipt_payload["receipt_sha256"] = source_status_materialization_receipt_sha256(
        receipt_payload
    )
    materialization_receipt = SourceStatusMaterializationReceipt.model_validate(
        receipt_payload
    )
    payload = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    return MaterializedSourceStatusInput(
        payload=payload,
        receipt=materialization_receipt,
        _construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "MaterializedSourceStatusInput",
    "SOURCE_STATUS_MATERIALIZATION_SCHEMA",
    "SourceStatusMaterializationError",
    "SourceStatusMaterializationReceipt",
    "materialize_verified_absence",
    "source_status_materialization_receipt_sha256",
]
