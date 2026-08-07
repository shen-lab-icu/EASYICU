"""Pandera adapter for generic dataframe contracts.

This adapter owns only generic tabular mechanics: named columns, declared
dtypes, nullability and non-clinical value domains.  It must never infer a
clinical concept, event time, patient entity, or causal role.  Validation is
strict and non-coercing, so malformed source data becomes an explicit receipt
rather than silently changing values to satisfy the schema.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

from .runtime import probe_external_adapter


@dataclass(frozen=True)
class PanderaColumnContract:
    """One generic column declaration; clinical meaning stays with EasyICU."""

    name: str
    dtype: str
    nullable: bool = False
    required: bool = True
    allowed_values: Tuple[object, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Pandera column contract requires a name")
        if not self.dtype.strip():
            raise ValueError("Pandera column contract requires a dtype")
        try:
            allowed_value_count = len(set(self.allowed_values))
        except TypeError as exc:
            raise ValueError("Pandera allowed values must be hashable") from exc
        if len(self.allowed_values) != allowed_value_count:
            raise ValueError("Pandera allowed values must be unique")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("Pandera minimum cannot exceed maximum")


@dataclass(frozen=True)
class PanderaDataFrameContract:
    """A generic schema.  ``strict=True`` rejects undeclared columns."""

    contract_id: str
    columns: Tuple[PanderaColumnContract, ...]
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("Pandera dataframe contract requires an id")
        if not self.columns:
            raise ValueError("Pandera dataframe contract requires columns")
        names = tuple(column.name for column in self.columns)
        if len(names) != len(set(names)):
            raise ValueError("Pandera dataframe contract column names must be unique")


@dataclass(frozen=True)
class PanderaValidationReceipt:
    """No-row-data receipt for generic dataframe validation."""

    contract_id: str
    status: Literal["validated", "invalid", "adapter_unavailable"]
    adapter_version: Optional[str]
    issue_code: Optional[str]
    validated_columns: Tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "easyicu.pandera_validation_receipt/1",
            "contract_id": self.contract_id,
            "status": self.status,
            "adapter_version": self.adapter_version,
            "issue_code": self.issue_code,
            "validated_columns": list(self.validated_columns),
        }


def validate_dataframe_contract(
    dataframe: Any,
    contract: PanderaDataFrameContract,
) -> PanderaValidationReceipt:
    """Validate one dataframe lazily, strictly and without type coercion.

    A missing dependency is intentionally reported as unavailable rather than
    falling back to a partial hand-written validator.  That makes a future
    approved image change visible in both evidence and tests.
    """

    runtime = probe_external_adapter("pandera_dataframe_contract_v1")
    column_names = tuple(column.name for column in contract.columns)
    if not runtime.available:
        return PanderaValidationReceipt(
            contract_id=contract.contract_id,
            status="adapter_unavailable",
            adapter_version=None,
            issue_code=runtime.issue_code,
            validated_columns=column_names,
        )

    pa = importlib.import_module("pandera.pandas")
    errors = importlib.import_module("pandera.errors")
    schema_columns: dict[str, Any] = {}
    for column in contract.columns:
        checks: list[Any] = []
        if column.allowed_values:
            checks.append(pa.Check.isin(list(column.allowed_values)))
        if column.minimum is not None:
            checks.append(pa.Check.ge(column.minimum))
        if column.maximum is not None:
            checks.append(pa.Check.le(column.maximum))
        schema_columns[column.name] = pa.Column(
            column.dtype,
            checks=checks or None,
            nullable=column.nullable,
            required=column.required,
            coerce=False,
        )
    schema = pa.DataFrameSchema(
        schema_columns,
        strict=contract.strict,
        coerce=False,
        name=contract.contract_id,
    )
    validation_errors = tuple(
        error
        for error in (
            getattr(errors, "SchemaError", None),
            getattr(errors, "SchemaErrors", None),
        )
        if isinstance(error, type) and issubclass(error, Exception)
    )
    try:
        schema.validate(dataframe, lazy=True)
    except validation_errors:
        # Do not include Pandera's raw failure cases: they can contain source
        # values.  The bound evidence artifact retains the detailed audit when
        # an approved executor chooses to persist it under its own PII policy.
        return PanderaValidationReceipt(
            contract_id=contract.contract_id,
            status="invalid",
            adapter_version=runtime.installed_version,
            issue_code="dataframe_contract_invalid",
            validated_columns=column_names,
        )
    return PanderaValidationReceipt(
        contract_id=contract.contract_id,
        status="validated",
        adapter_version=runtime.installed_version,
        issue_code=None,
        validated_columns=column_names,
    )


__all__ = [
    "PanderaColumnContract",
    "PanderaDataFrameContract",
    "PanderaValidationReceipt",
    "validate_dataframe_contract",
]
