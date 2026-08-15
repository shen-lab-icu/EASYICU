"""Verify Planner-owned cardinality and row-role consumption contracts.

This boundary checks an exact digest-bound table against a contract already
owned by the consuming plan step.  It does not choose which rows, outcomes, or
models matter scientifically.  A consumer with no explicit role selection must
preserve all rows; role-specific selection is valid only when the Planner names
the role column and complete expected roster.
"""

from __future__ import annotations

from ..canonical_json import sha256_file as _sha256_file

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..schema import ArtifactConsumptionContract


class ArtifactConsumptionError(ValueError):
    """A declared consumption rule does not match the sealed input artifact."""


def _read_role_column(path: Path, role_column: str) -> pd.Series:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, usecols=[role_column])
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t", usecols=[role_column])
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=[role_column])
    else:
        raise ArtifactConsumptionError("unsupported typed table format")
    return frame[role_column]


def verify_artifact_consumption(
    *,
    contract: ArtifactConsumptionContract,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a host receipt proving one consumption contract against a table."""

    if str(binding.get("identity_row", {}).get("input_key") or "") != (
        contract.input_key
    ):
        raise ArtifactConsumptionError("consumption input identity mismatch")
    product_contract = binding.get("product_contract")
    if not isinstance(product_contract, Mapping):
        raise ArtifactConsumptionError("typed table lacks a host product contract")
    row_count = product_contract.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ArtifactConsumptionError("typed table lacks a verified row_count")
    artifact_sha256 = str(binding.get("sha256") or "")
    path = Path(str(binding.get("absolute_path") or ""))
    if (
        len(artifact_sha256) != 64
        or not path.is_file()
        or _sha256_file(path) != artifact_sha256
    ):
        raise ArtifactConsumptionError("typed table bytes do not match the binding")

    receipt: dict[str, Any] = {
        "schema_version": "easyicu.verified_artifact_consumption/1",
        "input_key": contract.input_key,
        "mode": contract.mode,
        "artifact_sha256": artifact_sha256,
        "verified_row_count": row_count,
    }
    if contract.mode == "single_row":
        if row_count != 1:
            raise ArtifactConsumptionError(
                f"single_row requires exactly one row; observed {row_count}"
            )
        return receipt
    if contract.mode == "all_rows":
        return receipt

    role_column = str(contract.role_column)
    columns = product_contract.get("columns")
    if not isinstance(columns, list) or role_column not in columns:
        raise ArtifactConsumptionError(
            f"role column {role_column!r} is absent from the verified schema"
        )
    try:
        role_values = _read_role_column(path, role_column)
    except (KeyError, OSError, ValueError) as exc:
        raise ArtifactConsumptionError("could not verify role-column values") from exc
    if _sha256_file(path) != artifact_sha256:
        raise ArtifactConsumptionError("typed table changed during role verification")
    observed_roles = [str(value) for value in role_values.tolist()]
    expected_roles = list(contract.expected_roles)
    role_counts = {role: observed_roles.count(role) for role in expected_roles}
    extra_roles = sorted(set(observed_roles) - set(expected_roles))
    if (
        any(count != 1 for count in role_counts.values())
        or extra_roles
        or (len(observed_roles) != len(expected_roles))
    ):
        raise ArtifactConsumptionError(
            "one_per_role requires exactly one row for every declared role and no "
            "undeclared role rows"
        )
    receipt.update(
        {
            "role_column": role_column,
            "expected_roles": expected_roles,
            "verified_role_counts": role_counts,
        }
    )
    return receipt


__all__ = [
    "ArtifactConsumptionError",
    "verify_artifact_consumption",
]
