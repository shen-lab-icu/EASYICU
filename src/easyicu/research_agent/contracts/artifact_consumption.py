"""Verify and load the exact artifacts authorized by consumption contracts.

This boundary checks an exact digest-bound table against a contract already
owned by the consuming plan step.  It does not choose which rows, outcomes, or
models matter scientifically.  A consumer with no explicit role selection must
preserve all rows; role-specific selection is valid only when the Planner names
the role column and complete expected roster. Intake and execution share the
same containment, identity, digest, and table-shape checks when loading a bound
input.
"""

from __future__ import annotations

from ..canonical_json import sha256_file as _sha256_file

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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


#: Stable failure codes. Callers may branch on these; the messages may change.
BINDING_REASON_CODES = frozenset(
    {
        "manifest_unreadable",
        "manifest_step_mismatch",
        "binding_absent",
        "binding_widened",
        "binding_incomplete",
        "declared_kind_mismatch",
        "evidence_kind_mismatch",
        "product_identity_mismatch",
        "consumption_contract_mismatch",
        "path_not_contained",
        "unsupported_format",
        "digest_mismatch",
        "digest_changed_during_read",
        "product_contract_incomplete",
        "contract_columns_mismatch",
        "contract_row_count_mismatch",
    }
)


class TypedInputBindingError(RuntimeError):
    """A typed binding could not be honoured exactly as recorded."""

    def __init__(self, reason_code: str, message: str) -> None:
        if reason_code not in BINDING_REASON_CODES:  # pragma: no cover - guard
            raise AssertionError(f"undeclared binding reason code: {reason_code!r}")
        self.reason_code = reason_code
        super().__init__(f"{message} (reason_code={reason_code})")


@dataclass(frozen=True)
class BoundTypedInput:
    """One verified binding: the bytes, and what they were promised to be."""

    input_key: str
    path: Path
    sha256: str
    declared_kind: str
    evidence_kind: str
    product: str
    evidence_id: str
    columns: tuple[str, ...]
    row_count: int
    frame: pd.DataFrame
    binding: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_regular_file(path: Path, root: Path) -> Optional[Path]:
    """Return ``path`` only if it is a real file genuinely inside ``root``.

    Both the pre- and post-resolution containment checks are required: the
    first refuses a binding that names somewhere else, the second refuses one
    that reaches somewhere else through a link.
    """

    root = root.resolve()
    candidate = Path(path)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            return None
        parent = cursor.parent
        if parent == cursor:
            return None
        cursor = parent
    if not candidate.is_file():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def read_frame(path: Path, *, text_columns: Sequence[str] = ()) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, converters={name: str for name in text_columns})
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", converters={name: str for name in text_columns})
    raise TypedInputBindingError(
        "unsupported_format", "Typed input table format is unsupported"
    )


def _manifest_payload(resolved_inputs: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(resolved_inputs, Mapping):
        return dict(resolved_inputs)
    try:
        payload = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    except Exception as exc:
        raise TypedInputBindingError(
            "manifest_unreadable", "Resolved input manifest is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise TypedInputBindingError(
            "manifest_unreadable", "Resolved input manifest is not an object"
        )
    return payload


def _hex_digest(value: Any) -> Optional[str]:
    text = str(value or "")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        return None
    return text


def _verify_identity_row(
    *,
    identity_row: Any,
    input_key: str,
    binding: Mapping[str, Any],
) -> None:
    """A capsule must agree with its own identity record.

    The host writes ``identity_row`` as the single record of *what this input
    is*. If it disagrees with the fields beside it, one of the two was written
    by something that did not know about the other, and neither can then be
    trusted to say which bytes are authorised.
    """

    if identity_row is None:
        return
    if not isinstance(identity_row, Mapping):
        raise TypedInputBindingError(
            "product_identity_mismatch", "Typed input identity_row is not a record"
        )
    for field in ("input_key", "declared_kind", "product", "evidence_id", "sha256"):
        if field not in identity_row:
            continue
        expected = input_key if field == "input_key" else binding.get(field)
        if expected is None:
            continue
        if str(identity_row.get(field) or "") != str(expected or ""):
            raise TypedInputBindingError(
                "product_identity_mismatch",
                f"Typed input identity_row disagrees with the binding on {field}",
            )


def load_typed_input(
    *,
    input_key: str,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: Optional[str] = None,
    expected_declared_kind: Optional[str] = None,
    expected_evidence_kind: Optional[str] = None,
    expected_columns: Optional[Sequence[str]] = None,
    exclusive: bool = False,
    require_consumption_contract: bool = False,
    consumption_mode: str = "all_rows",
    minimum_row_count: int = 0,
    text_columns: Sequence[str] = (),
) -> BoundTypedInput:
    """Load exactly the artifact recorded for ``input_key``, verifying it fully.

    ``expected_declared_kind`` / ``expected_columns`` let a caller state the
    product it is competent to consume. They are checks, never inference: a
    caller that omits them still gets every containment, identity and digest
    guarantee, it simply accepts any shape.
    """

    payload = _manifest_payload(resolved_inputs)
    if step_id is not None and payload.get("step_id") != step_id:
        raise TypedInputBindingError(
            "manifest_step_mismatch",
            "resolved-input manifest does not belong to this step",
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise TypedInputBindingError(
            "binding_absent", "Resolved input manifest declares no inputs"
        )
    binding = inputs.get(input_key)
    if not isinstance(binding, dict):
        raise TypedInputBindingError(
            "binding_absent", f"Missing exact typed binding: {input_key}"
        )
    # Absence is checked before exclusivity so the two are never confused. A
    # widened binding means the host offered this step something extra it did
    # not ask for; a missing one means the thing it needs is not there at all.
    # Reporting the second as the first sends a reader looking for an input
    # that was never added.
    if exclusive and set(inputs) != {input_key}:
        extra = sorted(set(inputs) - {input_key})
        raise TypedInputBindingError(
            "binding_widened",
            f"binding for {input_key} is widened by {extra!r}; this consumer "
            "reads exactly one input",
        )

    relative_path = binding.get("relative_path")
    expected_sha256 = _hex_digest(binding.get("sha256"))
    contract = binding.get("product_contract")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or expected_sha256 is None
        or not isinstance(contract, dict)
    ):
        raise TypedInputBindingError(
            "binding_incomplete", f"Typed binding for {input_key} is incomplete"
        )

    declared_kind = str(binding.get("declared_kind") or "")
    if expected_declared_kind is not None and declared_kind != expected_declared_kind:
        raise TypedInputBindingError(
            "declared_kind_mismatch",
            f"Typed binding for {input_key} declares kind {declared_kind!r}, "
            f"not {expected_declared_kind!r}",
        )
    # ``declared_kind`` is what the *plan* called the product; ``evidence_kind``
    # is what the host verified the bytes physically are. They are not always
    # the same word -- a ``cohort:`` or ``artifact:`` product legitimately
    # resolves to a bound table -- so a caller that needs a real table asks for
    # the physical kind explicitly rather than inferring it from the plan's.
    evidence_kind = str(binding.get("evidence_kind") or "")
    if expected_evidence_kind is not None and evidence_kind != expected_evidence_kind:
        raise TypedInputBindingError(
            "evidence_kind_mismatch",
            f"Typed binding for {input_key} resolves to evidence of kind "
            f"{evidence_kind!r}, not {expected_evidence_kind!r}",
        )
    product = str(binding.get("product") or "")
    # Compare canonical identity to canonical identity.  ``declared_kind`` was
    # written by the binding registry, which canonicalises the Planner's alias
    # (`cohort:` is recorded as `dataset:`, deliberately, so plan DAG, declared
    # output validation and runtime binding share one identity).  Partitioning
    # the key here yields the Planner's *raw* prefix, so a plan that spelled its
    # cohort `cohort:analysis_set` -- one of the four spellings the planner
    # prompt lists as legal -- could never satisfy this check, and every typed
    # executor downstream failed on a `product_identity_mismatch`.
    #
    # It was worse than a refusal: the repair the message invites is to rewrite
    # the key to `dataset:analysis_set`, which is the canonical spelling, and
    # bindings are filed under the plan's spelling -- so that attempt failed as
    # `binding_absent`.  Both spellings failed, for opposite reasons, and the
    # step spent its whole repair budget discovering there was no third one.
    # Measured on a real run: four steps died this way in one plan.
    from .product_identity import typed_product as _typed_product

    canonical_key = _typed_product(input_key)
    canonical_binding = (
        _typed_product(f"{declared_kind}:{product}")
        if declared_kind and product
        else None
    )
    if canonical_key is not None and canonical_binding is not None:
        if canonical_binding[0] != canonical_key[0]:
            raise TypedInputBindingError(
                "product_identity_mismatch",
                f"Typed binding kind {declared_kind!r} does not match the input "
                f"key {input_key!r}",
            )
        if canonical_binding[1] != canonical_key[1]:
            raise TypedInputBindingError(
                "product_identity_mismatch",
                f"Typed binding product {product!r} does not match the input "
                f"key {input_key!r}",
            )
    _verify_identity_row(
        identity_row=binding.get("identity_row"),
        input_key=input_key,
        binding=binding,
    )

    consumption = binding.get("consumption_contract")
    if require_consumption_contract and not isinstance(consumption, Mapping):
        raise TypedInputBindingError(
            "consumption_contract_mismatch",
            f"Typed binding for {input_key} carries no consumption contract",
        )
    if isinstance(consumption, Mapping):
        if (
            consumption.get("input_key") != input_key
            or consumption.get("mode") != consumption_mode
            or consumption.get("artifact_sha256") != expected_sha256
        ):
            raise TypedInputBindingError(
                "consumption_contract_mismatch",
                f"Consumption contract for {input_key} does not authorise these "
                "bytes in this mode",
            )

    path = contained_regular_file(Path(run_dir) / relative_path, Path(run_dir))
    if path is None:
        raise TypedInputBindingError(
            "path_not_contained",
            f"Typed binding for {input_key} is not a contained regular file",
        )
    if sha256_file(path) != expected_sha256:
        raise TypedInputBindingError(
            "digest_mismatch",
            f"Typed binding for {input_key}: digest verification failed",
        )

    columns = contract.get("columns")
    row_count = contract.get("row_count")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(value, str) and value for value in columns)
        or len(columns) != len(set(columns))
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count < minimum_row_count
    ):
        raise TypedInputBindingError(
            "product_contract_incomplete",
            f"Typed binding for {input_key} has an incomplete product_contract",
        )
    if expected_columns is not None and columns != list(expected_columns):
        raise TypedInputBindingError(
            "contract_columns_mismatch",
            f"Typed binding for {input_key} promises a different product schema",
        )

    # Identity tokens in a delimited file are not measurements: preserve
    # zero prefixes and literal NA-like identifiers without weakening byte checks.
    frame = read_frame(path, text_columns=text_columns) if text_columns else read_frame(path)
    if list(frame.columns) != columns:
        raise TypedInputBindingError(
            "contract_columns_mismatch",
            f"Typed binding for {input_key}: bytes disagree with its product "
            "contract on columns",
        )
    if len(frame) != row_count:
        raise TypedInputBindingError(
            "contract_row_count_mismatch",
            f"Typed binding for {input_key}: bytes disagree with its product "
            "contract on row count",
        )
    # Re-hash after reading. The window between the first hash and the read is
    # small, but it is the whole window in which a swapped file would go
    # unnoticed, and closing it costs one pass over a file already in cache.
    if sha256_file(path) != expected_sha256:
        raise TypedInputBindingError(
            "digest_changed_during_read",
            f"Typed binding for {input_key} changed on disk while being read",
        )

    return BoundTypedInput(
        input_key=input_key,
        path=path,
        sha256=expected_sha256,
        declared_kind=declared_kind,
        evidence_kind=evidence_kind,
        product=product,
        evidence_id=str(binding.get("evidence_id") or ""),
        columns=tuple(columns),
        row_count=int(row_count),
        frame=frame,
        binding=dict(binding),
    )


__all__ = [
    "ArtifactConsumptionError",
    "BINDING_REASON_CODES",
    "BoundTypedInput",
    "TypedInputBindingError",
    "contained_regular_file",
    "load_typed_input",
    "read_frame",
    "sha256_file",
    "verify_artifact_consumption",
]
