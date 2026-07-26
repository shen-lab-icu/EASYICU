"""Host-owned receipts for actual typed-table consumption.

The resolved-input manifest selects one immutable evidence artifact.  This
module opens that exact regular file, verifies its bytes and row identity, and
returns an ephemeral load handle.  A durable receipt can be sealed only when
the host consumer presents the same, unchanged DataFrame object.  Merely
mentioning a path in generated code is never consumption authority.

This is a leaf authority primitive.  It does not select inputs, execute a
model, promote evidence, or decide whether a result is current.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Annotated, Any, Literal, Mapping, Sequence

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from ..canonical_json import canonical_json_bytes as _canonical_json_bytes
from ..schema import ValidationFinding
from .filesystem import AnchoredDirectory, AuthorityFilesystemError
from .run_input import canonical_sha256

TYPED_INPUT_CONSUMPTION_RECEIPT_SCHEMA = "easyicu.typed_input_consumption_receipt/1"
_RESOLVED_INPUTS_SCHEMA = "2.1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class TypedInputReceiptError(RuntimeError):
    """A typed input could not be bound to actual consumed table bytes."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TypedInputRowIdentity(_StrictFrozenModel):
    """Exact ordered row identity observed in the opened table."""

    column: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    sha256: _Sha256
    unique: Literal[True]
    missing_count: Literal[0]


class TypedInputRowIdentityNotApplicable(_StrictFrozenModel):
    """Receipt marker for host-declared inputs that do not align rows."""

    not_applicable: Literal[True]
    reason: str = Field(min_length=1)


class TypedInputConsumptionReceipt(_StrictFrozenModel):
    """Strict receipt joining one opened table to one code/step consumer."""

    schema_version: Literal["easyicu.typed_input_consumption_receipt/1"]
    input_key: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    artifact_sha256: _Sha256
    resolved_inputs_sha256: _Sha256
    resolved_input_binding_sha256: _Sha256
    artifact_relative_path: str = Field(min_length=1)
    opened_file_sha256: _Sha256
    opened_file_size_bytes: int = Field(ge=0)
    row_identity: TypedInputRowIdentity | TypedInputRowIdentityNotApplicable
    consumer_step_id: str = Field(min_length=1)
    consumer_code_sha256: _Sha256
    loaded_frame_sha256: _Sha256
    receipt_sha256: _Sha256

    @model_validator(mode="after")
    def _verify_internal_binding(self) -> "TypedInputConsumptionReceipt":
        if self.opened_file_sha256 != self.artifact_sha256:
            raise ValueError("opened file does not match artifact SHA-256")
        if typed_input_receipt_sha256(self) != self.receipt_sha256:
            raise ValueError("typed-input receipt SHA-256 mismatch")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedTypedInputLoad:
    """Ephemeral capability for one verified table loaded by the host."""

    frame: pd.DataFrame
    input_key: str
    evidence_id: str
    artifact_sha256: str
    resolved_inputs_sha256: str
    resolved_input_binding_sha256: str
    artifact_relative_path: str
    opened_file_size_bytes: int
    row_identity_column: str
    row_identity_sha256: str
    loaded_frame_sha256: str
    consumer_step_id: str
    consumer_code_sha256: str


@dataclass(frozen=True, slots=True)
class StepTypedInputReceiptVerification:
    """Structured step-level typed-input receipt verification result."""

    findings: tuple[ValidationFinding, ...]
    verified_inputs: Mapping[str, TypedInputConsumptionReceipt]


def typed_input_receipt_sha256(
    value: TypedInputConsumptionReceipt | Mapping[str, object],
) -> str:
    """Return the receipt's self-digest, excluding that digest field."""

    if isinstance(value, TypedInputConsumptionReceipt):
        payload: dict[str, object] = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("typed-input receipt must be a model or mapping")
    payload.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validated_sha256(value: object, *, name: str) -> str:
    candidate = str(value or "")
    if _SHA256_RE.fullmatch(candidate) is None:
        raise TypedInputReceiptError(f"{name} must be a SHA-256 digest")
    return candidate


def _validated_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypedInputReceiptError(f"{name} must be a non-empty string")
    return value.strip()


def _typed_input_product(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    declared_kind, product = value.split(":", 1)
    declared_kind = declared_kind.strip()
    product = product.strip()
    if not declared_kind or not product:
        return None
    return declared_kind, product


def _finding(
    *,
    issue_code: str,
    message: str,
    severity: Literal["info", "warning", "error"] = "error",
    input_key: str | None = None,
    detail: Mapping[str, object] | None = None,
) -> ValidationFinding:
    payload: dict[str, object] = {"issue_code": issue_code}
    if input_key is not None:
        payload["input_key"] = input_key
    if detail:
        payload.update(dict(detail))
    return ValidationFinding(
        validator="typed_input_receipt_completeness",
        severity=severity,
        message=message,
        detail=payload,
    )


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise TypedInputReceiptError(
                f"resolved-input manifest contains duplicate JSON key: {key}"
            )
        value[key] = item
    return value


def _lexically_contained(path: Path, *, root: Path, name: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    anchored_root = Path(os.path.abspath(os.fspath(root.expanduser())))
    try:
        candidate.relative_to(anchored_root)
    except ValueError as exc:
        raise TypedInputReceiptError(f"{name} is outside the run root") from exc
    return candidate


def _read_regular_bytes(path: Path, *, name: str) -> tuple[bytes, int]:
    try:
        with AnchoredDirectory.open(path.parent) as parent:
            with parent.open_regular(path.name) as handle:
                before = os.fstat(handle.fileno())
                payload = handle.read(int(before.st_size) + 1)
                after = os.fstat(handle.fileno())
    except (AuthorityFilesystemError, OSError) as exc:
        raise TypedInputReceiptError(f"cannot open {name} as a regular file") from exc
    if len(payload) != int(before.st_size) or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise TypedInputReceiptError(f"{name} changed while it was read")
    return payload, int(before.st_size)


def _read_manifest(
    path: Path,
    *,
    expected_sha256: str,
    run_root: Path,
) -> dict[str, object]:
    manifest_path = _lexically_contained(
        path,
        root=run_root,
        name="resolved-input manifest",
    )
    payload, _ = _read_regular_bytes(
        manifest_path,
        name="resolved-input manifest",
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise TypedInputReceiptError("resolved-input manifest SHA-256 mismatch")
    try:
        decoded = json.loads(payload, object_pairs_hook=_no_duplicate_object)
    except TypedInputReceiptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedInputReceiptError(
            "resolved-input manifest is not valid JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise TypedInputReceiptError("resolved-input manifest must be an object")
    if decoded.get("schema_version") != _RESOLVED_INPUTS_SCHEMA:
        raise TypedInputReceiptError("resolved-input manifest schema is unsupported")
    return decoded


def typed_input_row_identity_sha256(values: pd.Series) -> str:
    """Return the ordered row-identity digest used by typed-input receipts.

    Binding producers and receipt consumers must share this exact host-owned
    representation.  Keeping the helper public avoids a second, subtly
    incompatible row-identity digest at development-sample projection time.
    """

    digest = hashlib.sha256()
    for value in values.astype("string"):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "columns": [str(column) for column in frame.columns],
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "row_count": int(len(frame)),
            }
        )
    )
    try:
        hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
        digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    except (TypeError, ValueError):
        digest.update(
            frame.to_json(
                orient="split",
                date_format="iso",
                date_unit="ns",
                default_handler=str,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _load_table_from_regular_file(path: Path) -> tuple[pd.DataFrame, str, int]:
    try:
        with AnchoredDirectory.open(path.parent) as parent:
            with parent.open_regular(path.name) as handle:
                before = os.fstat(handle.fileno())
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                if handle.tell() != int(before.st_size):
                    raise TypedInputReceiptError(
                        "typed input changed while computing artifact SHA-256"
                    )
                handle.seek(0)
                suffix = path.suffix.lower()
                if suffix in {".parquet", ".pq"}:
                    frame = pd.read_parquet(handle)
                elif suffix == ".csv":
                    frame = pd.read_csv(handle)
                elif suffix == ".tsv":
                    frame = pd.read_csv(handle, sep="\t")
                elif suffix in {".xlsx", ".xls"}:
                    frame = pd.read_excel(handle)
                else:
                    raise TypedInputReceiptError(
                        f"unsupported typed table format: {suffix or '<none>'}"
                    )
                after = os.fstat(handle.fileno())
    except TypedInputReceiptError:
        raise
    except (AuthorityFilesystemError, OSError, ValueError, ImportError) as exc:
        raise TypedInputReceiptError(
            "cannot load typed input as a verified regular table"
        ) from exc
    except Exception as exc:
        # Pandas delegates to optional readers whose parse-error classes are
        # backend-specific.  None of those implementation details may escape
        # as a non-authority failure from this host boundary.
        raise TypedInputReceiptError(
            "cannot load typed input as a verified regular table"
        ) from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise TypedInputReceiptError("typed input changed while it was loaded")
    if not isinstance(frame, pd.DataFrame):
        raise TypedInputReceiptError("typed input loader did not return a DataFrame")
    return frame, digest.hexdigest(), int(before.st_size)


def _binding_for_input(
    manifest: Mapping[str, object],
    *,
    input_key: str,
    consumer_step_id: str,
) -> dict[str, object]:
    if manifest.get("step_id") != consumer_step_id:
        raise TypedInputReceiptError("resolved-input manifest consumer step mismatch")
    declared = manifest.get("planner_declared_inputs")
    if (
        not isinstance(declared, list)
        or any(not isinstance(item, str) for item in declared)
        or len(set(declared)) != len(declared)
        or declared.count(input_key) != 1
    ):
        raise TypedInputReceiptError(
            "input_key is not uniquely Planner-declared in resolved-input manifest"
        )
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or any(not isinstance(key, str) or key not in declared for key in inputs)
        or input_key not in inputs
    ):
        raise TypedInputReceiptError(
            "resolved-input bindings do not match Planner-declared typed inputs"
        )
    binding = inputs.get(input_key)
    if not isinstance(binding, Mapping):
        raise TypedInputReceiptError("input_key has no resolved-input binding")
    return dict(binding)


def _validate_binding_identity(
    binding: Mapping[str, object],
    *,
    input_key: str,
) -> tuple[str, str, str, Mapping[str, object]]:
    evidence_id = _validated_nonempty_string(
        binding.get("evidence_id"),
        name="evidence_id",
    )
    artifact_sha256 = _validated_sha256(
        binding.get("sha256"),
        name="artifact SHA-256",
    )
    identity = binding.get("identity_row")
    if not isinstance(identity, Mapping):
        raise TypedInputReceiptError("resolved-input identity row is missing")
    expected_identity = {
        "input_key": input_key,
        "declared_kind": binding.get("declared_kind"),
        "product": binding.get("product"),
        "evidence_id": evidence_id,
        "sha256": artifact_sha256,
        "produced_by_step": binding.get("produced_by_step"),
    }
    if dict(identity) != expected_identity:
        raise TypedInputReceiptError("resolved-input identity row mismatch")
    if binding.get("evidence_kind") != "table":
        raise TypedInputReceiptError("resolved typed input is not tabular evidence")
    contract = binding.get("product_contract")
    if not isinstance(contract, Mapping):
        raise TypedInputReceiptError("typed input product contract is missing")
    contract_identity = contract.get("identity_row")
    if contract_identity is not None and contract_identity != identity:
        raise TypedInputReceiptError("product contract identity row mismatch")
    return evidence_id, artifact_sha256, canonical_sha256(binding), contract


def _artifact_path(
    binding: Mapping[str, object],
    *,
    run_root: Path,
) -> tuple[Path, str]:
    relative_text = _validated_nonempty_string(
        binding.get("relative_path"),
        name="artifact relative path",
    )
    relative = Path(relative_text)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise TypedInputReceiptError("artifact relative path is unsafe")
    relative_text = relative.as_posix()
    selected = _lexically_contained(
        run_root / relative,
        root=run_root,
        name="typed input artifact",
    )
    absolute = _lexically_contained(
        Path(
            _validated_nonempty_string(
                binding.get("absolute_path"),
                name="artifact absolute path",
            )
        ),
        root=run_root,
        name="typed input artifact",
    )
    if selected != absolute:
        raise TypedInputReceiptError(
            "resolved-input relative and absolute artifact paths disagree"
        )
    return selected, relative_text


def _verify_table_contract(
    frame: pd.DataFrame,
    *,
    contract: Mapping[str, object],
) -> tuple[str, str]:
    columns = contract.get("columns")
    column_count = contract.get("column_count")
    if (
        not isinstance(columns, list)
        or any(not isinstance(column, str) for column in columns)
        or not isinstance(column_count, int)
        or isinstance(column_count, bool)
        or column_count != len(columns)
        or list(frame.columns) != columns
    ):
        raise TypedInputReceiptError("opened table violates product contract columns")
    identity_column = contract.get("row_identity_column")
    expected_row_count = contract.get("row_count")
    expected_identity_sha256 = contract.get("row_identity_sha256")
    if (
        not isinstance(identity_column, str)
        or not identity_column
        or identity_column not in frame.columns
        or not isinstance(expected_row_count, int)
        or isinstance(expected_row_count, bool)
        or expected_row_count < 0
        or _SHA256_RE.fullmatch(str(expected_identity_sha256 or "")) is None
    ):
        raise TypedInputReceiptError(
            "product contract lacks exact row identity authority"
        )
    identity = frame[identity_column]
    missing_count = int(identity.isna().sum())
    if missing_count:
        raise TypedInputReceiptError("opened table has missing row identity")
    normalized = identity.astype("string")
    if normalized.duplicated().any():
        raise TypedInputReceiptError("opened table has duplicate row identity")
    observed_identity_sha256 = typed_input_row_identity_sha256(identity)
    if (
        len(frame) != expected_row_count
        or observed_identity_sha256 != expected_identity_sha256
    ):
        raise TypedInputReceiptError(
            "opened table row identity does not match product contract"
        )
    return identity_column, observed_identity_sha256


def load_verified_typed_input_table(
    *,
    resolved_inputs_path: Path,
    expected_resolved_inputs_sha256: str,
    run_root: Path,
    input_key: str,
    consumer_step_id: str,
    consumer_code_sha256: str,
) -> VerifiedTypedInputLoad:
    """Open and verify exactly one Planner-bound typed table."""

    expected_manifest_sha256 = _validated_sha256(
        expected_resolved_inputs_sha256,
        name="resolved-input manifest SHA-256",
    )
    code_sha256 = _validated_sha256(
        consumer_code_sha256,
        name="consumer code SHA-256",
    )
    input_key = _validated_nonempty_string(input_key, name="input_key")
    consumer_step_id = _validated_nonempty_string(
        consumer_step_id,
        name="consumer step",
    )
    run_root = Path(run_root).expanduser()
    manifest = _read_manifest(
        Path(resolved_inputs_path),
        expected_sha256=expected_manifest_sha256,
        run_root=run_root,
    )
    binding = _binding_for_input(
        manifest,
        input_key=input_key,
        consumer_step_id=consumer_step_id,
    )
    evidence_id, artifact_sha256, binding_sha256, contract = _validate_binding_identity(
        binding, input_key=input_key
    )
    artifact_path, relative_path = _artifact_path(binding, run_root=run_root)
    frame, opened_sha256, file_size = _load_table_from_regular_file(artifact_path)
    if opened_sha256 != artifact_sha256:
        raise TypedInputReceiptError(
            "opened file does not match resolved artifact SHA-256"
        )
    identity_column, identity_sha256 = _verify_table_contract(
        frame,
        contract=contract,
    )
    return VerifiedTypedInputLoad(
        frame=frame,
        input_key=input_key,
        evidence_id=evidence_id,
        artifact_sha256=artifact_sha256,
        resolved_inputs_sha256=expected_manifest_sha256,
        resolved_input_binding_sha256=binding_sha256,
        artifact_relative_path=relative_path,
        opened_file_size_bytes=file_size,
        row_identity_column=identity_column,
        row_identity_sha256=identity_sha256,
        loaded_frame_sha256=_frame_digest(frame),
        consumer_step_id=consumer_step_id,
        consumer_code_sha256=code_sha256,
    )


def seal_typed_input_consumption(
    loaded: VerifiedTypedInputLoad,
    *,
    consumed_frame: pd.DataFrame,
) -> TypedInputConsumptionReceipt:
    """Seal consumption only for the exact unchanged host-loaded table."""

    if not isinstance(loaded, VerifiedTypedInputLoad):
        raise TypedInputReceiptError("receipt requires a verified typed-input load")
    if consumed_frame is not loaded.frame:
        raise TypedInputReceiptError(
            "consumer must use the same loaded DataFrame object"
        )
    current_frame_sha256 = _frame_digest(loaded.frame)
    if current_frame_sha256 != loaded.loaded_frame_sha256:
        raise TypedInputReceiptError("loaded DataFrame changed after verified load")
    identity = loaded.frame[loaded.row_identity_column]
    if (
        identity.isna().any()
        or identity.astype("string").duplicated().any()
        or typed_input_row_identity_sha256(identity) != loaded.row_identity_sha256
    ):
        raise TypedInputReceiptError("loaded row identity changed after verified load")
    payload: dict[str, object] = {
        "schema_version": TYPED_INPUT_CONSUMPTION_RECEIPT_SCHEMA,
        "input_key": loaded.input_key,
        "evidence_id": loaded.evidence_id,
        "artifact_sha256": loaded.artifact_sha256,
        "resolved_inputs_sha256": loaded.resolved_inputs_sha256,
        "resolved_input_binding_sha256": loaded.resolved_input_binding_sha256,
        "artifact_relative_path": loaded.artifact_relative_path,
        "opened_file_sha256": loaded.artifact_sha256,
        "opened_file_size_bytes": loaded.opened_file_size_bytes,
        "row_identity": {
            "column": loaded.row_identity_column,
            "row_count": int(len(loaded.frame)),
            "sha256": loaded.row_identity_sha256,
            "unique": True,
            "missing_count": 0,
        },
        "consumer_step_id": loaded.consumer_step_id,
        "consumer_code_sha256": loaded.consumer_code_sha256,
        "loaded_frame_sha256": loaded.loaded_frame_sha256,
    }
    payload["receipt_sha256"] = typed_input_receipt_sha256(payload)
    return TypedInputConsumptionReceipt.model_validate(payload)


def _parse_receipt(
    receipt: TypedInputConsumptionReceipt | Mapping[str, object],
) -> TypedInputConsumptionReceipt:
    if isinstance(receipt, TypedInputConsumptionReceipt):
        return receipt
    try:
        return TypedInputConsumptionReceipt.model_validate(receipt)
    except (ValidationError, TypeError, ValueError) as exc:
        raise TypedInputReceiptError("invalid receipt schema") from exc


def _receipt_contract_identity(
    receipt: TypedInputConsumptionReceipt,
) -> tuple[object, ...]:
    return (
        receipt.evidence_id,
        receipt.artifact_sha256,
        receipt.resolved_inputs_sha256,
        receipt.artifact_relative_path,
        receipt.opened_file_sha256,
        receipt.opened_file_size_bytes,
        receipt.loaded_frame_sha256,
    )


def verify_step_typed_input_receipts(
    *,
    planner_declared_inputs: Sequence[str],
    resolved_input_bindings: Mapping[str, Mapping[str, object]],
    resolved_inputs_sha256: str,
    consumer_step_id: str,
    consumer_code_sha256: str,
    receipts: Sequence[TypedInputConsumptionReceipt | Mapping[str, object]],
    row_identity_not_applicable: Sequence[str] = (),
) -> StepTypedInputReceiptVerification:
    """Verify the complete receipt set for one candidate step.

    This is a collection-level host gate.  It does not infer row keys or select
    inputs.  It checks that the Planner-declared typed file inputs, the
    host-resolved bindings, the current consumer identity, and the durable
    receipts describe the same exact set.
    """

    findings: list[ValidationFinding] = []
    verified: dict[str, TypedInputConsumptionReceipt] = {}

    try:
        expected_resolved_inputs_sha256 = _validated_sha256(
            resolved_inputs_sha256,
            name="resolved inputs SHA-256",
        )
    except TypedInputReceiptError as exc:
        expected_resolved_inputs_sha256 = ""
        findings.append(
            _finding(
                issue_code="invalid_resolved_inputs_sha256",
                message=str(exc),
            )
        )
    try:
        expected_code_sha256 = _validated_sha256(
            consumer_code_sha256,
            name="consumer code SHA-256",
        )
    except TypedInputReceiptError as exc:
        expected_code_sha256 = ""
        findings.append(
            _finding(issue_code="invalid_consumer_code_sha256", message=str(exc))
        )
    try:
        expected_step_id = _validated_nonempty_string(
            consumer_step_id,
            name="consumer step",
        )
    except TypedInputReceiptError as exc:
        expected_step_id = ""
        findings.append(_finding(issue_code="invalid_consumer_step", message=str(exc)))

    declared_file_inputs: list[str] = []
    seen_declared: set[str] = set()
    for item in planner_declared_inputs:
        if not isinstance(item, str) or not item.strip():
            findings.append(
                _finding(
                    issue_code="invalid_declared_input",
                    message="Planner-declared typed input must be a non-empty string.",
                )
            )
            continue
        if item in seen_declared:
            findings.append(
                _finding(
                    issue_code="duplicate_declared_input",
                    input_key=item,
                    message="Planner declared the same typed input more than once.",
                )
            )
            continue
        seen_declared.add(item)
        if _typed_input_product(item) is not None:
            declared_file_inputs.append(item)

    declared_set = set(declared_file_inputs)
    binding_keys = set(resolved_input_bindings)
    missing_bindings = sorted(declared_set - binding_keys)
    extra_bindings = sorted(binding_keys - declared_set)
    for input_key in missing_bindings:
        findings.append(
            _finding(
                issue_code="missing_resolved_binding",
                input_key=input_key,
                message="Planner-declared typed input has no host-resolved binding.",
            )
        )
    for input_key in extra_bindings:
        findings.append(
            _finding(
                issue_code="extra_resolved_binding",
                input_key=input_key,
                message="Host-resolved binding was not declared by the Planner.",
            )
        )

    not_applicable_inputs = set(row_identity_not_applicable)
    for input_key in sorted(not_applicable_inputs - declared_set):
        findings.append(
            _finding(
                issue_code="row_identity_not_applicable_for_unknown_input",
                input_key=input_key,
                message=(
                    "row_identity_not_applicable was asserted for an input that "
                    "is not Planner-declared for this step."
                ),
            )
        )

    receipts_by_input: dict[str, list[TypedInputConsumptionReceipt]] = {}
    for position, raw_receipt in enumerate(receipts):
        try:
            receipt = _parse_receipt(raw_receipt)
        except TypedInputReceiptError as exc:
            findings.append(
                _finding(
                    issue_code="invalid_receipt",
                    message=str(exc),
                    detail={"receipt_index": position},
                )
            )
            continue
        receipts_by_input.setdefault(receipt.input_key, []).append(receipt)

    receipt_inputs = set(receipts_by_input)
    for input_key in sorted(declared_set - receipt_inputs):
        findings.append(
            _finding(
                issue_code="missing_receipt",
                input_key=input_key,
                message="Planner-declared typed file input has no consumption receipt.",
            )
        )
    for input_key in sorted(receipt_inputs - declared_set):
        findings.append(
            _finding(
                issue_code="extra_receipt",
                input_key=input_key,
                message="Consumption receipt does not correspond to this step's declared inputs.",
            )
        )
    for input_key, grouped in sorted(receipts_by_input.items()):
        if len(grouped) > 1:
            findings.append(
                _finding(
                    issue_code="duplicate_receipt",
                    input_key=input_key,
                    message="Typed input has more than one consumption receipt.",
                    detail={"receipt_count": len(grouped)},
                )
            )

    file_identity_owner: dict[tuple[object, ...], str] = {}
    for input_key in sorted(declared_set):
        binding = resolved_input_bindings.get(input_key)
        if not isinstance(binding, Mapping):
            continue
        grouped = receipts_by_input.get(input_key, [])
        if len(grouped) != 1:
            continue
        receipt = grouped[0]

        typed_product = _typed_input_product(input_key)
        declared_kind = str(binding.get("declared_kind") or "").strip()
        product = str(binding.get("product") or "").strip()
        if typed_product != (declared_kind, product):
            findings.append(
                _finding(
                    issue_code="binding_identity_mismatch",
                    input_key=input_key,
                    message="Resolved binding does not match the Planner input identity.",
                    detail={"declared_kind": declared_kind, "product": product},
                )
            )
            continue
        if binding.get("evidence_kind") != "table":
            findings.append(
                _finding(
                    issue_code="non_table_typed_input_receipt",
                    input_key=input_key,
                    message=(
                        "Typed input receipt completeness currently verifies "
                        "file-backed table inputs only."
                    ),
                )
            )
            continue

        expected_artifact_sha = str(binding.get("sha256") or "")
        expected_evidence_id = str(binding.get("evidence_id") or "")
        expected_relative_path = str(binding.get("relative_path") or "")
        expected_binding_sha = canonical_sha256(binding)

        mismatched_fields: list[str] = []
        if receipt.evidence_id != expected_evidence_id:
            mismatched_fields.append("evidence_id")
        if receipt.artifact_sha256 != expected_artifact_sha:
            mismatched_fields.append("artifact_sha256")
        if receipt.opened_file_sha256 != expected_artifact_sha:
            mismatched_fields.append("opened_file_sha256")
        if receipt.artifact_relative_path != expected_relative_path:
            mismatched_fields.append("artifact_relative_path")
        if receipt.resolved_input_binding_sha256 != expected_binding_sha:
            mismatched_fields.append("resolved_input_binding_sha256")
        if (
            expected_resolved_inputs_sha256
            and receipt.resolved_inputs_sha256 != expected_resolved_inputs_sha256
        ):
            mismatched_fields.append("resolved_inputs_sha256")
        if expected_step_id and receipt.consumer_step_id != expected_step_id:
            mismatched_fields.append("consumer_step_id")
        if (
            expected_code_sha256
            and receipt.consumer_code_sha256 != expected_code_sha256
        ):
            mismatched_fields.append("consumer_code_sha256")
        if mismatched_fields:
            findings.append(
                _finding(
                    issue_code="receipt_binding_mismatch",
                    input_key=input_key,
                    message="Receipt does not match the current resolved binding.",
                    detail={"fields": sorted(mismatched_fields)},
                )
            )
            continue

        file_identity = _receipt_contract_identity(receipt)
        other_input = file_identity_owner.get(file_identity)
        if other_input is not None:
            findings.append(
                _finding(
                    issue_code="shared_file_identity",
                    input_key=input_key,
                    message=(
                        "The same consumed file receipt cannot authorize "
                        "multiple logical typed inputs."
                    ),
                    detail={"other_input_key": other_input},
                )
            )
            continue
        file_identity_owner[file_identity] = input_key

        contract = binding.get("product_contract")
        if not isinstance(contract, Mapping):
            if input_key in not_applicable_inputs:
                if not isinstance(
                    receipt.row_identity,
                    TypedInputRowIdentityNotApplicable,
                ):
                    findings.append(
                        _finding(
                            issue_code="row_identity_not_applicable_receipt_missing",
                            input_key=input_key,
                            message=(
                                "Host declared row identity not applicable, but "
                                "the receipt still carries row-alignment authority."
                            ),
                        )
                    )
                    continue
                verified[input_key] = receipt
                continue
            findings.append(
                _finding(
                    issue_code="missing_row_identity_contract",
                    input_key=input_key,
                    message=(
                        "Typed input requires row alignment but has no explicit "
                        "row-identity product contract."
                    ),
                )
            )
            continue

        identity_column = contract.get("row_identity_column")
        row_count = contract.get("row_count")
        row_identity_sha256 = contract.get("row_identity_sha256")
        has_exact_identity = (
            isinstance(identity_column, str)
            and bool(identity_column)
            and isinstance(row_count, int)
            and not isinstance(row_count, bool)
            and row_count >= 0
            and _SHA256_RE.fullmatch(str(row_identity_sha256 or "")) is not None
        )
        if not has_exact_identity:
            if input_key in not_applicable_inputs:
                if not isinstance(
                    receipt.row_identity,
                    TypedInputRowIdentityNotApplicable,
                ):
                    findings.append(
                        _finding(
                            issue_code="row_identity_not_applicable_receipt_missing",
                            input_key=input_key,
                            message=(
                                "Host declared row identity not applicable, but "
                                "the receipt still carries row-alignment authority."
                            ),
                        )
                    )
                    continue
                verified[input_key] = receipt
                continue
            findings.append(
                _finding(
                    issue_code="missing_row_identity_contract",
                    input_key=input_key,
                    message=(
                        "Typed input requires row alignment but its product "
                        "contract lacks exact row identity authority."
                    ),
                )
            )
            continue
        if input_key in not_applicable_inputs:
            findings.append(
                _finding(
                    issue_code="row_identity_not_applicable_conflicts_with_contract",
                    input_key=input_key,
                    message=(
                        "Host declared row identity not applicable for an input "
                        "that already has exact row-identity authority."
                    ),
                )
            )
            continue
        if not isinstance(receipt.row_identity, TypedInputRowIdentity):
            findings.append(
                _finding(
                    issue_code="row_identity_required",
                    input_key=input_key,
                    message="Receipt marks row identity not applicable for a row-aligned input.",
                )
            )
            continue
        if (
            receipt.row_identity.column != identity_column
            or receipt.row_identity.row_count != row_count
            or receipt.row_identity.sha256 != row_identity_sha256
        ):
            findings.append(
                _finding(
                    issue_code="row_identity_mismatch",
                    input_key=input_key,
                    message="Receipt row identity does not match the product contract.",
                )
            )
            continue
        verified[input_key] = receipt

    if findings:
        verified = {}
    return StepTypedInputReceiptVerification(
        findings=tuple(findings),
        verified_inputs=MappingProxyType(dict(verified)),
    )


def verify_typed_input_consumption_receipt(
    receipt: TypedInputConsumptionReceipt | Mapping[str, object],
    *,
    resolved_inputs_path: Path,
    expected_resolved_inputs_sha256: str,
    run_root: Path,
    input_key: str,
    consumer_step_id: str,
    consumer_code_sha256: str,
) -> TypedInputConsumptionReceipt:
    """Re-open authority bytes and verify one durable consumption receipt."""

    parsed = _parse_receipt(receipt)
    if parsed.input_key != input_key:
        raise TypedInputReceiptError("receipt input_key does not match consumer input")
    if parsed.consumer_step_id != consumer_step_id:
        raise TypedInputReceiptError("receipt consumer step does not match")
    if parsed.consumer_code_sha256 != consumer_code_sha256:
        raise TypedInputReceiptError("receipt consumer code does not match")
    loaded = load_verified_typed_input_table(
        resolved_inputs_path=resolved_inputs_path,
        expected_resolved_inputs_sha256=expected_resolved_inputs_sha256,
        run_root=run_root,
        input_key=input_key,
        consumer_step_id=consumer_step_id,
        consumer_code_sha256=consumer_code_sha256,
    )
    observed = seal_typed_input_consumption(loaded, consumed_frame=loaded.frame)
    if parsed.row_identity.row_count != observed.row_identity.row_count:
        raise TypedInputReceiptError("receipt row count does not match opened table")
    if parsed.row_identity != observed.row_identity:
        raise TypedInputReceiptError("receipt row identity does not match opened table")
    if parsed != observed:
        raise TypedInputReceiptError(
            "typed-input receipt does not match current authority bytes"
        )
    return parsed


__all__ = [
    "TYPED_INPUT_CONSUMPTION_RECEIPT_SCHEMA",
    "TypedInputConsumptionReceipt",
    "TypedInputReceiptError",
    "StepTypedInputReceiptVerification",
    "TypedInputRowIdentityNotApplicable",
    "TypedInputRowIdentity",
    "VerifiedTypedInputLoad",
    "load_verified_typed_input_table",
    "seal_typed_input_consumption",
    "typed_input_receipt_sha256",
    "verify_step_typed_input_receipts",
    "verify_typed_input_consumption_receipt",
]
