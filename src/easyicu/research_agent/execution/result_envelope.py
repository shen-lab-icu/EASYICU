"""Shadow canonicalization for research-agent step results.

This module is deliberately not wired into the live execution path yet.  It
compiles already-produced, registered step outputs into one strict,
versioned envelope without changing the raw artefacts.  During the migration
period the envelope is diagnostic-only and cannot grant paper authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import numbers
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)

JsonScalar = Union[StrictBool, StrictInt, StrictFloat, StrictStr, None]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRODUCT_RE = re.compile(
    r"^(?P<kind>[a-z][a-z0-9_.-]*):(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)$"
)
_SAFE_STRING_FIELD_TOKENS = frozenset(
    {
        "analysis",
        "complete",
        "effect",
        "event",
        "exposure",
        "family",
        "fit",
        "group",
        "kind",
        "method",
        "name",
        "outcome",
        "role",
        "scale",
        "status",
        "step",
        "unit",
        "variable",
    }
)
_VALUE_KEYS = (
    "value",
    "estimate",
    "result",
    "prevalence",
    "proportion",
    "percentage",
    "percent",
    "pct",
)
_LOW_KEYS = ("ci_low", "ci_95_low", "ci_lower", "lower")
_HIGH_KEYS = ("ci_high", "ci_95_high", "ci_upper", "upper")
_P_VALUE_KEYS = ("p_value", "p")
_NUMERATOR_KEYS = ("numerator", "positive_n", "event_n", "count")
_DENOMINATOR_KEYS = ("denominator", "denominator_n", "n_total", "total_n")
_MAX_SCALARS = 5_000
_MAX_DEPTH = 12


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CanonicalScalar(_StrictModel):
    field_path: str = Field(min_length=1, max_length=500)
    value: JsonScalar
    source: Literal["step_summary", "statistic_artifact"]
    product_id: str | None = None


class NormalizationReceipt(_StrictModel):
    operation: Literal[
        "bind_declared_product_identity",
        "container_path_to_relative",
        "numpy_scalar_to_builtin",
        "nullable_to_null",
        "path_to_relative",
        "authorized_path_to_evidence_ref",
    ]
    field_path: str = Field(min_length=1, max_length=500)
    before_type: str = Field(min_length=1, max_length=100)
    after_type: str = Field(min_length=1, max_length=100)
    product_id: str | None = None


class NormalizationIssue(_StrictModel):
    severity: Literal["warning", "error"]
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=500)
    field_path: str | None = Field(default=None, max_length=500)
    product_id: str | None = None


class StepArtifactRef(_StrictModel):
    product_id: str
    kind: str
    name: str
    relative_path: str
    media_type: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)


class CanonicalStatistic(_StrictModel):
    statistic_id: str
    product_id: str
    value: StrictInt | StrictFloat | None = None
    interval_low: StrictInt | StrictFloat | None = None
    interval_high: StrictInt | StrictFloat | None = None
    p_value: StrictInt | StrictFloat | None = None
    effect_scale: str | None = None
    unit: str | None = None
    numerator: StrictInt | StrictFloat | None = None
    denominator: StrictInt | StrictFloat | None = None
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fields: tuple[CanonicalScalar, ...] = ()


class StepPopulationResult(_StrictModel):
    eligible_n: StrictInt | None = Field(default=None, ge=0)
    analyzed_n: StrictInt | None = Field(default=None, ge=0)
    group_counts: tuple[CanonicalScalar, ...] = ()


class StepVariableBindings(_StrictModel):
    exposures: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()


class StepMissingDataResult(_StrictModel):
    declared_policy_ref: str | None = None
    executed_policy: str | None = None
    before_n: StrictInt | None = Field(default=None, ge=0)
    after_n: StrictInt | None = Field(default=None, ge=0)


class StepModelDiagnostic(_StrictModel):
    diagnostic_id: str
    status: str
    controlled_source_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class StepResultEnvelope(_StrictModel):
    """Strict shadow representation of one current step result."""

    schema_version: Literal["easyicu.step_result_envelope/1"] = (
        "easyicu.step_result_envelope/1"
    )
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step_id: str = Field(min_length=1, max_length=300)
    status: str | None = Field(default=None, max_length=100)
    planned_analysis_role: str | None = Field(default=None, max_length=100)
    product_contract_ref: str | None = Field(default=None, max_length=500)
    source_summary_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    raw_summary_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    ledger_record_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    population: StepPopulationResult | None = None
    variables: StepVariableBindings | None = None
    missing_data: StepMissingDataResult | None = None
    model_diagnostics: tuple[StepModelDiagnostic, ...] = ()
    input_evidence_refs: tuple[str, ...] = ()
    artifacts: tuple[StepArtifactRef, ...] = ()
    statistics: tuple[CanonicalStatistic, ...] = ()
    observed_scalars: tuple[CanonicalScalar, ...] = ()
    normalization_receipts: tuple[NormalizationReceipt, ...] = ()
    normalization_issues: tuple[NormalizationIssue, ...] = ()
    shadow: Literal[True] = True
    paper_authorized: Literal[False] = False


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _model_content_sha256(envelope: StepResultEnvelope) -> str:
    payload = envelope.model_dump(mode="json", exclude={"content_sha256"})
    return _sha256_bytes(_canonical_json_bytes(payload))


def verify_step_result_envelope(envelope: StepResultEnvelope) -> bool:
    """Verify the self-declared digest of an in-memory envelope."""

    return envelope.content_sha256 == _model_content_sha256(envelope)


def _type_label(value: Any) -> str:
    value_type = type(value)
    module = value_type.__module__
    if module in {"builtins", "pathlib"} or module.startswith(("numpy", "pandas")):
        return f"{module}.{value_type.__name__}"
    return "unsupported"


def _safe_string_field(field_path: str) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9]+", field_path.lower()) if token}
    return bool(tokens & _SAFE_STRING_FIELD_TOKENS)


def _path_field(field_path: str) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9]+", field_path.lower()) if token}
    return bool(tokens & {"file", "files", "path", "paths"})


def _coerce_scalar(
    value: Any,
    *,
    field_path: str,
    source: Literal["step_summary", "statistic_artifact"],
    product_id: str | None,
    authorized_path_refs: Mapping[str, str],
    receipts: list[NormalizationReceipt],
    issues: list[NormalizationIssue],
) -> JsonScalar | object:
    path_was_authorized = False
    if value is None or isinstance(value, (bool, int, float, str)):
        normalized: Any = value
    elif type(value).__module__.startswith("numpy") and callable(
        getattr(value, "item", None)
    ):
        normalized = value.item()
        if not isinstance(normalized, (bool, int, float, str)):
            issues.append(
                NormalizationIssue(
                    severity="warning",
                    code="unsupported_scalar_type",
                    message="A NumPy value did not reduce to a supported scalar.",
                    field_path=field_path,
                    product_id=product_id,
                )
            )
            return _OMIT
        receipts.append(
            NormalizationReceipt(
                operation="numpy_scalar_to_builtin",
                field_path=field_path,
                before_type=_type_label(value),
                after_type=_type_label(normalized),
                product_id=product_id,
            )
        )
    elif (
        type(value).__module__.startswith("pandas") and type(value).__name__ == "NAType"
    ):
        receipts.append(
            NormalizationReceipt(
                operation="nullable_to_null",
                field_path=field_path,
                before_type=_type_label(value),
                after_type="builtins.NoneType",
                product_id=product_id,
            )
        )
        normalized = None
    elif isinstance(value, Path):
        if value.is_absolute():
            bound_ref = authorized_path_refs.get(value.as_posix())
            if bound_ref is None:
                issues.append(
                    NormalizationIssue(
                        severity="error",
                        code="absolute_unbound_path",
                        message="An absolute path was not bound to current evidence.",
                        field_path=field_path,
                        product_id=product_id,
                    )
                )
                return _OMIT
            normalized = bound_ref
            path_was_authorized = True
            receipts.append(
                NormalizationReceipt(
                    operation="authorized_path_to_evidence_ref",
                    field_path=field_path,
                    before_type=_type_label(value),
                    after_type="evidence.ref",
                    product_id=product_id,
                )
            )
        else:
            normalized = value.as_posix()
            receipts.append(
                NormalizationReceipt(
                    operation="path_to_relative",
                    field_path=field_path,
                    before_type=_type_label(value),
                    after_type="builtins.str",
                    product_id=product_id,
                )
            )
    elif isinstance(value, numbers.Integral):
        normalized = int(value)
        receipts.append(
            NormalizationReceipt(
                operation="numpy_scalar_to_builtin",
                field_path=field_path,
                before_type=_type_label(value),
                after_type="builtins.int",
                product_id=product_id,
            )
        )
    elif isinstance(value, numbers.Real):
        normalized = float(value)
        receipts.append(
            NormalizationReceipt(
                operation="numpy_scalar_to_builtin",
                field_path=field_path,
                before_type=_type_label(value),
                after_type="builtins.float",
                product_id=product_id,
            )
        )
    else:
        issues.append(
            NormalizationIssue(
                severity="warning",
                code="unsupported_scalar_type",
                message="A scalar with an unsupported runtime type was omitted.",
                field_path=field_path,
                product_id=product_id,
            )
        )
        return _OMIT

    if isinstance(normalized, float) and not math.isfinite(normalized):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="nonfinite_scalar",
                message="A non-finite numeric scalar cannot enter the canonical envelope.",
                field_path=field_path,
                product_id=product_id,
            )
        )
        return _OMIT
    if isinstance(normalized, str):
        if _path_field(field_path) and normalized.startswith(("/", "\\")):
            bound_ref = authorized_path_refs.get(normalized)
            if bound_ref is None:
                issues.append(
                    NormalizationIssue(
                        severity="error",
                        code="absolute_unbound_path",
                        message="An absolute path was not bound to current evidence.",
                        field_path=field_path,
                        product_id=product_id,
                    )
                )
                return _OMIT
            normalized = bound_ref
            path_was_authorized = True
            receipts.append(
                NormalizationReceipt(
                    operation="authorized_path_to_evidence_ref",
                    field_path=field_path,
                    before_type="path.string",
                    after_type="evidence.ref",
                    product_id=product_id,
                )
            )
        if not path_was_authorized and (
            len(normalized) > 128 or not _safe_string_field(field_path)
        ):
            issues.append(
                NormalizationIssue(
                    severity="warning",
                    code="untyped_string_omitted",
                    message="Untyped free text was omitted from the canonical envelope.",
                    field_path=field_path,
                    product_id=product_id,
                )
            )
            return _OMIT
    return normalized


_OMIT = object()


def _flatten_scalars(
    payload: Any,
    *,
    source: Literal["step_summary", "statistic_artifact"],
    product_id: str | None,
    authorized_path_refs: Mapping[str, str],
    receipts: list[NormalizationReceipt],
    issues: list[NormalizationIssue],
    prefix: str = "",
    depth: int = 0,
) -> list[CanonicalScalar]:
    if depth > _MAX_DEPTH:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="maximum_normalization_depth_exceeded",
                message="Nested output exceeded the canonical normalization depth.",
                field_path=prefix or None,
                product_id=product_id,
            )
        )
        return []
    if isinstance(payload, Mapping):
        scalars: list[CanonicalScalar] = []
        for raw_key, value in sorted(payload.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_key, str):
                issues.append(
                    NormalizationIssue(
                        severity="error",
                        code="non_string_mapping_key",
                        message="Canonical output mappings require string keys.",
                        field_path=prefix or None,
                        product_id=product_id,
                    )
                )
                continue
            if not prefix and raw_key == "output_files":
                continue
            child_path = f"{prefix}.{raw_key}" if prefix else raw_key
            scalars.extend(
                _flatten_scalars(
                    value,
                    source=source,
                    product_id=product_id,
                    authorized_path_refs=authorized_path_refs,
                    receipts=receipts,
                    issues=issues,
                    prefix=child_path,
                    depth=depth + 1,
                )
            )
            if len(scalars) >= _MAX_SCALARS:
                issues.append(
                    NormalizationIssue(
                        severity="error",
                        code="maximum_scalar_count_exceeded",
                        message="Output exceeded the canonical scalar-count limit.",
                        field_path=child_path,
                        product_id=product_id,
                    )
                )
                return scalars[:_MAX_SCALARS]
        return scalars
    if isinstance(payload, (list, tuple)):
        scalars = []
        for index, value in enumerate(payload):
            child_path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            scalars.extend(
                _flatten_scalars(
                    value,
                    source=source,
                    product_id=product_id,
                    authorized_path_refs=authorized_path_refs,
                    receipts=receipts,
                    issues=issues,
                    prefix=child_path,
                    depth=depth + 1,
                )
            )
            if len(scalars) >= _MAX_SCALARS:
                return scalars[:_MAX_SCALARS]
        return scalars
    scalar = _coerce_scalar(
        payload,
        field_path=prefix or "value",
        source=source,
        product_id=product_id,
        authorized_path_refs=authorized_path_refs,
        receipts=receipts,
        issues=issues,
    )
    if scalar is _OMIT:
        return []
    return [
        CanonicalScalar(
            field_path=prefix or "value",
            value=scalar,
            source=source,
            product_id=product_id,
        )
    ]


def _safe_json_loads(raw: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(raw.decode("utf-8"), parse_constant=reject_constant)


def _artifact_relative_path(
    raw_path: Any,
    *,
    output_dir: Path,
    container_output_roots: Sequence[str],
    product_id: str,
    receipts: list[NormalizationReceipt],
    issues: list[NormalizationIssue],
) -> Path | None:
    if not isinstance(raw_path, (str, os.PathLike)):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_product_path_type",
                message="A registered product path must be a string or path-like value.",
                field_path="output_files",
                product_id=product_id,
            )
        )
        return None
    rendered = os.fspath(raw_path)
    candidate_relative: Path | None = None
    if PurePosixPath(rendered).is_absolute():
        absolute = PurePosixPath(rendered)
        for raw_root in container_output_roots:
            root = PurePosixPath(raw_root)
            if not root.is_absolute():
                continue
            try:
                relative = absolute.relative_to(root)
            except ValueError:
                continue
            candidate_relative = Path(*relative.parts)
            receipts.append(
                NormalizationReceipt(
                    operation="container_path_to_relative",
                    field_path="output_files",
                    before_type="container.absolute_path",
                    after_type="host.relative_path",
                    product_id=product_id,
                )
            )
            break
        if candidate_relative is None:
            issues.append(
                NormalizationIssue(
                    severity="error",
                    code="absolute_unbound_product_path",
                    message="An absolute product path was not bound to an authorized container output root.",
                    field_path="output_files",
                    product_id=product_id,
                )
            )
            return None
    else:
        candidate_relative = Path(rendered)
    if not candidate_relative.parts or any(
        part in {"", ".", ".."} for part in candidate_relative.parts
    ):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="unsafe_relative_product_path",
                message="A product path was empty or contained an unsafe segment.",
                field_path="output_files",
                product_id=product_id,
            )
        )
        return None
    output_root = output_dir.resolve()
    candidate = output_dir / candidate_relative
    if any(
        (output_dir.joinpath(*candidate_relative.parts[:index])).is_symlink()
        for index in range(1, len(candidate_relative.parts) + 1)
    ):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="symlink_product_path",
                message="A registered product path traversed a symbolic link.",
                field_path="output_files",
                product_id=product_id,
            )
        )
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(output_root)
    except (FileNotFoundError, OSError, ValueError):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="missing_or_outside_product_path",
                message="A registered product was missing or outside the output directory.",
                field_path="output_files",
                product_id=product_id,
            )
        )
        return None
    if not resolved.is_file():
        issues.append(
            NormalizationIssue(
                severity="error",
                code="non_file_product_path",
                message="A registered product did not resolve to a regular file.",
                field_path="output_files",
                product_id=product_id,
            )
        )
        return None
    return candidate_relative


def _first_finite_number(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    product_id: str,
    field_name: str,
    issues: list[NormalizationIssue],
) -> int | float | None:
    values: list[int | float] = []
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(
                NormalizationIssue(
                    severity="error",
                    code="invalid_statistic_numeric_field",
                    message="A typed statistic numeric field was not a finite number.",
                    field_path=key,
                    product_id=product_id,
                )
            )
            continue
        if not math.isfinite(float(value)):
            issues.append(
                NormalizationIssue(
                    severity="error",
                    code="nonfinite_statistic_field",
                    message="A typed statistic contained a non-finite numeric value.",
                    field_path=key,
                    product_id=product_id,
                )
            )
            continue
        values.append(value)
    if not values:
        return None
    first = float(values[0])
    if any(
        not math.isclose(float(value), first, rel_tol=1e-12, abs_tol=1e-12)
        for value in values[1:]
    ):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="conflicting_statistic_fields",
                message=f"Equivalent {field_name} fields contained conflicting values.",
                product_id=product_id,
            )
        )
        return None
    return values[0]


def _safe_identifier(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized and len(normalized) <= 128:
            return normalized
    return None


def _parse_statistic(
    *,
    product_id: str,
    statistic_name: str,
    artifact: StepArtifactRef,
    artifact_bytes: bytes,
    receipts: list[NormalizationReceipt],
    issues: list[NormalizationIssue],
) -> CanonicalStatistic | None:
    if not artifact.relative_path.lower().endswith(".json"):
        issues.append(
            NormalizationIssue(
                severity="warning",
                code="unsupported_statistic_media_type",
                message="A non-JSON statistic was registered but not parsed in envelope v1.",
                product_id=product_id,
            )
        )
        return None
    try:
        payload = _safe_json_loads(artifact_bytes)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_statistic_json",
                message="A registered statistic was not strict UTF-8 JSON.",
                product_id=product_id,
            )
        )
        return None
    if not isinstance(payload, Mapping):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_statistic_shape",
                message="A registered statistic JSON product was not an object.",
                product_id=product_id,
            )
        )
        return None
    declared = _safe_identifier(payload, ("name", "statistic"))
    if declared is not None and declared != statistic_name:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="conflicting_statistic_identity",
                message="The statistic payload identity conflicted with the declared product.",
                product_id=product_id,
            )
        )
        return None
    if declared is None:
        receipts.append(
            NormalizationReceipt(
                operation="bind_declared_product_identity",
                field_path="name",
                before_type="missing",
                after_type="planner_declared_product",
                product_id=product_id,
            )
        )
    fields = _flatten_scalars(
        payload,
        source="statistic_artifact",
        product_id=product_id,
        authorized_path_refs={},
        receipts=receipts,
        issues=issues,
    )
    value = _first_finite_number(
        payload,
        _VALUE_KEYS,
        product_id=product_id,
        field_name="point estimate",
        issues=issues,
    )
    low = _first_finite_number(
        payload,
        _LOW_KEYS,
        product_id=product_id,
        field_name="interval lower bound",
        issues=issues,
    )
    high = _first_finite_number(
        payload,
        _HIGH_KEYS,
        product_id=product_id,
        field_name="interval upper bound",
        issues=issues,
    )
    p_value = _first_finite_number(
        payload,
        _P_VALUE_KEYS,
        product_id=product_id,
        field_name="p value",
        issues=issues,
    )
    numerator = _first_finite_number(
        payload,
        _NUMERATOR_KEYS,
        product_id=product_id,
        field_name="numerator",
        issues=issues,
    )
    denominator = _first_finite_number(
        payload,
        _DENOMINATOR_KEYS,
        product_id=product_id,
        field_name="denominator",
        issues=issues,
    )
    return CanonicalStatistic(
        statistic_id=statistic_name,
        product_id=product_id,
        value=value,
        interval_low=low,
        interval_high=high,
        p_value=p_value,
        effect_scale=_safe_identifier(
            payload,
            ("effect_scale", "effect_measure", "scale"),
        ),
        unit=_safe_identifier(payload, ("unit", "units")),
        numerator=numerator,
        denominator=denominator,
        source_artifact_sha256=artifact.sha256,
        fields=tuple(fields),
    )


def normalize_step_result_shadow(
    *,
    step_id: str,
    step_summary: Any,
    output_dir: Path,
    status: str | None = None,
    planned_analysis_role: str | None = None,
    product_contract_ref: str | None = None,
    source_summary_bytes: bytes | None = None,
    raw_summary_artifact_bytes: bytes | None = None,
    ledger_record_sha256: str | None = None,
    container_output_roots: Sequence[str] = (),
    authorized_path_refs: Mapping[str, str] | None = None,
) -> StepResultEnvelope:
    """Compile one result into a strict, non-authoritative shadow envelope.

    Raw files are read but never written.  Product identities come exclusively
    from the registered ``output_files`` mapping, not from filename heuristics.
    """

    if ledger_record_sha256 is not None and not _SHA256_RE.fullmatch(
        ledger_record_sha256
    ):
        raise ValueError("ledger_record_sha256 must be a lowercase SHA-256 digest")
    receipts: list[NormalizationReceipt] = []
    issues: list[NormalizationIssue] = []
    authorized_path_refs = dict(authorized_path_refs or {})
    summary_mapping = step_summary if isinstance(step_summary, Mapping) else {}
    if not isinstance(step_summary, Mapping):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_step_summary_shape",
                message="The current step summary was not an object.",
            )
        )
    observed_scalars = _flatten_scalars(
        summary_mapping,
        source="step_summary",
        product_id=None,
        authorized_path_refs=authorized_path_refs,
        receipts=receipts,
        issues=issues,
    )
    artifacts: list[StepArtifactRef] = []
    statistics: list[CanonicalStatistic] = []
    output_files = summary_mapping.get("output_files")
    if output_files is not None and not isinstance(output_files, Mapping):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_output_files_shape",
                message="The output_files registration was not an object.",
                field_path="output_files",
            )
        )
        output_files = {}
    for raw_product_id, raw_path in sorted(
        (output_files or {}).items(),
        key=lambda item: str(item[0]),
    ):
        product_id = str(raw_product_id or "").strip()
        match = _PRODUCT_RE.fullmatch(product_id)
        if match is None:
            issues.append(
                NormalizationIssue(
                    severity="error",
                    code="invalid_product_identity",
                    message="A registered product did not use a valid kind:name identity.",
                    field_path="output_files",
                    product_id=product_id or None,
                )
            )
            continue
        relative = _artifact_relative_path(
            raw_path,
            output_dir=output_dir,
            container_output_roots=container_output_roots,
            product_id=product_id,
            receipts=receipts,
            issues=issues,
        )
        if relative is None:
            continue
        source = output_dir / relative
        raw_bytes = source.read_bytes()
        kind = match.group("kind")
        name = match.group("name")
        artifact = StepArtifactRef(
            product_id=product_id,
            kind=kind,
            name=name,
            relative_path=relative.as_posix(),
            media_type=mimetypes.guess_type(relative.name)[0],
            sha256=_sha256_bytes(raw_bytes),
            byte_size=len(raw_bytes),
        )
        artifacts.append(artifact)
        if kind == "statistic":
            parsed = _parse_statistic(
                product_id=product_id,
                statistic_name=name,
                artifact=artifact,
                artifact_bytes=raw_bytes,
                receipts=receipts,
                issues=issues,
            )
            if parsed is not None:
                statistics.append(parsed)

    if source_summary_bytes is None:
        try:
            source_summary_bytes = _canonical_json_bytes(summary_mapping)
        except (TypeError, ValueError):
            issues.append(
                NormalizationIssue(
                    severity="warning",
                    code="source_summary_digest_unavailable",
                    message=(
                        "An in-memory summary could not be serialized without "
                        "coercion; no raw source digest was invented."
                    ),
                )
            )
    provisional = StepResultEnvelope(
        content_sha256="0" * 64,
        step_id=step_id,
        status=status,
        planned_analysis_role=planned_analysis_role,
        product_contract_ref=product_contract_ref,
        source_summary_sha256=(
            _sha256_bytes(source_summary_bytes)
            if source_summary_bytes is not None
            else None
        ),
        raw_summary_artifact_sha256=(
            _sha256_bytes(raw_summary_artifact_bytes)
            if raw_summary_artifact_bytes is not None
            else None
        ),
        ledger_record_sha256=ledger_record_sha256,
        input_evidence_refs=tuple(sorted(set(authorized_path_refs.values()))),
        artifacts=tuple(artifacts),
        statistics=tuple(statistics),
        observed_scalars=tuple(observed_scalars),
        normalization_receipts=tuple(receipts),
        normalization_issues=tuple(issues),
    )
    return provisional.model_copy(
        update={"content_sha256": _model_content_sha256(provisional)}
    )


def write_shadow_step_result_envelope(
    envelope: StepResultEnvelope,
    target_path: Path,
    *,
    source_output_dir: Path | None = None,
) -> None:
    """Atomically write a verified shadow envelope outside raw outputs."""

    if not verify_step_result_envelope(envelope):
        raise ValueError("refusing to write an envelope with an invalid content digest")
    target = target_path.resolve()
    if source_output_dir is not None:
        source_root = source_output_dir.resolve()
        try:
            target.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise ValueError("shadow envelopes must not be written into raw outputs")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(envelope.model_dump(mode="json"))
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".shadow",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


__all__ = [
    "CanonicalScalar",
    "CanonicalStatistic",
    "NormalizationIssue",
    "NormalizationReceipt",
    "StepArtifactRef",
    "StepMissingDataResult",
    "StepModelDiagnostic",
    "StepPopulationResult",
    "StepResultEnvelope",
    "StepVariableBindings",
    "normalize_step_result_shadow",
    "verify_step_result_envelope",
    "write_shadow_step_result_envelope",
]
