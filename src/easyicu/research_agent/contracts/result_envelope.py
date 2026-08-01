"""Shadow canonicalization for research-agent step results.

This module is deliberately not wired into the live execution path yet.  It
compiles already-produced, registered step outputs into one strict,
versioned envelope without changing the raw artefacts.  During the migration
period the envelope is diagnostic-only and cannot grant paper authority.
"""

from __future__ import annotations

import csv
import io
import json
import math
import mimetypes
import numbers
import os
import re
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from types import MappingProxyType
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

from ..canonical_json import (
    canonical_json_bytes,
    sha256_bytes as _sha256_bytes,
)
from .fraction_scale import is_scale_descriptor_field

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
#: The reader for a registered ``statistic:<name>`` product, as data.
#:
#: A statistic file is refused unless it parses to a JSON object, and its
#: fields are recovered by trying these aliases in order.  Generated code has
#: to write a shape it was never shown otherwise: a real run wrote a
#: one-element list of exactly the right object and was refused with
#: ``invalid_statistic_shape`` after producing every other output correctly.
#: The Coder directive renders this mapping rather than describing it, so a
#: new alias here reaches the model instead of drifting away from it.
STATISTIC_PAYLOAD_KEY_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "point estimate": _VALUE_KEYS,
        "interval lower bound": _LOW_KEYS,
        "interval upper bound": _HIGH_KEYS,
        "p value": _P_VALUE_KEYS,
        "numerator": _NUMERATOR_KEYS,
        "denominator": _DENOMINATOR_KEYS,
    }
)

#: Where a model step's summary names its term-level coefficient companion.
#:
#: The exact-replay path for robustness variants needs the fitted coefficients,
#: not just the reported effect, and it finds them by this key.  It used to
#: read ``diagnostic_companions.coefficients`` alone and fall back to a fixed
#: ``coefficients.csv`` -- a filename no producer has ever written.  Measured
#: over every recorded run: 334 step summaries, of which 23 carry
#: ``model_contracts``; ``coefficient_table`` appears 10 times (the
#: deterministic owner writes it), ``coefficient_file`` 3 times (a Coder-written
#: summary), ``diagnostic_companions`` once, and ``coefficients.csv`` exists
#: zero times.  So the reader resolved 1 of 23, and the replay path it guards
#: was unreachable in every other run.
#:
#: Order is preference, not permission: a summary declaring more than one is
#: read by the first, and any of them being unreadable is a refusal rather
#: than a fall-through to the next.
MODEL_SUMMARY_COEFFICIENT_TABLE_KEYS: tuple[str, ...] = (
    "diagnostic_companions.coefficients",
    "coefficient_table",
    "coefficient_file",
)


def model_summary_coefficient_filename(summary: Mapping[str, Any]) -> str | None:
    """Return the coefficient companion filename a model summary declares.

    ``None`` means *this summary does not name one*, which is a refusal: the
    caller must not guess a filename, because a guessed name that happens to
    exist would bind the replay to a table nobody declared.
    """

    if not isinstance(summary, Mapping):
        return None
    for key in MODEL_SUMMARY_COEFFICIENT_TABLE_KEYS:
        declared: Any = summary
        for part in key.split("."):
            if declared is None:
                break
            if not isinstance(declared, Mapping):
                # The summary *has* this path and it is not navigable, which
                # is a broken declaration.  Moving on to the next spelling
                # would answer with a different table than the one it tried
                # to name, and the substitution would be invisible.
                return None
            declared = declared.get(part)
        if declared is None:
            continue
        if not isinstance(declared, str):
            return None
        filename = declared.strip()
        if (
            not filename
            or PurePosixPath(filename).name != filename
            or not filename.lower().endswith(".csv")
        ):
            return None
        return filename
    return None


#: Where a model summary states the analysis its primary estimate came from.
#:
#: The complete-case equivalence proof needs the exposure, the outcome and the
#: adjustment set, because a model fitted on one adjustment set and a
#: complete-case restriction taken over a different one are different analyses.
#: It read ``summary["analysis_definition"]`` and nothing else.
#:
#: Measured over every recorded run: 358 step summaries, 27 carry
#: ``model_contracts``, 12 of those state exposure + outcome + covariates, and
#: exactly **one** writes ``analysis_definition`` -- a one-off Coder summary. A
#: repository-wide search for the name returns the reader and a single test
#: fixture, both added in the same commit. The host published a contract only
#: its own test could satisfy, so the proof was unreachable in production from
#: the day it was written.
#:
#: The deterministic primary owner states the same three facts as flat keys, so
#: they are published here rather than a second nested spelling being demanded
#: of it. Order is preference, not permission.
MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY = "analysis_definition"
MODEL_SUMMARY_EXPOSURE_KEYS: tuple[str, ...] = ("exposure", "exposure_source")
MODEL_SUMMARY_OUTCOME_KEYS: tuple[str, ...] = ("outcome",)
MODEL_SUMMARY_COVARIATE_KEYS: tuple[str, ...] = ("covariates", "adjustment_covariates")


def _clean_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    return name or None


def _clean_covariates(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    cleaned: list[str] = []
    for item in value:
        name = _clean_name(item)
        if name is None:
            return None
        cleaned.append(name)
    return cleaned


def _first_declared(source: Mapping[str, Any], keys: Sequence[str]) -> tuple[bool, Any]:
    for key in keys:
        if key in source:
            return True, source[key]
    return False, None


def model_summary_analysis_definition(
    summary: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the exposure/outcome/covariates a model summary states, or ``None``.

    ``None`` means *this summary does not state its analysis*, which is a
    refusal. The caller must not reconstruct the adjustment set from anywhere
    else: a proof taken over a different set than the model used is a different
    analysis reported under the same label, and no downstream check can see it.

    An empty covariate list is a real answer -- an unadjusted primary model
    states exactly that -- so it is returned rather than treated as absent.
    """

    if not isinstance(summary, Mapping):
        return None

    nested = summary.get(MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY)
    if nested is not None:
        if not isinstance(nested, Mapping):
            return None
        source: Mapping[str, Any] = nested
    else:
        source = summary

    has_exposure, raw_exposure = _first_declared(source, MODEL_SUMMARY_EXPOSURE_KEYS)
    has_outcome, raw_outcome = _first_declared(source, MODEL_SUMMARY_OUTCOME_KEYS)
    if not has_exposure or not has_outcome:
        return None
    exposure = _clean_name(raw_exposure)
    outcome = _clean_name(raw_outcome)
    if exposure is None or outcome is None:
        return None

    # Two spellings of the adjustment set that disagree is the one case where
    # answering at all would be worse than refusing: whichever is picked, the
    # summary itself says the other is also true, and the proof would silently
    # be taken over a set the model may not have used.
    declared: list[list[str]] = []
    for key in MODEL_SUMMARY_COVARIATE_KEYS:
        if key not in source:
            continue
        covariates = _clean_covariates(source[key])
        if covariates is None:
            return None
        declared.append(covariates)
    if not declared:
        return None
    if any(entry != declared[0] for entry in declared[1:]):
        return None

    return {"exposure": exposure, "outcome": outcome, "covariates": list(declared[0])}


_MAX_SCALARS = 5_000
_MAX_DEPTH = 12
_MAX_TABLE_BYTES = 16 * 1024 * 1024
_MAX_TABLE_COLUMNS = 256
_MAX_TABLE_ROWS = 20_000
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_SCALAR_PATH_TOKEN_RE = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


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
        "percent_to_fraction",
        "profile_registered_table",
        "bind_registered_model_contract",
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


class CanonicalTableProfile(_StrictModel):
    product_id: str
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: StrictInt = Field(ge=0)
    columns: tuple[str, ...]
    semantic_roles: tuple[
        Literal[
            "effect_estimate",
            "generic",
            "group_summary",
            "missingness",
            "model_diagnostic",
            "population_flow",
            "prevalence",
        ],
        ...,
    ]


class CanonicalPopulationCount(_StrictModel):
    count_id: str
    role: Literal[
        "analyzed",
        "cohort",
        "complete_case",
        "denominator",
        "dropped",
        "eligible",
        "event",
        "excluded",
        "non_event",
        "numerator",
        "source",
    ]
    value: StrictInt = Field(ge=0)
    source_product_id: str
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CanonicalGroupCount(_StrictModel):
    group_id: str = Field(min_length=1, max_length=128)
    value: StrictInt = Field(ge=0)
    source_product_id: str
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StepPopulationResult(_StrictModel):
    eligible_n: StrictInt | None = Field(default=None, ge=0)
    analyzed_n: StrictInt | None = Field(default=None, ge=0)
    counts: tuple[CanonicalPopulationCount, ...] = ()
    group_counts: tuple[CanonicalGroupCount, ...] = ()


class StepVariableBindings(_StrictModel):
    exposures: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()


class StepMissingVariableResult(_StrictModel):
    variable: str = Field(min_length=1, max_length=128)
    denominator_n: StrictInt | None = Field(default=None, ge=0)
    nonmissing_n: StrictInt | None = Field(default=None, ge=0)
    missing_n: StrictInt = Field(ge=0)
    missing_fraction: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    source_product_id: str
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StepMissingDataResult(_StrictModel):
    declared_policy_ref: str | None = None
    executed_policy: str | None = None
    before_n: StrictInt | None = Field(default=None, ge=0)
    after_n: StrictInt | None = Field(default=None, ge=0)
    variables: tuple[StepMissingVariableResult, ...] = ()


class StepModelDiagnostic(_StrictModel):
    diagnostic_id: str
    status: str
    model_family: str | None = None
    fit_method: str | None = None
    converged: StrictBool | None = None
    separation_detected: StrictBool | None = None
    penalized: StrictBool | None = None
    analyzed_n: StrictInt | None = Field(default=None, ge=0)
    event_n: StrictInt | None = Field(default=None, ge=0)
    controlled_source_artifact_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class StepResultEnvelope(_StrictModel):
    """Strict shadow representation of one current step result."""

    schema_version: Literal[
        "easyicu.step_result_envelope/1",
        "easyicu.step_result_envelope/2",
    ] = "easyicu.step_result_envelope/2"
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
    tables: tuple[CanonicalTableProfile, ...] = ()
    statistics: tuple[CanonicalStatistic, ...] = ()
    observed_scalars: tuple[CanonicalScalar, ...] = ()
    normalization_receipts: tuple[NormalizationReceipt, ...] = ()
    normalization_issues: tuple[NormalizationIssue, ...] = ()
    shadow: Literal[True] = True
    paper_authorized: Literal[False] = False


def rebuild_observed_scalar_tree(
    scalars: Sequence[CanonicalScalar],
) -> dict[str, Any] | None:
    """Rebuild the canonical summary tree, rejecting ambiguous scalar paths."""

    root: dict[str, Any] = {}
    for scalar in scalars:
        tokens: list[str | int] = []
        for match in _SCALAR_PATH_TOKEN_RE.finditer(scalar.field_path):
            raw_index = match.group(2)
            tokens.append(
                int(raw_index) if raw_index is not None else str(match.group(1))
            )
        if not tokens or not isinstance(tokens[0], str):
            return None
        rendered = str(tokens[0])
        for token in tokens[1:]:
            rendered += f"[{token}]" if isinstance(token, int) else f".{token}"
        if rendered != scalar.field_path:
            return None

        current: Any = root
        for index, token in enumerate(tokens):
            final = index == len(tokens) - 1
            if isinstance(token, str):
                if not isinstance(current, dict):
                    return None
                if final:
                    if token in current and current[token] != scalar.value:
                        return None
                    current[token] = scalar.value
                    continue
                expected = list if isinstance(tokens[index + 1], int) else dict
                child = current.setdefault(token, expected())
            else:
                if not isinstance(current, list):
                    return None
                while len(current) <= token:
                    current.append(None)
                if final:
                    if current[token] is not None and current[token] != scalar.value:
                        return None
                    current[token] = scalar.value
                    continue
                expected = list if isinstance(tokens[index + 1], int) else dict
                child = current[token]
                if child is None:
                    child = expected()
                    current[token] = child
            if not isinstance(child, expected):
                return None
            current = child
    return root


def _canonical_json_bytes(payload: Any) -> bytes:
    # The result-envelope wire contract intentionally includes one final LF.
    return canonical_json_bytes(payload, trailing_newline=True)


def _model_content_sha256(envelope: StepResultEnvelope) -> str:
    excluded = {"content_sha256"}
    if envelope.schema_version == "easyicu.step_result_envelope/1":
        excluded.add("tables")
    payload = envelope.model_dump(mode="json", exclude=excluded)
    return _sha256_bytes(_canonical_json_bytes(payload))


def verify_step_result_envelope(envelope: StepResultEnvelope) -> bool:
    """Verify the self-declared digest of an in-memory envelope."""

    return envelope.content_sha256 == _model_content_sha256(envelope)


def rebind_step_result_status(
    envelope: StepResultEnvelope,
    *,
    status: str,
) -> StepResultEnvelope:
    """Return a verified copy of an envelope re-bound to a final step status.

    Only the ``status`` field and the recomputed ``content_sha256`` change.
    No cohort, CSV, JSON, or artifact bytes are re-read: the rebind operates
    solely on the already-compiled in-memory envelope, so canonicalization
    never scales a second time with cohort size.

    The input envelope's declared digest MUST already verify -- a tampered or
    otherwise inconsistent envelope is rejected rather than laundered into a
    fresh digest.  ``status`` must be a non-empty string; the terminal-vs-non
    -terminal policy is owned by the caller (the sidecar lifecycle), not by
    this digest primitive.
    """

    if not isinstance(status, str) or not status.strip():
        raise ValueError("step-result status rebinding requires a non-empty status")
    if not verify_step_result_envelope(envelope):
        raise ValueError(
            "refusing to rebind a step-result envelope with an invalid content digest"
        )
    rebound = envelope.model_copy(update={"status": status})
    rebound = rebound.model_copy(
        update={"content_sha256": _model_content_sha256(rebound)}
    )
    if not verify_step_result_envelope(rebound):
        raise ValueError("rebound step-result envelope failed self-verification")
    return rebound


def _type_label(value: Any) -> str:
    value_type = type(value)
    module = value_type.__module__
    if module in {"builtins", "pathlib"} or module.startswith(("numpy", "pandas")):
        return f"{module}.{value_type.__name__}"
    return "unsupported"


def _safe_string_field(field_path: str) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9]+", field_path.lower()) if token}
    leaf = re.sub(
        r"[^a-z0-9]+",
        "_",
        re.split(r"[.\[]", field_path.lower())[-1],
    ).strip("_")
    return bool(tokens & _SAFE_STRING_FIELD_TOKENS) or is_scale_descriptor_field(leaf)


def _path_field(field_path: str) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9]+", field_path.lower()) if token}
    return bool(tokens & {"file", "files", "path", "paths"})


def _container_relative_scalar_path(
    rendered: str,
    *,
    container_output_roots: Sequence[str],
) -> str | None:
    """Return a safe relative form for an exact container output path."""

    absolute = PurePosixPath(rendered)
    if not absolute.is_absolute():
        return None
    for raw_root in container_output_roots:
        root = PurePosixPath(raw_root)
        if not root.is_absolute():
            continue
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            continue
        if not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            continue
        return relative.as_posix()
    return None


def _coerce_scalar(
    value: Any,
    *,
    field_path: str,
    source: Literal["step_summary", "statistic_artifact"],
    product_id: str | None,
    authorized_path_refs: Mapping[str, str],
    container_output_roots: Sequence[str],
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
            container_relative = _container_relative_scalar_path(
                value.as_posix(),
                container_output_roots=container_output_roots,
            )
            if bound_ref is not None:
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
            elif container_relative is not None:
                normalized = container_relative
                path_was_authorized = True
                receipts.append(
                    NormalizationReceipt(
                        operation="container_path_to_relative",
                        field_path=field_path,
                        before_type="container.absolute_path",
                        after_type="host.relative_path",
                        product_id=product_id,
                    )
                )
            else:
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
            container_relative = _container_relative_scalar_path(
                normalized,
                container_output_roots=container_output_roots,
            )
            if bound_ref is not None:
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
            elif container_relative is not None:
                normalized = container_relative
                path_was_authorized = True
                receipts.append(
                    NormalizationReceipt(
                        operation="container_path_to_relative",
                        field_path=field_path,
                        before_type="container.absolute_path",
                        after_type="host.relative_path",
                        product_id=product_id,
                    )
                )
            else:
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
    container_output_roots: Sequence[str],
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
                    container_output_roots=container_output_roots,
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
                    container_output_roots=container_output_roots,
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
        container_output_roots=container_output_roots,
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
        container_output_roots=(),
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


@dataclass
class _CompiledRegisteredOutputs:
    tables: list[CanonicalTableProfile] = field(default_factory=list)
    statistics: list[CanonicalStatistic] = field(default_factory=list)
    population_counts: list[CanonicalPopulationCount] = field(default_factory=list)
    group_counts: list[CanonicalGroupCount] = field(default_factory=list)
    missing_variables: list[StepMissingVariableResult] = field(default_factory=list)
    model_diagnostics: list[StepModelDiagnostic] = field(default_factory=list)
    exposures: set[str] = field(default_factory=set)
    outcomes: set[str] = field(default_factory=set)
    covariates: set[str] = field(default_factory=set)


def _safe_csv_rows(
    raw: bytes,
    *,
    product_id: str,
    profile_required: bool,
    issues: list[NormalizationIssue],
) -> tuple[tuple[str, ...], list[dict[str, str]]] | None:
    if len(raw) > _MAX_TABLE_BYTES:
        issues.append(
            NormalizationIssue(
                severity="error" if profile_required else "warning",
                code="registered_table_too_large",
                message=(
                    "A registered result table exceeded the canonical byte limit."
                    if profile_required
                    else (
                        "An opaque data artifact exceeded the canonical table "
                        "profiling byte limit; its digest-bound artifact reference "
                        "was retained without expanding row-level content."
                    )
                ),
                product_id=product_id,
            )
        )
        return None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_table_encoding",
                message="A registered table was not strict UTF-8 CSV.",
                product_id=product_id,
            )
        )
        return None
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_table_csv",
                message="A registered table was not valid CSV.",
                product_id=product_id,
            )
        )
        return None
    if not rows:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="empty_registered_table",
                message="A registered table did not contain a header row.",
                product_id=product_id,
            )
        )
        return None
    header = tuple(value.strip() for value in rows[0])
    if (
        not header
        or len(header) > _MAX_TABLE_COLUMNS
        or any(not value or len(value) > 128 for value in header)
        or len(set(header)) != len(header)
    ):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_table_header",
                message="A registered table had empty, duplicate, or excessive columns.",
                product_id=product_id,
            )
        )
        return None
    body = rows[1:]
    if len(body) > _MAX_TABLE_ROWS:
        issues.append(
            NormalizationIssue(
                severity="error" if profile_required else "warning",
                code="registered_table_row_limit_exceeded",
                message=(
                    "A registered result table exceeded the canonical row limit."
                    if profile_required
                    else (
                        "An opaque data artifact exceeded the canonical table "
                        "profiling row limit; its digest-bound artifact reference "
                        "was retained without expanding row-level content."
                    )
                ),
                product_id=product_id,
            )
        )
        return None
    mapped: list[dict[str, str]] = []
    for index, row in enumerate(body):
        if len(row) != len(header):
            issues.append(
                NormalizationIssue(
                    severity="error",
                    code="invalid_registered_table_row_width",
                    message="A registered table row did not match its header width.",
                    field_path=f"row[{index}]",
                    product_id=product_id,
                )
            )
            return None
        mapped.append(dict(zip(header, row, strict=True)))
    return header, mapped


def _csv_number(
    row: Mapping[str, str],
    key: str,
    *,
    product_id: str,
    row_index: int,
    issues: list[NormalizationIssue],
) -> int | float | None:
    raw = row.get(key)
    if raw is None or not raw.strip():
        return None
    rendered = raw.strip()
    try:
        value = float(rendered)
    except ValueError:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_numeric_cell",
                message="A recognized numeric table cell was not numeric.",
                field_path=f"row[{row_index}].{key}",
                product_id=product_id,
            )
        )
        return None
    if not math.isfinite(value):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="nonfinite_registered_numeric_cell",
                message="A recognized numeric table cell was not finite.",
                field_path=f"row[{row_index}].{key}",
                product_id=product_id,
            )
        )
        return None
    if re.fullmatch(r"[+-]?\d+", rendered):
        return int(rendered)
    return value


def _csv_count(
    row: Mapping[str, str],
    key: str,
    *,
    product_id: str,
    row_index: int,
    issues: list[NormalizationIssue],
) -> int | None:
    value = _csv_number(
        row,
        key,
        product_id=product_id,
        row_index=row_index,
        issues=issues,
    )
    if value is None:
        return None
    if value < 0 or not float(value).is_integer():
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_count",
                message="A recognized count cell was negative or non-integral.",
                field_path=f"row[{row_index}].{key}",
                product_id=product_id,
            )
        )
        return None
    return int(value)


def _csv_bool(
    row: Mapping[str, str],
    key: str,
    *,
    product_id: str,
    row_index: int,
    issues: list[NormalizationIssue],
) -> bool | None:
    raw = row.get(key)
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip().lower()
    if normalized not in {"true", "false"}:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_boolean_cell",
                message="A recognized boolean table cell was not true or false.",
                field_path=f"row[{row_index}].{key}",
                product_id=product_id,
            )
        )
        return None
    return normalized == "true"


def _safe_table_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        return None
    return normalized


def _table_semantic_roles(
    columns: set[str],
    rows: Sequence[Mapping[str, str]],
) -> tuple[str, ...]:
    roles: set[str] = set()
    if (
        {"variable", "group", "denominator_n"}.issubset(columns)
        and "schema_version" in columns
        and all(
            not row.get("schema_version")
            or row["schema_version"].strip()
            in {
                "easyicu.table_one_result/1",
                "easyicu.table_one_result/2",
                "easyicu.table_one_result/3",
            }
            for row in rows
        )
    ):
        roles.add("group_summary")
    if {"n_at_start", "n_remaining", "n_excluded"}.issubset(columns) or {
        "n_at_start_rows",
        "n_remaining_rows",
        "n_excluded_rows",
    }.issubset(columns):
        roles.add("population_flow")
    if (
        "variable" in columns
        and "missing_n" in columns
        and columns.intersection({"n_full", "n_total", "cohort_n", "denominator_n"})
        and columns.intersection({"missing_pct", "missing_percent", "fraction_missing"})
    ):
        roles.add("missingness")
    if (
        columns.intersection({"estimate", "odds_ratio"})
        and columns.intersection({"ci_low", "ci_95_low"})
        and columns.intersection({"ci_high", "ci_95_high"})
    ):
        roles.add("effect_estimate")
    if (
        columns.intersection({"numerator", "positive_n"})
        and columns.intersection({"denominator", "denominator_n"})
        and columns.intersection({"prevalence", "proportion"})
    ):
        roles.add("prevalence")
    if {"model_id", "fit_status"}.issubset(columns):
        roles.add("model_diagnostic")
    return tuple(sorted(roles or {"generic"}))


def _row_identity(row: Mapping[str, str], row_index: int) -> str:
    for key in (
        "model_id",
        "summary",
        "criterion_id",
        "stage",
        "variable",
        "outcome",
        "analysis",
    ):
        identifier = _safe_table_identifier(row.get(key))
        if identifier is not None:
            return identifier
    return f"row_{row_index}"


def _normalized_fraction(
    row: Mapping[str, str],
    *,
    fraction_keys: Sequence[str],
    percent_keys: Sequence[str],
    product_id: str,
    row_index: int,
    field_name: str,
    receipts: list[NormalizationReceipt],
    issues: list[NormalizationIssue],
) -> float | None:
    candidates: list[float] = []
    for key in fraction_keys:
        value = _csv_number(
            row,
            key,
            product_id=product_id,
            row_index=row_index,
            issues=issues,
        )
        if value is not None:
            candidates.append(float(value))
    for key in percent_keys:
        value = _csv_number(
            row,
            key,
            product_id=product_id,
            row_index=row_index,
            issues=issues,
        )
        if value is not None:
            candidates.append(float(value) / 100.0)
            receipts.append(
                NormalizationReceipt(
                    operation="percent_to_fraction",
                    field_path=f"row[{row_index}].{key}",
                    before_type="percent",
                    after_type="fraction",
                    product_id=product_id,
                )
            )
    if not candidates:
        return None
    if any(value < 0.0 or value > 1.0 for value in candidates):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_fraction",
                message=f"A recognized {field_name} was outside [0, 1].",
                field_path=f"row[{row_index}]",
                product_id=product_id,
            )
        )
        return None
    first = candidates[0]
    if any(
        not math.isclose(value, first, rel_tol=1e-9, abs_tol=1e-12)
        for value in candidates[1:]
    ):
        issues.append(
            NormalizationIssue(
                severity="error",
                code="conflicting_registered_fraction",
                message=f"Equivalent {field_name} fields contained conflicting values.",
                field_path=f"row[{row_index}]",
                product_id=product_id,
            )
        )
        return None
    return first


def _reported_fraction_rounding_tolerance(
    row: Mapping[str, str],
    *,
    fraction_keys: Sequence[str],
    percent_keys: Sequence[str],
) -> float:
    """Return half of the coarsest reported unit on the fraction scale."""

    tolerance = 1e-12
    for keys, scale in ((fraction_keys, Decimal(1)), (percent_keys, Decimal(100))):
        for key in keys:
            raw = str(row.get(key) or "").strip()
            if not raw:
                continue
            try:
                value = Decimal(raw)
            except InvalidOperation:
                continue
            if not value.is_finite():
                continue
            quantum = Decimal(1).scaleb(value.as_tuple().exponent)
            tolerance = max(
                tolerance,
                float(abs(quantum) / (Decimal(2) * scale)),
            )
    return tolerance


def _compile_registered_table(
    *,
    artifact: StepArtifactRef,
    raw: bytes,
    receipts: list[NormalizationReceipt],
    issues: list[NormalizationIssue],
) -> _CompiledRegisteredOutputs:
    compiled = _CompiledRegisteredOutputs()
    parsed = _safe_csv_rows(
        raw,
        product_id=artifact.product_id,
        profile_required=artifact.kind == "table",
        issues=issues,
    )
    if parsed is None:
        return compiled
    header, rows = parsed
    columns = set(header)
    roles = _table_semantic_roles(columns, rows)
    compiled.tables.append(
        CanonicalTableProfile(
            product_id=artifact.product_id,
            source_artifact_sha256=artifact.sha256,
            row_count=len(rows),
            columns=header,
            semantic_roles=roles,
        )
    )
    receipts.append(
        NormalizationReceipt(
            operation="profile_registered_table",
            field_path="output_files",
            before_type="registered.csv",
            after_type="canonical.table_profile",
            product_id=artifact.product_id,
        )
    )

    is_group_summary = "group_summary" in roles
    group_values: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        identity = _row_identity(row, row_index)
        if is_group_summary:
            group_id = str(row.get("group") or "").strip()
            denominator = _csv_count(
                row,
                "denominator_n",
                product_id=artifact.product_id,
                row_index=row_index,
                issues=issues,
            )
            if group_id and denominator is not None:
                previous = group_values.get(group_id)
                if previous is not None and previous != denominator:
                    issues.append(
                        NormalizationIssue(
                            severity="error",
                            code="conflicting_registered_group_count",
                            message="Repeated group rows had conflicting denominators.",
                            field_path=f"row[{row_index}].denominator_n",
                            product_id=artifact.product_id,
                        )
                    )
                else:
                    group_values[group_id] = denominator
            continue

        if "population_flow" in roles:
            group_id = str(row.get("exposure_level") or "").strip()
            grouped = group_id and group_id.lower() not in {"all", "overall"}
            for key, role in (
                ("n_at_start", "source"),
                ("n_at_start_rows", "source"),
                ("n_remaining", "eligible"),
                ("n_remaining_rows", "eligible"),
                ("n_excluded", "excluded"),
                ("n_excluded_rows", "excluded"),
            ):
                value = _csv_count(
                    row,
                    key,
                    product_id=artifact.product_id,
                    row_index=row_index,
                    issues=issues,
                )
                if value is None:
                    continue
                if grouped and role == "eligible":
                    group_values[group_id] = value
                    continue
                if grouped:
                    continue
                compiled.population_counts.append(
                    CanonicalPopulationCount(
                        count_id=f"{artifact.name}:{identity}:{key}",
                        role=role,
                        value=value,
                        source_product_id=artifact.product_id,
                        source_artifact_sha256=artifact.sha256,
                    )
                )

        if "missingness" in roles:
            variable = _safe_table_identifier(row.get("variable"))
            missing_n = _csv_count(
                row,
                "missing_n",
                product_id=artifact.product_id,
                row_index=row_index,
                issues=issues,
            )
            if variable is not None and missing_n is not None:
                explicit_full_n = None
                for key in ("n_full", "n_total", "cohort_n"):
                    explicit_full_n = _csv_count(
                        row,
                        key,
                        product_id=artifact.product_id,
                        row_index=row_index,
                        issues=issues,
                    )
                    if explicit_full_n is not None:
                        break
                nonmissing_n = None
                for key in ("n_nonmissing", "nonmissing_n"):
                    nonmissing_n = _csv_count(
                        row,
                        key,
                        product_id=artifact.product_id,
                        row_index=row_index,
                        issues=issues,
                    )
                    if nonmissing_n is not None:
                        break
                partition_full_n = (
                    nonmissing_n + missing_n if nonmissing_n is not None else None
                )
                # A cohort splits into observed, missing, and *not applicable*.
                # Reading only the first two asserts that the value is
                # semantically applicable to every subject, which is false for
                # any conditional quantity. ``MissingnessProfile`` already types
                # this: ``not_applicable_n`` is "rows where absence is expected
                # under the typed semantics", defaulting to 0 -- so a row that
                # does not declare one is unaffected.
                #
                # Measured 2026-07-29: a real E1 step was failed for the
                # ``death_time`` row of its own audit -- n_total=1000,
                # n_nonmissing=102, missing_n=0, not_applicable_n=898. The row
                # was right: 102 patients died, so only they have a death time,
                # and calling the other 898 "missing" would claim 89.8 % of
                # death times were absent when those patients simply did not
                # die. Under the full partition it reconciles exactly:
                # 102 + 0 + 898 = 1000.
                not_applicable_n = _csv_count(
                    row,
                    "not_applicable_n",
                    product_id=artifact.product_id,
                    row_index=row_index,
                    issues=issues,
                )
                if (
                    explicit_full_n is not None
                    and partition_full_n is not None
                    and explicit_full_n != partition_full_n + (not_applicable_n or 0)
                ):
                    issues.append(
                        NormalizationIssue(
                            severity="error",
                            code="inconsistent_registered_missingness_partition",
                            message=(
                                "Explicit full denominator disagreed with the "
                                "non-missing plus missing plus not-applicable "
                                "partition."
                            ),
                            field_path=f"row[{row_index}]",
                            product_id=artifact.product_id,
                        )
                    )
                denominator_n = (
                    explicit_full_n if explicit_full_n is not None else partition_full_n
                )
                if denominator_n is None:
                    # ``denominator_n`` is intentionally last: descriptive rows
                    # often use it for the non-missing summary denominator while
                    # reporting missingness against the full cohort.  An explicit
                    # full-cohort field or the complete nonmissing/missing
                    # partition is therefore stronger authority.
                    denominator_n = _csv_count(
                        row,
                        "denominator_n",
                        product_id=artifact.product_id,
                        row_index=row_index,
                        issues=issues,
                    )
                missing_fraction = _normalized_fraction(
                    row,
                    fraction_keys=("fraction_missing",),
                    percent_keys=("missing_pct", "missing_percent"),
                    product_id=artifact.product_id,
                    row_index=row_index,
                    field_name="missing fraction",
                    receipts=receipts,
                    issues=issues,
                )
                if (
                    denominator_n is not None
                    and denominator_n > 0
                    and missing_fraction is not None
                    and not math.isclose(
                        missing_n / denominator_n,
                        missing_fraction,
                        rel_tol=1e-8,
                        abs_tol=_reported_fraction_rounding_tolerance(
                            row,
                            fraction_keys=("fraction_missing",),
                            percent_keys=("missing_pct", "missing_percent"),
                        ),
                    )
                ):
                    issues.append(
                        NormalizationIssue(
                            severity="error",
                            code="inconsistent_registered_missingness",
                            message="Missing count and fraction did not share one denominator.",
                            field_path=f"row[{row_index}]",
                            product_id=artifact.product_id,
                        )
                    )
                compiled.missing_variables.append(
                    StepMissingVariableResult(
                        variable=variable,
                        denominator_n=denominator_n,
                        nonmissing_n=nonmissing_n,
                        missing_n=missing_n,
                        missing_fraction=missing_fraction,
                        source_product_id=artifact.product_id,
                        source_artifact_sha256=artifact.sha256,
                    )
                )
                role = str(row.get("role") or "").strip().lower()
                if role == "primary_exposure":
                    compiled.exposures.add(variable)
                elif role in {"target_outcome", "outcome"}:
                    compiled.outcomes.add(variable)
                elif role in {"adjustment", "covariate"}:
                    compiled.covariates.add(variable)

        if "effect_estimate" in roles or "prevalence" in roles:
            effect_scale = _safe_table_identifier(row.get("effect_scale"))
            value: int | float | None
            if "prevalence" in roles:
                value = _csv_number(
                    row,
                    "prevalence",
                    product_id=artifact.product_id,
                    row_index=row_index,
                    issues=issues,
                )
            elif effect_scale and "odds_ratio" in effect_scale:
                value = _csv_number(
                    row,
                    "odds_ratio",
                    product_id=artifact.product_id,
                    row_index=row_index,
                    issues=issues,
                )
                if value is None:
                    value = _csv_number(
                        row,
                        "estimate",
                        product_id=artifact.product_id,
                        row_index=row_index,
                        issues=issues,
                    )
            else:
                value = _csv_number(
                    row,
                    "estimate",
                    product_id=artifact.product_id,
                    row_index=row_index,
                    issues=issues,
                )
            low = _csv_number(
                row,
                "ci_low" if "ci_low" in columns else "ci_95_low",
                product_id=artifact.product_id,
                row_index=row_index,
                issues=issues,
            )
            high = _csv_number(
                row,
                "ci_high" if "ci_high" in columns else "ci_95_high",
                product_id=artifact.product_id,
                row_index=row_index,
                issues=issues,
            )
            if "prevalence" in roles and low is None and high is None:
                low = _normalized_fraction(
                    row,
                    fraction_keys=(),
                    percent_keys=("ci_lower_pct",),
                    product_id=artifact.product_id,
                    row_index=row_index,
                    field_name="interval lower bound",
                    receipts=receipts,
                    issues=issues,
                )
                high = _normalized_fraction(
                    row,
                    fraction_keys=(),
                    percent_keys=("ci_upper_pct",),
                    product_id=artifact.product_id,
                    row_index=row_index,
                    field_name="interval upper bound",
                    receipts=receipts,
                    issues=issues,
                )
            if value is not None:
                statistic_id = f"{artifact.name}:{identity}:{row_index}"
                numerator = None
                denominator = None
                if "prevalence" in roles:
                    for key in ("numerator", "positive_n"):
                        numerator = _csv_count(
                            row,
                            key,
                            product_id=artifact.product_id,
                            row_index=row_index,
                            issues=issues,
                        )
                        if numerator is not None:
                            break
                    for key in ("denominator", "denominator_n"):
                        denominator = _csv_count(
                            row,
                            key,
                            product_id=artifact.product_id,
                            row_index=row_index,
                            issues=issues,
                        )
                        if denominator is not None:
                            break
                compiled.statistics.append(
                    CanonicalStatistic(
                        statistic_id=statistic_id,
                        product_id=artifact.product_id,
                        value=value,
                        interval_low=low,
                        interval_high=high,
                        p_value=_csv_number(
                            row,
                            "p_value",
                            product_id=artifact.product_id,
                            row_index=row_index,
                            issues=issues,
                        ),
                        effect_scale=effect_scale,
                        unit=_safe_table_identifier(row.get("outcome_unit"))
                        or _safe_table_identifier(row.get("unit")),
                        numerator=numerator,
                        denominator=denominator,
                        source_artifact_sha256=artifact.sha256,
                    )
                )

        for key, role in (
            ("n_full", "source"),
            ("cohort_n", "cohort"),
            ("n_complete_case", "complete_case"),
            ("n_dropped", "dropped"),
            ("n", "analyzed"),
            ("event_n", "event"),
            ("non_event_n", "non_event"),
            ("numerator", "numerator"),
            ("denominator", "denominator"),
        ):
            if key not in columns:
                continue
            if key in {"numerator", "denominator"} and "prevalence" not in roles:
                continue
            value = _csv_count(
                row,
                key,
                product_id=artifact.product_id,
                row_index=row_index,
                issues=issues,
            )
            if value is not None:
                compiled.population_counts.append(
                    CanonicalPopulationCount(
                        count_id=f"{artifact.name}:{identity}:{row_index}:{key}",
                        role=role,
                        value=value,
                        source_product_id=artifact.product_id,
                        source_artifact_sha256=artifact.sha256,
                    )
                )

        outcome = _safe_table_identifier(row.get("outcome"))
        if outcome is not None:
            compiled.outcomes.add(outcome)
        exposure = _safe_table_identifier(
            row.get("exposure_source") or row.get("exposure")
        )
        if exposure is not None and str(row.get("exposure_role") or "").lower() in {
            "primary",
            "exposure",
            "",
        }:
            compiled.exposures.add(exposure)
        if {"model_id", "fit_status"}.issubset(columns):
            model_id = _safe_table_identifier(row.get("model_id"))
            fit_status = _safe_table_identifier(row.get("fit_status"))
            if model_id is not None and fit_status is not None:
                compiled.model_diagnostics.append(
                    StepModelDiagnostic(
                        diagnostic_id=f"{model_id}@{artifact.product_id}",
                        status=fit_status,
                        model_family=_safe_table_identifier(row.get("model_family")),
                        fit_method=_safe_table_identifier(row.get("fit_method")),
                        converged=_csv_bool(
                            row,
                            "converged",
                            product_id=artifact.product_id,
                            row_index=row_index,
                            issues=issues,
                        ),
                        separation_detected=_csv_bool(
                            row,
                            (
                                "separation_detected"
                                if "separation_detected" in columns
                                else "separation"
                            ),
                            product_id=artifact.product_id,
                            row_index=row_index,
                            issues=issues,
                        ),
                        penalized=_csv_bool(
                            row,
                            "penalized",
                            product_id=artifact.product_id,
                            row_index=row_index,
                            issues=issues,
                        ),
                        analyzed_n=_csv_count(
                            row,
                            "n",
                            product_id=artifact.product_id,
                            row_index=row_index,
                            issues=issues,
                        ),
                        event_n=_csv_count(
                            row,
                            "event_n",
                            product_id=artifact.product_id,
                            row_index=row_index,
                            issues=issues,
                        ),
                        controlled_source_artifact_sha256=artifact.sha256,
                    )
                )

    for group_id, value in sorted(group_values.items()):
        if len(group_id) > 128:
            issues.append(
                NormalizationIssue(
                    severity="error",
                    code="invalid_registered_group_identity",
                    message="A registered group identity exceeded the canonical limit.",
                    product_id=artifact.product_id,
                )
            )
            continue
        compiled.group_counts.append(
            CanonicalGroupCount(
                group_id=group_id,
                value=value,
                source_product_id=artifact.product_id,
                source_artifact_sha256=artifact.sha256,
            )
        )
    return compiled


def _strict_optional_bool(
    payload: Mapping[str, Any],
    key: str,
    *,
    product_id: str,
    issues: list[NormalizationIssue],
) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not bool:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_model_boolean",
                message="A model diagnostic boolean was not a strict JSON boolean.",
                field_path=key,
                product_id=product_id,
            )
        )
        return None
    return value


def _strict_optional_json_count(
    payload: Mapping[str, Any],
    key: str,
    *,
    product_id: str,
    issues: list[NormalizationIssue],
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_model_count",
                message="A model diagnostic count was not a non-negative integer.",
                field_path=key,
                product_id=product_id,
            )
        )
        return None
    return value


def _compile_registered_model_artifact(
    *,
    artifact: StepArtifactRef,
    raw: bytes,
    receipts: list[NormalizationReceipt],
    issues: list[NormalizationIssue],
) -> _CompiledRegisteredOutputs:
    compiled = _CompiledRegisteredOutputs()
    if not artifact.relative_path.lower().endswith(".json"):
        return compiled
    try:
        payload = _safe_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return compiled
    if not isinstance(payload, Mapping):
        return compiled
    contract = payload.get("model_contract")
    if not isinstance(contract, Mapping):
        return compiled
    model_id = _safe_table_identifier(contract.get("model_id"))
    fit_status = _safe_table_identifier(contract.get("fit_status"))
    if model_id is None or fit_status is None:
        issues.append(
            NormalizationIssue(
                severity="error",
                code="invalid_registered_model_contract",
                message="A registered model contract lacked a safe model id or fit status.",
                product_id=artifact.product_id,
            )
        )
        return compiled
    compiled.model_diagnostics.append(
        StepModelDiagnostic(
            diagnostic_id=f"{model_id}@{artifact.product_id}",
            status=fit_status,
            model_family=_safe_table_identifier(contract.get("model_family")),
            fit_method=_safe_table_identifier(contract.get("fit_method")),
            converged=_strict_optional_bool(
                contract,
                "converged",
                product_id=artifact.product_id,
                issues=issues,
            ),
            separation_detected=_strict_optional_bool(
                contract,
                "separation_detected",
                product_id=artifact.product_id,
                issues=issues,
            ),
            penalized=_strict_optional_bool(
                contract,
                "penalized",
                product_id=artifact.product_id,
                issues=issues,
            ),
            analyzed_n=_strict_optional_json_count(
                contract,
                "n",
                product_id=artifact.product_id,
                issues=issues,
            ),
            event_n=_strict_optional_json_count(
                contract,
                "event_n",
                product_id=artifact.product_id,
                issues=issues,
            ),
            controlled_source_artifact_sha256=artifact.sha256,
        )
    )
    outcome = _safe_table_identifier(contract.get("outcome"))
    exposure = _safe_table_identifier(
        contract.get("exposure_source") or contract.get("exposure")
    )
    if outcome is not None:
        compiled.outcomes.add(outcome)
    if exposure is not None:
        compiled.exposures.add(exposure)
    receipts.append(
        NormalizationReceipt(
            operation="bind_registered_model_contract",
            field_path="model_contract",
            before_type="registered.json",
            after_type="canonical.model_diagnostic",
            product_id=artifact.product_id,
        )
    )
    return compiled


def _extend_compilation(
    target: _CompiledRegisteredOutputs,
    source: _CompiledRegisteredOutputs,
) -> None:
    target.tables.extend(source.tables)
    target.statistics.extend(source.statistics)
    target.population_counts.extend(source.population_counts)
    target.group_counts.extend(source.group_counts)
    target.missing_variables.extend(source.missing_variables)
    target.model_diagnostics.extend(source.model_diagnostics)
    target.exposures.update(source.exposures)
    target.outcomes.update(source.outcomes)
    target.covariates.update(source.covariates)


def _unique_count(
    counts: Sequence[CanonicalPopulationCount],
    roles: set[str],
) -> int | None:
    values = {item.value for item in counts if item.role in roles}
    return next(iter(values)) if len(values) == 1 else None


def _selected_count(
    counts: Sequence[CanonicalPopulationCount],
    *,
    preferred_role: str,
    fallback_roles: set[str] = frozenset(),
) -> int | None:
    preferred = [item.value for item in counts if item.role == preferred_role]
    if preferred:
        return preferred[-1]
    return _unique_count(counts, fallback_roles)


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
        container_output_roots=container_output_roots,
        receipts=receipts,
        issues=issues,
    )
    artifacts: list[StepArtifactRef] = []
    statistics: list[CanonicalStatistic] = []
    compiled_outputs = _CompiledRegisteredOutputs()
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
        if relative.suffix.lower() == ".csv" and kind in {"artifact", "table"}:
            _extend_compilation(
                compiled_outputs,
                _compile_registered_table(
                    artifact=artifact,
                    raw=raw_bytes,
                    receipts=receipts,
                    issues=issues,
                ),
            )
        if kind == "artifact" and relative.suffix.lower() == ".json":
            _extend_compilation(
                compiled_outputs,
                _compile_registered_model_artifact(
                    artifact=artifact,
                    raw=raw_bytes,
                    receipts=receipts,
                    issues=issues,
                ),
            )

    statistics.extend(compiled_outputs.statistics)
    population = None
    if compiled_outputs.population_counts or compiled_outputs.group_counts:
        population = StepPopulationResult(
            eligible_n=_selected_count(
                compiled_outputs.population_counts,
                preferred_role="eligible",
                fallback_roles={"cohort"},
            ),
            analyzed_n=_selected_count(
                compiled_outputs.population_counts,
                preferred_role="analyzed",
                fallback_roles={"complete_case"},
            ),
            counts=tuple(compiled_outputs.population_counts),
            group_counts=tuple(compiled_outputs.group_counts),
        )
    variables = None
    if (
        compiled_outputs.exposures
        or compiled_outputs.outcomes
        or compiled_outputs.covariates
    ):
        variables = StepVariableBindings(
            exposures=tuple(sorted(compiled_outputs.exposures)),
            outcomes=tuple(sorted(compiled_outputs.outcomes)),
            covariates=tuple(sorted(compiled_outputs.covariates)),
        )
    missing_data = None
    if compiled_outputs.missing_variables:
        missing_data = StepMissingDataResult(
            before_n=_selected_count(
                compiled_outputs.population_counts,
                preferred_role="source",
                fallback_roles={"cohort"},
            ),
            after_n=_selected_count(
                compiled_outputs.population_counts,
                preferred_role="analyzed",
                fallback_roles={"complete_case"},
            ),
            variables=tuple(compiled_outputs.missing_variables),
        )

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
        population=population,
        variables=variables,
        missing_data=missing_data,
        model_diagnostics=tuple(compiled_outputs.model_diagnostics),
        input_evidence_refs=tuple(sorted(set(authorized_path_refs.values()))),
        artifacts=tuple(artifacts),
        tables=tuple(compiled_outputs.tables),
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
    "CanonicalGroupCount",
    "CanonicalPopulationCount",
    "CanonicalScalar",
    "CanonicalStatistic",
    "CanonicalTableProfile",
    "MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY",
    "MODEL_SUMMARY_COEFFICIENT_TABLE_KEYS",
    "MODEL_SUMMARY_COVARIATE_KEYS",
    "MODEL_SUMMARY_EXPOSURE_KEYS",
    "MODEL_SUMMARY_OUTCOME_KEYS",
    "NormalizationIssue",
    "NormalizationReceipt",
    "StepArtifactRef",
    "StepMissingDataResult",
    "StepMissingVariableResult",
    "StepModelDiagnostic",
    "StepPopulationResult",
    "STATISTIC_PAYLOAD_KEY_ALIASES",
    "StepResultEnvelope",
    "StepVariableBindings",
    "model_summary_analysis_definition",
    "model_summary_coefficient_filename",
    "normalize_step_result_shadow",
    "rebind_step_result_status",
    "rebuild_observed_scalar_tree",
    "verify_step_result_envelope",
    "write_shadow_step_result_envelope",
]
