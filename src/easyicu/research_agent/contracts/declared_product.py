"""Execution-time boundary between planned and produced step products.

The planner owns scientific scope through typed ``kind:product`` entries in
``AnalysisStep.expected_outputs``.  A successful script must realise those
products in its machine-readable summary, and it may not silently widen a
non-figure/non-effect step into a publication figure or effect analysis.

This module only validates declarations and registrations.  It never chooses
an exposure, outcome, cohort, estimator, or analysis method.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..repair_registry import is_sealed_renderer_repair, repair_metadata_for
from ..schema import AnalysisStep, ValidationFinding
from ..trajectory.plan_contract import (
    trajectory_role_result_findings,
    trajectory_role_scope_summary_findings,
)
from .typed_schema import (
    _tabular_artifact_columns,
    merge_host_table_contract,
    typed_product_prompt_facts,
    typed_product_schema_receipt,
)

_FIGURE_KINDS = frozenset({"figure", "plot", "chart", "fig", "heatmap"})
_FAILED_STATUSES = frozenset(
    {
        "blocked",
        "error",
        "failed",
        "execution_failed",
        "contract_failed",
        "fail_closed",
        "failed_closed",
        "repair_failed",
        "skipped",
        "skipped_dependency_failed",
    }
)
_OUTPUT_CONTAINER_KEYS = frozenset(
    {"output_files", "output_artifacts", "outputs", "figure_files"}
)
_PRODUCT_DESCRIPTOR_FIELDS = frozenset(
    {
        "filename",
        "kind",
        "name",
        "path",
        "product_type",
        "relative_path",
        "role",
        "value",
    }
)
_DIRECT_FIGURE_KEYS = frozenset({"figure_file", "figure_path"})
_FIGURE_SUFFIXES = frozenset({".png", ".svg", ".pdf", ".tif", ".tiff"})
_KNOWN_FILE_SUFFIXES = frozenset(
    {
        *_FIGURE_SUFFIXES,
        ".csv",
        ".tsv",
        ".parquet",
        ".feather",
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".log",
        ".pkl",
        ".pickle",
        ".joblib",
        ".npy",
        ".npz",
    }
)

# Typed-product capabilities are shared by planning, Coder guidance, and the
# execution-time evidence resolver.  Keep the parser permissive so callers can
# report an unsupported ``kind:name`` token precisely, but keep the capabilities
# themselves closed: a plan must not appear executable when the runtime cannot
# bind the declared product to current evidence.
RUNTIME_TYPED_INPUT_EVIDENCE_KINDS: Mapping[str, frozenset[str]] = {
    "artifact": frozenset({"log", "table"}),
    "dataset": frozenset({"table"}),
    "figure": frozenset({"figure"}),
    "log": frozenset({"log"}),
    "manifest": frozenset({"log"}),
    "model": frozenset({"log"}),
    "statistic": frozenset({"statistic"}),
    "table": frozenset({"table"}),
}
RUNTIME_BINDABLE_TYPED_INPUT_KINDS = frozenset(RUNTIME_TYPED_INPUT_EVIDENCE_KINDS)

# ``audit`` and ``test`` have explicit output-registration contracts, while a
# terminal ``report`` is materialised by the writer rather than consumed as a
# step input.  They are therefore valid plan outputs but are deliberately not
# runtime-bindable inputs.  Adding a new kind requires implementing its physical
# registration and (if consumable) digest-bound evidence mapping first.
PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS = frozenset(
    {
        *RUNTIME_BINDABLE_TYPED_INPUT_KINDS,
        "audit",
        "report",
        "test",
    }
)


def is_failed_step_status(value: Any) -> bool:
    """Return whether a generated summary explicitly reports failure.

    Generated code sometimes emits a more specific status such as
    ``failed_provenance_audit``.  Treat the host-owned failure prefixes as
    fail-closed instead of relying only on a finite spelling allowlist.
    """

    status = _normalise(value)
    return status in _FAILED_STATUSES or status.startswith(("fail_", "failed_"))


_EFFECT_PRODUCT_BASES = frozenset(
    {
        "adjusted_effect",
        "adjusted_effect_estimate",
        "adjusted_effect_estimates",
        "adjusted_association",
        "adjusted_associations",
        "adjusted_association_estimate",
        "adjusted_association_estimates",
        "adjusted_hr",
        "adjusted_odds_ratio",
        "adjusted_odds_ratios",
        "adjusted_or",
        "adjusted_rd",
        "adjusted_rr",
        "association_estimate",
        "association_estimates",
        "causal_effect",
        "coefficient",
        "coefficients",
        "effect_estimate",
        "effect_estimates",
        "effect_forest",
        "hazard_ratio",
        "interaction_pvalue",
        "odds_ratio",
        "or_estimate",
        "or_estimates",
        "or_forest",
        "overall_effect",
        "primary_association",
        "primary_association_estimate",
        "primary_effect",
        "primary_estimate",
        "primary_hr",
        "primary_or",
        "primary_rd",
        "primary_rr",
        "relative_risk",
        "rd_estimate",
        "rd_estimates",
        "rd_forest",
        "rr_estimate",
        "rr_estimates",
        "rr_forest",
        "risk_difference",
        "risk_ratio",
        "hr_estimate",
        "hr_estimates",
        "hr_forest",
        "subgroup_effect",
        "subgroup_effects",
        "treatment_effect",
        "adjusted_logistic_regression_primary",
    }
)

_PRODUCT_SLOT_SUFFIXES: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "absolute_risk": (("absolute", "risk"),),
    "distribution": (("distribution",),),
    "availability": (
        ("measurement", "availability"),
        ("availability",),
        ("availability", "panel"),
        ("measurement", "coverage"),
        ("source", "coverage"),
    ),
    "cohort_flow": (
        ("cohort", "flow"),
        ("eligibility", "flow"),
    ),
    "attrition_audit": (
        ("attrition",),
        ("attrition", "audit"),
    ),
    "primary_estimand": (
        ("adjusted", "effect"),
        ("adjusted", "association"),
        ("primary", "estimand"),
    ),
    "precision_audit": (
        ("precision", "audit"),
        ("interval", "width", "audit"),
    ),
    "robustness_plot": (
        ("robustness", "plot"),
        ("sensitivity", "plot"),
        ("robustness", "forest"),
        ("sensitivity", "forest"),
    ),
    "robustness_denominator_audit": (
        ("robustness", "denominator", "audit"),
        ("sensitivity", "denominator", "audit"),
        ("model", "denominator", "audit"),
    ),
}

# A planner product's subject must name the scientific entity being displayed,
# not another display archetype.  Without this structural guard a nested role
# such as ``kaplan_meier_curve_distribution`` can be stripped to the subject
# ``kaplan_meier_curve`` and then laundered into an ordinary distribution slot.
# These are generic display-role terminals, not benchmark or clinical tokens.
_NESTED_DISPLAY_ROLE_SUFFIXES: tuple[tuple[str, ...], ...] = (
    ("curve",),
    ("diagram",),
    ("forest",),
    ("heatmap",),
    ("plot",),
)

_OPERATIONAL_SUBJECT_SUFFIXES = frozenset(
    {
        "any",
        "count",
        "ever",
        "first",
        "flag",
        "indicator",
        "last",
        "max",
        "mean",
        "measured",
        "median",
        "min",
        "n",
        "observed",
        "raw",
        "sum",
        "value",
    }
)

# This subtree is a host-verified receipt, not a registry of scientific
# outputs.  The execution layer separately rejects these markers when they
# were not installed by an authorized sealed renderer, so excluding the digest
# map here cannot let generated code claim renderer authority.
_HOST_RECEIPT_SUBTREES = frozenset({"sealed_renderer_parent_digests"})


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _assignment_artifact_frame(
    path: Path,
    *,
    columns: Sequence[str],
):
    """Read only assignment-product identity and score columns."""

    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=list(columns))
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", usecols=list(columns))
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns))
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def typed_product_binding_contract(
    *,
    product_name: str,
    step_summary: Mapping[str, Any],
    artifact_path: Path,
    authoritative_cohort_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return producer-declared coordinates needed to consume a typed product.

    This binds metadata already chosen by the producer. It never selects a
    cohort, exposure, outcome, model, method, or estimand for the consumer.
    """

    normalized_product = _normalise(product_name)
    if normalized_product == "exposure_definition":
        authoritative_exposure = str(
            step_summary.get("authoritative_exposure") or ""
        ).strip()
        derived_exposure = str(step_summary.get("derived_exposure") or "").strip()
        derived_rule = str(step_summary.get("derived_exposure_rule") or "").strip()
        locked_cohort_n = step_summary.get("locked_cohort_n")
        if not (
            authoritative_exposure
            and authoritative_exposure.isidentifier()
            and derived_exposure
            and derived_exposure.isidentifier()
            and derived_rule
            and isinstance(locked_cohort_n, int)
            and not isinstance(locked_cohort_n, bool)
            and locked_cohort_n >= 0
            and artifact_path.suffix.lower() == ".json"
            and authoritative_cohort_path is not None
        ):
            return None
        try:
            artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(artifact_payload, Mapping):
            return None
        if (
            str(artifact_payload.get("authoritative_exposure") or "").strip()
            != authoritative_exposure
            or str(artifact_payload.get("derived_exposure") or "").strip()
            != derived_exposure
            or str(artifact_payload.get("rule") or "").strip() != derived_rule
            or artifact_payload.get("locked_cohort_n") != locked_cohort_n
        ):
            return None
        cohort_columns = set(_tabular_artifact_columns(authoritative_cohort_path))
        if authoritative_exposure not in cohort_columns:
            return None
        contract = {
            "artifact_type": "exposure_definition",
            "executable_column": authoritative_exposure,
            "exposure_column": authoritative_exposure,
            "authoritative_primary_exposure": authoritative_exposure,
            "derived_exposure": derived_exposure,
            "rule": derived_rule,
            "locked_cohort_n": locked_cohort_n,
        }
        for key in (
            "window",
            "aggregation_rule",
            "usable_variation",
            "weighted_association_feasibility",
            "failure_reason",
        ):
            if artifact_payload.get(key) is not None:
                contract[key] = artifact_payload[key]
        return contract
    exact_contract = step_summary.get(product_name)
    if isinstance(exact_contract, Mapping):
        contract = dict(exact_contract)
        if normalized_product == "primary_exposure_definition":
            declared_columns = {
                str(contract.get(key) or "").strip()
                for key in ("column", "executable_column", "exposure_column")
                if str(contract.get(key) or "").strip()
            }
            if len(declared_columns) != 1:
                return None
            executable_column = next(iter(declared_columns))
            artifact_columns = set(_tabular_artifact_columns(artifact_path))
            if artifact_columns and executable_column not in artifact_columns:
                return None
            contract["executable_column"] = executable_column
            contract["exposure_column"] = executable_column
            contract["authoritative_primary_exposure"] = executable_column
            declared_windows = {
                str(contract.get(key) or "").strip()
                for key in ("window", "time_window")
                if str(contract.get(key) or "").strip()
            }
            if len(declared_windows) > 1:
                return None
            if len(declared_windows) == 1:
                time_window = next(iter(declared_windows))
                contract["window"] = time_window
                contract["time_window"] = time_window
        # Assignment-model and confounder-set contracts are host-owned.  Their
        # schema must be derived from the registered artifact and canonical
        # producer roster below; an arbitrary same-name mapping in the summary
        # is not authority and must never bypass those checks.
        if normalized_product not in {
            "assignment_model",
            "prespecified_confounder_set",
        }:
            return contract
    if normalized_product == "prespecified_confounder_set":
        if artifact_path.suffix.lower() != ".json":
            return None
        try:
            artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(artifact_payload, Mapping):
            return None
        selected = artifact_payload.get("selected_covariates")
        if not isinstance(selected, list):
            return None
        covariates = [str(value).strip() for value in selected]
        if (
            not covariates
            or any(not value for value in covariates)
            or len(set(covariates)) != len(covariates)
        ):
            return None
        contract: dict[str, Any] = {
            "covariates": covariates,
            "source_field": "selected_covariates",
        }
        for key in ("ordinal_encoding", "missingness_plan"):
            value = artifact_payload.get(key)
            if isinstance(value, Mapping):
                contract[key] = dict(value)
        return contract
    if normalized_product != "assignment_model":
        return None

    # Assignment scores are row-level scientific inputs.  Internal
    # self-consistency is not enough: an arbitrary, unique set of identifiers
    # with plausible probabilities must not become a valid typed product.  The
    # host therefore requires the producer artifact to be a row-aligned view of
    # the immutable analysis cohort.  Individual models may leave scores null
    # outside their Planner-owned analysis set, but they may not substitute a
    # different population.
    if authoritative_cohort_path is None:
        return None

    raw_models = step_summary.get("assignment_models")
    if not isinstance(raw_models, list):
        return None
    columns = _tabular_artifact_columns(artifact_path)
    normalized_columns = {column: _normalise(column) for column in columns}
    bound_models: list[dict[str, Any]] = []
    fitted_identities: set[tuple[str, str]] = set()
    for raw_model in raw_models:
        if not isinstance(raw_model, Mapping):
            return None
        model = dict(raw_model)
        fit_status = _normalise(raw_model.get("fit_status") or raw_model.get("status"))
        if fit_status != "fitted":
            continue
        model_id = _normalise(raw_model.get("model_id"))
        analysis_set = _normalise(raw_model.get("analysis_set"))
        if not model_id or not analysis_set:
            return None
        identity = (model_id, analysis_set)
        if identity in fitted_identities:
            return None
        fitted_identities.add(identity)
        identifiers = {
            _normalise(raw_model.get(key))
            for key in ("analysis_set", "model_id")
            if _normalise(raw_model.get(key))
        }
        identifiers.update(
            identifier.removeprefix("assignment_")
            for identifier in list(identifiers)
            if identifier.startswith("assignment_")
        )
        candidates: list[str] = []
        for column, normalized_column in normalized_columns.items():
            for prefix in ("propensity_score_", "propensity_", "ps_"):
                if (
                    normalized_column.startswith(prefix)
                    and normalized_column[len(prefix) :] in identifiers
                ):
                    candidates.append(column)
                    break
        unique_candidates = sorted(set(candidates))
        declared_score_columns = {
            str(raw_model.get(key) or "").strip()
            for key in ("propensity_score_column", "score_column")
            if str(raw_model.get(key) or "").strip()
        }
        if len(declared_score_columns) > 1:
            return None
        if declared_score_columns:
            declared_score = next(iter(declared_score_columns))
            if declared_score not in columns:
                return None
            if unique_candidates and declared_score not in unique_candidates:
                return None
            unique_candidates = [declared_score]
        if len(unique_candidates) != 1:
            return None
        model["model_id"] = str(raw_model.get("model_id"))
        model["analysis_set"] = str(raw_model.get("analysis_set"))
        model["fit_status"] = "fitted"
        model["propensity_score_column"] = unique_candidates[0]
        bound_models.append(model)
    if not bound_models:
        return None
    declared_identity_columns = {
        str(value).strip()
        for value in (
            step_summary.get("assignment_model_row_key"),
            step_summary.get("row_identity_column"),
            *[
                model.get("row_identity_column")
                for model in bound_models
                if isinstance(model, Mapping)
            ],
        )
        if str(value or "").strip()
    }
    if len(declared_identity_columns) > 1:
        return None
    if declared_identity_columns:
        identity_column = next(iter(declared_identity_columns))
        if identity_column not in columns:
            return None
    else:
        identity_candidates = [
            column
            for column in columns
            if _normalise(column)
            in {
                "encounter_id",
                "patient_id",
                "row_id",
                "row_index",
                "stay_id",
                "subject_id",
            }
        ]
        if len(identity_candidates) != 1:
            return None
        identity_column = identity_candidates[0]
    score_columns = [str(model["propensity_score_column"]) for model in bound_models]
    try:
        frame = _assignment_artifact_frame(
            artifact_path,
            columns=[identity_column, *score_columns],
        )
    except Exception:
        return None
    if frame is None or frame.empty:
        return None
    identity = frame[identity_column]
    if identity.isna().any() or identity.duplicated().any():
        return None
    import numpy as np
    import pandas as pd

    try:
        cohort_path = Path(authoritative_cohort_path)
        suffix = cohort_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            if identity_column == "row_index":
                import pyarrow.parquet as pq

                cohort_n = int(pq.ParquetFile(cohort_path).metadata.num_rows)
                expected_identity = pd.Series(range(cohort_n), name=identity_column)
            else:
                cohort_frame = pd.read_parquet(cohort_path, columns=[identity_column])
                expected_identity = cohort_frame[identity_column]
        elif suffix == ".csv":
            if identity_column == "row_index":
                cohort_n = len(pd.read_csv(cohort_path, usecols=[0]))
                expected_identity = pd.Series(range(cohort_n), name=identity_column)
            else:
                cohort_frame = pd.read_csv(cohort_path, usecols=[identity_column])
                expected_identity = cohort_frame[identity_column]
        elif suffix == ".tsv":
            if identity_column == "row_index":
                cohort_n = len(pd.read_csv(cohort_path, sep="\t", usecols=[0]))
                expected_identity = pd.Series(range(cohort_n), name=identity_column)
            else:
                cohort_frame = pd.read_csv(
                    cohort_path, sep="\t", usecols=[identity_column]
                )
                expected_identity = cohort_frame[identity_column]
        else:
            return None
    except Exception:
        return None
    expected_identity = expected_identity.reset_index(drop=True)
    observed_identity = identity.reset_index(drop=True)
    if (
        len(observed_identity) != len(expected_identity)
        or expected_identity.isna().any()
        or expected_identity.duplicated().any()
        or not observed_identity.astype("string").equals(
            expected_identity.astype("string")
        )
    ):
        return None

    def _identity_digest(values: "pd.Series") -> str:
        digest = hashlib.sha256()
        for value in values.astype("string"):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    row_identity_sha256 = _identity_digest(observed_identity)

    for model in bound_models:
        score_column = str(model["propensity_score_column"])
        score = pd.to_numeric(frame[score_column], errors="coerce")
        nonmissing = score.notna()
        values = score[nonmissing].to_numpy(dtype=float)
        if (
            not values.size
            or not np.isfinite(values).all()
            or bool(((values < 0.0) | (values > 1.0)).any())
        ):
            return None
        declared_n = model.get("n")
        if (
            isinstance(declared_n, bool)
            or declared_n is not None
            and (
                not isinstance(declared_n, int)
                or declared_n < 1
                or declared_n != int(nonmissing.sum())
            )
        ):
            return None
        model["row_identity_column"] = identity_column
        model["analysis_set_n"] = int(nonmissing.sum())
        model["analysis_set_identity_sha256"] = _identity_digest(
            observed_identity[nonmissing].reset_index(drop=True)
        )
    return {
        "row_identity_column": identity_column,
        "row_count": len(observed_identity),
        "row_identity_sha256": row_identity_sha256,
        "authoritative_cohort_sha256": _sha256_file(Path(authoritative_cohort_path)),
        "models": bound_models,
    }


def _canonical_kind(value: object) -> str:
    kind = _normalise(value)
    if kind in _FIGURE_KINDS:
        return "figure"
    # A cohort is a scientifically scoped tabular dataset, not a distinct
    # physical evidence class.  Canonicalising the Planner-facing alias here
    # keeps plan DAG construction, declared-output validation, and runtime
    # evidence binding on the same identity.  The alias never chooses or
    # modifies cohort membership; it only closes the typed product boundary.
    if kind == "cohort":
        return "dataset"
    if kind in {"metric", "statistics"}:
        return "statistic"
    return kind


def typed_product(value: object) -> tuple[str, str] | None:
    """Return the shared canonical identity for a ``kind:product`` token."""

    kind, separator, product = str(value or "").strip().partition(":")
    if not separator:
        return None
    canonical_kind = _canonical_kind(kind)
    product_name = _normalise(Path(product).name)
    for suffix in sorted(_KNOWN_FILE_SUFFIXES, key=len, reverse=True):
        normalised_suffix = _normalise(suffix)
        if product_name.endswith(f"_{normalised_suffix}"):
            product_name = product_name[: -(len(normalised_suffix) + 1)]
            break
    if not canonical_kind or not product_name:
        return None
    return canonical_kind, product_name


_PRIMARY_ANALYSIS_COHORT_METHODS = frozenset(
    {
        "cohort_construction",
        "cohort_definition",
        "cohort_definition_and_attrition",
        "cohort_definition_with_attrition",
        "eligibility_definition",
        "primary_cohort_definition",
    }
)
_PRIMARY_ANALYSIS_COHORT_DATA_KINDS = frozenset({"artifact", "dataset", "table"})
_PRIMARY_ANALYSIS_COHORT_FLOW_PRODUCTS = frozenset(
    {
        "attrition",
        "attrition_by_rule",
        "cohort_attrition",
        "cohort_denominator",
        "cohort_denominators",
        "cohort_flow",
        "eligibility_flow",
    }
)


def _is_primary_analysis_cohort_method(value: object) -> bool:
    """Recognise a case-neutral primary cohort construction method family.

    Planner method labels are agent-authored and may use an equivalent phrase
    instead of one host spelling.  Match the method's semantic tokens while
    excluding sensitivity/robustness riders; typed outputs and the locked plan
    still decide whether the step may own the primary cohort.
    """

    method = _normalise(value)
    if method in _PRIMARY_ANALYSIS_COHORT_METHODS:
        return True
    tokens = frozenset(part for part in method.split("_") if part)
    if tokens & {
        "external",
        "matched",
        "overlap",
        "robustness",
        "secondary",
        "sensitivity",
        "subgroup",
        "validation",
    }:
        return False
    return bool(tokens & {"cohort", "eligibility"}) and bool(
        tokens & {"construction", "definition", "filter"}
    )


def _primary_analysis_cohort_product(raw: object) -> tuple[str, str] | None:
    """Return one exact primary-cohort product identity.

    ``analysis_cohort`` is the legacy closed product across its supported
    physical aliases. Any non-empty product in the explicit Planner-facing
    ``cohort:`` namespace is also a closed population identity candidate; the
    plan-level cohort name and surrounding method/output contract decide whether
    it owns the primary cohort. A model-specific ``table:analysis_set`` or
    ``artifact:analysis_set`` must not become a primary population merely
    because a cohort/attrition step exists nearby.
    """

    raw_kind, separator, raw_product = str(raw or "").strip().partition(":")
    raw_product = raw_product.strip()
    if (
        not separator
        or not raw_product
        or Path(raw_product).name != raw_product
        or "/" in raw_product
        or "\\" in raw_product
    ):
        return None
    parsed = typed_product(raw)
    if parsed is None:
        return None
    kind, name = parsed
    if kind not in _PRIMARY_ANALYSIS_COHORT_DATA_KINDS:
        return None
    if name == "analysis_cohort":
        return parsed
    if _normalise(raw_kind) == "cohort":
        return parsed
    return None


def _declares_reserved_primary_cohort_product(step: AnalysisStep) -> bool:
    """Return whether a step claims one legacy globally reserved identity."""

    for raw in step.expected_outputs or []:
        raw_kind, _, _ = str(raw or "").strip().partition(":")
        parsed = _primary_analysis_cohort_product(raw)
        if parsed is None:
            continue
        _, name = parsed
        if name == "analysis_cohort" or (
            _normalise(raw_kind) == "cohort" and name == "analysis_set"
        ):
            return True
    return False


def _primary_analysis_cohort_product_matches_plan(
    raw: object, *, plan: Any
) -> tuple[str, str] | None:
    """Bind a named ``cohort:`` product to the Planner-locked cohort only."""

    product = _primary_analysis_cohort_product(raw)
    if product is None:
        return None
    raw_kind, _, _ = str(raw or "").strip().partition(":")
    _, name = product
    if name == "analysis_cohort" or (
        _normalise(raw_kind) == "cohort" and name == "analysis_set"
    ):
        return product
    cohort_name = _normalise(getattr(getattr(plan, "cohort", None), "name", None))
    if _normalise(raw_kind) == "cohort" and cohort_name and name == cohort_name:
        return product
    return None


def _declares_explicit_cohort_namespace(raw: object) -> bool:
    """Recognize an explicit cohort claim even when its product is malformed."""

    raw_kind, separator, raw_product = str(raw or "").strip().partition(":")
    return bool(separator and raw_product.strip() and _normalise(raw_kind) == "cohort")


def _primary_analysis_cohort_attrition_candidate(step: AnalysisStep) -> bool:
    if not _is_primary_analysis_cohort_method(step.method):
        return False
    outputs = list(step.expected_outputs or [])
    parsed = [typed_product(raw) for raw in outputs]
    return any(
        _primary_analysis_cohort_product(raw) is not None
        or _declares_explicit_cohort_namespace(raw)
        for raw in outputs
    ) and any(
        product is not None
        and product[0] == "table"
        and product[1] in _PRIMARY_ANALYSIS_COHORT_FLOW_PRODUCTS
        for product in parsed
    )


def _primary_analysis_cohort_attrition_step(step: AnalysisStep) -> bool:
    """Return whether a step claims primary cohort construction + attrition.

    Unlike ``_primary_analysis_cohort_attrition_candidate``, this predicate
    deliberately does not require a valid closed cohort output.  It lets plan
    preflight reject a definition/log artifact that was incorrectly declared
    in place of the materialised cohort before Coder or cohort execution runs.
    """

    if not _is_primary_analysis_cohort_method(step.method):
        return False
    return any(
        product is not None
        and product[0] == "table"
        and product[1] in _PRIMARY_ANALYSIS_COHORT_FLOW_PRODUCTS
        for product in (typed_product(raw) for raw in step.expected_outputs or [])
    )


def primary_analysis_cohort_producer_uses_universe(
    *, step: AnalysisStep, plan: Any
) -> bool:
    """Bind only one closed primary-cohort producer to the raw universe.

    The host materialises the Planner-locked cohort before ordinary analysis
    steps run.  A mixed cohort+attrition producer must nevertheless see the raw
    universe so it can report truthful exclusions.  This predicate changes only
    the input role: the Agent still owns every cohort criterion and output.
    """

    if not _primary_analysis_cohort_attrition_candidate(step):
        return False

    raw_outputs = list(step.expected_outputs or [])
    parsed_outputs = [typed_product(raw) for raw in raw_outputs]
    if not parsed_outputs or any(product is None for product in parsed_outputs):
        return False
    analysis_cohort_products = 0
    has_attrition = False
    for raw, parsed in zip(raw_outputs, parsed_outputs, strict=True):
        kind, name = parsed  # type: ignore[misc]
        if _primary_analysis_cohort_product_matches_plan(raw, plan=plan) is not None:
            analysis_cohort_products += 1
            continue
        if kind == "table" and name in _PRIMARY_ANALYSIS_COHORT_FLOW_PRODUCTS:
            has_attrition = True
            continue
        return False
    if analysis_cohort_products != 1 or not has_attrition:
        return False

    producers = [
        candidate
        for candidate in getattr(plan, "steps", ())
        if _declares_reserved_primary_cohort_product(candidate)
        or any(
            _primary_analysis_cohort_product_matches_plan(raw, plan=plan) is not None
            for raw in (candidate.expected_outputs or [])
        )
    ]
    return len(producers) == 1 and producers[0].step_id == step.step_id


def _primary_analysis_cohort_product_owner_finding(
    *,
    step: AnalysisStep,
    plan: Any,
    validator: str,
) -> ValidationFinding | None:
    """Return the shared structural owner finding for one cohort step.

    This check uses only the Planner-declared method and typed products.  It is
    therefore safe to run before code generation as well as inside the later
    execution-time integrity gate.  Keeping the finding here prevents the two
    stages from drifting into different definitions of a closed owner.
    """

    if not _primary_analysis_cohort_attrition_step(step):
        return None
    if primary_analysis_cohort_producer_uses_universe(step=step, plan=plan):
        return None
    declared_closed_candidates = [
        str(raw or "").strip()
        for raw in step.expected_outputs or []
        if _primary_analysis_cohort_product(raw) is not None
        or _declares_explicit_cohort_namespace(raw)
    ]
    issue = (
        "primary_cohort_product_owner_ambiguous"
        if declared_closed_candidates
        else "primary_cohort_product_missing"
    )
    detail: dict[str, Any] = {
        "issue": issue,
        "step_id": step.step_id,
    }
    if issue == "primary_cohort_product_missing":
        detail["declared_closed_candidates"] = declared_closed_candidates
    return ValidationFinding(
        validator=validator,
        severity="error",
        message=(
            "A primary cohort construction + attrition step must declare exactly "
            "one materialised closed cohort product and be the plan's unique "
            "primary-cohort owner. A definition or protocol artifact is not a "
            "cohort dataset."
        ),
        detail=detail,
    )


def primary_analysis_cohort_plan_findings(*, plan: Any) -> list[ValidationFinding]:
    """Validate primary-cohort typed-product ownership before Coder execution.

    The host does not select a cohort or edit eligibility here.  It only checks
    whether a Planner-declared mixed cohort/attrition step is structurally
    closed and uniquely owns the locked primary cohort.  The probe-aware
    replanner may then repair declarations without changing scientific choices.
    """

    findings: list[ValidationFinding] = []
    for step in getattr(plan, "steps", ()):
        finding = _primary_analysis_cohort_product_owner_finding(
            step=step,
            plan=plan,
            validator="plan_primary_analysis_cohort_integrity",
        )
        if finding is not None:
            findings.append(finding)
    return findings


def _contained_regular_output_file(out_dir: Path, value: object) -> Path:
    """Return one link-free regular file lexically contained in ``out_dir``.

    This helper runs inside the macOS execution sandbox.  Calling
    :meth:`Path.resolve` there asks the kernel to inspect ancestors such as
    ``/private`` that are intentionally outside the sandbox allowlist.  The
    host already supplies the output root, so containment is enforced without
    ancestor traversal. Output-local absolute aliases are first reduced
    lexically to a relative path; outside paths and parent components remain
    invalid. Every output-local component is then checked for symlinks, and
    hard-linked files are rejected.
    """

    root = Path(out_dir)
    raw = str(value or "").strip()
    supplied = Path(raw)
    if supplied.is_absolute():
        try:
            relative = supplied.relative_to(root.absolute())
        except ValueError as exc:
            raise ValueError("output path must be contained") from exc
    else:
        relative = supplied
    if not raw or ".." in relative.parts:
        raise ValueError("output path must be contained")
    try:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("output root is not a regular directory")
        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("output path traverses a symlink")
        stat_result = candidate.stat()
    except OSError as exc:
        raise ValueError("output file is unavailable") from exc
    if not candidate.is_file() or stat_result.st_nlink != 1:
        raise ValueError("output path is not a single-link regular file")
    return candidate


def read_digest_bound_artifact_snapshot(
    *, parent_out: Path, artifact_digests: Mapping[str, str]
) -> dict[str, bytes]:
    """Read one link-free parent snapshot and verify the exact bytes once.

    Renderers must parse the returned bytes (for example through
    :class:`io.BytesIO`) rather than reopening the source paths.  This makes a
    file replacement after verification irrelevant to the authorized render.
    """

    root = Path(parent_out)
    if root.is_symlink() or not root.is_dir() or not artifact_digests:
        raise ValueError("parent artifact root is unavailable")
    snapshot: dict[str, bytes] = {}
    for raw_name, raw_digest in artifact_digests.items():
        name = str(raw_name or "").strip()
        digest = str(raw_digest or "").strip().lower()
        path = Path(name)
        candidate = root / name
        if (
            not name
            or path.name != name
            or path.is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or candidate.is_symlink()
        ):
            raise ValueError("parent artifact seal contains an unsafe entry")
        try:
            stat_result = candidate.stat()
            payload = candidate.read_bytes()
        except OSError as exc:
            raise ValueError("parent artifact is unavailable") from exc
        if (
            not candidate.is_file()
            or stat_result.st_nlink != 1
            or hashlib.sha256(payload).hexdigest() != digest
        ):
            raise ValueError("parent artifact does not match its authorized digest")
        snapshot[name] = payload
    return snapshot


def _planner_product_slot_and_subject(
    product_name: str, allowed_slots: set[str]
) -> tuple[str, tuple[str, ...], str]:
    tokens = tuple(part for part in _normalise(product_name).split("_") if part)
    matches: set[tuple[str, tuple[str, ...], tuple[str, ...], str]] = set()
    for slot in allowed_slots:
        for role_tokens in _PRODUCT_SLOT_SUFFIXES.get(slot, ()):
            if (
                len(tokens) >= len(role_tokens)
                and tokens[-len(role_tokens) :] == role_tokens
            ):
                matches.add(
                    (slot, role_tokens, tokens[: -len(role_tokens)], "subject_role")
                )
            if (
                len(tokens) > len(role_tokens) + 1
                and tokens[: len(role_tokens)] == role_tokens
                and tokens[len(role_tokens)] == "by"
            ):
                matches.add(
                    (
                        slot,
                        role_tokens,
                        tokens[len(role_tokens) + 1 :],
                        "role_by_subject",
                    )
                )
    if matches:
        longest_by_slot_and_syntax = {
            (slot, syntax): max(
                len(role_tokens)
                for candidate_slot, role_tokens, _subject, candidate_syntax in matches
                if candidate_slot == slot and candidate_syntax == syntax
            )
            for slot, _role_tokens, _subject, syntax in matches
        }
        matches = {
            match
            for match in matches
            if len(match[1]) == longest_by_slot_and_syntax[(match[0], match[3])]
        }
    if len(matches) != 1:
        raise ValueError(
            "Planner figure role does not map uniquely to an authorized product slot"
        )
    slot, _role_tokens, subject, syntax = matches.pop()
    if not subject and syntax == "role_by_subject":
        raise ValueError("Planner figure role has an empty scientific subject")
    if any(
        tuple(subject[index : index + len(role_suffix)]) == role_suffix
        for role_suffix in _NESTED_DISPLAY_ROLE_SUFFIXES
        for index in range(len(subject) - len(role_suffix) + 1)
    ):
        raise ValueError(
            "Planner figure role nests an incompatible display archetype in "
            "its scientific subject"
        )
    return slot, subject, syntax


def _subject_tokens_compatible(
    declared_subject: tuple[str, ...], authoritative_subject: tuple[str, ...]
) -> bool:
    """Match a display label to one host-owned operational exposure.

    Operational columns may add a standard aggregation suffix (for example
    ``marker_max``). Human-readable aliases must be supplied explicitly by the
    host context; lexical prefix guesses are intentionally forbidden. This
    comparison is only one half of authorization: the exact declared subject
    must also occur in a typed direct-parent anchor.
    """

    reduced = list(authoritative_subject)
    while len(reduced) > 1 and reduced[-1] in _OPERATIONAL_SUBJECT_SUFFIXES:
        reduced.pop()
    return bool(declared_subject) and declared_subject in {
        authoritative_subject,
        tuple(reduced),
    }


def _contains_token_sequence(
    container: tuple[str, ...], candidate: tuple[str, ...]
) -> bool:
    return bool(candidate) and any(
        container[index : index + len(candidate)] == candidate
        for index in range(len(container) - len(candidate) + 1)
    )


def authorize_declared_figure_product_slots(
    *,
    declared_products: Sequence[str],
    renderer_repair_id: str,
    planner_parent_anchors: Sequence[str],
    authoritative_display_subjects: Sequence[str] = (),
) -> dict[str, str]:
    """Build the host-owned exact product-to-slot authorization.

    Product suffixes describe a display role, but a suffix alone is not
    authority: ``kaplan_meier_curve_distribution`` must not become an ordinary
    descriptive distribution merely because its last token matches.  A
    non-empty subject prefix is therefore accepted only when it is anchored to
    the beginning of a typed product or structured input from the host-recorded
    parent planning request.  Physical filenames and coder-written summaries
    are never semantic authority.  The exact Planner method/family selects the
    sealed renderer separately.
    """

    if not is_sealed_renderer_repair(renderer_repair_id):
        raise ValueError("renderer repair id is not an exact sealed renderer")
    allowed_slots = set(repair_metadata_for(renderer_repair_id).figure_product_slots)
    parent_token_sequences: set[tuple[str, ...]] = set()
    for value in planner_parent_anchors:
        parsed_parent = typed_product(value)
        parent_name = (
            parsed_parent[1]
            if parsed_parent is not None
            else (
                _file_stem(value)
                if Path(str(value or "")).name != "step_summary.json"
                else ""
            )
        )
        tokens = tuple(part for part in parent_name.split("_") if part)
        if tokens:
            parent_token_sequences.add(tokens)
    canonical: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    slot_bindings: dict[str, str] = {}
    claimed_slots: set[str] = set()
    for raw in declared_products:
        parsed = typed_product(raw)
        if parsed is None or parsed[0] != "figure" or parsed in seen:
            raise ValueError("Planner products are not unique typed figure roles")
        seen.add(parsed)
        canonical.append(parsed)
        slot, subject, syntax = _planner_product_slot_and_subject(
            parsed[1], allowed_slots
        )
        parent_subject_anchored = any(
            (
                (
                    len(parent_tokens) >= len(subject)
                    and parent_tokens[: len(subject)] == subject
                )
                if syntax == "subject_role"
                else _contains_token_sequence(parent_tokens, subject)
            )
            for parent_tokens in parent_token_sequences
        )
        if subject and not parent_subject_anchored:
            raise ValueError(
                "Planner figure role subject is not anchored to a verified "
                "Planner direct-parent product"
            )
        if syntax == "role_by_subject":
            authoritative_tokens = {
                tuple(part for part in _normalise(value).split("_") if part)
                for value in authoritative_display_subjects
                if _normalise(value)
            }
            if not authoritative_tokens or not any(
                _subject_tokens_compatible(subject, expected)
                for expected in authoritative_tokens
            ):
                raise ValueError(
                    "Planner role-by-subject figure does not match the host-owned "
                    "display subject"
                )
        if slot in claimed_slots:
            raise ValueError(
                "Planner figure roles do not map to distinct contract product slots"
            )
        slot_bindings[f"{parsed[0]}:{parsed[1]}"] = slot
        claimed_slots.add(slot)
    if not canonical:
        raise ValueError("Planner declared no typed figure products")
    return slot_bindings


def bind_declared_figure_products(
    *,
    out_dir: Path,
    declared_products: Sequence[str],
    authorized_product_slots: Mapping[str, str],
    renderer_repair_id: str,
    renderer_implementation_sha256: str,
    renderer_parent_digests: Mapping[str, str],
) -> bool:
    """Bind Planner-owned figure roles to one verified multi-panel bundle.

    Closed renderers own only rendering.  The Planner still owns the logical
    products, so the host embeds its exact ``expected_outputs`` in the trusted
    adapter and this helper registers those roles without inventing a new
    figure, statistic, cohort, or method.  Multiple planned roles may point to
    one canonical bundle only when its contract contains at least one panel per
    role; otherwise the binding fails closed.
    """

    if not is_sealed_renderer_repair(renderer_repair_id):
        raise ValueError("renderer repair id is not an exact sealed renderer")
    metadata = repair_metadata_for(renderer_repair_id)
    allowed_slots = set(metadata.figure_product_slots)
    if not allowed_slots:
        raise ValueError("sealed renderer declares no authorized product slots")
    if not re.fullmatch(r"[0-9a-f]{64}", renderer_implementation_sha256):
        raise ValueError("renderer implementation digest is invalid")
    canonical_parent_digests = {
        str(name): str(digest).strip().lower()
        for name, digest in renderer_parent_digests.items()
    }
    if "step_summary.json" not in canonical_parent_digests or any(
        Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for name, digest in canonical_parent_digests.items()
    ):
        raise ValueError("renderer parent digest seal is invalid")

    out_dir = Path(out_dir)
    try:
        summary_path = _contained_regular_output_file(out_dir, "step_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("rendered step_summary.json is unreadable") from exc
    if not isinstance(summary, dict) or summary.get("rendering_only") is not True:
        raise ValueError("rendered summary does not declare rendering_only=true")

    canonical: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in declared_products:
        parsed = typed_product(raw)
        if parsed is None or parsed[0] != "figure" or parsed in seen:
            raise ValueError("Planner products are not unique typed figure roles")
        seen.add(parsed)
        canonical.append((parsed[0], parsed[1], str(raw).strip()))
    if not canonical:
        raise ValueError("Planner declared no typed figure products")
    canonical_names = {f"{kind}:{name}" for kind, name, _raw in canonical}
    supplied_slots = {
        str(product): _normalise(slot)
        for product, slot in authorized_product_slots.items()
    }
    if set(supplied_slots) != canonical_names:
        raise ValueError(
            "host product-slot authorization does not match Planner products"
        )
    if any(slot not in allowed_slots for slot in supplied_slots.values()) or len(
        supplied_slots.values()
    ) != len(set(supplied_slots.values())):
        raise ValueError("host product-slot authorization is invalid")

    figure_path_value = summary.get("figure_path")
    if not isinstance(figure_path_value, str) or not figure_path_value.strip():
        raise ValueError("rendered summary has no canonical figure_path")
    try:
        figure_path = _contained_regular_output_file(out_dir, figure_path_value)
    except ValueError as exc:
        raise ValueError("canonical figure is missing or outside STEP_OUT_DIR") from exc
    if figure_path.suffix.lower() not in _FIGURE_SUFFIXES:
        raise ValueError("canonical figure_path is not a supported figure file")

    contract_value = summary.get("figure_contract")
    if not isinstance(contract_value, str) or not contract_value.strip():
        raise ValueError("rendered summary has no figure_contract")
    try:
        contract_path = _contained_regular_output_file(out_dir, contract_value)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "figure contract is unreadable or outside STEP_OUT_DIR"
        ) from exc
    panels = contract.get("panels") if isinstance(contract, Mapping) else None
    panel_ids: set[str] = set()
    slot_to_panels: dict[str, set[str]] = defaultdict(set)
    for panel in panels or []:
        if not isinstance(panel, Mapping):
            continue
        panel_id = str(panel.get("panel_id") or "").strip()
        if not panel_id or panel_id in panel_ids:
            raise ValueError("figure contract panel ids are missing or duplicated")
        panel_ids.add(panel_id)
        panel_metadata = panel.get("metadata")
        raw_slots = (
            panel_metadata.get("planner_product_slots")
            if isinstance(panel_metadata, Mapping)
            else None
        )
        if raw_slots is None:
            continue
        if not isinstance(raw_slots, list) or not raw_slots:
            raise ValueError("panel planner_product_slots must be a non-empty list")
        normalized_slots = [_normalise(slot) for slot in raw_slots]
        if any(
            not slot or slot not in allowed_slots for slot in normalized_slots
        ) or len(normalized_slots) != len(set(normalized_slots)):
            raise ValueError("panel declares an unauthorized or duplicate product slot")
        for slot in normalized_slots:
            slot_to_panels[slot].add(panel_id)

    slot_bindings: dict[str, dict[str, object]] = {}
    for kind, name, _raw in canonical:
        canonical_name = f"{kind}:{name}"
        slot = supplied_slots[canonical_name]
        candidate_panels = slot_to_panels.get(slot, set())
        if not candidate_panels:
            raise ValueError(
                "authorized product slot is not anchored to a contract panel"
            )
        slot_bindings[canonical_name] = {
            "slot": slot,
            "panel_ids": sorted(candidate_panels),
        }

    output_files = summary.get("output_files")
    if not isinstance(output_files, dict):
        output_files = {}
    else:
        output_files = dict(output_files)
    for kind, name, _raw in canonical:
        output_files[f"{kind}:{name}"] = figure_path_value
    summary["output_files"] = output_files
    summary["planner_bound_figure_products"] = [
        f"{kind}:{name}" for kind, name, _raw in canonical
    ]
    summary["planner_product_slot_bindings"] = slot_bindings
    summary["planner_product_binding"] = "shared_multi_panel_bundle"
    summary["sealed_renderer_repair"] = renderer_repair_id
    summary["sealed_renderer_implementation_sha256"] = renderer_implementation_sha256
    summary["sealed_renderer_parent_digests"] = dict(
        sorted(canonical_parent_digests.items())
    )

    payload = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=out_dir,
        prefix=".step_summary.planner_binding.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, summary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


# Internal compatibility alias; lineage and product-scope validation must share
# the public parser above rather than growing independent token grammars.
_typed_product = typed_product


def _file_stem(value: object) -> str:
    name = Path(str(value or "").strip()).name
    suffix = Path(name).suffix.lower()
    if suffix in _KNOWN_FILE_SUFFIXES:
        name = name[: -len(suffix)]
    return _normalise(name)


def _file_kinds(value: object) -> frozenset[str]:
    suffix = Path(str(value or "").strip()).suffix.lower()
    if suffix in _FIGURE_SUFFIXES:
        return frozenset({"figure"})
    if suffix in {".csv", ".tsv"}:
        return frozenset({"table", "artifact", "dataset", "test"})
    if suffix in {".parquet", ".feather"}:
        return frozenset({"artifact", "dataset", "table"})
    if suffix in {".pkl", ".pickle", ".joblib"}:
        return frozenset({"model", "artifact"})
    if suffix in {".npy", ".npz"}:
        return frozenset({"artifact", "dataset", "model"})
    if suffix in {".md", ".txt", ".log", ".jsonl"}:
        return frozenset({"log", "artifact"})
    if suffix == ".json":
        return frozenset({"artifact", "manifest", "log", "model", "test"})
    return frozenset()


_AUDIT_PHYSICAL_FILE_KINDS = frozenset({"log", "table"})


def _descriptor_path_is_compatible(*, kind: str, path: str) -> bool:
    """Return whether an exact typed descriptor can use *path*.

    ``audit`` is a semantic product role rather than a file suffix.  It must
    therefore be authorised only by an explicit structured descriptor, while
    its physical payload may be a table or a machine-readable/log sidecar.
    Keeping this compatibility here avoids teaching suffix-only path inference
    that every CSV or JSON file is an audit product.
    """

    physical_kinds = _file_kinds(path)
    if kind == "audit":
        return bool(physical_kinds & _AUDIT_PHYSICAL_FILE_KINDS)
    return kind in physical_kinds


def _is_file_path(value: object) -> bool:
    return isinstance(value, str) and Path(value.strip()).suffix.lower() in (
        _KNOWN_FILE_SUFFIXES
    )


def _iter_paths(value: Any) -> Iterable[str]:
    if _is_file_path(value):
        yield str(value).strip()
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_paths(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_paths(child)


def _registered_output_paths(summary: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                if key in _OUTPUT_CONTAINER_KEYS:
                    paths.extend(_iter_paths(child))
                elif isinstance(child, (Mapping, list, tuple)):
                    visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(summary)
    return list(dict.fromkeys(paths))


def _registered_product_paths(
    summary: Mapping[str, Any],
    *,
    product_name: str,
    allowed_kinds: frozenset[str],
    out_dir: Path,
) -> list[str]:
    explicit: list[str] = []
    legacy: list[str] = []
    target_descriptor_invalid = False

    def matches(value: object) -> bool:
        parsed = typed_product(value)
        return bool(
            parsed is not None
            and parsed[0] in allowed_kinds
            and parsed[1] == product_name
        )

    def visit_container(value: Any) -> None:
        nonlocal target_descriptor_invalid

        if isinstance(value, Mapping):
            descriptor_keys = {_normalise(raw_key) for raw_key in value}
            if descriptor_keys.intersection(_PRODUCT_DESCRIPTOR_FIELDS):
                raw_descriptor_kind = value.get("kind")
                raw_product_type = value.get("product_type")
                descriptor_name = value.get("name")
                raw_kind_identity = typed_product(raw_descriptor_kind)
                product_type_identity = typed_product(raw_product_type)
                descriptor_kind_families = {
                    family
                    for family in (
                        (
                            raw_kind_identity[0]
                            if raw_kind_identity is not None
                            else _canonical_kind(raw_descriptor_kind)
                        ),
                        (
                            product_type_identity[0]
                            if product_type_identity is not None
                            else _canonical_kind(raw_product_type)
                        ),
                    )
                    if family
                }
                name_identity = typed_product(descriptor_name)
                semantic_identities = {
                    identity
                    for identity in (
                        raw_kind_identity,
                        product_type_identity,
                        name_identity,
                    )
                    if identity is not None
                }
                if name_identity is None and descriptor_name is not None:
                    semantic_identities.update(
                        identity
                        for family in descriptor_kind_families
                        for identity in (typed_product(f"{family}:{descriptor_name}"),)
                        if identity is not None
                    )
                identity_targets_product = any(
                    identity[0] in allowed_kinds and identity[1] == product_name
                    for identity in semantic_identities
                )
                direct_paths = [
                    path
                    for field in ("path", "relative_path", "filename")
                    for path in _iter_paths(value.get(field))
                ]
                targets_product = identity_targets_product or (
                    not semantic_identities
                    and (
                        (
                            _file_stem(descriptor_name) == product_name
                            or any(
                                _file_stem(path) == product_name
                                for path in direct_paths
                            )
                        )
                        and (
                            not descriptor_kind_families
                            or bool(descriptor_kind_families & allowed_kinds)
                        )
                    )
                )

                # Descriptor fields form a closed authority boundary.  Never
                # recurse into metadata/value or reuse these paths as legacy
                # filename evidence after descriptor validation fails.
                if isinstance(raw_descriptor_kind, str) and ":" in raw_descriptor_kind:
                    shorthand = _typed_kind_shorthand_receipt(value, out_dir=out_dir)
                    if shorthand is not None:
                        descriptor, candidate = shorthand
                        if (
                            descriptor[0] in allowed_kinds
                            and descriptor[1] == product_name
                        ):
                            try:
                                explicit.append(
                                    str(candidate.relative_to(Path(out_dir).absolute()))
                                )
                            except ValueError:
                                target_descriptor_invalid = True
                            return
                    if targets_product:
                        target_descriptor_invalid = True
                    return

                if identity_targets_product:
                    if direct_paths:
                        explicit.extend(direct_paths[:1])
                    else:
                        target_descriptor_invalid = True
                elif targets_product:
                    target_descriptor_invalid = True
                return

            # Exact typed-key mappings are the legacy structured form.  They
            # remain authoritative even when the physical filename is not the
            # logical product name.
            for raw_key, child in value.items():
                if matches(raw_key):
                    paths = list(_iter_paths(child))
                    if paths:
                        explicit.extend(paths)
                    else:
                        target_descriptor_invalid = True
                elif isinstance(child, (Mapping, list, tuple)):
                    visit_container(child)
                elif _is_file_path(child) and _file_stem(child) == product_name:
                    legacy.append(str(child).strip())
            return

        if isinstance(value, (list, tuple)):
            for child in value:
                visit_container(child)
            return

        if _is_file_path(value) and _file_stem(value) == product_name:
            legacy.append(str(value).strip())

    def visit_summary(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                if _normalise(raw_key) in _OUTPUT_CONTAINER_KEYS:
                    visit_container(child)
                elif isinstance(child, (Mapping, list, tuple)):
                    visit_summary(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit_summary(child)

    visit_summary(summary)
    if target_descriptor_invalid:
        return []
    if explicit:
        return list(dict.fromkeys(explicit))
    return list(dict.fromkeys(legacy))


def _integral_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


_PRIMARY_RAW_DENOMINATOR_FIELDS = (
    "n_universe",
    "universe_n",
    "n_input_universe",
)
_PRIMARY_FINAL_DENOMINATOR_FIELDS = (
    "n_final_analysis_cohort",
    "n_analysis_cohort",
    "n_final_cohort",
    "final_cohort_n",
)


def _coherent_integral_count_columns(
    table: Any,
    names: Sequence[str],
) -> tuple[str | None, dict[str, list[int | None]], str | None]:
    """Read synonymous structural-count columns without first-match masking."""

    present = [name for name in names if name in table.columns]
    values_by_column = {
        name: [_integral_count(value) for value in table[name].tolist()]
        for name in present
    }
    if any(value is None for values in values_by_column.values() for value in values):
        return (present[0] if present else None), values_by_column, "nonintegral"
    if len(present) > 1:
        reference = values_by_column[present[0]]
        if any(values_by_column[name] != reference for name in present[1:]):
            return present[0], values_by_column, "disagree"
    return (present[0] if present else None), values_by_column, None


def _coherent_integral_mapping_counts(
    mapping: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[int | None, dict[str, int | None], str | None]:
    """Read all present synonymous count fields from one declared mapping."""

    present = [name for name in names if name in mapping]
    values_by_field = {name: _integral_count(mapping[name]) for name in present}
    if any(value is None for value in values_by_field.values()):
        return None, values_by_field, "nonintegral"
    if len(present) > 1:
        reference = values_by_field[present[0]]
        if any(values_by_field[name] != reference for name in present[1:]):
            return reference, values_by_field, "disagree"
    return (values_by_field[present[0]] if present else None), values_by_field, None


def _assert_exact_frame_values_equal(left: Any, right: Any) -> None:
    """Compare host values exactly while ignoring storage-only dtype changes.

    Parquet round-trips may represent the same nullable flag as either
    ``boolean {False, True, NA}`` or ``float {0.0, 1.0, NaN}``.  Pandas'
    frame comparator treats those values as different even with dtype checks
    disabled.  Authority needs a narrower rule: column order and missingness
    must match exactly, and every nonmissing scalar must compare exactly after
    conversion from NumPy/Pandas scalar wrappers to Python scalars.  No numeric
    tolerance, string coercion, or scientific recoding is permitted.
    """

    if left.shape != right.shape:
        raise AssertionError(f"frame shapes differ: {left.shape} != {right.shape}")
    if list(left.columns) != list(right.columns):
        raise AssertionError("frame columns differ")

    def python_scalar(value: Any) -> Any:
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (TypeError, ValueError):
                pass
        return value

    for position, column in enumerate(left.columns):
        left_series = left.iloc[:, position].reset_index(drop=True)
        right_series = right.iloc[:, position].reset_index(drop=True)
        left_missing = left_series.isna().tolist()
        right_missing = right_series.isna().tolist()
        if left_missing != right_missing:
            raise AssertionError(f"column {column!r} has different missingness")
        for row, (left_value, right_value, missing) in enumerate(
            zip(left_series.tolist(), right_series.tolist(), left_missing, strict=True)
        ):
            if missing:
                continue
            left_scalar = python_scalar(left_value)
            right_scalar = python_scalar(right_value)
            try:
                equal = bool(left_scalar == right_scalar)
            except (TypeError, ValueError):
                equal = False
            if not equal:
                raise AssertionError(
                    f"column {column!r} differs at row {row}: "
                    f"{left_scalar!r} != {right_scalar!r}"
                )


def primary_analysis_cohort_integrity_findings(
    *,
    step: AnalysisStep,
    plan: Any,
    step_summary: Mapping[str, Any],
    out_dir: Path,
    universe_path: Path,
    authoritative_cohort_path: Path,
    context: Any = None,
) -> list[ValidationFinding]:
    """Verify one Agent-produced primary cohort against host-locked authority.

    This gate never chooses eligibility.  It replays the Planner-owned typed
    cohort definition on the raw universe, then checks the produced row identity
    and declared attrition accounting before the attempt may become current.
    """

    if not _primary_analysis_cohort_attrition_step(step):
        return []

    def finding(issue: str, message: str, **detail: Any) -> list[ValidationFinding]:
        return [
            ValidationFinding(
                validator="primary_analysis_cohort_integrity",
                severity="error",
                message=message,
                detail={"issue": issue, "step_id": step.step_id, **detail},
            )
        ]

    owner_finding = _primary_analysis_cohort_product_owner_finding(
        step=step,
        plan=plan,
        validator="primary_analysis_cohort_integrity",
    )
    if owner_finding is not None:
        return [owner_finding]

    try:
        import pandas as pd

        from ..cohort.schema import (
            CohortDefinition,
            _planner_declared_context_column_bindings,
            _resolve_predicate_column,
            build_cohort,
            coerce_cohort_definition,
        )

        universe = pd.read_parquet(universe_path).reset_index(drop=True)
        authoritative = pd.read_parquet(authoritative_cohort_path).reset_index(
            drop=True
        )
        definition = coerce_cohort_definition(getattr(plan, "cohort", None))
        replayed = universe.copy()
        expected_remaining_counts = [int(len(replayed))]
        expected_exclusion_counts = [0]
        expected_criterion_ids: list[str] = []
        accepted_criterion_ids: list[tuple[str, ...]] = []
        if definition is not None:
            column_bindings = _planner_declared_context_column_bindings(
                definition=definition,
                plan=plan,
                context=context,
                columns=replayed.columns,
            )
            ordered_predicates = [
                ("include", predicate) for predicate in definition.inclusion
            ] + [("exclude", predicate) for predicate in definition.exclusion]
            for order, (kind, predicate) in enumerate(ordered_predicates, start=1):
                before_n = int(len(replayed))
                resolved_column = _resolve_predicate_column(
                    replayed.columns,
                    predicate.concept_id,
                    predicate.aggregation,
                    column_bindings=column_bindings,
                )
                if resolved_column is None:
                    raise ValueError(
                        "locked cohort predicate has no host-resolved materialized "
                        f"column: {predicate.concept_id!r}"
                    )
                one = CohortDefinition(
                    name=f"criterion_{order}",
                    inclusion=(predicate,) if kind == "include" else (),
                    exclusion=(predicate,) if kind == "exclude" else (),
                )
                replayed = build_cohort(
                    one,
                    replayed,
                    column_bindings=column_bindings,
                ).reset_index(drop=True)
                after_n = int(len(replayed))
                concept = _normalise(predicate.concept_id)
                canonical_id = f"{kind}_{order:02d}_{concept}"
                resolved_id = f"{kind}_{order:02d}_{_normalise(resolved_column)}"
                expected_criterion_ids.append(canonical_id)
                accepted_criterion_ids.append(
                    tuple(dict.fromkeys((canonical_id, resolved_id)))
                )
                expected_remaining_counts.append(after_n)
                expected_exclusion_counts.append(before_n - after_n)
    except Exception as exc:
        return finding(
            "authority_replay_unavailable",
            "The primary analysis-cohort product could not be verified against "
            "the raw universe and Planner-locked cohort definition.",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )

    criterion_alias_to_canonical: dict[str, str] = {}
    for canonical_id, aliases in zip(
        expected_criterion_ids,
        accepted_criterion_ids,
        strict=True,
    ):
        for alias in aliases:
            existing = criterion_alias_to_canonical.setdefault(alias, canonical_id)
            if existing != canonical_id:
                return finding(
                    "attrition_rule_identity_ambiguous",
                    "Host-resolved cohort predicate columns do not provide a "
                    "unique attrition-rule identity.",
                    ambiguous_alias=alias,
                    canonical_criterion_ids=[existing, canonical_id],
                )

    def canonicalise_reported_rule_ids(
        reported: Sequence[str],
    ) -> list[str] | None:
        canonical: list[str] = []
        for rule_id in reported:
            resolved = criterion_alias_to_canonical.get(rule_id)
            if resolved is None:
                return None
            canonical.append(resolved)
        return canonical

    raw_n = int(len(universe))
    locked_n = int(len(authoritative))
    if len(replayed) != locked_n:
        return finding(
            "locked_cohort_replay_count_mismatch",
            "The Planner-locked cohort replay does not match the authoritative "
            "materialised cohort denominator.",
            raw_universe_n=raw_n,
            replayed_cohort_n=int(len(replayed)),
            authoritative_cohort_n=locked_n,
        )

    identity_column = next(
        (
            candidate
            for candidate in ("stay_id", "encounter_id", "row_id")
            if all(
                candidate in frame.columns
                and not frame[candidate].isna().any()
                and not frame[candidate].duplicated().any()
                for frame in (replayed, authoritative)
            )
        ),
        None,
    )
    if identity_column is None:
        return finding(
            "row_identity_unverifiable",
            "The primary analysis cohort has no shared unique row identity; "
            "cohort membership cannot be sealed safely.",
        )

    expected_identity = (
        replayed[identity_column].astype("string").reset_index(drop=True)
    )
    authoritative_identity = (
        authoritative[identity_column].astype("string").reset_index(drop=True)
    )
    if not expected_identity.equals(authoritative_identity):
        return finding(
            "authoritative_cohort_identity_mismatch",
            "The materialised analysis cohort does not match the row identities "
            "obtained by replaying the Planner-locked definition.",
            identity_column=identity_column,
            raw_universe_n=raw_n,
            authoritative_cohort_n=locked_n,
        )
    try:
        _assert_exact_frame_values_equal(
            replayed.loc[:, authoritative.columns].reset_index(drop=True),
            authoritative,
        )
    except (AssertionError, KeyError) as exc:
        return finding(
            "authoritative_cohort_value_mismatch",
            "The materialised analysis cohort changes host columns relative to "
            "the Planner-locked cohort replay.",
            error=str(exc)[:300],
        )

    declared_cohort_products = {
        product[1]
        for raw in (step.expected_outputs or [])
        if (product := _primary_analysis_cohort_product_matches_plan(raw, plan=plan))
        is not None
    }
    if len(declared_cohort_products) != 1:
        return finding(
            "analysis_cohort_product_ambiguous",
            "The primary analysis-cohort product must have exactly one "
            "registered typed identity.",
            declared_products=sorted(declared_cohort_products),
        )
    cohort_product = next(iter(declared_cohort_products))
    cohort_candidates = _registered_product_paths(
        step_summary,
        product_name=cohort_product,
        allowed_kinds=_PRIMARY_ANALYSIS_COHORT_DATA_KINDS,
        out_dir=out_dir,
    )
    if len(cohort_candidates) != 1:
        return finding(
            "analysis_cohort_product_ambiguous",
            "The declared primary analysis-cohort product must resolve to exactly one "
            "registered output file.",
            product=cohort_product,
            candidates=cohort_candidates,
        )
    try:
        produced_path = _contained_regular_output_file(
            Path(out_dir), cohort_candidates[0]
        )
        if produced_path.suffix.lower() == ".parquet":
            produced = pd.read_parquet(produced_path).reset_index(drop=True)
        elif produced_path.suffix.lower() == ".feather":
            produced = pd.read_feather(produced_path).reset_index(drop=True)
        elif produced_path.suffix.lower() in {".csv", ".tsv"}:
            produced = pd.read_csv(
                produced_path,
                sep="\t" if produced_path.suffix.lower() == ".tsv" else ",",
            ).reset_index(drop=True)
        else:
            raise ValueError("unsupported analysis-cohort file type")
    except Exception as exc:
        return finding(
            "analysis_cohort_product_unreadable",
            "The declared analysis_cohort product is unreadable or outside the "
            "step output directory.",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
    if (
        identity_column not in produced.columns
        or produced[identity_column].isna().any()
        or produced[identity_column].duplicated().any()
        or not produced[identity_column]
        .astype("string")
        .reset_index(drop=True)
        .equals(authoritative_identity)
    ):
        return finding(
            "analysis_cohort_identity_mismatch",
            "The produced analysis_cohort does not preserve the exact ordered row "
            "identity of the Planner-locked cohort.",
            identity_column=identity_column,
            produced_cohort_n=int(len(produced)),
            authoritative_cohort_n=locked_n,
        )
    try:
        _assert_exact_frame_values_equal(
            produced.loc[:, authoritative.columns].reset_index(drop=True),
            authoritative,
        )
    except (AssertionError, KeyError) as exc:
        return finding(
            "analysis_cohort_value_mismatch",
            "The produced analysis_cohort changes or omits authoritative cohort "
            "columns; only additional derived columns are allowed.",
            error=str(exc)[:300],
        )

    reported_raw_n, raw_denominators_by_field, raw_denominator_issue = (
        _coherent_integral_mapping_counts(
            step_summary,
            _PRIMARY_RAW_DENOMINATOR_FIELDS,
        )
    )
    reported_final_n, final_denominators_by_field, final_denominator_issue = (
        _coherent_integral_mapping_counts(
            step_summary,
            _PRIMARY_FINAL_DENOMINATOR_FIELDS,
        )
    )
    if (
        raw_denominator_issue == "nonintegral"
        or final_denominator_issue == "nonintegral"
    ):
        return finding(
            "cohort_denominator_fields_nonintegral",
            "The primary cohort summary contains a non-integral structural "
            "denominator in one of its synonymous fields.",
            raw_denominators_by_field=raw_denominators_by_field,
            final_denominators_by_field=final_denominators_by_field,
        )
    if raw_denominator_issue == "disagree" or final_denominator_issue == "disagree":
        return finding(
            "cohort_denominator_fields_disagree",
            "The primary cohort summary contains contradictory synonymous "
            "structural denominator fields.",
            raw_denominators_by_field=raw_denominators_by_field,
            final_denominators_by_field=final_denominators_by_field,
        )
    if reported_raw_n != raw_n or reported_final_n != locked_n:
        return finding(
            "cohort_denominator_mismatch",
            "The primary cohort summary does not report the raw-universe and "
            "locked final denominators exactly.",
            expected_universe_n=raw_n,
            reported_universe_n=reported_raw_n,
            expected_final_n=locked_n,
            reported_final_n=reported_final_n,
        )

    declared_flow_products = {
        name
        for raw in (step.expected_outputs or [])
        if (parsed := typed_product(raw)) is not None
        for kind, name in (parsed,)
        if kind == "table" and name in _PRIMARY_ANALYSIS_COHORT_FLOW_PRODUCTS
    }
    for product in sorted(declared_flow_products):
        candidates = _registered_product_paths(
            step_summary,
            product_name=product,
            allowed_kinds=frozenset({"table"}),
            out_dir=out_dir,
        )
        if len(candidates) != 1:
            return finding(
                "attrition_product_ambiguous",
                f"The declared {product} product must resolve to exactly one "
                "registered table.",
                product=product,
                candidates=candidates,
            )
        try:
            table_path = _contained_regular_output_file(Path(out_dir), candidates[0])
            if table_path.suffix.lower() == ".parquet":
                table = pd.read_parquet(table_path)
            elif table_path.suffix.lower() == ".feather":
                table = pd.read_feather(table_path)
            else:
                table = pd.read_csv(
                    table_path,
                    sep="\t" if table_path.suffix.lower() == ".tsv" else ",",
                )
        except Exception as exc:
            return finding(
                "attrition_product_unreadable",
                f"The declared {product} table is unreadable or outside the step "
                "output directory.",
                product=product,
                error_type=type(exc).__name__,
                error=str(exc)[:300],
            )
        if table.empty:
            return finding(
                "attrition_counts_unverifiable",
                f"The declared {product} table has no verifiable count sequence.",
                product=product,
            )
        # Only typed identity columns carry an exact host-owned predicate ID.
        # Legacy ``criterion`` remains descriptive free text; its row order and
        # verified counts provide the compatibility identity without routing on
        # case-specific wording.
        canonical_identity_columns = [
            name
            for name in ("criterion_id", "attrition_category")
            if name in table.columns
        ]
        normalised_identity_by_column = {
            name: [_normalise(value) for value in table[name].tolist()]
            for name in canonical_identity_columns
        }
        if len(canonical_identity_columns) > 1:
            reference = normalised_identity_by_column[canonical_identity_columns[0]]
            if any(
                normalised_identity_by_column[name] != reference
                for name in canonical_identity_columns[1:]
            ):
                return finding(
                    "attrition_identity_columns_disagree",
                    f"The declared {product} table has contradictory canonical "
                    "Planner-predicate identity columns.",
                    product=product,
                    identity_columns=canonical_identity_columns,
                    identities_by_column=normalised_identity_by_column,
                )
        canonical_identity_column = (
            canonical_identity_columns[0] if canonical_identity_columns else None
        )

        remaining_column, remaining_by_column, remaining_issue = (
            _coherent_integral_count_columns(
                table,
                ("n_remaining", "n_remaining_rows"),
            )
        )
        partition_count_column, partition_by_column, partition_issue = (
            _coherent_integral_count_columns(table, ("n", "n_rows"))
        )
        if remaining_column is None and partition_count_column is None:
            denominator_row = table.iloc[0].to_dict()
            table_raw_n, table_raw_by_field, table_raw_issue = (
                _coherent_integral_mapping_counts(
                    denominator_row,
                    _PRIMARY_RAW_DENOMINATOR_FIELDS,
                )
            )
            table_final_n, table_final_by_field, table_final_issue = (
                _coherent_integral_mapping_counts(
                    denominator_row,
                    _PRIMARY_FINAL_DENOMINATOR_FIELDS,
                )
            )
            if table_raw_issue == "nonintegral" or table_final_issue == "nonintegral":
                return finding(
                    "cohort_denominator_fields_nonintegral",
                    f"The declared {product} table contains a non-integral "
                    "structural denominator in one of its synonymous fields.",
                    product=product,
                    raw_denominators_by_field=table_raw_by_field,
                    final_denominators_by_field=table_final_by_field,
                )
            if table_raw_issue == "disagree" or table_final_issue == "disagree":
                return finding(
                    "cohort_denominator_fields_disagree",
                    f"The declared {product} table contains contradictory "
                    "synonymous structural denominator fields.",
                    product=product,
                    raw_denominators_by_field=table_raw_by_field,
                    final_denominators_by_field=table_final_by_field,
                )
            if (
                product in {"cohort_denominator", "cohort_denominators"}
                and len(table) == 1
                and table_raw_n == raw_n
                and table_final_n == locked_n
            ):
                continue
            return finding(
                "attrition_counts_unverifiable",
                f"The declared {product} table has no verifiable count sequence.",
                product=product,
            )
        if remaining_column is not None:
            count_column = remaining_column
            count_values_by_column = remaining_by_column
            count_issue = remaining_issue
        else:
            count_column = partition_count_column
            count_values_by_column = partition_by_column
            count_issue = partition_issue
        assert count_column is not None
        if count_issue == "nonintegral":
            return finding(
                "attrition_counts_nonintegral",
                f"The declared {product} table contains invalid structural counts.",
                product=product,
                count_column=count_column,
                count_columns=list(count_values_by_column),
            )
        if count_issue == "disagree":
            return finding(
                "attrition_count_columns_disagree",
                f"The declared {product} table has contradictory synonymous "
                "structural-count columns.",
                product=product,
                count_columns=list(count_values_by_column),
                values_by_column=count_values_by_column,
            )
        integral_counts = [
            int(value)
            for value in count_values_by_column[count_column]
            if value is not None
        ]
        if any(value < 0 for value in integral_counts):
            return finding(
                "attrition_counts_negative",
                f"The declared {product} table contains negative structural counts.",
                product=product,
                count_column=count_column,
            )

        # Sequential flow schema: every row records the population remaining
        # after one rule.  When exclusions are explicit, each transition must
        # reconcile row by row; a matching grand total alone is insufficient.
        if remaining_column is not None:
            if integral_counts[0] != raw_n or integral_counts[-1] != locked_n:
                return finding(
                    "attrition_endpoints_mismatch",
                    f"The declared {product} table does not run from the raw "
                    "universe to the locked analysis cohort.",
                    product=product,
                    expected_universe_n=raw_n,
                    reported_first_n=integral_counts[0],
                    expected_final_n=locked_n,
                    reported_last_n=integral_counts[-1],
                )
            if any(
                current > previous
                for previous, current in zip(
                    integral_counts, integral_counts[1:], strict=False
                )
            ):
                return finding(
                    "attrition_counts_increase",
                    f"The declared {product} remaining-count sequence increases.",
                    product=product,
                )
            excluded_column, exclusions_by_column, exclusion_issue = (
                _coherent_integral_count_columns(
                    table,
                    (
                        "n_excluded_at_step",
                        "n_removed_from_prior_stage",
                        "n_excluded_rows",
                    ),
                )
            )
            if excluded_column is None:
                return finding(
                    "attrition_exclusions_unverifiable",
                    f"The declared {product} table does not report per-step "
                    "exclusions.",
                    product=product,
                )
            if exclusion_issue == "nonintegral":
                return finding(
                    "attrition_exclusions_nonintegral",
                    f"The declared {product} table contains invalid per-step "
                    "exclusion counts.",
                    product=product,
                    count_columns=list(exclusions_by_column),
                )
            if exclusion_issue == "disagree":
                return finding(
                    "attrition_count_columns_disagree",
                    f"The declared {product} table has contradictory synonymous "
                    "per-step exclusion-count columns.",
                    product=product,
                    count_columns=list(exclusions_by_column),
                    values_by_column=exclusions_by_column,
                )
            integral_exclusions = [
                int(value)
                for value in exclusions_by_column[excluded_column]
                if value is not None
            ]
            reported_remaining = list(integral_counts)
            reported_exclusions = list(integral_exclusions)
            has_terminal_row = False
            if (
                len(reported_remaining) == len(expected_remaining_counts) + 1
                and reported_remaining[-1] == locked_n
                and reported_exclusions[-1] == 0
            ):
                has_terminal_row = True
                reported_remaining.pop()
                reported_exclusions.pop()
            if reported_remaining != expected_remaining_counts:
                return finding(
                    "attrition_stage_counts_mismatch",
                    f"The declared {product} remaining counts do not match the "
                    "ordered Planner-locked cohort predicates.",
                    product=product,
                    expected_remaining_counts=expected_remaining_counts,
                    reported_remaining_counts=reported_remaining,
                )
            if reported_exclusions != expected_exclusion_counts:
                return finding(
                    "attrition_transitions_do_not_conserve",
                    f"The declared {product} exclusions do not match each "
                    "ordered Planner-locked cohort predicate.",
                    product=product,
                    expected_exclusions=expected_exclusion_counts,
                    reported_exclusions=reported_exclusions,
                )
            if canonical_identity_column is not None:
                reported_rule_ids = normalised_identity_by_column[
                    canonical_identity_column
                ]
                expected_rule_ids = ["universe", *expected_criterion_ids]
                reported_predicate_ids = (
                    reported_rule_ids[:-1] if has_terminal_row else reported_rule_ids
                )
                reported_boundary_id = (
                    reported_predicate_ids[0] if reported_predicate_ids else None
                )
                canonical_reported_predicates = canonicalise_reported_rule_ids(
                    reported_predicate_ids[1:]
                )
                allowed_terminal_ids = {
                    "analysis_cohort",
                    "final_analysis_cohort",
                    "final_cohort",
                    "primary_analysis_cohort",
                }
                terminal_id_valid = (
                    not has_terminal_row
                    or reported_rule_ids[-1] in allowed_terminal_ids
                )
                if (
                    reported_boundary_id != "universe"
                    or canonical_reported_predicates != expected_criterion_ids
                    or not terminal_id_valid
                ):
                    return finding(
                        "attrition_sequence_rule_ids_mismatch",
                        f"The declared {product} canonical identity sequence does "
                        "not bind each row to the corresponding Planner-locked "
                        "cohort predicate.",
                        product=product,
                        identity_column=canonical_identity_column,
                        expected_criterion_ids=expected_rule_ids,
                        accepted_criterion_ids=[
                            ["universe"],
                            *[list(aliases) for aliases in accepted_criterion_ids],
                        ],
                        reported_criterion_ids=reported_rule_ids,
                    )
            start_column = (
                "n_at_start_rows" if "n_at_start_rows" in table.columns else None
            )
            if start_column is not None:
                starts = [
                    _integral_count(value) for value in table[start_column].tolist()
                ]
                expected_starts = [raw_n, *integral_counts[:-1]]
                if (
                    any(value is None for value in starts)
                    or [int(value) for value in starts if value is not None]
                    != expected_starts
                ):
                    return finding(
                        "attrition_start_counts_do_not_conserve",
                        f"The declared {product} start counts do not match the "
                        "preceding remaining denominators.",
                        product=product,
                        expected_start_counts=expected_starts,
                    )
            continue

        # Partition schema: one denominator row, one retained row, and zero or
        # more mutually exclusive excluded partitions.  A bare ``n`` column is
        # never treated as an ordered flow because that would make arbitrary
        # intermediate values unverifiable.
        assert partition_count_column is not None
        role_columns = [
            name
            for name in ("status", "partition_status", "row_role", "role")
            if name in table.columns
        ]
        normalised_roles_by_column = {
            name: [_normalise(value) for value in table[name].tolist()]
            for name in role_columns
        }
        if len(role_columns) > 1:
            reference = normalised_roles_by_column[role_columns[0]]
            if any(
                normalised_roles_by_column[name] != reference
                for name in role_columns[1:]
            ):
                return finding(
                    "attrition_role_columns_disagree",
                    f"The declared {product} partition table has contradictory "
                    "synonymous row-role columns.",
                    product=product,
                    role_columns=role_columns,
                    roles_by_column=normalised_roles_by_column,
                )
        role_column = role_columns[0] if role_columns else None
        if role_column is None:
            if partition_count_column == "n_rows":
                reported_remaining = list(integral_counts)
                if (
                    len(reported_remaining) == len(expected_remaining_counts) + 1
                    and reported_remaining[-1] == locked_n
                ):
                    reported_remaining.pop()
                if reported_remaining == expected_remaining_counts:
                    continue
                return finding(
                    "attrition_stage_counts_mismatch",
                    f"The declared {product} row-count sequence does not match "
                    "the ordered Planner-locked cohort predicates.",
                    product=product,
                    expected_remaining_counts=expected_remaining_counts,
                    reported_remaining_counts=reported_remaining,
                )
            return finding(
                "attrition_partition_roles_unverifiable",
                f"The declared {product} partition table has no explicit row roles.",
                product=product,
                count_column=partition_count_column,
            )
        roles = normalised_roles_by_column[role_column]
        denominator_rows = [
            index for index, role in enumerate(roles) if role == "denominator"
        ]
        retained_rows = [
            index for index, role in enumerate(roles) if role == "retained"
        ]
        excluded_rows = [
            index for index, role in enumerate(roles) if role == "excluded"
        ]
        if (
            len(denominator_rows) != 1
            or len(retained_rows) != 1
            or len(denominator_rows) + len(retained_rows) + len(excluded_rows)
            != len(table)
        ):
            return finding(
                "attrition_partition_roles_invalid",
                f"The declared {product} partition table must contain exactly "
                "one denominator row, one retained row, and only explicit "
                "excluded partitions otherwise.",
                product=product,
                role_column=role_column,
            )
        reported_denominator = integral_counts[denominator_rows[0]]
        reported_retained = integral_counts[retained_rows[0]]
        reported_excluded_counts = [integral_counts[index] for index in excluded_rows]
        reported_excluded = sum(reported_excluded_counts)
        if (
            reported_denominator != raw_n
            or reported_retained != locked_n
            or reported_excluded != raw_n - locked_n
        ):
            return finding(
                "attrition_partitions_do_not_conserve",
                f"The declared {product} partitions do not reconcile the raw "
                "and final cohort denominators.",
                product=product,
                expected_universe_n=raw_n,
                reported_universe_n=reported_denominator,
                expected_retained_n=locked_n,
                reported_retained_n=reported_retained,
                expected_excluded_n=raw_n - locked_n,
                reported_excluded_n=reported_excluded,
            )
        category_column = canonical_identity_column
        if expected_criterion_ids and category_column is None:
            return finding(
                "attrition_partition_rule_ids_unverifiable",
                f"The declared {product} partition table does not identify the "
                "Planner-locked cohort rules attached to each exclusion.",
                product=product,
                expected_criterion_ids=expected_criterion_ids,
            )
        if expected_criterion_ids:
            assert category_column is not None
            denominator_id = _normalise(
                table.iloc[denominator_rows[0]][category_column]
            )
            retained_id = _normalise(table.iloc[retained_rows[0]][category_column])
            if denominator_id != "universe" or retained_id not in {
                "analysis_cohort",
                "final_analysis_cohort",
                "final_cohort",
                "primary_analysis_cohort",
            }:
                return finding(
                    "attrition_partition_boundary_ids_mismatch",
                    f"The declared {product} canonical identity column does not "
                    "identify the universe and retained analysis cohort exactly.",
                    product=product,
                    identity_column=category_column,
                    reported_denominator_id=denominator_id,
                    reported_retained_id=retained_id,
                )
            reported_rule_ids = [
                _normalise(table.iloc[index][category_column])
                for index in excluded_rows
            ]
            canonical_reported_rule_ids = canonicalise_reported_rule_ids(
                reported_rule_ids
            )
            if (
                canonical_reported_rule_ids is None
                or len(canonical_reported_rule_ids)
                != len(set(canonical_reported_rule_ids))
                or set(canonical_reported_rule_ids) != set(expected_criterion_ids)
            ):
                return finding(
                    "attrition_partition_rule_ids_mismatch",
                    f"The declared {product} excluded partitions do not identify "
                    "each Planner-locked cohort predicate exactly once.",
                    product=product,
                    expected_criterion_ids=expected_criterion_ids,
                    accepted_criterion_ids=[
                        list(aliases) for aliases in accepted_criterion_ids
                    ],
                    reported_criterion_ids=reported_rule_ids,
                )
            reported_by_rule = dict(
                zip(
                    canonical_reported_rule_ids,
                    reported_excluded_counts,
                    strict=True,
                )
            )
            expected_by_rule = dict(
                zip(
                    expected_criterion_ids,
                    expected_exclusion_counts[1:],
                    strict=True,
                )
            )
        else:
            reported_by_rule = {}
            expected_by_rule = {}
        if reported_by_rule != expected_by_rule:
            return finding(
                "attrition_partition_rule_counts_mismatch",
                f"The declared {product} excluded partitions do not match the "
                "Planner-locked cohort predicates.",
                product=product,
                expected_by_rule=expected_by_rule,
                reported_by_rule=reported_by_rule,
            )

    return []


def _summary_scalar_products(value: Any) -> set[tuple[str, str]]:
    """Return exact statistic/log keys backed by non-null scalar values."""

    products: set[tuple[str, str]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            keys = {_normalise(raw_key) for raw_key in node}
            if keys.intersection(_PRODUCT_DESCRIPTOR_FIELDS):
                return
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                if key in _HOST_RECEIPT_SUBTREES or key in _OUTPUT_CONTAINER_KEYS:
                    continue
                if isinstance(child, Mapping) or isinstance(child, (list, tuple)):
                    visit(child)
                    continue
                valid = (
                    child is not None
                    and child != ""
                    and not (isinstance(child, bool) and child is False)
                    and not _is_explicit_effect_absence_flag(key, child)
                    and not _is_file_path(child)
                    and not _is_noncomputed_result_status(child)
                )
                if isinstance(child, float):
                    valid = math.isfinite(child)
                if valid and key:
                    products.add(("statistic", key))
                    products.add(("log", key))
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return products


def _finite_json_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


_NONCOMPUTED_RESULT_STATUSES = frozenset(
    {
        "deferred",
        "non_estimable",
        "not_applicable",
        "not_computed",
        "not_estimable",
        "not_estimated",
        "skipped",
        "unavailable",
    }
)


def _is_noncomputed_result_status(value: Any) -> bool:
    """Return whether a scalar explicitly says no scientific result exists."""

    return isinstance(value, str) and _normalise(value) in _NONCOMPUTED_RESULT_STATUSES


def _is_explicit_effect_absence_flag(name: object, value: Any) -> bool:
    """Recognise a true boolean that explicitly says no effect result exists.

    These flags document output scope; they are not themselves scientific
    statistics.  Keep the rule deliberately narrow: only a literal boolean
    ``True`` and a normalized ``no_effect...`` field qualify.  Positive flags
    such as ``effect_estimates_created=True`` remain fail-closed.
    """

    return value is True and _normalise(name).startswith("no_effect")


def _valid_inline_statistic_descriptor(
    value: Mapping[str, Any], descriptor: tuple[str, str]
) -> bool:
    """Validate the closed pathless statistic receipt used by accounting steps."""

    if set(value) != {"kind", "name", "role", "value"}:
        return False
    if descriptor[0] != "statistic" or _normalise(value["role"]) != descriptor[1]:
        return False
    return _finite_json_number(value["value"])


def _typed_kind_shorthand_receipt(
    value: Mapping[str, Any], *, out_dir: Path | None
) -> tuple[tuple[str, str], Path] | None:
    """Resolve one closed ``kind:product`` descriptor to its exact file."""

    raw_kind = value.get("kind")
    if not isinstance(raw_kind, str) or raw_kind.strip().count(":") != 1:
        return None
    descriptor = _typed_product(raw_kind)
    if (
        descriptor is None
        or descriptor[0] not in PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS
        or out_dir is None
    ):
        return None

    if "name" in value:
        raw_name = value.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None
        name_descriptor = _typed_product(raw_name)
        if name_descriptor is None:
            name_descriptor = _typed_product(f"{descriptor[0]}:{raw_name}")
        if name_descriptor != descriptor:
            return None
    if "product_type" in value and (
        _canonical_kind(value.get("product_type")) != descriptor[0]
    ):
        return None

    # Normalize only the caller's coordinate system.  ``absolute()`` makes an
    # absolute descriptor comparable with a relative ``out_dir`` without
    # resolving any symlink or granting authority outside the lexical root.
    root = Path(out_dir).absolute()
    candidates: list[Path] = []
    unresolved_relative_paths: list[Path] = []
    for field in ("path", "relative_path", "filename"):
        if field not in value:
            continue
        raw_path = value.get(field)
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        supplied_path = Path(raw_path.strip())
        if ".." in supplied_path.parts:
            return None
        if supplied_path.is_absolute():
            try:
                supplied_path = supplied_path.relative_to(root)
            except ValueError:
                return None
        try:
            candidates.append(_contained_regular_output_file(root, supplied_path))
        except ValueError:
            if Path(raw_path.strip()).is_absolute():
                return None
            unresolved_relative_paths.append(supplied_path)
    if not candidates:
        return None

    authoritative_stat = candidates[0].stat()
    authoritative_identity = (authoritative_stat.st_dev, authoritative_stat.st_ino)
    if any(
        (candidate.stat().st_dev, candidate.stat().st_ino) != authoritative_identity
        for candidate in candidates[1:]
    ):
        return None

    # A run-relative alias such as ``steps/03/outputs/result.csv`` has meaning
    # only after another field has identified the authoritative output-local
    # file.  Project it from ancestors of ``out_dir`` and accept it solely when
    # it resolves, without links, to that same inode.  This never grants
    # authority from a filename or ancestor lookup alone.
    for relative_path in unresolved_relative_paths:
        matching_candidate: Path | None = None
        for ancestor in root.parents:
            projected = ancestor / relative_path
            try:
                output_relative = projected.relative_to(root)
                candidate = _contained_regular_output_file(root, output_relative)
                candidate_stat = candidate.stat()
            except (ValueError, OSError):
                continue
            if (candidate_stat.st_dev, candidate_stat.st_ino) == authoritative_identity:
                matching_candidate = candidate
                break
        if matching_candidate is None:
            return None

    if not _descriptor_path_is_compatible(kind=descriptor[0], path=str(candidates[0])):
        return None
    return descriptor, candidates[0]


def _registered_products(
    summary: Mapping[str, Any],
    *,
    out_dir: Path | None = None,
) -> tuple[set[tuple[str, str]], list[tuple[str, bool]]]:
    """Collect typed/file products and figure paths from output containers."""

    products: set[tuple[str, str]] = set()
    figure_paths: list[tuple[str, bool]] = []

    def is_actual_output(path: str) -> bool:
        if out_dir is None:
            return True
        root = out_dir.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError):
            return False
        return resolved.is_file()

    def add_path(path: str, *, explicit_figure_list: bool = False) -> None:
        kinds = _file_kinds(path)
        stem = _file_stem(path)
        products.update((kind, stem) for kind in kinds if stem)
        if "figure" in kinds:
            figure_paths.append((path, explicit_figure_list))

    def add_container(value: Any, *, explicit_figure_list: bool = False) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                add_container(item, explicit_figure_list=explicit_figure_list)
            return
        if isinstance(value, Mapping):
            # Coder outputs commonly use a structured product receipt rather
            # than a ``kind:name -> path`` mapping.  Bind only the closed
            # ``kind``/``name``/``path`` shape and only when the physical file
            # class is compatible with the canonical typed kind.  This lets a
            # harmless filename prefix differ from the logical product name
            # without turning arbitrary summary prose into product authority.
            raw_descriptor_kind = value.get("kind")
            if isinstance(raw_descriptor_kind, str) and ":" in raw_descriptor_kind:
                shorthand = _typed_kind_shorthand_receipt(value, out_dir=out_dir)
                if shorthand is not None:
                    shorthand_descriptor, shorthand_path = shorthand
                    products.add(shorthand_descriptor)
                    if shorthand_descriptor[0] == "figure":
                        figure_paths.append((str(shorthand_path), explicit_figure_list))
                return

            descriptor_kind = raw_descriptor_kind or value.get("product_type")
            descriptor_name = value.get("name")
            descriptor_keys = {_normalise(raw_key) for raw_key in value}
            descriptor_candidate = bool(
                descriptor_keys.intersection(_PRODUCT_DESCRIPTOR_FIELDS)
            )
            typed_descriptor_name = _typed_product(descriptor_name)
            canonical_descriptor_kind = _canonical_kind(descriptor_kind)
            if typed_descriptor_name is not None:
                descriptor = (
                    typed_descriptor_name
                    if canonical_descriptor_kind == typed_descriptor_name[0]
                    else None
                )
            else:
                descriptor = _typed_product(
                    f"{descriptor_kind}:{descriptor_name}"
                    if descriptor_kind is not None and descriptor_name is not None
                    else ""
                )
            descriptor_path = next(
                (
                    value.get(key)
                    for key in ("path", "relative_path", "filename")
                    if isinstance(value.get(key), str) and str(value.get(key)).strip()
                ),
                None,
            )
            descriptor_paths = [
                path for path in _iter_paths(descriptor_path) if is_actual_output(path)
            ]
            scalar_descriptor = (
                descriptor is not None
                and _valid_inline_statistic_descriptor(value, descriptor)
            )
            compatible_descriptor_path = descriptor is not None and any(
                _descriptor_path_is_compatible(kind=descriptor[0], path=path)
                for path in descriptor_paths
            )
            if scalar_descriptor or compatible_descriptor_path:
                products.add(descriptor)
            if descriptor_candidate:
                if compatible_descriptor_path:
                    for path in descriptor_paths:
                        add_path(path, explicit_figure_list=explicit_figure_list)
                return
            for raw_role, child in value.items():
                role = _typed_product(raw_role)
                paths = [path for path in _iter_paths(child) if is_actual_output(path)]
                if role is not None:
                    role_kind, role_name = role
                    compatible_path = any(
                        role_kind in _file_kinds(path) for path in paths
                    )
                    scalar_registration = (
                        role_kind in {"statistic", "log"}
                        and not isinstance(child, (Mapping, list, tuple, set))
                        and child is not None
                        and child != ""
                    )
                    if compatible_path or scalar_registration:
                        products.add(role)
                elif paths:
                    role_name = _normalise(raw_role)
                    for path in paths:
                        products.update(
                            (kind, role_name) for kind in _file_kinds(path) if role_name
                        )
                for path in paths:
                    # A JSON file registered under an exact matching typed log
                    # role is an auxiliary sidecar, not four extra scientific
                    # products merely because JSON is a multi-purpose format.
                    # Keep suffix inference for every scientific role and for
                    # mismatched log filenames so an innocuous key cannot
                    # launder an effect-bearing output path.
                    if (
                        role is not None
                        and role[0] == "log"
                        and role[0] in _file_kinds(path)
                        and _file_stem(path) == role[1]
                    ):
                        continue
                    add_path(path, explicit_figure_list=explicit_figure_list)
            return
        for path in _iter_paths(value):
            add_path(path, explicit_figure_list=explicit_figure_list)

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                if key in _OUTPUT_CONTAINER_KEYS:
                    add_container(child, explicit_figure_list=key == "figure_files")
                elif key in _DIRECT_FIGURE_KEYS:
                    add_container(child, explicit_figure_list=True)
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(summary)
    products.update(_summary_scalar_products(summary))
    return products, figure_paths


def _has_product_registry(value: Any) -> bool:
    """Whether a summary opted into the machine-readable output registry."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if _normalise(raw_key) in {
                "output_files",
                "output_artifacts",
                "outputs",
            }:
                return True
            if isinstance(child, (Mapping, list, tuple)) and _has_product_registry(
                child
            ):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_product_registry(child) for child in value)
    return False


def _effect_bearing_name(name: str) -> bool:
    normalised = _normalise(name)
    # Input/design audits and provenance sidecars may name the effect they
    # inspect, but they are not authority to estimate or register that effect.
    # Any numeric effect nested inside such a summary is still caught by the
    # scalar-path walk below; this exemption applies only to the typed product
    # role itself.
    if any(
        normalised.endswith(f"_{suffix}")
        for suffix in (
            "audit",
            "contract",
            "definition",
            "diagnostic",
            "diagnostics",
            "input_audit",
            "lineage",
            "provenance",
            "trace",
        )
    ):
        return False
    return any(
        _contains_product_role(normalised, base) for base in _EFFECT_PRODUCT_BASES
    )


def _contains_product_role(name: str, role: str) -> bool:
    """Match a normalized multi-token role at any underscore boundary."""

    return name == role or f"_{role}_" in f"_{name}_"


def effect_bearing_product(value: object) -> bool:
    """Return whether a typed product name denotes a scientific effect."""

    parsed = typed_product(value)
    return parsed is not None and _effect_bearing_name(parsed[1])


def effect_bearing_name(value: object) -> bool:
    """Return whether an untyped/canonical product name denotes an effect."""

    return _effect_bearing_name(str(value or ""))


_EFFECT_MEASURE_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "odds_ratio": (
        "odds_ratio",
        "adjusted_odds_ratio",
        "adjusted_odds_ratios",
        "primary_or",
        "adjusted_or",
        "or_estimate",
        "or_estimates",
        "or_forest",
    ),
    "risk_ratio": (
        "risk_ratio",
        "relative_risk",
        "primary_rr",
        "adjusted_rr",
        "rr_estimate",
        "rr_estimates",
        "rr_forest",
    ),
    "hazard_ratio": (
        "hazard_ratio",
        "primary_hr",
        "adjusted_hr",
        "hr_estimate",
        "hr_estimates",
        "hr_forest",
    ),
    "risk_difference": (
        "risk_difference",
        "primary_rd",
        "adjusted_rd",
        "rd_estimate",
        "rd_estimates",
        "rd_forest",
    ),
    "coefficient": ("coefficient", "coefficients"),
}

# These abbreviations are meaningful only when they are the complete typed
# product/column name.  Boundary matching would misclassify ordinary ICU
# products such as ``hr_trajectory`` (heart rate), ``rr_distribution``
# (respiratory rate), or prose-like ``included_or_excluded_counts``.
_EXACT_EFFECT_MEASURE_NAMES: Mapping[str, str] = {
    "or": "odds_ratio",
    "rr": "risk_ratio",
    "hr": "hazard_ratio",
    "rd": "risk_difference",
}


def effect_measure_family(value: object) -> str | None:
    """Return an explicit effect scale encoded in a typed product name."""

    parsed = typed_product(value)
    if parsed is None:
        return None
    name = parsed[1]
    exact_family = _EXACT_EFFECT_MEASURE_NAMES.get(name)
    if exact_family is not None:
        return exact_family
    for family, prefixes in _EFFECT_MEASURE_PREFIXES.items():
        if any(_contains_product_role(name, prefix) for prefix in prefixes):
            return family
    return None


_EFFECT_ROLE_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "interaction": ("interaction", "interaction_pvalue"),
    "subgroup": ("subgroup", "subgroup_effect", "subgroup_effects"),
    "treatment": ("treatment", "treatment_effect"),
    "causal": ("causal", "causal_effect"),
}


def effect_role_family(value: object) -> str | None:
    """Return a non-interchangeable scientific effect role, if explicit."""

    parsed = typed_product(value)
    if parsed is None:
        return None
    name = parsed[1]
    for family, prefixes in _EFFECT_ROLE_PREFIXES.items():
        if any(_contains_product_role(name, prefix) for prefix in prefixes):
            return family
    return None


_EFFECT_ESTIMAND_TIER_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "primary": ("primary",),
    "secondary": ("secondary",),
    "sensitivity": ("sensitivity", "robust", "robustness"),
    "corroborative": ("corroborative",),
}


def effect_estimand_tier(value: object) -> str | None:
    """Return a non-interchangeable primary/supporting estimand tier."""

    parsed = typed_product(value)
    if parsed is None:
        return None
    name = parsed[1]
    for tier, prefixes in _EFFECT_ESTIMAND_TIER_PREFIXES.items():
        if any(_contains_product_role(name, prefix) for prefix in prefixes):
            return tier
    return None


def effect_adjustment_family(value: object) -> str | None:
    """Return an explicit adjusted versus crude/unadjusted qualifier."""

    parsed = typed_product(value)
    if parsed is None:
        return None
    name = parsed[1]
    if _contains_product_role(name, "adjusted"):
        return "adjusted"
    if any(
        _contains_product_role(name, qualifier) for qualifier in ("unadjusted", "crude")
    ):
        return "unadjusted"
    return None


_AUXILIARY_LOG_SUFFIXES = (
    "_audit",
    "_contract",
    "_diagnostic",
    "_diagnostics",
    "_lineage",
    "_process",
    "_provenance",
    "_render_trace",
    "_rendering_trace",
    "_trace",
)


def _auxiliary_log_name(name: str) -> bool:
    normalised = _normalise(name)
    return any(normalised.endswith(suffix) for suffix in _AUXILIARY_LOG_SUFFIXES)


def _effect_summary_paths(summary: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []

    def visit(node: Any, prefix: str = "") -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                path = f"{prefix}.{key}" if prefix else key
                if key in _HOST_RECEIPT_SUBTREES:
                    continue
                if isinstance(child, Mapping) or isinstance(child, (list, tuple)):
                    visit(child, path)
                elif (
                    key
                    and _effect_bearing_name(key)
                    and child is not None
                    and child != ""
                    and not (isinstance(child, bool) and child is False)
                    and not _is_explicit_effect_absence_flag(key, child)
                    and not _is_file_path(child)
                    and not _is_noncomputed_result_status(child)
                ):
                    paths.append(path)
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, f"{prefix}[{index}]")

    visit(summary)
    return sorted(set(paths))


def _undeclared_figure_bundle(
    figure_paths: Sequence[tuple[str, bool]],
) -> dict[str, list[str]]:
    by_stem: dict[str, set[str]] = defaultdict(set)
    explicit_stems: set[str] = set()
    for path, explicit in figure_paths:
        stem = _file_stem(path)
        if not stem:
            continue
        by_stem[stem].add(Path(path).suffix.lower())
        if explicit:
            explicit_stems.add(stem)
    return {
        stem: sorted(suffixes)
        for stem, suffixes in by_stem.items()
        if len(suffixes) >= 2 or stem in explicit_stems
    }


def _assignment_model_completion_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    declared: set[tuple[str, str]],
) -> list[ValidationFinding]:
    """Require a declared assignment-model artifact to contain a fitted model.

    This validates realization of the Planner-owned method; it does not select
    the exposure, covariates, model family, cohort, or estimand.
    """

    if ("artifact", "assignment_model") not in declared:
        return []

    raw_models = step_summary.get("assignment_models")
    models = raw_models if isinstance(raw_models, list) else []
    fitted_models = [
        model
        for model in models
        if isinstance(model, Mapping)
        and _normalise(model.get("fit_status") or model.get("status")) == "fitted"
    ]
    if fitted_models:
        return []

    exposure = step_summary.get("exposure")
    exposure_resolution = (
        exposure.get("resolution")
        if isinstance(exposure, Mapping)
        and isinstance(exposure.get("resolution"), Mapping)
        else {}
    )
    model_diagnostics = [
        {
            "model_id": model.get("model_id"),
            "fit_status": _normalise(model.get("fit_status") or model.get("status")),
            "n": model.get("n"),
            "exposure_event_n": model.get("exposure_event_n"),
            "exposure_non_event_n": model.get("exposure_non_event_n"),
            "error": model.get("error"),
        }
        for model in models
        if isinstance(model, Mapping)
    ]
    overall_event_n = exposure.get("event_n") if isinstance(exposure, Mapping) else None
    overall_non_event_n = (
        exposure.get("non_event_n") if isinstance(exposure, Mapping) else None
    )
    exposure_class_collapse = bool(
        isinstance(overall_event_n, int)
        and isinstance(overall_non_event_n, int)
        and overall_event_n > 0
        and overall_non_event_n > 0
        and any(
            (
                diagnostic.get("exposure_event_n") == 0
                or diagnostic.get("exposure_non_event_n") == 0
            )
            for diagnostic in model_diagnostics
        )
    )
    return [
        ValidationFinding(
            validator="declared_product_contract",
            severity="error",
            message=(
                f"Step {step.step_id} declared an assignment-model artifact but "
                "registered no assignment model whose summary roster has "
                "`fit_status` exactly `fitted`. Empty, placeholder, or "
                "noncanonical model rosters are not completed products."
            ),
            detail={
                "kind": "assignment_model_unfitted",
                "step_id": step.step_id,
                "model_statuses": [
                    _normalise(model.get("fit_status") or model.get("status"))
                    for model in models
                    if isinstance(model, Mapping)
                ],
                "exposure_resolution_status": _normalise(
                    exposure_resolution.get("status")
                ),
                "exposure_resolution_reason": exposure_resolution.get("reason"),
                "overall_exposure_event_n": overall_event_n,
                "overall_exposure_non_event_n": overall_non_event_n,
                "model_diagnostics": model_diagnostics,
                "exposure_class_collapse_after_eligibility": exposure_class_collapse,
                "repair_constraint": (
                    "The resolved exposure has both classes before model-set "
                    "eligibility but at least one model set has lost a class. "
                    "Audit analysis-set and timing eligibility symmetrically "
                    "for exposed and unexposed rows; do not redefine the "
                    "Planner-owned exposure, cohort, model, or estimand."
                    if exposure_class_collapse
                    else None
                ),
            },
        )
    ]


def _declared_diagnostic_completion_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    declared: set[tuple[str, str]],
) -> list[ValidationFinding]:
    """Do not let a placeholder diagnostic artifact satisfy a planned result."""

    if not any("diagnostic" in name for _kind, name in declared):
        return []
    diagnostic_status = _normalise(step_summary.get("diagnostic_status"))
    skipped_reason = str(step_summary.get("skipped_reason") or "").strip()
    incomplete_statuses = {
        *_FAILED_STATUSES,
        "not_computable",
        "not_computed",
        "unavailable",
    }
    if diagnostic_status not in incomplete_statuses and not skipped_reason:
        return []
    return [
        ValidationFinding(
            validator="declared_product_contract",
            severity="error",
            message=(
                f"Step {step.step_id} declared a diagnostic result but registered "
                "only a not-computed, unavailable, failed, or skipped placeholder. "
                "A file's presence does not complete the planned diagnostic."
            ),
            detail={
                "kind": "declared_diagnostic_not_completed",
                "step_id": step.step_id,
                "diagnostic_status": diagnostic_status or None,
                "skipped_reason": skipped_reason or None,
            },
        )
    ]


def declared_product_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    effect_method_authorized: bool,
    effect_figure_source_authorized: bool = False,
    out_dir: Path | None = None,
) -> list[ValidationFinding]:
    """Validate declared-product realization and scientific output scope."""

    reported_status = _normalise(step_summary.get("status"))
    if is_failed_step_status(reported_status):
        return []

    declared = {
        product
        for raw in (step.expected_outputs or [])
        if (product := _typed_product(raw)) is not None
    }
    registered, figure_paths = _registered_products(step_summary, out_dir=out_dir)
    findings: list[ValidationFinding] = []
    findings.extend(
        trajectory_role_scope_summary_findings(
            step=step,
            step_summary=step_summary,
        )
    )
    findings.extend(
        trajectory_role_result_findings(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
        )
    )
    findings.extend(
        _assignment_model_completion_findings(
            step=step,
            step_summary=step_summary,
            declared=declared,
        )
    )
    findings.extend(
        _declared_diagnostic_completion_findings(
            step=step,
            step_summary=step_summary,
            declared=declared,
        )
    )

    # Older direct unit fixtures predate ``output_files`` and validate only
    # their own numeric payload.  A real execution supplies ``out_dir`` and is
    # always held to the product boundary, even if its script tries to evade the
    # gate by omitting the modern registry entirely.
    missing = (
        sorted(declared - registered)
        if out_dir is not None or _has_product_registry(step_summary)
        else []
    )
    if missing:
        findings.append(
            ValidationFinding(
                validator="declared_product_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} did not realise every typed product "
                    "declared by the plan in step_summary output registrations."
                ),
                detail={
                    "kind": "declared_product_missing",
                    "step_id": step.step_id,
                    "missing_products": [f"{kind}:{name}" for kind, name in missing],
                    "declared_products": [
                        f"{kind}:{name}" for kind, name in sorted(declared)
                    ],
                    "registered_products": [
                        f"{kind}:{name}" for kind, name in sorted(registered)
                    ],
                },
            )
        )

    declares_figure = any(kind == "figure" for kind, _name in declared)
    figure_bundle = _undeclared_figure_bundle(figure_paths)
    if figure_bundle and not declares_figure:
        findings.append(
            ValidationFinding(
                validator="declared_product_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} produced a figure bundle without a "
                    "typed figure product in expected_outputs. Figure rendering "
                    "must remain in its declared figure owner step."
                ),
                detail={
                    "kind": "undeclared_figure_bundle",
                    "step_id": step.step_id,
                    "figure_bundle": figure_bundle,
                },
            )
        )

    if not effect_method_authorized:
        declared_effects = sorted(
            f"{kind}:{name}"
            for kind, name in declared
            if _effect_bearing_name(name)
            and not (kind == "log" and _auxiliary_log_name(name))
            and not (effect_figure_source_authorized and kind == "figure")
        )
        registered_effects = sorted(
            f"{kind}:{name}"
            for kind, name in registered
            if _effect_bearing_name(name)
            and not (
                kind == "log" and (kind, name) in declared and _auxiliary_log_name(name)
            )
            and not (
                effect_figure_source_authorized
                and kind == "figure"
                and (kind, name) in declared
            )
        )
        summary_effects = _effect_summary_paths(step_summary)
        if declared_effects or registered_effects or summary_effects:
            findings.append(
                ValidationFinding(
                    validator="declared_product_contract",
                    severity="error",
                    message=(
                        f"Step {step.step_id} lacks a closed effect-output contract "
                        "but declared or registered effect-bearing scientific output. "
                        "Use an agent-planned effect-method owner with a typed, "
                        "machine-readable effect result."
                    ),
                    detail={
                        "kind": "unauthorized_effect_product",
                        "step_id": step.step_id,
                        "planned_method": step.method,
                        "declared_effect_products": declared_effects,
                        "registered_effect_products": registered_effects,
                        "summary_effect_paths": summary_effects,
                    },
                )
            )

    return findings


__all__ = [
    "bind_declared_figure_products",
    "declared_product_contract_findings",
    "effect_adjustment_family",
    "effect_bearing_name",
    "effect_bearing_product",
    "effect_estimand_tier",
    "effect_measure_family",
    "effect_role_family",
    "typed_product",
    "typed_product_binding_contract",
    "typed_product_schema_receipt",
]
