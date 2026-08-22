"""Plan-level DAG contract for agent-decomposed trajectory phenotyping.

The planner may keep representation, candidate selection, stability freezing,
and descriptive characterization in one step or split them across steps.  This
module validates only that those agent-declared roles form one closed,
unambiguous artifact DAG.  It never selects a feature representation,
clustering method, cluster count, eligibility threshold, or scientific runner.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ClusterSelectionManifest,
    ResearchContext,
    ValidationFinding,
)

_ROLE_ORDER = (
    "representation",
    "candidate_selection",
    "stability_freeze",
    "characterization",
)

_ROLE_CANONICAL_OUTPUTS: Mapping[str, Tuple[str, ...]] = {
    "representation": (
        "table:trajectory_membership",
        "manifest:trajectory_representation_schema",
    ),
    "candidate_selection": (
        "manifest:cluster_selection",
        "manifest:candidate_cluster_solution_schema",
    ),
    "stability_freeze": (
        "manifest:trajectory_missingness_policy",
        "table:cluster_assignments",
        "table:cluster_stability",
        "table:cluster_stability_assignments",
    ),
    "characterization": (
        "table:trajectory_profiles",
        "table:cluster_sizes",
    ),
}

STABILITY_EXECUTOR_INPUTS = frozenset(
    {
        "artifact:trajectory_representation",
        "artifact:candidate_cluster_assignments",
        "manifest:cluster_selection",
        "manifest:trajectory_representation_schema",
        "manifest:candidate_cluster_solution_schema",
    }
)

# Canonical method head for the typed supporting calculator.  Scientific fit
# choices remain in the upstream manifests and planner-owned spec; this token
# prevents an unrelated method (for example cluster-robust regression) from
# claiming the calculator merely by declaring stability-shaped filenames.
TRAJECTORY_STABILITY_METHOD_HEAD = "trajectory_cluster_stability"
TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD = (
    "trajectory_cluster_stability_characterization"
)

#: The method key a group-discovery study declares when it has no fixed-window
#: trajectory. It is ``llm_coded`` with no runner in the method suite, which is
#: the registry's own way of saying general cluster stability stays agent-coded.
#: A contract test binds this to that registry entry so a rename cannot leave
#: this guide naming a method the Planner is not allowed to use.
_GENERAL_CLUSTER_STABILITY_METHOD = "cluster_stability"

TRAJECTORY_REPRESENTATION_SCHEMA_VERSION = "easyicu.trajectory_representation_schema/2"
TRAJECTORY_CANDIDATE_SOLUTION_SCHEMA_VERSION = (
    "easyicu.candidate_cluster_solution_schema/2"
)
OBSERVED_DATA_DIAG_GMM_METHOD = (
    "observed_data_diagonal_gaussian_mixture_candidate_selection"
)
OBSERVED_DATA_DIAG_GMM_MODEL_FAMILY = "latent_class_diagonal_gaussian_mixture"
OBSERVED_DATA_DIAG_GMM_FIT_METHOD = "observed_data_em_diagonal_gaussian_mixture"

STABILITY_EXECUTOR_OUTPUTS = frozenset(
    {
        "artifact:stability_freeze",
        "artifact:cluster_assignments",
        "manifest:cluster_stability_spec",
        "manifest:trajectory_missingness_policy",
        "table:cluster_assignments",
        "table:cluster_stability",
        "table:cluster_stability_assignments",
        "table:cluster_assignment_provenance",
    }
)

STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS = frozenset(
    {
        *STABILITY_EXECUTOR_OUTPUTS,
        "table:trajectory_profiles",
        "table:cluster_sizes",
    }
)

_CHARACTERIZATION_OUTCOME_PRODUCTS = frozenset(
    {
        "outcome_by_cluster",
        "cluster_outcomes",
        "cluster_outcome_summary",
        "cluster_mortality",
        "cluster_mortality_descriptive",
    }
)

_FIGURE_KINDS = frozenset({"figure", "plot", "chart", "fig", "heatmap"})
_ARTIFACT_KINDS = frozenset({"artifact", "dataset", "manifest"})

_REPRESENTATION_PRODUCTS = frozenset(
    {
        ("artifact", "trajectory_features"),
        ("artifact", "trajectory_feature_matrix"),
        ("artifact", "trajectory_representation"),
        ("dataset", "trajectory_features"),
        ("dataset", "trajectory_feature_matrix"),
        ("table", "trajectory_membership"),
        ("table", "trajectory_features"),
    }
)
_SELECTION_PRODUCTS = frozenset(
    {
        ("manifest", "cluster_selection"),
        ("table", "cluster_selection"),
        ("statistic", "cluster_selection"),
    }
)
_CANDIDATE_SOLUTION_PRODUCTS = frozenset(
    {
        ("artifact", "candidate_cluster_fits"),
        ("artifact", "cluster_assignments"),
        ("artifact", "stable_cluster_assignments"),
        ("dataset", "cluster_assignments"),
        ("model", "candidate_cluster_fits"),
        ("table", "cluster_assignments"),
    }
)
_STABILITY_PRODUCTS = frozenset(
    {
        ("manifest", "cluster_stability"),
        ("statistic", "cluster_stability"),
        ("table", "cluster_stability"),
    }
)
_STABILITY_ASSIGNMENT_PRODUCTS = frozenset(
    {
        ("artifact", "stable_cluster_assignments"),
        ("artifact", "cluster_assignments"),
        ("dataset", "stable_cluster_assignments"),
        ("table", "cluster_assignments"),
        ("table", "cluster_stability_assignments"),
    }
)
_CHARACTERIZATION_PRODUCTS = frozenset(
    {
        ("dataset", "trajectory_profiles"),
        ("table", "cluster_characteristics"),
        ("table", "cluster_profiles"),
        ("table", "phenotype_profiles"),
        ("table", "trajectory_profiles"),
    }
)

_REDUNDANT_SPLIT_ROLE_OUTPUTS: Mapping[str, frozenset[Tuple[str, str]]] = {
    "candidate_selection": frozenset({("table", "cluster_number_selection")}),
    "characterization": frozenset({("table", "cluster_sizes")}),
}


@dataclass(frozen=True)
class TrajectoryPlanDagEvaluation:
    """Deterministic result of the plan-level trajectory DAG audit."""

    applies: bool
    role_owners: Mapping[str, str]
    artifact_producers: Mapping[str, str]
    artifact_edges: Tuple[Tuple[str, str, str], ...]
    findings: Tuple[ValidationFinding, ...]


def _normalise_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _method_head(value: object) -> str:
    return _normalise_token(value).split("_with_", 1)[0]


def _tokens(value: object) -> frozenset[str]:
    return frozenset(token for token in _normalise_token(value).split("_") if token)


def _declared_product(value: object) -> Optional[Tuple[str, str]]:
    text = str(value or "").strip().lower()
    kind, separator, product = text.partition(":")
    if not separator:
        return None
    kind = _normalise_token(kind)
    product = product.rsplit("/", 1)[-1]
    product = re.sub(r"\.(?:csv|json|parquet|feather|tsv)$", "", product)
    product = _normalise_token(product)
    if not kind or not product:
        return None
    return kind, product


def _step_products(step: AnalysisStep) -> frozenset[Tuple[str, str]]:
    return frozenset(
        product
        for raw in (step.expected_outputs or [])
        if (product := _declared_product(raw)) is not None
    )


def _step_typed_inputs(step: AnalysisStep) -> frozenset[Tuple[str, str]]:
    return frozenset(
        product
        for raw in (step.inputs or [])
        if (product := _declared_product(raw)) is not None
    )


def _step_artifact_inputs(step: AnalysisStep) -> frozenset[str]:
    return frozenset(
        product for kind, product in _step_typed_inputs(step) if kind in _ARTIFACT_KINDS
    )


def _step_artifact_outputs(step: AnalysisStep) -> frozenset[str]:
    return frozenset(
        product for kind, product in _step_products(step) if kind in _ARTIFACT_KINDS
    )


def _product_tokens(
    products: frozenset[Tuple[str, str]],
) -> Tuple[Tuple[str, frozenset[str]], ...]:
    return tuple((kind, _tokens(product)) for kind, product in products)


def _is_window_manifest_product(product: Tuple[str, str]) -> bool:
    kind, name = product
    tokens = _tokens(name)
    return (
        kind in _ARTIFACT_KINDS
        and "manifest" in tokens
        and bool(tokens & {"trajectory", "window", "windows"})
    )


def _has_product_evidence(
    products: frozenset[Tuple[str, str]],
    *,
    required: frozenset[str],
    any_of: frozenset[str] = frozenset(),
    kinds: frozenset[str] = frozenset(),
) -> bool:
    for kind, tokens in _product_tokens(products):
        if kinds and kind not in kinds:
            continue
        if not required <= tokens:
            continue
        if any_of and not (tokens & any_of):
            continue
        return True
    return False


def _method_family_evidence(method: str) -> frozenset[str]:
    """Return bounded method-family evidence from whole normalized tokens.

    Method names remain agent-owned.  The contract recognizes general method
    families rather than an exact string allowlist, but never treats a raw
    substring (for example ``cluster`` inside ``clustered``) as ownership.
    Typed products are still required separately by :func:`_role_qualifies`.
    """

    tokens = _tokens(method)
    families: set[str] = set()
    clustering = bool(tokens & {"cluster", "clustering", "phenotyping", "kmeans"})
    clustering = clustering or {"k", "means"} <= tokens
    clustering = clustering or {"latent", "class"} <= tokens
    clustering = clustering or {"gaussian", "mixture"} <= tokens
    if "representation" in tokens or (
        "functional" in tokens and bool(tokens & {"feature", "features", "basis"})
    ):
        families.add("representation")
    if clustering:
        families.add("candidate_selection")
    if clustering and bool(
        tokens & {"stability", "consensus", "bootstrap", "resampling"}
    ):
        families.add("stability_freeze")
    if "characterization" in tokens or (
        "descriptive" in tokens
        and bool(tokens & {"profile", "profiles", "phenotype", "phenotypes"})
    ):
        families.add("characterization")
    return frozenset(families)


def _representation_product_evidence(
    products: frozenset[Tuple[str, str]],
) -> bool:
    if products & _REPRESENTATION_PRODUCTS:
        return True
    return _has_product_evidence(
        products,
        required=frozenset({"trajectory"}),
        any_of=frozenset({"representation", "feature", "features", "matrix"}),
        kinds=frozenset({"artifact", "dataset", "table"}),
    )


def _candidate_product_evidence(
    products: frozenset[Tuple[str, str]],
) -> bool:
    if products & _SELECTION_PRODUCTS and products & _CANDIDATE_SOLUTION_PRODUCTS:
        return True
    has_candidate_set = _has_product_evidence(
        products,
        required=frozenset({"candidate"}),
        any_of=frozenset({"model", "models", "fit", "fits", "assignments"}),
        kinds=frozenset({"artifact", "dataset", "model", "table"}),
    )
    has_comparison = _has_product_evidence(
        products,
        required=frozenset({"candidate"}),
        any_of=frozenset({"criterion", "criteria", "selection", "comparison"}),
        kinds=frozenset({"manifest", "statistic", "table"}),
    ) or _has_product_evidence(
        products,
        required=frozenset({"cluster", "selection"}),
        kinds=frozenset({"manifest", "statistic", "table"}),
    )
    return has_candidate_set and has_comparison


def _stability_product_evidence(
    products: frozenset[Tuple[str, str]],
) -> bool:
    has_stability = bool(products & _STABILITY_PRODUCTS) or _has_product_evidence(
        products,
        required=frozenset({"stability"}),
        any_of=frozenset({"cluster", "clustering", "freeze"}),
        kinds=frozenset({"artifact", "dataset", "manifest", "statistic", "table"}),
    )
    has_frozen_assignment = bool(products & _STABILITY_ASSIGNMENT_PRODUCTS) or (
        _has_product_evidence(
            products,
            required=frozenset({"cluster", "assignments"}),
            kinds=frozenset({"artifact", "dataset", "table"}),
        )
        and _has_product_evidence(
            products,
            required=frozenset({"stability"}),
            any_of=frozenset({"freeze", "frozen"}),
            kinds=frozenset({"artifact", "dataset", "manifest"}),
        )
    )
    return has_stability and has_frozen_assignment


def _characterization_product_evidence(
    products: frozenset[Tuple[str, str]],
) -> bool:
    if products & _CHARACTERIZATION_PRODUCTS:
        return True
    return _has_product_evidence(
        products,
        required=frozenset({"cluster"}),
        any_of=frozenset({"characteristic", "characteristics", "profile", "profiles"}),
        kinds=frozenset({"artifact", "dataset", "table"}),
    )


def _role_qualifies(
    role: str,
    *,
    method: str,
    products: frozenset[Tuple[str, str]],
) -> bool:
    method_families = _method_family_evidence(method)
    if role == "representation":
        return bool(
            method_families & {"representation", "candidate_selection"}
        ) and _representation_product_evidence(products)
    if role == "candidate_selection":
        return "candidate_selection" in method_families and _candidate_product_evidence(
            products
        )
    if role == "stability_freeze":
        return bool(
            method_families & {"stability_freeze", "candidate_selection"}
        ) and _stability_product_evidence(products)
    if role == "characterization":
        return bool(
            method_families & {"characterization", "candidate_selection"}
        ) and _characterization_product_evidence(products)
    raise ValueError(f"Unknown trajectory role: {role}")


def trajectory_step_roles(step: AnalysisStep) -> frozenset[str]:
    """Return scientific roles proven by method-family + typed products."""

    method = _method_head(step.method)
    products = _step_products(step)
    return frozenset(
        role
        for role in _ROLE_ORDER
        if _role_qualifies(role, method=method, products=products)
    )


def _trajectory_output_role(name: object) -> Optional[str]:
    """Classify only closed trajectory product names, never prose substrings."""

    tokens = _tokens(Path(str(name or "")).stem)
    if not tokens:
        return None
    if "trajectory" in tokens and bool(
        tokens & {"representation", "membership", "feature", "features"}
    ):
        return "representation"
    if "candidate" in tokens and bool(
        tokens & {"cluster", "clustering", "fit", "fits", "model", "models"}
    ):
        return "candidate_selection"
    if {"cluster", "selection"} <= tokens:
        return "candidate_selection"
    if "stability" in tokens and bool(
        tokens
        & {
            "adjusted",
            "ari",
            "bootstrap",
            "cluster",
            "clustering",
            "consensus",
            "rand",
            "resampling",
        }
    ):
        return "stability_freeze"
    if {"cluster", "assignments"} <= tokens or {
        "trajectory",
        "missingness",
        "policy",
    } <= tokens:
        return "stability_freeze"
    if "cluster" in tokens and bool(
        tokens
        & {
            "profile",
            "profiles",
            "characteristic",
            "characteristics",
            "outcome",
            "outcomes",
            "mortality",
            "sizes",
        }
    ):
        return "characterization"
    if {"trajectory", "profiles"} <= tokens or {"outcome", "cluster"} <= tokens:
        return "characterization"
    return None


def trajectory_role_scope_summary_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, object],
) -> List[ValidationFinding]:
    """Fail closed when one role writes another role's scientific products."""

    owned_roles = trajectory_step_roles(step)
    if not owned_roles:
        return []
    produced_by_role: Dict[str, set[str]] = defaultdict(set)

    def collect(node: object) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                role = _trajectory_output_role(raw_key)
                if role is not None:
                    produced_by_role[role].add(str(raw_key))
                collect(child)
        elif isinstance(node, (list, tuple, set)):
            for child in node:
                collect(child)
        elif isinstance(node, str):
            role = _trajectory_output_role(node)
            if role is not None:
                produced_by_role[role].add(node)

    for key in ("outputs", "output_files", "diagnostic_files"):
        container = step_summary.get(key)
        if container is not None:
            collect(container)
    unauthorized = {
        role: sorted(values)
        for role, values in sorted(produced_by_role.items())
        if role not in owned_roles
    }
    if not unauthorized:
        return []
    return [
        ValidationFinding(
            validator="trajectory_role_scope",
            severity="error",
            message=(
                f"Step {step.step_id} produced scientific products owned by a "
                "different trajectory DAG role. Keep representation, candidate "
                "selection, stability/freeze, and characterization boundaries explicit."
            ),
            detail={
                "kind": "trajectory_role_product_out_of_scope",
                "step_id": step.step_id,
                "owned_roles": sorted(owned_roles),
                "unauthorized_products_by_role": unauthorized,
            },
        )
    ]


def trajectory_role_result_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, object],
    out_dir: Path | None = None,
) -> List[ValidationFinding]:
    """Validate role-local replay metadata without choosing the science."""

    roles = trajectory_step_roles(step)
    findings: List[ValidationFinding] = []
    if out_dir is not None and "representation" in roles:
        findings.extend(
            _trajectory_representation_schema_findings(step=step, out_dir=out_dir)
        )
    if out_dir is not None and "candidate_selection" in roles:
        findings.extend(
            _trajectory_candidate_schema_findings(step=step, out_dir=out_dir)
        )
    if "candidate_selection" not in roles:
        return findings
    raw_selection = step_summary.get("cluster_selection")
    try:
        selection = ClusterSelectionManifest.model_validate(raw_selection)
    except Exception as exc:
        return [
            *findings,
            ValidationFinding(
                validator="trajectory_role_result",
                severity="error",
                message=(
                    f"Step {step.step_id} did not produce a valid typed cluster "
                    "selection with at least two finite candidates and one selected k."
                ),
                detail={
                    "kind": "trajectory_candidate_selection_invalid",
                    "step_id": step.step_id,
                    "validation_error": str(exc),
                },
            ),
        ]

    selected_value = next(
        item.criterion_value
        for item in selection.candidates
        if item.n_clusters == selection.selected_n_clusters
    )
    candidate_values = [item.criterion_value for item in selection.candidates]
    issues: List[str] = []
    if selection.selection_rule == "minimum" and not math.isclose(
        selected_value,
        min(candidate_values),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        issues.append("minimum rule did not select the finite minimum")
    if selection.selection_rule == "maximum" and not math.isclose(
        selected_value,
        max(candidate_values),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        issues.append("maximum rule did not select the finite maximum")
    for count_key in ("n_clusters", "cluster_count"):
        raw_count = step_summary.get(count_key)
        if raw_count is None:
            continue
        try:
            reported_count = int(raw_count)
        except (TypeError, ValueError):
            issues.append(f"{count_key} is not an integer")
            continue
        if reported_count != selection.selected_n_clusters:
            issues.append(f"{count_key} differs from selected_n_clusters")
    if not issues:
        return findings
    return [
        *findings,
        ValidationFinding(
            validator="trajectory_role_result",
            severity="error",
            message=(
                f"Step {step.step_id} cluster-selection metadata does not replay "
                "the agent-declared selection rule."
            ),
            detail={
                "kind": "trajectory_candidate_selection_replay_mismatch",
                "step_id": step.step_id,
                "issues": issues,
                "selection": selection.model_dump(mode="json"),
            },
        ),
    ]


def _read_role_manifest(path: Path) -> tuple[Mapping[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return None, "manifest root is not an object"
    return payload, None


def _role_schema_finding(
    *, step: AnalysisStep, kind: str, message: str, issues: Sequence[str]
) -> ValidationFinding:
    return ValidationFinding(
        validator="trajectory_role_result",
        severity="error",
        message=message,
        detail={"kind": kind, "step_id": step.step_id, "issues": list(issues)},
    )


def _trajectory_representation_schema_findings(
    *, step: AnalysisStep, out_dir: Path
) -> List[ValidationFinding]:
    path = out_dir / "trajectory_representation_schema.json"
    payload, read_error = _read_role_manifest(path)
    if payload is None:
        return [
            _role_schema_finding(
                step=step,
                kind="trajectory_representation_schema_invalid",
                message="The trajectory representation schema is absent or unreadable.",
                issues=[read_error or "unknown read error"],
            )
        ]
    issues: List[str] = []
    if payload.get("schema_version") != TRAJECTORY_REPRESENTATION_SCHEMA_VERSION:
        issues.append("schema_version is missing or unsupported")
    if payload.get("anchor_provenance") not in {"task_contract", "agent_declared"}:
        issues.append("anchor_provenance is not task_contract or agent_declared")
    trailing = payload.get("trailing_na_policy")
    required_trailing = {
        "zero_imputation": False,
        "eligibility_uses_observed_window_count": True,
        "profile_summaries_ignore_missing": True,
    }
    if not isinstance(trailing, Mapping) or any(
        trailing.get(key) is not value for key, value in required_trailing.items()
    ):
        issues.append("trailing_na_policy is not the structured missingness contract")
    scaling = payload.get("coordinate_scaling")
    required_scaling = {
        "method": "pooled_coordinate_wise_z_score",
        "ddof": 0,
        "observed_value_policy": "direct_or_owner_locf_available",
        "missing_value_policy": "preserve_missing_exclude_from_likelihood",
        "zero_variance_action": "fail_closed",
    }
    if not isinstance(scaling, Mapping) or any(
        scaling.get(key) != value for key, value in required_scaling.items()
    ):
        issues.append("coordinate_scaling is not the frozen z-score contract")
    evidence = payload.get("evidence_state_policy")
    required_evidence = {
        "direct_observed": "include",
        "owner_locf_available": "include_and_audit",
        "unavailable": "exclude",
        "additional_clustering_stage_imputation": "none",
    }
    if not isinstance(evidence, Mapping) or any(
        evidence.get(key) != value for key, value in required_evidence.items()
    ):
        issues.append("evidence_state_policy is not the owner-receipt contract")
    for field in (
        "id_column",
        "observation_family",
        "observation_columns",
        "profile_columns",
        "representation_columns",
        "frozen_population_n",
        "representation_sha256",
    ):
        if payload.get(field) in (None, "", []):
            issues.append(f"{field} is missing")
    if not issues:
        return []
    return [
        _role_schema_finding(
            step=step,
            kind="trajectory_representation_schema_incomplete",
            message=(
                "The representation step executed, but its typed schema cannot "
                "authorize a downstream stability calculation."
            ),
            issues=issues,
        )
    ]


def _trajectory_candidate_schema_findings(
    *, step: AnalysisStep, out_dir: Path
) -> List[ValidationFinding]:
    path = out_dir / "candidate_cluster_solution_schema.json"
    payload, read_error = _read_role_manifest(path)
    if payload is None:
        return [
            _role_schema_finding(
                step=step,
                kind="trajectory_candidate_schema_invalid",
                message="The candidate solution schema is absent or unreadable.",
                issues=[read_error or "unknown read error"],
            )
        ]
    issues: List[str] = []
    if payload.get("schema_version") != TRAJECTORY_CANDIDATE_SOLUTION_SCHEMA_VERSION:
        issues.append("schema_version is missing or unsupported")
    if _normalise_token(step.method) == OBSERVED_DATA_DIAG_GMM_METHOD:
        expected = {
            "model_family": OBSERVED_DATA_DIAG_GMM_MODEL_FAMILY,
            "fit_method": OBSERVED_DATA_DIAG_GMM_FIT_METHOD,
            "covariance_type": "diag",
        }
        for field, value in expected.items():
            if _normalise_token(payload.get(field)) != value:
                issues.append(
                    f"{field} does not match the declared observed-data method"
                )
    for field in (
        "id_column",
        "representation_columns",
        "selected_n_clusters",
        "selected_model_id",
        "assignment_column",
        "criterion",
        "selection_rule",
        "direction",
        "selected_criterion_value",
        "representation_schema_sha256",
        "candidate_assignments_sha256",
        "coordinate_scaling",
    ):
        if payload.get(field) in (None, "", []):
            issues.append(f"{field} is missing")
    if not issues:
        return []
    return [
        _role_schema_finding(
            step=step,
            kind="trajectory_candidate_schema_incomplete",
            message=(
                "The candidate-selection step executed, but its typed method/schema "
                "contract does not authorize the declared stability refit."
            ),
            issues=issues,
        )
    ]


@dataclass(frozen=True)
class TrajectoryRoleRequirement:
    """What a step must declare to own one trajectory role.

    ``_role_qualifies`` decides ownership from method-family tokens plus typed
    products, and every one of those sets is a literal in this module.  The
    refusal used to say only "missing one structured scientific role", which
    names the gap without naming anything the Planner can declare -- and none
    of the four role names appears anywhere in the Planner prompt, so the
    contract was enforced without ever being stated.  Rendering the sets here
    keeps the message and the predicate the same fact.
    """

    role: str
    method_tokens: tuple[str, ...]
    product_groups: tuple[frozenset[str], ...]

    def sentence(self) -> str:
        groups = "; and ".join(
            "one of " + ", ".join(sorted(group)) for group in self.product_groups
        )
        return (
            "An owner needs a method naming one of "
            f"{', '.join(self.method_tokens)} and expected_outputs containing "
            f"{groups}."
        )


def _spelled(products: frozenset[Tuple[str, str]]) -> frozenset[str]:
    return frozenset(f"{kind}:{name}" for kind, name in products)


def role_declaration_requirement(role: str) -> TrajectoryRoleRequirement:
    """The canonical, unambiguous way to own ``role``.

    Each role also has token-heuristic branches; those stay available but are
    not published, because a published set has to be one a Planner can copy
    exactly rather than a rule it has to re-derive.
    """

    if role == "representation":
        return TrajectoryRoleRequirement(
            role,
            ("representation", "functional basis/features"),
            (_spelled(_REPRESENTATION_PRODUCTS),),
        )
    if role == "candidate_selection":
        return TrajectoryRoleRequirement(
            role,
            ("cluster", "clustering", "phenotyping", "kmeans", "latent class"),
            (_spelled(_SELECTION_PRODUCTS), _spelled(_CANDIDATE_SOLUTION_PRODUCTS)),
        )
    if role == "stability_freeze":
        return TrajectoryRoleRequirement(
            role,
            ("cluster stability", "consensus", "bootstrap", "resampling"),
            (_spelled(_STABILITY_PRODUCTS), _spelled(_STABILITY_ASSIGNMENT_PRODUCTS)),
        )
    if role == "characterization":
        return TrajectoryRoleRequirement(
            role,
            ("characterization", "descriptive profile/phenotype"),
            (_spelled(_CHARACTERIZATION_PRODUCTS),),
        )
    raise ValueError(f"Unknown trajectory role: {role}")


def trajectory_plan_contract_applies(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    long_trajectory_bound: bool = False,
) -> bool:
    """Return the non-heuristic trigger for the run-level trajectory contract.

    Once the agent stamps a fixed-window plan as ``trajectory_clustering``, a
    missing or artifact-only role cannot make the contract disappear.  The
    evaluator will instead return explicit plan-contract errors.

    Trajectories reach this host in two representations.  ``ResearchContext``
    only ever carries the wide one, because ``fixed_window_trajectory`` is
    inferred by parsing a column name (``<family>_h<start>_<end>``).  The long
    one -- ``stay_id, charttime, concept, value_num`` -- is materialised,
    digested and bound as a typed run input, and has no column names to parse,
    so a run holding 19,067,154 verified trajectory rows still presented zero
    trajectory variables here and had its whole plan refused.  Callers that can
    see the bound tier say so with ``long_trajectory_bound``; the default keeps
    the wide-column behaviour exactly for callers that cannot.
    """

    if _normalise_token(plan.analysis_type) != "trajectory_clustering":
        return False
    if long_trajectory_bound:
        return True
    return trajectory_context_is_bound(context)


def trajectory_context_is_bound(context: ResearchContext) -> bool:
    """Return whether the context carries a typed fixed-window trajectory.

    This is the single context-side applicability boundary shared by prompts
    and result gates.  A trajectory may be materialized as a typed long input
    or represented by fixed-window wide variables; ordinary stay-level
    clustering has neither and must not inherit the trajectory role contract.
    """

    materialized_trajectory = getattr(
        getattr(context, "materialized_inputs", None), "trajectory", None
    )
    if materialized_trajectory is not None:
        return True
    return any(
        variable.fixed_window_trajectory is not None
        for variable in (context.variables or [])
    )


def _finding(kind: str, message: str, **detail: object) -> ValidationFinding:
    return ValidationFinding(
        validator="plan_contract",
        severity="error",
        message=message,
        detail={"kind": kind, **detail},
    )


def _cycle_nodes(
    step_ids: Sequence[str],
    edges: Sequence[Tuple[str, str, str]],
) -> List[str]:
    adjacency: Dict[str, set[str]] = {step_id: set() for step_id in step_ids}
    indegree: Dict[str, int] = {step_id: 0 for step_id in step_ids}
    for producer, consumer, _artifact in edges:
        if consumer not in adjacency[producer]:
            adjacency[producer].add(consumer)
            indegree[consumer] += 1
    queue = deque(step_id for step_id in step_ids if indegree[step_id] == 0)
    while queue:
        step_id = queue.popleft()
        for child in sorted(adjacency[step_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return sorted(step_id for step_id, degree in indegree.items() if degree > 0)


def evaluate_trajectory_plan_dag(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    long_trajectory_bound: bool = False,
) -> TrajectoryPlanDagEvaluation:
    """Validate the agent-declared trajectory plan without rewriting it.

    This re-asks the applicability question itself, so a caller that threaded
    ``long_trajectory_bound`` into its own guard but not into here still gets
    the wide-column answer.  H3 was refused a second time for exactly that: the
    outer guard had the flag, this call did not, and the run ended on the same
    "no validated fixed-window trajectory contract" it started with.
    """

    spec_steps = [
        step
        for step in (plan.steps or [])
        if step.trajectory_stability_spec is not None
    ]
    if not trajectory_plan_contract_applies(
        plan=plan,
        context=context,
        long_trajectory_bound=long_trajectory_bound,
    ):
        findings = (
            (
                _finding(
                    "trajectory_stability_spec_without_trajectory_plan",
                    "A trajectory stability spec is attached to a plan that has "
                    "no validated fixed-window trajectory contract.",
                    step_ids=[step.step_id for step in spec_steps],
                ),
            )
            if spec_steps
            else ()
        )
        return TrajectoryPlanDagEvaluation(bool(spec_steps), {}, {}, (), findings)

    findings: List[ValidationFinding] = []
    steps = list(plan.steps or [])
    step_index = {step.step_id: index for index, step in enumerate(steps)}
    products_by_step = {step.step_id: _step_products(step) for step in steps}
    artifact_inputs = {step.step_id: _step_artifact_inputs(step) for step in steps}
    artifact_outputs = {step.step_id: _step_artifact_outputs(step) for step in steps}

    role_candidates: Dict[str, List[str]] = {role: [] for role in _ROLE_ORDER}
    for step in steps:
        method = _method_head(step.method)
        products = products_by_step[step.step_id]
        has_figure = any(kind in _FIGURE_KINDS for kind, _ in products)
        for role in _ROLE_ORDER:
            qualifies = _role_qualifies(role, method=method, products=products)
            if not qualifies:
                continue
            if has_figure:
                findings.append(
                    _finding(
                        "trajectory_role_owner_has_figure_output",
                        "A trajectory scientific-role owner also declares a figure "
                        "product. Keep figure rendering in a separate non-owner step.",
                        role=role,
                        step_id=step.step_id,
                    )
                )
                continue
            role_candidates[role].append(step.step_id)

    role_owners: Dict[str, str] = {}
    for role in _ROLE_ORDER:
        candidates = role_candidates[role]
        if len(candidates) == 1:
            role_owners[role] = candidates[0]
        elif not candidates:
            requirement = role_declaration_requirement(role)
            findings.append(
                _finding(
                    "trajectory_role_missing",
                    "The trajectory plan declares no owner for the "
                    f"{role!r} role. {requirement.sentence()}",
                    role=role,
                    required_method_family_tokens=list(requirement.method_tokens),
                    qualifying_products=[
                        sorted(group) for group in requirement.product_groups
                    ],
                )
            )
        else:
            findings.append(
                _finding(
                    "trajectory_role_ambiguous",
                    "More than one plan step claims the same trajectory role.",
                    role=role,
                    candidate_step_ids=candidates,
                )
            )

    if len(spec_steps) > 1:
        findings.append(
            _finding(
                "trajectory_stability_spec_owner_ambiguous",
                "Only one dedicated stability owner may carry the standard "
                "trajectory stability spec.",
                step_ids=[step.step_id for step in spec_steps],
            )
        )
    for spec_step in spec_steps:
        local_roles = trajectory_step_roles(spec_step)
        local_method_head = _method_head(spec_step.method)
        local_inputs = {str(value).strip().lower() for value in spec_step.inputs}
        local_outputs = {
            str(value).strip().lower() for value in spec_step.expected_outputs
        }
        candidate_owner = role_owners.get("candidate_selection")
        closed_stability_contract = (
            local_method_head == TRAJECTORY_STABILITY_METHOD_HEAD
            and local_roles == frozenset({"stability_freeze"})
            and local_outputs == STABILITY_EXECUTOR_OUTPUTS
        )
        closed_stability_characterization_contract = (
            local_method_head == TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD
            and local_roles == frozenset({"stability_freeze", "characterization"})
            and local_outputs == STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS
        )
        if (
            not (
                closed_stability_contract
                or closed_stability_characterization_contract
            )
            or role_owners.get("stability_freeze") != spec_step.step_id
            or candidate_owner is None
            or candidate_owner == spec_step.step_id
            or step_index.get(candidate_owner, len(steps))
            >= step_index.get(spec_step.step_id, -1)
            or local_inputs != STABILITY_EXECUTOR_INPUTS
        ):
            findings.append(
                _finding(
                    "trajectory_stability_spec_contract_invalid",
                    "A planner-owned trajectory stability spec requires one later, "
                    "dedicated stability owner using method head "
                    f"{TRAJECTORY_STABILITY_METHOD_HEAD!r} (or the closed combined "
                    "stability-characterization method), the matching standard "
                    "executor input/output contract, and a distinct preceding "
                    "candidate owner.",
                    step_id=spec_step.step_id,
                    declared_method_head=local_method_head,
                    required_method_heads=[
                        TRAJECTORY_STABILITY_METHOD_HEAD,
                        TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD,
                    ],
                    roles=sorted(local_roles),
                    candidate_owner_step_id=candidate_owner,
                    declared_inputs=sorted(local_inputs),
                    required_inputs=sorted(STABILITY_EXECUTOR_INPUTS),
                    declared_outputs=sorted(local_outputs),
                    required_output_contracts=[
                        sorted(STABILITY_EXECUTOR_OUTPUTS),
                        sorted(STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS),
                    ],
                )
            )
    for step in steps:
        for kind, product in sorted(products_by_step[step.step_id]):
            if kind in _FIGURE_KINDS or _is_window_manifest_product((kind, product)):
                continue
            product_role = _trajectory_output_role(f"{kind}:{product}")
            expected_owner = role_owners.get(product_role or "")
            if expected_owner is None or expected_owner == step.step_id:
                continue
            findings.append(
                _finding(
                    "trajectory_role_product_owner_mismatch",
                    "A typed trajectory scientific product is declared outside "
                    "its unique role owner.",
                    role=product_role,
                    typed_product=f"{kind}:{product}",
                    expected_owner_step_id=expected_owner,
                    declared_owner_step_id=step.step_id,
                )
            )

    producer_candidates: Dict[str, List[str]] = defaultdict(list)
    for step in steps:
        for artifact in sorted(artifact_outputs[step.step_id]):
            producer_candidates[artifact].append(step.step_id)

    artifact_producers: Dict[str, str] = {}
    for artifact, producers in sorted(producer_candidates.items()):
        if len(producers) == 1:
            artifact_producers[artifact] = producers[0]
        else:
            findings.append(
                _finding(
                    "trajectory_artifact_producer_ambiguous",
                    "A plan artifact must have exactly one producer.",
                    artifact=artifact,
                    producer_step_ids=producers,
                )
            )

    typed_product_producers: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for step in steps:
        for product in sorted(products_by_step[step.step_id]):
            typed_product_producers[product].append(step.step_id)
    for (kind, product), producers in sorted(typed_product_producers.items()):
        if len(producers) <= 1:
            continue
        findings.append(
            _finding(
                "trajectory_typed_product_producer_ambiguous",
                "A typed trajectory scientific product must have exactly one "
                "plan owner.",
                typed_product=f"{kind}:{product}",
                producer_step_ids=producers,
            )
        )
    for consumer in steps:
        for kind, product in sorted(_step_typed_inputs(consumer)):
            producers = typed_product_producers.get((kind, product), [])
            if not producers:
                findings.append(
                    _finding(
                        "trajectory_typed_product_orphan",
                        "A consumed typed trajectory product has no declared "
                        "producer.",
                        typed_product=f"{kind}:{product}",
                        consumer_step_id=consumer.step_id,
                    )
                )
                continue
            if len(producers) != 1:
                continue
            producer = producers[0]
            if step_index[producer] >= step_index[consumer.step_id]:
                findings.append(
                    _finding(
                        "trajectory_typed_product_producer_not_preceding_consumer",
                        "A consumed typed trajectory product must come from one "
                        "earlier plan step.",
                        typed_product=f"{kind}:{product}",
                        producer_step_id=producer,
                        consumer_step_id=consumer.step_id,
                    )
                )

    edges: List[Tuple[str, str, str]] = []
    for consumer in steps:
        for artifact in sorted(artifact_inputs[consumer.step_id]):
            producers = producer_candidates.get(artifact, [])
            if not producers:
                findings.append(
                    _finding(
                        "trajectory_artifact_orphan",
                        "A consumed plan artifact has no declared producer.",
                        artifact=artifact,
                        consumer_step_id=consumer.step_id,
                    )
                )
                continue
            if len(producers) != 1:
                continue
            producer = producers[0]
            edges.append((producer, consumer.step_id, artifact))
            if step_index[producer] >= step_index[consumer.step_id]:
                findings.append(
                    _finding(
                        "trajectory_artifact_producer_not_preceding_consumer",
                        "A consumed artifact must come from one earlier plan step.",
                        artifact=artifact,
                        producer_step_id=producer,
                        consumer_step_id=consumer.step_id,
                    )
                )

    cycle_step_ids = _cycle_nodes(
        [step.step_id for step in steps],
        edges,
    )
    if cycle_step_ids:
        findings.append(
            _finding(
                "trajectory_artifact_cycle",
                "The trajectory artifact dependency graph contains a cycle.",
                cycle_step_ids=cycle_step_ids,
            )
        )

    for upstream_role, downstream_role in zip(_ROLE_ORDER, _ROLE_ORDER[1:]):
        upstream = role_owners.get(upstream_role)
        downstream = role_owners.get(downstream_role)
        if upstream is None or downstream is None or upstream == downstream:
            continue
        shared = sorted(artifact_outputs[upstream] & artifact_inputs[downstream])
        if not shared:
            findings.append(
                _finding(
                    "trajectory_role_edge_missing",
                    "Adjacent trajectory roles are not connected by a declared "
                    "artifact edge.",
                    upstream_role=upstream_role,
                    upstream_step_id=upstream,
                    downstream_role=downstream_role,
                    downstream_step_id=downstream,
                )
            )

    representation_owner = role_owners.get("representation")
    if representation_owner is not None:
        variables_by_name = {
            variable.name: variable
            for variable in (context.variables or [])
            if variable.fixed_window_trajectory is not None
        }
        representation_step = steps[step_index[representation_owner]]
        window_source_step = representation_step
        selected_by_family: Dict[str, List[Tuple[float, float, str]]] = defaultdict(
            list
        )
        direct_window_inputs = [
            str(input_name)
            for input_name in (representation_step.inputs or [])
            if str(input_name) in variables_by_name
        ]
        direct_family_counts: Dict[str, int] = defaultdict(int)
        for input_name in direct_window_inputs:
            metadata = variables_by_name[input_name].fixed_window_trajectory
            assert metadata is not None
            direct_family_counts[metadata.family] += 1
        if not any(count >= 2 for count in direct_family_counts.values()):
            typed_inputs = _step_typed_inputs(representation_step)
            manifest_inputs = sorted(
                product
                for product in typed_inputs
                if _is_window_manifest_product(product)
            )
            manifest_producers = sorted(
                {
                    producers[0]
                    for _kind, manifest in manifest_inputs
                    for producers in (producer_candidates.get(manifest, []),)
                    if len(producers) == 1
                }
            )
            # This branch presumes one of two topologies: the representation
            # reads wide window columns directly, or it reads a panel someone
            # else built and therefore must consume that producer's manifest.
            # A LONG-bound run has a third: the representation reads the bound
            # long trajectory (stay_id / charttime / concept / value_num) and
            # emits the window manifest ITSELF, because it is the step that
            # chooses the windows. Requiring it to consume a manifest from an
            # upstream producer asks it to import provenance it is the source
            # of -- the same wide-topology assumption as the window-family rule
            # below, one rule over.
            representation_emits_the_manifest = long_trajectory_bound and any(
                _is_window_manifest_product(product)
                for product in _step_products(representation_step)
            )
            if representation_emits_the_manifest:
                # It IS the window source: it reads the bound long trajectory
                # and declares the windows it chose. There is no upstream
                # producer to resolve, and no panel lineage to import.
                pass
            elif not manifest_inputs:
                findings.append(
                    _finding(
                        "trajectory_window_manifest_missing",
                        "A representation built from an upstream panel must consume "
                        "a typed trajectory-window manifest from that panel's producer.",
                        step_id=representation_owner,
                        long_trajectory_bound=long_trajectory_bound,
                    )
                )
            elif len(manifest_producers) != 1:
                findings.append(
                    _finding(
                        "trajectory_window_manifest_producer_unresolved",
                        "The consumed trajectory-window manifest must resolve to one "
                        "upstream producer.",
                        step_id=representation_owner,
                        manifest_inputs=[
                            f"{kind}:{name}" for kind, name in manifest_inputs
                        ],
                        producer_step_ids=manifest_producers,
                    )
                )
            else:
                producer_id = manifest_producers[0]
                consumed_from_producer = {
                    name
                    for kind, name in typed_inputs
                    if kind in _ARTIFACT_KINDS
                    and name in artifact_outputs[producer_id]
                    and not _is_window_manifest_product((kind, name))
                }
                if not consumed_from_producer:
                    findings.append(
                        _finding(
                            "trajectory_window_panel_lineage_missing",
                            "The window manifest and the upstream trajectory panel "
                            "must share one declared producer and both be consumed by "
                            "the representation owner.",
                            step_id=representation_owner,
                            manifest_producer_step_id=producer_id,
                        )
                    )
                else:
                    window_source_step = steps[step_index[producer_id]]

        for input_name in window_source_step.inputs or []:
            variable = variables_by_name.get(str(input_name))
            if variable is None or variable.fixed_window_trajectory is None:
                continue
            metadata = variable.fixed_window_trajectory
            selected_by_family[metadata.family].append(
                (
                    float(metadata.window_start_hours),
                    float(metadata.window_end_hours),
                    variable.name,
                )
            )
        multi_window_families = {
            family: sorted(windows)
            for family, windows in selected_by_family.items()
            if len(windows) >= 2
        }
        # The ONLY way this rule can be satisfied is a declared input whose
        # variable carries ``fixed_window_trajectory`` -- metadata inferred by
        # parsing a wide column name (``<family>_h<start>_<end>``).  A run whose
        # trajectory is bound in the LONG tier has no such column to parse:
        # this module's own applicability docstring records that a run holding
        # 19,067,154 verified trajectory rows "still presented zero trajectory
        # variables here".  So for a long-bound run the rule applies and cannot
        # be satisfied by any plan -- h3 has never passed step 01 in any
        # recorded run, and this is why.
        #
        # This is the THIRD time the tier flag was threaded to one decision in
        # this file and not another; the docstring above already records two.
        # The waiver is narrow: the plan must still declare the window manifest,
        # which is where a long-bound run's windows are recorded and where they
        # are verified after the representation step actually runs. A long-bound
        # plan that declares no manifest still fails, one line below.
        long_tier_defers_windows_to_the_manifest = long_trajectory_bound and any(
            _is_window_manifest_product(product)
            for step in (representation_step, window_source_step)
            for product in (
                *_step_typed_inputs(step),
                *_step_products(step),
            )
        )
        if not multi_window_families and not long_tier_defers_windows_to_the_manifest:
            findings.append(
                _finding(
                    "trajectory_window_family_not_resolved",
                    "The representation owner does not select at least two fixed "
                    "windows from one declared family, either directly or through "
                    "a typed upstream panel manifest.",
                    step_id=representation_owner,
                    window_source_step_id=window_source_step.step_id,
                    selected_families=sorted(selected_by_family),
                    long_trajectory_bound=long_trajectory_bound,
                )
            )
        available_by_family: Dict[str, List[Tuple[float, float, str]]] = defaultdict(
            list
        )
        for variable in variables_by_name.values():
            metadata = variable.fixed_window_trajectory
            assert metadata is not None
            available_by_family[metadata.family].append(
                (
                    float(metadata.window_start_hours),
                    float(metadata.window_end_hours),
                    variable.name,
                )
            )
        for family, windows in sorted(multi_window_families.items()):
            selected_bins = {(start, end) for start, end, _name in windows}
            horizon_start = min(start for start, _end, _name in windows)
            horizon_end = max(end for _start, end, _name in windows)
            omitted_internal = sorted(
                name
                for start, end, name in available_by_family.get(family, [])
                if start >= horizon_start - 1e-9
                and end <= horizon_end + 1e-9
                and (start, end) not in selected_bins
            )
            if omitted_internal:
                findings.append(
                    _finding(
                        "trajectory_internal_window_gap",
                        "The representation silently omits available fixed-window "
                        "bins inside one selected family horizon.",
                        step_id=representation_owner,
                        window_source_step_id=window_source_step.step_id,
                        family=family,
                        horizon_start_hours=horizon_start,
                        horizon_end_hours=horizon_end,
                        omitted_columns=omitted_internal,
                    )
                )

    return TrajectoryPlanDagEvaluation(
        applies=True,
        role_owners=dict(role_owners),
        artifact_producers=dict(artifact_producers),
        artifact_edges=tuple(sorted(set(edges))),
        findings=tuple(findings),
    )


def trajectory_plan_dag_findings(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    long_trajectory_bound: bool = False,
) -> List[ValidationFinding]:
    """Return plan-contract errors for a stamped fixed-window trajectory DAG."""

    return list(
        evaluate_trajectory_plan_dag(
            plan=plan,
            context=context,
            long_trajectory_bound=long_trajectory_bound,
        ).findings
    )


def _normalise_redundant_split_role_outputs(
    *,
    plan: AnalysisPlan,
) -> tuple[AnalysisPlan, Dict[str, List[str]], Dict[str, List[str]]]:
    """Drop only redundant products misplaced on a dedicated stability owner.

    Older saved plans sometimes asked the stability step to repeat a cluster-
    count selection table or cluster-size table even though separate candidate
    and characterization owners already existed.  Those repeats are schema
    aliases of canonical products, not independent scientific decisions.  The
    migration is deliberately narrow: it acts only when all three dedicated
    owners are independently proven by their method family and non-redundant
    typed products.  Ambiguous or monolithic plans remain untouched and fail
    through the ordinary DAG audit.
    """

    anchor_products = {
        "candidate_selection": _REDUNDANT_SPLIT_ROLE_OUTPUTS["candidate_selection"],
        "stability_freeze": frozenset(),
        "characterization": _REDUNDANT_SPLIT_ROLE_OUTPUTS["characterization"],
    }
    candidates: Dict[str, List[str]] = {role: [] for role in anchor_products}
    for step in plan.steps or []:
        method = _method_head(step.method)
        products = _step_products(step)
        for role, ignored in anchor_products.items():
            if _role_qualifies(role, method=method, products=products - ignored):
                candidates[role].append(step.step_id)
    if any(len(step_ids) != 1 for step_ids in candidates.values()):
        return plan, {}, {}

    owners = {role: step_ids[0] for role, step_ids in candidates.items()}
    stability_owner = owners["stability_freeze"]
    if stability_owner in {
        owners["candidate_selection"],
        owners["characterization"],
    }:
        return plan, {}, {}

    removals: Dict[str, List[str]] = defaultdict(list)
    revised_steps: List[AnalysisStep] = []
    for step in plan.steps or []:
        if step.step_id != stability_owner:
            revised_steps.append(step)
            continue
        kept: List[str] = []
        for raw_output in step.expected_outputs or []:
            product = _declared_product(raw_output)
            target_role = next(
                (
                    role
                    for role, redundant in _REDUNDANT_SPLIT_ROLE_OUTPUTS.items()
                    if product in redundant
                ),
                None,
            )
            if target_role is None or owners[target_role] == stability_owner:
                kept.append(raw_output)
                continue
            removals[step.step_id].append(str(raw_output))
        revised_steps.append(
            step.model_copy(update={"expected_outputs": kept})
            if removals.get(step.step_id)
            else step
        )
    if not removals:
        return plan, {}, {}

    input_removals: Dict[str, List[str]] = defaultdict(list)
    characterization_owner = owners["characterization"]
    final_steps: List[AnalysisStep] = []
    removed_cluster_sizes = "table:cluster_sizes" in removals.get(stability_owner, [])
    removed_cluster_selection = "table:cluster_number_selection" in removals.get(
        stability_owner, []
    )
    for step in revised_steps:
        kept_inputs: List[str] = []
        inputs_changed = False
        for raw_input in step.inputs or []:
            product = _declared_product(raw_input)
            if (
                step.step_id == characterization_owner
                and removed_cluster_sizes
                and product == ("table", "cluster_sizes")
            ):
                input_removals[step.step_id].append(str(raw_input))
                inputs_changed = True
            elif removed_cluster_selection and product == (
                "table",
                "cluster_number_selection",
            ):
                input_removals[step.step_id].append(str(raw_input))
                if "manifest:cluster_selection" not in kept_inputs and (
                    "manifest:cluster_selection" not in (step.inputs or [])
                ):
                    kept_inputs.append("manifest:cluster_selection")
                inputs_changed = True
            else:
                kept_inputs.append(raw_input)
        final_steps.append(
            step.model_copy(update={"inputs": kept_inputs}) if inputs_changed else step
        )
    return (
        plan.model_copy(update={"steps": final_steps}),
        dict(removals),
        dict(input_removals),
    )


def augment_trajectory_plan_products(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Add only canonical replay products to the agent's existing role owners.

    This is schema normalization, not scientific planning: step identities,
    methods, order, and scientific choices remain unchanged.  A split
    stability owner is bound to the candidate owner's canonical selection
    manifest, and narrowly recognized redundant cross-role aliases are removed.
    Missing or ambiguous roles are left for
    :func:`trajectory_plan_dag_findings` to block rather than being invented by
    the framework.
    """

    normalized, removed, removed_inputs = _normalise_redundant_split_role_outputs(
        plan=plan
    )
    normalization_findings: List[ValidationFinding] = []
    if removed or removed_inputs:
        normalization_findings.append(
            ValidationFinding(
                validator="plan_contract",
                severity="info",
                message=(
                    "Removed redundant selection/characterization aliases from "
                    "a dedicated trajectory stability owner."
                ),
                detail={
                    "kind": "trajectory_redundant_split_role_outputs_removed",
                    "removed_outputs_by_step": removed,
                    "removed_inputs_by_step": removed_inputs,
                    "preserved_step_ids": [step.step_id for step in plan.steps],
                },
            )
        )

    evaluation = evaluate_trajectory_plan_dag(plan=normalized, context=context)
    if not evaluation.applies or set(evaluation.role_owners) != set(_ROLE_ORDER):
        if normalized == plan:
            return plan, normalization_findings
        return (
            normalized.model_copy(update={"revision": max(1, int(plan.revision)) + 1}),
            normalization_findings,
        )

    additions: Dict[str, List[str]] = defaultdict(list)
    input_additions: Dict[str, List[str]] = defaultdict(list)
    for role, outputs in _ROLE_CANONICAL_OUTPUTS.items():
        owner = evaluation.role_owners[role]
        step = next(item for item in normalized.steps if item.step_id == owner)
        for output in outputs:
            if output not in (step.expected_outputs or []):
                additions[owner].append(output)

    characterization_owner = evaluation.role_owners["characterization"]
    characterization = next(
        item for item in normalized.steps if item.step_id == characterization_owner
    )
    declared_products = {
        product
        for output in characterization.expected_outputs or []
        if (parsed := _declared_product(output)) is not None
        for product in (parsed[1],)
    }
    if (
        declared_products & _CHARACTERIZATION_OUTCOME_PRODUCTS
        and "table:outcome_by_cluster" not in (characterization.expected_outputs or [])
    ):
        additions[characterization_owner].append("table:outcome_by_cluster")

    candidate_owner = evaluation.role_owners["candidate_selection"]
    stability_owner = evaluation.role_owners["stability_freeze"]
    stability = next(
        item for item in normalized.steps if item.step_id == stability_owner
    )
    if stability.trajectory_stability_spec is not None:
        for output in sorted(STABILITY_EXECUTOR_OUTPUTS):
            if output not in (stability.expected_outputs or []):
                additions[stability_owner].append(output)
    if (
        candidate_owner != stability_owner
        and stability.trajectory_stability_spec is None
        and "manifest:cluster_selection" not in (stability.inputs or [])
    ):
        input_additions[stability_owner].append("manifest:cluster_selection")

    representation_owner = evaluation.role_owners["representation"]
    candidate = next(
        item for item in normalized.steps if item.step_id == candidate_owner
    )
    if representation_owner != candidate_owner and (
        "manifest:trajectory_representation_schema" not in (candidate.inputs or [])
    ):
        input_additions[candidate_owner].append(
            "manifest:trajectory_representation_schema"
        )
    if representation_owner != stability_owner and (
        "manifest:trajectory_representation_schema" not in (stability.inputs or [])
    ):
        input_additions[stability_owner].append(
            "manifest:trajectory_representation_schema"
        )
    if candidate_owner != stability_owner and (
        "manifest:candidate_cluster_solution_schema" not in (stability.inputs or [])
    ):
        input_additions[stability_owner].append(
            "manifest:candidate_cluster_solution_schema"
        )

    if not additions and not input_additions and normalized == plan:
        return plan, normalization_findings
    revised_steps = [
        (
            step.model_copy(
                update={
                    "expected_outputs": [
                        *(step.expected_outputs or []),
                        *additions.get(step.step_id, []),
                    ],
                    "inputs": [
                        *(step.inputs or []),
                        *input_additions.get(step.step_id, []),
                    ],
                }
            )
            if step.step_id in additions or step.step_id in input_additions
            else step
        )
        for step in normalized.steps
    ]
    revised = normalized.model_copy(
        update={
            "steps": revised_steps,
            "revision": max(1, int(plan.revision)) + 1,
        }
    )
    augmentation_findings = list(normalization_findings)
    if additions or input_additions:
        augmentation_findings.append(
            ValidationFinding(
                validator="plan_contract",
                severity="info",
                message=(
                    "Added canonical replay products to the existing agent-declared "
                    "trajectory DAG roles without changing scientific ownership."
                ),
                detail={
                    "kind": "trajectory_canonical_products_added",
                    "added_outputs_by_step": {
                        step_id: list(outputs)
                        for step_id, outputs in sorted(additions.items())
                    },
                    "added_inputs_by_step": {
                        step_id: list(inputs)
                        for step_id, inputs in sorted(input_additions.items())
                    },
                    "preserved_step_ids": [step.step_id for step in plan.steps],
                },
            )
        )
    return revised, augmentation_findings


def trajectory_planner_contract_guide(
    *,
    context: ResearchContext,
    analysis_type: object,
    long_trajectory_bound: bool = False,
) -> str:
    """Return the case-neutral planning schema for trajectory DAGs.

    ``long_trajectory_bound`` mirrors the parameter
    :func:`trajectory_plan_contract_applies` already takes, and for the same
    reason: a trajectory reaches this host in two representations, and only
    the wide one leaves ``fixed_window_trajectory`` on a variable to find.
    The gate learned that; this guide did not, so a run holding a bound long
    trajectory was judged against a contract it was never shown -- the
    Planner wrote one combined clustering-and-stability step, the gate
    demanded two typed owners, and the whole plan was refused before any
    step ran. The default keeps the wide-column behaviour byte-identical
    for callers that cannot see the bound tier.
    """

    if _normalise_token(analysis_type) != "trajectory_clustering" or not (
        long_trajectory_bound
        or any(
            variable.fixed_window_trajectory is not None
            for variable in (context.variables or [])
        )
    ):
        return ""
    return (
        "FIXED-WINDOW TRAJECTORY PLAN CONTRACT (scientific choices remain yours):\n"
        "Declare four typed roles: representation, candidate selection, "
        "stability/freeze, and descriptive characterization. They may be four "
        "steps, one monolithic scientific step, or a hybrid; in particular one "
        "scientific step may own both candidate selection and stability/freeze. "
        "Never mix a scientific role with figure outputs. Methods may use any "
        "accurate method-family name, but role products must use this canonical "
        "replay schema:\n"
        "- representation: declare the chosen representation artifact, "
        "`table:trajectory_membership`, and "
        "`manifest:trajectory_representation_schema`;\n"
        "- candidate selection: declare a candidate fit/assignment artifact and "
        "both `manifest:cluster_selection` and "
        "`manifest:candidate_cluster_solution_schema`;\n"
        "- stability/freeze: declare the frozen-solution assignment artifact, "
        "`manifest:trajectory_missingness_policy`, `table:cluster_assignments`, "
        "`table:cluster_stability`, and "
        "`table:cluster_stability_assignments`;\n"
        "- characterization: declare `table:trajectory_profiles` and "
        "`table:cluster_sizes`, plus `table:outcome_by_cluster` only when an "
        "outcome description is planned.\n"
        "Each typed scientific product has one role owner. A separate stability "
        "owner must consume `manifest:cluster_selection` and the candidate "
        "fit/assignment artifacts; it must not repeat selection tables, cluster "
        "sizes, profiles, outcomes, or figures. Connect separate owners through "
        "explicit typed producer/consumer edges. "
        + (
            # The wide tier only. A long-bound run has no `<family>_h<start>_<end>`
            # column to list, so telling it to list them describes work it cannot
            # do -- and the contract then refuses it for not having done it.
            (
                "If representation reads raw fixed-window columns directly, list "
                "them in its inputs. If it instead reads an upstream aligned "
                "panel, the panel producer must list the raw fixed-window columns "
                "in its inputs, produce both the panel and "
                "`manifest:trajectory_window_manifest`, and the representation "
                "owner must consume both. "
            )
            if not long_trajectory_bound
            else (
                # MEASURED: rendered for a long-bound run this guide mentioned the
                # long tier ZERO times -- no charttime, no value_num, no stay_id --
                # while the contract refused the plan for windows only a wide run
                # can declare. h3 never passed step 01 in any recorded run. One
                # recorded plan happened to declare the manifest and one did not;
                # neither was ever told to.
                "This run's trajectory is bound as the LONG representation, not "
                "as wide columns: one row per stay per time per concept, with "
                "the observation time on each row. There are no "
                "`<family>_h<start>_<end>` columns to list, so do NOT plan "
                "around them. The host hands that table to the step's CODE at "
                "runtime, the same way it hands over the cohort -- it is NOT a "
                "listable step input and has no name in the executable roster, "
                "so do NOT put it in `inputs` and do not invent a name for it. "
                "Plan the representation owner to derive at least two fixed "
                "windows from one concept family out of that table, and it MUST "
                "declare `manifest:trajectory_window_manifest` among its own "
                "expected_outputs -- it is the window source, so there is no "
                "upstream producer to consume one from. "
            )
        )
        + "The manifest records ordered "
        "source columns with family and window boundaries; it is provenance, not "
        "permission for the framework to choose a horizon, method, k, threshold, "
        "or missing-data policy.\n"
        "A dedicated stability owner may opt into the standard observed-data "
        "diagonal-GMM stability calculator by carrying a complete "
        "trajectory_stability_spec. In that case its method head must be exactly "
        f"{TRAJECTORY_STABILITY_METHOD_HEAD}; its inputs must be exactly "
        "artifact:trajectory_representation, "
        "artifact:candidate_cluster_assignments, "
        "manifest:cluster_selection, "
        "manifest:trajectory_representation_schema, and "
        "manifest:candidate_cluster_solution_schema. Its outputs must be exactly "
        "artifact:stability_freeze, artifact:cluster_assignments, "
        "manifest:cluster_stability_spec, "
        "manifest:trajectory_missingness_policy, table:cluster_assignments, "
        "table:cluster_stability, table:cluster_stability_assignments, and "
        "table:cluster_assignment_provenance. The spec is your "
        "scientific decision packet, and it asks you only for the decisions "
        "that are yours: n_resamples, and exactly one of sample_fraction (<1) "
        "or sample_size. Add minimum_mean_stability only when mean stability "
        "must gate freezing; leaving it out reports stability without making a "
        "binary accept/reject claim, and a failed threshold requests a new "
        "planner revision and never makes the calculator choose a different k. "
        "You may also override the recorded defaults for base_seed, "
        "refit_max_iter, refit_tolerance and refit_regularization. Declare "
        "nothing else in the spec. The v1 calculator has exactly one "
        "resampling scheme, one comparison metric, one label-alignment rule "
        "and one refit engine, so those fields have exactly one legal value; "
        "the host fills them in and records them in the spec and its digest. "
        "Retyping them can only misspell them, and any field name outside the "
        "spec is rejected rather than ignored. Leave "
        "trajectory_stability_spec null for other fit families or any monolithic "
        "candidate+stability step; those remain agent-coded and fail closed if "
        "unsupported."
    )


def non_trajectory_clustering_stability_guide(
    *,
    context: ResearchContext,
    analysis_type: object,
    long_trajectory_bound: bool = False,
) -> str:
    """Tell a group-discovery study without a trajectory how to declare stability.

    The converse of :func:`trajectory_planner_contract_guide`, and gated on the
    same predicate so exactly one of the two ever speaks.

    A sepsis-subphenotyping fixture asks to cluster first-24h summaries -- one
    row per stay, no trajectory in either representation -- and requires a
    cluster-stability audit.  ``trajectory_stability_spec`` is the only typed
    stability field a Planner can see, so it declared that, and the plan was
    refused for attaching a stability spec with no validated fixed-window
    trajectory contract behind it.  The study was asked for something it had no
    legal way to declare.

    The method registry already separates the two: ``cluster_stability`` is
    ``llm_coded`` with no runner, and the trajectory entry's own notes say
    "general cluster stability remains agent-coded".  That answer simply never
    reached the Planner.
    """

    if _normalise_token(analysis_type) != "trajectory_clustering":
        return ""
    if long_trajectory_bound or any(
        variable.fixed_window_trajectory is not None
        for variable in (context.variables or [])
    ):
        # The fixed-window contract applies; the other guide speaks instead.
        return ""
    return (
        "CLUSTER STABILITY WITHOUT A FIXED-WINDOW TRAJECTORY:\n"
        "This run has no fixed-window trajectory in either representation, so "
        "the typed trajectory stability calculator does not apply to it. Leave "
        "`trajectory_stability_spec` null on every step: attaching it without a "
        "validated fixed-window trajectory contract is refused, and that "
        "refusal stops the whole plan. Declare the stability work as an "
        f"ordinary analysis step with method `{_GENERAL_CLUSTER_STABILITY_METHOD}` "
        "(bootstrap / consensus / adjusted Rand). It is agent-coded, so state "
        "the resampling scheme, the agreement metric and the decision rule in "
        "the step intent rather than in a typed spec."
    )


def trajectory_role_code_contract(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    applies: bool = True,
) -> str:
    """Return role-local schemas for an agent-decomposed trajectory DAG."""

    if not applies:
        return ""

    # The SCHEMA is still selected from typed products, never task prose. What
    # the context supplies here is not a schema choice but a VOCABULARY: the
    # exact concept ids materialized in the bound trajectory.
    #
    # verify42 is why. The bound table held sofa2, sofa2_resp, sofa2_coag,
    # sofa2_liver, sofa2_cardio, sofa2_cns, sofa2_renal and lact -- all eight
    # present, none unavailable -- and the script queried sofa_resp, sofa_coag,
    # sofa_liver: every name one character off, built from the question's phrase
    # "SOFA components and lactate". Telling it to read the column helped
    # (verify43 began using sofa2) but left it still assembling names. Naming
    # them removes the guess.
    materialized_concepts: Tuple[str, ...] = ()
    trajectory_input = getattr(
        getattr(context, "materialized_inputs", None), "trajectory", None
    )
    if trajectory_input is not None:
        materialized_concepts = tuple(
            str(item)
            for item in (getattr(trajectory_input, "materialized_concepts", None) or ())
            if str(item).strip()
        )
    products = _step_products(step)
    declarations = {product for _kind, product in products}
    sections: List[str] = []
    window_manifests = sorted(
        product for product in products if _is_window_manifest_product(product)
    )
    if window_manifests:
        manifest_names = ", ".join(f"{name}.json" for _kind, name in window_manifests)
        sections.append(
            "UPSTREAM WINDOW-PANEL ROLE: write the declared manifest JSON "
            f"({manifest_names}) with panel_product and families; each family "
            "entry must contain family, ordered_source_columns, and for every "
            "source column its exact name, window_start_hours, and "
            "window_end_hours. Derive these fields from the declared source "
            "inputs. Do not silently omit an internal available bin."
        )
    if materialized_concepts and (
        window_manifests or "trajectory_membership" in declarations
    ):
        sections.append(
            "BOUND TRAJECTORY VOCABULARY: the long trajectory input materializes "
            "exactly these concept ids, and its `concept` column contains these "
            f"values verbatim: {', '.join(sorted(materialized_concepts))}. Select "
            "the families your role needs from THIS list by exact string; do not "
            "derive a concept id from the research question's wording, and do not "
            "report a family as absent without checking this list first."
        )
    if "trajectory_membership" in declarations:
        sections.append(
            "REPRESENTATION ROLE: write trajectory_membership.csv with the "
            "agent-selected id column plus observed_window_count, "
            "meets_min_observed_windows, included_in_clustering, and "
            "exclusion_reason. In step_summary.json also declare "
            "observation_family, observation_columns (in model order), "
            "min_observed_windows, profile_columns, "
            "profile_summary_statistic (mean or median), time_axis='relative_hours', "
            "anchor, anchor_provenance, anchor_source, and trailing_na_policy. "
            "Also write trajectory_representation_schema.json with "
            "schema_version='easyicu.trajectory_representation_schema/2' and those exact "
            "agent-chosen fields plus id_column, representation_columns in model "
            "order, frozen_population_n, and representation_sha256 computed from "
            "the exact representation artifact bytes. This typed manifest is the downstream "
            "authority for representation semantics; do not make later roles infer "
            "them from filenames, column substrings, or prose. Set "
            "anchor_provenance to task_contract or agent_declared, and encode "
            "trailing_na_policy as the object {zero_imputation:false, "
            "eligibility_uses_observed_window_count:true, "
            "profile_summaries_ignore_missing:true}, never as prose. "
            "Declare coordinate_scaling as {method:'pooled_coordinate_wise_z_score', "
            "ddof:0, observed_value_policy:'direct_or_owner_locf_available', "
            "missing_value_policy:'preserve_missing_exclude_from_likelihood', "
            "zero_variance_action:'fail_closed'} and evidence_state_policy as "
            "{direct_observed:'include', owner_locf_available:'include_and_audit', "
            "unavailable:'exclude', additional_clustering_stage_imputation:'none'}. "
            "Do not impute unobserved trajectory cells with zero. This role only "
            "builds the representation: do not fit clusters, select k, freeze "
            "assignments, characterize profiles, or analyze outcomes here."
        )
    if "cluster_selection" in declarations:
        owned_roles = trajectory_step_roles(step)
        role_boundary = ""
        if "stability_freeze" not in owned_roles:
            role_boundary += (
                " This owner does not own stability/freeze: do not run bootstrap, "
                "resampling, consensus, or stability refits, and do not write "
                "stability products."
            )
        if "characterization" not in owned_roles:
            role_boundary += (
                " This owner does not own characterization: do not write cluster "
                "profiles, cluster characteristics, cluster sizes, or outcomes."
            )
        sections.append(
            "CANDIDATE-SELECTION ROLE: consume the declared upstream trajectory "
            "representation artifact as the clustering model matrix; do not "
            "reconstruct trajectory features from COHORT_PARQUET (the locked "
            "cohort may be used only for identifier reconciliation or audits). "
            "Treat the representation rows as the upstream owner's already-frozen "
            "eligible population; do not reapply cohort, anchor, or observed-window "
            "eligibility unless an explicit membership artifact is also a declared "
            "input. When an upstream scaling summary names "
            "scaled_representation_column, use those exact named columns as the "
            "model coordinates; do not concatenate missingness indicators, rank "
            "intermediates, scaled coordinates, and profile summaries into duplicate "
            "representations. "
            "Write cluster_selection.json using the "
            "typed candidate schema: criterion, selection_rule (minimum or "
            "maximum), direction (minimize or maximize), "
            "selected_n_clusters, at least two finite candidates with "
            "n_clusters and criterion_value, and rationale. Repeat the exact "
            "object as step_summary.cluster_selection and report n_clusters "
            "and clustering_method. Candidate fit/assignment artifacts used by a "
            "separate stability owner must also preserve the selected method "
            "family, exact representation_columns, selected_n_clusters (or the "
            "full cluster_selection object), id_column, and candidate assignment "
            "labels. cluster_selection.json is a closed selection-only manifest: "
            "write only criterion, selection_rule, direction, selected_n_clusters, "
            "candidates, and rationale. Do not add role, id_column, "
            "clustering_method, model_family, fit_method, or selected_model_id to "
            "that manifest. Give every candidate model record a stable model_id; "
            "the exact identifier, method, and selected-model metadata belong in "
            "the candidate solution schema and candidate model metadata so a "
            "downstream stability owner never has to guess them. "
            "Consume trajectory_representation_schema.json and write "
            "candidate_cluster_solution_schema.json with "
            "schema_version='easyicu.candidate_cluster_solution_schema/2', its exact id_column and "
            "representation_columns plus clustering_method/model_family, "
            "fit_method, covariance_type, selected_n_clusters, selected_model_id, "
            "assignment_column, criterion, selection_rule, direction, and selected "
            "criterion value. For method "
            "observed_data_diagonal_gaussian_mixture_candidate_selection, the "
            "implementation and schema must use "
            "model_family='latent_class_diagonal_gaussian_mixture', "
            "fit_method='observed_data_em_diagonal_gaussian_mixture', and "
            "covariance_type='diag'; median/zero imputation followed by sklearn "
            "GaussianMixture is a different method and is not permitted. "
            "Copy the exact coordinate_scaling object from the representation "
            "schema into the candidate solution schema so candidate fitting and "
            "deterministic refits bind the same scaling contract. "
            "Bind the exact consumed bytes by copying the host-provided SHA-256 "
            "digests into representation_schema_sha256 and "
            "candidate_assignments_sha256; do not predict or reconstruct host "
            "evidence identifiers. "
            "This manifest records the "
            "agent's already-made selection; it must not introduce a new choice. "
            "The agent owns the method, criterion, and k." + role_boundary
        )
    if declarations & {
        "cluster_stability_spec",
        "trajectory_missingness_policy",
        "cluster_assignments",
        "cluster_stability_assignments",
    }:
        sections.append(
            "STABILITY/FREEZE ROLE: read only the exact files bound in "
            "EASYICU_RESOLVED_INPUTS_JSON for the declared upstream representation, "
            "trajectory-representation schema, cluster-selection manifest, candidate "
            "solution schema, and candidate fit/assignment artifacts. Treat the two "
            "typed schema manifests as authoritative and fail closed if either is "
            "missing or disagrees with the bound data. "
            "Apply each schema to its own contract: representation_columns and "
            "profile columns must exist in the trajectory representation, while only "
            "id_column and assignment_column must exist in candidate assignments. "
            "Never require representation coordinates to be duplicated in the "
            "assignment table. Resolve selected_n_clusters, selected_model_id, "
            "clustering_method/model_family, fit_method, covariance_type, id_column, "
            "assignment_column, and representation_columns from the candidate-solution "
            "and representation schemas first; use legacy selection/model artifacts "
            "only to cross-check those already-frozen values, not as a reason to reject "
            "a schema because the old manifest omitted a field. Stability refits must "
            "use the selected candidate record's exact model family, fit_method, "
            "covariance structure, coordinate order, and missing-data handling. Do not "
            "substitute a complete-data estimator, impute missing coordinates, or drop "
            "rows when the selected candidate fit used an observed-data method. "
            "The upstream representation rows and identifiers are the frozen "
            "eligible population: do not read COHORT_PARQUET, scan raw fixed-window "
            "or trajectory columns, or reapply cohort, anchor, observed-window, "
            "adult, or other eligibility rules. Reuse selected_n_clusters and the "
            "selected clustering method from the candidate-selection artifacts; "
            "use their exact representation_columns in the same order, and copy "
            "the selected candidate labels into the final cluster assignments. "
            "Read id_column from the upstream selection/model metadata. For a "
            "legacy artifact that lacks id_column, resolve it only when the frozen "
            "representation and candidate-assignment table have exactly one shared "
            "column, that column is complete and unique in both tables, and the two "
            "full identifier sets are equal; otherwise fail closed. Never select an "
            "identifier by its name, substring, column position, or distinct ratio. "
            "Read selected_model_id from the selection/model metadata. For a legacy "
            "artifact without model ids, accept only one fitted candidate-model record "
            "whose n_clusters equals selected_n_clusters and whose method family agrees "
            "with the frozen selection; derive a provenance id from the bound candidate-"
            "models evidence_id plus selected_n_clusters and record that derivation. "
            "Fail closed if zero or multiple records match. "
            "Read clustering_method from the selection manifest. For a legacy manifest "
            "without it, accept a method family only when the bound candidate-model "
            "artifact's top-level metadata and every fitted candidate record resolve "
            "to one identical normalized method family; record that legacy derivation "
            "and fail closed on absence or disagreement. Never infer the method from "
            "step ids, filenames, or prose. "
            "Do not compare candidate k values or choose a new method, k, coordinate "
            "layer, population, or reference assignment. If any required upstream "
            "field is absent or inconsistent, fail closed in step_summary.json "
            "instead of inferring or reconstructing it. A dedicated stability "
            "owner may carry a complete planner-owned trajectory_stability_spec; "
            "when present, materialize cluster_stability_spec.json by binding that "
            "unchanged packet to the current upstream evidence digests. The executor "
            "must not supply any missing resampling, seed, refit, comparison, "
            "alignment, or decision field. Write trajectory_missingness_policy.json "
            "with id_column, observation_family, observation_columns (in model order), "
            "the exact model representation_columns, min_observed_windows, "
            "profile_columns, profile_summary_statistic, "
            "clustering_method, n_clusters, time_axis='relative_hours', anchor, "
            "anchor_provenance, anchor_source, and trailing_na_policy={zero_imputation:false, "
            "eligibility_uses_observed_window_count:true, "
            "profile_summaries_ignore_missing:true}. Write cluster_assignments.csv "
            "with id_column and cluster; cluster_stability.csv with resample_id, "
            "n_overlap, adjusted_rand_index, clustering_method, refit_model_id, "
            "seed, sampling_method, sample_n, sample_id_hash, and selected_n_clusters; "
            "and cluster_stability_assignments.csv with resample_id, id_column, "
            "reference_cluster, and resampled_cluster. Use at least two genuinely "
            "distinct resamples/refits of that same method and same k, with distinct "
            "seeds, refit_model_id values, and sampled-id hashes; never use the "
            "outcome to form clusters. This owner writes no candidate-selection "
            "table, cluster sizes, profiles, outcome summaries, or figures."
        )
    if declarations & {"trajectory_profiles", "cluster_sizes", "outcome_by_cluster"}:
        outcome_clause = (
            " If outcome_by_cluster is declared, write outcome_by_cluster.csv: "
            "for a binary outcome use cluster,n,outcome_n,event_n,outcome_rate; "
            "for a non-binary outcome use cluster,n,outcome_n,summary_statistic,value."
            if "outcome_by_cluster" in declarations
            else ""
        )
        sections.append(
            "CHARACTERIZATION ROLE: read the frozen assignments and write "
            "trajectory_profiles.csv with cluster, source_column, "
            "window_start_hours, window_end_hours, summary_statistic, value, "
            "and n_observed; write cluster_sizes.csv with cluster and n. "
            "Profiles must be recomputed from the original source columns, "
            "preserve their declared scale, and ignore missing cells according "
            "to the agent-declared policy." + outcome_clause
        )
    if not sections:
        return ""
    return (
        "\n\nCROSS-STEP FIXED-WINDOW TRAJECTORY CONTRACT "
        "(standardized artifacts only; scientific choices remain agent-owned):\n- "
        + "\n- ".join(sections)
    )


__all__ = [
    "STABILITY_EXECUTOR_INPUTS",
    "STABILITY_EXECUTOR_OUTPUTS",
    "STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS",
    "TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD",
    "TRAJECTORY_STABILITY_METHOD_HEAD",
    "non_trajectory_clustering_stability_guide",
    "TrajectoryPlanDagEvaluation",
    "augment_trajectory_plan_products",
    "evaluate_trajectory_plan_dag",
    "TrajectoryRoleRequirement",
    "role_declaration_requirement",
    "trajectory_plan_contract_applies",
    "trajectory_context_is_bound",
    "trajectory_plan_dag_findings",
    "trajectory_planner_contract_guide",
    "trajectory_role_code_contract",
    "trajectory_role_result_findings",
    "trajectory_role_scope_summary_findings",
    "trajectory_step_roles",
]
