"""Plan-level DAG contract for agent-decomposed trajectory phenotyping.

The planner may keep representation, candidate selection, stability freezing,
and descriptive characterization in one step or split them across steps.  This
module validates only that those agent-declared roles form one closed,
unambiguous artifact DAG.  It never selects a feature representation,
clustering method, cluster count, eligibility threshold, or scientific runner.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import (
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
    "representation": ("table:trajectory_membership",),
    "candidate_selection": ("manifest:cluster_selection",),
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
    if clustering and bool(tokens & {"stability", "consensus", "bootstrap", "resampling"}):
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
        any_of=frozenset(
            {"characteristic", "characteristics", "profile", "profiles"}
        ),
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
    if "stability" in tokens and bool(tokens & {"cluster", "clustering"}):
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
) -> List[ValidationFinding]:
    """Validate role-local replay metadata without choosing the science."""

    roles = trajectory_step_roles(step)
    if "candidate_selection" not in roles:
        return []
    raw_selection = step_summary.get("cluster_selection")
    try:
        selection = ClusterSelectionManifest.model_validate(raw_selection)
    except Exception as exc:
        return [
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
            )
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
        return []
    return [
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
        )
    ]


def trajectory_plan_contract_applies(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> bool:
    """Return the non-heuristic trigger for the run-level trajectory contract.

    Once the agent stamps a fixed-window plan as ``trajectory_clustering``, a
    missing or artifact-only role cannot make the contract disappear.  The
    evaluator will instead return explicit plan-contract errors.
    """

    return _normalise_token(plan.analysis_type) == "trajectory_clustering" and any(
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
) -> TrajectoryPlanDagEvaluation:
    """Validate the agent-declared trajectory plan without rewriting it."""

    if not trajectory_plan_contract_applies(plan=plan, context=context):
        return TrajectoryPlanDagEvaluation(False, {}, {}, (), ())

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
            findings.append(
                _finding(
                    "trajectory_role_missing",
                    "The trajectory plan is missing one structured scientific role.",
                    role=role,
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
                product for product in typed_inputs if _is_window_manifest_product(product)
            )
            manifest_producers = sorted(
                {
                    producers[0]
                    for _kind, manifest in manifest_inputs
                    for producers in (producer_candidates.get(manifest, []),)
                    if len(producers) == 1
                }
            )
            if not manifest_inputs:
                findings.append(
                    _finding(
                        "trajectory_window_manifest_missing",
                        "A representation built from an upstream panel must consume "
                        "a typed trajectory-window manifest from that panel's producer.",
                        step_id=representation_owner,
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
        if not multi_window_families:
            findings.append(
                _finding(
                    "trajectory_window_family_not_resolved",
                    "The representation owner does not select at least two fixed "
                    "windows from one declared family, either directly or through "
                    "a typed upstream panel manifest.",
                    step_id=representation_owner,
                    window_source_step_id=window_source_step.step_id,
                    selected_families=sorted(selected_by_family),
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
) -> List[ValidationFinding]:
    """Return plan-contract errors for a stamped fixed-window trajectory DAG."""

    return list(evaluate_trajectory_plan_dag(plan=plan, context=context).findings)


def augment_trajectory_plan_products(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Add only canonical replay products to the agent's existing role owners.

    This is schema normalization, not scientific planning: step identities,
    methods, inputs, order, and all existing outputs remain unchanged.  Missing
    or ambiguous roles are left for :func:`trajectory_plan_dag_findings` to
    block rather than being invented by the framework.
    """

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)
    if not evaluation.applies or set(evaluation.role_owners) != set(_ROLE_ORDER):
        return plan, []

    additions: Dict[str, List[str]] = defaultdict(list)
    for role, outputs in _ROLE_CANONICAL_OUTPUTS.items():
        owner = evaluation.role_owners[role]
        step = next(item for item in plan.steps if item.step_id == owner)
        for output in outputs:
            if output not in (step.expected_outputs or []):
                additions[owner].append(output)

    characterization_owner = evaluation.role_owners["characterization"]
    characterization = next(
        item for item in plan.steps if item.step_id == characterization_owner
    )
    declared_products = {
        product
        for output in characterization.expected_outputs or []
        if (parsed := _declared_product(output)) is not None
        for product in (parsed[1],)
    }
    if (
        declared_products & _CHARACTERIZATION_OUTCOME_PRODUCTS
        and "table:outcome_by_cluster"
        not in (characterization.expected_outputs or [])
    ):
        additions[characterization_owner].append("table:outcome_by_cluster")

    if not additions:
        return plan, []
    revised_steps = [
        step.model_copy(
            update={
                "expected_outputs": [
                    *(step.expected_outputs or []),
                    *additions.get(step.step_id, []),
                ]
            }
        )
        if step.step_id in additions
        else step
        for step in plan.steps
    ]
    revised = plan.model_copy(
        update={
            "steps": revised_steps,
            "revision": max(1, int(plan.revision)) + 1,
        }
    )
    return revised, [
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
                "preserved_step_ids": [step.step_id for step in plan.steps],
            },
        )
    ]


def trajectory_planner_contract_guide(
    *,
    context: ResearchContext,
    analysis_type: object,
) -> str:
    """Return the case-neutral planning schema for trajectory DAGs."""

    if _normalise_token(analysis_type) != "trajectory_clustering" or not any(
        variable.fixed_window_trajectory is not None
        for variable in (context.variables or [])
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
        "- representation: declare the chosen representation artifact and "
        "`table:trajectory_membership`;\n"
        "- candidate selection: declare a candidate fit/assignment artifact and "
        "`manifest:cluster_selection`;\n"
        "- stability/freeze: declare the frozen-solution assignment artifact, "
        "`manifest:trajectory_missingness_policy`, `table:cluster_assignments`, "
        "`table:cluster_stability`, and "
        "`table:cluster_stability_assignments`;\n"
        "- characterization: declare `table:trajectory_profiles` and "
        "`table:cluster_sizes`, plus `table:outcome_by_cluster` only when an "
        "outcome description is planned.\n"
        "Connect separate owners through explicit typed producer/consumer edges. "
        "If representation reads raw fixed-window columns directly, list them in "
        "its inputs. If it instead reads an upstream aligned panel, the panel "
        "producer must list the raw fixed-window columns in its inputs, produce "
        "both the panel and `manifest:trajectory_window_manifest`, and the "
        "representation owner must consume both. The manifest records ordered "
        "source columns with family and window boundaries; it is provenance, not "
        "permission for the framework to choose a horizon, method, k, threshold, "
        "or missing-data policy."
    )


def trajectory_role_code_contract(
    *,
    context: ResearchContext,
    step: AnalysisStep,
) -> str:
    """Return role-local schemas for an agent-decomposed trajectory DAG."""

    del context  # Schema is selected from typed products, never task prose.
    products = _step_products(step)
    declarations = {product for _kind, product in products}
    sections: List[str] = []
    window_manifests = sorted(
        product for product in products if _is_window_manifest_product(product)
    )
    if window_manifests:
        manifest_names = ", ".join(
            f"{name}.json" for _kind, name in window_manifests
        )
        sections.append(
            "UPSTREAM WINDOW-PANEL ROLE: write the declared manifest JSON "
            f"({manifest_names}) with panel_product and families; each family "
            "entry must contain family, ordered_source_columns, and for every "
            "source column its exact name, window_start_hours, and "
            "window_end_hours. Derive these fields from the declared source "
            "inputs. Do not silently omit an internal available bin."
        )
    if "trajectory_membership" in declarations:
        sections.append(
            "REPRESENTATION ROLE: write trajectory_membership.csv with the "
            "agent-selected id column plus observed_window_count, "
            "meets_min_observed_windows, included_in_clustering, and "
            "exclusion_reason. In step_summary.json also declare "
            "observation_family, ordered observation_columns, "
            "min_observed_windows, profile_columns, "
            "profile_summary_statistic (mean or median), time_axis='relative_hours', "
            "anchor, anchor_provenance, anchor_source, and trailing_na_policy. "
            "Do not impute unobserved trajectory cells with zero. This role only "
            "builds the representation: do not fit clusters, select k, freeze "
            "assignments, characterize profiles, or analyze outcomes here."
        )
    if "cluster_selection" in declarations:
        sections.append(
            "CANDIDATE-SELECTION ROLE: consume the declared upstream trajectory "
            "representation artifact as the clustering model matrix; do not "
            "reconstruct trajectory features from COHORT_PARQUET (the locked "
            "cohort may be used only for identifier reconciliation or audits). "
            "Write cluster_selection.json using the "
            "typed candidate schema: criterion, selection_rule (minimum or "
            "maximum), direction (minimize or maximize), "
            "selected_n_clusters, at least two finite candidates with "
            "n_clusters and criterion_value, and rationale. Repeat the exact "
            "object as step_summary.cluster_selection and report n_clusters "
            "and clustering_method. The agent owns the method, criterion, and k."
        )
    if declarations & {
        "trajectory_missingness_policy",
        "cluster_assignments",
        "cluster_stability_assignments",
    }:
        sections.append(
            "STABILITY/FREEZE ROLE: read the declared upstream representation "
            "and candidate-selection artifacts. Write trajectory_missingness_policy.json "
            "with id_column, observation_family, ordered observation_columns, "
            "min_observed_windows, profile_columns, profile_summary_statistic, "
            "clustering_method, n_clusters, time_axis='relative_hours', anchor, "
            "anchor_provenance, anchor_source, and trailing_na_policy={zero_imputation:false, "
            "eligibility_uses_observed_window_count:true, "
            "profile_summaries_ignore_missing:true}. Write cluster_assignments.csv "
            "with id_column and cluster; cluster_stability.csv with resample_id, "
            "n_overlap, adjusted_rand_index, clustering_method, refit_model_id, "
            "seed, sampling_method, sample_n, sample_id_hash, and selected_n_clusters; "
            "and cluster_stability_assignments.csv with resample_id, id_column, "
            "reference_cluster, and resampled_cluster. Use at least two genuinely "
            "distinct resamples/refits and never use the outcome to form clusters."
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
            "to the agent-declared policy."
            + outcome_clause
        )
    if not sections:
        return ""
    return (
        "\n\nCROSS-STEP FIXED-WINDOW TRAJECTORY CONTRACT "
        "(standardized artifacts only; scientific choices remain agent-owned):\n- "
        + "\n- ".join(sections)
    )


__all__ = [
    "TrajectoryPlanDagEvaluation",
    "augment_trajectory_plan_products",
    "evaluate_trajectory_plan_dag",
    "trajectory_plan_contract_applies",
    "trajectory_plan_dag_findings",
    "trajectory_planner_contract_guide",
    "trajectory_role_code_contract",
    "trajectory_role_result_findings",
    "trajectory_role_scope_summary_findings",
    "trajectory_step_roles",
]
