"""Typed plan dependency graph, validation, and bounded step preservation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..authority.step_recovery import StepRecoverySignature
from ..contracts.declared_product import (
    PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS,
    RUNTIME_BINDABLE_TYPED_INPUT_KINDS,
    typed_product,
)
from ..contracts.step_families import _effect_contract_applies, _prediction_contract_applies, _step_is_figure_only
from .figure_step_contract import _parent_step_id_for_figure_step, _step_produces_figure
from ..schema import AnalysisPlan, AnalysisStep, ValidationFinding

def _step_is_primary_estimand_model(step: AnalysisStep) -> bool:
    """True when ``step`` is a result-bearing PRIMARY model (the estimand).

    Requires the Planner's typed ``primary`` role, a compatible method family,
    and a structured result product. Free-text id/intent tokens and
    preparation-only outputs do not establish ownership of the primary
    estimand.
    """

    if step.planned_analysis_role != "primary":
        return False

    # Exclude only a PURE figure/render child, not a combined model+figure step
    # (which the replanner can emit before the figure/table splitter runs). Both
    # contract helpers below already require a closed result-bearing product, so
    # a combined step that owns the estimand stays primary.
    if _step_is_figure_only(step):
        return False
    # Both helpers normalize only the ``<head>`` of a ``<head>_with_<rider>``
    # method and require a closed result-bearing product.  Thus a legitimate
    # mixed-effects model with a cohort-robust rider remains primary, while a
    # propensity-preparation or audit step cannot qualify through prose.
    return _effect_contract_applies(step) or _prediction_contract_applies(step)


def _step_is_baseline_context_table(step: AnalysisStep) -> bool:
    """True for a structured Table 1 / baseline-context analysis step.

    Match only the step id and declared outputs. Replan repair prose often
    mentions missing baseline context without owning a baseline artifact, so
    intent and free-form method text are deliberately excluded.
    """

    if _step_produces_figure(step):
        return False
    structured = " ".join(
        [step.step_id or "", " ".join(step.expected_outputs or [])]
    ).lower()
    return any(
        token in structured
        for token in (
            "table_one",
            "table one",
            "baseline_context",
            "baseline context",
            "baseline_table",
            "baseline table",
            "baseline_characteristics",
            "baseline characteristics",
        )
    )


def _typed_plan_dependency_graph(
    steps: Sequence[AnalysisStep],
) -> Tuple[Dict[str, Set[str]], List[ValidationFinding]]:
    """Build the unique producer graph for every typed ``kind:product`` input.

    The graph is deliberately method-agnostic.  Scientific methods remain
    planner-owned; this helper only enforces the execution fact that a typed
    input must have one declared producer in the same plan.  Missing and
    ambiguous producers are reported rather than guessed.
    """

    producers: Dict[Tuple[str, str], List[str]] = {}
    findings: List[ValidationFinding] = []
    for step in steps:
        for raw_output in step.expected_outputs or []:
            product = typed_product(raw_output)
            if product is None:
                continue
            if product[0] not in PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS:
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan output uses a product kind that the "
                            "runtime cannot materialise; the plan must be revised "
                            "before execution."
                        ),
                        detail={
                            "reason": "typed_output_kind_not_materializable",
                            "producer_step_id": step.step_id,
                            "typed_product": f"{product[0]}:{product[1]}",
                            "supported_kinds": sorted(
                                PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS
                            ),
                        },
                    )
                )
                continue
            producers.setdefault(product, []).append(step.step_id)

    dependencies: Dict[str, Set[str]] = {step.step_id: set() for step in steps}
    for step in steps:
        for raw_input in step.inputs or []:
            product = typed_product(raw_input)
            if product is None:
                continue
            if product[0] not in RUNTIME_BINDABLE_TYPED_INPUT_KINDS:
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan input uses a product kind that the "
                            "runtime cannot bind to current evidence; the plan "
                            "must be revised before execution."
                        ),
                        detail={
                            "reason": "typed_input_kind_not_runtime_bindable",
                            "consumer_step_id": step.step_id,
                            "typed_product": f"{product[0]}:{product[1]}",
                            "supported_kinds": sorted(
                                RUNTIME_BINDABLE_TYPED_INPUT_KINDS
                            ),
                        },
                    )
                )
                continue
            owner_ids = sorted(set(producers.get(product, [])))
            if not owner_ids:
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan input has no declared producer; the "
                            "plan must be revised before execution."
                        ),
                        detail={
                            "reason": "typed_input_producer_missing",
                            "consumer_step_id": step.step_id,
                            "typed_product": f"{product[0]}:{product[1]}",
                        },
                    )
                )
                continue
            if len(owner_ids) != 1:
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan input has multiple declared producers; "
                            "the framework cannot choose one on the agent's behalf."
                        ),
                        detail={
                            "reason": "typed_input_producer_ambiguous",
                            "consumer_step_id": step.step_id,
                            "typed_product": f"{product[0]}:{product[1]}",
                            "producer_step_ids": owner_ids,
                        },
                    )
                )
                continue
            producer_id = owner_ids[0]
            if producer_id != step.step_id:
                dependencies[step.step_id].add(producer_id)

    # Figure children created by the plan splitter remain paired with their
    # direct parent even when a legacy child omitted its typed table input.
    step_ids = set(dependencies)
    for step in steps:
        if not _step_produces_figure(step):
            continue
        parent_id = _parent_step_id_for_figure_step(step)
        if parent_id in step_ids and parent_id != step.step_id:
            dependencies[step.step_id].add(parent_id)
    return dependencies, findings


def _stable_topological_plan_steps(
    steps: Sequence[AnalysisStep],
    dependencies: Mapping[str, Set[str]],
) -> Tuple[List[AnalysisStep], List[str]]:
    """Return a stable producer-before-consumer order and any cycle members."""

    step_by_id = {step.step_id: step for step in steps}
    original_index = {step.step_id: idx for idx, step in enumerate(steps)}
    active_ids = set(step_by_id)
    remaining = {
        step_id: set(dependencies.get(step_id, set())) & active_ids
        for step_id in active_ids
    }
    dependents: Dict[str, Set[str]] = {step_id: set() for step_id in active_ids}
    for consumer_id, producer_ids in remaining.items():
        for producer_id in producer_ids:
            dependents[producer_id].add(consumer_id)

    ready = sorted(
        (step_id for step_id, producer_ids in remaining.items() if not producer_ids),
        key=lambda step_id: original_index[step_id],
    )
    ordered_ids: List[str] = []
    while ready:
        step_id = ready.pop(0)
        ordered_ids.append(step_id)
        for consumer_id in sorted(
            dependents[step_id], key=lambda value: original_index[value]
        ):
            remaining[consumer_id].discard(step_id)
            if not remaining[consumer_id] and consumer_id not in ordered_ids:
                ready.append(consumer_id)
        ready.sort(key=lambda value: original_index[value])

    cycle_ids = sorted(
        active_ids - set(ordered_ids), key=lambda value: original_index[value]
    )
    if cycle_ids:
        return list(steps), cycle_ids
    return [step_by_id[step_id] for step_id in ordered_ids], []


def _typed_plan_dag_findings(plan: AnalysisPlan) -> List[ValidationFinding]:
    """Validate the generic typed product DAG without choosing any science."""

    steps = list(plan.steps or [])
    dependencies, findings = _typed_plan_dependency_graph(steps)
    original_index = {step.step_id: idx for idx, step in enumerate(steps)}
    for consumer_id, producer_ids in dependencies.items():
        for producer_id in sorted(producer_ids):
            if original_index.get(producer_id, -1) >= original_index.get(
                consumer_id, len(steps)
            ):
                findings.append(
                    ValidationFinding(
                        validator="plan_typed_dag",
                        severity="error",
                        message=(
                            "A typed plan producer must precede its consumer; the "
                            "plan requires topological repair before execution."
                        ),
                        detail={
                            "reason": "typed_input_producer_not_preceding_consumer",
                            "producer_step_id": producer_id,
                            "consumer_step_id": consumer_id,
                        },
                    )
                )
    _ordered, cycle_ids = _stable_topological_plan_steps(steps, dependencies)
    if cycle_ids:
        findings.append(
            ValidationFinding(
                validator="plan_typed_dag",
                severity="error",
                message=(
                    "The typed plan dependency graph contains a cycle and cannot "
                    "be executed without planner revision."
                ),
                detail={
                    "reason": "typed_dependency_cycle",
                    "cycle_step_ids": cycle_ids,
                },
            )
        )
    return findings


def _step_recovery_contract(step: AnalysisStep) -> dict[str, Any]:
    """Persist the inspectable recovery identity of one dropped plan step."""

    signature = StepRecoverySignature.from_step(step)
    payload = signature.model_dump(mode="json")
    if signature.family_primary_result_requirement is None:
        payload.pop("family_primary_result_requirement", None)
    return {
        "step_id": signature.step_id,
        "planned_analysis_role": signature.planned_analysis_role,
        "method": signature.method,
        "expected_outputs": list(signature.expected_outputs),
        "recovery_signature": payload,
        "recovery_signature_sha256": signature.canonical_digest(),
    }


def _cap_plan_preserving_figure_steps(
    *,
    plan: AnalysisPlan,
    cap: int,
    protected_step_ids: Optional[Sequence[str]] = None,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Truncate a plan without orphaning required figure steps.

    Figure-only child steps produced by the splitter are load-bearing only when
    their upstream source step remains in the plan. Treat the parent and child
    as a small dependency unit: a cap may displace another non-figure step to
    keep both, but it must not preserve a figure child by replacing its parent.

    The first genuine primary-estimand model and first structured baseline /
    Table 1 step are article-contract anchors as well. Replan repair steps can
    push these anchors past ``steps[:cap]``; silently dropping either makes a
    busy plan incapable of answering the research question.
    """

    steps = list(plan.steps or [])
    if cap <= 0:
        return plan, []

    # Even a plan already under the numerical cap still needs a stable typed
    # dependency order.  Reordering unique producer edges is structural only;
    # missing, ambiguous, or cyclic edges remain fail-closed findings.
    if len(steps) <= cap:
        dependencies, findings = _typed_plan_dependency_graph(steps)
        ordered, cycle_ids = _stable_topological_plan_steps(steps, dependencies)
        if cycle_ids:
            findings.append(
                ValidationFinding(
                    validator="planner",
                    severity="error",
                    message=(
                        "The plan has a typed dependency cycle; planner revision "
                        "is required before execution."
                    ),
                    detail={
                        "reason": "typed_dependency_cycle",
                        "cycle_step_ids": cycle_ids,
                    },
                )
            )
        elif [step.step_id for step in ordered] != [step.step_id for step in steps]:
            findings.append(
                ValidationFinding(
                    validator="planner",
                    severity="warning",
                    message=(
                        "Reordered plan steps into stable typed producer-before-"
                        "consumer order."
                    ),
                    detail={
                        "reason": "typed_dependency_topological_reorder",
                        "original_step_ids": [step.step_id for step in steps],
                        "reordered_step_ids": [step.step_id for step in ordered],
                    },
                )
            )
        return plan.model_copy(update={"steps": ordered}), findings

    step_by_id = {step.step_id: step for step in steps}
    original_index = {step.step_id: idx for idx, step in enumerate(steps)}
    kept_ids = {step.step_id for step in steps[:cap]}
    protected_ids = {
        str(step_id) for step_id in (protected_step_ids or []) if step_id in step_by_id
    }
    # Role authority is method-family agnostic.  A survival, phenotyping, or
    # causal primary result is just as load-bearing as an association model and
    # must not disappear merely because it sits beyond the numerical cap.
    primary_owner = next(
        (step for step in steps if step.planned_analysis_role == "primary"),
        None,
    )
    if primary_owner is not None:
        protected_ids.add(primary_owner.step_id)
    for predicate in (_step_is_baseline_context_table,):
        owner = next((step for step in steps if predicate(step)), None)
        if owner is not None:
            protected_ids.add(owner.step_id)
    kept_ids.update(protected_ids)
    original_kept_ids = set(kept_ids)
    preserved_step_ids: List[str] = []
    displaced_step_ids: List[str] = []

    def _protected_parent_ids(ids: set[str]) -> set[str]:
        protected: set[str] = set()
        for step_id in ids:
            step = step_by_id.get(step_id)
            if step is None or not _step_produces_figure(step):
                continue
            parent_id = _parent_step_id_for_figure_step(step)
            if parent_id in ids:
                protected.add(parent_id)
        return protected

    def _remove_displaceable(required_ids: set[str]) -> bool:
        protected = set(required_ids) | protected_ids | _protected_parent_ids(kept_ids)
        candidates = [
            step_id
            for step_id in kept_ids
            if step_id not in protected
            and not _step_produces_figure(step_by_id[step_id])
        ]
        if not candidates:
            candidates = [step_id for step_id in kept_ids if step_id not in protected]
        if not candidates:
            return False
        displaced_id = max(candidates, key=lambda sid: original_index.get(sid, -1))
        kept_ids.remove(displaced_id)
        displaced_step_ids.append(displaced_id)
        return True

    # A protected article-contract anchor may sit beyond the initial
    # first-``cap`` slice. Make room immediately rather than relying on a later
    # figure step to happen to trigger eviction.
    while len(kept_ids) > cap:
        if not _remove_displaceable(set()):
            break

    for step in steps[cap:]:
        if not _step_produces_figure(step):
            continue
        parent_id = _parent_step_id_for_figure_step(step)
        required_ids = {step.step_id}
        if parent_id in step_by_id:
            required_ids.add(parent_id)
        if required_ids <= kept_ids:
            continue
        added_ids: List[str] = []
        removed_before = list(displaced_step_ids)
        for required_id in sorted(
            required_ids - kept_ids,
            key=lambda sid: original_index.get(sid, len(steps)),
        ):
            kept_ids.add(required_id)
            added_ids.append(required_id)
        while len(kept_ids) > cap:
            if not _remove_displaceable(required_ids):
                for added_id in added_ids:
                    kept_ids.discard(added_id)
                displaced_step_ids = removed_before
                break
        if step.step_id in kept_ids and step.step_id not in original_kept_ids:
            preserved_step_ids.append(step.step_id)

    # Dependency closure outranks display preservation.  A retained consumer is
    # never allowed to lose its unique typed producer merely to fit one more
    # figure under the cap.
    dependencies, _full_plan_dependency_findings = _typed_plan_dependency_graph(steps)

    def _expand_dependency_closure(ids: Set[str]) -> Set[str]:
        closed = set(ids)
        pending = list(ids)
        while pending:
            consumer_id = pending.pop()
            for producer_id in dependencies.get(consumer_id, set()):
                if producer_id not in closed:
                    closed.add(producer_id)
                    pending.append(producer_id)
        return closed

    kept_ids = _expand_dependency_closure(kept_ids)
    hard_protected_ids = _expand_dependency_closure(set(protected_ids))

    # Remove dependency leaves: first non-protected rendering leaves, then
    # other non-protected leaves.  Removing a consumer can make its producers
    # removable on the next pass, while no retained consumer is orphaned.
    while len(kept_ids) > cap:
        required_as_producer = {
            producer_id
            for consumer_id in kept_ids
            for producer_id in dependencies.get(consumer_id, set())
            if producer_id in kept_ids
        }
        leaf_candidates = [
            step_id
            for step_id in kept_ids
            if step_id not in hard_protected_ids and step_id not in required_as_producer
        ]
        if not leaf_candidates:
            break
        figure_leaves = [
            step_id
            for step_id in leaf_candidates
            if _step_produces_figure(step_by_id[step_id])
        ]
        candidates = figure_leaves or leaf_candidates
        displaced_id = max(candidates, key=lambda sid: original_index.get(sid, -1))
        kept_ids.remove(displaced_id)
        displaced_step_ids.append(displaced_id)

    kept = [step for step in steps if step.step_id in kept_ids]
    _retained_dependencies, dependency_findings = _typed_plan_dependency_graph(kept)
    kept_dependencies = {
        step_id: set(dependencies.get(step_id, set())) & kept_ids
        for step_id in kept_ids
    }
    kept, cycle_ids = _stable_topological_plan_steps(kept, kept_dependencies)
    dropped_ids = [step.step_id for step in steps if step.step_id not in kept_ids]
    dependency_displaced_figure_step_ids = [
        step_id for step_id in preserved_step_ids if step_id not in kept_ids
    ]
    preserved_step_ids = [
        step_id for step_id in preserved_step_ids if step_id in kept_ids
    ]
    capped = plan.model_copy(update={"steps": kept})
    # Name the scientific products that were dropped, not only the step ids. A
    # reader of the findings — or of the manuscript that quotes them — cannot
    # tell from "dropped: 13_x, 14_y" that the run no longer contains, say, the
    # calibration figure or the PH diagnostic it was asked for. The step ids are
    # internal; the declared outputs are the thing whose absence changes what
    # the analysis means.
    dropped_outputs = sorted(
        {
            str(output).strip()
            for step in steps
            if step.step_id in set(dropped_ids)
            for output in (getattr(step, "expected_outputs", None) or ())
            if str(output).strip()
        }
    )
    dropped_step_products = [
        _step_recovery_contract(step)
        for step in steps
        if step.step_id in set(dropped_ids)
    ]
    findings = [
        ValidationFinding(
            validator="planner",
            severity="warning",
            message=(
                f"Initial plan had {len(steps)} steps; truncated to "
                f"max_total_steps={cap}. Dropped: "
                f"{', '.join(dropped_ids[:6])}"
                + (" ..." if len(dropped_ids) > 6 else "")
                + (
                    "; the analysis no longer produces "
                    + ", ".join(dropped_outputs[:6])
                    + (" ..." if len(dropped_outputs) > 6 else "")
                    if dropped_outputs
                    else ""
                )
            ),
            detail={
                "dropped_step_ids": dropped_ids,
                "dropped_expected_outputs": dropped_outputs,
                "dropped_step_products": dropped_step_products,
                "plan_truncated": True,
                "cap": cap,
                "protected_step_ids": sorted(protected_ids),
                "preserved_figure_step_ids": preserved_step_ids,
                "dependency_displaced_figure_step_ids": (
                    dependency_displaced_figure_step_ids
                ),
                "displaced_step_ids": displaced_step_ids,
            },
        )
    ]
    findings.extend(dependency_findings)
    if len(kept_ids) > cap:
        findings.append(
            ValidationFinding(
                validator="planner",
                severity="error",
                message=(
                    "The plan cap cannot be satisfied without dropping a protected "
                    "step or one of its typed producers; planner revision is required."
                ),
                detail={
                    "reason": "typed_dependency_closure_exceeds_cap",
                    "cap": cap,
                    "retained_step_ids": [step.step_id for step in kept],
                    "protected_step_ids": sorted(hard_protected_ids),
                },
            )
        )
    if cycle_ids:
        findings.append(
            ValidationFinding(
                validator="planner",
                severity="error",
                message=(
                    "The retained plan has a typed dependency cycle; planner "
                    "revision is required before execution."
                ),
                detail={
                    "reason": "typed_dependency_cycle",
                    "cycle_step_ids": cycle_ids,
                },
            )
        )
    return capped, findings

