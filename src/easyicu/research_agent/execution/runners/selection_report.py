"""Explain why no deterministic executor claimed a step.

A step that no standard executor owns falls to the stochastic Coder without
saying so.  Twice now the only way to learn *which* owner declined, and on
which predicate, was to reconstruct the step by hand and call the ownership
predicates one at a time — for E1 Step 02 (a frozen input product-name tuple
did not recognise a product id from the host's own study-design playbook) and
for E1 Step 04 (a replanner-enriched third declared product left every closed
missingness contract behind).

This module records that decline as data, using the very same predicates the
selector uses, so the report cannot drift away from the decision it explains.
It never changes a selection: it only observes one.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ...schema import AnalysisPlan, AnalysisStep

__all__ = [
    "STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION",
    "standard_executor_candidate_report",
]

STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION = "easyicu.standard_executor_candidates/1"


def _candidate_predicates(
    *,
    plan: AnalysisPlan,
    resolved_bindings: Mapping[str, Any] | None,
) -> tuple[tuple[str, str, Callable[[AnalysisStep], bool]], ...]:
    """Bind every ownership predicate the selector consults, in its order.

    Entries are ``(kind, analysis_kind, predicate)``.  ``owner`` entries are the
    ones that can actually claim a step; ``detail`` entries are sub-predicates
    recorded only to locate *which* half of a composite owner said no, and must
    never be read as a claim.
    """

    # Imported here so this observability surface never becomes a second import
    # path that the selector's own module graph has to route around.
    from .cohort_summary_executor import cohort_summary_executor_owns_step
    from .deterministic_missingness import (
        is_compact_missingness_measurement_contract,
        is_measurement_bias_audit_contract,
        is_missingness_complete_case_contract,
        is_missingness_measurement_availability_contract,
        missingness_audit_executor_owns_step,
        missingness_audit_input_scope_supported,
    )
    from .exposure_outcome_distribution_figure_executor import (
        exposure_outcome_distribution_figure_executor_owns_step,
    )
    from .missingness_measurement_figure_executor import (
        missingness_measurement_figure_executor_owns_step,
    )
    from .prevalence_mortality_figure_executor import (
        prevalence_mortality_figure_executor_owns_step,
    )
    from .prevalence_outcome_figure_executor import (
        prevalence_outcome_figure_executor_owns_step,
    )
    from .table_one_executor import table_one_executor_owns_step
    from .trajectory_stability_executor import (
        trajectory_stability_executor_owns_step,
    )

    return (
        ("owner", "descriptive_cohort_summary", cohort_summary_executor_owns_step),
        (
            "owner",
            "prevalence_outcome_figure",
            prevalence_outcome_figure_executor_owns_step,
        ),
        (
            "owner",
            "prevalence_mortality_figure",
            prevalence_mortality_figure_executor_owns_step,
        ),
        (
            "owner",
            "exposure_outcome_distribution_figure",
            lambda step: exposure_outcome_distribution_figure_executor_owns_step(
                step,
                resolved_bindings=resolved_bindings,
                display_labels=plan.display_labels,
            ),
        ),
        (
            "owner",
            "missingness_measurement_figure",
            missingness_measurement_figure_executor_owns_step,
        ),
        ("owner", "grouped_table_one", table_one_executor_owns_step),
        ("owner", "missingness_audit", missingness_audit_executor_owns_step),
        # The missingness family is the one whose ownership is decided by three
        # separate closed contracts, so record which of them matched.  A step
        # that satisfies the input scope but no contract is the enrichment case:
        # the science grew a product and silently lost its deterministic owner.
        (
            "detail",
            "missingness_audit:input_scope",
            missingness_audit_input_scope_supported,
        ),
        (
            "detail",
            "missingness_audit:availability_contract",
            lambda step: is_missingness_measurement_availability_contract(
                step.method, step.expected_outputs
            ),
        ),
        (
            "detail",
            "missingness_audit:complete_case_contract",
            lambda step: is_missingness_complete_case_contract(
                step.method, step.expected_outputs
            ),
        ),
        (
            "detail",
            "missingness_audit:compact_contract",
            lambda step: is_compact_missingness_measurement_contract(
                step.method, step.expected_outputs
            ),
        ),
        (
            "detail",
            "missingness_audit:measurement_bias_contract",
            lambda step: is_measurement_bias_audit_contract(
                step.method, step.expected_outputs
            ),
        ),
        (
            "owner",
            "trajectory_cluster_stability",
            lambda step: trajectory_stability_executor_owns_step(step, plan=plan),
        ),
    )


def standard_executor_candidate_report(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    resolved_bindings: Mapping[str, Any] | None = None,
    claimed_by: str | None = None,
) -> dict[str, Any]:
    """Return the typed record of which owners considered this step.

    ``claimed_by`` is the ``analysis_kind`` the selector actually chose, or
    ``None`` when the step fell through to the Coder.  The coordinates the
    predicates key on — declared method and typed products — are recorded
    alongside, because those are what a reader needs in order to tell an
    unsupported analysis apart from a supported one wearing a new name.
    """

    candidates: list[dict[str, Any]] = []
    for kind, analysis_kind, predicate in _candidate_predicates(
        plan=plan, resolved_bindings=resolved_bindings
    ):
        entry: dict[str, Any] = {"kind": kind, "analysis_kind": analysis_kind}
        try:
            entry["owns"] = bool(predicate(step))
        except Exception as exc:  # observability must never fail a step
            entry["owns"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        candidates.append(entry)
    return {
        "schema_version": STANDARD_EXECUTOR_CANDIDATE_SCHEMA_VERSION,
        "step_id": str(step.step_id),
        "claimed_by": claimed_by,
        "owning_candidates": [
            entry["analysis_kind"]
            for entry in candidates
            if entry["kind"] == "owner" and entry["owns"]
        ],
        "declared_method": str(step.method or ""),
        "declared_outputs": [str(value) for value in (step.expected_outputs or ())],
        "declared_typed_inputs": [
            str(value) for value in (step.inputs or ()) if ":" in str(value)
        ],
        "declared_raw_input_count": sum(
            1 for value in (step.inputs or ()) if ":" not in str(value)
        ),
        "candidates": candidates,
    }
