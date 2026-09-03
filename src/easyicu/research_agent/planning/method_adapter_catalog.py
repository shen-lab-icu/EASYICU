"""Planner-visible contracts for the first high-frequency method adapters.

The statistical implementations and execution selectors already have their
own owners.  This module does not reimplement either.  It publishes the small,
dependency-neutral contract a Planner needs to distinguish a real host adapter
from a reviewed kernel that still requires Coder-generated glue.

Every entry is deliberately capped at ``analysis_only``.  A family capability
and the normal evidence/reportability gates may impose a lower ceiling; this
catalog can never raise it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal, Tuple

__all__ = [
    "HIGH_FREQUENCY_METHOD_ADAPTERS",
    "MethodAdapterContract",
    "MethodAdapterGapError",
    "get_method_adapter_contract",
    "method_adapter_catalog_receipt",
    "require_method_adapter_contract",
]

AdapterScope = Literal["full_action", "typed_subcontract"]
AdapterClaimCeiling = Literal["analysis_only"]


@dataclass(frozen=True, slots=True)
class MethodAdapterContract:
    """One case-neutral route from a scientific action to a host owner."""

    adapter_id: str
    action_id: str
    scope: AdapterScope
    owner_module: str
    owner_entrypoint: str
    selection_kind: str
    required_declarations: Tuple[str, ...]
    validation_test_refs: Tuple[str, ...]
    claim_ceiling: AdapterClaimCeiling = "analysis_only"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MethodAdapterGapError(ValueError):
    """Typed, stable failure for an action without a registered adapter."""

    code = "method_adapter_not_registered"

    def __init__(self, action_id: str) -> None:
        self.action_id = str(action_id or "").strip()
        super().__init__(f"{self.code}: {self.action_id!r}")


def _adapter_group(
    action_ids: Tuple[str, ...],
    *,
    owner_module: str,
    owner_entrypoint: str,
    selection_kind: str,
    required_declarations: Tuple[str, ...],
    validation_test_refs: Tuple[str, ...],
    typed_subcontracts: Tuple[str, ...] = (),
) -> Tuple[MethodAdapterContract, ...]:
    typed = frozenset(typed_subcontracts)
    return tuple(
        MethodAdapterContract(
            adapter_id=action_id.replace(".", "_") + "_v1",
            action_id=action_id,
            scope=("typed_subcontract" if action_id in typed else "full_action"),
            owner_module=owner_module,
            owner_entrypoint=owner_entrypoint,
            selection_kind=selection_kind,
            required_declarations=required_declarations,
            validation_test_refs=validation_test_refs,
        )
        for action_id in action_ids
    )


HIGH_FREQUENCY_METHOD_ADAPTERS: Tuple[MethodAdapterContract, ...] = (
    *_adapter_group(
        ("descriptive.table_one",),
        owner_module=("easyicu.research_agent.execution.runners.table_one_executor"),
        owner_entrypoint="table_one_executor_code",
        selection_kind="grouped_table_one",
        required_declarations=(
            "one digest-bound typed cohort",
            "TableOneSpec with closed grouping levels and variable summaries",
            "table:table_one output contract",
        ),
        validation_test_refs=(
            "tests/research_agent/core/test_table_one_executor.py::test_standard_executor_selects_table_one_before_any_coder_path",
            "tests/research_agent/core/test_table_one_executor.py::test_table_one_executor_rejects_tampered_bound_cohort",
        ),
    ),
    *_adapter_group(
        (
            "descriptive.missingness_audit",
            "association.missingness_audit",
        ),
        owner_module=(
            "easyicu.research_agent.execution.runners.deterministic_missingness"
        ),
        owner_entrypoint="missingness_measurement_audit_code",
        selection_kind="missingness_audit",
        required_declarations=(
            "one closed cohort authority",
            "explicit audited variables or measurement products",
            "declared missingness output roles",
        ),
        validation_test_refs=(
            "tests/research_agent/execution/test_deterministic_missingness_runner.py::test_structured_availability_contract_is_selected_before_any_coder_path",
            "tests/research_agent/execution/test_deterministic_missingness_runner.py::test_genuinely_absent_declared_input_still_blocks",
        ),
    ),
    *_adapter_group(
        ("association.ordinal_trend",),
        owner_module=(
            "easyicu.research_agent.execution.runners.ordered_stratified_executor"
        ),
        owner_entrypoint="ordered_stratified_executor_code",
        selection_kind="ordered_stratified_analysis",
        required_declarations=(
            "one digest-bound typed cohort",
            "one upstream primary model requirement",
            "closed ordered levels plus binary and continuous outcomes",
        ),
        validation_test_refs=(
            "tests/research_agent/core/test_ordered_stratified_executor.py::test_typed_owner_executes_and_replays_without_coder",
            "tests/research_agent/core/test_ordered_stratified_executor.py::test_method_label_without_typed_spec_does_not_select_owner",
        ),
        typed_subcontracts=("association.ordinal_trend",),
    ),
    *_adapter_group(
        ("association.adjusted_association",),
        owner_module=(
            "easyicu.research_agent.execution.runners.adjusted_association_executor"
        ),
        owner_entrypoint="adjusted_association_executor_code",
        selection_kind="adjusted_association_estimates",
        required_declarations=(
            "one digest-bound typed cohort",
            "one exact primary model requirement",
            "typed exposure and covariate term roster",
        ),
        validation_test_refs=(
            "tests/research_agent/core/test_adjusted_association_executor.py::test_the_real_fresh19_primary_step_is_claimed_once_covariates_are_declared",
            "tests/research_agent/core/test_adjusted_association_executor.py::test_an_unimplemented_method_family_is_not_claimed",
        ),
    ),
    *_adapter_group(
        ("association.robustness_panel",),
        owner_module=(
            "easyicu.research_agent.execution.runners.deterministic_robustness"
        ),
        owner_entrypoint="robustness_sensitivity_preflight_code",
        selection_kind="robustness_sensitivity",
        required_declarations=(
            "one completed and locked primary model",
            "prespecified robustness replay products",
            "exact cohort and estimator identity",
        ),
        validation_test_refs=(
            "tests/research_agent/core/test_deterministic_robustness_preflight.py::test_preflight_fails_closed_without_completed_primary_estimate",
            "tests/research_agent/core/test_deterministic_robustness_preflight.py::test_preflight_fails_closed_without_locked_specs",
        ),
    ),
    *_adapter_group(
        (
            "prediction.discrimination_calibration",
            "prediction.calibration_metrics",
            "prediction.decision_curve",
            "prediction.internal_validation",
        ),
        owner_module=(
            "easyicu.research_agent.execution.runners.prediction_model_executor"
        ),
        owner_entrypoint="prediction_model_executor_code",
        selection_kind="prediction_model",
        required_declarations=(
            "one digest-bound typed cohort",
            "exact predictor, outcome, split and preprocessing contract",
            "one registered prediction action and its exact products",
        ),
        validation_test_refs=(
            "tests/research_agent/core/test_prediction_model_executor.py::test_prediction_owner_selects_only_exact_action_contract",
            "tests/research_agent/core/test_prediction_model_executor.py::test_prediction_workflow_is_group_safe_source_bound_and_renderable",
        ),
    ),
    *_adapter_group(
        (
            "time_to_event.cox_hr",
            "time_to_event.km_logrank",
            "time_to_event.ph_check",
        ),
        owner_module=(
            "easyicu.research_agent.execution.runners.survival_primary_executor"
        ),
        owner_entrypoint="survival_primary_executor_code",
        selection_kind="survival_primary_cox",
        required_declarations=(
            "one digest-bound typed cohort",
            "exact time origin, unit, event and censoring contract",
            "one Cox model requirement and proportional-hazards policy",
        ),
        validation_test_refs=(
            "tests/research_agent/core/test_survival_primary_executor.py::test_primary_survival_contract_selects_a_fully_sealed_host_executor",
            "tests/research_agent/core/test_survival_primary_executor.py::test_primary_survival_plan_cannot_fall_back_to_coder_without_bound_input",
        ),
        typed_subcontracts=(
            "time_to_event.km_logrank",
            "time_to_event.ph_check",
        ),
    ),
    *_adapter_group(
        (
            "phenotyping.cluster_solution",
            "phenotyping.k_selection",
            "phenotyping.cluster_stability",
            "phenotyping.cluster_sizes",
            "phenotyping.outcome_by_cluster",
        ),
        owner_module=(
            "easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor"
        ),
        owner_entrypoint="cross_sectional_phenotyping_executor_code",
        selection_kind="cross_sectional_phenotyping",
        required_declarations=(
            "one digest-bound typed cohort",
            "closed feature roster excluding outcomes",
            "candidate k grid, selection rule and stability design",
        ),
        validation_test_refs=(
            "tests/research_agent/core/test_cross_sectional_phenotyping_executor.py::test_phenotyping_owner_selects_only_the_exact_action_contract",
            "tests/research_agent/core/test_cross_sectional_phenotyping_executor.py::test_phenotyping_workflow_is_outcome_excluding_typed_and_renderable",
        ),
        typed_subcontracts=(
            "phenotyping.cluster_sizes",
            "phenotyping.outcome_by_cluster",
        ),
    ),
    *_adapter_group(
        ("phenotyping.trajectory_cluster_stability",),
        owner_module=(
            "easyicu.research_agent.execution.runners.trajectory_stability_executor"
        ),
        owner_entrypoint="trajectory_stability_executor_code",
        selection_kind="trajectory_cluster_stability",
        required_declarations=(
            "signed trajectory representation and selected candidate",
            "exact resampling, refit, alignment and seed policy",
            "typed upstream assignment and feature products",
        ),
        validation_test_refs=(
            "tests/research_agent/figures/test_trajectory_stability_executor.py::test_executor_is_case_neutral_and_replayable",
            "tests/research_agent/figures/test_trajectory_stability_executor.py::test_executor_fails_closed_on_untrusted_input_binding",
        ),
    ),
)


def _validate_catalog() -> None:
    adapter_ids = [item.adapter_id for item in HIGH_FREQUENCY_METHOD_ADAPTERS]
    action_ids = [item.action_id for item in HIGH_FREQUENCY_METHOD_ADAPTERS]
    if not 15 <= len(action_ids) <= 20:
        raise RuntimeError(
            "high-frequency method-adapter batch must contain 15-20 actions"
        )
    if len(adapter_ids) != len(set(adapter_ids)):
        raise RuntimeError("method-adapter ids must be unique")
    if len(action_ids) != len(set(action_ids)):
        raise RuntimeError("method-adapter action ids must be unique")
    for item in HIGH_FREQUENCY_METHOD_ADAPTERS:
        if not all(
            (
                item.adapter_id,
                item.action_id,
                item.owner_module,
                item.owner_entrypoint,
                item.selection_kind,
                *item.required_declarations,
                *item.validation_test_refs,
            )
        ):
            raise RuntimeError(f"method-adapter contract is incomplete: {item!r}")


_validate_catalog()
_BY_ACTION = {item.action_id: item for item in HIGH_FREQUENCY_METHOD_ADAPTERS}


def get_method_adapter_contract(action_id: str) -> MethodAdapterContract | None:
    """Return the exact adapter contract, or ``None`` without guessing."""

    return _BY_ACTION.get(str(action_id or "").strip())


def require_method_adapter_contract(action_id: str) -> MethodAdapterContract:
    """Return one exact adapter or raise a stable fail-closed error."""

    contract = get_method_adapter_contract(action_id)
    if contract is None:
        raise MethodAdapterGapError(action_id)
    return contract


def method_adapter_catalog_receipt() -> dict[str, object]:
    """Return a stable, digest-bound coverage receipt for quick acceptance."""

    entries = [item.to_dict() for item in HIGH_FREQUENCY_METHOD_ADAPTERS]
    canonical = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "easyicu.method_adapter_catalog/1",
        "adapter_count": len(entries),
        "catalog_sha256": hashlib.sha256(canonical).hexdigest(),
        "action_ids": [item.action_id for item in HIGH_FREQUENCY_METHOD_ADAPTERS],
        "claim_ceiling": "analysis_only",
    }
