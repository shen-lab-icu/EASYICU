"""Typed registry and provenance ledger for deterministic repairs.

P0 of ENG-REPAIR1 keeps existing repair behavior unchanged while making every
repair auditable.  The registry is intentionally conservative: anything that
fits or substitutes an analytical method is classified as
``METHOD_SUBSTITUTION`` so strict-mode enforcement can fail closed in P2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


class RepairClass(str, Enum):
    """Integrity class for deterministic repairs."""

    SYNTACTIC = "syntactic"
    STRUCTURAL = "structural"
    CONTRACT_FILL = "contract_fill"
    METHOD_SUBSTITUTION = "method_substitution"


class RepairExecutionPolicy(str, Enum):
    """Whether later generated code may replace an authorized repair.

    Most repairs remain ordinary mutable code transforms.  A closed-source
    figure renderer is different: its exact parent product, input digests, and
    rendering-only scope were authorized before execution.  Letting a later
    coder repair replace that adapter would silently hand scientific product
    ownership back to generated code while retaining the renderer's provenance.
    """

    MUTABLE = "mutable"
    SEALED_RENDERER = "sealed_renderer"


class Repair(Protocol):
    """Minimal protocol future repair objects must satisfy."""

    repair_id: str
    repair_class: RepairClass


@dataclass(frozen=True)
class RepairMetadata:
    """Static metadata attached to a deterministic repair."""

    repair_id: str
    repair_class: RepairClass
    invariants: Tuple[str, ...] = ()
    introduces_numbers: bool = False
    requires_disclosure: bool = False
    selection_rule_required: bool = False
    execution_policy: RepairExecutionPolicy = RepairExecutionPolicy.MUTABLE
    figure_product_slots: Tuple[str, ...] = ()
    planner_methods: Tuple[str, ...] = ()
    planner_method_required: bool = True
    planner_parent_output_role_groups: Tuple[Tuple[Tuple[str, ...], ...], ...] = ()
    implementation_modules: Tuple[str, ...] = ()
    description: str = ""
    classification_source: str = "exact"


@dataclass(frozen=True)
class RepairProvenance:
    """One attempted or applied repair record."""

    repair_id: str
    repair_class: str
    step_id: Optional[str]
    trigger: Dict[str, Any] = field(default_factory=dict)
    transformation: str = ""
    invariants_checked: Tuple[str, ...] = ()
    invariants_passed: Optional[bool] = None
    invariant_status: str = "unverified"
    invariant_failures: Tuple[str, ...] = ()
    introduces_numbers: bool = False
    requires_disclosure: bool = False
    selection_rule: Optional[str] = None
    outcome: str = "applied"
    model_id: Optional[str] = None
    applied_at: str = ""
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    classification_source: str = "exact"


STRUCTURAL_INVARIANTS = ("row_set_unchanged", "n_unchanged")
CONTRACT_FILL_INVARIANTS = (
    "source_values_preexisting",
    "deterministic_selection_rule",
    "selected_value_surfaced",
)
METHOD_SUBSTITUTION_INVARIANTS = ("requires_disclosure",)


class InvariantStatus(str, Enum):
    """Three-state outcome of a runtime invariant evaluation.

    P0 recorded a bare ``invariants_passed`` that defaulted to ``True`` for any
    repair declaring invariants, even though nothing was actually checked.  P1
    replaces that with an honest three-state result so an unverified invariant
    is never reported as a pass.
    """

    VERIFIED_PASS = "verified_pass"
    VERIFIED_FAIL = "verified_fail"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class RepairObservedState:
    """Observable state captured before/after a repair, for invariant checks.

    Fields are all optional: a checker returns ``None`` (unverified) when the
    state it needs is absent, so a repair applied at the code-rewrite layer
    (where row-level data is not observable) is honestly recorded as
    ``UNVERIFIED`` rather than a fake pass.
    """

    row_count: Optional[int] = None
    id_values: Optional[Tuple[Any, ...]] = None
    value_pool: Optional[Tuple[Any, ...]] = None
    filled_value: Optional[Any] = None


@dataclass(frozen=True)
class InvariantEvaluation:
    """Result of evaluating all declared invariants for one repair application."""

    status: str
    passed: Optional[bool]
    checked: Tuple[str, ...]
    failures: Tuple[str, ...]


def _check_n_unchanged(
    before: Optional[RepairObservedState], after: Optional[RepairObservedState]
) -> Optional[bool]:
    if before is None or after is None:
        return None
    if before.row_count is None or after.row_count is None:
        return None
    return before.row_count == after.row_count


def _check_row_set_unchanged(
    before: Optional[RepairObservedState], after: Optional[RepairObservedState]
) -> Optional[bool]:
    if before is None or after is None:
        return None
    if before.id_values is None or after.id_values is None:
        return None
    return frozenset(before.id_values) == frozenset(after.id_values)


def _check_source_values_preexisting(
    after: Optional[RepairObservedState],
) -> Optional[bool]:
    if after is None or after.filled_value is None or after.value_pool is None:
        return None
    return after.filled_value in set(after.value_pool)


def _check_deterministic_selection_rule(selection_rule: Optional[str]) -> bool:
    return bool(selection_rule and str(selection_rule).strip())


def evaluate_invariants(
    metadata: RepairMetadata,
    *,
    before_state: Optional[RepairObservedState] = None,
    after_state: Optional[RepairObservedState] = None,
    selection_rule: Optional[str] = None,
) -> InvariantEvaluation:
    """Run each declared invariant and return an honest three-state result.

    A declared invariant whose required state is unavailable is left
    unevaluated (``None``); it is never silently counted as a pass.  Status is
    ``VERIFIED_FAIL`` if any checkable invariant failed, ``VERIFIED_PASS`` only
    if every declared invariant was checked and held (vacuously true when a
    repair declares no invariants), otherwise ``UNVERIFIED``.
    """

    results: Dict[str, Optional[bool]] = {}
    for invariant in metadata.invariants:
        if invariant == "n_unchanged":
            results[invariant] = _check_n_unchanged(before_state, after_state)
        elif invariant == "row_set_unchanged":
            results[invariant] = _check_row_set_unchanged(before_state, after_state)
        elif invariant == "source_values_preexisting":
            results[invariant] = _check_source_values_preexisting(after_state)
        elif invariant == "deterministic_selection_rule":
            results[invariant] = _check_deterministic_selection_rule(selection_rule)
        elif invariant == "requires_disclosure":
            results[invariant] = metadata.requires_disclosure is True
        else:
            # e.g. selected_value_surfaced — not observable at this layer yet.
            results[invariant] = None

    checked = tuple(name for name, value in results.items() if value is not None)
    failures = tuple(name for name, value in results.items() if value is False)

    if failures:
        return InvariantEvaluation(
            InvariantStatus.VERIFIED_FAIL.value, False, checked, failures
        )
    if not metadata.invariants:
        return InvariantEvaluation(InvariantStatus.VERIFIED_PASS.value, True, (), ())
    if checked and len(checked) == len(metadata.invariants):
        return InvariantEvaluation(
            InvariantStatus.VERIFIED_PASS.value, True, checked, ()
        )
    return InvariantEvaluation(InvariantStatus.UNVERIFIED.value, None, checked, ())


def _meta(
    repair_id: str,
    repair_class: RepairClass,
    *,
    invariants: Sequence[str] = (),
    introduces_numbers: bool = False,
    requires_disclosure: bool = False,
    selection_rule_required: bool = False,
    execution_policy: RepairExecutionPolicy = RepairExecutionPolicy.MUTABLE,
    figure_product_slots: Sequence[str] = (),
    planner_methods: Sequence[str] = (),
    planner_method_required: bool = True,
    planner_parent_output_role_groups: Sequence[Sequence[Sequence[str]]] = (),
    implementation_modules: Sequence[str] = (),
    description: str = "",
) -> RepairMetadata:
    return RepairMetadata(
        repair_id=repair_id,
        repair_class=repair_class,
        invariants=tuple(invariants),
        introduces_numbers=introduces_numbers,
        requires_disclosure=requires_disclosure,
        selection_rule_required=selection_rule_required,
        execution_policy=execution_policy,
        figure_product_slots=tuple(figure_product_slots),
        planner_methods=tuple(str(value) for value in planner_methods),
        planner_method_required=planner_method_required,
        planner_parent_output_role_groups=tuple(
            tuple(tuple(str(token) for token in suffix) for suffix in alternatives)
            for alternatives in planner_parent_output_role_groups
        ),
        implementation_modules=tuple(implementation_modules),
        description=description,
    )


_SYNTACTIC_REPAIRS = {
    "attrition_rule_id_canonicalization_v1",
    "boolean_mask_reduction_precedence_v1",
    "boolean_reduction_identity_v1",
    "categorical_distribution_clinical_bin_role_v1",
    "categorical_declared_order_check_v1",
    "closed_counts_declared_levels_binding_v1",
    "closed_counts_direct_host_call_v1",
    "closed_counts_level_column_v1",
    "closed_counts_stable_keywords_v1",
    "figure_contract_source_data_schema_v1",
    "fstring_runtime_quote_compat_v1",
    "host_helper_keyword_only_call_v1",
    "local_helper_unpack_receipt_v1",
    "local_read_before_assignment_hoist_v1",
    "matplotlib_patch_source_rows_v1",
    "measurement_provenance_envelope_alias_v1",
    "measurement_receipt_stable_binding_v1",
    "missing_os_import_v1",
    "pandas_boolean_index_alignment_v1",
    "pandas_merge_dynamic_column_collision_guard_v1",
    "plausibility_range_schema_keys_v1",
    "strip_python_prefix_v1",
    "restore_shadowed_json_module_v1",
    "replace_hallucinated_figure_utils_import_v1",
    "prediction_calibration_import_fix_v1",
    "publication_export_audit_paths_v1",
    "raw_contract_document_fallback_v1",
    "raw_contract_list_type_assertion_v1",
    "raw_contract_mapping_iteration_v1",
    "relocate_known_host_helper_import_v1",
    "resolved_context_digest_load_v1",
    "resolved_input_consumption_contract_owner_v1",
    "resolved_input_manifest_env_v1",
    "resolved_input_identity_key_v1",
    "resolved_input_json_document_adapter_v1",
    "resolved_input_run_root_v1",
    "resolved_typed_input_precedence_v1",
    "scalar_cast_before_reduction_v1",
    "statsmodels_interval_method_label_v1",
    "strict_numeric_input_result_projection_v1",
    "superseded_manual_provenance_receipt_v1",
    "table_one_planner_spec_binding_v1",
    "unresolved_input_binding_receipts_v1",
    "validation_finding_json_default_v1",
    "undefined_mapping_near_match_alias_v1",
}

_STRUCTURAL_REPAIRS = {
    "all_rows_outcome_coordinate_filter_v1",
    "all_rows_profile_roles_display_v1",
    "arbitrary_column_fallback_fail_closed_v1",
    "audit_only_companion_value_selector_v1",
    "availability_fraction_component_denominator_v1",
    "binary_domain_authored_feasibility_v1",
    "bound_figure_source_projection_v1",
    "bound_figure_source_projection_v2",
    "bound_percentage_identity_guard_v1",
    "cohort_csv_to_parquet_v1",
    "cohort_file_direct_read_v1",
    "flag_only_plausibility_range_retention_v1",
    "categorical_level_reconciliation_guard_v1",
    "conditional_nonfinite_fail_closed_guard_v1",
    "direct_bound_figure_source_materialization_v1",
    "direct_bound_figure_source_projection_v1",
    "host_schema_numeric_alias_v1",
    "dedupe_predictor_numeric_design_v1",
    "dedupe_required_cols_outcome_v1",
    "include_outcome_in_all_vars_v1",
    "inline_missing_to_jsonable_utils_v1",
    "json_dump_numpy_key_sanitizer_v1",
    "host_validation_helper_reraise_v1",
    "local_wilson_proportion_confint_v1",
    "lossy_numeric_coercion_guard_v1",
    "llm_proven_numeric_domain_guards_v1",
    "matplotlib_errorbar_xerr_shape_v1",
    "measurement_provenance_summary_mapping_v1",
    "measurement_provenance_summary_mapping_v2",
    "measurement_provenance_host_receipts_v1",
    "measurement_provenance_before_outputs_v1",
    "nonfinite_audit_preserve_observed_v1",
    "nonfinite_audit_host_strict_boundary_v2",
    "nonfinite_missing_mask_conflation_v1",
    "non_tabular_companion_row_gate_v1",
    "normalize_first_time_companion_v1",
    "observed_binary_primary_exposure_guard_v1",
    "penalized_convergence_contract_v1",
    "penalized_convergence_contract_v2",
    "primary_predictor_safe_summary_lookup_v1",
    "raw_input_physical_superset_guard_v1",
    "host_receipt_source_envelope_v1",
    "provenance_bidirectional_pair_scan_v1",
    "provenance_checked_status_contract_v1",
    "provenance_custom_helper_to_host_receipt_v1",
    "provenance_fail_closed_guard_v1",
    "provenance_helper_reraise_v1",
    "proportion_confint_nobs_keyword_v1",
    "publication_bundle_promote_v1",
    "returned_coercion_loss_guard_v1",
    "render_only_effect_echo_suppression_v1",
    "remove_pandas_cut_observed_keyword_v1",
    "sklearn_bool_imputer_cast_v1",
    "sibling_figure_exports_promote_v1",
    "statsmodels_conf_int_filter_axis_v1",
    "statsmodels_dummy_design_float_v1",
    "statsmodels_helper_design_float_v1",
    "strict_numeric_nonfinite_guard_v1",
    "structured_analysis_role_selection_v1",
    "table_one_binary_key_string_v1",
    "text_distribution_denominator_from_counts_v1",
    "typed_output_normalization_v1",
    "unavailable_figure_full_source_projection_v1",
    "unused_nullable_numeric_validation_v1",
    # Only renderers with an exact direct-parent product, closed source schema,
    # and no scientific selection are automatic.  Heuristic result-table,
    # exposure, outcome, or model selection remains METHOD_SUBSTITUTION by the
    # conservative unknown-id fallback until it gains a typed source contract.
    "ordered_category_distribution_publication_bundle_v1",
    "ordered_category_distribution_availability_publication_bundle_v2",
    "distribution_availability_publication_bundle_from_parent_outputs_v1",
    "continuous_measurement_audit_publication_bundle_v1",
    "absolute_risk_incidence_prevalence_publication_bundle_v1",
    "association_publication_bundle_from_planned_model_contract_v1",
    "cohort_flow_publication_bundle_from_parent_outputs_v1",
    "sensitivity_publication_bundle_from_locked_summary_v1",
    "missingness_publication_bundle_from_parent_outputs_v1",
    # Step-summary salvage that faithfully relocates the agent's own output
    # (stdout JSON / a named summary artefact) into step_summary.json. No new
    # numbers are introduced; it is a representation/location change.
    "summary_salvage_stdout_json_v1",
    "summary_salvage_named_json_v1",
    "summary_output_registry_canonicalization_v1",
    "sklearn_runtime_object_diagnostics_v1",
    # Adds the missing ``elif var == "age"`` branch to a covariate-coding
    # metadata table so the loop stops KeyError-ing on the one demographic
    # covariate that has no measured-indicator entry. The model, rows, and
    # fill strategy are untouched; the appended row only reports what the
    # script already does, with values computed from the existing dataframe.
    "age_covariate_no_measured_indicator_v1",
}


# These adapters have a closed direct-parent product, digest-bound inputs, and
# rendering-only scope.  Their outputs may be audited or rejected, but their
# code must never be replaced by a later LLM repair while the registry still
# credits the structural renderer.
_SEALED_RENDERER_REPAIRS = {
    "ordered_category_distribution_publication_bundle_v1",
    "ordered_category_distribution_availability_publication_bundle_v2",
    "distribution_availability_publication_bundle_from_parent_outputs_v1",
    "continuous_measurement_audit_publication_bundle_v1",
    "absolute_risk_incidence_prevalence_publication_bundle_v1",
    "association_publication_bundle_from_planned_model_contract_v1",
    "cohort_flow_publication_bundle_from_parent_outputs_v1",
    "sensitivity_publication_bundle_from_locked_summary_v1",
    "missingness_publication_bundle_from_parent_outputs_v1",
}

_SEALED_RENDERER_PRODUCT_SLOTS: Dict[str, Tuple[str, ...]] = {
    "ordered_category_distribution_publication_bundle_v1": (
        "distribution",
        "availability",
    ),
    "ordered_category_distribution_availability_publication_bundle_v2": (
        "distribution",
        "availability",
    ),
    "distribution_availability_publication_bundle_from_parent_outputs_v1": (
        "distribution",
        "availability",
    ),
    "continuous_measurement_audit_publication_bundle_v1": (
        "distribution",
        "availability",
    ),
    "absolute_risk_incidence_prevalence_publication_bundle_v1": ("absolute_risk",),
    "association_publication_bundle_from_planned_model_contract_v1": (
        "primary_estimand",
        "precision_audit",
    ),
    "cohort_flow_publication_bundle_from_parent_outputs_v1": (
        "cohort_flow",
        "attrition_audit",
    ),
    "sensitivity_publication_bundle_from_locked_summary_v1": (
        "robustness_plot",
        "robustness_denominator_audit",
    ),
    "missingness_publication_bundle_from_parent_outputs_v1": (
        "missingness_measurement",
    ),
}

# Sealed renderers are selected only from the host-recorded Planner contract.
# Each outer tuple item is one required logical parent-output role; its inner
# tuple lists exact terminal-token alternatives for that role.  No prose or
# physical filename participates in this registry.
from .planning.method_vocabulary import (
    ADJUSTED_ASSOCIATION_MODELS,
    BINARY_OUTCOME_INCIDENCE_AND_ABSOLUTE_RISK,
    COHORT_DEFINITION,
    COHORT_DEFINITION_SENSITIVITY,
    DISTRIBUTION_SUMMARY_AND_MISSINGNESS_AUDIT,
    EXPOSURE_DISTRIBUTION_AND_MISSINGNESS_AUDIT,
    MISSINGNESS_SOURCE_AVAILABILITY_AUDIT,
    ORDINAL_EXPOSURE_DERIVATION_AND_QUALITY_CONTROL,
    RIGHT_SKEWED_DISTRIBUTION_AND_MEASUREMENT_AVAILABILITY_AUDIT,
)

_SEALED_RENDERER_PLANNER_METHODS: Dict[str, Tuple[str, ...]] = {
    "ordered_category_distribution_publication_bundle_v1": (
        ORDINAL_EXPOSURE_DERIVATION_AND_QUALITY_CONTROL,
    ),
    # v2 is an additive compatibility renderer for an already-sealed Planner
    # product pair.  It is authorized by exact typed output roles plus a
    # digest-bound ordinal/schema contract, never by a model-authored method
    # string.  Keeping this tuple empty prevents accidental text routing.
    "ordered_category_distribution_availability_publication_bundle_v2": (),
    "distribution_availability_publication_bundle_from_parent_outputs_v1": (
        EXPOSURE_DISTRIBUTION_AND_MISSINGNESS_AUDIT,
    ),
    "continuous_measurement_audit_publication_bundle_v1": (
        RIGHT_SKEWED_DISTRIBUTION_AND_MEASUREMENT_AVAILABILITY_AUDIT,
        DISTRIBUTION_SUMMARY_AND_MISSINGNESS_AUDIT,
    ),
    "absolute_risk_incidence_prevalence_publication_bundle_v1": (
        BINARY_OUTCOME_INCIDENCE_AND_ABSOLUTE_RISK,
    ),
    "association_publication_bundle_from_planned_model_contract_v1": (
        ADJUSTED_ASSOCIATION_MODELS,
    ),
    "cohort_flow_publication_bundle_from_parent_outputs_v1": (COHORT_DEFINITION,),
    "sensitivity_publication_bundle_from_locked_summary_v1": (
        COHORT_DEFINITION_SENSITIVITY,
    ),
    "missingness_publication_bundle_from_parent_outputs_v1": (
        MISSINGNESS_SOURCE_AVAILABILITY_AUDIT,
    ),
}
_SEALED_RENDERER_PARENT_OUTPUT_ROLE_GROUPS: Dict[
    str, Tuple[Tuple[Tuple[str, ...], ...], ...]
] = {
    "ordered_category_distribution_publication_bundle_v1": ((("distribution",),),),
    "ordered_category_distribution_availability_publication_bundle_v2": (
        (("distribution",),),
        (
            ("measurement", "availability"),
            ("availability",),
        ),
    ),
    "distribution_availability_publication_bundle_from_parent_outputs_v1": (
        (("distribution",),),
        (
            ("measurement", "audit"),
            ("availability",),
            ("measurement", "coverage"),
            ("source", "coverage"),
            ("missingness",),
        ),
    ),
    "continuous_measurement_audit_publication_bundle_v1": (
        (("distribution",),),
        (("missingness",),),
    ),
    "absolute_risk_incidence_prevalence_publication_bundle_v1": (
        (("outcome", "incidence"),),
        (("exposure", "prevalence"),),
    ),
    "association_publication_bundle_from_planned_model_contract_v1": (
        (("adjusted", "association", "estimates"),),
    ),
    "cohort_flow_publication_bundle_from_parent_outputs_v1": (
        (("cohort", "flow"),),
        (("attrition",),),
    ),
    "sensitivity_publication_bundle_from_locked_summary_v1": (
        (("robustness", "summary"),),
    ),
    "missingness_publication_bundle_from_parent_outputs_v1": (
        (("missingness", "audit"),),
        (("measurement", "source", "audit"),),
    ),
}

_COMMON_SEALED_RENDERER_MODULES = (
    "easyicu.research_agent.pipeline",
    "easyicu.research_agent.contracts.declared_product",
    "easyicu.research_agent.repair_registry",
    "easyicu.research_agent.figures.publication",
)
_SEALED_RENDERER_IMPLEMENTATION_MODULES: Dict[str, Tuple[str, ...]] = {
    "ordered_category_distribution_publication_bundle_v1": (
        *_COMMON_SEALED_RENDERER_MODULES,
        "easyicu.research_agent.figures.ordered_distribution",
    ),
    "ordered_category_distribution_availability_publication_bundle_v2": (
        *_COMMON_SEALED_RENDERER_MODULES,
        "easyicu.research_agent.authority.figure_renderer",
        "easyicu.research_agent.figures.ordered_distribution",
    ),
    "distribution_availability_publication_bundle_from_parent_outputs_v1": (
        *_COMMON_SEALED_RENDERER_MODULES,
        "easyicu.research_agent.figures.distribution_availability",
    ),
    "continuous_measurement_audit_publication_bundle_v1": (
        *_COMMON_SEALED_RENDERER_MODULES,
        "easyicu.research_agent.figures.continuous_measurement_audit",
    ),
    "absolute_risk_incidence_prevalence_publication_bundle_v1": (
        *_COMMON_SEALED_RENDERER_MODULES,
        "easyicu.research_agent.figures.absolute_risk",
    ),
    "association_publication_bundle_from_planned_model_contract_v1": (
        *_COMMON_SEALED_RENDERER_MODULES,
    ),
    "cohort_flow_publication_bundle_from_parent_outputs_v1": (
        *_COMMON_SEALED_RENDERER_MODULES,
    ),
    "sensitivity_publication_bundle_from_locked_summary_v1": (
        *_COMMON_SEALED_RENDERER_MODULES,
    ),
    "missingness_publication_bundle_from_parent_outputs_v1": (
        *_COMMON_SEALED_RENDERER_MODULES,
        "easyicu.research_agent.figures.missingness_source",
    ),
}

_CONTRACT_FILL_REPAIRS: set[str] = set()

_METHOD_SUBSTITUTION_REPAIRS = {
    # Drops overadjustment covariates the overadjustment_auditor objectively
    # named (mediator/collider-style adjustors). Changing the adjustment set
    # changes the estimand specification, so it must be disclosed even though
    # the trigger is an auditor finding rather than a model failure.
    "drop_overadjustment_covariates_v1",
    "categorical_primary_association_selection_v1",
    "cut_bins_flatten_v1",
    "derived_analysis_cohort_materialization_v1",
    "dtype_coerce_v1",
    "filter_x_cols_after_dummy_encoding_v1",
    "filter_x_cols_before_dropna_after_dummy_encoding_v1",
    "logit_regularized_fit_v1",
    "logreg_impute_v1",
    # Reduces the design matrix to a full-rank column subset (preserving the
    # primary predictor) and swaps sm.Logit for sm.GLM(Binomial) after a
    # singular-matrix null result. Both the covariate set and the estimator
    # change, so results require disclosure.
    "rank_safe_statsmodels_design_v1",
    "missing_indicator_source_df_v1",
    "prediction_preserve_categorical_before_ohe_v1",
    "primary_predictor_omitted_from_design_v1",
    "publication_bundle_promote_script_v1",
    "publication_contract_optional_v1",
    "robustness_missingness_contract_v1",
    "robustness_encode_sex_before_numeric_checks_v1",
    "robustness_predictor_design_and_plot_v1",
    "seaborn_matplotlib_fallback_v1",
    "sex_binary_encode_for_logit_v1",
    "sex_covariate_numeric_loop_guard_v1",
    "sex_numeric_coercion_before_dropna_v1",
    "statsmodels_endog_exog_index_align_v1",
    "strip_unknown_cols_from_list_literals_v1",
    "summary_salvage_minimal_contract_v1",
    "zero_impute_to_complete_case_v1",
    "outcome_incidence_descriptive_repair_v1",
    "prediction_discrimination_template_v1",
    "prediction_split_minimal_v1",
    "table_one_descriptive_repair_v1",
    "validation_nonconvergence_fallback_v1",
}


# Generic code repair may fix syntax or representation, but it must not replace
# an agent-authored scientific or descriptive analysis. Standard products such
# as Table One and outcome incidence belong in explicitly routed
# AuxiliaryRunners with typed inputs; the old syntax-triggered templates remain
# available only as historical provenance and direct unit-test fixtures.
AUTOMATIC_METHOD_SUBSTITUTION_ALLOWLIST: frozenset[str] = frozenset()


REPAIR_METADATA: Dict[str, RepairMetadata] = {
    **{
        repair_id: _meta(repair_id, RepairClass.SYNTACTIC)
        for repair_id in _SYNTACTIC_REPAIRS
    },
    **{
        repair_id: _meta(
            repair_id,
            RepairClass.STRUCTURAL,
            invariants=STRUCTURAL_INVARIANTS,
            execution_policy=(
                RepairExecutionPolicy.SEALED_RENDERER
                if repair_id in _SEALED_RENDERER_REPAIRS
                else RepairExecutionPolicy.MUTABLE
            ),
            figure_product_slots=_SEALED_RENDERER_PRODUCT_SLOTS.get(repair_id, ()),
            planner_methods=_SEALED_RENDERER_PLANNER_METHODS.get(repair_id, ()),
            planner_method_required=(
                repair_id
                != "ordered_category_distribution_availability_publication_bundle_v2"
            ),
            planner_parent_output_role_groups=(
                _SEALED_RENDERER_PARENT_OUTPUT_ROLE_GROUPS.get(repair_id, ())
            ),
            implementation_modules=_SEALED_RENDERER_IMPLEMENTATION_MODULES.get(
                repair_id, ()
            ),
        )
        for repair_id in _STRUCTURAL_REPAIRS
    },
    **{
        repair_id: _meta(
            repair_id,
            RepairClass.CONTRACT_FILL,
            invariants=CONTRACT_FILL_INVARIANTS,
            selection_rule_required=True,
        )
        for repair_id in _CONTRACT_FILL_REPAIRS
    },
    **{
        repair_id: _meta(
            repair_id,
            RepairClass.METHOD_SUBSTITUTION,
            invariants=METHOD_SUBSTITUTION_INVARIANTS,
            introduces_numbers=True,
            requires_disclosure=True,
        )
        for repair_id in _METHOD_SUBSTITUTION_REPAIRS
    },
}


_PATTERN_METADATA: Tuple[Tuple[str, RepairMetadata], ...] = (
    (
        "strip_fake_easyicu_import_",
        _meta("strip_fake_easyicu_import_*_v1", RepairClass.SYNTACTIC),
    ),
    (
        "undefined_helper_stub_",
        _meta(
            "undefined_helper_stub_*_v1",
            RepairClass.METHOD_SUBSTITUTION,
            invariants=METHOD_SUBSTITUTION_INVARIANTS,
            introduces_numbers=True,
            requires_disclosure=True,
        ),
    ),
)


def repair_metadata_for(repair_id: str) -> RepairMetadata:
    """Return metadata for ``repair_id``.

    Unknown repairs are classified conservatively as method substitutions.  P0
    records them without changing soft-mode behavior; P2 can then block them
    in strict mode rather than letting an unreviewed repair silently through.
    """

    if repair_id in REPAIR_METADATA:
        return REPAIR_METADATA[repair_id]
    for prefix, metadata in _PATTERN_METADATA:
        if repair_id.startswith(prefix):
            return RepairMetadata(
                repair_id=repair_id,
                repair_class=metadata.repair_class,
                invariants=metadata.invariants,
                introduces_numbers=metadata.introduces_numbers,
                requires_disclosure=metadata.requires_disclosure,
                selection_rule_required=metadata.selection_rule_required,
                execution_policy=metadata.execution_policy,
                figure_product_slots=metadata.figure_product_slots,
                planner_methods=metadata.planner_methods,
                planner_method_required=metadata.planner_method_required,
                planner_parent_output_role_groups=(
                    metadata.planner_parent_output_role_groups
                ),
                implementation_modules=metadata.implementation_modules,
                description=metadata.description,
                classification_source=f"pattern:{metadata.repair_id}",
            )
    return RepairMetadata(
        repair_id=repair_id,
        repair_class=RepairClass.METHOD_SUBSTITUTION,
        invariants=METHOD_SUBSTITUTION_INVARIANTS,
        introduces_numbers=True,
        requires_disclosure=True,
        description="Conservative fallback for unclassified repair id.",
        classification_source="fallback:unknown_method_substitution",
    )


def automatic_repair_allowed(
    repair_id: str,
    *,
    step: Any = None,
    sealed_renderer_wrapper: bool = False,
) -> bool:
    """Whether a deterministic repair may run without analyst authorization.

    Unknown IDs inherit the conservative METHOD_SUBSTITUTION classification and
    are denied. ``step`` is accepted so this stays the single policy boundary
    if a future typed AuxiliaryRunner is authorized.
    """

    del step
    metadata = repair_metadata_for(repair_id)
    if (
        metadata.execution_policy is RepairExecutionPolicy.SEALED_RENDERER
        and not sealed_renderer_wrapper
    ):
        return False
    return bool(
        metadata.repair_class is not RepairClass.METHOD_SUBSTITUTION
        or repair_id in AUTOMATIC_METHOD_SUBSTITUTION_ALLOWLIST
    )


def is_sealed_renderer_repair(repair_id: str) -> bool:
    """Return whether *repair_id* is an immutable closed-source renderer."""

    metadata = repair_metadata_for(repair_id)
    return (
        metadata.classification_source == "exact"
        and metadata.repair_class is RepairClass.STRUCTURAL
        and metadata.execution_policy is RepairExecutionPolicy.SEALED_RENDERER
    )


def sealed_renderer_metadata() -> Tuple[RepairMetadata, ...]:
    """Return the closed renderer registry in deterministic id order."""

    return tuple(
        REPAIR_METADATA[repair_id] for repair_id in sorted(_SEALED_RENDERER_REPAIRS)
    )


def assert_repair_metadata_invariants(metadata: RepairMetadata) -> None:
    """Assert class-level integrity invariants for one repair declaration."""

    if metadata.repair_class == RepairClass.METHOD_SUBSTITUTION:
        if not metadata.introduces_numbers or not metadata.requires_disclosure:
            raise AssertionError(
                f"{metadata.repair_id} is METHOD_SUBSTITUTION but is not "
                "marked as introducing numbers and requiring disclosure."
            )
    if metadata.repair_class == RepairClass.CONTRACT_FILL:
        if not metadata.selection_rule_required:
            raise AssertionError(
                f"{metadata.repair_id} is CONTRACT_FILL but does not require "
                "a deterministic selection rule."
            )
    if metadata.repair_class == RepairClass.STRUCTURAL and not metadata.invariants:
        raise AssertionError(
            f"{metadata.repair_id} is STRUCTURAL but declares no invariants."
        )
    if metadata.execution_policy is RepairExecutionPolicy.SEALED_RENDERER:
        if metadata.repair_class is not RepairClass.STRUCTURAL:
            raise AssertionError(
                f"{metadata.repair_id} is a sealed renderer but is not STRUCTURAL."
            )
        if metadata.introduces_numbers:
            raise AssertionError(
                f"{metadata.repair_id} is a sealed renderer but introduces numbers."
            )
        if not metadata.figure_product_slots or len(
            metadata.figure_product_slots
        ) != len(set(metadata.figure_product_slots)):
            raise AssertionError(
                f"{metadata.repair_id} is a sealed renderer without unique "
                "figure product slots."
            )
        if metadata.planner_method_required and (
            not metadata.planner_methods
            or len(metadata.planner_methods) != len(set(metadata.planner_methods))
        ):
            raise AssertionError(
                f"{metadata.repair_id} is a sealed renderer without unique "
                "Planner methods."
            )
        if not metadata.planner_method_required and metadata.planner_methods:
            raise AssertionError(
                f"{metadata.repair_id} disables method routing but still declares "
                "Planner methods."
            )
        if not metadata.planner_parent_output_role_groups or any(
            not alternatives
            for alternatives in metadata.planner_parent_output_role_groups
        ):
            raise AssertionError(
                f"{metadata.repair_id} is a sealed renderer without complete "
                "Planner parent-output roles."
            )
        if not metadata.implementation_modules or len(
            metadata.implementation_modules
        ) != len(set(metadata.implementation_modules)):
            raise AssertionError(
                f"{metadata.repair_id} is a sealed renderer without unique "
                "implementation modules."
            )


def assert_registry_invariants() -> None:
    """Assert static invariants for every explicitly registered repair."""

    for metadata in REPAIR_METADATA.values():
        assert_repair_metadata_invariants(metadata)


def _hash_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_repair_provenance(
    *,
    repair_id: str,
    step_id: Optional[str],
    trigger: Optional[Dict[str, Any]] = None,
    transformation: Optional[str] = None,
    outcome: str = "applied",
    model_id: Optional[str] = None,
    before_text: Optional[str] = None,
    after_text: Optional[str] = None,
    selection_rule: Optional[str] = None,
    before_state: Optional[RepairObservedState] = None,
    after_state: Optional[RepairObservedState] = None,
) -> RepairProvenance:
    """Build a provenance record for a repair application.

    Invariants are evaluated at runtime against the supplied observable state;
    the result is recorded honestly (``verified_pass`` / ``verified_fail`` /
    ``unverified``).  ``invariants_passed`` is derived from that evaluation and
    is ``None`` when nothing could be checked — it is never defaulted to a pass.
    """

    metadata = repair_metadata_for(repair_id)
    assert_repair_metadata_invariants(metadata)
    evaluation = evaluate_invariants(
        metadata,
        before_state=before_state,
        after_state=after_state,
        selection_rule=selection_rule,
    )
    return RepairProvenance(
        repair_id=repair_id,
        repair_class=metadata.repair_class.value,
        step_id=step_id,
        trigger=trigger or {},
        transformation=transformation or metadata.description or repair_id,
        invariants_checked=evaluation.checked,
        invariants_passed=evaluation.passed,
        invariant_status=evaluation.status,
        invariant_failures=evaluation.failures,
        introduces_numbers=metadata.introduces_numbers,
        requires_disclosure=metadata.requires_disclosure,
        selection_rule=selection_rule,
        outcome=outcome,
        model_id=model_id,
        applied_at=datetime.now(timezone.utc).isoformat(),
        before_hash=_hash_text(before_text),
        after_hash=_hash_text(after_text),
        classification_source=metadata.classification_source,
    )


class RepairLedger:
    """Append-only JSON ledger for repair provenance records."""

    schema_version = "easyicu.repair_ledger/1"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: List[RepairProvenance] = []
        self.write()

    @property
    def records(self) -> Tuple[RepairProvenance, ...]:
        return tuple(self._records)

    def append(self, provenance: RepairProvenance) -> None:
        self._records.append(provenance)
        self.write()

    def append_application(
        self,
        *,
        repair_id: str,
        step_id: Optional[str],
        trigger: Optional[Dict[str, Any]] = None,
        transformation: Optional[str] = None,
        outcome: str = "applied",
        model_id: Optional[str] = None,
        before_text: Optional[str] = None,
        after_text: Optional[str] = None,
        selection_rule: Optional[str] = None,
        before_state: Optional[RepairObservedState] = None,
        after_state: Optional[RepairObservedState] = None,
    ) -> RepairProvenance:
        provenance = make_repair_provenance(
            repair_id=repair_id,
            step_id=step_id,
            trigger=trigger,
            transformation=transformation,
            outcome=outcome,
            model_id=model_id,
            before_text=before_text,
            after_text=after_text,
            selection_rule=selection_rule,
            before_state=before_state,
            after_state=after_state,
        )
        self.append(provenance)
        return provenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repairs": [asdict(record) for record in self._records],
        }

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


__all__ = [
    "InvariantEvaluation",
    "InvariantStatus",
    "Repair",
    "RepairClass",
    "RepairExecutionPolicy",
    "RepairLedger",
    "RepairMetadata",
    "RepairObservedState",
    "RepairProvenance",
    "REPAIR_METADATA",
    "assert_registry_invariants",
    "assert_repair_metadata_invariants",
    "evaluate_invariants",
    "is_sealed_renderer_repair",
    "make_repair_provenance",
    "repair_metadata_for",
    "sealed_renderer_metadata",
]
