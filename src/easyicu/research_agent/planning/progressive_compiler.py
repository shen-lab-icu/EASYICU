"""Compile a Progressive Planner skeleton into the existing AnalysisPlan.

The compiler owns only deterministic materialization.  It never relaxes an
``AnalysisPlan`` validator and it never invents an estimand.  Ambiguous choices
fail with a stable owner/reason/step coordinate so the Planner can replace the
unlocked suffix instead of rewriting an already-valid prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from pydantic import ValidationError

from ..authority.declared_levels import observed_levels_for
from ..canonical_json import canonical_sha256
from ..cohort.schema import materialized_input_column_authority
from ..contracts.declared_product import PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS
from ..contracts.claim_ceiling import DescriptiveClaimContract
from ..contracts.model_terms import ModelTermSpec, level_spelling
from ..contracts.model_tokens import (
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
)
from ..contracts.product_identity import typed_product
from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    CohortDefinitionSpec,
    CohortEligibilityCriterion,
    ExposureOutcomeDistributionSpec,
    ExposureOutcomeRiskDifferenceContrast,
    MeasurementAuditProduct,
    MeasurementAuditSpec,
    KnowHowDecision,
    PlannedModelRequirement,
    ResearchContext,
    RobustnessReplayProduct,
    RobustnessReplaySpec,
    TableOneSpec,
    TableOneVariableSpec,
)
from .analysis_types import (
    canonical_analysis_family,
    get_analysis_type,
    validate_host_authorized_analysis_family,
)
from .cohort_contract import (
    CohortDefinition,
    CohortSchemaError,
    ConceptPredicate,
    TimeWindow,
    cohort_concept_id_scope,
    validate_cohort_definition,
)
from .dependence_authority import context_counts_only_authority
from .literature_contract import LiteratureDesignBinding
from .method_literature import method_binding_support
from .progressive_contract import (
    PROGRESSIVE_HOST_COMPILED_OUTPUTS,
    ProgressiveCompiledStepReceipt,
    ProgressiveCohortIntent,
    ProgressiveLiteratureBinding,
    ProgressivePlanCompileError,
    ProgressivePlanCompileReceipt,
    ProgressivePlanFoundation,
    ProgressivePlanSkeleton,
    ProgressiveSkeletonStep,
    progressive_module_ids_for_analysis_types,
)
from .robustness_contract import RobustnessSpec
from .preplan_know_how import verify_know_how_decisions
from .scientific_action_catalog import (
    ScientificActionGapError,
    scientific_action_for_id,
    validate_plan_scientific_action_selections,
)
from .scientific_review import post_baseline_exposure


_OWNER = "easyicu.planning.progressive_compiler_v1"
_MODULE_OUTPUT_ROLES: Mapping[str, frozenset[str]] = {
    "cohort_definition": frozenset({"analysis_cohort", "cohort_flow"}),
    "table_one": frozenset({"table_one"}),
    "exposure_outcome_distribution": frozenset({"exposure_outcome_distribution"}),
    "measurement_audit": frozenset(
        {
            "measurement_missingness",
            "missingness_profile",
            "measurement_source",
            "measurement_process",
            "event_timing",
            "component_completeness",
            "analytic_denominators",
        }
    ),
    "adjusted_association": frozenset(
        {"adjusted_association_estimates", "adjusted_association_model"}
    ),
    "robustness_replay": frozenset(
        {
            "robustness_matrix",
            "robustness_summary",
            "specification_grid",
            "membership_change",
            "outcome_label_executability",
            "missingness_strategy_notes",
            "primary_effect",
            "complete_case_n",
        }
    ),
    "custom_analysis": frozenset({"scientific_sensitivity", "custom"}),
    "visualization": frozenset({"figure"}),
    "report": frozenset({"report"}),
}
_METHOD_BY_MODULE: Mapping[str, str] = {
    "cohort_definition": "cohort_definition_and_attrition",
    "table_one": "table_one",
    "exposure_outcome_distribution": "descriptive",
    "measurement_audit": "missing_data",
    "adjusted_association": "adjusted_association_models",
    "robustness_replay": "robustness_sensitivity",
    "visualization": "visualization",
    "report": "feasibility_protocol",
}
_COHORT_FRAME_ONLY_MODULES = frozenset(
    {
        "cohort_definition",
        "table_one",
        "exposure_outcome_distribution",
        "measurement_audit",
        "adjusted_association",
    }
)


@dataclass(frozen=True)
class _CompiledStep:
    step: AnalysisStep
    outputs: tuple[str, ...]


def _fail(
    code: str,
    message: str,
    *,
    step: ProgressiveSkeletonStep | None = None,
    step_index: int | None = None,
    path: str | None = None,
) -> ProgressivePlanCompileError:
    return ProgressivePlanCompileError(
        code,
        message,
        step_id=step.step_id if step is not None else None,
        step_index=step_index,
        path=path,
    )


def _variable_index(context: ResearchContext) -> dict[str, Any]:
    return {variable.name: variable for variable in context.variables}


def progressive_cohort_concept_ids(
    context: ResearchContext,
    variable_names: Sequence[str],
) -> tuple[str, ...]:
    """Return the sealed cohort concepts and materialized columns for a run.

    Foundation transport and deterministic plan validation must use this same
    authority. ResearchContext variable names are physical columns in the
    sealed analysis input; source and derivation concepts remain eligible when
    the dictionary exposes them through those selected variables.
    """

    selected = set(variable_names)
    values: list[str] = list(variable_names)
    for variable in context.variables:
        if variable.name not in selected:
            continue
        values.extend(
            str(value).strip()
            for value in (
                variable.source_concept,
                *variable.derived_from_concepts,
            )
            if str(value or "").strip()
        )
    return tuple(dict.fromkeys(values))


def _context_cohort_concept_ids(context: ResearchContext) -> tuple[str, ...]:
    return progressive_cohort_concept_ids(
        context,
        tuple(variable.name for variable in context.variables),
    )


def _require_variables(
    names: Iterable[str],
    *,
    variables: Mapping[str, Any],
    step: ProgressiveSkeletonStep,
    step_index: int,
    path: str,
) -> list[str]:
    cleaned = list(dict.fromkeys(str(name or "").strip() for name in names))
    missing = [name for name in cleaned if name not in variables]
    if missing:
        raise _fail(
            "progressive_unknown_variable",
            f"variables are absent from the sealed ResearchContext: {missing!r}",
            step=step,
            step_index=step_index,
            path=path,
        )
    return cleaned


def _compile_cohort_intent(
    cohort_intent: ProgressiveCohortIntent,
) -> CohortDefinition:
    def predicate(item: Any) -> ConceptPredicate:
        return ConceptPredicate(
            concept_id=item.concept_id,
            time_window=TimeWindow(
                anchor=item.anchor,
                start_offset_hours=item.start_offset_hours,
                end_offset_hours=item.end_offset_hours,
            ),
            aggregation=item.aggregation,
            op=item.op,
            value=item.value.materialize(),
        )

    return CohortDefinition(
        name=cohort_intent.name,
        inclusion=tuple(predicate(item) for item in cohort_intent.inclusion),
        exclusion=tuple(predicate(item) for item in cohort_intent.exclusion),
        selection_mode=cohort_intent.selection_mode,
    )


def _validate_progressive_cohort_intent(
    cohort_intent: ProgressiveCohortIntent,
    *,
    context: ResearchContext,
) -> CohortDefinition:
    try:
        with cohort_concept_id_scope(_context_cohort_concept_ids(context)):
            cohort = _compile_cohort_intent(cohort_intent)
            validate_cohort_definition(cohort)
    except CohortSchemaError as exc:
        raise _fail(
            "progressive_foundation_cohort_invalid",
            str(exc),
            path="cohort",
        ) from exc
    return cohort


def validate_progressive_foundation(
    foundation: ProgressivePlanFoundation,
    *,
    context: ResearchContext,
    analysis_type: str,
) -> None:
    """Fail before step generation when a sealed Foundation cannot compile."""

    _validate_progressive_cohort_intent(foundation.cohort, context=context)
    if (
        str(analysis_type or "").strip().casefold() == "descriptive_epidemiology"
        and foundation.robustness_intents
    ):
        raise _fail(
            "progressive_descriptive_robustness_unavailable",
            "descriptive_epidemiology has no fitted primary effect or interval; "
            "use typed measurement and denominator audits instead of "
            "effect-style robustness intents",
            path="robustness_intents",
        )
    variables = {item.name: item for item in context.variables}
    for intent in foundation.robustness_intents:
        if intent.missing_strategy != "complete_case":
            raise _fail(
                "progressive_robustness_intent_not_replayable",
                "the progressive v1 robustness replay can compile only an "
                "explicit complete-case missing-data specification; timing, "
                "cohort/readmission, outcome-definition, and functional-form "
                "analyses require a separate custom_analysis step",
                path=f"robustness_intents.{intent.spec_id}",
            )
        conditional_event_times = [
            name
            for name in intent.complete_case_variables
            if (
                (descriptor := variables.get(name)) is not None
                and descriptor.observation_semantics is not None
                and descriptor.observation_semantics.kind
                == "conditional_event_time"
            )
        ]
        if conditional_event_times:
            raise _fail(
                "progressive_complete_case_includes_not_applicable_time",
                "complete-case membership cannot require conditional event-time "
                "columns whose event-absent rows are typed as not applicable: "
                + ", ".join(conditional_event_times),
                path=(
                    f"robustness_intents.{intent.spec_id}."
                    "complete_case_variables"
                ),
            )


def _compile_robustness_intents(
    skeleton: ProgressivePlanSkeleton,
    *,
    context: ResearchContext,
    variables: Mapping[str, Any],
) -> list[RobustnessSpec]:
    compiled: list[RobustnessSpec] = []
    reserved_coordinates = set(
        materialized_input_column_authority(context).reserved_navigation_coordinates
    )
    primary_model_variables = tuple(
        dict.fromkeys(
            name
            for step in skeleton.steps
            if (
                step.module_id == "adjusted_association"
                and step.planned_analysis_role == "primary"
            )
            for name in (
                step.primary_exposure,
                step.outcome,
                *(term.name for term in step.model_terms),
            )
            if name
        )
    )
    for item in skeleton.robustness_intents:
        missing_override: Optional[dict[str, Any]] = None
        if item.missing_strategy == "complete_case":
            complete_case_variables = list(
                dict.fromkeys(
                    name
                    for name in (
                        *item.complete_case_variables,
                        *primary_model_variables,
                    )
                    if name not in reserved_coordinates
                )
            )
            missing = [
                name for name in complete_case_variables if name not in variables
            ]
            if missing:
                raise _fail(
                    "progressive_unknown_robustness_variable",
                    f"complete-case variables are not in ResearchContext: {missing!r}",
                    path=f"robustness_intents.{item.spec_id}.complete_case_variables",
                )
            missing_override = {
                "strategy": "complete_case",
                "variables": complete_case_variables,
                "audit_flags": None,
            }
        compiled.append(
            RobustnessSpec(
                spec_id=item.spec_id,
                axis=item.axis,
                description=item.description,
                missing_override=missing_override,
            )
        )
    return compiled


def _canonical_outputs(step: ProgressiveSkeletonStep) -> list[tuple[str, str]]:
    standard = list(PROGRESSIVE_HOST_COMPILED_OUTPUTS.get(step.module_id, ()))
    declared = [(item.product_id, item.semantic_role) for item in step.outputs]
    by_product: dict[str, str] = {}
    for product, role in [*standard, *declared]:
        prior = by_product.get(product)
        if prior is not None and prior != role:
            raise ValueError(
                f"product {product!r} has conflicting roles {prior!r} and {role!r}"
            )
        by_product[product] = role
    return list(by_product.items())


def _validate_outputs(
    outputs: Sequence[tuple[str, str]],
    *,
    step: ProgressiveSkeletonStep,
    step_index: int,
) -> None:
    allowed_roles = _MODULE_OUTPUT_ROLES[step.module_id]
    for product_id, semantic_role in outputs:
        parsed = typed_product(product_id)
        if parsed is None or parsed[0] not in PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS:
            raise _fail(
                "progressive_unsupported_product",
                f"output {product_id!r} is not materializable by the runtime",
                step=step,
                step_index=step_index,
                path="outputs",
            )
        if semantic_role not in allowed_roles:
            raise _fail(
                "progressive_output_role_mismatch",
                f"semantic role {semantic_role!r} does not belong to module "
                f"{step.module_id!r}; allowed={sorted(allowed_roles)!r}",
                step=step,
                step_index=step_index,
                path="outputs",
            )
        kind = parsed[0]
        if step.module_id == "measurement_audit" and kind != "table":
            raise _fail(
                "progressive_measurement_product_kind",
                "measurement audit products must be table products",
                step=step,
                step_index=step_index,
                path="outputs",
            )
        if step.module_id == "visualization" and kind != "figure":
            raise _fail(
                "progressive_figure_product_kind",
                "visualization outputs must use kind 'figure'",
                step=step,
                step_index=step_index,
                path="outputs",
            )
        if step.module_id == "report" and kind != "report":
            raise _fail(
                "progressive_report_product_kind",
                "report module outputs must use kind 'report'",
                step=step,
                step_index=step_index,
                path="outputs",
            )


def _identity_column(
    *,
    context: ResearchContext,
    step: ProgressiveSkeletonStep,
    step_index: int,
) -> str:
    declared = [name for name in step.raw_inputs if name in context.cohort.id_columns]
    if len(declared) == 1:
        return declared[0]
    ids = list(context.cohort.id_columns)
    if len(ids) == 1:
        return ids[0]
    raise _fail(
        "progressive_identity_column_ambiguous",
        "cohort definition needs exactly one stable row identity; declare one "
        f"of {ids!r} in raw_inputs",
        step=step,
        step_index=step_index,
        path="raw_inputs",
    )


def _eligibility_criteria(
    context: ResearchContext,
    skeleton: ProgressivePlanSkeleton,
) -> list[CohortEligibilityCriterion]:
    descriptions = [
        *[str(value) for value in context.cohort.inclusion_criteria],
        *[f"Exclude: {value}" for value in context.cohort.exclusion_criteria],
    ]
    if not descriptions and skeleton.cohort.selection_mode == "predicate_filtered":
        for group, items in (
            ("include", skeleton.cohort.inclusion),
            ("exclude", skeleton.cohort.exclusion),
        ):
            for item in items:
                descriptions.append(
                    f"{group} {item.concept_id} {item.op} within "
                    f"{item.anchor}[{item.start_offset_hours:g},"
                    f"{item.end_offset_hours:g})h using {item.aggregation}"
                )
    return [
        CohortEligibilityCriterion(
            criterion_id=f"criterion_{index:02d}",
            description=description,
        )
        for index, description in enumerate(descriptions, start=1)
    ]


def _table_one_variable_kind(variable: Any, levels: Sequence[Any]) -> str:
    if variable.is_ordinal:
        return "ordinal"
    dtype = str(variable.dtype or "").lower()
    if levels and (
        len(levels) == 2
        or dtype.startswith(("object", "str", "string", "category", "bool"))
    ):
        return "categorical"
    return "continuous"


def _compile_table_one(
    *,
    context: ResearchContext,
    variables: Mapping[str, Any],
    step: ProgressiveSkeletonStep,
    step_index: int,
) -> TableOneSpec:
    group_by = str(step.table_one_group_by or "")
    row_intents = [
        item for item in step.table_one_variables if item.name != group_by
    ]
    if not row_intents:
        raise _fail(
            "progressive_table_one_rows_missing",
            "Table 1 needs at least one row variable distinct from group_by",
            step=step,
            step_index=step_index,
            path="table_one_variables",
        )
    _require_variables(
        [group_by, *(item.name for item in row_intents)],
        variables=variables,
        step=step,
        step_index=step_index,
        path="table_one_variables",
    )
    group_levels = observed_levels_for(name=group_by, variables=dict(variables))
    if len(group_levels) < 2:
        raise _fail(
            "progressive_table_one_group_levels_unavailable",
            f"host has no closed two-level-or-more domain for {group_by!r}",
            step=step,
            step_index=step_index,
            path="table_one_group_by",
        )
    rows: list[TableOneVariableSpec] = []
    for index, item in enumerate(row_intents):
        variable = variables[item.name]
        levels = observed_levels_for(name=item.name, variables=dict(variables))
        kind = _table_one_variable_kind(variable, levels)
        if kind == "continuous" and item.summary == "count_percent":
            raise _fail(
                "progressive_table_one_summary_incompatible",
                f"continuous variable {item.name!r} cannot use count_percent",
                step=step,
                step_index=step_index,
                path=f"table_one_variables[{index}]",
            )
        if kind == "categorical" and item.summary != "count_percent":
            raise _fail(
                "progressive_table_one_summary_incompatible",
                f"categorical variable {item.name!r} requires count_percent",
                step=step,
                step_index=step_index,
                path=f"table_one_variables[{index}]",
            )
        if kind in {"categorical", "ordinal"} and item.summary == "count_percent":
            if len(levels) < 2:
                raise _fail(
                    "progressive_table_one_levels_unavailable",
                    f"closed levels are unavailable for {item.name!r}",
                    step=step,
                    step_index=step_index,
                    path=f"table_one_variables[{index}]",
                )
            row_levels = list(levels)
        else:
            row_levels = list(levels) if kind == "ordinal" and levels else []
        if step.table_one_mode == "descriptive_smd_only":
            test = "none_descriptive_smd_only"
        elif item.summary in {"mean_sd", "both"}:
            test = "welch_t_or_anova"
        elif item.summary == "median_iqr":
            test = "mann_whitney_or_kruskal"
        else:
            test = "chi_square_with_fisher_exact_for_sparse_2x2"
        rows.append(
            TableOneVariableSpec(
                name=item.name,
                variable_kind=kind,
                summary=item.summary,
                test=test,
                levels=row_levels,
            )
        )
    descriptive = step.table_one_mode == "descriptive_smd_only"
    try:
        return TableOneSpec(
            schema_version=(
                "easyicu.table_one/2" if descriptive else "easyicu.table_one/1"
            ),
            group_by=group_by,
            group_levels=list(group_levels),
            variables=rows,
            include_overall=True,
            missing_group_policy="fail_closed",
            missingness_display="n_percent_by_group",
            p_values_required=not descriptive,
            p_value_adjustment=(
                "not_applicable_repeated_units"
                if descriptive
                else "none_descriptive_table"
            ),
            standardized_difference_mode="auto_binary_groups",
        )
    except ValidationError as exc:
        finding = exc.errors(include_input=False)[0]
        raise _fail(
            "progressive_table_one_contract_invalid",
            f"compiled Table 1 violates its typed contract: {finding['msg']}",
            step=step,
            step_index=step_index,
            path="table_one_variables",
        ) from exc


def _level_at(
    levels: Sequence[Any],
    index: int | None,
    *,
    label: str,
    step: ProgressiveSkeletonStep,
    step_index: int,
) -> Any:
    if index is None or index >= len(levels):
        raise _fail(
            "progressive_level_index_out_of_range",
            f"{label} index {index!r} is outside closed domain of size {len(levels)}",
            step=step,
            step_index=step_index,
            path=label,
        )
    return levels[index]


def _compile_distribution(
    *,
    variables: Mapping[str, Any],
    step: ProgressiveSkeletonStep,
    step_index: int,
    counts_only: bool = False,
) -> ExposureOutcomeDistributionSpec:
    exposure = str(step.primary_exposure or "")
    outcome = str(step.outcome or "")
    _require_variables(
        [exposure, outcome],
        variables=variables,
        step=step,
        step_index=step_index,
        path="distribution_variables",
    )
    exposure_levels = observed_levels_for(name=exposure, variables=dict(variables))
    outcome_levels = observed_levels_for(name=outcome, variables=dict(variables))
    if len(exposure_levels) < 2 or len(outcome_levels) < 2:
        raise _fail(
            "progressive_distribution_levels_unavailable",
            "exposure and outcome both require closed domains with at least two levels",
            step=step,
            step_index=step_index,
            path="distribution_variables",
        )
    event = _level_at(
        outcome_levels,
        step.event_level_index,
        label="event_level_index",
        step=step,
        step_index=step_index,
    )
    if counts_only:
        # The typed StudyContext has already forbidden uncertainty and effect
        # contrasts. Compile only the observed denominators, counts, and
        # proportions; model-supplied contrast indexes carry no authority.
        return ExposureOutcomeDistributionSpec(
            schema_version="easyicu.exposure_outcome_distribution/3",
            exposure=exposure,
            exposure_levels=list(exposure_levels),
            outcome=outcome,
            outcome_levels=list(outcome_levels),
            outcome_positive_value=event,
            level_match_policy="exact_typed",
            denominator_policy=step.denominator_policy,
            missing_exposure_policy=step.missing_exposure_policy,
            missing_outcome_policy=step.missing_outcome_policy,
            undeclared_outcome_policy="fail_closed",
            interval_method="none_counts_only",
            repeated_unit_interval_method=None,
            risk_difference_contrast=None,
            dependence=None,
            confidence_level=None,
        )
    reference = _level_at(
        exposure_levels,
        step.reference_exposure_level_index,
        label="reference_exposure_level_index",
        step=step,
        step_index=step_index,
    )
    comparison = _level_at(
        exposure_levels,
        step.comparison_exposure_level_index,
        label="comparison_exposure_level_index",
        step=step,
        step_index=step_index,
    )
    if (
        step.reference_exposure_level_index
        == step.comparison_exposure_level_index
    ):
        raise _fail(
            "progressive_distribution_contrast_not_distinct",
            "risk-difference comparison and reference levels must differ",
            step=step,
            step_index=step_index,
            path="comparison_exposure_level_index",
        )
    try:
        return ExposureOutcomeDistributionSpec(
            schema_version="easyicu.exposure_outcome_distribution/2",
            exposure=exposure,
            exposure_levels=list(exposure_levels),
            outcome=outcome,
            outcome_levels=list(outcome_levels),
            outcome_positive_value=event,
            level_match_policy="exact_typed",
            denominator_policy=step.denominator_policy,
            missing_exposure_policy=step.missing_exposure_policy,
            missing_outcome_policy=step.missing_outcome_policy,
            undeclared_outcome_policy="fail_closed",
            interval_method="wilson",
            repeated_unit_interval_method="patient_cluster_robust_wald",
            risk_difference_contrast=ExposureOutcomeRiskDifferenceContrast(
                reference_exposure_level=reference,
                comparison_exposure_level=comparison,
            ),
            confidence_level=step.confidence_level,
        )
    except ValidationError as exc:
        finding = exc.errors(include_input=False)[0]
        field = ".".join(str(value) for value in finding["loc"])
        raise _fail(
            "progressive_distribution_spec_invalid",
            "compiled exposure/outcome distribution violates its typed "
            f"contract: {finding['msg']}",
            step=step,
            step_index=step_index,
            path=field or "exposure_outcome_distribution",
        ) from exc


def _compile_model_terms(
    *,
    variables: Mapping[str, Any],
    step: ProgressiveSkeletonStep,
    step_index: int,
) -> tuple[list[ModelTermSpec], list[str], list[str], str, str]:
    names = [item.name for item in step.model_terms]
    _require_variables(
        names,
        variables=variables,
        step=step,
        step_index=step_index,
        path="model_terms",
    )
    if len(names) != len(set(names)):
        raise _fail(
            "progressive_model_term_duplicate",
            "model term names must be unique",
            step=step,
            step_index=step_index,
            path="model_terms",
        )
    compiled: list[ModelTermSpec] = []
    for index, item in enumerate(step.model_terms):
        observed = observed_levels_for(name=item.name, variables=dict(variables))
        if item.coding == "continuous":
            levels = None
            reference = None
            transform = "identity"
        elif item.coding == "ordinal_linear":
            if len(observed) < 2:
                raise _fail(
                    "progressive_model_levels_unavailable",
                    f"ordinal term {item.name!r} has no closed ordered domain",
                    step=step,
                    step_index=step_index,
                    path=f"model_terms[{index}]",
                )
            levels = [level_spelling(value) for value in observed]
            reference = None
            transform = "declared_level_index"
        else:
            minimum = 2
            if len(observed) < minimum or (
                item.coding == "binary" and len(observed) != 2
            ):
                raise _fail(
                    "progressive_model_levels_unavailable",
                    f"{item.coding} term {item.name!r} has incompatible closed domain",
                    step=step,
                    step_index=step_index,
                    path=f"model_terms[{index}]",
                )
            levels = [level_spelling(value) for value in observed]
            reference_value = _level_at(
                observed,
                item.reference_level_index,
                label=f"model_terms[{index}].reference_level_index",
                step=step,
                step_index=step_index,
            )
            reference = level_spelling(reference_value)
            transform = "treatment_contrast"
        compiled.append(
            ModelTermSpec(
                name=item.name,
                role=item.role,
                coding=item.coding,
                levels=levels,
                reference_level=reference,
                transform=transform,
            )
        )
    exposures = [item for item in compiled if item.role == "exposure"]
    if len(exposures) != 1 or exposures[0].name != step.primary_exposure:
        raise _fail(
            "progressive_primary_exposure_term_mismatch",
            "model terms require exactly one exposure matching primary_exposure",
            step=step,
            step_index=step_index,
            path="model_terms",
        )
    covariates = [item.name for item in compiled if item.role == "covariate"]
    exposure_term = exposures[0]
    exposure_levels = list(exposure_term.levels or ())
    reference = str(exposure_term.reference_level or "")
    if exposure_levels:
        if len(exposure_levels) == 2:
            primary_contrast = next(
                value for value in exposure_levels if value != reference
            )
        else:
            observed = observed_levels_for(
                name=exposure_term.name, variables=dict(variables)
            )
            value = _level_at(
                observed,
                step.primary_contrast_level_index,
                label="primary_contrast_level_index",
                step=step,
                step_index=step_index,
            )
            primary_contrast = level_spelling(value)
            if primary_contrast == reference:
                raise _fail(
                    "progressive_primary_contrast_is_reference",
                    "primary contrast must differ from the exposure reference",
                    step=step,
                    step_index=step_index,
                    path="primary_contrast_level_index",
                )
    else:
        primary_contrast = ""
    return compiled, covariates, exposure_levels, reference, primary_contrast


def _compile_adjusted_association(
    *,
    variables: Mapping[str, Any],
    step: ProgressiveSkeletonStep,
    step_index: int,
) -> list[PlannedModelRequirement]:
    exposure = str(step.primary_exposure or "")
    outcome = str(step.outcome or "")
    _require_variables(
        [exposure, outcome],
        variables=variables,
        step=step,
        step_index=step_index,
        path="adjusted_association",
    )
    terms, covariates, exposure_levels, reference, primary_contrast = (
        _compile_model_terms(
            variables=variables,
            step=step,
            step_index=step_index,
        )
    )
    method_family = (
        ASSOCIATION_LOGIT_ESTIMATOR
        if step.outcome_type == "binary"
        else ASSOCIATION_OLS_ESTIMATOR
    )
    return [
        PlannedModelRequirement(
            requirement_id=f"{step.step_id}_primary",
            outcome=outcome,
            outcome_type=step.outcome_type,
            method_family=method_family,
            exposure_source=exposure,
            analysis_role="primary",
            analysis_set="source_aware",
            required_for_step_success=True,
            covariates=covariates,
            model_terms=terms,
            exposure_levels=exposure_levels or None,
            exposure_reference_level=reference or None,
            primary_contrast_level=primary_contrast or None,
        )
    ]


def _compile_measurement_spec(
    outputs: Sequence[tuple[str, str]],
    *,
    step: ProgressiveSkeletonStep,
    step_index: int,
) -> MeasurementAuditSpec:
    try:
        return MeasurementAuditSpec(
            products=[
                MeasurementAuditProduct(
                    product_id=product_id.split(":", 1)[1],
                    audit=role,
                )
                for product_id, role in outputs
            ]
        )
    except ValidationError as exc:
        finding = exc.errors(include_input=False)[0]
        raise _fail(
            "progressive_measurement_audit_spec_invalid",
            "compiled measurement audit violates its typed contract: "
            f"{finding['msg']}",
            step=step,
            step_index=step_index,
            path="outputs",
        ) from exc


def _compile_robustness_spec(
    outputs: Sequence[tuple[str, str]],
    *,
    step: ProgressiveSkeletonStep,
    step_index: int,
) -> RobustnessReplaySpec:
    try:
        return RobustnessReplaySpec(
            products=[
                RobustnessReplayProduct(
                    product_id=product_id.split(":", 1)[1],
                    output=role,
                )
                for product_id, role in outputs
            ]
        )
    except ValidationError as exc:
        finding = exc.errors(include_input=False)[0]
        raise _fail(
            "progressive_robustness_replay_spec_invalid",
            "compiled robustness replay violates its typed contract: "
            f"{finding['msg']}",
            step=step,
            step_index=step_index,
            path="outputs",
        ) from exc


def _compile_inputs(
    *,
    context: ResearchContext,
    skeleton: ProgressivePlanSkeleton,
    step: ProgressiveSkeletonStep,
    step_index: int,
    variables: Mapping[str, Any],
    producers: Mapping[str, str],
    outputs_by_step: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[ArtifactConsumptionContract]]:
    reserved_coordinates = set(
        materialized_input_column_authority(context).reserved_navigation_coordinates
    )
    raw_names = list(step.raw_inputs)
    if step.module_id == "measurement_audit":
        # Observation semantics are host-verified context authority.  An audit
        # that receives only the model-selected representative column cannot
        # re-run the count/flag/status reconciliation and silently reports a
        # complete event status as ordinary measurement availability.  Carry
        # the small typed dependency closure mechanically; the model still
        # chooses whether a measurement-audit module belongs in the plan.
        for descriptor in context.variables:
            semantics = descriptor.observation_semantics
            if semantics is None:
                continue
            raw_names.extend(
                value
                for value in (
                    descriptor.name,
                    semantics.event_count_column,
                    semantics.measured_column,
                    semantics.representative_column,
                    semantics.event_status_column,
                )
                if value and value in variables
            )
    if (
        step.module_id == "custom_analysis"
        and step.planned_analysis_role == "sensitivity"
        and step.sensitivity_spec_ids
    ):
        primary_producer_ids = {
            reference.producer_step_id
            for reference in step.product_inputs
            if reference.product_id == "table:adjusted_association_estimates"
        }
        for upstream in skeleton.steps:
            if (
                upstream.step_id in primary_producer_ids
                and upstream.planned_analysis_role == "primary"
            ):
                raw_names.extend(upstream.raw_inputs)
    raw = _require_variables(
        [name for name in raw_names if name not in reserved_coordinates],
        variables=variables,
        step=step,
        step_index=step_index,
        path="raw_inputs",
    )
    inputs = list(raw)
    if (
        step.module_id != "cohort_definition"
        and "artifact:analysis_cohort" in producers
    ):
        inputs.append("artifact:analysis_cohort")
    refs = list(step.product_inputs)
    if step.module_id == "visualization" and not refs:
        for dependency in step.depends_on:
            for product_id in outputs_by_step.get(dependency, ()):
                parsed = typed_product(product_id)
                if parsed is not None and parsed[0] in {"table", "statistic"}:
                    inputs.append(product_id)
    for reference in refs:
        owner = producers.get(reference.product_id)
        if owner is None:
            raise _fail(
                "progressive_product_reference_mismatch",
                f"{reference.product_id!r} has no preceding host-registered "
                f"owner; declared producer was {reference.producer_step_id!r}",
                step=step,
                step_index=step_index,
                path="product_inputs",
            )
        # Product ids have one preceding owner by construction. Resolve that
        # owner from the host registry instead of making the model repeat an
        # already-known edge correctly in two fields. Exact host executors in
        # ``_COHORT_FRAME_ONLY_MODULES`` read only the sealed cohort plus their
        # declared raw columns/specification. An outline dependency can order
        # those steps, but its table/report product must not become a second
        # data-frame input that the executor neither reads nor receipts.
        if step.module_id not in _COHORT_FRAME_ONLY_MODULES:
            inputs.append(reference.product_id)
    inputs = list(dict.fromkeys(inputs))
    consumption = [
        ArtifactConsumptionContract(
            input_key=value,
            mode="all_rows",
            role_column=None,
            expected_roles=[],
        )
        for value in inputs
        if step.module_id == "visualization" and value.startswith("table:")
    ]
    return inputs, consumption


def _compile_literature(
    step: ProgressiveSkeletonStep,
    *,
    allowed_citations: frozenset[str],
    step_index: int,
    host_reporting_source_key: str | None = None,
    host_interpretation_source_key: str | None = None,
) -> tuple[list[str], list[LiteratureDesignBinding]]:
    bindings_by_key: dict[str, list[ProgressiveLiteratureBinding]] = {}
    for item in step.literature_bindings:
        bindings_by_key.setdefault(item.citation_key, []).append(item)
    host_bindings = (
        (
            host_reporting_source_key,
            ("reporting",),
            "Apply the host-sealed article reporting standard to this study's "
            "methods and results.",
        ),
        (
            host_interpretation_source_key,
            ("outcome",),
            "Report an absolute outcome measure alongside each model ratio "
            "estimate so interpretation is not ratio-only.",
        ),
    )
    for source_key, required_elements, application in host_bindings:
        if source_key is None:
            continue
        existing_elements = {
            element
            for binding in bindings_by_key.get(source_key, ())
            for element in binding.design_elements
        }
        missing_elements = [
            element for element in required_elements if element not in existing_elements
        ]
        if missing_elements:
            bindings_by_key.setdefault(source_key, []).append(
                ProgressiveLiteratureBinding(
                    citation_key=source_key,
                    design_elements=missing_elements,
                    application=application,
                    divergence=None,
                )
            )
    keys = list(bindings_by_key)
    unknown = sorted(set(keys) - allowed_citations)
    if unknown:
        raise _fail(
            "progressive_unknown_literature_source",
            f"citation keys are outside the sealed run roster: {unknown!r}",
            step=step,
            step_index=step_index,
            path="literature_bindings",
        )

    compiled: list[LiteratureDesignBinding] = []
    for citation_key, bindings in bindings_by_key.items():
        design_elements = list(
            dict.fromkeys(
                element for binding in bindings for element in binding.design_elements
            )
        )
        applications = list(dict.fromkeys(binding.application for binding in bindings))
        divergences = list(
            dict.fromkeys(
                binding.divergence
                for binding in bindings
                if binding.divergence is not None
            )
        )
        application = "\n".join(applications)
        divergence = "\n".join(divergences) if divergences else None
        try:
            binding = LiteratureDesignBinding(
                citation_key=citation_key,
                design_elements=design_elements,
                application=application,
                divergence=divergence,
            )
        except ValidationError as exc:
            finding = exc.errors(include_input=False)[0]
            field = ".".join(str(value) for value in finding["loc"])
            code = (
                "progressive_literature_merge_overflow"
                if finding["type"] == "string_too_long"
                else "progressive_literature_merge_invalid"
            )
            raise _fail(
                code,
                f"coalescing citation {citation_key!r} violates the {field} "
                f"contract: {finding['msg']}",
                step=step,
                step_index=step_index,
                path=f"literature_bindings.{field}",
            ) from exc
        compiled.append(binding)
    return keys, compiled


def _compile_one_step(
    *,
    context: ResearchContext,
    skeleton: ProgressivePlanSkeleton,
    step: ProgressiveSkeletonStep,
    step_index: int,
    variables: Mapping[str, Any],
    allowed_citations: frozenset[str],
    producers: Mapping[str, str],
    outputs_by_step: Mapping[str, Sequence[str]],
    host_reporting_source_key: str | None = None,
    host_interpretation_source_key: str | None = None,
) -> _CompiledStep:
    try:
        output_pairs = _canonical_outputs(step)
    except ValueError as exc:
        raise _fail(
            "progressive_conflicting_output_role",
            str(exc),
            step=step,
            step_index=step_index,
            path="outputs",
        ) from exc
    _validate_outputs(output_pairs, step=step, step_index=step_index)
    inputs, consumption = _compile_inputs(
        context=context,
        skeleton=skeleton,
        step=step,
        step_index=step_index,
        variables=variables,
        producers=producers,
        outputs_by_step=outputs_by_step,
    )
    citation_keys, literature = _compile_literature(
        step,
        allowed_citations=allowed_citations,
        step_index=step_index,
        host_reporting_source_key=host_reporting_source_key,
        host_interpretation_source_key=host_interpretation_source_key,
    )
    method = step.custom_method or _METHOD_BY_MODULE[step.module_id]
    kwargs: dict[str, Any] = {
        "step_id": step.step_id,
        "planned_analysis_role": step.planned_analysis_role,
        "intent": step.objective,
        "inputs": inputs,
        "expected_outputs": [product for product, _role in output_pairs],
        "method": method,
        "scientific_action_id": step.scientific_action_id,
        "icu_rule_refs": [],
        "sensitivity_spec_ids": list(step.sensitivity_spec_ids),
        "literature_citation_keys": citation_keys,
        "literature_design_bindings": literature,
        "input_consumption_contracts": consumption,
    }
    if step.module_id == "cohort_definition":
        kwargs["cohort_definition_spec"] = CohortDefinitionSpec(
            identity_column=_identity_column(
                context=context,
                step=step,
                step_index=step_index,
            ),
            eligibility_criteria=_eligibility_criteria(context, skeleton),
        )
    elif step.module_id == "table_one":
        spec = _compile_table_one(
            context=context,
            variables=variables,
            step=step,
            step_index=step_index,
        )
        kwargs["table_one_spec"] = spec
        kwargs["inputs"] = list(
            dict.fromkeys(
                [
                    *inputs,
                    spec.group_by,
                    *(item.name for item in spec.variables),
                ]
            )
        )
    elif step.module_id == "exposure_outcome_distribution":
        spec = _compile_distribution(
            variables=variables,
            step=step,
            step_index=step_index,
            counts_only=context_counts_only_authority(context),
        )
        kwargs["exposure_outcome_distribution_spec"] = spec
        kwargs["scientific_capability"] = "descriptive_exposure_outcome_distribution_v1"
        if (
            step.planned_analysis_role == "primary"
            and post_baseline_exposure(context)[0]
        ):
            kwargs["descriptive_claim"] = DescriptiveClaimContract(
                unresolved_limitations=(
                    "post_baseline_exposure_opportunity_unresolved",
                )
            )
        kwargs["inputs"] = list(dict.fromkeys([*inputs, spec.exposure, spec.outcome]))
    elif step.module_id == "measurement_audit":
        kwargs["measurement_audit_spec"] = _compile_measurement_spec(
            output_pairs,
            step=step,
            step_index=step_index,
        )
    elif step.module_id == "adjusted_association":
        kwargs["model_requirements"] = _compile_adjusted_association(
            variables=variables,
            step=step,
            step_index=step_index,
        )
        kwargs["scientific_capability"] = "association_adjusted_v1"
        kwargs["inputs"] = list(
            dict.fromkeys(
                [
                    *inputs,
                    str(step.primary_exposure),
                    str(step.outcome),
                    *(item.name for item in step.model_terms),
                ]
            )
        )
    elif step.module_id == "robustness_replay":
        kwargs["robustness_replay_spec"] = _compile_robustness_spec(
            output_pairs,
            step=step,
            step_index=step_index,
        )
    try:
        compiled = AnalysisStep.model_validate(kwargs)
    except ValidationError as exc:
        raise _fail(
            "progressive_analysis_step_invalid",
            str(exc),
            step=step,
            step_index=step_index,
            path="compiled_step",
        ) from exc
    return _CompiledStep(
        step=compiled,
        outputs=tuple(product for product, _role in output_pairs),
    )


def _preflight_step_findings(
    *,
    skeleton: ProgressivePlanSkeleton,
    context: ResearchContext,
    canonical_type: str,
    variables: Mapping[str, Any],
    allowed_citations: frozenset[str],
) -> None:
    """Report independent step defects together before suffix materialization.

    The compiler used to stop at the first invalid step.  A model could then
    repair that coordinate only to expose an unrelated later defect, consuming
    one Provider call per finding.  This preflight uses the same owner helpers
    as materialization, preserves their stable reason codes, and returns one
    earliest suffix coordinate plus the complete set of currently observable
    findings.  Final compilation still reruns every check and remains the
    authority.
    """

    findings: list[ProgressivePlanCompileError] = []
    producers: dict[str, str] = {}
    outputs_by_step: dict[str, tuple[str, ...]] = {}
    for index, step in enumerate(skeleton.steps):
        if step.scientific_action_id is not None:
            try:
                scientific_action_for_id(
                    analysis_type=canonical_type,
                    action_id=step.scientific_action_id,
                )
            except ScientificActionGapError as exc:
                findings.append(
                    _fail(
                        "progressive_scientific_action_invalid",
                        str(exc),
                        step=step,
                        step_index=index,
                        path="scientific_action_id",
                    )
                )

        try:
            output_pairs = _canonical_outputs(step)
            _validate_outputs(output_pairs, step=step, step_index=index)
        except ProgressivePlanCompileError as exc:
            findings.append(exc)
            output_pairs = []
        except ValueError as exc:
            findings.append(
                _fail(
                    "progressive_conflicting_output_role",
                    str(exc),
                    step=step,
                    step_index=index,
                    path="outputs",
                )
            )
            output_pairs = []

        output_ids = tuple(product for product, _role in output_pairs)
        for product in output_ids:
            prior = producers.get(product)
            if prior is not None:
                findings.append(
                    _fail(
                        "progressive_product_has_multiple_owners",
                        f"product {product!r} is already produced by {prior!r}",
                        step=step,
                        step_index=index,
                        path="outputs",
                    )
                )

        if output_pairs:
            try:
                _compile_one_step(
                    context=context,
                    skeleton=skeleton,
                    step=step,
                    step_index=index,
                    variables=variables,
                    allowed_citations=allowed_citations,
                    producers=producers,
                    outputs_by_step=outputs_by_step,
                )
            except ProgressivePlanCompileError as exc:
                findings.append(exc)

        for product in output_ids:
            producers.setdefault(product, step.step_id)
        outputs_by_step[step.step_id] = output_ids

    unique: list[ProgressivePlanCompileError] = []
    seen: set[tuple[object, ...]] = set()
    for finding in findings:
        key = (
            finding.reason_code,
            finding.step_id,
            finding.step_index,
            finding.path,
            finding.details["message"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    if not unique:
        return
    if len(unique) == 1:
        single = unique[0]
        single.details["findings"] = [dict(single.details)]
        raise single

    ordered = sorted(
        unique,
        key=lambda item: (
            item.step_index is None,
            item.step_index if item.step_index is not None else len(skeleton.steps),
            item.reason_code,
        ),
    )
    first = ordered[0]
    summary = "; ".join(
        f"{item.step_id or '<plan>'}:{item.reason_code}:{item.path or '<root>'}"
        for item in ordered
    )
    raise ProgressivePlanCompileError(
        "progressive_compile_batch_invalid",
        f"{len(ordered)} independent compiler findings must be fixed together: "
        f"{summary}",
        step_id=first.step_id,
        step_index=first.step_index,
        path=first.path,
        findings=[item.details for item in ordered],
    )


def assert_immutable_prefix(
    *,
    prior_receipt: ProgressivePlanCompileReceipt,
    revised_skeleton: ProgressivePlanSkeleton,
    locked_step_count: int,
) -> None:
    """Refuse a suffix revision that changes any compiled prefix step."""

    count = max(0, int(locked_step_count))
    if count > len(prior_receipt.compiled_steps) or count > len(revised_skeleton.steps):
        raise _fail(
            "progressive_locked_prefix_length_invalid",
            "locked prefix exceeds the prior receipt or revised skeleton",
            path="steps",
        )
    for index in range(count):
        expected = prior_receipt.compiled_steps[index]
        observed_step = revised_skeleton.steps[index]
        observed_sha = canonical_sha256(observed_step.model_dump(mode="json"))
        if (
            expected.step_id != observed_step.step_id
            or expected.skeleton_sha256 != observed_sha
        ):
            raise _fail(
                "progressive_locked_prefix_changed",
                "suffix revision attempted to alter an immutable compiled step",
                step=observed_step,
                step_index=index,
                path=f"steps[{index}]",
            )


def compile_progressive_plan(
    *,
    skeleton: ProgressivePlanSkeleton,
    context: ResearchContext,
    allowed_literature_citation_keys: Sequence[str] = (),
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None = None,
    host_reporting_method_source_keys: Sequence[str] = (),
) -> tuple[AnalysisPlan, ProgressivePlanCompileReceipt]:
    """Compile and validate one skeleton without weakening final plan gates."""

    canonical_type = canonical_analysis_family(skeleton.analysis_type)
    if canonical_type is None:
        raise _fail(
            "progressive_unknown_analysis_type",
            f"unknown analysis type {skeleton.analysis_type!r}",
            path="analysis_type",
        )
    try:
        validate_host_authorized_analysis_family(context, canonical_type)
    except ValueError as exc:
        raise _fail(
            "progressive_analysis_type_unauthorized",
            str(exc),
            path="analysis_type",
        ) from exc
    validate_progressive_foundation(
        ProgressivePlanFoundation(
            cohort=skeleton.cohort,
            display_labels=skeleton.display_labels,
            robustness_intents=skeleton.robustness_intents,
            know_how_decisions=skeleton.know_how_decisions,
        ),
        context=context,
        analysis_type=canonical_type,
    )
    allowed_modules = set(
        progressive_module_ids_for_analysis_types((canonical_type,))
    )
    for index, step in enumerate(skeleton.steps):
        if step.module_id not in allowed_modules:
            raise _fail(
                "progressive_analysis_module_unavailable",
                f"module {step.module_id!r} is unavailable for analysis type "
                f"{canonical_type!r}",
                step=step,
                step_index=index,
                path="module_id",
            )
    variables = _variable_index(context)
    cohort = _validate_progressive_cohort_intent(
        skeleton.cohort,
        context=context,
    )
    allowed_citations = frozenset(
        str(value).strip() for value in allowed_literature_citation_keys
    )
    host_reporting_keys = tuple(
        dict.fromkeys(
            str(value or "").strip()
            for value in host_reporting_method_source_keys
            if str(value or "").strip()
        )
    )
    host_reporting_target = next(
        (
            (index, step)
            for index, step in enumerate(skeleton.steps)
            if step.planned_analysis_role in {"primary", "secondary", "sensitivity"}
        ),
        (None, None),
    )
    target_index, target_step = host_reporting_target
    unavailable = sorted(set(host_reporting_keys) - allowed_citations)
    if unavailable:
        raise _fail(
            "progressive_host_reporting_source_unavailable",
            "article reporting source keys are outside the sealed run roster: "
            f"{unavailable!r}",
            step=target_step,
            step_index=target_index,
            path="host_reporting_method_source_keys",
        )
    invalid = sorted(
        key
        for key in host_reporting_keys
        if "reporting_standard"
        not in method_binding_support(key, ["reporting"])["matched_layers"]
    )
    if invalid:
        raise _fail(
            "progressive_host_reporting_source_invalid",
            "article reporting source keys lack reporting method-card authority: "
            f"{invalid!r}",
            step=target_step,
            step_index=target_index,
            path="host_reporting_method_source_keys",
        )
    host_reporting_source_key = (
        host_reporting_keys[0] if len(host_reporting_keys) == 1 else None
    )
    host_interpretation_source_key = (
        host_reporting_source_key
        if host_reporting_source_key is not None
        and "interpretation"
        in method_binding_support(
            host_reporting_source_key,
            ["outcome"],
        )["matched_layers"]
        else None
    )
    _preflight_step_findings(
        skeleton=skeleton,
        context=context,
        canonical_type=canonical_type,
        variables=variables,
        allowed_citations=allowed_citations,
    )
    producers: dict[str, str] = {}
    outputs_by_step: dict[str, tuple[str, ...]] = {}
    compiled_steps: list[AnalysisStep] = []
    step_receipts: list[ProgressiveCompiledStepReceipt] = []
    prefix_payload: list[dict[str, Any]] = []
    for index, skeleton_step in enumerate(skeleton.steps):
        if skeleton_step.scientific_action_id is not None:
            try:
                scientific_action_for_id(
                    analysis_type=canonical_type,
                    action_id=skeleton_step.scientific_action_id,
                )
            except ScientificActionGapError as exc:
                raise _fail(
                    "progressive_scientific_action_invalid",
                    str(exc),
                    step=skeleton_step,
                    step_index=index,
                    path="scientific_action_id",
                ) from exc
        compiled = _compile_one_step(
            context=context,
            skeleton=skeleton,
            step=skeleton_step,
            step_index=index,
            variables=variables,
            allowed_citations=allowed_citations,
            producers=producers,
            outputs_by_step=outputs_by_step,
            host_reporting_source_key=(
                host_reporting_source_key if index == target_index else None
            ),
            host_interpretation_source_key=(
                host_interpretation_source_key
                if skeleton_step.model_terms
                else None
            ),
        )
        for product in compiled.outputs:
            prior = producers.get(product)
            if prior is not None:
                raise _fail(
                    "progressive_product_has_multiple_owners",
                    f"product {product!r} is already produced by {prior!r}",
                    step=skeleton_step,
                    step_index=index,
                    path="outputs",
                )
            producers[product] = skeleton_step.step_id
        outputs_by_step[skeleton_step.step_id] = compiled.outputs
        compiled_steps.append(compiled.step)
        skeleton_payload = skeleton_step.model_dump(mode="json")
        compiled_payload = compiled.step.model_dump(mode="json")
        prefix_payload.append(skeleton_payload)
        step_receipts.append(
            ProgressiveCompiledStepReceipt(
                step_id=skeleton_step.step_id,
                skeleton_sha256=canonical_sha256(skeleton_payload),
                compiled_step_sha256=canonical_sha256(compiled_payload),
                immutable_prefix_sha256=canonical_sha256(prefix_payload),
            )
        )
    try:
        robustness = _compile_robustness_intents(
            skeleton,
            context=context,
            variables=variables,
        )
        know_how_decisions = [
            KnowHowDecision.model_validate(item.model_dump(mode="json"))
            for item in skeleton.know_how_decisions
        ]
        if allowed_know_how_decisions is None:
            if know_how_decisions:
                raise _fail(
                    "progressive_know_how_authority_absent",
                    "skeleton declares know-how decisions without a retrieved authority",
                    path="know_how_decisions",
                )
        else:
            verify_know_how_decisions(
                know_how_decisions,
                allowed_know_how_decisions,
            )
        analysis_type_spec = get_analysis_type(canonical_type)
        with cohort_concept_id_scope(_context_cohort_concept_ids(context)):
            plan = AnalysisPlan.model_validate(
                {
                    "research_question": context.research_question,
                    "analysis_type": canonical_type,
                    "steps": [
                        step.model_dump(mode="json") for step in compiled_steps
                    ],
                    "cohort": cohort.to_dict(),
                    "endpoint": (
                        context.endpoint.model_dump(mode="json")
                        if context.endpoint is not None
                        else None
                    ),
                    "robustness_specs": [item.to_dict() for item in robustness],
                    "display_labels": {
                        item.key: item.value for item in skeleton.display_labels
                    },
                    "know_how_decisions": [
                        item.model_dump(mode="json") for item in know_how_decisions
                    ],
                    "rationale": skeleton.rationale,
                    "revision": 1,
                }
            )
        validate_plan_scientific_action_selections(
            plan=plan,
            inferred_analysis_type=analysis_type_spec.key,
            require_result_actions=True,
        )
    except ProgressivePlanCompileError:
        raise
    except (ValidationError, ValueError, ScientificActionGapError) as exc:
        raise _fail(
            "progressive_analysis_plan_invalid",
            str(exc),
            path="compiled_plan",
        ) from exc
    skeleton_payload = skeleton.model_dump(mode="json")
    plan_payload = plan.model_dump(mode="json")
    return plan, ProgressivePlanCompileReceipt(
        skeleton_sha256=canonical_sha256(skeleton_payload),
        analysis_plan_sha256=canonical_sha256(plan_payload),
        compiled_steps=step_receipts,
    )


__all__ = [
    "assert_immutable_prefix",
    "compile_progressive_plan",
    "progressive_cohort_concept_ids",
    "validate_progressive_foundation",
]
