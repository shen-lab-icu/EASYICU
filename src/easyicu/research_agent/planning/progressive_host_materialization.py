"""Deterministically materialize low-entropy Progressive Planner steps.

The Planner owns question-specific design and action selection in the outline.
This owner only projects already-sealed coordinates into repetitive execution
fields.  Returning ``None`` means that scientific judgment is still required;
the caller must use the Planner for that one step.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..authority.declared_levels import observed_levels_for
from ..canonical_json import canonical_sha256
from ..contracts.figure_plan import ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS
from ..contracts.ordered_stratified import (
    PARENT_PRODUCT as ORDERED_STRATIFIED_PARENT_PRODUCT,
    SCIENTIFIC_ACTION_ID as ORDERED_STRATIFIED_ACTION_ID,
)
from ..schema import ResearchContext
from .method_literature import METHOD_CARDS
from .ordinal_multi_outcome import resolve_ordinal_multi_outcome_contract
from .progressive_contract import (
    ProgressiveLiteratureBinding,
    ProgressiveOutlineStep,
    ProgressiveOutputIntent,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
    ProgressiveProductRef,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
    ProgressiveTableOneVariable,
)
from .scientific_action_catalog import scientific_action_for_id


_MODULE_DESIGN_ELEMENTS: Mapping[str, tuple[str, ...]] = {
    "cohort_definition": ("population", "time_zero", "reporting"),
    "table_one": ("population", "reporting"),
    "exposure_outcome_distribution": (
        "exposure",
        "outcome",
        "dependence",
        "reporting",
    ),
    "measurement_audit": ("missing_data", "reporting"),
    "adjusted_association": (
        "adjustment",
        "dependence",
        "estimand",
        "exposure",
        "functional_form",
        "interpretation",
        "missing_data",
        "outcome",
        "reporting",
        "robustness",
        "time_zero",
    ),
    "absolute_risk_context": ("estimand", "outcome", "reporting"),
    "robustness_replay": ("robustness", "missing_data"),
    "visualization": ("reporting",),
    "report": ("reporting",),
}


def progressive_module_method_source_keys(
    module_id: str,
    allowed_keys: Sequence[str],
) -> tuple[str, ...]:
    """Return sealed method sources applicable to one progressive module.

    The mapping is the same typed design-element authority used when the host
    materializes low-entropy steps. It may wire an already sealed method card
    to a step, but it cannot invent a citation or a question-specific design
    decision.
    """

    desired = set(_MODULE_DESIGN_ELEMENTS.get(str(module_id or ""), ()))
    if not desired:
        return ()
    allowed = tuple(dict.fromkeys(str(key or "").strip() for key in allowed_keys))
    return tuple(
        key
        for key in allowed
        if key
        and any(
            card.source_key == key and desired & set(card.design_elements)
            for card in METHOD_CARDS
        )
    )


def _bindings(step: ProgressiveOutlineStep) -> list[ProgressiveLiteratureBinding]:
    desired = set(_MODULE_DESIGN_ELEMENTS.get(step.module_id, ()))
    if not desired:
        return []
    bindings = []
    for key in step.literature_citation_keys:
        cards = [card for card in METHOD_CARDS if card.source_key == key]
        supported = set(
            element for card in cards for element in card.design_elements
        )
        elements = sorted(desired & supported)
        if not elements:
            continue
        card_ids = ", ".join(card.id for card in cards if desired & set(card.design_elements))
        bindings.append(
            ProgressiveLiteratureBinding(
                citation_key=key,
                design_elements=elements,
                application=(
                    f"Apply the host-curated method card(s) {card_ids} only "
                    f"to the declared {step.module_id} coordinate and retain "
                    "the run's evidence ceiling."
                ),
                divergence=None,
            )
        )
    return bindings


def _rendering_bindings(
    *,
    outline: ProgressivePlanOutline,
    step: ProgressiveOutlineStep,
    existing: Sequence[ProgressiveLiteratureBinding],
) -> list[ProgressiveLiteratureBinding]:
    """Reuse selected design decisions for deterministic figure/report wiring."""

    bindings = list(existing)
    bound_keys = {binding.citation_key for binding in bindings}
    selection = outline.design_selection
    if selection is None:
        return bindings
    dimensions = {
        "visualization": {"table_and_figure_completeness"},
        "report": {"table_and_figure_completeness", "conclusion_boundaries"},
    }.get(step.module_id, set())
    decisions = [
        decision
        for decision in selection.selected.literature_design_decisions
        if decision.dimension in dimensions
    ]
    for key in step.literature_citation_keys:
        if key in bound_keys:
            continue
        rationales = [
            decision.rationale
            for decision in decisions
            if key in decision.citation_keys
        ]
        if not rationales:
            continue
        bindings.append(
            ProgressiveLiteratureBinding(
                citation_key=key,
                design_elements=["reporting"],
                application=(
                    "Apply the selected, literature-bound design decision to "
                    f"the {step.module_id} coordinate: " + " ".join(rationales)
                ),
                divergence=None,
            )
        )
        bound_keys.add(key)
    return bindings


def _output(product_id: str, role: str) -> ProgressiveOutputIntent:
    return ProgressiveOutputIntent(product_id=product_id, semantic_role=role)


def _product_name(step_id: str) -> str:
    return step_id if step_id[:1].isalpha() else f"step_{step_id}"


def _refs(
    available: Sequence[tuple[str, str]],
    dependencies: Sequence[str],
    *,
    result_only: bool = False,
) -> list[ProgressiveProductRef]:
    allowed = set(dependencies)
    refs = []
    for producer, product in available:
        if producer not in allowed:
            continue
        if result_only and not product.startswith(("table:", "statistic:")):
            continue
        refs.append(ProgressiveProductRef(producer_step_id=producer, product_id=product))
    return refs


def _exact_profile_refs(
    available: Sequence[tuple[str, str]],
    dependencies: Sequence[str],
    products: Sequence[str],
) -> list[ProgressiveProductRef]:
    allowed = set(dependencies)
    refs = []
    for product in products:
        matches = [
            producer
            for producer, candidate in available
            if producer in allowed and candidate == product
        ]
        if len(matches) != 1:
            return []
        refs.append(
            ProgressiveProductRef(
                producer_step_id=matches[0],
                product_id=product,
            )
        )
    return refs


def _table_summary(descriptor: object) -> str:
    levels = observed_levels_for(
        name=descriptor.name, variables={descriptor.name: descriptor}
    )
    dtype = str(descriptor.dtype or "").casefold()
    if descriptor.is_ordinal or len(levels) >= 2 or dtype.startswith(
        ("object", "str", "string", "category", "bool")
    ):
        return "count_percent"
    return "both"


def _common_step(
    step: ProgressiveOutlineStep,
    *,
    raw_inputs: Sequence[str],
    literature_bindings: Sequence[ProgressiveLiteratureBinding],
    product_inputs: Sequence[ProgressiveProductRef] = (),
    outputs: Sequence[ProgressiveOutputIntent] = (),
    **kwargs: object,
) -> ProgressiveSkeletonStep:
    return ProgressiveSkeletonStep(
        step_id=step.step_id,
        planned_analysis_role=step.planned_analysis_role,
        module_id=step.module_id,
        objective=step.objective,
        depends_on=list(step.depends_on),
        raw_inputs=list(raw_inputs),
        product_inputs=list(product_inputs),
        outputs=list(outputs),
        scientific_action_id=step.scientific_action_id,
        literature_bindings=list(literature_bindings),
        **kwargs,
    )


def normalize_progressive_cohort_identity(
    materialization: ProgressiveStepMaterialization,
    *,
    context: ResearchContext,
) -> ProgressiveStepMaterialization:
    """Project a uniquely proven row identity into a Planner cohort step."""

    step = materialization.step
    if step.module_id != "cohort_definition":
        return materialization
    identity = _cohort_row_identity(context, step.raw_inputs)
    if identity is None:
        return materialization
    declared_ids = {name for name in step.raw_inputs if name in context.cohort.id_columns}
    normalized_inputs = [
        name
        for name in step.raw_inputs
        if name not in declared_ids or name == identity
    ]
    if identity not in normalized_inputs:
        normalized_inputs.insert(0, identity)
    if normalized_inputs == step.raw_inputs:
        return materialization
    normalized_step = step.model_copy(
        update={"raw_inputs": normalized_inputs}
    )
    return materialization.model_copy(update={"step": normalized_step})


def normalize_progressive_action_contract(
    materialization: ProgressiveStepMaterialization,
    *,
    context: ResearchContext,
    outline_step: ProgressiveOutlineStep,
    available_product_refs: Sequence[tuple[str, str]],
) -> ProgressiveStepMaterialization:
    """Close exact action-owned inputs without replacing Planner judgment.

    The outline remains the authority for selecting a scientific action and
    its variables. Once that choice matches a typed action contract, asking
    the model to repeat the same raw-column and upstream-product wiring during
    step materialization creates avoidable failure entropy. Project only the
    uniquely implied execution coordinates; ambiguous contexts stay unchanged
    and fail through the ordinary compiler.
    """

    step = materialization.step
    if (
        step.scientific_action_id != ORDERED_STRATIFIED_ACTION_ID
        or outline_step.scientific_action_id != ORDERED_STRATIFIED_ACTION_ID
        or step.module_id != "custom_analysis"
        or step.planned_analysis_role != "secondary"
    ):
        return materialization
    contract = resolve_ordinal_multi_outcome_contract(context)
    if contract is None or set(outline_step.variable_names) != set(contract.variables):
        return materialization
    parent_owners = [
        producer
        for producer, product_id in available_product_refs
        if product_id == ORDERED_STRATIFIED_PARENT_PRODUCT
        and producer in step.depends_on
    ]
    if len(parent_owners) != 1:
        return materialization
    normalized_step = step.model_copy(
        update={
            "raw_inputs": list(contract.variables),
            "product_inputs": [
                ProgressiveProductRef(
                    producer_step_id=parent_owners[0],
                    product_id=ORDERED_STRATIFIED_PARENT_PRODUCT,
                )
            ],
        }
    )
    if normalized_step == step:
        return materialization
    return materialization.model_copy(update={"step": normalized_step})


def _cohort_row_identity(
    context: ResearchContext,
    raw_inputs: Sequence[str],
) -> str | None:
    """Resolve one host-proven row identity without guessing among IDs."""

    declared_ids = tuple(
        name for name in raw_inputs if name in context.cohort.id_columns
    )
    if len(declared_ids) == 1:
        return declared_ids[0]

    provenance = context.cohort.provenance
    if provenance.get("analysis_unit") != "icu_stay":
        return None
    stay_ids = provenance.get("stay_id_columns")
    if not isinstance(stay_ids, list):
        return None
    proven_stay_ids = tuple(
        name
        for name in stay_ids
        if isinstance(name, str) and name in context.cohort.id_columns
    )
    if len(proven_stay_ids) != 1:
        return None
    proven = proven_stay_ids[0]
    if declared_ids and proven not in declared_ids:
        return None
    return proven


def host_materialize_progressive_step(
    *,
    context: ResearchContext,
    outline: ProgressivePlanOutline,
    outline_step: ProgressiveOutlineStep,
    foundation: ProgressivePlanFoundation,
    available_product_refs: Sequence[tuple[str, str]],
) -> ProgressiveStepMaterialization | None:
    """Return a host-owned materialization, or ``None`` for model judgment."""

    module = outline_step.module_id
    variables = {item.name: item for item in context.variables}
    raw = [name for name in outline_step.variable_names if name in variables]
    bindings = _bindings(outline_step)
    if module in {"visualization", "report"}:
        bindings = _rendering_bindings(
            outline=outline,
            step=outline_step,
            existing=bindings,
        )
    if {binding.citation_key for binding in bindings} != set(
        outline_step.literature_citation_keys
    ):
        # Dynamic design-analogue cards carry question-specific applications
        # that this mechanical owner cannot reconstruct from a citation key.
        return None
    action = None
    if outline_step.scientific_action_id:
        action = scientific_action_for_id(
            analysis_type=outline.analysis_type,
            action_id=outline_step.scientific_action_id,
        )
        if action.adapter_status != "full_action":
            return None

    if module == "cohort_definition":
        identity = _cohort_row_identity(context, raw)
        declared_ids = {name for name in raw if name in context.cohort.id_columns}
        if len(declared_ids) > 1 and identity is None:
            return None
        cohort_raw = [
            name
            for name in raw
            if name not in declared_ids or name == identity
        ]
        if identity is not None and identity not in cohort_raw:
            cohort_raw.insert(0, identity)
        skeleton = _common_step(
            outline_step,
            raw_inputs=cohort_raw,
            literature_bindings=bindings,
        )
    elif module == "table_one":
        exposure = context.primary_exposure
        if not exposure or exposure not in raw:
            return None
        rows = [
            ProgressiveTableOneVariable(name=name, summary=_table_summary(variables[name]))
            for name in raw
            if name != exposure
        ]
        if not rows:
            return None
        skeleton = _common_step(
            outline_step,
            raw_inputs=raw,
            literature_bindings=bindings,
            table_one_group_by=exposure,
            table_one_mode="descriptive_smd_only",
            table_one_variables=rows,
        )
    elif module == "exposure_outcome_distribution":
        exposure, outcome = context.primary_exposure, context.target_outcome
        if not exposure or not outcome or exposure not in raw or outcome not in raw:
            return None
        exposure_levels = observed_levels_for(name=exposure, variables=variables)
        outcome_levels = observed_levels_for(name=outcome, variables=variables)
        if len(exposure_levels) < 2 or len(outcome_levels) != 2:
            return None
        skeleton = _common_step(
            outline_step,
            raw_inputs=raw,
            literature_bindings=bindings,
            primary_exposure=exposure,
            outcome=outcome,
            outcome_type="binary",
            event_level_index=1,
            reference_exposure_level_index=0,
            comparison_exposure_level_index=1,
            denominator_policy="observed_outcome_rows",
            missing_exposure_policy="exclude_from_denominator",
            missing_outcome_policy="exclude_from_denominator",
            confidence_level=0.95,
        )
    elif module == "measurement_audit":
        # The deterministic runner owns execution after the Planner declares
        # which audit questions this study needs.  The outline currently does
        # not carry that variable-length output-role roster, so choosing it
        # here would silently replace a scientific requirement.
        return None
    elif module == "robustness_replay" and action is not None:
        skeleton = _common_step(
            outline_step,
            raw_inputs=raw,
            literature_bindings=bindings,
            sensitivity_spec_ids=[item.spec_id for item in foundation.robustness_intents],
        )
    elif module == "visualization":
        refs = _exact_profile_refs(
            available_product_refs,
            outline_step.depends_on,
            ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS,
        ) or _refs(
            available_product_refs,
            outline_step.depends_on,
            result_only=True,
        )
        if not refs or len(refs) > 4:
            return None
        skeleton = _common_step(
            outline_step,
            raw_inputs=(),
            literature_bindings=bindings,
            product_inputs=refs,
            outputs=(_output(f"figure:{_product_name(outline_step.step_id)}", "figure"),),
        )
    elif module == "report":
        refs = _refs(available_product_refs, outline_step.depends_on)
        if not refs:
            return None
        skeleton = _common_step(
            outline_step,
            raw_inputs=(),
            literature_bindings=bindings,
            product_inputs=refs,
            outputs=(_output(f"report:{_product_name(outline_step.step_id)}", "report"),),
        )
    else:
        return None

    return ProgressiveStepMaterialization(
        outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
        foundation=None,
        step=skeleton,
    )


__all__ = [
    "host_materialize_progressive_step",
    "normalize_progressive_action_contract",
    "normalize_progressive_cohort_identity",
    "progressive_module_method_source_keys",
]
