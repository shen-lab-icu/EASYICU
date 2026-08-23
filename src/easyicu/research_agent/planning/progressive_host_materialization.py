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
from ..schema import ResearchContext
from .method_literature import METHOD_CARDS
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
    "exposure_outcome_distribution": ("exposure", "outcome", "reporting"),
    "measurement_audit": ("missing_data", "reporting"),
    "robustness_replay": ("robustness", "missing_data"),
    "visualization": ("reporting",),
    "report": ("reporting",),
}


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
        literature_bindings=_bindings(step),
        **kwargs,
    )


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
    action = None
    if outline_step.scientific_action_id:
        action = scientific_action_for_id(
            analysis_type=outline.analysis_type,
            action_id=outline_step.scientific_action_id,
        )
        if action.adapter_status != "full_action":
            return None

    if module == "cohort_definition":
        skeleton = _common_step(outline_step, raw_inputs=raw)
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
            sensitivity_spec_ids=[item.spec_id for item in foundation.robustness_intents],
        )
    elif module == "visualization":
        refs = _refs(available_product_refs, outline_step.depends_on, result_only=True)
        if not refs or len(refs) > 4:
            return None
        skeleton = _common_step(
            outline_step,
            raw_inputs=(),
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


__all__ = ["host_materialize_progressive_step"]
