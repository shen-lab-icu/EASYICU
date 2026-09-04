"""Zero-dependency authority for composite concept output names.

Some EasyICU loaders emit several user-facing columns from one extraction
source.  The mapping is shared by the concept catalog, export/runtime metadata,
and the research-agent cohort binder.  It lives outside ``easyicu.concept`` so
consumers do not import that package's executable loader facade merely to read
declarative metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

COMPOSITE_CONCEPT_OUTPUT_SOURCES: dict[str, str] = {
    "aki_reference": "kdigo_aki",
    "aki_stage_reference": "kdigo_aki",
    "aki_severe_reference": "kdigo_aki",
    "aki_stage_creat_reference": "kdigo_aki",
    "aki_stage_uo_reference": "kdigo_aki",
    "aki_stage_rrt_reference": "kdigo_aki",
    "aki_stage_reference_smoothed_6h": "kdigo_aki",
    "aki_reference_profile": "kdigo_aki",
    "aki_reference_status": "kdigo_aki",
    "aki_reference_authority": "kdigo_aki",
    "aki_reference_reliability_grade": "kdigo_aki",
    "aki_reference_fidelity": "kdigo_aki",
    "aki_reference_rrt_scope": "kdigo_aki",
    "aki_reference_uses_future": "kdigo_aki",
    "kidney_observation_window_coverage": "kdigo_aki",
    "creatinine_evidence_status": "kdigo_aki",
    "creatinine_evidence_reason": "kdigo_aki",
    "urine_evidence_status": "kdigo_aki",
    "rrt_evidence_status": "kdigo_aki",
    "aki_source_native": "kdigo_aki",
    "aki_stage_source_native": "kdigo_aki",
    "aki_severe_source_native": "kdigo_aki",
    "aki_stage_creat_source_native": "kdigo_aki",
    "aki_stage_uo_source_native": "kdigo_aki",
    "aki_stage_crrt_source_native": "kdigo_aki",
    "aki_stage_source_native_smoothed": "kdigo_aki",
    "aki_source_native_profile": "kdigo_aki",
    "aki_source_native_status": "kdigo_aki",
    "aki_source_native_authority": "kdigo_aki",
    "aki_source_native_reliability_grade": "kdigo_aki",
    "aki_source_native_fidelity": "kdigo_aki",
    "aki_source_native_ascertainment": "kdigo_aki",
    "aki_source_native_reason": "kdigo_aki",
    "aki_source_native_time_scale": "kdigo_aki",
    "aki_source_native_uses_future": "kdigo_aki",
    "creat_low_past_48hr": "kdigo_creat",
    "creat_low_past_7day": "kdigo_creat",
    "creat_baseline_n_48h": "kdigo_creat",
    "creat_baseline_n_7d": "kdigo_creat",
    "creat_baseline_source": "kdigo_creat",
    "creat_pre_icu_history_observed": "kdigo_creat",
    "uo_rt_6hr": "kdigo_uo",
    "uo_rt_12hr": "kdigo_uo",
    "uo_rt_24hr": "kdigo_uo",
    "circ_event": "circ_failure_loader",
    "circ_failure": "circ_failure_loader",
    "sep3_sofa1": "sep3",
    "charlson": "comorbidity_loader",
    "elixhauser": "comorbidity_loader",
    "mort_28d": "outcomes_loader",
    "mort_90d": "outcomes_loader",
    "mort_365d": "outcomes_loader",
    "followup_days_28d": "outcomes_loader",
    "followup_days_90d": "outcomes_loader",
    "followup_days_365d": "outcomes_loader",
    "icu_free_days_28": "outcomes_loader",
    "vent_free_days_28": "outcomes_loader",
    "icu_readmission": "outcomes_loader",
    "culture_positive": "microbiology_loader",
    "bld_culture_positive": "microbiology_loader",
}

# Executable aliases only; provenance pseudo-loaders above are not load tokens.
CONCEPT_OUTPUT_LOAD_SOURCES: Mapping[str, str] = MappingProxyType(
    {"sep3_sofa1": "sep3"}
)


class ConceptLoadPlanReason(str, Enum):
    INVALID_COLLECTION = "concept_load_plan.invalid_collection"
    EMPTY_REQUEST = "concept_load_plan.empty_request"
    INVALID_CONCEPT_TYPE = "concept_load_plan.invalid_concept_type"
    INVALID_CONCEPT_ID = "concept_load_plan.invalid_concept_id"


class ConceptLoadPlanError(ValueError):
    def __init__(
        self,
        reason: ConceptLoadPlanReason,
        message: str,
        *,
        position: int | None = None,
    ) -> None:
        self.reason = reason
        self.reason_code = reason.value
        self.position = position
        super().__init__(f"{reason.value}: {message}")


@dataclass(frozen=True, slots=True)
class ConceptMaterializationBinding:
    """One compiled public-output projection from an executable source."""

    output_concept: str
    source_concept: str


@dataclass(frozen=True, slots=True)
class ConceptLoadPlan:
    output_concepts: tuple[str, ...]
    source_concepts: tuple[str, ...]
    materializations: tuple[ConceptMaterializationBinding, ...]


_CONCEPT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def compile_concept_load_plan(output_concepts: Sequence[str]) -> ConceptLoadPlan:
    """Compile public outputs into their de-duplicated extraction sources.

    Both output concepts and source concepts preserve first-seen order.
    Only executable aliases in ``CONCEPT_OUTPUT_LOAD_SOURCES`` are rewritten;
    provenance-only composite mappings are not load tokens.
    """

    if isinstance(output_concepts, (str, bytes, bytearray)) or not isinstance(
        output_concepts, Sequence
    ):
        raise ConceptLoadPlanError(
            ConceptLoadPlanReason.INVALID_COLLECTION,
            "output_concepts must be an ordered sequence of concept identifiers",
        )
    if not output_concepts:
        raise ConceptLoadPlanError(
            ConceptLoadPlanReason.EMPTY_REQUEST,
            "at least one output concept is required",
        )

    output_order: list[str] = []
    for position, output_concept in enumerate(output_concepts):
        if not isinstance(output_concept, str):
            raise ConceptLoadPlanError(
                ConceptLoadPlanReason.INVALID_CONCEPT_TYPE,
                "every output concept must be a string",
                position=position,
            )
        if not _CONCEPT_ID_PATTERN.fullmatch(output_concept):
            raise ConceptLoadPlanError(
                ConceptLoadPlanReason.INVALID_CONCEPT_ID,
                "concept identifiers must match ^[a-z][a-z0-9_]*$",
                position=position,
            )
        if output_concept not in output_order:
            output_order.append(output_concept)

    output_sources = tuple(
        (concept, CONCEPT_OUTPUT_LOAD_SOURCES.get(concept, concept))
        for concept in output_order
    )
    return ConceptLoadPlan(
        output_concepts=tuple(output_order),
        source_concepts=tuple(dict.fromkeys(source for _, source in output_sources)),
        materializations=tuple(
            ConceptMaterializationBinding(
                output_concept=output,
                source_concept=source,
            )
            for output, source in output_sources
            if output != source
        ),
    )


__all__ = [
    "COMPOSITE_CONCEPT_OUTPUT_SOURCES",
    "CONCEPT_OUTPUT_LOAD_SOURCES",
    "ConceptLoadPlan",
    "ConceptLoadPlanError",
    "ConceptLoadPlanReason",
    "ConceptMaterializationBinding",
    "compile_concept_load_plan",
]
