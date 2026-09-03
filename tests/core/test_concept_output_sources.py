from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from easyicu.concept_output_sources import (
    CONCEPT_OUTPUT_LOAD_SOURCES,
    ConceptLoadPlanError,
    ConceptLoadPlanReason,
    ConceptMaterializationBinding,
    compile_concept_load_plan,
)


def test_compile_load_plan_resolves_execution_alias_and_deduplicates_stably() -> None:
    plan = compile_concept_load_plan(
        [
            "sep3_sofa1",
            "aki",
            "aki_stage",
            "hr",
            "aki_stage_creat",
            "charlson",
            "aki",
        ]
    )

    assert plan.output_concepts == (
        "sep3_sofa1",
        "aki",
        "aki_stage",
        "hr",
        "aki_stage_creat",
        "charlson",
    )
    assert plan.source_concepts == (
        "sep3",
        "aki",
        "aki_stage",
        "hr",
        "aki_stage_creat",
        "charlson",
    )
    assert plan.materializations == (
        ConceptMaterializationBinding(
            output_concept="sep3_sofa1",
            source_concept="sep3",
        ),
    )


def test_execution_alias_map_is_frozen_and_excludes_provenance_pseudo_loaders() -> None:
    assert dict(CONCEPT_OUTPUT_LOAD_SOURCES) == {"sep3_sofa1": "sep3"}

    with pytest.raises(TypeError):
        CONCEPT_OUTPUT_LOAD_SOURCES["aki"] = "kdigo_aki"  # type: ignore[index]


def test_load_plan_is_frozen() -> None:
    plan = compile_concept_load_plan(["sep3_sofa1", "aki_stage_rrt"])

    with pytest.raises(FrozenInstanceError):
        plan.source_concepts = ("kdigo_uo",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.materializations[0].source_concept = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("hr", ConceptLoadPlanReason.INVALID_COLLECTION),
        ({"hr": True}, ConceptLoadPlanReason.INVALID_COLLECTION),
        ({"hr"}, ConceptLoadPlanReason.INVALID_COLLECTION),
        ((item for item in ["hr"]), ConceptLoadPlanReason.INVALID_COLLECTION),
        ([], ConceptLoadPlanReason.EMPTY_REQUEST),
        (["hr", None], ConceptLoadPlanReason.INVALID_CONCEPT_TYPE),
        (["hr", ""], ConceptLoadPlanReason.INVALID_CONCEPT_ID),
        (["hr", " HR"], ConceptLoadPlanReason.INVALID_CONCEPT_ID),
        (["hr", "hr-rate"], ConceptLoadPlanReason.INVALID_CONCEPT_ID),
    ],
)
def test_compile_load_plan_fails_with_stable_reason_codes(
    payload: object,
    reason: ConceptLoadPlanReason,
) -> None:
    with pytest.raises(ConceptLoadPlanError) as caught:
        compile_concept_load_plan(payload)  # type: ignore[arg-type]

    assert caught.value.reason is reason
    assert caught.value.reason_code == reason.value
    assert str(caught.value).startswith(f"{reason.value}:")


def test_compile_load_plan_accepts_ordered_tuple_boundary() -> None:
    plan = compile_concept_load_plan(("mort_28d", "death", "mort_28d"))

    assert plan.output_concepts == ("mort_28d", "death")
    assert plan.source_concepts == ("mort_28d", "death")
    assert plan.materializations == ()
