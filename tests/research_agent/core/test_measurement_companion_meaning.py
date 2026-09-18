"""Observation counts and flags must not masquerade as clinical values."""

import pytest

from easyicu.concept.metadata_projection import (
    ColumnProjectionSpec,
    ConceptColumnMetadata,
    ConceptColumnRole,
    derive_concept_column_metadata,
    project_concept_column_metadata,
)
from easyicu.concept.schema import ConceptDefinition
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.research_context.outbound import outbound_safe_context_payload
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize(
    "role,phrase,time_origin,time_unit",
    [
        (ConceptColumnRole.COUNT, "non-null observation count", None, None),
        (ConceptColumnRole.MEASUREMENT_STATUS, "measurement availability", None, None),
        (ConceptColumnRole.FIRST_OBSERVATION_TIME, "first observation time", "icu_admission", "h"),
        (ConceptColumnRole.LAST_OBSERVATION_TIME, "last observation time", "icu_admission", "h"),
    ],
)
def test_producer_labels_the_representation_not_just_the_source(
    derived, role, phrase, time_origin, time_unit,
):
    definition = ConceptDefinition.from_name_and_payload(
        "burden_score", {"description": "Comorbidity burden score", "sources": {}},
    )
    spec = ColumnProjectionSpec(
        column_name="opaque_coordinate", source_concept="burden_score", role=role,
        time_origin=time_origin, time_unit=time_unit,
    )
    source = project_concept_column_metadata(
        definition, source_database="miiv",
        spec=ColumnProjectionSpec(
            column_name="burden_score", source_concept="burden_score",
            role=ConceptColumnRole.VALUE,
        ),
    )
    if derived:
        projected = derive_concept_column_metadata(source, spec=spec)
    else:
        projected = project_concept_column_metadata(
            definition, spec=spec, source_database="miiv",
        )
    assert phrase in projected.description.lower()
    assert "Comorbidity burden score" in projected.description
    assert projected.description != source.description
    assert projected.source_concept == source.source_concept
    assert projected.source_lineage == source.source_lineage

    # Frozen, older metadata remains readable without rewriting its bytes or label.
    archived = projected.to_dict()
    archived["description"] = "Comorbidity burden score"
    assert ConceptColumnMetadata.from_dict(archived).to_dict() == archived


@pytest.mark.parametrize(
    "transform", ["window_nonnull_count", "window_measurement_status"],
)
def test_planner_receives_the_closed_metadata_role_and_transform(transform):
    variable = ConceptDescriptor(
        name="opaque_coordinate", role="meta", dtype="float64",
        source_concept="burden_score", unit_normalization=transform,
        observed_domain={"n_unique": 3, "min": 888.123, "max": 999.321},
    )
    context = ResearchContext(
        research_question="Describe the cohort.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
        variables=[variable],
    )
    projected = outbound_safe_context_payload(context)["variables"][0]
    assert projected["role"] == "meta"
    assert projected["materialized_representation"] == transform
    assert "888.123" not in str(projected)
    assert "999.321" not in str(projected)


def test_unrecognized_representation_text_cannot_leave_the_host():
    context = ResearchContext(
        research_question="Describe the cohort.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
        variables=[ConceptDescriptor(
            name="opaque_coordinate", role="meta", dtype="float64",
            unit_normalization="private free-form value",
        )],
    )
    projected = outbound_safe_context_payload(context)["variables"][0]
    assert "materialized_representation" not in projected
    assert "private free-form value" not in str(projected)


def test_outline_cards_share_safe_representation_authority_with_step_prompts():
    context = ResearchContext(
        research_question="Describe the cohort.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
        variables=[ConceptDescriptor(
            name="opaque_coordinate", role="meta", dtype="float64",
            source_concept="burden_score", description="private free-form label",
            unit_normalization="window_nonnull_count",
            derived_from_concepts=["not_an_authorized_concept"],
            observed_domain={"levels": [123.456, 789.123], "n_unique": 2},
        )],
    )
    card = ProgressivePlannerAgent._retrieved_data_cards(context, ("opaque_coordinate",))[0]
    assert card["role"] == "meta"
    assert card["materialized_representation"] == "window_nonnull_count"
    assert card["table_one_restriction"] == "measurement_audit_only_unless_question_anchor"
    for private in ("burden_score", "private free-form label", "not_an_authorized_concept", "123.456", "789.123"):
        assert private not in str(card)
