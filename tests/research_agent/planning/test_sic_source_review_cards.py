"""Source limitations are offered for discussion without authorizing execution."""
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import (
    ReviewOnlyTimeOrigin, StudyRequirements, assess_study, review_only_time_origin,
)
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext
from pydantic import ValidationError
import pytest


def _card(database: str, *, source_concept: str = "los_icu") -> dict:
    variable = ConceptDescriptor(name="recorded_length", dtype="float64", role="outcome", source_concept=source_concept)
    context = ResearchContext(
        research_question="Early lactate and ICU stay length",
        cohort=CohortDescriptor(cohort_name="metadata only", database=database, n_stays=0),
        variables=[variable], target_outcome=variable.name,
    )
    return ProgressivePlannerAgent._retrieved_data_cards(context, [variable.name])[0]


def test_sic_proxy_is_review_only_and_exact_endpoint_remains_refused():
    candidate = _card("sic")["source_boundary_review"]
    assert candidate["status"] == "review_only_not_selected_or_executable"
    assert candidate["start"]["column"] == "ICUOffset"
    assert candidate["end"]["column"] == "TimeOfStay"
    origin = candidate["time_origin_candidate"]
    assert origin["database"] == "sic"
    assert origin["source_identity_scope"] == "packaged_registry_only"
    assert origin["requested_origin"] == "icu_admission"
    assert origin["proposed_origin"] == "first_intensive_or_intermediate_care_transfer"
    assert origin["source_field"]["column"] == "ICUOffset"
    assert origin["executable"] is False
    assert origin["review_required"] is True
    assert origin["execution_contract_available"] is False
    with pytest.raises(ValidationError):
        ReviewOnlyTimeOrigin.model_validate({**origin, "executable": True})
    assert "not a verified continuous ICU stay" in candidate["limitation"]
    decision = assess_study(research_capability_profile("sic"), StudyRequirements(
        family="association", outcome="los_icu", concepts=("lact",),
    ))
    assert decision.status == "refusal"
    assert any(item.capability == "icu_los" for item in decision.missing)


def test_candidates_do_not_change_other_sources_or_unrelated_concepts():
    assert "source_boundary_review" not in _card("miiv")
    assert "source_boundary_review" not in _card("sic", source_concept="lact")
    profile = research_capability_profile("sic")
    assert review_only_time_origin(profile.model_copy(update={"source_registered": False})) is None
    assert review_only_time_origin(profile.model_copy(update={"schema_database": "miiv"})) is None
