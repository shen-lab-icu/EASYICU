"""Source application retains the reviewed capability owner's boundaries."""
import json
from types import SimpleNamespace

import pytest

from easyicu.databases.research_approval import CapabilityApprovalError, require_approved_database_capabilities
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import UserPreferences
from tests.support.source_substitutions import _approve, _review, context, plan


def test_unknown_concepts_remain_unassessed_in_an_applied_proposal(tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = context("hirid", unit="hospitalization", outcome="death")
    ctx = ctx.model_copy(update={"primary_exposure": "synthetic_assay"})
    bound = bind_source_substitutions(ctx)
    compiled = plan(bound)
    review = _review(bound, compiled)
    decision = review.facts["database_capability_review"]["decision"]
    assert decision["status"] == "substitution"
    assert decision["unassessed_concepts"] == ["synthetic_assay"]
    assert review.approval_allowed
    require_approved_database_capabilities(_approve(tmp_path, EvidenceStore(tmp_path), bound, compiled, review), tmp_path)


def test_an_unregistered_cohort_keeps_its_binding_and_population_behavior():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = context("miiv")
    ctx = ctx.model_copy(update={"cohort": ctx.cohort.model_copy(update={"database": "synthetic_export"}),
                                 "user_preferences": UserPreferences()})
    assert bind_source_substitutions(ctx) == ctx
    assert bind_source_substitutions(ctx, restoring=True) == ctx
    spec = PopulationSpec.model_validate({"criteria": [{"id": "c1", "kind": "age_years", "role": "include",
        "source": "question", "quote": "Include the stated age group.", "min_years": 95}]})
    compiled = compile_population(spec, ctx)
    assert not compiled.blocking
    assert compiled.cohort_definition().inclusion[0].value == 95
    assert "database_capability_binding" not in compiled.record()


def test_binding_does_not_swallow_a_registered_metadata_error(monkeypatch):
    from easyicu.databases import research_substitutions as owner

    ctx = context("miiv")

    def broken(database):
        raise KeyError("synthetic_metadata_error")

    monkeypatch.setattr(owner, "research_capability_profile", broken)
    with pytest.raises(KeyError, match="synthetic_metadata_error"):
        owner.bind_source_substitutions(ctx)


def test_preplanner_binding_resolves_only_the_requested_recipe(monkeypatch):
    from easyicu.databases import research_capabilities as owner
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = context("miiv").model_copy(update={"primary_exposure": "sofa_coag",
                                             "user_preferences": UserPreferences()})
    requested = []

    def resolve(name, database):
        requested.append(name)
        return "full", None, ()

    monkeypatch.setattr(owner, "_derivation_status", resolve)
    assert bind_source_substitutions(ctx) == ctx
    assert requested and set(requested) == {"sofa_coag"}


@pytest.mark.parametrize("covariates", [None, []])
def test_an_applied_proposal_keeps_an_undeclared_adjustment_roster(covariates):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("aumc", outcome="death"))
    compiled = plan(ctx)
    compiled.steps[0].model_requirements[0].covariates = covariates
    assert _review(ctx, compiled).approval_allowed
    assert compiled.steps[0].model_requirements[0].covariates == covariates


@pytest.mark.parametrize("remove_binding", [False, True])
def test_changing_an_approved_source_to_an_unknown_profile_never_bypasses_review(tmp_path, remove_binding):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("hirid", outcome="death"))
    compiled = plan(ctx)
    result = _approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, _review(ctx, compiled))
    result.context = ctx.model_copy(update={"cohort": ctx.cohort.model_copy(update={"database": "synthetic_export"})})
    if remove_binding:
        result.context = result.context.model_copy(update={"user_preferences": UserPreferences(),
            "cohort": result.context.cohort.model_copy(update={"provenance": {}})})
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_review_mismatch"):
        require_approved_database_capabilities(result, tmp_path)


@pytest.mark.parametrize("field,value", [("identifier", "invented_id"), ("outcome_meaning", "death"),
                                        ("effective_analysis_unit", "hospitalization")])
def test_restore_checks_the_entire_source_binding(field, value):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("hirid", outcome="death"))
    constraints = json.loads(ctx.user_preferences.data_constraints)
    constraints["database_capability_binding"][field] = value
    changed = ctx.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    with pytest.raises(ValueError, match="capability_substitution_binding_drift"):
        bind_source_substitutions(changed, restoring=True)
    with pytest.raises(ValueError, match="capability_substitution_binding_drift"):
        compile_population(PopulationSpec(criteria=[]), changed)


def test_a_refusal_never_creates_a_source_binding():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = context("hirid", unit="hospitalization", outcome="death")
    constraints = json.loads(ctx.user_preferences.data_constraints)
    constraints["database_study_requirements"]["concepts"] = ["los_hosp"]
    ctx = ctx.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    assert bind_source_substitutions(ctx) == ctx


def test_lost_binding_is_detected_before_population_compilation():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("hirid", outcome="death"))
    changed = ctx.model_copy(update={"user_preferences": UserPreferences()})
    with pytest.raises(ValueError, match="capability_substitution_binding_drift"):
        bind_source_substitutions(changed)


def test_restoration_never_rewrites_the_sealed_source_domain():
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("aumc", outcome="death"))
    variables = [v.model_copy(update={"is_ordinal": False, "ordinal_levels": None}) if v.name == "age" else v
                 for v in ctx.variables]
    changed = ctx.model_copy(update={"variables": variables})
    with pytest.raises(ValueError, match="capability_substitution_binding_drift"):
        bind_source_substitutions(changed, restoring=True)


def test_deleting_all_original_coordinates_cannot_erase_an_approved_substitution(tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("hirid", outcome="death"))
    compiled = plan(ctx)
    result = _approve(tmp_path, EvidenceStore(tmp_path), ctx, compiled, _review(ctx, compiled))
    result.context = ctx.model_copy(update={"user_preferences": UserPreferences(),
        "cohort": ctx.cohort.model_copy(update={"provenance": {}})})
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_review_mismatch"):
        require_approved_database_capabilities(result, tmp_path)


def test_the_timing_binding_discloses_a_monitoring_proxy():
    from easyicu.databases.research_substitutions import bind_source_substitutions, source_binding

    ctx = bind_source_substitutions(context("hirid", outcome="death"))
    timing = source_binding(ctx)["mortality_timing"]
    assert timing["meaning"] == "icu_death"
    assert timing["time_kind"] == "last_observation_proxy"
    assert timing["event_time"]["column"] == "datetime"
    assert timing["status"]["column"] == "discharge_status"


def test_execute_requires_the_bound_native_identifier(tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions

    ctx = bind_source_substitutions(context("hirid", unit="hospitalization", outcome="death"))
    compiled = plan(ctx)
    changed = ctx.model_copy(update={"cohort": ctx.cohort.model_copy(update={"id_columns": ["synthetic_id"]})})
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
        require_approved_database_capabilities(SimpleNamespace(context=changed, plan=compiled), tmp_path)


def test_execute_rejects_misordered_band_levels(tmp_path):
    from easyicu.databases.research_substitutions import bind_source_substitutions
    from easyicu.research_agent.schema import ModelTermSpec

    ctx = bind_source_substitutions(context("aumc", outcome="icu_death"))
    compiled = plan(ctx)
    requirement = compiled.steps[0].model_requirements[0]
    requirement.covariates = ["age"]
    requirement.model_terms = [ModelTermSpec(name="age", role="covariate", coding="categorical",
        levels=["65", "30", "75"], reference_level="65", transform="treatment_contrast")]
    with pytest.raises(CapabilityApprovalError, match="capability_substitution_not_applied"):
        require_approved_database_capabilities(SimpleNamespace(context=ctx, plan=compiled), tmp_path)
