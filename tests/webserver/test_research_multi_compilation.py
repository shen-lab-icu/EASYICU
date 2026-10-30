"""Target plans recompile the reviewed design; copies cannot authorize drift."""
from dataclasses import replace

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver.research_followup_compile import compile_new_study
from easyicu.webserver.research_followup_contracts import CompletedResult
from easyicu.webserver.research_multi_plan import compile_same_study, require_same_design
from easyicu.webserver.research_multi_contracts import MultiDatabaseError
from tests.support.baseline_association import baseline_context


def source():
    context = baseline_context("sex", outcome="unit_death", covariates=("age",))
    context.cohort.database = "miiv"
    context, plan = compile_new_study(context, family="association_study")
    return CompletedResult("source", "a" * 64, plan, context, frozenset(), ())


# Databases that record age in years; banded sources are covered below.
@pytest.mark.parametrize("database", ["miiv", "eicu", "mimic"])
def test_recompilation_uses_target_metadata_and_retains_scientific_coordinates(database):
    original = source()
    target = original.context.model_copy(deep=True)
    target.cohort.database = database
    target.cohort.n_stays = 17
    target.cohort.provenance = {"synthetic_target_identity": database}
    before = original.context.model_dump_json(), original.plan.model_dump_json()
    context, plan, decision = compile_same_study(original, target)
    assert context.cohort.database == database
    assert context.cohort.n_stays == 17
    assert context.cohort.provenance["synthetic_target_identity"] == database
    assert context.endpoint == original.context.endpoint
    assert context.primary_exposure == original.context.primary_exposure
    assert plan.steps is not original.plan.steps
    require_same_design(plan, original.plan)
    assert decision.database == database
    assert before == (original.context.model_dump_json(), original.plan.model_dump_json())


@pytest.mark.parametrize("defect", ["drop_step", "outcome", "model", "covariate"])
def test_same_label_cannot_disguise_a_different_estimand(defect):
    original = source().plan
    altered = original.model_copy(deep=True)
    primary = next(step for step in altered.steps if step.planned_analysis_role == "primary")
    requirement = primary.model_requirements[0]
    if defect == "drop_step":
        altered.steps.pop()
    elif defect == "outcome":
        requirement.outcome = "another_synthetic_outcome"
    elif defect == "model":
        primary.method = "another_estimator"
    else:
        requirement.covariates = []
    with pytest.raises(MultiDatabaseError, match="design_changed"):
        require_same_design(altered, original)


def test_template_cannot_silently_omit_a_reviewed_additional_step():
    original = source()
    extra = original.plan.steps[-1].model_copy(update={"step_id": "extra_check", "method": "another_check"})
    changed = original.plan.model_copy(update={"steps": [*original.plan.steps, extra]})
    with pytest.raises(MultiDatabaseError, match="design_changed"):
        compile_same_study(replace(original, plan=changed), original.context)


@pytest.mark.parametrize("database", ["aumc", "hirid"])
def test_a_binned_covariate_is_recompiled_not_treated_as_measured_years(database):
    original = source()
    target = original.context.model_copy(deep=True)
    target.cohort.database = database
    context, plan, _ = compile_same_study(original, target)
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    age = next(term for term in primary.model_requirements[0].model_terms if term.name == "age")
    assert age.coding == "categorical"
    assert canonical_sha256(plan.model_dump(mode="json")) != canonical_sha256(original.plan.model_dump(mode="json"))
    assert context.cohort.provenance["database_capability_binding"]["binned_covariates"]


def test_sic_approximate_age_stays_approximate_and_requires_source_review():
    from easyicu.databases.research_substitutions import approximate_variables_used

    original = source()
    target = original.context.model_copy(deep=True)
    target.cohort.database = "sic"
    context, plan, _ = compile_same_study(original, target)
    binding = context.cohort.provenance["database_capability_binding"]
    assert "age" in binding["approximate_covariates"]
    assert "age" not in binding["binned_covariates"]
    assert "age" in approximate_variables_used(context, plan)


def test_fu1_owner_uses_the_same_real_compiler_without_source_observations(monkeypatch):
    from easyicu.webserver import research_multi_followup as module

    original = source()
    monkeypatch.setattr(module, "select_source", lambda _: ({}, "a" * 64))
    owner = module.RegisteredDatabaseOwner()
    decision = owner.decide(original, database="eicu")
    context, plan = owner.dry_compile(original, decision=decision)
    assert context.cohort.database == "eicu" and context.cohort.n_stays == 0
    assert "synthetic_target_identity" not in context.cohort.provenance
    require_same_design(plan, original.plan)
