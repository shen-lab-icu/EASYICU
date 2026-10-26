"""Family and database admission for result follow-ups uses real dry compilers."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.capability_registry import resolve_primary_capability
from easyicu.research_agent.planning.family_spec import build_family_spec_request, build_prediction_skeleton, validate_family_plan_spec
from easyicu.research_agent.planning.family_spec.contract import FamilyPlanSpec
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_resume import assemble_progressive_skeleton
from easyicu.research_agent.schema import UserPreferences
from easyicu.webserver.research_followup_compile import compile_new_study
from easyicu.webserver.research_followup_contracts import CompletedResult, DatabaseDecision, FollowupReading
from easyicu.webserver.research_followups import suggest_followups
from tests.support.baseline_association import baseline_context
from tests.support.survival_proposal import survival_context
from tests.research_agent.planning.family_spec_fixtures import _prediction_context, _prediction_payload


def _source(family):
    context = baseline_context("chronic_flag", covariates=("age", "sex"))
    context.cohort.database = "miiv"
    if family == "survival":
        context = survival_context(user_preferences=UserPreferences(
            inferred_analysis_family="survival", covariate_selection="exact", covariates=["age"],
            covariate_rationales={"age": "The study specifies age as a baseline confounder."},
            covariate_temporal_roles={"age": "baseline_static"},
        ))
    if family == "prediction_model":
        context = _prediction_context()
        context.cohort.database = "miiv"
        request = build_family_spec_request(context, analysis_types=(family,), variable_roster=[v.name for v in context.variables],
                                            allowed_literature_citation_keys=())
        spec = FamilyPlanSpec.model_validate(_prediction_payload(request, features=["age", "sex", "hr_max"]))
        validate_family_plan_spec(spec, request)
        draft = build_prediction_skeleton(request, spec)
        plan, _ = compile_progressive_plan(context=context, skeleton=assemble_progressive_skeleton(
            outline=draft.outline, foundation=draft.foundation.foundation, steps=[m.step for m in draft.materializations]))
    else:
        context, plan = compile_new_study(context, family=family)
    return CompletedResult("source", "a" * 64, plan, context,
                           frozenset(key for step in plan.steps for key in step.expected_outputs), tuple(plan.steps))


@pytest.mark.parametrize("family,expected", [
    ("association_study", "descriptive_context"), ("survival", "descriptive_context"),
    ("prediction_model", "descriptive_context"), ("descriptive_epidemiology", "complementary_association"),
])
def test_each_family_offers_a_real_compilable_delta(family, expected):
    source = _source(family)
    # An unfinished absolute-risk context is not part of this result fixture.
    steps = tuple(s for s in source.completed_steps if "table:absolute_risk_context" not in s.expected_outputs)
    source = replace(source, completed_steps=steps)
    before = source.context.model_dump_json(), source.plan.model_dump_json()
    offered = suggest_followups(source)
    assert expected in {row.kind for row in offered.suggestions}
    for row in offered.suggestions:
        delta = row.delta
        assert row.target_action == "new_study"
        proposed, plan = compile_new_study(source.context, family=delta.analysis_family,
                                           source_plan=source.plan, primary_exposure=delta.primary_exposure)
        assert plan == delta.plan
        verdict = resolve_primary_capability(analysis_type=plan.analysis_type, plan=plan)
        assert verdict.coherent and verdict.execution_owner == "host_deterministic"
        assert delta.source_reading == FollowupReading.from_context(source.context, family=family)
        assert delta.source_reading.source_context_sha256 == canonical_sha256(source.context.model_dump(mode="json"))
        assert proposed.cohort == source.context.cohort
        assert plan.cohort.inclusion == source.plan.cohort.inclusion
        assert plan.cohort.exclusion == source.plan.cohort.exclusion
        assert plan.cohort.selection_mode == source.plan.cohort.selection_mode
        assert proposed.endpoint == source.context.endpoint
    assert before == (source.context.model_dump_json(), source.plan.model_dump_json())
    assert offered == suggest_followups(source)


def test_a_binary_result_with_paired_time_can_offer_survival():
    survival = _source("survival")
    # The same sealed timing/endpoint reading can describe a completed binary
    # analysis; only the result's family differs in this synthetic fixture.
    source = replace(survival, plan=survival.plan.model_copy(update={"analysis_type": "association_study"}))
    offered = suggest_followups(source)
    survival_row = next(row for row in offered.suggestions if row.kind == "complementary_survival")
    assert survival_row.delta.plan.analysis_type == "survival"
    missing = source.context.model_copy(update={"variables": [v for v in source.context.variables if v.name != "followup_days_90d"]})
    assert "complementary_survival" not in {row.kind for row in suggest_followups(replace(source, context=missing)).suggestions}


def test_a_refused_source_database_offers_nothing():
    source = _source("survival")
    source.context.cohort.database = "eicu"
    result = suggest_followups(source)
    assert result.suggestions == []
    assert "followup_database_refused" in result.unavailable_actions


def test_already_reported_description_is_not_suggested_again():
    source = _source("association_study")
    assert "descriptive_context" not in {row.kind for row in suggest_followups(source).suggestions}


@pytest.mark.parametrize("coherent,execution,validation", [
    (True, "agent_coded", "reportable"), (False, "host_deterministic", "reportable"),
    (True, "host_deterministic", "unsupported"),
])
def test_unclaimed_capability_is_never_offered(monkeypatch, coherent, execution, validation):
    from easyicu.webserver import research_followup_compile as owner

    source = _source("descriptive_epidemiology")
    monkeypatch.setattr(owner, "resolve_primary_capability", lambda **_: SimpleNamespace(
        coherent=coherent, execution_owner=execution, scientific_validation=validation))
    assert suggest_followups(source).suggestions == []


def test_survival_followup_does_not_select_an_unbound_adjustment_roster():
    with pytest.raises(ValueError, match="followup_family_unavailable"):
        compile_new_study(survival_context(), family="survival")


def test_prediction_description_selects_only_a_declared_baseline_predictor():
    source = _source("prediction_model")
    context = source.context.model_copy(deep=True)
    sex = next(v for v in context.variables if v.name == "sex")
    context.variables.append(sex.model_copy(update={"name": "aaa_unused"}))
    result = suggest_followups(replace(source, context=context))
    assert result.suggestions[0].delta.primary_exposure == "sex"
    context = context.model_copy(update={"variables": [v for v in context.variables if v.name != "sex"]})
    assert suggest_followups(replace(source, context=context)).suggestions == []


def test_unbound_adjustment_is_not_invented():
    source = _source("descriptive_epidemiology")
    source.context.user_preferences.covariate_selection = "planner_selectable"
    source.context.user_preferences.covariates = []
    source.context.user_preferences.covariate_rationales = {}
    source.context.user_preferences.covariate_temporal_roles = {}
    # Remove the source's exact roster as well as any host-recognized
    # baseline defaults; a follow-up cannot select confounders from prose.
    source.context.user_preferences.data_constraints = json.dumps({"stated_adjustments": [{
        "variable": "absent", "concept": "absent", "words": "the stated adjustment", "source_status": "unavailable",
    }]})
    assert suggest_followups(source).suggestions == []


class DatabaseOwner:
    def __init__(self, status="exact"):
        self.status = status
        self.compiled = []

    def registered_databases(self):
        return ["eicu", "miiv", "eicu"]

    def decide(self, source, *, database):
        return DatabaseDecision(database=database, source_context_sha256=canonical_sha256(source.context.model_dump(mode="json")),
                                status=self.status, reason_codes=("synthetic_capability_decision",))

    def dry_compile(self, source, *, decision):
        self.compiled.append(decision.database)
        context = source.context.model_copy(deep=True)
        context.cohort.database = decision.database
        return compile_new_study(context, family=source.plan.analysis_type, source_plan=source.plan)


@pytest.mark.parametrize("status", ["exact", "substitution", "refusal"])
def test_only_eligible_registered_databases_reach_the_launch_compiler(status):
    source = _source("descriptive_epidemiology")
    owner = DatabaseOwner(status)
    result = suggest_followups(source, database_owner=owner)
    rows = [row for row in result.suggestions if row.target_action == "other_database"]
    assert len(rows) == (0 if status == "refusal" else 1)
    assert owner.compiled == ([] if status == "refusal" else ["eicu"])
    if rows:
        assert rows[0].delta.capability_decision.status == status
        assert rows[0].delta.target_reading.database == "eicu"
    assert result.unavailable_actions == ()


def test_database_decision_must_bind_the_source_and_target():
    source = _source("descriptive_epidemiology")
    owner = DatabaseOwner()
    owner.decide = lambda *_args, **kwargs: DatabaseDecision(database=kwargs["database"], source_context_sha256="b" * 64,
                                                            status="exact", reason_codes=())
    assert not any(row.target_action == "other_database" for row in suggest_followups(source, database_owner=owner).suggestions)
    assert owner.compiled == []


@pytest.mark.parametrize("mismatch", ["decision_database", "compiled_database", "compiled_family", "unclaimed_owner", "scope"])
def test_other_database_admission_rechecks_the_compiled_target(mismatch, monkeypatch):
    from easyicu.webserver import research_followup_databases as owner_module

    source = _source("survival" if mismatch == "scope" else "descriptive_epidemiology")
    owner = DatabaseOwner()
    target = source.context.model_copy(deep=True)
    target.cohort.database = "aumc" if mismatch == "compiled_database" else "eicu"
    plan = source.plan
    if mismatch == "compiled_family":
        target, plan = compile_new_study(target, family="association_study", source_plan=source.plan)
    owner.dry_compile = lambda *_args, **_kwargs: (target, plan)
    if mismatch == "decision_database":
        original = owner.decide
        owner.decide = lambda *args, **kwargs: original(*args, **kwargs).model_copy(update={"database": "aumc"})
    if mismatch == "unclaimed_owner":
        monkeypatch.setattr(owner_module, "resolve_primary_capability", lambda **_: SimpleNamespace(
            coherent=True, execution_owner="agent_coded", scientific_validation="reportable"))
    assert not any(row.target_action == "other_database" for row in suggest_followups(source, database_owner=owner).suggestions)


def test_multi_database_without_its_owner_is_explicitly_unavailable():
    result = suggest_followups(_source("descriptive_epidemiology"))
    assert "multi_database_unavailable" in result.unavailable_actions
    assert not any(row.target_action == "other_database" for row in result.suggestions)


def test_rank_limit_and_ids_are_stable_across_registry_order():
    source = _source("descriptive_epidemiology")
    owner = DatabaseOwner()
    databases = ["eicu", "miiv", "aumc", "hirid", "mimic", "sic", "eicu"]
    owner.registered_databases = lambda: databases
    first = suggest_followups(source, database_owner=owner)
    databases.reverse()
    second = suggest_followups(source, database_owner=owner)
    assert first == second
    assert len(first.suggestions) == 4
    assert first.suggestions[0].kind == "complementary_association"
    ids = [row.suggestion_id for row in first.suggestions]
    assert len(ids) == len(set(ids))
    assert ids[1:] == sorted(ids[1:])
