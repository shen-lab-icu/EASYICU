"""Declared missingness policies and optional summaries on synthetic metadata."""
from __future__ import annotations

from tests.support.descriptive_plan import _context, _payload, _plan
from tests.support.distribution_policies import EXCLUDE_MISSING, STOP_MISSING

import json
import hashlib

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import (
    family_spec_response_shape,
    family_spec_structured_output_request,
    family_spec_user_prompt,
    parse_family_plan_spec,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT, run_exposure_outcome_distribution_from_env,
)
from easyicu.research_agent.planning.family_spec import build_descriptive_skeleton
from easyicu.research_agent.planning.family_spec.contract import (
    AcceptedBaselineRow,
    DistributionMissingPolicies,
    FamilyPlanSpec,
    FamilySpecError,
    validate_family_plan_spec,
)
from easyicu.research_agent.schema import MissingnessProfile
from easyicu.research_agent.reporting.article_contract import build_article_analysis_contract

from .family_spec_fixtures import _descriptive_context, _request, _run


@pytest.fixture
def distribution_rows(tmp_path, monkeypatch):
    """Exercise the public runner with digest-bound synthetic inputs."""
    cohort = tmp_path / "cohort.parquet"
    manifest = tmp_path / "resolved_inputs.json"
    out = tmp_path / "outputs"
    monkeypatch.setenv("STEP_OUT_DIR", str(out))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest))

    def run(frame, *, spec):
        frame.to_parquet(cohort, index=False)
        identity = {
            "input_key": "artifact:analysis_cohort", "product": "analysis_cohort",
            "declared_kind": "artifact", "evidence_id": "synthetic-cohort",
            "sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
        }
        manifest.write_text(json.dumps({"step_id": "synthetic_distribution", "inputs": {
            "artifact:analysis_cohort": {
                **identity, "relative_path": cohort.name, "identity_row": identity,
                "product_contract": {"columns": list(frame), "row_count": len(frame)},
            },
        }}))
        summary = run_exposure_outcome_distribution_from_env(
            spec_payload=spec.model_dump(mode="json"), typed_cohort_input="artifact:analysis_cohort",
        )
        return pd.read_csv(out / summary["output_files"][EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]).to_dict("records")

    return run


def _rows():
    # The two NA sex rows represent one source-domain failure and one raw NA,
    # already combined by the source factor owner. No source values are guessed.
    return pd.DataFrame({
        'sex': ['Female', 'Female', 'Female', 'Male', 'Male', 'Male', None, None],
        'death': pd.Series([1, 0, pd.NA, 0, 1, pd.NA, 1, 0], dtype='Int8'),
    })


def test_a_declared_stop_policy_cannot_accept_missingness_without_a_new_plan(distribution_rows):
    spec = _plan(policies=STOP_MISSING).steps[1].exposure_outcome_distribution_spec
    assert spec.interval_method == 'none_counts_only'
    with pytest.raises(RuntimeError, match='missing_exposure_policy'):
        distribution_rows(_rows().assign(death=0), spec=spec)
    with pytest.raises(RuntimeError, match='missing_outcome_policy'):
        distribution_rows(_rows().dropna(subset=['sex']), spec=spec)
    # The approved policy remains fail-closed after execution refuses.
    assert spec.missing_outcome_policy == 'fail_closed'


def test_zero_metadata_rows_require_a_policy_and_expose_unknown_missingness_on_both_routes():
    request = _request(_context(), cohort_mode='all_input_rows')
    assert request.distribution_missingness_unobserved
    assert request.distribution_missing_policies is None
    payload = _payload(request)
    payload.pop('distribution_missing_policies')
    with pytest.raises(FamilySpecError, match='family_spec_missing_policies_required'):
        parse_family_plan_spec(json.dumps(payload), request)
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    assert 'distribution_missing_policies' in schema['required']
    policies = schema['properties']['distribution_missing_policies']
    assert policies['type'] == 'object'
    assert set(policies['required']) == set(STOP_MISSING)
    for key in STOP_MISSING:
        assert set(policies['properties'][key]['enum']) == {STOP_MISSING[key], EXCLUDE_MISSING[key]}
    assert 'distribution_missing_policies' in family_spec_response_shape(request)
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    assert '"missingness_unobserved": true' in prompt


def test_declared_counts_plan_omits_baseline_and_reports_each_denominator_without_imputing_death(distribution_rows):
    plan = _plan()
    assert [s.step_id for s in plan.steps] == [
        'cohort_accounting', 'exposure_outcome_distribution', 'measurement_audit',
        'family_cohort_flow_figure',
    ]
    assert not any(s.table_one_spec is not None for s in plan.steps)
    spec = plan.steps[1].exposure_outcome_distribution_spec
    assert spec.model_dump(include=set(EXCLUDE_MISSING)) == EXCLUDE_MISSING
    assert spec.outcome_levels == [0, 1] and spec.interval_method == 'none_counts_only'
    assert spec.confidence_level is None and spec.risk_difference_contrast is None
    assert '不生成基线表' in ' '.join(plan.design_selection.selected.reviewable_plan)
    rows = distribution_rows(_rows(), spec=spec)
    groups = {r['exposure_level']: r for r in rows if r['exposure_level'] in {'Female', 'Male'}}
    assert set(groups) == {'Female', 'Male'}
    for row in groups.values():
        assert (row['n_rows'], row['outcome_denominator'], row['outcome_events'], row['outcome_missing_n']) == (3, 2, 1, 1)
        assert row['missing_exposure_excluded_n'] == 2
        assert row['outcome_rate_pct'] == 50.0
        assert pd.isna(row['ci_low_pct']) and pd.isna(row['ci_high_pct'])
    assert len(_rows()) == 8  # Distribution exclusions do not rewrite the cohort.


def test_an_explicit_stop_policy_is_never_changed_when_prepared_rows_have_missing_values(distribution_rows):
    spec = _plan(policies=STOP_MISSING).steps[1].exposure_outcome_distribution_spec
    with pytest.raises(RuntimeError, match='missing_exposure_policy'):
        distribution_rows(_rows(), spec=spec)
    assert spec.model_dump(include=set(STOP_MISSING)) == STOP_MISSING


def test_unmapped_observed_sex_and_outcome_levels_still_fail_closed(distribution_rows):
    spec = _plan().steps[1].exposure_outcome_distribution_spec
    for column, value in [('sex', 'Other'), ('death', 2)]:
        frame = _rows()
        frame.loc[0, column] = value
        with pytest.raises(RuntimeError, match='undeclared|declared'):
            distribution_rows(frame, spec=spec)


@pytest.mark.parametrize('retained', [False, True])
def test_requested_or_previously_accepted_baseline_rows_are_retained(retained):
    request = _request(_context(), cohort_mode='all_input_rows')
    if retained:
        request = request.model_copy(update={'accepted_baseline_rows': [
            AcceptedBaselineRow(required='age', columns=['age'], summary='both'),
        ]})
    spec = FamilyPlanSpec.model_validate(_payload(request, baseline=[] if retained else ['age']))
    draft = build_descriptive_skeleton(request, spec)
    baseline = next(m.step for m in draft.materializations if m.step.step_id == 'baseline_context')
    assert [v.name for v in baseline.table_one_variables] == ['age']


@pytest.mark.parametrize("policy,denominator", [
    (EXCLUDE_MISSING, "all_declared_rows"), (STOP_MISSING, "observed_outcome_rows"),
])
def test_inconsistent_denominator_is_rejected_before_any_plan_can_be_approved(policy, denominator):
    with pytest.raises(ValidationError, match='denominator'):
        DistributionMissingPolicies.model_validate({**policy, 'denominator_policy': denominator})


def test_fixture_does_not_replace_an_explicit_invalid_policy_with_an_exclusion():
    request = _request(_context(), cohort_mode='all_input_rows')
    with pytest.raises(ValidationError):
        FamilyPlanSpec.model_validate(_payload(request, policies={}))


def test_optional_descriptive_baseline_does_not_remove_other_family_or_primary_obligations():
    context = _context()
    contract = build_article_analysis_contract(context, analysis_type='descriptive_epidemiology')
    assert set(contract.required_roles) == {'cohort_accounting', 'data_quality', 'distribution'}
    assert 'baseline_context' in contract.recommended_roles
    for family in ['association_study', 'prediction_model']:
        assert 'baseline_context' in build_article_analysis_contract(context, analysis_type=family).required_roles


def test_missing_policies_are_not_a_choice_on_an_association_request():
    from .family_spec_fixtures import _context as association_context, _spec_payload

    request = _request(association_context())
    spec = FamilyPlanSpec.model_validate({
        **_spec_payload(request), "distribution_missing_policies": STOP_MISSING,
    })
    with pytest.raises(FamilySpecError, match="missing_policies_not_applicable"):
        validate_family_plan_spec(spec, request)


@pytest.mark.parametrize("coordinate", ["primary_exposure", "target_outcome"])
def test_missing_context_variables_are_refused_before_descriptive_request_construction(coordinate):
    context = _context().model_copy(update={coordinate: "unregistered_variable"})
    with pytest.raises(FamilySpecError, match="template_unavailable"):
        _request(context)


@pytest.mark.parametrize("candidate,observed_default", [
    (STOP_MISSING, EXCLUDE_MISSING), (EXCLUDE_MISSING, STOP_MISSING),
])
def test_candidate_policy_is_the_reviewable_choice_even_when_request_has_a_default(candidate, observed_default):
    request = _request(_context(), cohort_mode="all_input_rows").model_copy(update={
        "distribution_missing_policies": DistributionMissingPolicies.model_validate(observed_default),
    })
    spec = FamilyPlanSpec.model_validate(_payload(request, policies=candidate))
    validate_family_plan_spec(spec, request)
    draft = build_descriptive_skeleton(request, spec)
    distribution, = [item.step for item in draft.materializations
                     if item.step.step_id == "exposure_outcome_distribution"]
    assert distribution.model_dump(include=set(candidate)) == candidate
    assert request.distribution_missing_policies.model_dump() == observed_default
