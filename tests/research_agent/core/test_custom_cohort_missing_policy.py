"""A stated custom measurement criterion controls unknown rows explicitly."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.cohort.schema import CohortDataError, _build_cohort_with_flow
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition, CohortSchemaError, ConceptPredicate, TimeWindow,
    cohort_definition_sha,
)


def _measurement(policy=None, *, op="<=", concept="crea", value=4.0):
    return ConceptPredicate(
        concept_id=concept,
        time_window=TimeWindow(anchor="icu_admit", start_offset_hours=0.0, end_offset_hours=24.0),
        aggregation="first", op=op, value=value, missing_policy=policy,
    )


@pytest.mark.parametrize("kind", ["inclusion", "exclusion"])
@pytest.mark.parametrize("policy,unknown_kept", [("exclude", False), ("retain", True)])
def test_missing_handling_is_a_cohort_decision_for_both_criterion_roles(kind, policy, unknown_kept):
    data = pd.DataFrame({"stay_id": [1, 2, 3, 4], "crea": pd.Series([2.0, 4.0, 5.0, None], dtype="Float64")})
    cohort, flow = _build_cohort_with_flow(
        CohortDefinition(name="custom", **{kind: (_measurement(policy),)}), data,
    )
    # The two observed cutoff values are handled independently of missingness;
    # the boundary 4 is included by <=.
    expected = {1, 2} if kind == "inclusion" else {3}
    if unknown_kept:
        expected.add(4)
    assert set(cohort.stay_id) == expected
    assert flow[1]["n_excluded_missing"] == (0 if unknown_kept else 1)
    assert flow[1]["n_retained_missing"] == (1 if unknown_kept else 0)


@pytest.mark.parametrize("kind", ["inclusion", "exclusion"])
def test_stop_refuses_undetermined_measurements_only_in_remaining_cohort(kind):
    data = pd.DataFrame({"stay_id": [1, 2, 3], "age": [17, 18, 20], "crea": [None, 4.0, None]})
    definition = CohortDefinition(name="custom", inclusion=(_measurement(op=">=", concept="age", value=18),), **{kind: (_measurement("stop"),)} if kind == "exclusion" else {})
    if kind == "inclusion":
        definition = CohortDefinition(name="custom", inclusion=(_measurement(op=">=", concept="age", value=18), _measurement("stop")))
    with pytest.raises(CohortDataError, match="selected missing policy requires stopping"):
        _build_cohort_with_flow(definition, data)
    known_remaining = data.iloc[:2]
    cohort, flow = _build_cohort_with_flow(definition, known_remaining)
    assert flow[-1]["n_retained_missing"] == 0
    assert set(cohort.stay_id) == ({2} if kind == "inclusion" else set())


def test_policy_is_bound_to_definition_and_legacy_serialization_is_unchanged():
    legacy = _measurement()
    payload = legacy.to_dict()
    assert "missing_policy" not in payload
    assert ConceptPredicate.from_dict(payload) == legacy
    explicit = _measurement("retain")
    assert ConceptPredicate.from_dict(explicit.to_dict()) == explicit
    assert cohort_definition_sha(CohortDefinition(name="custom", inclusion=(legacy,))) != cohort_definition_sha(CohortDefinition(name="custom", inclusion=(explicit,)))


@pytest.mark.parametrize("policy", ["", "ignore", [], 0])
def test_invalid_policies_are_rejected(policy):
    with pytest.raises(CohortSchemaError, match="missing_policy"):
        _measurement(policy)


@pytest.mark.parametrize("op", ["missing", "not_missing"])
@pytest.mark.parametrize("policy", ["exclude", "retain", "stop"])
def test_an_explicit_missingness_predicate_cannot_have_a_conflicting_policy(op, policy):
    with pytest.raises(CohortSchemaError, match="missingness predicate"):
        _measurement(policy, op=op, value=None)

    from easyicu.research_agent.schema import AnalysisPlan

    legacy = AnalysisPlan(
        research_question="Select according to a recorded missing value.",
        cohort=CohortDefinition(name="custom", inclusion=(_measurement(None, op=op, value=None),)),
        steps=[],
    ).model_dump(mode="json")
    legacy["cohort"]["inclusion"][0]["missing_policy"] = policy
    with pytest.raises(ValueError, match="missingness predicate"):
        AnalysisPlan.model_validate(legacy)


def test_missing_at_an_event_time_is_unknown_not_a_proven_absence():
    from easyicu.research_agent.planning.cohort_contract import cohort_concept_id_scope

    with cohort_concept_id_scope(["custom_event"]):
        predicate = ConceptPredicate(
            concept_id="custom_event",
            time_window=TimeWindow(anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=24.0),
            aggregation="first", op="==", value=1, missing_policy="exclude",
        )
        data = pd.DataFrame({"stay_id": [1, 2, 3, 4], "custom_event": [1, 1, 0, None], "custom_event_time": [5.0, None, None, None]})
        cohort, flow = _build_cohort_with_flow(CohortDefinition(name="event", exclusion=(predicate,)), data)
    assert set(cohort.stay_id) == {3}
    assert flow[1]["n_excluded_missing"] == 2


def test_retained_missing_counts_participate_in_disclosure_review():
    from easyicu.research_agent.gates.publication_disclosure import is_subject_count_name
    assert is_subject_count_name("n_retained_missing")


def test_custom_policy_flows_into_recoverable_analysis_cohort_receipt(tmp_path):
    import json
    from easyicu.research_agent.cohort.schema import (
        load_materialized_analysis_cohort_result, materialize_locked_analysis_cohort,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    universe = tmp_path / "universe.parquet"
    pd.DataFrame({"stay_id": [1, 2, 3], "crea": [3.0, 5.0, None]}).to_parquet(universe, index=False)
    plan = AnalysisPlan(
        research_question="Patients with first creatinine at most 4 mg/dL in the stated window; retain unknowns.",
        cohort=CohortDefinition(name="custom", inclusion=(_measurement("retain"),)),
        robustness_specs=[], steps=[],
    )
    result = materialize_locked_analysis_cohort(run_dir=tmp_path, plan=plan, universe_path=universe)
    assert set(pd.read_parquet(result["path"])["stay_id"]) == {1, 3}
    flow = pd.read_csv(result["flow_path"])
    assert flow["n_retained_missing"].tolist() == [0, 1]
    provenance = json.loads((tmp_path / "cohort_analysis_provenance.json").read_text())
    assert provenance["cohort_definition"]["inclusion"][0]["missing_policy"] == "retain"
    assert provenance["cohort_flow"][1]["n_retained_missing"] == 1
    assert load_materialized_analysis_cohort_result(run_dir=tmp_path, plan=plan)


@pytest.mark.parametrize("op,value", [("<=", 4), (">=", 4), ("!=", 4)])
@pytest.mark.parametrize("kind", ["inclusion", "exclusion"])
@pytest.mark.parametrize("policy", ["exclude", "retain", "stop"])
def test_joint_criteria_apply_each_comparison_and_preserve_unknown_policy(op, value, kind, policy):
    # The first criterion narrows the population; a missing value on a row
    # already excluded by age must never cause the second criterion to stop.
    data = pd.DataFrame({"stay_id": [1, 2, 3, 4, 5], "age": [17, 18, 30, 50, 40], "crea": pd.Series([None, 3, 4, 5, None], dtype="Float64")})
    age = _measurement(op=">=", concept="age", value=18)
    target = _measurement(policy, op=op, value=value)
    definition = CohortDefinition(name="joint", inclusion=(age, target) if kind == "inclusion" else (age,), exclusion=(target,) if kind == "exclusion" else ())
    if policy == "stop":
        with pytest.raises(CohortDataError, match="missing policy"):
            _build_cohort_with_flow(definition, data)
        return
    cohort, flow = _build_cohort_with_flow(definition, data)
    observed = {2: 3, 3: 4, 4: 5}
    matches = {stay for stay, x in observed.items() if {"<=": x <= value, ">=": x >= value, "!=": x != value}[op]}
    expected = matches if kind == "inclusion" else set(observed) - matches
    if policy == "retain":
        expected.add(5)
    assert set(cohort.stay_id) == expected
    assert flow[-1]["n_retained_missing"] == int(policy == "retain")
    assert flow[-1]["n_excluded_missing"] == int(policy == "exclude")


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("kind", ["inclusion", "exclusion"])
@pytest.mark.parametrize("policy", ["exclude", "retain", "stop"])
def test_event_presence_or_absence_keeps_unknown_distinct(level, kind, policy):
    from easyicu.research_agent.planning.cohort_contract import cohort_concept_id_scope

    data = pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5], "custom_event": [1.0, 1.0, 1.0, 0.0, None],
        "custom_event_time": [5.0, None, 30.0, None, None],
    })
    with cohort_concept_id_scope(["custom_event"]):
        pred = ConceptPredicate(
            concept_id="custom_event", time_window=TimeWindow("icu_admission", 0.0, 24.0),
            aggregation="first", op="==", value=level, missing_policy=policy,
        )
        definition = CohortDefinition(name="events", **{kind: (pred,)})
        if policy == "stop":
            with pytest.raises(CohortDataError, match="missing policy"):
                _build_cohort_with_flow(definition, data)
            return
        cohort, flow = _build_cohort_with_flow(definition, data)
    matches = {1} if level == 1 else {3, 4}
    known = {1, 3, 4}
    expected = matches if kind == "inclusion" else known - matches
    if policy == "retain":
        expected |= {2, 5}
    assert set(cohort.stay_id) == expected
    assert flow[-1]["n_retained_missing"] == (2 if policy == "retain" else 0)
    assert flow[-1]["n_excluded_missing"] == (2 if policy == "exclude" else 0)


def test_stop_does_not_publish_partial_analysis_data(tmp_path):
    from easyicu.research_agent.cohort.schema import materialize_locked_analysis_cohort
    from easyicu.research_agent.schema import AnalysisPlan

    universe = tmp_path / "universe.parquet"
    pd.DataFrame({"stay_id": [1, 2], "crea": [3.0, None]}).to_parquet(universe, index=False)
    plan = AnalysisPlan(
        research_question="Keep those with a reviewed measurement criterion.",
        cohort=CohortDefinition(name="custom", inclusion=(_measurement("stop"),)),
        robustness_specs=[], steps=[],
    )
    result = materialize_locked_analysis_cohort(run_dir=tmp_path, plan=plan, universe_path=universe)
    assert result["status"] == "error"
    assert "selected missing policy requires stopping" in result["error"]
    assert not (tmp_path / "cohort_analysis.parquet").exists()
    assert not (tmp_path / "cohort_analysis_flow.csv").exists()
    assert not (tmp_path / "cohort_analysis_provenance.json").exists()


def test_legacy_plan_json_omits_unset_policy_while_explicit_policy_round_trips():
    from easyicu.research_agent.schema import AnalysisPlan

    def plan(policy):
        return AnalysisPlan(
            research_question="Cohort selection.",
            cohort=CohortDefinition(name="custom", inclusion=(_measurement(policy),)),
            robustness_specs=[], steps=[],
        )
    legacy = plan(None).model_dump(mode="json")
    assert "missing_policy" not in legacy["cohort"]["inclusion"][0]
    stated = plan("exclude").model_dump(mode="json")
    assert stated["cohort"]["inclusion"][0]["missing_policy"] == "exclude"
    assert AnalysisPlan.model_validate(stated).cohort.inclusion[0].missing_policy == "exclude"


@pytest.mark.parametrize("database", ["miiv", "eicu_demo"])
def test_two_generic_measurements_compile_and_select_for_two_database_families(database):
    import json
    from easyicu.research_agent.planning.population_compile import compile_population
    from easyicu.research_agent.planning.population_spec import PopulationSpec
    from easyicu.research_agent.schema import (
        CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences, VariableRole,
    )

    # These are synthetic descriptors with explicitly recorded units and summary
    # windows, not a claim about either database's native lab-unit completeness.
    context = ResearchContext(
        research_question="Include first creatinine at most 4 mg/dL within the first 24 hours; exclude maximum lactate at least 2 mmol/L within the first 24 hours; exclude missing values.",
        cohort=CohortDescriptor(cohort_name="synthetic", database=database, n_stays=5, id_columns=["stay_id"], outcome_columns=["death"]),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(name="crea_first", source_concept="crea", role=VariableRole.LAB, dtype="float64", unit="mg/dL"),
            ConceptDescriptor(name="lact_max", source_concept="lact", role=VariableRole.LAB, dtype="float64", unit="mmol/L"),
        ],
        user_preferences=UserPreferences(data_constraints=json.dumps({"materialization_window": {"role": "outer_observation_window", "anchor": "ICU admission", "hours": 24}})),
    )
    spec = PopulationSpec.model_validate({"criteria": [
        {"id": "c1", "quote": "Include first creatinine at most 4 mg/dL within the first 24 hours", "source": "question", "role": "include", "kind": "measurement", "concept": "crea", "summary": "first", "window": {"start_hours": 0, "end_hours": 24}, "op": "<=", "value": 4, "unit": "mg/dL", "missing_policy": "exclude"},
        {"id": "c2", "quote": "exclude maximum lactate at least 2 mmol/L within the first 24 hours", "source": "question", "role": "exclude", "kind": "measurement", "concept": "lact", "summary": "max", "window": {"start_hours": 0, "end_hours": 24}, "op": ">=", "value": 2, "unit": "mmol/L", "missing_policy": "exclude"},
    ]})
    compiled = compile_population(spec, context, time_zero_hours=24.0)
    assert [item.disposition for item in compiled.criteria] == ["applied_by_plan", "applied_by_plan"]
    # The same first-day criteria cannot be applied at ICU admission time:
    # their measurements may occur later than time zero.
    premature = compile_population(spec, context, time_zero_hours=0.0)
    assert all(item.reason == "population_determined_after_time_zero" for item in premature.criteria)
    data = pd.DataFrame({"stay_id": [1, 2, 3, 4, 5], "crea_first": [3., 4., 5., None, 2.], "lact_max": [1., 2., 1., 1., None]})
    cohort, flow = _build_cohort_with_flow(compiled.cohort_definition(), data)
    assert set(cohort.stay_id) == {1}
    assert flow[1]["n_excluded_missing"] == 1
    assert flow[2]["n_excluded_missing"] == 1


def test_stop_on_a_typed_universe_explains_why_and_does_not_publish(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pyarrow as pa
    from easyicu.research_agent.cohort.schema import materialize_locked_analysis_cohort
    from easyicu.research_agent.intake import materialized_metadata
    from easyicu.research_agent.intake.materialized_metadata import MaterializedMetadataError
    from easyicu.research_agent.schema import AnalysisPlan

    universe = tmp_path / "universe.parquet"
    frame = pd.DataFrame({"stay_id": [1, 2], "crea": [3.0, None]})
    frame.to_parquet(universe, index=False)
    typed_parent = SimpleNamespace(authority=SimpleNamespace(semantic_provenance={}))
    monkeypatch.setattr(materialized_metadata, "load_verified_materialized_cohort_authority", lambda *_: typed_parent)
    monkeypatch.setattr(materialized_metadata, "read_verified_materialized_cohort_table", lambda *_, **__: pa.Table.from_pandas(frame))
    plan = AnalysisPlan(
        research_question="Apply a criterion with a reviewed stop-on-missing policy.",
        cohort=CohortDefinition(name="custom", inclusion=(_measurement("stop"),)),
        robustness_specs=[], steps=[],
    )
    with pytest.raises(MaterializedMetadataError, match="selected missing-data policy requires stopping"):
        materialize_locked_analysis_cohort(run_dir=tmp_path, plan=plan, universe_path=universe)
    assert not (tmp_path / "cohort_analysis.parquet").exists()
