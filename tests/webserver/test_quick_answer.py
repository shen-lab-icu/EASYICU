"""Host quick answers use synthetic prepared rows, never a provider or ASGI."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import pandas as pd
import pytest

from easyicu.concept.catalog import CONCEPT_DICTIONARY
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.research_agent.schema import (
    CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences, VariableRole,
)
from easyicu.webserver.quick_answer import (
    QuickApproval, QuickReading, QuickStop, bind_prepared_source, execute_quick_answer,
    prepare_quick_answer,
)
from easyicu.webserver.quick_answer.planning import bind_concept_names, qualification_stop


@pytest.fixture
def prepared(tmp_path):
    frame = pd.DataFrame({
        "stay_id": range(60), "subject_id": range(60),
        "sex": ["Male"] * 30 + ["Female"] * 30,
        "age": [40.0] * 60,
        "lact_max": [1.0, 2.0, 3.0] * 10 + [2.0, 3.0, 4.0] * 10,
        "icu_death": [0] * 40 + [1] * 20,
    })
    path = tmp_path / "synthetic.parquet"
    frame.to_parquet(path, index=False)
    identity = SourceIdentity(database="eicu_demo", registration_id="src_synthetic")
    variables = [
        ConceptDescriptor(name=name, dtype="int64", role="id")
        for name in ("stay_id", "subject_id")
    ] + [
        ConceptDescriptor(name="sex", dtype="object", role="demographic",
                          observed_domain={"levels": ["Male", "Female"]}),
        ConceptDescriptor(name="age", dtype="float64", role="demographic", unit="years"),
        ConceptDescriptor(name="lact_max", source_concept="lact", dtype="float64",
                          role="lab", unit="mmol/L", analysis_window="icu_admission[0,24]h"),
        ConceptDescriptor(name="icu_death", dtype="int64", role="outcome",
                          observed_domain={"levels": [0, 1]}),
    ]
    context = ResearchContext(
        research_question="Prepared source",
        cohort=CohortDescriptor(cohort_name="Synthetic stays", database="eicu_demo",
                                n_stays=60, id_columns=["stay_id", "subject_id"]),
        variables=variables,
        user_preferences=UserPreferences(data_constraints=json.dumps({
            "source_identity": identity.to_dict(),
        })),
    )
    return bind_prepared_source(path, context=context, identity=identity,
                                unit_column="stay_id", patient_column="subject_id")


def reading(question="What proportion of stays are male?", **updates):
    return QuickReading.model_validate({
        "question": question, "intent": "describe", "population": {"criteria": []},
        "measure": {"concept": "sex", "statistic": "proportion", "level": "Male"},
        **updates,
    })


def test_one_approval_runs_the_existing_kernel_and_binds_every_number(prepared, tmp_path):
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    assert proposal.status == "approval_required"
    assert proposal.card.source == prepared.identity.to_dict()
    start = monotonic()
    answer = execute_quick_answer(
        proposal, approval=QuickApproval(card_sha256=proposal.card.sha256),
        source=prepared, provider=None, run_dir=tmp_path / "run",
    )
    assert monotonic() - start < 10
    assert answer.status == "ok"
    assert answer.schema_version == "easyicu.quick_answer/1"
    values = {row.statistic: row.value for row in answer.rows}
    assert values["percentage"] == 50
    assert values["numerator"] == 30
    assert values["denominator"] == 60
    assert len(answer.sentences) == len(answer.rows)
    assert all(row.evidence.source == prepared.identity.to_dict() for row in answer.rows)
    assert all(sentence.evidence == row.evidence
               for sentence, row in zip(answer.sentences, answer.rows, strict=True))
    assert answer.expand.reading == reading()
    assert answer.expand.action == "expand_into_full_study"
    assert not list((tmp_path / "run").glob("*.parquet"))


def rebound(source, *, frame=None, database=None, context=None, patient_column="subject_id"):
    if frame is not None:
        frame.to_parquet(source.path, index=False)
    context = context or source.context
    identity = source.identity
    if database:
        identity = replace(identity, database=database)
        context = context.model_copy(update={
            "cohort": context.cohort.model_copy(update={"database": database}),
            "user_preferences": UserPreferences(data_constraints=json.dumps({"source_identity": identity.to_dict()})),
        })
    return bind_prepared_source(source.path, context=context, identity=identity,
                                unit_column=source.unit_column, patient_column=patient_column)


def run(proposal, source, tmp_path, *, provider=None, accept_substitutions=False):
    return execute_quick_answer(
        proposal, approval=QuickApproval(card_sha256=proposal.card.sha256,
                                        accept_substitutions=accept_substitutions),
        source=source, provider=provider, run_dir=tmp_path / "run",
    )


@pytest.mark.parametrize("statistic,expected", [
    ("mean", {"mean": 2.5}), ("median", {"median": 2.5, "q1": 2.0, "q3": 3.0}),
    ("iqr", {"median": 2.5, "q1": 2.0, "q3": 3.0}), ("count", {"nonmissing_n": 60}),
])
def test_windowed_measure_uses_the_descriptive_kernel(prepared, tmp_path, monkeypatch, statistic, expected):
    from easyicu.webserver.quick_answer import analysis

    calls = []
    original = analysis.cohort_summary_table

    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(analysis, "cohort_summary_table", traced)
    request = reading("Summarize the first-day peak lactate.", measure={
        "concept": "lact", "statistic": statistic, "summary": "max",
        "window": {"start_hours": 0, "end_hours": 24},
    })
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.status == "approval_required", proposal.reason
    answer = run(proposal, prepared, tmp_path)
    assert answer.status == "ok", answer.reason
    actual = {row.statistic: row.value for row in answer.rows}
    assert all(actual[key] == value for key, value in expected.items())
    # A mean is not reported with quartiles.
    assert statistic in {"median", "iqr"} or not {"q1", "q3"} & actual.keys()
    assert calls == [True]


@pytest.mark.parametrize("statistic,test", [("mean", "welch_t"), ("median", "mann_whitney_u"),
                                           ("proportion", "chi_square")])
def test_two_group_comparisons_reuse_the_standard_test(prepared, tmp_path, monkeypatch, statistic, test):
    from easyicu.webserver.quick_answer import analysis

    calls = []
    original = analysis.build_grouped_table_one

    def traced(frame, spec):
        calls.append(spec)
        return original(frame, spec)

    monkeypatch.setattr(analysis, "build_grouped_table_one", traced)
    measure = ({"concept": "icu_death", "statistic": statistic, "level": 1} if statistic == "proportion" else
               {"concept": "lact", "statistic": statistic, "summary": "max",
                "window": {"start_hours": 0, "end_hours": 24}})
    # Give both groups enough events and non-events for aggregate egress.
    if statistic == "proportion":
        frame = pd.read_parquet(prepared.path)
        frame["icu_death"] = [0] * 15 + [1] * 15 + [0] * 12 + [1] * 18
        prepared = rebound(prepared, frame=frame)
    proposal = prepare_quick_answer(reading("Compare the observed values by sex.", intent="compare",
                                           grouping=["sex"], measure=measure), source=prepared, provider=None)
    assert proposal.status == "approval_required", proposal.reason
    answer = run(proposal, prepared, tmp_path)
    assert answer.status == "ok", answer.reason
    assert len(calls) == 1 and len(calls[0].group_levels) == 2
    assert any(row.statistic == "p_value:" + test and 0 <= row.value <= 1 for row in answer.rows)
    for row, sentence in zip(answer.rows, answer.sentences, strict=True):
        assert sentence.text == f"The recorded value is {row.value}."


@pytest.mark.parametrize("question,updates,reason", [
    ("Compare mortality accounting for age.", {}, "quick_requires_adjusted_study"),
    ("Compare mortality after balancing sex.", {}, "quick_requires_adjusted_study"),
    ("校正年龄后，两组死亡比例是多少？", {}, "quick_requires_adjusted_study"),
    ("What proportion are still alive at day 28?", {}, "quick_requires_survival_study"),
    ("What proportion died within 28 days?", {}, "quick_requires_survival_study"),
    ("到第 28 天的存活比例？", {}, "quick_requires_survival_study"),
    ("Predict the average next-day value.", {}, "quick_requires_prediction_study"),
    ("What is the effect of treatment?", {}, "quick_requires_causal_study"),
    ("Describe clusters.", {}, "quick_requires_clustering_study"),
    ("A simple proportion?", {"adjustments": ["age"]}, "quick_requires_adjusted_study"),
    ("A simple proportion?", {"endpoint_kind": "fixed_horizon"}, "quick_requires_survival_study"),
    ("A simple proportion?", {"measure": {"concept": "mort_28d", "statistic": "proportion"}}, "quick_requires_survival_study"),
    ("A simple proportion?", {"ambiguities": ["population"]}, "quick_reading_ambiguous"),
    ("A simple proportion?", {"grouping": ["sex", "age"]}, "quick_grouping_scope_exceeded"),
    ("A simple proportion?", {"intent": "multistep"}, "quick_requires_multistep_study"),
])
def test_complex_or_ambiguous_readings_offer_full_study(prepared, question, updates, reason):
    request = reading(question, **updates)
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.status == "full_study" and proposal.reason == reason
    assert proposal.card is None and proposal.expand.reading == request


def test_original_question_cannot_be_replaced_by_a_simpler_reading(prepared):
    proposal = prepare_quick_answer(reading(), question="Adjust the proportion for age.", source=prepared, provider=None)
    assert proposal.reason == "quick_reading_ambiguous"
    assert proposal.expand.reading.question == "Adjust the proportion for age."


def test_a_clinical_name_binds_to_the_concept_the_question_names(prepared, tmp_path):
    # pilot1 10-26: the interpreter typed this question completely, but by
    # clinical name, and filed its conventional readings as ambiguities.
    question = "在 eICU demo 中，入 ICU 第一天最高乳酸的中位数是多少？男女之间有差别吗？"
    request = reading(question, intent="compare", grouping=["gender"], measure={
        "concept": "lactate", "statistic": "median", "summary": "max",
        "window": {"start_hours": 0, "end_hours": 24},
    })
    proposal = prepare_quick_answer(request, source=prepared, provider=None, question=question)
    assert proposal.status == "approval_required", proposal.reason
    plan = proposal.card.plan
    assert (plan.measure_column, plan.group_column) == ("lact_max", "sex")
    assert (plan.reading.measure.concept, plan.reading.grouping) == ("lact", ("sex",))
    assert run(proposal, prepared, tmp_path).status == "ok"


@pytest.mark.parametrize("name", ["icu_type", "unit_type", "unittype", "ICU type", "ICU 类型", "type of ICU"])
def test_an_icu_type_name_binds_to_the_unit_type_concept(name):
    from easyicu.webserver.study_intent import concept_named_by

    assert concept_named_by(name) == "icu_unit_type"
    assert concept_named_by("icu") is None and concept_named_by("ICU patients") is None


def with_group(prepared, concept, levels):
    frame = pd.read_parquet(prepared.path)
    frame[concept] = [levels[0]] * 30 + [levels[1]] * 30
    variable = ConceptDescriptor(
        name=concept, dtype=str(frame[concept].dtype), observed_domain={"levels": list(levels)},
    )
    variables = [v for v in prepared.context.variables if v.name != concept] + [variable]
    return rebound(prepared, frame=frame, context=prepared.context.model_copy(update={"variables": variables}))


@pytest.mark.parametrize(("concept", "label", "levels"), [
    ("sex", CONCEPT_DICTIONARY["sex"][0], ("Male", "Female")),
    ("sex", CONCEPT_DICTIONARY["sex"][1], ("Male", "Female")),
    ("adm", CONCEPT_DICTIONARY["adm"][0], ("med", "surg")),
    ("adm", CONCEPT_DICTIONARY["adm"][1], ("med", "surg")),
    ("loop_diuretic", CONCEPT_DICTIONARY["loop_diuretic"][0], (0, 1)),
    ("loop_diuretic", CONCEPT_DICTIONARY["loop_diuretic"][0].lower(), (0, 1)),
    ("loop_diuretic", CONCEPT_DICTIONARY["loop_diuretic"][1], (0, 1)),
])
def test_a_registered_display_label_binds_the_prepared_group(prepared, concept, label, levels):
    question = f"What is the median age by {label}?"
    request = reading(question, grouping=[label], measure={"concept": "age", "statistic": "median"})
    source = with_group(prepared, concept, levels)

    assert bind_concept_names(request).grouping == (concept,)
    proposal = prepare_quick_answer(request, source=source, provider=None, question=question)
    assert (proposal.status, proposal.reason) == ("approval_required", "quick_qualified")
    assert proposal.card.plan.group_column == concept
    assert proposal.card.plan.group_levels == levels


def test_an_exact_group_label_takes_priority_over_a_shorter_phrase_match(prepared):
    # The case-insensitive catalog recognizes dotted I. Its lowercase form
    # contains a combining dot, so the phrase reader sees only "blood pressure".
    label = CONCEPT_DICTIONARY["dbp"][0].replace("i", "İ", 1)
    question = f"What is the median age by {label}?"
    request = reading(question, grouping=[label], measure={"concept": "age", "statistic": "median"})
    source = with_group(with_group(prepared, "dbp", (60, 80)), "sbp", (100, 120))

    proposal = prepare_quick_answer(request, source=source, provider=None, question=question)
    assert (proposal.status, proposal.reason) == ("approval_required", "quick_qualified")
    assert proposal.card.plan.group_column == "dbp"
    assert proposal.card.plan.group_levels == (60, 80)


@pytest.mark.parametrize("label", ["local triage category", CONCEPT_DICTIONARY["o2sat"][1]])
def test_an_unknown_or_ambiguous_group_name_stops_with_a_typed_reason(prepared, label):
    question = f"What is the median age by {label}?"
    request = reading(question, grouping=[label], measure={"concept": "age", "statistic": "median"})
    source = with_group(with_group(prepared, "spo2", (94, 98)), "o2sat", (93, 97))

    assert bind_concept_names(request).grouping == (label,)
    proposal = prepare_quick_answer(request, source=source, provider=None, question=question)
    assert (proposal.status, proposal.reason) == ("stopped", "quick_capability_unassessed")
    assert proposal.card is None


@pytest.mark.parametrize(("replay", "update_distribution", "stop"), [
    ("r2", False, None),
    ("r3", False, "quick_reading_ambiguous"),
    ("r3", True, None),
])
def test_demo_icu_type_replays_use_the_canonical_group(prepared, replay, update_distribution, stop):
    path = Path(__file__).with_name("fixtures") / "quick_answer" / f"dH5_{replay}.reading.json"
    request = QuickReading.model_validate(json.loads(path.read_text(encoding="utf-8"))["reading"])
    if update_distribution:
        # Replay the recorded label with the distribution contract landed after this reading.
        request = request.model_copy(update={
            "measure": request.measure.model_copy(update={"statistic": "distribution"}),
            "ambiguities": (),
        })
    source = with_group(prepared, "icu_unit_type", ("MICU", "Med-Surg ICU"))

    bound = bind_concept_names(request)
    assert bound.grouping == ("icu_unit_type",)
    assert qualification_stop(bound) == stop
    proposal = prepare_quick_answer(request, source=source, provider=None, question=request.question)
    if stop:
        assert (proposal.status, proposal.reason) == ("full_study", stop)
        assert proposal.card is None
    else:
        assert (proposal.status, proposal.reason) == ("approval_required", "quick_qualified")
        assert (proposal.card.plan.measure_column, proposal.card.plan.group_column) == ("lact_max", "icu_unit_type")
        assert proposal.card.plan.reading.measure.statistic == request.measure.statistic


def test_icu_stay_by_icu_type_is_answered_not_left_unassessed(tmp_path):
    # pilot1 11-16 (dD3): the interpreter typed the grouping as "icu_type";
    # unbound, it stopped as "not assessed for this database" although the
    # eICU demo carries both the unit type and ICU length of stay.
    frame = pd.DataFrame({
        "stay_id": range(40), "subject_id": range(40),
        "icu_unit_type": ["MICU"] * 20 + ["Med-Surg ICU"] * 20,
        "los_icu": [1.0, 2.0, 3.0, 4.0] * 5 + [2.0, 3.0, 4.0, 5.0] * 5,
    })
    path = tmp_path / "unit_type.parquet"
    frame.to_parquet(path, index=False)
    identity = SourceIdentity(database="eicu_demo", registration_id="src_unit_type")
    variables = [ConceptDescriptor(name=name, dtype="int64", role="id") for name in ("stay_id", "subject_id")] + [
        ConceptDescriptor(name="icu_unit_type", dtype="object", role="demographic",
                          observed_domain={"levels": ["MICU", "Med-Surg ICU"]}),
        ConceptDescriptor(name="los_icu", dtype="float64", role="outcome", unit="days"),
    ]
    context = ResearchContext(
        research_question="Prepared source",
        cohort=CohortDescriptor(cohort_name="Synthetic stays", database="eicu_demo",
                                n_stays=40, id_columns=["stay_id", "subject_id"]),
        variables=variables,
        user_preferences=UserPreferences(data_constraints=json.dumps({"source_identity": identity.to_dict()})),
    )
    source = bind_prepared_source(path, context=context, identity=identity,
                                  unit_column="stay_id", patient_column="subject_id")
    question = "在 eICU demo 中，不同 ICU 类型的患者，ICU 住院时长有什么差别？"
    request = reading(question, intent="compare", grouping=["icu_type"],
                      measure={"concept": "icu_los", "statistic": "median"})
    proposal = prepare_quick_answer(request, source=source, provider=None, question=question)
    assert proposal.status == "approval_required", proposal.reason
    assert (proposal.card.plan.measure_column, proposal.card.plan.group_column) == ("los_icu", "icu_unit_type")
    assert run(proposal, source, tmp_path).status == "ok"


def test_a_name_that_reads_as_no_concept_stays_as_typed(prepared):
    request = reading("Summarize the first-day peak glucose.", measure={
        "concept": "glucose", "statistic": "median", "summary": "max",
        "window": {"start_hours": 0, "end_hours": 24},
    })
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.card is None and proposal.reason == "quick_capability_unassessed"
    assert proposal.expand.reading.measure.concept == "glucose"


def test_the_interpreter_is_told_a_conventional_reading_is_not_an_ambiguity():
    text = QuickReading.model_json_schema()["properties"]["ambiguities"]["description"]
    assert "is not an ambiguity" in text and "stops the quick answer" in text
    assert all(term in text for term in ("distribution", "spread", "summary", "measure.statistic"))


@pytest.mark.parametrize("question", [
    "入 ICU 第一天接受有创机械通气的患者占多少？",
    "What share of stays received invasive ventilation on the first day?",
])
def test_a_named_kind_the_data_do_not_separate_is_not_answered_on_every_kind(prepared, question):
    proposal = prepare_quick_answer(reading(question), source=prepared, provider=None)
    assert proposal.status == "full_study" and proposal.reason == "quick_named_level_undistinguished"
    assert proposal.card is None


def test_missing_window_falls_back_without_guessing(prepared):
    proposal = prepare_quick_answer(reading(measure={"concept": "lact", "statistic": "median", "summary": "max"}),
                                    source=prepared, provider=None)
    assert proposal.status == "full_study" and proposal.reason == "quick_window_unresolved"


def test_population_compiles_and_filters_before_counting(prepared, tmp_path):
    frame = pd.read_parquet(prepared.path)
    frame["age"] = [16.0] * 20 + [40.0] * 40
    prepared = rebound(prepared, frame=frame)
    request = reading("Count stays aged at least 18 years.", measure={"statistic": "count"}, population={"criteria": [{
        "id": "c1", "kind": "age_years", "source": "question", "role": "include",
        "quote": "aged at least 18 years", "min_years": 18,
    }]})
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.status == "approval_required", proposal.reason
    assert proposal.card.plan.population["criteria"][0]["disposition"] == "applied_by_plan"
    answer = run(proposal, prepared, tmp_path)
    assert answer.status == "ok", answer.reason
    assert [(row.statistic, row.value) for row in answer.rows] == [("cohort_n", 40)]


def test_population_counts_can_use_one_group_without_a_measure(prepared, tmp_path):
    request = reading("Count stays by sex.", measure={"statistic": "count"}, grouping=["sex"])
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.status == "approval_required", proposal.reason
    answer = run(proposal, prepared, tmp_path)
    assert answer.status == "ok", answer.reason
    assert {(row.group, row.value) for row in answer.rows} == {
        ("Overall", 60), ("Male", 30), ("Female", 30),
    }
    assert all(row.statistic == "cohort_n" for row in answer.rows)


def test_substitution_is_disclosed_and_requires_same_card_approval(prepared, tmp_path):
    from easyicu.research_agent.providers.mocks import MockLLMClient

    prepared = rebound(prepared, database="hirid")
    provider = MockLLMClient()
    request = reading("What is the hospital death proportion?",
                      measure={"concept": "hospital_death", "statistic": "proportion", "level": 1})
    proposal = prepare_quick_answer(request, source=prepared, provider=provider)
    assert proposal.status == "approval_required", proposal.reason
    assert proposal.card.capability.status == "substitution"
    assert proposal.card.plan.effective_reading.measure.concept == "icu_death"
    assert run(proposal, prepared, tmp_path, provider=provider).reason == "quick_substitution_approval_required"
    assert not (tmp_path / "run").exists()
    answer = run(proposal, prepared, tmp_path, provider=provider, accept_substitutions=True)
    assert answer.status == "ok", answer.reason
    assert answer.expand.reading == request
    assert answer.expand.effective_reading.measure.concept == "icu_death"


def test_refusal_names_the_missing_capability(prepared):
    from easyicu.research_agent.providers.mocks import MockLLMClient

    proposal = prepare_quick_answer(reading(measure={"concept": "los_hosp", "statistic": "median"}),
                                    source=rebound(prepared, database="hirid"), provider=MockLLMClient())
    assert proposal.status == "stopped" and proposal.reason == "quick_capability_refused"
    assert proposal.capability.missing[0].code == "hospital_los_unavailable"


def test_credentialed_source_needs_a_model_connection(prepared):
    proposal = prepare_quick_answer(reading(), source=rebound(prepared, database="eicu"), provider=None)
    assert proposal.reason == "quick_qualifying_provider_required"
    assert proposal.card is None


def test_credentialed_source_needs_no_data_handling_declaration(prepared):
    # Recorded, not required (owner decision 2026-10-27).
    proposal = prepare_quick_answer(reading(), source=rebound(prepared, database="eicu"), provider=object())
    assert proposal.reason == "quick_qualified"
    assert proposal.card is not None


def test_source_identity_must_be_the_contexts_exact_registration(prepared):
    with pytest.raises(QuickStop, match="quick_source_identity_mismatch"):
        bind_prepared_source(prepared.path, context=prepared.context,
                             identity=replace(prepared.identity, registration_id="src_other"), unit_column="stay_id")


def test_approval_is_bound_to_the_whole_card(prepared, tmp_path):
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    answer = execute_quick_answer(proposal, approval=QuickApproval(card_sha256="0" * 64),
                                   source=prepared, provider=None, run_dir=tmp_path / "run")
    assert answer.reason == "quick_approval_mismatch" and not (tmp_path / "run").exists()
    changed = proposal.model_copy(update={"card": proposal.card.model_copy(update={"interval_policy": "changed"})})
    assert run(changed, prepared, tmp_path).reason == "quick_approval_mismatch"


def test_changed_source_stops_before_execution(prepared, tmp_path):
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    frame = pd.read_parquet(prepared.path)
    frame["sex"] = "Male"
    frame.to_parquet(prepared.path, index=False)
    assert run(proposal, prepared, tmp_path).reason == "quick_source_changed"
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("change,reason", [
    ("small_complement", "quick_small_cells_suppressed"),
    ("unknown_level", "quick_levels_changed"),
    ("duplicate_unit", "quick_unit_identity_invalid"),
])
def test_egress_and_row_identity_fail_closed(prepared, tmp_path, change, reason):
    frame = pd.read_parquet(prepared.path)
    if change == "small_complement":
        frame["sex"] = ["Male"] * 59 + ["Female"]
    elif change == "unknown_level":
        frame.loc[0, "sex"] = "unrecognized"
    else:
        frame.loc[0, "stay_id"] = 1
    prepared = rebound(prepared, frame=frame)
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    answer = run(proposal, prepared, tmp_path)
    assert answer.reason == reason and not answer.rows and not answer.sentences
    assert not (tmp_path / "run" / "evidence").exists()


def test_a_comparison_cannot_assume_repeated_stays_are_independent(prepared, tmp_path):
    frame = pd.read_parquet(prepared.path)
    frame.loc[1, "subject_id"] = 0
    prepared = rebound(prepared, frame=frame)
    request = reading(intent="compare", grouping=["sex"], measure={"concept": "age", "statistic": "median"})
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert run(proposal, prepared, tmp_path).reason == "quick_repeated_units_require_full_study"


@pytest.mark.parametrize("question", ["eICU demo 里男性占多少？", "How many male stays?", "Counts only, please."])
def test_simple_questions_qualify_without_changing_the_normal_study_design(prepared, question):
    before = prepared.context.model_dump(mode="json")
    proposal = prepare_quick_answer(reading(question), source=prepared, provider=None)
    assert proposal.status == "approval_required"
    assert proposal.card.interval_policy == "point_summaries_only"
    assert prepared.context.model_dump(mode="json") == before


def test_grouped_descriptions_keep_all_declared_groups(prepared, tmp_path):
    proposal = prepare_quick_answer(reading(grouping=["sex"], measure={"concept": "age", "statistic": "median"}),
                                    source=prepared, provider=None)
    answer = run(proposal, prepared, tmp_path)
    assert answer.status == "ok", answer.reason
    assert {row.group: row.value for row in answer.rows if row.statistic == "median"} == {
        "Overall": 40, "Male": 40, "Female": 40,
    }


def test_creatinine_distribution_by_sex_uses_the_quick_summary(prepared, tmp_path):
    frame = pd.read_parquet(prepared.path).assign(
        crea=[1.0, 2.0, 3.0] * 6 + [None] * 12 + [2.0, 3.0, 4.0] * 10,
    )
    context = prepared.context.model_copy(update={"variables": [
        *prepared.context.variables, ConceptDescriptor(name="crea", dtype="float64", role="lab", unit="mg/dL"),
    ]})
    source = rebound(prepared, frame=frame, context=context)
    request = reading("肌酐按性别的分布如何？", grouping=["sex"],
                      measure={"concept": "crea", "statistic": "distribution"})
    proposal = prepare_quick_answer(request, source=source, provider=None)
    assert (proposal.status, proposal.reason) == ("approval_required", "quick_qualified")
    assert proposal.card.plan.test is None
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok", answer.reason
    rows_by_group = {group: {row.statistic: row.value for row in answer.rows if row.group == group}
                     for group in {row.group for row in answer.rows}}
    assert set(rows_by_group) == {"Overall", "Male", "Female"}
    assert rows_by_group["Male"] == pytest.approx({
        "cohort_n": 30, "nonmissing_n": 18, "missing_n": 12, "mean": 2, "std": (12 / 17) ** .5,
        "median": 2, "q1": 1, "q3": 3,
    })
    assert rows_by_group["Female"] == pytest.approx({
        "cohort_n": 30, "nonmissing_n": 30, "missing_n": 0, "mean": 3, "std": (20 / 29) ** .5,
        "median": 3, "q1": 2, "q3": 4,
    })
    assert all(row.unit == "mg/dL" for row in answer.rows if row.statistic in {"mean", "std", "median", "q1", "q3"})


def test_categorical_distribution_reports_every_level_within_each_group(prepared, tmp_path):
    frame = pd.read_parquet(prepared.path).assign(icu_death=[0] * 18 + [1] * 12 + [0] * 12 + [1] * 18)
    source = rebound(prepared, frame=frame)
    request = reading("Describe the distribution of observed ICU discharge status by sex.", grouping=["sex"],
                      measure={"concept": "icu_death", "statistic": "distribution"})
    proposal = prepare_quick_answer(request, source=source, provider=None)
    assert (proposal.status, proposal.reason) == ("approval_required", "quick_qualified")
    assert proposal.card.plan.measure_levels == (0, 1)
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok", answer.reason
    for group, level_counts in {"Overall": (30, 30), "Male": (18, 12), "Female": (12, 18)}.items():
        total = sum(level_counts)
        for level, count in enumerate(level_counts):
            assert {row.statistic: row.value for row in answer.rows if row.group == group and row.level == str(level)} == {
                "numerator": count, "denominator": total, "percentage": 100 * count / total,
            }


@pytest.mark.parametrize(("concept", "expected"), [
    ("age", {}), ("icu_death", {("0", "numerator"): 0, ("0", "denominator"): 0,
                              ("1", "numerator"): 0, ("1", "denominator"): 0}),
])
def test_distribution_keeps_counts_when_all_values_are_missing(prepared, tmp_path, concept, expected):
    frame = pd.read_parquet(prepared.path)
    frame[concept] = float("nan")
    source = rebound(prepared, frame=frame)
    request = reading("Describe the distribution.", measure={"concept": concept, "statistic": "distribution"})
    proposal = prepare_quick_answer(request, source=source, provider=None)
    assert proposal.status == "approval_required", proposal.reason
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok", answer.reason
    assert {(row.level, row.statistic): row.value for row in answer.rows} == {
        ("", "cohort_n"): 60, ("", "missing_n"): 60, ("", "nonmissing_n"): 0, **expected,
    }


@pytest.mark.parametrize(("question", "statistic", "grouping", "ambiguities", "reason"), [
    ("Describe the renal measure's distribution by sex.", "distribution", ["sex"],
     ["Either creatinine or urine output"], "quick_reading_ambiguous"),
    ("Describe the distribution of creatinine by sex and ICU type.", "distribution",
     ["sex", "icu_unit_type"], [], "quick_grouping_scope_exceeded"),
    ("Describe the distribution of creatinine by sex.", "median", ["sex"],
     ["Could mean quantiles, spread, or a figure"], "quick_reading_ambiguous"),
])
def test_distribution_keeps_quick_scope_and_any_recorded_ambiguity(
    prepared, question, statistic, grouping, ambiguities, reason,
):
    request = reading(question, grouping=grouping, ambiguities=ambiguities,
                      measure={"concept": "crea", "statistic": statistic})
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert (proposal.status, proposal.reason) == ("full_study", reason)
    assert proposal.card is None and proposal.expand.reading == request


def test_missing_values_are_counted_without_changing_the_observed_denominator(prepared, tmp_path):
    frame = pd.read_parquet(prepared.path)
    frame.loc[:11, "sex"] = None
    prepared = rebound(prepared, frame=frame)
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    answer = run(proposal, prepared, tmp_path)
    assert answer.status == "ok", answer.reason
    values = {row.statistic: row.value for row in answer.rows}
    assert (values["missing_n"], values["denominator"], values["percentage"]) == (12, 48, 37.5)


def test_nonfinite_measurement_never_becomes_a_missing_value(prepared, tmp_path):
    frame = pd.read_parquet(prepared.path)
    frame.loc[0, "age"] = float("inf")
    prepared = rebound(prepared, frame=frame)
    proposal = prepare_quick_answer(reading(measure={"concept": "age", "statistic": "mean"}), source=prepared, provider=None)
    assert run(proposal, prepared, tmp_path).reason == "quick_input_not_testable"


def test_comparison_requires_verified_patient_identity(prepared):
    request = reading(intent="compare", grouping=["sex"], measure={"concept": "age", "statistic": "mean"})
    proposal = prepare_quick_answer(request, source=rebound(prepared, patient_column=None), provider=None)
    assert proposal.reason == "quick_independent_units_unverified"


def grouped(source, patients, *, issued=True):
    """Key the rows as a verified grouping does: ``p<patient>:s<stay>``, named by the context."""
    frame = pd.read_parquet(source.path).drop(columns=["subject_id"])
    frame.insert(0, "patient_stay_id", [f"p{p}:s{s}" for p, s in zip(patients, frame.pop("stay_id"), strict=True)])
    frame.to_parquet(source.path, index=False)
    variables = [ConceptDescriptor(name="patient_stay_id", dtype="object", role="id"),
                 *(v for v in source.context.variables if v.name not in {"stay_id", "subject_id"})]
    provenance = {"replacement_row_identity": {
        "output_identity_column": "patient_stay_id", "mapping_file_sha256": "a" * 64, "mapped_cohort_rows": len(frame),
        "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
    }} if issued else {}
    context = source.context.model_copy(update={"variables": variables, "cohort": source.context.cohort.model_copy(
        update={"id_columns": ["patient_stay_id"], "provenance": provenance})})
    return bind_prepared_source(source.path, context=context, identity=source.identity,
                                unit_column="patient_stay_id", patient_column="patient_stay_id")


def test_a_verified_grouping_key_names_each_stays_patient(prepared, tmp_path):
    source = grouped(prepared, range(60))
    request = reading(intent="compare", grouping=["sex"], measure={"concept": "age", "statistic": "median"})
    proposal = prepare_quick_answer(request, source=source, provider=None)
    assert proposal.status == "approval_required", proposal.reason
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok", answer.reason
    assert any(row.statistic.startswith("p_value:") for row in answer.rows)


def test_a_patient_with_two_stays_stops_the_comparison(prepared, tmp_path):
    source = grouped(prepared, [0, *range(59)])
    request = reading(intent="compare", grouping=["sex"], measure={"concept": "age", "statistic": "median"})
    proposal = prepare_quick_answer(request, source=source, provider=None)
    assert proposal.status == "approval_required", proposal.reason
    answer = run(proposal, source, tmp_path)
    assert answer.reason == "quick_repeated_units_require_full_study" and not answer.rows


def test_a_repeated_patient_stop_offers_the_first_stay_only_where_it_can_be_prepared(prepared, tmp_path):
    source = grouped(prepared, [0, *range(59)])
    constraints = json.loads(source.context.user_preferences.data_constraints)
    orderable = source.context.model_copy(update={"user_preferences": UserPreferences(
        data_constraints=json.dumps({**constraints, "first_icu_stays_orderable": True}))})
    source = bind_prepared_source(source.path, context=orderable, identity=source.identity,
                                  unit_column=source.unit_column, patient_column=source.patient_column)
    request = reading(intent="compare", grouping=["sex"], measure={"concept": "age", "statistic": "median"})
    proposal = prepare_quick_answer(request, source=source, provider=None)
    assert run(proposal, source, tmp_path).reason == "quick_repeated_units_ask_first_stay"


def test_a_stay_key_is_never_its_own_patient(prepared):
    with pytest.raises(QuickStop, match="quick_unit_identity_missing"):
        grouped(prepared, range(60), issued=False)
    with pytest.raises(QuickStop, match="quick_unit_identity_missing"):
        rebound(prepared, patient_column="stay_id")


def test_one_run_is_not_executed_twice(prepared, tmp_path):
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    assert run(proposal, prepared, tmp_path).status == "ok"
    assert run(proposal, prepared, tmp_path).reason == "quick_run_already_exists"


def test_changed_plan_with_a_recomputed_digest_is_still_recompiled(prepared, tmp_path):
    from easyicu.webserver.quick_answer.planning import card_digest

    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    card = proposal.card.model_copy(update={"plan": proposal.card.plan.model_copy(update={"measure_column": "age"})})
    card = card.model_copy(update={"sha256": card_digest(card)})
    proposal = proposal.model_copy(update={"card": card})
    assert run(proposal, prepared, tmp_path).reason == "quick_plan_changed"


@pytest.mark.parametrize("whitespace_only", [False, True])
def test_projection_rechecks_approval_evidence(prepared, tmp_path, whitespace_only):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.quick_answer.evidence import project_quick_answer

    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    answer = run(proposal, prepared, tmp_path)
    store = EvidenceStore(tmp_path / "run")
    result = store.get(answer.rows[0].evidence.evidence_id)
    receipt = store.get(result.inputs[0])
    path = tmp_path / "run" / receipt.relative_path
    path.write_text(path.read_text() + " " if whitespace_only else "{}")
    with pytest.raises(QuickStop, match="quick_evidence_binding_failed"):
        project_quick_answer(run_dir=tmp_path / "run", proposal=proposal,
                             evidence_id=result.evidence_id, sha256=result.sha256)


def test_an_unsupported_population_is_never_silently_dropped(prepared):
    request = reading("Describe the male proportion on the first ICU stay.", population={"criteria": [{
        "id": "c1", "kind": "first_icu_stay", "source": "question", "role": "include",
        "quote": "first ICU stay",
    }]})
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.reason == "quick_population_not_prepared" and proposal.card is None


def test_multiple_population_operations_require_full_study(prepared):
    request = reading("Use first ICU stay aged at least 18 years.", population={"criteria": [{
        "id": "c1", "kind": "first_icu_stay", "source": "question", "role": "include", "quote": "first ICU stay",
    }, {
        "id": "c2", "kind": "age_years", "source": "question", "role": "include",
        "quote": "aged at least 18 years", "min_years": 18,
    }]})
    assert prepare_quick_answer(request, source=prepared, provider=None).reason == "quick_requires_multistep_study"


def test_a_measure_requires_the_exact_prepared_window(prepared):
    request = reading(measure={"concept": "lact", "statistic": "median", "summary": "max",
                              "window": {"start_hours": 0, "end_hours": 48}})
    assert prepare_quick_answer(request, source=prepared, provider=None).reason == "quick_window_unresolved"


def test_more_than_two_groups_get_per_group_summaries_without_a_test(prepared):
    # The group count is not inferred from the question: a third level is
    # summarised as its own group, and no omnibus test silently includes it.
    variables = [v.model_copy(update={"observed_domain": {"levels": ["Male", "Female", "Unknown"]}})
                 if v.name == "sex" else v for v in prepared.context.variables]
    prepared = rebound(prepared, context=prepared.context.model_copy(update={"variables": variables}))
    request = reading(intent="compare", grouping=["sex"], measure={"concept": "age", "statistic": "mean"})
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.status == "approval_required"
    assert proposal.card.plan.test is None
    assert proposal.card.plan.group_levels == ("Male", "Female", "Unknown")
    assert proposal.card.plan.qualification[-1] == "multi_group_summaries_without_test"


def test_a_three_group_comparison_reports_each_group_and_no_p_value(prepared, tmp_path):
    frame = pd.read_parquet(prepared.path)
    frame["sex"] = ["Male"] * 20 + ["Female"] * 20 + ["Unknown"] * 20
    variables = [v.model_copy(update={"observed_domain": {"levels": ["Male", "Female", "Unknown"]}})
                 if v.name == "sex" else v for v in prepared.context.variables]
    source = rebound(prepared, frame=frame, context=prepared.context.model_copy(update={"variables": variables}))
    request = reading("Compare lactate by sex.", intent="compare", grouping=["sex"],
                      measure={"concept": "lact", "statistic": "median", "summary": "max",
                               "window": {"start_hours": 0, "end_hours": 24}})
    proposal = prepare_quick_answer(request, source=source, provider=None)
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok"
    medians = {row.group: row.value for row in answer.rows if row.statistic == "median"}
    assert set(medians) >= {"Male", "Female", "Unknown"}
    assert not [row for row in answer.rows if row.statistic.startswith("p_value")]


def test_selected_proportion_never_receives_an_omnibus_test(prepared):
    variables = [v.model_copy(update={"observed_domain": {"levels": [0, 1, 2]}})
                 if v.name == "icu_death" else v for v in prepared.context.variables]
    prepared = rebound(prepared, context=prepared.context.model_copy(update={"variables": variables}))
    request = reading("Compare the selected category proportion by sex.", intent="compare",
                      grouping=["sex"], measure={"concept": "icu_death", "statistic": "proportion", "level": 1})
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.status == "full_study"
    assert proposal.reason == "quick_comparison_ambiguous"
    assert proposal.card is None and proposal.expand.reading == request


@pytest.mark.parametrize("label", ["Overall", "Missing", "Comparison", "Other groups (pooled)"])
def test_reserved_partition_labels_cannot_ambiguously_name_a_group(prepared, label):
    variables = [v.model_copy(update={"observed_domain": {"levels": [label, "Other"]}})
                 if v.name == "sex" else v for v in prepared.context.variables]
    prepared = rebound(prepared, context=prepared.context.model_copy(update={"variables": variables}))
    request = reading("Count stays by sex.", measure={"statistic": "count"}, grouping=["sex"])
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.status == "full_study"
    assert proposal.reason == "quick_grouping_ambiguous"
    assert proposal.card is None and proposal.expand.reading == request


def test_identifier_role_cannot_be_presented_as_a_measure(prepared):
    variables = [v.model_copy(update={"role": VariableRole.ID}) if v.name == "sex" else v for v in prepared.context.variables]
    # Round-trip validation retains typed roles, just as prepared-context loading does.
    context = ResearchContext.model_validate(prepared.context.model_copy(update={"variables": variables}).model_dump(mode="json"))
    proposal = prepare_quick_answer(reading(), source=rebound(prepared, context=context), provider=None)
    assert proposal.reason == "quick_identifier_forbidden"


def test_large_prepared_scan_is_refused_before_allocating_rows(prepared, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver.quick_answer import analysis

    original = analysis.pq.ParquetFile

    def oversized(stream):
        file = original(stream)
        return SimpleNamespace(schema=file.schema, metadata=SimpleNamespace(
            num_rows=2_000_001, num_row_groups=0,
        ), read=lambda **_kwargs: pytest.fail("unbounded scan"))

    monkeypatch.setattr(analysis.pq, "ParquetFile", oversized)
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    assert run(proposal, prepared, tmp_path).reason == "quick_prepared_source_too_large"


def test_offline_budget_on_an_incrementally_written_prepared_source(prepared):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # A bounded row-group fixture, generated on disk in this lane, not a large
    # in-memory frame. TemporaryDirectory removes every generated byte.
    lane = Path(__file__).resolve().parents[2] / ".codex_lane" / "tmp"
    lane.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=lane, prefix="quick-timing-") as directory:
        directory = Path(directory)
        path = directory / "prepared.parquet"
        batch = pd.read_parquet(prepared.path)
        with pq.ParquetWriter(path, pa.Table.from_pandas(batch, preserve_index=False).schema) as writer:
            for offset in range(0, 60_000, len(batch)):
                writer.write_table(pa.Table.from_pandas(batch.assign(
                    stay_id=batch.stay_id + offset, subject_id=batch.subject_id + offset,
                ), preserve_index=False))
        context = prepared.context.model_copy(update={"cohort": prepared.context.cohort.model_copy(update={"n_stays": 60_000})})
        source = bind_prepared_source(path, context=context, identity=prepared.identity,
                                     unit_column="stay_id", patient_column="subject_id")
        start = monotonic()
        proposal = prepare_quick_answer(reading(), source=source, provider=None)
        answer = run(proposal, source, directory)
        elapsed = monotonic() - start
        assert answer.status == "ok", answer.reason
        assert elapsed < 30
        assert next(row.value for row in answer.rows if row.statistic == "denominator") == 60_000
        print(f"quick offline prepared path: {elapsed:.3f}s for 60000 synthetic stays")


@pytest.mark.parametrize("tamper", ["digest", "source", "missing_claim", "claim_value", "field", "file",
                                    "file_whitespace", "claim_tolerance", "source_metadata", "binding_metadata", "data_metadata"])
def test_every_numeric_cell_requires_exact_evidence(prepared, tmp_path, monkeypatch, tamper):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.quick_answer.evidence import project_quick_answer

    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    answer = run(proposal, prepared, tmp_path)
    coordinate = answer.rows[0].evidence
    digest = coordinate.sha256
    if tamper == "digest":
        digest = "0" * 64
    elif tamper == "source":
        proposal = proposal.model_copy(update={"card": proposal.card.model_copy(update={
            "source": {"database": "eicu_demo", "registration_id": "src_wrong"},
        })})
    elif tamper in {"file", "file_whitespace"}:
        store = EvidenceStore(tmp_path / "run")
        path = tmp_path / "run" / store.get(coordinate.evidence_id).relative_path
        path.write_bytes(path.read_bytes() + b" " if tamper == "file_whitespace" else
                         path.read_bytes().replace(b'"value": 60', b'"value": 61'))
    elif tamper in {"source_metadata", "binding_metadata", "data_metadata"}:
        store = EvidenceStore(tmp_path / "run")
        record = store.get(coordinate.evidence_id)
        key = {"source_metadata": "source_identity", "binding_metadata": "source_binding",
               "data_metadata": "source_data_sha256"}[tamper]
        store.update_record(record.evidence_id, metadata={**record.metadata, key: "wrong"})
    else:
        original = EvidenceStore.numeric_claims

        def claims(store):
            items = original(store)
            if tamper == "missing_claim":
                return []
            item = next(claim for claim in items if claim.source_field == coordinate.source_field)
            if tamper == "claim_value":
                item.canonical += 1
            elif tamper == "claim_tolerance":
                item.tolerance = 1
            else:
                item.source_field = "rows[999].value"
            return items

        monkeypatch.setattr(EvidenceStore, "numeric_claims", claims)
    with pytest.raises(QuickStop, match="quick_evidence_binding_failed"):
        project_quick_answer(run_dir=tmp_path / "run", proposal=proposal,
                             evidence_id=coordinate.evidence_id, sha256=digest)


def test_a_population_quote_must_come_from_this_question(prepared):
    request = reading(population={"criteria": [{
        "id": "c1", "kind": "age_years", "source": "question", "role": "include",
        "quote": "aged at least 18 years", "min_years": 18,
    }]})
    assert prepare_quick_answer(request, source=prepared, provider=None).reason == "quick_reading_ambiguous"


def test_unassessed_vocabulary_cannot_be_claimed_as_supported(prepared):
    variables = [*prepared.context.variables, ConceptDescriptor(name="local_metric", dtype="float64")]
    prepared = rebound(prepared, context=prepared.context.model_copy(update={"variables": variables}))
    proposal = prepare_quick_answer(reading(measure={"concept": "local_metric", "statistic": "median"}),
                                    source=prepared, provider=None)
    assert proposal.reason == "quick_capability_unassessed"


def test_context_cannot_change_under_the_prepared_handle(prepared, tmp_path):
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    prepared.context.cohort.n_stays = 61
    assert run(proposal, prepared, tmp_path).reason == "quick_source_changed"


@pytest.mark.parametrize("female_n,expected", [(0, "ok"), (10, "stopped"), (11, "ok")])
def test_aggregate_suppression_uses_the_existing_count_floor(prepared, tmp_path, female_n, expected):
    frame = pd.read_parquet(prepared.path)
    frame["sex"] = ["Male"] * (len(frame) - female_n) + ["Female"] * female_n
    prepared = rebound(prepared, frame=frame)
    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    assert run(proposal, prepared, tmp_path).status == expected


def test_the_card_records_the_exact_prepared_data_digest(prepared):
    from easyicu.research_agent.canonical_json import canonical_sha256

    proposal = prepare_quick_answer(reading(), source=prepared, provider=None)
    assert proposal.card.plan.source_data_sha256 == prepared.data_sha256
    assert proposal.card.plan.context_sha256 == canonical_sha256(prepared.context.model_dump(mode="json"))


def test_population_receipts_do_not_copy_private_host_metadata_into_the_card(prepared):
    from easyicu.research_agent.acquisition.first_icu_stay import SELECTION_POLICY
    from easyicu.research_agent.intake.materialized_metadata import FIRST_ICU_STAY_PER_HOSPITALIZATION_RESTRICTION_SCHEMA

    receipt = {
        "schema_version": FIRST_ICU_STAY_PER_HOSPITALIZATION_RESTRICTION_SCHEMA,
        "coordinate_sha256": "a" * 64, "authority_coordinates": SELECTION_POLICY,
        "private_host_path": "/private/fixture/mapping.parquet", "private_rows": ["subject_fixture"],
    }
    context = prepared.context.model_copy(update={"cohort": prepared.context.cohort.model_copy(update={
        "provenance": {"first_icu_stay_per_hospitalization_restriction": receipt},
    })})
    prepared = rebound(prepared, context=context)
    request = reading("Count the first ICU stay per hospitalization.", measure={"statistic": "count"},
                      population={"criteria": [{
                          "id": "c1", "kind": "first_icu_stay_per_hospitalization", "source": "question",
                          "role": "include", "quote": "first ICU stay per hospitalization",
                      }]})
    proposal = prepare_quick_answer(request, source=prepared, provider=None)
    assert proposal.status == "approval_required", proposal.reason
    assert "private_host_path" not in proposal.model_dump_json()
    assert "subject_fixture" not in proposal.model_dump_json()
    assert proposal.card.plan.population["criteria"][0]["disposition"] == "applied_by_source"


LACT_MEDIAN = {"concept": "lact", "statistic": "median", "summary": "max",
               "window": {"start_hours": 0, "end_hours": 24}}


def four_groups(prepared, sizes, *, lact=None):
    frame = pd.read_parquet(prepared.path)
    frame["sex"] = sum(([level] * n for level, n in sizes.items()), [])
    if lact is not None:
        frame["lact_max"] = lact
    variables = [v.model_copy(update={"observed_domain": {"levels": list(sizes)}})
                 if v.name == "sex" else v for v in prepared.context.variables]
    return rebound(prepared, frame=frame, context=prepared.context.model_copy(update={"variables": variables}))


def by_group(answer, statistic):
    return {row.group: row.value for row in answer.rows if row.statistic == statistic}


def test_small_groups_are_pooled_instead_of_withholding_the_answer(prepared, tmp_path):
    source = four_groups(prepared, {"Male": 24, "Female": 24, "Unknown": 7, "Other": 5})
    proposal = prepare_quick_answer(reading("Lactate by sex.", grouping=["sex"], measure=LACT_MEDIAN),
                                    source=source, provider=None)
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok", answer.reason
    assert set(by_group(answer, "median")) == {"Overall", "Male", "Female", "Other groups (pooled)"}
    assert by_group(answer, "cohort_n") == {"Overall": 60, "Male": 24, "Female": 24, "Other groups (pooled)": 12}
    assert not {row.group for row in answer.rows} & {"Unknown", "Other"}


def test_a_pool_still_below_the_floor_absorbs_the_next_smallest_group(prepared, tmp_path):
    source = four_groups(prepared, {"Male": 30, "Female": 20, "Unknown": 6, "Other": 4})
    request = reading("Compare lactate by sex.", intent="compare", grouping=["sex"], measure=LACT_MEDIAN)
    proposal = prepare_quick_answer(request, source=source, provider=None)
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok", answer.reason
    assert by_group(answer, "cohort_n") == {"Overall": 60, "Male": 30, "Other groups (pooled)": 30}
    assert not [row for row in answer.rows if row.statistic.startswith("p_value")]


def test_a_group_with_few_values_shows_its_size_but_no_statistic(prepared, tmp_path):
    lact = [2.0] * 30 + [3.0] * 5 + [None] * 25
    source = four_groups(prepared, {"Male": 30, "Female": 30}, lact=lact)
    proposal = prepare_quick_answer(reading("Lactate by sex.", grouping=["sex"], measure=LACT_MEDIAN),
                                    source=source, provider=None)
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok", answer.reason
    assert set(by_group(answer, "median")) == {"Overall", "Male"}
    assert by_group(answer, "cohort_n") == {"Overall": 60, "Male": 30, "Female": 30}
    female = {row.statistic for row in answer.rows if row.group == "Female"}
    assert female == {"cohort_n"}
    # Overall's split would reveal Female's 5 values by subtraction.
    assert "nonmissing_n" not in {row.statistic for row in answer.rows if row.group == "Overall"}


def test_a_few_missing_values_withhold_only_the_missing_split(prepared, tmp_path):
    lact = [None] * 3 + [2.0] * 57
    source = four_groups(prepared, {"Male": 30, "Female": 30}, lact=lact)
    proposal = prepare_quick_answer(reading("Lactate by sex.", grouping=["sex"], measure=LACT_MEDIAN),
                                    source=source, provider=None)
    answer = run(proposal, source, tmp_path)
    assert answer.status == "ok", answer.reason
    assert set(by_group(answer, "median")) == {"Overall", "Male", "Female"}
    assert set(by_group(answer, "missing_n")) == {"Female"}
    assert set(by_group(answer, "nonmissing_n")) == {"Female"}


def test_a_two_group_test_never_pools_and_still_stops_on_a_small_group(prepared, tmp_path):
    source = four_groups(prepared, {"Male": 55, "Female": 5})
    request = reading("Compare lactate by sex.", intent="compare", grouping=["sex"], measure=LACT_MEDIAN)
    proposal = prepare_quick_answer(request, source=source, provider=None)
    assert proposal.card.plan.test is not None
    answer = run(proposal, source, tmp_path)
    assert answer.reason == "quick_small_cells_suppressed" and not answer.rows
