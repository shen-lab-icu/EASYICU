"""Host quick answers use synthetic prepared rows, never a provider or ASGI."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import pandas as pd
import pytest

from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.research_agent.schema import (
    CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences, VariableRole,
)
from easyicu.webserver.quick_answer import (
    QuickApproval, QuickReading, QuickStop, bind_prepared_source, execute_quick_answer,
    prepare_quick_answer,
)


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
    ("mean", {"mean": 2.5}), ("median", {"median": 2.5}),
    ("iqr", {"q1": 2.0, "q3": 3.0}), ("count", {"nonmissing_n": 60}),
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


@pytest.mark.parametrize("provider", [None, object()])
def test_credentialed_source_needs_existing_provider_clearance(prepared, provider):
    proposal = prepare_quick_answer(reading(), source=rebound(prepared, database="eicu"), provider=provider)
    assert proposal.reason == "quick_qualifying_provider_required"
    assert proposal.card is None


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


def test_group_count_is_not_inferred_from_the_question(prepared):
    variables = [v.model_copy(update={"observed_domain": {"levels": ["Male", "Female", "Unknown"]}})
                 if v.name == "sex" else v for v in prepared.context.variables]
    prepared = rebound(prepared, context=prepared.context.model_copy(update={"variables": variables}))
    request = reading(intent="compare", grouping=["sex"], measure={"concept": "age", "statistic": "mean"})
    assert prepare_quick_answer(request, source=prepared, provider=None).reason == "quick_two_groups_required"


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


@pytest.mark.parametrize("label", ["Overall", "Missing", "Comparison"])
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
