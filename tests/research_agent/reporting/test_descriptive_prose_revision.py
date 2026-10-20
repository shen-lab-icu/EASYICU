"""Synthetic temperature groups exercise the reader prose and its authority."""

from copy import deepcopy
import json
import re
from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts.exposure_group_rules import grouping_rule_texts, grouping_rule_values
from easyicu.research_agent.orchestration.exposure_group_labels import exposure_group_label_rows
from easyicu.research_agent.reporting.descriptive_report_facts import (
    compile_counts_only_report_facts, compile_continuous_outcome_report_facts,
    manuscript_group_labels, render_descriptive_report_claims, writer_question_coverage,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_result_facts import result_fact_carriage, result_fact_rows
from easyicu.research_agent.reporting.manuscript_surface import render_reader_manuscript
from tests.support.descriptive_report_fixtures import DRAFT, LABELS, _observed, _plan, _store


def _context(unit="icu_stay"):
    return SimpleNamespace(cohort=SimpleNamespace(provenance={"analysis_unit": unit}), variables=[])


def _parameters(n=2):
    return {
        "concept": "temp", "window": {"start_hours": 0, "end_hours": 12},
        "scale": "nominal", "unmeasured": "exclude", "source_columns": {"max": "temp_max"},
        "groups": [
            {"id": f"g{i + 1}", "rule": {"summary": "max", "op": ">", "value": 39.5 - i, "unit": "degC"}}
            for i in range(n)
        ] + [{"id": f"g{n + 1}", "rule": "otherwise"}],
    }


def _group_record(n=2):
    return {
        "schema_version": "easyicu.exposure_groupings_record/1",
        "compiled": {"groupings": [{
            "disposition": "applied", "variable": "temperature_group",
            "compared": {"reference": 1, "contrast": 2},
            "labels": {f"g{i + 1}": f"分组{i + 1}" for i in range(n + 1)},
            "derivation": _parameters(n),
        }]},
    }


@pytest.mark.parametrize("unit,noun", [("icu_stay", "ICU stays"), ("patient", "patients")])
def test_analysis_cohort_uses_source_count_and_context_unit(unit, noun):
    records, evidence = _observed()
    facts = compile_counts_only_report_facts(
        records, evidence=evidence, reader_display_labels=LABELS, context=_context(unit),
    )
    text = render_descriptive_report_claims(DRAFT, facts)
    assert f"The analysis cohort comprised 107 {noun}" in text
    assert "The analysis cohort comprised 100" not in text
    assert f"Of the 100 {noun}" in text
    assert "7 of the 107" in text and "excluded from these shares" in text
    assert "observations" not in text
    assert {fact.cohort_source_field for fact in facts if fact.cohort_n} == {
        "source_row_count_reconciliation.source_rows",
    }


@pytest.mark.parametrize("owner", ["table_one", "study_occurrence"])
def test_full_compiler_passes_the_analysis_unit_to_cohort_fact_owners(tmp_path, monkeypatch, owner):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
    from easyicu.research_agent.reporting.descriptive_report_facts import compile_primary_counts_only_report_facts

    if owner == "table_one":
        summary = {"status": "ok", "analysis_family": "grouped_table_one", "cohort_n": 100,
                   "variables": ["age"], "cohort_input_key": "artifact:analysis_cohort",
                   "output_files": {"table:table_one": "table_one.csv"}}
    else:
        summary = _observed()[0][0]["step_summary"]
        summary.update(analysis_role="secondary", analysis_set="bound_typed_cohort",
                       typed_cohort_input="cohort:study_population")
        summary["descriptive_estimates"].update(analysis_role="secondary", analysis_set="bound_typed_cohort")
        del summary["source_row_count_reconciliation"]
    store = EvidenceStore(tmp_path)
    source = store.register_json(kind="statistic", filename="unit.json", evidence_id="unit",
                                 produced_by_step="unit", payload=summary, description="Synthetic cohort summary")
    rows = [{"step_id": "unit", "status": "ok", "step_summary": summary,
             "step_summary_evidence_id": source.evidence_id, "evidence_ids": [source.evidence_id]}]
    monkeypatch.setattr(RegisteredOutputEnvelopeConsumer, "authoritative_writer_records",
                        lambda self, records, **kwargs: records)
    facts = compile_primary_counts_only_report_facts(
        rows, evidence=store, reader_display_labels=LABELS, context=_context("patient"),
    )
    assert facts and all("100 patients" in fact.text for fact in facts)
    assert all("observations" not in fact.text and "ICU stays" not in fact.text for fact in facts)


@pytest.mark.parametrize("n,expected", [
    (1, "not maximum > 39.5 degC"),
    (2, "neither maximum > 39.5 degC nor maximum > 38.5 degC"),
    (3, "none of: maximum > 39.5 degC; maximum > 38.5 degC; maximum > 37.5 degC"),
])
def test_otherwise_rule_and_record_reader_agree(n, expected):
    parameters = _parameters(n)
    assert grouping_rule_texts(parameters)[f"g{n + 1}"].endswith(": " + expected)
    row = exposure_group_label_rows(_group_record(n), record_sha256="a" * 64)[0]
    assert row["levels"][-1]["rule"].endswith(": " + expected)
    assert grouping_rule_values(parameters)[f"g{n + 1}"] == (0.0, 12.0, *[39.5 - i for i in range(n)])


@pytest.mark.parametrize("n,expected", [
    (2, "maximum > 38.5 degC, without maximum > 39.5 degC"),
    (3, "maximum > 37.5 degC, without maximum > 39.5 degC or maximum > 38.5 degC"),
    (4, "maximum > 36.5 degC, without maximum > 39.5 degC or maximum > 38.5 degC or maximum > 37.5 degC"),
])
def test_conditioned_rule_excludes_every_earlier_match(n, expected):
    assert grouping_rule_texts(_parameters(n))[f"g{n}"].endswith(": " + expected)


def test_combined_sentences_abstract_capitalization_and_binding(tmp_path):
    store, records = _store(tmp_path)
    context = _context()
    facts = (*compile_counts_only_report_facts(
        records[:1], evidence=store, reader_display_labels=LABELS, context=context,
        scientific_claims=store.scientific_claims(),
    ), *compile_continuous_outcome_report_facts(
        records, evidence=store, reader_display_labels=LABELS, context=context, plan=_plan(),
    ))
    text = render_descriptive_report_claims(DRAFT, facts)
    bound, bindings, untraced = bind_numeric_values(
        store.bind_manuscript(text, per_step_records=records), evidence=store, per_step_records=records,
    )
    assert bindings and not untraced
    assert writer_question_coverage(bound, plan=_plan(), facts=facts, evidence=store) == []
    assert len(result_fact_carriage(bound, facts=result_fact_rows(facts),
                                   claims=store.scientific_claims(), evidence=store).carried) == 2
    reader = render_reader_manuscript(bound)
    abstract, results = reader.split("## Results")
    assert abstract.count("Of the 100 ICU stays") == results.count("Of the 100 ICU stays") == 1
    assert "60 (60.00%) had Afebrile and 40 (40.00%) Febrile" in results
    assert "Readmission was 10.00% (6 of 60 ICU stays)" in results
    assert "and 20.00% (8 of 40 ICU stays)" in results
    assert results.count("Median recovery time") == abstract.count("Median recovery time") == 1
    assert results.count("Median support duration") == abstract.count("Median support duration") == 1
    assert "4.25 hours (IQR 2.5–8.125)" in results
    assert "6.5 hours (IQR 3.75–9.25)" in results
    assert "18.25 hours (IQR 9.125–24.5)" in results
    assert "0 missing" not in reader and "2 missing" in results and "1 missing" in results
    assert "observations" not in reader and "“" not in reader
    assert not re.search(r" {2,}[,.;:)]", reader)
    assert all(line[0].isupper() for line in reader.splitlines() if line and line[0].isalpha())
    assert "The analysis cohort" not in abstract
    assert render_descriptive_report_claims(text, facts) == text


def test_definition_precedes_shares_and_rule_labels_omit_measure_window(tmp_path):
    store, records = _store(tmp_path)
    group = _group_record(1)
    group["compiled"]["groupings"][0]["variable"] = "exposure"
    store.register_text(kind="log", filename="groups.json", evidence_id="exposure_groupings",
                        text=json.dumps(group), description="Synthetic temperature grouping")
    summary = deepcopy(records[0]["step_summary"])
    for rows in summary["descriptive_estimates"].values():
        if isinstance(rows, list):
            for row in rows:
                row["level"] += 1
    records[0]["step_summary"] = summary
    facts = compile_counts_only_report_facts(
        records[:1], evidence=store, reader_display_labels=LABELS, context=_context(),
    )
    rendered = render_descriptive_report_claims(DRAFT, facts)
    results = rendered.split("## Results")[1]
    abstract = rendered.split("## Results")[0]
    assert results.count("Groups were defined from") == 1
    assert results.index("Groups were defined from") < results.index("Of the")
    assert "from ICU admission to 12 h" in results
    assert "over 0-12 h" not in results
    assert "recorded from ICU admission to 12 h" in abstract
    assert "that window" not in abstract and "Groups were defined" not in abstract
    assert "had maximum above 39.5 degC" in results
    labels, _ = manuscript_group_labels(store, {}, language="en")
    assert labels["exposure=2"] == "not maximum above 39.5 degC"
    _, bindings, untraced = bind_numeric_values(rendered, evidence=store, per_step_records=records)
    assert bindings and not untraced


def test_plan_card_projection_keeps_the_complete_rule_text():
    from easyicu.webserver.pi_copilot.exposure_group_notes import project_exposure_groups
    rows = exposure_group_label_rows(_group_record(3), record_sha256="a" * 64)
    projected = project_exposure_groups(rows)
    assert projected["status"] == "shown"
    assert [row["rule"] for row in projected["groupings"][0]["levels"]] == [
        row["rule"] for row in rows[0]["levels"]
    ]
    assert "none of:" in projected["groupings"][0]["levels"][-1]["rule"]


def test_code_only_groups_still_cite_the_abstracts_recorded_window(tmp_path):
    store, records = _store(tmp_path)
    group = _group_record(1)
    item = group["compiled"]["groupings"][0]
    item["variable"] = "exposure"
    item["labels"] = {}
    item["derivation"]["window"]["end_hours"] = 168.5
    store.register_text(kind="log", filename="groups.json", evidence_id="exposure_groupings",
                        text=json.dumps(group), description="Synthetic unlabelled grouping")
    facts = compile_counts_only_report_facts(
        records[:1], evidence=store, reader_display_labels={}, context=_context(),
    )
    rendered = render_descriptive_report_claims(DRAFT, facts)
    _, bindings, untraced = bind_numeric_values(rendered, evidence=store, per_step_records=records)
    assert bindings and not untraced
    abstract = rendered.split("## Results")[0]
    assert "168.5 h" in abstract and "{evidence:exposure_groupings}" in abstract


def test_full_cohort_shares_do_not_call_unmeasured_levels_recorded(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    records, _ = _observed()
    summary = records[0]["step_summary"]
    summary["analysis_set"] = summary["descriptive_estimates"]["analysis_set"] = "bound_typed_cohort"
    del summary["source_row_count_reconciliation"]
    for key in ("exposure_prevalence", "outcome_absolute_risks"):
        for row, code in zip(summary["descriptive_estimates"][key], (1, 3)):
            row["level"] = code
    group = _group_record(1)
    item = group["compiled"]["groupings"][0]
    item.update(variable="exposure", labels={"g1": "Febrile", "g2": "Other", "gU": "Unmeasured"})
    item["derivation"]["unmeasured"] = "own_group"
    store = EvidenceStore(tmp_path)
    store.register_json(kind="statistic", filename="summary.json", evidence_id="summary",
                        produced_by_step="distribution", payload=summary, description="Synthetic complete cohort")
    store.register_json(kind="log", filename="groups.json", evidence_id="exposure_groupings",
                        payload=group, description="Synthetic explicit unmeasured level")
    facts = compile_counts_only_report_facts(
        records, evidence=store, reader_display_labels={}, context=_context(),
    )
    text = render_descriptive_report_claims(DRAFT, facts)
    assert "Of the 100 ICU stays in the analysis cohort" in text
    assert "40 (40.00%) Unmeasured" in text
    assert "with temperature recorded" not in text


def test_historical_method_rules_keep_their_exact_words_and_binding(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.manuscript_method_facts import method_fact_binding, place_manuscript_method_facts
    from easyicu.research_agent.schema import ResearchContext

    parameters = _parameters()
    grouping = {
        "variable": "temperature_group", "concept": "temp", "scale": "nominal",
        "parameters": parameters,
        "levels": [{"code": i, "group": f"g{i}", "label": f"Category {i}"} for i in range(1, 4)],
    }
    for version in (None, 2):
        item = {**grouping, **({"rule_wording_version": version} if version else {})}
        context = ResearchContext.model_validate({
            "research_question": "Describe recovery across temperature groups.",
            "cohort": {"cohort_name": "synthetic", "database": "synthetic", "n_stays": 107,
                       "provenance": {"exposure_group_labels": [item]}},
            "variables": [{"name": "temperature_group", "dtype": "int64"}],
        })
        store = EvidenceStore(tmp_path / str(version))
        store.register_text(kind="log", filename="context.json", evidence_id="research_context",
                            text=context.model_dump_json(), description="Synthetic versioned context",
                            producer="pipeline", generation_mode="system")
        original = (store.root / store.get("research_context").relative_path).read_bytes()
        facts = store.manuscript_method_facts()
        text, _ = place_manuscript_method_facts("## Methods\n\n### Variables\n", facts)
        assert ("none of maximum" if version is None else "neither maximum") in text
        _, bindings, untraced = bind_numeric_values(
            store.bind_manuscript(text, per_step_records=[]), evidence=store,
            per_step_records=[], **method_fact_binding(store, []),
        )
        assert bindings and not untraced
        assert (store.root / store.get("research_context").relative_path).read_bytes() == original


def test_new_context_projection_selects_current_rule_wording():
    from easyicu.research_agent.research_context.builder import _exposure_group_projections
    from easyicu.research_agent.contracts.exposure_group_rules import planned_group_column

    column = planned_group_column(
        _parameters(), variable="temperature_group",
        compared={"reference": 1, "contrast": 2},
        labels={"g1": "Warm", "g2": "Mild", "g3": "Other"},
        groupings_record_sha256="a" * 64,
    ).record()
    projection = _exposure_group_projections(None, {"exposure_groups": [column]})
    assert projection["exposure_group_labels"][0]["rule_wording_version"] == 2


def test_continuous_overall_and_unmeasured_appear_once_after_groups(tmp_path):
    store, records = _store(tmp_path)
    source = store.get("continuous_table")
    path = store.root / source.relative_path
    text = path.read_text() + (
        "overall,recovery,Overall,107,105,2,4.75,2.75,8.75,hours\n"
        "exposure_level,recovery,Missing,7,7,0,3.75,1.75,6.75,hours\n"
    )
    updated = store.register_text(kind="table", filename="continuous.csv", evidence_id="continuous_table",
                        description="Synthetic aggregate with overall and unmeasured",
                        produced_by_step="secondary", text=text, on_sha_change="new_id")
    records[1]["evidence_ids"].remove("continuous_table")
    records[1]["evidence_ids"].append(updated.evidence_id)
    facts = compile_continuous_outcome_report_facts(
        records, evidence=store, reader_display_labels=LABELS, context=_context(), plan=_plan(),
    )
    rendered = render_descriptive_report_claims(DRAFT, facts).split("## Results")[1]
    assert rendered.count("the overall median was") == 1
    assert rendered.count("with unmeasured exposure") == 1
    assert rendered.index("in Afebrile") < rendered.index("the overall median") < rendered.index("with unmeasured exposure")
    assert "0 missing" not in rendered
    _, bindings, untraced = bind_numeric_values(rendered, evidence=store, per_step_records=records)
    assert bindings and not untraced
