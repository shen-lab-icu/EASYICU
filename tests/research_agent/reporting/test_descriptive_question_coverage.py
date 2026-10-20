"""Synthetic temperature groups and recovery outcomes, with no patient data."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.reporting.descriptive_report_facts import (
    compile_counts_only_report_facts,
    render_descriptive_report_claims,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.report_fact import DescriptiveReportFact

from tests.support.descriptive_report_fixtures import DRAFT, LABELS, _observed, _plan, _store


def _continuous(store, records):
    from easyicu.research_agent.reporting.descriptive_report_facts import compile_continuous_outcome_report_facts
    return compile_continuous_outcome_report_facts(
        records, evidence=store, reader_display_labels=LABELS, plan=_plan(),
    )


def test_observed_exposure_reports_shares_denominator_and_unmeasured():
    records, evidence = _observed()
    before = deepcopy(records)
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels=LABELS)
    text = render_descriptive_report_claims(DRAFT, facts)
    assert "60 (60.00%) had Afebrile" in text
    assert "exposure recorded" in text
    assert "7" in text and "no exposure recorded" in text and "107" in text
    assert "Afebrile" in text and "Febrile" in text
    assert records == before


def test_the_observed_exposure_subset_is_not_reported_as_the_full_cohort():
    records, evidence = _observed()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels=LABELS)
    text = render_descriptive_report_claims(DRAFT, facts)
    assert "The analysis cohort comprised 100" not in text
    assert "7 of the 107 records" in text


@pytest.mark.parametrize("field,value", [
    ("source_rows", 108), ("analyzed_rows", 99), ("excluded_missing_exposure_rows", -1),
])
def test_observed_subset_rejects_a_false_partition(field, value):
    records, evidence = _observed()
    records[0]["step_summary"]["source_row_count_reconciliation"][field] = value
    with pytest.raises(ValueError):
        compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels=LABELS)


def test_secondary_facts_stay_in_the_secondary_section():
    fact = DescriptiveReportFact("Secondary analyses", "Median recovery was 4 hours", "table",
                                 "a" * 64, ("median",), required_result_sections=("Results",))
    text = render_descriptive_report_claims(DRAFT, [fact])
    assert fact.scaffold in text.split("### Secondary analyses")[1].split("## Discussion")[0]
    assert fact.scaffold not in text.split("### Descriptive results")[1].split("### Secondary analyses")[0]


def test_one_interpretive_token_compares_all_bound_groups(tmp_path):
    store, records = _store(tmp_path)
    facts = compile_counts_only_report_facts(
        records[:1], evidence=store, reader_display_labels=LABELS,
        scientific_claims=store.scientific_claims(),
    )
    claim = store.scientific_claims()[0]
    text = DRAFT + "{claim:" + claim.claim_ref + "}\n"
    rendered = render_descriptive_report_claims(text, facts)
    conclusion = rendered.split("## Conclusion")[1]
    assert "10.00%" in conclusion and "20.00%" in conclusion
    assert "Afebrile" in conclusion and "Febrile" in conclusion
    assert "descriptive" in conclusion and "unadjusted" in conclusion and "causal" in conclusion
    assert "risk difference" not in conclusion and "significant" not in conclusion
    assert render_descriptive_report_claims(rendered, facts) == rendered


def test_each_secondary_outcome_has_quantiles_counts_units_and_numeric_binding(tmp_path):
    store, records = _store(tmp_path)
    facts = _continuous(store, records)
    assert len(facts) == 4
    text = render_descriptive_report_claims(DRAFT, facts)
    assert "recovery time" in text and "support duration" in text
    assert "4.25" in text and "2.5" in text and "8.125" in text
    assert "58" in text and "missing" in text and "hours" in text
    bound, bindings, untraced = bind_numeric_values(text, evidence=store, per_step_records=records)
    assert not untraced and bindings and "continuous_table" in bound
    assert all(f.evidence_id == "continuous_table" for f in facts)
    assert {f.outcome for f in facts} == {"recovery", "support"}
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(text.replace("4.25", "4.999"), evidence=store, per_step_records=records)


@pytest.mark.parametrize("kind", ["stale", "wrong_owner", "inactive", "inactive_summary"])
def test_secondary_table_authority_is_not_inferred_from_its_filename(tmp_path, kind):
    store, records = _store(tmp_path)
    record = store.get("continuous_table")
    if kind == "stale":
        path = store.root / record.relative_path
        path.write_text(path.read_text().replace("4.25", "4.50"))
    elif kind == "wrong_owner":
        record.produced_by_step = "another_step"
    elif kind == "inactive":
        records[1]["evidence_ids"].remove("continuous_table")
    else:
        records[1]["evidence_ids"].remove("continuous_summary")
    with pytest.raises(ValueError):
        _continuous(store, records)


def test_continuous_results_cannot_switch_the_plans_grouping(tmp_path):
    from easyicu.research_agent.reporting.descriptive_report_facts import compile_continuous_outcome_report_facts
    store, records = _store(tmp_path)
    plan = _plan()
    plan.steps[1].inputs[1] = "another_grouping"
    with pytest.raises(ValueError):
        compile_continuous_outcome_report_facts(
            records, evidence=store, reader_display_labels=LABELS, plan=plan,
        )


def test_secondary_code_only_levels_remain_readable_without_invented_meanings(tmp_path):
    from easyicu.research_agent.reporting.descriptive_report_facts import compile_continuous_outcome_report_facts
    store, records = _store(tmp_path, grouping_variable="temperature_group")
    plan = _plan()
    plan.steps[1].inputs[1] = "temperature_group"
    facts = compile_continuous_outcome_report_facts(
        records, evidence=store, reader_display_labels=LABELS, plan=plan,
    )
    text = render_descriptive_report_claims(DRAFT, facts)
    assert "in temperature group=0" in text and "in temperature group=1" in text
    assert "Febrile" not in text and "Afebrile" not in text
    _, bindings, untraced = bind_numeric_values(text, evidence=store, per_step_records=records)
    assert bindings and not untraced


def test_plan_coverage_detects_an_uncompiled_part_and_wrong_citation(tmp_path):
    from easyicu.research_agent.reporting.descriptive_report_facts import writer_question_coverage
    store, records = _store(tmp_path)
    facts = (*compile_counts_only_report_facts(
        records[:1], evidence=store, reader_display_labels=LABELS,
    ), *_continuous(store, records))
    rendered = render_descriptive_report_claims(DRAFT, facts)
    assert writer_question_coverage(rendered, plan=_plan(), facts=facts, evidence=store) == []
    removed = [f for f in facts if f.outcome != "support"]
    missing = writer_question_coverage(render_descriptive_report_claims(DRAFT, removed),
                                      plan=_plan(), facts=removed, evidence=store)
    assert len(missing) == 1 and missing[0].severity == "error"
    assert missing[0].detail["outcome"] == "support"
    assert missing[0].detail["product_id"] == "table:continuous_outcome_summary"
    assert len(writer_question_coverage(DRAFT, plan=_plan(), facts=(), evidence=store)) == 4
    assert writer_question_coverage(rendered.replace("{evidence:continuous_table}", "{evidence:summary}"),
                                    plan=_plan(), facts=facts, evidence=store)
    assert writer_question_coverage("<!--" + rendered + "-->", plan=_plan(), facts=facts, evidence=store)
    wrong = [replace(f, product_id="table:unrelated") for f in facts]
    assert writer_question_coverage(rendered, plan=_plan(), facts=wrong, evidence=store)
    stale = [replace(f, source_sha256="b" * 64) for f in facts]
    assert writer_question_coverage(rendered, plan=_plan(), facts=stale, evidence=store)


def test_a_missingness_sentence_does_not_answer_the_group_share_question(tmp_path):
    from easyicu.research_agent.reporting.descriptive_report_facts import writer_question_coverage
    store, records = _store(tmp_path)
    facts = (*compile_counts_only_report_facts(
        records[:1], evidence=store, reader_display_labels=LABELS,
    ), *_continuous(store, records))
    without_shares = [f for f in facts if f.outcome != "exposure_distribution"]
    without_shares.append(DescriptiveReportFact(
        "Cohort characteristics", "Seven records had unmeasured exposure", "summary",
        store.get("summary").sha256, (), step_id="distribution",
        product_id="table:exposure_outcome_distribution", outcome="exposure_availability",
    ))
    missing = writer_question_coverage(
        render_descriptive_report_claims(DRAFT, without_shares),
        plan=_plan(), facts=without_shares, evidence=store,
    )
    assert [f.detail["outcome"] for f in missing] == ["exposure_distribution"]


def test_full_report_compiler_and_final_binding_keep_all_question_parts(tmp_path, monkeypatch):
    from easyicu.research_agent.audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
    from easyicu.research_agent.reporting.descriptive_report_facts import compile_primary_counts_only_report_facts
    from easyicu.research_agent.reporting.descriptive_report_facts import writer_question_coverage

    store, records = _store(tmp_path)
    # Envelope admission has its own tests; this checks the writer's wiring
    # after that boundary using real registered sources and strict binding.
    monkeypatch.setattr(RegisteredOutputEnvelopeConsumer, "authoritative_writer_records",
                        lambda self, records, **kwargs: records)
    facts = compile_primary_counts_only_report_facts(
        records, evidence=store, reader_display_labels=LABELS, plan=_plan(),
    )
    rendered = render_descriptive_report_claims(DRAFT, facts)
    bound, bindings, untraced = bind_numeric_values(
        store.bind_manuscript(rendered, per_step_records=records),
        evidence=store, per_step_records=records,
    )
    assert bindings and not untraced
    assert writer_question_coverage(bound, plan=_plan(), facts=facts, evidence=store) == []
    assert "recovery time" in bound and "support duration" in bound and "no exposure recorded" in bound


def test_uncovered_question_blocks_a_passing_writer_critique(tmp_path, monkeypatch):
    from easyicu.research_agent.reporting import write_phase
    from easyicu.research_agent.schema import CritiqueReport

    store, records = _store(tmp_path)
    facts = compile_counts_only_report_facts(
        records[:1], evidence=store, reader_display_labels=LABELS,
        scientific_claims=store.scientific_claims(),
    )
    monkeypatch.setattr(write_phase, "compile_primary_counts_only_report_facts", lambda *a, **k: facts)
    monkeypatch.setattr(write_phase, "_persist_reader_artifacts", lambda **k: None)
    monkeypatch.setattr(write_phase, "_persist_manuscript_quality_artifacts", lambda **k: [])
    findings = []
    result = write_phase._bind_and_review_manuscript(
        SimpleNamespace(_evidence_enforcement_mode=EvidenceEnforcementMode.STRICT),
        critic=SimpleNamespace(review_manuscript=lambda **k: CritiqueReport(reviewer="synthetic", status="pass")),
        evidence=store, findings=findings, literature=None, per_step_records=records,
        current_evidence_names=[r.evidence_id for r in store.records()],
        scaffold=DRAFT, writer_error_message=None, writer_probe_mode=False,
        writer_probe_failed_steps=(), run_dir=store.root, reader_display_labels=LABELS,
        manuscript_language="en", plan=_plan(),
    )
    missing = [f for f in findings if f.validator == "writer_question_coverage"]
    assert {f.detail["outcome"] for f in missing} == {"recovery", "support"}
    assert result.manuscript_critique.status == "blocked"


def test_group_labels_use_the_record_and_match_float_codes(tmp_path):
    from easyicu.research_agent.reporting.descriptive_report_facts import manuscript_group_labels
    store, _ = _store(tmp_path)
    record = {
        "schema_version": "easyicu.exposure_groupings_record/1",
        "compiled": {"groupings": [{
            "disposition": "applied", "variable": "temperature_group",
            "compared": {"reference": 1, "contrast": 2},
            "labels": {"g1": "Febrile", "g2": "Afebrile"},
            "derivation": {
                "concept": "temp", "window": {"start_hours": 0, "end_hours": 12},
                "scale": "nominal", "unmeasured": "exclude", "source_columns": {"max": "temp_max"},
                "groups": [{"id": "g1", "rule": {"summary": "max", "op": ">", "value": 38.5, "unit": "degC"}},
                           {"id": "g2", "rule": "otherwise"}],
            },
        }]},
    }
    store.register_text(kind="log", text=json.dumps(record), filename="groups.json",
                        evidence_id="exposure_groupings", description="Synthetic group meanings")
    labels, citations = manuscript_group_labels(store, {"temperature_group=1": "Group one"}, language="en")
    assert labels["temperature_group=1.0"] == "Febrile"
    assert labels["temperature_group=2"] == "Afebrile"
    assert citations["temperature_group=1.0"] == ("exposure_groupings",)
    record["compiled"]["groupings"][0]["labels"] = {"g1": "发热（>38.5 degC）", "g2": "未发热"}
    store2 = EvidenceStore(tmp_path / "other")
    store2.register_text(kind="log", text=json.dumps(record), filename="groups.json",
                         evidence_id="exposure_groupings", description="Chinese group names")
    english, _ = manuscript_group_labels(store2, {}, language="en")
    chinese, _ = manuscript_group_labels(store2, {}, language="zh")
    assert "发热" not in english["temperature_group=1.0"]
    assert "38.5" in english["temperature_group=1.0"]
    assert chinese["temperature_group=1.0"] == "发热（高于38.5 degC）"
    record["compiled"]["groupings"][0]["labels"] = {}
    store3 = EvidenceStore(tmp_path / "codes")
    store3.register_text(kind="log", text=json.dumps(record), filename="groups.json",
                         evidence_id="exposure_groupings", description="Code-only levels")
    codes, _ = manuscript_group_labels(store3, {}, language="en")
    assert codes.get("temperature_group=1.0", "temperature_group=1.0") == "temperature_group=1.0"


def test_english_rule_labels_bind_their_metadata_numbers(tmp_path):
    from easyicu.research_agent.reporting.descriptive_report_facts import manuscript_group_labels

    store, records = _store(tmp_path)
    grouping = {
        "schema_version": "easyicu.exposure_groupings_record/1",
        "compiled": {"groupings": [{
            "disposition": "applied", "variable": "temperature_group",
            "compared": {"reference": 1, "contrast": 2},
            "labels": {"g1": "发热", "g2": "未发热"},
            "derivation": {
                "concept": "temp", "window": {"start_hours": 0, "end_hours": 12},
                "scale": "nominal", "unmeasured": "exclude", "source_columns": {"max": "temp_max"},
                "groups": [{"id": "g1", "rule": {"summary": "max", "op": ">", "value": 38.5, "unit": "degC"}},
                           {"id": "g2", "rule": "otherwise"}],
            },
        }]},
    }
    store.register_text(kind="log", text=json.dumps(grouping), filename="groups.json",
                        evidence_id="exposure_groupings", description="Synthetic label parameters")
    labels, _ = manuscript_group_labels(store, {}, language="en")
    fact = DescriptiveReportFact(
        "Primary outcome", f"Observed readmission in “{labels['temperature_group=1.0']}” was 10.00%",
        "summary", store.get("summary").sha256, (),
        supporting_evidence_ids=("exposure_groupings",),
    )
    _, bindings, untraced = bind_numeric_values(fact.scaffold, evidence=store, per_step_records=records)
    assert not untraced and {c.evidence_id for c in bindings.values()} == {"summary", "exposure_groupings"}
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(fact.scaffold.replace("38.5", "39.9"), evidence=store, per_step_records=records)
