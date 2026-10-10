"""A prediction's comparison with an existing score is reported by the host.

The static prediction owner records, per comparator, both AUROCs with their
intervals, their paired difference and, when calibration was compared, both
Brier scores, in a table and in its summary.  The host states them in fixed
sentences whose every number is read from the verified table and must equal
the summary, which the numeric binder checks under STRICT evidence.  The
Abstract carries the comparison when the question asked for it, and
Limitations says so when the comparator was computed over another window
than the model's.  Synthetic records only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.audits.manuscript_claims import audit_manuscript_numeric_claims
from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.contracts.prediction_execution import PREDICTION_BENCHMARK_PRODUCT
from easyicu.research_agent.reporting.benchmark_report_facts import (
    BENCHMARK_LIMITATIONS_MISSING,
    BENCHMARK_REPORT_REQUIREMENT_DRIFT,
    BENCHMARK_REPORT_SOURCE_MISMATCH,
    BenchmarkReportError,
    audit_bound_benchmark_limitations,
    benchmark_window_limitations,
    compile_benchmark_report_facts,
)
from easyicu.research_agent.reporting.descriptive_report_facts import (
    missing_primary_result_facts,
    render_descriptive_report_claims,
)
from easyicu.research_agent.reporting.manuscript_method_facts import (
    place_manuscript_method_facts,
)

_STEP = "benchmark_comparison"
_COLUMN = "apache_iv_pred_hosp_mort"
_LABELS = {_COLUMN: "APACHE IVa predicted mortality"}
MANUSCRIPT = (
    "# Title\n\n## Abstract\n\n**Results:**\n\n**Conclusions:**\nCaution.\n\n"
    "## Results\n\n### Model performance\n\n### Secondary analyses\n\n"
    "## Discussion\n\nBoundary.\n\n## Limitations\n\nOne.\n\n## Conclusion\n\nCaution."
)


def _comparison(*, missing: int, calibration: bool, relation: str) -> dict[str, Any]:
    validation = 1200
    model = {"auroc": 0.8123456789, "auroc_ci_low": 0.7901234567, "auroc_ci_high": 0.8345678912}
    comparator = {"auroc": 0.7012345678, "auroc_ci_low": 0.6723456789, "auroc_ci_high": 0.7298765432}
    if calibration:
        model.update(brier_score=0.1234567891, calibration_intercept=0.0123, calibration_slope=0.9812)
        comparator.update(brier_score=0.1398765432, calibration_intercept=-0.2345, calibration_slope=1.1234)
    return {
        "comparator_column": _COLUMN,
        "comparator_concept": _COLUMN,
        "comparator_kind": "probability",
        "validation_n": validation,
        "comparator_missing_n": missing,
        "comparison_n": validation - missing,
        "comparison_event_n": 210 - missing // 10,
        "comparison_subject_n": validation - missing,
        "calibration_status": "compared" if calibration else "calibration_not_compared",
        "calibration_reason": "" if calibration else "score_scale",
        "comparator_predicts": "death",
        "outcome_concept": "death",
        "comparator_information_window": "icu_admission[0,24]h",
        "prediction_time_hours": 6.0,
        "information_window_relation": relation,
        "information_window_differs": relation != "same",
        "model": model,
        "comparator": comparator,
        "auroc_difference": model["auroc"] - comparator["auroc"],
        "auroc_difference_ci_low": 0.0812345678,
        "auroc_difference_ci_high": 0.1412345678,
        "auroc_difference_p_value": None,
        "interval_method": "patient_stratified_bootstrap_percentile_95pct",
    }


def _table(comparison: dict[str, Any]) -> pd.DataFrame:
    """The executor's table rows for one comparison, from the same values."""

    shared = {
        key: comparison[key]
        for key in ("comparator_column", "validation_n", "comparator_missing_n",
                    "comparison_n", "comparison_event_n", "information_window_differs")
    }
    model, comparator = comparison["model"], comparison["comparator"]
    rows = [{
        **shared, "metric": "auroc",
        "model_value": model["auroc"], "model_ci_low": model["auroc_ci_low"],
        "model_ci_high": model["auroc_ci_high"], "comparator_value": comparator["auroc"],
        "comparator_ci_low": comparator["auroc_ci_low"],
        "comparator_ci_high": comparator["auroc_ci_high"],
        "difference": comparison["auroc_difference"],
        "difference_ci_low": comparison["auroc_difference_ci_low"],
        "difference_ci_high": comparison["auroc_difference_ci_high"],
    }]
    for metric in ("brier_score", "calibration_intercept", "calibration_slope"):
        if metric in model:
            rows.append({**shared, "metric": metric,
                         "model_value": model[metric], "comparator_value": comparator[metric]})
    return pd.DataFrame(rows)


def _registered(
    tmp_path: Path,
    *,
    missing: int = 0,
    calibration: bool = True,
    relation: str = "comparator_ends_after_prediction_time",
    table_shift: float = 0.0,
):
    comparison = _comparison(missing=missing, calibration=calibration, relation=relation)
    summary = {
        "step_id": _STEP,
        "status": "ok",
        "method": "deterministic_prediction_benchmark_comparison",
        "reportable_benchmark_comparison": {
            "prediction_time_hours": 6.0,
            "outcome_concept": "death",
            "comparisons": [comparison],
        },
        "output_files": {PREDICTION_BENCHMARK_PRODUCT: "benchmark_comparison.csv"},
    }
    out = tmp_path / "step"
    out.mkdir()
    table = _table(comparison)
    table.loc[table["metric"].eq("auroc"), "model_value"] += table_shift
    table.to_csv(out / "benchmark_comparison.csv", index=False)
    (out / "step_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(tmp_path / "run")
    store.register_file(
        kind="statistic", source_path=out / "step_summary.json", evidence_id="benchmark_summary",
        description="Benchmark comparison", produced_by_step=_STEP,
        generation_mode="deterministic_standard",
    )
    store.register_file(
        kind="table", source_path=out / "benchmark_comparison.csv",
        description="Benchmark comparison table", produced_by_step=_STEP,
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=_STEP, evidence_id="benchmark_summary", summary=summary,
    )
    records = [{
        "step_id": _STEP, "status": "ok", "step_summary": summary,
        "step_summary_evidence_id": "benchmark_summary", "evidence_ids": ["benchmark_summary"],
    }]
    return store, records


#: The primary model's own AUROC, which a prediction run registers beside the
#: comparison, so the auditor checks every AUROC a sentence cites.
_PRIMARY = {
    "step_id": "primary_performance", "status": "ok",
    "step_summary": {"auroc": 0.8, "auroc_ci_low": 0.78, "auroc_ci_high": 0.82},
}


def _bound(store, records, facts) -> tuple[str, str]:
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    projected = render_descriptive_report_claims(MANUSCRIPT, facts)
    bound, bindings, untraced = bind_numeric_values(
        projected, evidence=store, enforcement_mode=EvidenceEnforcementMode.STRICT,
        per_step_records=records,
    )
    assert bindings and not untraced
    return projected, bound


def test_a_full_comparison_is_stated_and_every_number_binds(tmp_path) -> None:
    store, records = _registered(tmp_path)

    facts = compile_benchmark_report_facts(records, evidence=store, reader_display_labels=_LABELS)

    comparison, brier = facts
    assert comparison.text == (
        "In the 1,200 validation stays (210 events), the model's AUROC was 0.812 "
        "(95% CI 0.790 to 0.835) and the AUROC of APACHE IVa predicted mortality was "
        "0.701 (95% CI 0.672 to 0.730); the difference between the two AUROCs (the "
        "model's minus that of APACHE IVa predicted mortality) was 0.111 (95% CI 0.081 "
        "to 0.141)"
    )
    assert brier.text == (
        "On the same stays, the Brier score was 0.123 for the model and 0.140 for "
        "APACHE IVa predicted mortality"
    )
    assert {fact.subsection for fact in facts} == {"Secondary analyses"}
    # Without a plan no comparison counts as one the question asked for.
    assert {fact.required_result_sections for fact in facts} == {("Results",)}
    projected, bound = _bound(store, records, facts)
    secondary = projected.split("### Secondary analyses", 1)[1].split("## Discussion")[0]
    assert all(fact.scaffold in secondary for fact in facts)
    assert "step=benchmark_comparison" in bound and "field=reportable_benchmark_comparison" in bound
    assert missing_primary_result_facts(bound, facts) == {}
    # The prediction auditor reads both AUROCs and both intervals as the step's.
    assert audit_manuscript_numeric_claims(bound, per_step_records=[_PRIMARY, *records]) == []


def test_a_partial_comparison_states_the_stays_it_used(tmp_path) -> None:
    store, records = _registered(tmp_path, missing=200, calibration=False)

    (fact,) = compile_benchmark_report_facts(records, evidence=store, reader_display_labels=_LABELS)

    assert fact.text.startswith(
        "APACHE IVa predicted mortality was recorded for 1,000 of the 1,200 validation "
        "stays (190 events among them), and the comparison used these stays: the model's "
        "AUROC was 0.812"
    )
    assert fact.source_fields[:3] == tuple(
        f"reportable_benchmark_comparison.comparisons[0].{field}"
        for field in ("comparison_n", "validation_n", "comparison_event_n")
    )
    _bound(store, records, (fact,))


def test_a_table_that_disagrees_with_its_summary_stops_the_report(tmp_path) -> None:
    store, records = _registered(tmp_path, table_shift=1e-9)

    with pytest.raises(BenchmarkReportError) as stopped:
        compile_benchmark_report_facts(records, evidence=store, reader_display_labels=_LABELS)

    assert stopped.value.code == BENCHMARK_REPORT_SOURCE_MISMATCH
    assert "model_value" in str(stopped.value)


def test_a_run_without_a_comparison_states_none(tmp_path) -> None:
    store, records = _registered(tmp_path)
    other = [{**records[0], "step_id": "primary_performance",
              "step_summary": {"status": "ok", "auroc": 0.8}}]

    assert compile_benchmark_report_facts(other, evidence=store, reader_display_labels=_LABELS) == ()
    assert benchmark_window_limitations(other, evidence=store, reader_display_labels=_LABELS) == ()


def _asked_for(tmp_path, monkeypatch, *, planned: str, executed: str):
    """A run whose planning record judged the question's benchmark ``planned``,
    and whose executed plan the same record judges ``executed``."""

    from easyicu.research_agent.planning import question_requirements

    store, records = _registered(tmp_path)
    record = {
        "schema_version": "recorded",
        "judged": [{"id": "q1", "kind": "benchmark", "disposition": planned}],
    }
    (Path(store.root) / question_requirements.QUESTION_REQUIREMENTS_FILENAME).write_text(
        json.dumps(record), encoding="utf-8"
    )
    plan = SimpleNamespace(steps=[SimpleNamespace(step_id=_STEP, planned_analysis_role="secondary")])
    judged: list[tuple[Any, Any]] = []

    def judge(read, *, plan):
        judged.append((read, plan))
        requirement = SimpleNamespace(id="q1", kind="benchmark", concepts=(_COLUMN,))
        return SimpleNamespace(judged=[SimpleNamespace(
            requirement=requirement, disposition=executed, owner_step_ids=(_STEP,),
        )])

    monkeypatch.setattr(question_requirements, "judge_recorded_requirements", judge)
    return store, records, plan, judged, record


def test_a_comparison_the_question_asked_for_is_in_the_abstract(tmp_path, monkeypatch) -> None:
    store, records, plan, judged, record = _asked_for(
        tmp_path, monkeypatch, planned="covered", executed="covered"
    )

    comparison, brier = compile_benchmark_report_facts(
        records, evidence=store, reader_display_labels=_LABELS, plan=plan,
    )

    assert judged == [(record, plan)]
    assert comparison.required_result_sections == ("Abstract", "Results")
    assert brier.required_result_sections == ("Results",)
    projected, bound = _bound(store, records, (comparison, brier))
    abstract = projected.split("## Results", 1)[0]
    assert comparison.scaffold in abstract and brier.scaffold not in abstract
    assert missing_primary_result_facts(bound, (comparison, brier)) == {}


@pytest.mark.parametrize(
    ("planned", "executed"), [("covered", "not_covered"), ("not_covered", "covered")]
)
def test_a_plan_that_drifted_from_its_judged_requirements_stops_the_report(
    tmp_path, monkeypatch, planned: str, executed: str
) -> None:
    store, records, plan, _judged, _record = _asked_for(
        tmp_path, monkeypatch, planned=planned, executed=executed
    )

    with pytest.raises(BenchmarkReportError) as stopped:
        compile_benchmark_report_facts(
            records, evidence=store, reader_display_labels=_LABELS, plan=plan,
        )
    assert stopped.value.code == BENCHMARK_REPORT_REQUIREMENT_DRIFT


def test_a_comparator_computed_past_the_prediction_time_is_a_limitation(tmp_path) -> None:
    store, records = _registered(tmp_path)

    (limitation,) = benchmark_window_limitations(
        records, evidence=store, reader_display_labels=_LABELS,
    )

    assert limitation.text == (
        "APACHE IVa predicted mortality is computed from data recorded up to 24 h after "
        "ICU admission, while the model's prediction time is 6 h, so the comparison "
        "credits APACHE IVa predicted mortality with information the model does not use"
    )
    placed, fields = place_manuscript_method_facts(MANUSCRIPT, (limitation,))
    assert fields == (limitation.source_field,)
    assert placed.split("## Limitations", 1)[1].lstrip().startswith(limitation.scaffold)
    bound = store.bind_manuscript(placed, per_step_records=records)
    audit = {"evidence": store, "per_step_records": records, "reader_display_labels": _LABELS}
    assert audit_bound_benchmark_limitations(bound, **audit) is None
    dropped = audit_bound_benchmark_limitations(
        store.bind_manuscript(MANUSCRIPT, per_step_records=records), **audit
    )
    assert dropped.detail["reason_code"] == BENCHMARK_LIMITATIONS_MISSING


def test_the_write_phase_audits_the_limitation_with_the_host_facts(tmp_path) -> None:
    # The write phase runs the comparator's audit beside the host's other facts.
    from easyicu.research_agent.reporting.write_phase import _bound_host_fact_findings

    store, records = _registered(tmp_path)
    bound = store.bind_manuscript(MANUSCRIPT, per_step_records=records)

    findings = _bound_host_fact_findings(bound, store, records, _LABELS, None, "en")

    assert BENCHMARK_LIMITATIONS_MISSING in {
        finding.detail.get("reason_code") for finding in findings
    }


def test_a_comparator_on_the_models_window_needs_no_limitation(tmp_path) -> None:
    store, records = _registered(tmp_path, relation="same")

    assert benchmark_window_limitations(records, evidence=store, reader_display_labels=_LABELS) == ()


def test_the_prediction_auditor_needs_the_comparisons_typed_blocks(tmp_path) -> None:
    """Read from the flat keys the summary once had, the same sentence fails the auditor."""

    store, records = _registered(tmp_path)
    facts = compile_benchmark_report_facts(records, evidence=store, reader_display_labels=_LABELS)
    _projected, bound = _bound(store, records, facts)
    flat = json.loads(json.dumps(records[0]["step_summary"]))
    (comparison,) = flat["reportable_benchmark_comparison"]["comparisons"]
    model, comparator = comparison.pop("model"), comparison.pop("comparator")
    comparison.update(model_auroc=model["auroc"], comparator_auroc=comparator["auroc"])

    findings = audit_manuscript_numeric_claims(
        bound, per_step_records=[_PRIMARY, {**records[0], "step_summary": flat}],
    )

    assert {finding.detail.get("reason") for finding in findings} == {
        "cited_step_does_not_register_metric"
    }
