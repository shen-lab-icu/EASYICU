"""How often the exposure occurs in the study cohort reaches the report.

A question that asks how often its exposure occurs is answered by a secondary
distribution on the host-republished study cohort.  Its counts, shares and
intervals become host-placed Abstract and Results sentences copied from the
verified summary, so the answer does not depend on the Writer quoting a
secondary table.  Summaries and labels here are synthetic.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode, EvidenceStore
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    percentage,
    wilson_interval,
)
from easyicu.research_agent.reporting.descriptive_report_facts import (
    _compile_study_population_occurrence_report_facts,
    missing_primary_result_facts,
    render_descriptive_report_claims,
)

LABELS = {"early_injury=0": "No early injury", "early_injury=1": "Early injury"}
MANUSCRIPT = (
    "## Abstract\n\n**Results:**\n\n**Conclusions:**\nCaution.\n\n"
    "## Results\n\n### Cohort characteristics\n\n### Primary association\n\n"
    "### Secondary analyses\n\n## Discussion\n\nBoundary.\n\n## Conclusion\n\nCaution."
)


def _row(index: int, count: int, denominator: int, *, interval: bool, events: bool = False) -> dict[str, Any]:
    """One recorded level, with the executor's own share and Wilson interval."""

    low, high = wilson_interval(count, denominator, confidence_level=0.95) if interval else (None, None)
    proportion = count / denominator
    return {
        "level_index": index,
        "level": index,
        "events" if events else "n": count,
        "denominator": denominator,
        "estimate_pct": percentage(count, denominator),
        "standard_error_pct": (
            round(100.0 * math.sqrt(proportion * (1.0 - proportion) / denominator), 6)
            if interval else None
        ),
        "ci_low_pct": low,
        "ci_high_pct": high,
        "confidence_level": 0.95 if interval else None,
        "interval_method": "wilson" if interval else "none_counts_only",
        "covariance": "binomial_independent" if interval else "none_counts_only",
        "cluster_count": None,
    }


def _summary(*, cohort: str = "cohort:study_population", interval: bool = True) -> dict[str, Any]:
    method = "wilson" if interval else "none_counts_only"
    return {
        "status": "ok",
        "analysis_family": "descriptive",
        "interpretation_class": "exposure_outcome_distribution",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "analysis_role": "secondary",
        "analysis_set": "bound_typed_cohort",
        "cohort_n": 100,
        "exposure": "early_injury",
        "outcome": "death",
        "interval_method": method,
        "effective_interval_method": method,
        "confidence_level": 0.95 if interval else None,
        "typed_cohort_input": cohort,
        "adjusted_effect": None,
        "descriptive_estimates": {
            "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
            "analysis_role": "secondary",
            "analysis_set": "bound_typed_cohort",
            "interpretation_ceiling": "descriptive_unadjusted_not_causal",
            "dependence": None,
            "exposure_prevalence": [
                _row(0, 70, 100, interval=interval),
                _row(1, 30, 100, interval=interval),
            ],
            "outcome_absolute_risks": [
                _row(0, 7, 70, interval=interval, events=True),
                _row(1, 6, 30, interval=interval, events=True),
            ],
            "risk_difference": None,
        },
        "descriptive_contrast": None,
        "output_files": {"table:exposure_outcome_distribution": "distribution.csv"},
    }


def _interval_text(count: int, denominator: int) -> str:
    low, high = wilson_interval(count, denominator, confidence_level=0.95)
    return f"95% CI, {low:.2f}% to {high:.2f}%"


def _registered(tmp_path, summary: dict[str, Any], *, step_id: str = "exposure_occurrence"):
    store = EvidenceStore(tmp_path / "run")
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    store.register_file(
        kind="statistic", source_path=source, evidence_id="occurrence_summary",
        description="Exposure occurrence", produced_by_step="exposure_occurrence",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id="exposure_occurrence", evidence_id="occurrence_summary", summary=summary,
    )
    records = [{
        "step_id": step_id, "status": "ok", "step_summary": summary,
        "step_summary_evidence_id": "occurrence_summary", "evidence_ids": ["occurrence_summary"],
    }]
    return store, records


def test_each_level_is_copied_with_its_interval(tmp_path) -> None:
    summary = _summary()
    store, records = _registered(tmp_path, summary)
    before = deepcopy(records)

    facts = _compile_study_population_occurrence_report_facts(records, store, LABELS)

    assert [fact.text for fact in facts] == [
        f"In the study cohort, 70 of 100 observations (70.00%; {_interval_text(70, 100)}) "
        "were in the “No early injury” group",
        f"In the study cohort, 30 of 100 observations (30.00%; {_interval_text(30, 100)}) "
        "were in the “Early injury” group",
    ]
    assert {fact.subsection for fact in facts} == {"Cohort characteristics"}
    assert {fact.required_result_sections for fact in facts} == {("Abstract", "Results")}
    assert all(fact.cohort_n is None and fact.replaces_claim_ref is None for fact in facts)
    assert facts[1].source_fields == tuple(
        f"descriptive_estimates.exposure_prevalence[1].{key}"
        for key in ("level", "n", "denominator", "estimate_pct", "ci_low_pct", "ci_high_pct", "confidence_level")
    )
    assert records == before


def test_a_counts_only_occurrence_carries_no_interval(tmp_path) -> None:
    store, records = _registered(tmp_path, _summary(interval=False))

    facts = _compile_study_population_occurrence_report_facts(records, store, LABELS)

    assert facts[0].text == (
        "In the study cohort, 70 of 100 observations (70.00%) were in the “No early injury” group"
    )


@pytest.mark.parametrize(
    "cohort", ["artifact:analysis_cohort", "cohort:analysis_set", "cohort:landmark_cohort"]
)
def test_a_distribution_on_another_cohort_is_not_the_occurrence(tmp_path, cohort) -> None:
    store, records = _registered(tmp_path, _summary(cohort=cohort))

    assert _compile_study_population_occurrence_report_facts(records, store, LABELS) == ()


def _break(kind: str, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    rows = summary["descriptive_estimates"]["exposure_prevalence"]
    if kind == "foreign_owner":
        records[0]["step_id"] = "another_step"
    elif kind == "share_not_count_over_denominator":
        rows[0]["estimate_pct"] = rows[0]["estimate_pct"] + 1.0
    elif kind == "interval_misses_share":
        rows[0]["ci_high_pct"] = rows[0]["estimate_pct"] - 1.0
    elif kind == "interval_missing":
        rows[0]["ci_low_pct"] = None
    elif kind == "counts_only_with_interval":
        rows[0].update(interval_method="none_counts_only")
    elif kind == "counts_do_not_partition":
        rows[1] = _row(1, 20, 100, interval=True)
    elif kind == "count_exceeds_denominator":
        rows[1].update(n=130, denominator=100)
    elif kind == "levels_out_of_order":
        rows.reverse()
    elif kind == "duplicate_level":
        rows[1]["level"] = 0
    elif kind == "boolean_count":
        rows[0]["n"] = True


@pytest.mark.parametrize(
    "kind",
    [
        "foreign_owner",
        "share_not_count_over_denominator",
        "interval_misses_share",
        "interval_missing",
        "counts_only_with_interval",
        "counts_do_not_partition",
        "count_exceeds_denominator",
        "levels_out_of_order",
        "duplicate_level",
        "boolean_count",
    ],
)
def test_a_contradictory_summary_is_refused(tmp_path, kind) -> None:
    summary = _summary()
    records = [{"step_id": "exposure_occurrence"}]
    _break(kind, summary, records)
    store, registered = _registered(tmp_path, summary, step_id=records[0]["step_id"])

    with pytest.raises(ValueError):
        _compile_study_population_occurrence_report_facts(registered, store, LABELS)


def test_the_placed_sentences_bind_every_number_under_strict_evidence(tmp_path) -> None:
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    store, records = _registered(tmp_path, _summary())
    facts = _compile_study_population_occurrence_report_facts(records, store, LABELS)

    projected = render_descriptive_report_claims(MANUSCRIPT, facts)
    bound, bindings, untraced = bind_numeric_values(
        projected, evidence=store, enforcement_mode=EvidenceEnforcementMode.STRICT,
        per_step_records=records,
    )

    abstract, results = projected.split("## Results", 1)
    assert all(fact.scaffold in abstract and fact.scaffold in results for fact in facts)
    assert bindings and not untraced
    assert missing_primary_result_facts(bound, facts) == {}


def test_the_shared_report_admission_carries_the_occurrence(tmp_path, monkeypatch) -> None:
    """The full write and the report-only repair admit facts through one owner."""

    from easyicu.research_agent.audits import envelope_consumers
    from easyicu.research_agent.reporting.descriptive_report_facts import (
        compile_primary_counts_only_report_facts,
    )

    store, records = _registered(tmp_path, _summary())
    # Envelope admission has its own contracts; here it passes the verified record.
    monkeypatch.setattr(
        envelope_consumers.RegisteredOutputEnvelopeConsumer,
        "authoritative_writer_records",
        lambda self, completed_step_records, *, evidence_store: list(completed_step_records),
    )

    facts = compile_primary_counts_only_report_facts(
        records, evidence=store, reader_display_labels=LABELS,
    )

    assert [fact.text for fact in facts] == [
        fact.text
        for fact in _compile_study_population_occurrence_report_facts(records, store, LABELS)
    ]
