"""A signed survival suite states its landmark cohort under Cohort characteristics.

The suite owns its Table 1, so no grouped Table 1 fact names the population it
analysed.  A Writer's cohort sentence uses the suite's own coordinates
("landmark", "exposure status"), which the strict Results grammar does not
admit, so Cohort characteristics ended empty and every signed survival study
failed the manuscript audit whatever its results.

The shared report-fact owner now copies the suite's recorded landmark
population, in the analysis unit of its typed reporting envelope, and places
it after the strict gate.  Synthetic study and seeded synthetic rows only
(renal replacement therapy and 90-day mortality).
"""

from __future__ import annotations

import copy
import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.result_envelope import (
    normalize_step_result_shadow,
    rebuild_observed_scalar_tree,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.reporting.descriptive_report_facts import (
    _compile_survival_cohort_report_facts,
    render_descriptive_report_claims,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_quality import audit_manuscript_quality
from easyicu.research_agent.reporting.manuscript_result_structure import required_result_subsections
from tests.support.survival_sealed import (
    bound_survival_plan,
    run_signed_suite,
    sealed_survival,
    synthetic_survival_rows,
)

STEP = "primary_survival_suite"
EVIDENCE = "statistic_step_summary_primary_survival_suite"


def _suite(tmp_path):
    context, authority = sealed_survival(tmp_path)
    plan = bound_survival_plan(context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
    summary = json.loads(json.dumps(run_signed_suite(authority, synthetic_survival_rows(), tmp_path / "out")))
    return plan, summary


def _registered(tmp_path, summary):
    """The summary registered as the host runner registers a signed owner's."""

    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir)
    store.register_file(
        kind="statistic", description="Signed survival suite summary", source_path=source,
        evidence_id=EVIDENCE, produced_by_step=STEP, producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(step_id=STEP, evidence_id=EVIDENCE, summary=summary)
    records = [{
        "step_id": STEP, "status": "ok", "step_summary": summary,
        "step_summary_evidence_id": EVIDENCE, "evidence_ids": [EVIDENCE],
    }]
    return store, records


def _writer_projection(tmp_path, records):
    """The Writer reads each summary through its sealed envelope's scalar tree."""

    output_dir = tmp_path / "projection"
    output_dir.mkdir()
    envelope = normalize_step_result_shadow(
        step_id=STEP, step_summary=records[0]["step_summary"], output_dir=output_dir, status="ok",
    )
    return [{**records[0], "step_summary": rebuild_observed_scalar_tree(envelope.observed_scalars)}]


def _filtered_draft(plan) -> str:
    """A survival manuscript after the strict gate removed the Writer's cohort prose."""

    subsections = "".join(f"### {heading}\n\n" for heading in required_result_subsections(plan))
    return (
        "# Synthetic survival study\n\n## Abstract\n\n"
        "**Background:** Renal replacement therapy is common in the ICU.\n\n"
        "**Methods:** We fitted a prespecified landmark survival suite.\n\n"
        "**Results:** See Figure 1.\n\n**Conclusions:** Independent validation is required.\n\n"
        "## Introduction\n\nThe question is prognostic.\n\n"
        f"## Results\n\n{subsections}"
        "## Discussion\n\nInterpretation stays in the Discussion.\n\n"
        "## Conclusion\n\nIndependent validation is required.\n"
    )


def _cohort(text: str) -> str:
    return text.split("### Cohort characteristics\n", 1)[1].split("\n### ", 1)[0]


def _empty_subsections(text: str, plan) -> set[str]:
    audit = audit_manuscript_quality(text, analysis_plan=plan, require_administrative_sections=False)
    return {
        excerpt
        for finding in audit.findings
        if finding.code == "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY"
        for excerpt in finding.excerpts
    }


def test_the_landmark_cohort_fills_cohort_characteristics_and_binds(tmp_path):
    plan, summary = _suite(tmp_path)
    store, records = _registered(tmp_path, summary)
    projected = _writer_projection(tmp_path, records)
    # The projection keeps the envelope's coordinates but not its version;
    # the digest-verified source summary is the typed authority.
    assert "schema_version" not in projected[0]["step_summary"]["reportable_survival_results"]

    [fact] = _compile_survival_cohort_report_facts(projected, store)
    unit = summary["reportable_survival_results"]["analysis_unit"]
    assert fact.text == (
        f"The landmark analysis cohort included {summary['n_landmark_population']:,} {unit}"
    )
    assert fact.source_fields == ("n_landmark_population",)
    assert fact.required_result_sections == ("Results",)
    assert _compile_survival_cohort_report_facts(records, store) == (fact,)

    draft = _filtered_draft(plan)
    assert "Cohort characteristics" in _empty_subsections(draft, plan)
    placed = render_descriptive_report_claims(draft, (fact,))
    assert fact.scaffold in _cohort(placed)
    assert "Cohort characteristics" not in _empty_subsections(placed, plan)
    assert render_descriptive_report_claims(placed, (fact,)) == placed

    bound = store.bind_manuscript(placed, per_step_records=records)
    _, binding, untraced = bind_numeric_values(bound, evidence=store, per_step_records=records)
    assert untraced == []
    assert {claim.source_field for claim in binding.values()} == {"n_landmark_population"}


def test_the_shared_report_admission_carries_the_landmark_cohort(tmp_path, monkeypatch):
    """The full write and the report-only repair admit facts through one owner."""

    from easyicu.research_agent.audits import envelope_consumers
    from easyicu.research_agent.reporting.descriptive_report_facts import (
        compile_primary_counts_only_report_facts,
    )

    _plan, summary = _suite(tmp_path)
    store, records = _registered(tmp_path, summary)
    projected = _writer_projection(tmp_path, records)
    # Envelope admission has its own contracts; here it returns the projection.
    monkeypatch.setattr(
        envelope_consumers.RegisteredOutputEnvelopeConsumer,
        "authoritative_writer_records",
        lambda self, completed_step_records, *, evidence_store: projected,
    )

    facts = compile_primary_counts_only_report_facts(records, evidence=store, reader_display_labels={})

    assert facts == _compile_survival_cohort_report_facts(projected, store)
    assert len(facts) == 1


def test_an_envelope_signed_before_typed_reporting_states_no_cohort(tmp_path):
    _plan, summary = _suite(tmp_path)
    legacy = copy.deepcopy(summary)
    legacy["reportable_survival_results"]["schema_version"] = "easyicu.survival_reporting/1"
    store, records = _registered(tmp_path, legacy)

    assert _compile_survival_cohort_report_facts(records, store) == ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_landmark_population", 12.5, "recorded integer count"),
        ("n_landmark_population", 0, "recorded integer count"),
        ("n_source", 1, "exceeds its source cohort"),
        ("analysis_unit", "ICU stays {evidence:other}", "reader noun phrase"),
    ],
)
def test_a_count_or_unit_the_suite_did_not_record_is_refused(tmp_path, field, value, message):
    _plan, summary = _suite(tmp_path)
    forged = copy.deepcopy(summary)
    if field == "analysis_unit":
        forged["reportable_survival_results"][field] = value
    else:
        forged[field] = value
    store, records = _registered(tmp_path, forged)

    with pytest.raises(ValueError, match=message):
        _compile_survival_cohort_report_facts(records, store)
