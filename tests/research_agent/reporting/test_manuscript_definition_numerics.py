"""Study definitions authorize exact spans, never equal-valued results."""

from pathlib import Path

import pytest

from easyicu.research_agent.audits.manuscript_claims import audit_manuscript_numeric_claims
from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementError, EvidenceStore
from easyicu.research_agent.reporting.manuscript_post import (
    bind_numeric_values, drop_untraceable_numeric_sentences,
)
from easyicu.research_agent.schema import AnalysisPlan


@pytest.fixture
def definition_store(tmp_path):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    plan = AnalysisPlan.model_validate({
        "research_question": "Creatinine groups and readmission", "steps": [{
            "step_id": "cohort", "intent": "Describe eligibility", "method": "descriptive",
            "expected_outputs": ["artifact:analysis_cohort", "table:cohort_flow"],
            "cohort_definition_spec": {
                "identity_column": "stay_id", "eligibility_criteria": [{
                    "criterion_id": "age", "description": "patients aged 65 years or older",
                }],
            },
        }],
        "display_labels": {
            "creatinine_group=high": "creatinine at least 2 mg/dL",
            "creatinine_group=low": "creatinine below 1 mg/dL",
            "severity=high": "SOFA at least 8 points",
            "discrimination_group": "AUROC at least 0.60",
            "trace_group": "marker at least 1 × 10⁻³ mg/dL",
        },
    })
    store.register_json(kind="log", description="Registered synthetic plan",
                        payload=plan.model_dump(mode="json"), filename="plan.json",
                        evidence_id="analysis_plan", producer="planner", generation_mode="system")
    summary = {"odds_ratio": 1.73, "ci_low": 1.21, "ci_high": 2.47}
    store.register_json(kind="statistic", description="Synthetic contrast", payload=summary,
                        filename="contrast.json", evidence_id="contrast", produced_by_step="model")
    store.register_step_summary_numerics(step_id="model", evidence_id="contrast", summary=summary)
    records = [{"step_id": "model", "status": "ok", "step_summary": summary,
                "evidence_ids": ["contrast"], "step_summary_evidence_id": "contrast"}]
    return store, records


RESULT = (
    "After adjustment for age, creatinine at least 2 mg/dL versus creatinine below 1 mg/dL "
    "was associated with readmission (odds ratio, 1.73; 95% CI, 1.21 to 2.47) {evidence:contrast}."
)


def _chain(store, records, text):
    bound = store.bind_manuscript(text, per_step_records=records)
    kept, removed = drop_untraceable_numeric_sentences(bound, evidence=store, per_step_records=records)
    final, bindings, untraced = bind_numeric_values(kept, evidence=store, per_step_records=records)
    return kept, removed, final, bindings, untraced


@pytest.mark.parametrize("section", ["Abstract", "Results", "Conclusion"])
def test_labelled_threshold_result_keeps_estimate_and_definition_sources(definition_store, section):
    store, records = definition_store
    kept, removed, final, bindings, untraced = _chain(store, records, f"## {section}\n\n{RESULT}")
    assert removed == [] and untraced == [] and "1.73" in kept
    assert {c.value for c in bindings.values() if c.evidence_id == "contrast"} == {"1.73", "1.21", "2.47"}
    assert {c.value for c in bindings.values() if c.evidence_id == "analysis_plan"} == {"1", "2"}
    assert "field=display_labels" in final and "sha256=" in final
    assert audit_manuscript_numeric_claims(final, per_step_records=records) == []


@pytest.mark.parametrize("extra", ["with 2 events", "with 8 deaths", "with 65 events", "with 991 events"])
def test_definition_value_does_not_authorize_an_extra_result(definition_store, extra):
    store, records = definition_store
    text = RESULT.replace("was associated", extra + ", was associated")
    kept, removed, _, _, unknown = _chain(store, records, text)
    assert not unknown and "1.73" not in kept
    assert len(removed) == 1
    assert removed[0]["untraced"] == [extra.split()[1]]


def test_definition_numeral_elsewhere_remains_untraced(definition_store):
    store, records = definition_store
    kept, removed, _, _, _ = _chain(store, records, RESULT + "\n\nThere were 2 events {evidence:contrast}.")
    assert "1.73" in kept and "There were" not in kept
    assert [r["untraced"] for r in removed] == [["2"]]


@pytest.mark.parametrize("definition", ["patients aged 65 years or older", "SOFA at least 8 points"])
def test_exact_plan_definition_thresholds_bind_without_result_claims(definition_store, definition):
    store, records = definition_store
    _, removed, _, bindings, unknown = _chain(store, records, f"## Methods\n\nWe included {definition}.")
    assert not removed and not unknown
    assert len(bindings) == 1 and next(iter(bindings.values())).evidence_id == "analysis_plan"


@pytest.mark.parametrize("replacement", ["creatinine at least 3 mg/dL", "creatinine at least 2.0 mg/dL"])
def test_edited_or_partial_label_does_not_gain_definition_authority(definition_store, replacement):
    store, records = definition_store
    text = RESULT.replace("creatinine at least 2 mg/dL", replacement)
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(text, evidence=store, per_step_records=records)
    assert len(drop_untraceable_numeric_sentences(text, evidence=store, per_step_records=records)[1]) == 1


def test_stale_registered_plan_gives_no_label_release(definition_store):
    store, records = definition_store
    record = store.get("analysis_plan")
    Path(store.root, record.relative_path).write_text("{}")
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(RESULT, evidence=store, per_step_records=records)


def test_numeric_auditor_consumes_definition_authority_from_the_binder(definition_store):
    store, records = definition_store
    summary = {"auroc": 0.74}
    store.register_json(kind="statistic", description="Synthetic discrimination", payload=summary,
                        filename="metric.json", evidence_id="metric", produced_by_step="discrimination")
    store.register_step_summary_numerics(step_id="discrimination", evidence_id="metric", summary=summary)
    records.append({"step_id": "discrimination", "status": "ok", "step_summary": summary,
                    "evidence_ids": ["metric"], "step_summary_evidence_id": "metric"})
    text = ("We defined the subgroup as AUROC at least 0.60 {evidence:analysis_plan}.\n\n"
            "The AUROC was 0.74 {evidence:metric}.")
    _, removed, final, _, unknown = _chain(store, records, text)
    assert not removed and not unknown
    assert audit_manuscript_numeric_claims(final, per_step_records=records) == []
    _, removed, final, _, _ = _chain(store, records, text + "\n\nThe AUROC was 0.60 {evidence:metric}.")
    assert [r["untraced"] for r in removed] == [["0.60"]]
    assert audit_manuscript_numeric_claims(final, per_step_records=records) == []


def test_approved_plan_overrides_proposed_definition_labels(definition_store):
    import json

    from easyicu.research_agent.authority.plan_lifecycle import (
        ApprovedExecutablePlan, NormalizedPlan, ProposedPlan, persist_approved_executable_plan,
    )
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority

    store, records = definition_store
    plan = AnalysisPlan.model_validate(json.loads(Path(store.root, store.get("analysis_plan").relative_path).read_text()))
    plan.display_labels = {"creatinine_group=high": "creatinine at least 3 mg/dL"}
    normalized = NormalizedPlan.create(proposed=ProposedPlan.create(plan=plan, source="planner_llm"),
                                       transformation_receipts=(), plan=plan)
    approved = ApprovedExecutablePlan.create(normalized=normalized,
                plan_review_authority=PlanReviewAuthority.create(plan=plan), decision_set_sha256="a" * 64)
    persist_approved_executable_plan(run_dir=store.root, evidence=store, approved=approved)
    text = "We defined the exposure as creatinine at least 3 mg/dL."
    _, removed, final, bindings, _ = _chain(store, records, text)
    assert not removed and len(bindings) == 1
    claim = next(iter(bindings.values()))
    assert claim.evidence_id == "approved_executable_plan_revision_1"
    assert claim.source_field == "plan_payload.display_labels.creatinine_group=high"
    assert "authority=definition" in final
    with pytest.raises(EvidenceEnforcementError):
        bind_numeric_values(RESULT, evidence=store, per_step_records=records)


def test_a_scientific_notation_definition_is_one_numeric_occurrence(definition_store):
    store, records = definition_store
    text = "We used marker at least 1 × 10⁻³ mg/dL."
    _, removed, final, bindings, unknown = _chain(store, records, text)
    assert not removed and not unknown
    assert len(bindings) == 1
    assert next(iter(bindings.values())).value == "1 × 10⁻³"
    assert final.count("1 × 10⁻³") == 2  # Its one prose occurrence and source value in the footnote.
