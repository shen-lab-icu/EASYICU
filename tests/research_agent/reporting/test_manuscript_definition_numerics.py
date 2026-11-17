"""Study definition authority stays scoped to the occurrences it owns."""

from pathlib import Path
import re

import pytest

from easyicu.research_agent.audits.manuscript_claims import audit_manuscript_numeric_claims
from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementError, EvidenceStore
from easyicu.research_agent.reporting.manuscript_post import (
    bind_numeric_values, drop_untraceable_numeric_sentences, repair_miscited_numeric_citations,
)
from easyicu.research_agent.schema import AnalysisPlan, ResearchContext


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


def _grouping(concept, threshold, unit):
    return {
        "variable": f"{concept}_group_x1", "concept": concept, "scale": "ordinal",
        "levels": [
            {"code": 1, "group": "g1", "label": "Lower measurements"},
            {"code": 2, "group": "g2", "label": "Elevated measurements"},
        ],
        "parameters": {
            "concept": concept, "window": None, "scale": "ordinal",
            "unmeasured": "exclude", "source_columns": {"value": concept},
            "groups": [
                {"id": group, "rule": {"summary": "value", "op": op, "value": threshold, "unit": unit}}
                for group, op in (("g1", "<"), ("g2", ">="))
            ],
        },
    }


def _register_context(store, context):
    return store.register_json(
        kind="log", description="Frozen synthetic research context",
        payload=context.model_dump(mode="json"), filename="research_context.json",
        evidence_id="research_context", producer="pipeline", generation_mode="system",
        aliases=["RUN_CONTEXT"],
    )


@pytest.fixture
def typed_definition_store(definition_store):
    store, records = definition_store
    context = ResearchContext.model_validate({
        "research_question": "SOFA groups and readmission",
        "cohort": {"cohort_name": "synthetic", "database": "eicu", "n_stays": 100,
                   "provenance": {"exposure_group_labels": [
                       _grouping("sofa", 8, "points"),
                       _grouping("creatinine", 2, "mg/dL"),
                   ]}},
        "primary_exposure": "sofa_group_x1", "target_outcome": "readmission",
        "variables": [
            {"name": "sofa", "dtype": "int64", "description": "SOFA score", "unit": "points"},
            {"name": "creatinine", "dtype": "float64", "unit": "mg/dL"},
            {"name": "glucose", "dtype": "float64", "unit": "mg/dL"},
            {"name": "urine_creatinine", "dtype": "float64", "unit": "mg/dL"},
        ],
    })
    _register_context(store, context)
    return store, records


PARAPHRASED = "SOFA score was represented by two ordered categories: under 8.0 points and 8.0 points or higher"


@pytest.mark.parametrize("text, values, fields", [
    (PARAPHRASED, ["8.0", "8.0"], ["groups[0].rule.value", "groups[1].rule.value"]),
    ("The primary contrast was SOFA of 8 points or higher versus below 8 points, "
     "with the under-8 group as the reference", ["8", "8", "8"],
     ["groups[1].rule.value", "groups[0].rule.value"]),
    ("We compared SOFA >= 8 vs. < 8", ["8", "8"], ["groups[1].rule.value", "groups[0].rule.value"]),
    ("We grouped creatinine by a cutoff of 2.0 mg/dL", ["2.0"], ["groups[0].rule.value"]),
    ("We used SOFA level 2 as the comparison group", ["2"], ["levels[1].code"]),
    ("Creatinine was dichotomized at 2 mg/dL", ["2"], ["groups[0].rule.value"]),
])
def test_a_cited_typed_definition_binds_each_paraphrased_occurrence(typed_definition_store, text, values, fields):
    store, records = typed_definition_store
    registry_before = [claim.to_dict() for claim in store.numeric_claims()]
    _, removed, final, bindings, untraced = _chain(store, records, text + " {evidence:RUN_CONTEXT}.")
    assert removed == [] and untraced == []
    mentions, _ = final.split("\n\n", 1)
    assert re.findall(r"(\d+(?:\.\d+)?)\[\^claim_\d+\]", mentions) == values
    assert [claim.source_field.split(".parameters.")[-1].split("exposure_group_labels[0].")[-1]
            for claim in bindings.values()] == fields
    assert {claim.evidence_id for claim in bindings.values()} == {"research_context"}
    assert f"sha256={store.get('research_context').sha256}; authority=definition" in final
    assert audit_manuscript_numeric_claims(final, per_step_records=records) == []
    assert [claim.to_dict() for claim in store.numeric_claims()] == registry_before


@pytest.mark.parametrize("text", [
    PARAPHRASED + ".",
    PARAPHRASED + " according to research_context.",
    PARAPHRASED + " {evidence:contrast}.",
    "Definitions came from the study {evidence:research_context}. " + PARAPHRASED + ".",
    "Definitions came from the study. {evidence:research_context} " + PARAPHRASED + ".",
    "The SOFA threshold was an OR of 8.0 {evidence:research_context}.",
    "SOFA was elevated in 8% of stays {evidence:research_context}.",
    "The mean creatinine was 2.0 mg/dL {evidence:research_context}.",
    "The creatinine threshold was 2.0 mg/dL (95% CI 1.2 to 3.4) {evidence:research_context}.",
    "The creatinine threshold was 2.0 mg/dL (median) {evidence:research_context}.",
    "For SOFA we reported IQR 8.0 to 9.0 {evidence:research_context}.",
    "For SOFA we reported n=8 {evidence:research_context}.",
    "For SOFA we reported p=8.0 {evidence:research_context}.",
    "SOFA below 8.0 points <!-- {evidence:research_context} -->.",
    "The creatinine cutoff was 2.0%, as a proportion {evidence:research_context}.",
    "Glucose below 2.0 mg/dL defined eligibility {evidence:research_context}.",
    "Urine creatinine below 2.0 mg/dL defined eligibility {evidence:research_context}.",
    "We recorded creatinine and excluded glucose below 2.0 mg/dL {evidence:research_context}.",
    "We used creatinine below 2.0 mmol/L {evidence:research_context}.",
    "Creatinine was above 2.0 mg/dL {evidence:research_context}.",
    "The discrimination threshold was 0.60 {evidence:research_context}.",
])
def test_a_matching_literal_without_definition_authority_is_removed(typed_definition_store, text):
    store, records = typed_definition_store
    kept, removed, _, _, _ = _chain(store, records, text)
    assert PARAPHRASED not in kept
    assert not kept.strip() or kept.strip().startswith("Definitions came from the study")
    assert len(removed) == 1


def test_a_definition_citation_cannot_release_a_drifted_context(typed_definition_store):
    store, records = typed_definition_store
    text = store.bind_manuscript(PARAPHRASED + " {evidence:research_context}.", per_step_records=records)
    source = store.get("research_context")
    Path(store.root, source.relative_path).write_text("{}")
    _, removed = drop_untraceable_numeric_sentences(text, evidence=store, per_step_records=records)
    assert [row["untraced"] for row in removed] == [["8.0", "8.0"]]


def test_cited_threshold_and_equal_valued_result_retain_separate_sources(typed_definition_store):
    store, records = typed_definition_store
    summary = {"odds_ratio": 8.0}
    store.register_json(kind="statistic", description="An equal-valued result", payload=summary,
                        filename="equal.json", evidence_id="equal_result", produced_by_step="equal")
    store.register_step_summary_numerics(step_id="equal", evidence_id="equal_result", summary=summary)
    records.append({"step_id": "equal", "status": "ok", "step_summary": summary,
                    "evidence_ids": ["equal_result"], "step_summary_evidence_id": "equal_result"})
    text = ("For SOFA below 8 points, the OR was 8.0 {evidence:research_context} "
            "{evidence:equal_result}.")
    _, removed, final, bindings, untraced = _chain(store, records, text)
    assert removed == [] and untraced == []
    assert [(claim.value, claim.evidence_id) for claim in bindings.values()] == [
        ("8", "research_context"), ("8", "equal_result"),
    ]
    assert final.count("authority=definition") == 1
    assert "8.0[^claim_2]" in final
    assert audit_manuscript_numeric_claims(final, per_step_records=records) == []
    definition = "We compared SOFA below 8.0 points {evidence:research_context}."
    assert repair_miscited_numeric_citations(definition, evidence=store) == (definition, [])


def test_cohort_threshold_and_outcome_window_use_the_typed_plan_fields(tmp_path):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    _register_context(store, ResearchContext.model_validate({
        "research_question": "Study ventilator-free days by creatinine",
        "cohort": {"cohort_name": "synthetic", "database": "eicu", "n_stays": 100},
        "variables": [
            {"name": "age", "dtype": "float64", "unit": "years"},
            {"name": "free_days", "dtype": "int64", "description": "Ventilator-free days"},
        ],
    }))
    plan = AnalysisPlan.model_validate({
        "research_question": "Study ventilator-free days by creatinine",
        "cohort": {"name": "synthetic", "inclusion": [{
            "concept_id": "age", "op": ">=", "value": 65, "aggregation": "first",
            "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 1},
        }]},
        "steps": [{
            "step_id": "model", "intent": "Estimate an association", "method": "adjusted_association_models",
            "expected_outputs": ["table:adjusted_association_estimates"],
            "model_requirements": [{
                "requirement_id": "model", "outcome": "free_days", "outcome_type": "free_days",
                "method_family": "statsmodels_ordered_logit", "exposure_source": "creatinine",
                "analysis_role": "primary", "analysis_set": "source_aware",
                "outcome_definition": {
                    "estimand": "Common odds ratio for more support-free days",
                    "free_days": {"horizon_days": 37, "support_concept": "mech_vent", "death_score": 0},
                },
            }],
        }],
    })
    store.register_json(kind="log", description="Registered synthetic plan",
                        payload=plan.model_dump(mode="json"), filename="plan.json",
                        evidence_id="analysis_plan", producer="planner", generation_mode="system")
    text = (
        "We included patients with age at ICU admission of 65.0 years or older {evidence:research_context}.\n\n"
        "Ventilator-free days were assessed within 37 days {evidence:research_context}."
    )
    _, removed, final, bindings, untraced = _chain(store, [], text)
    assert removed == [] and untraced == []
    assert [claim.source_field for claim in bindings.values()] == [
        "cohort.inclusion[0].value",
        "steps[0].model_requirements[0].outcome_definition.free_days.horizon_days",
    ]
    assert {claim.evidence_id for claim in bindings.values()} == {"analysis_plan"}
    assert final.count("authority=definition") == 2

    result = "Ventilator-free days were 37.0 days {evidence:research_context}."
    _, removed, _, _, _ = _chain(store, [], result)
    assert [row["untraced"] for row in removed] == [["37.0"]]
