"""Study definition authority stays scoped to the occurrences it owns."""

from copy import deepcopy
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


def _register_context(store, context, *, producer="pipeline"):
    return store.register_json(
        kind="log", description="Frozen synthetic research context",
        payload=context.model_dump(mode="json"), filename="research_context.json",
        evidence_id="research_context", producer=producer, generation_mode="system",
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
    ("Creatinine was split at 2 mg/dL versus 2.0 mg/dL", ["2", "2.0"], ["groups[0].rule.value"]),
    ("We compared SOFA code 2 and code 1", ["2", "1"], ["levels[1].code", "levels[0].code"]),
    ("The creatinine cutoff was 2.0 mg/dL versus below 2 mg/dL", ["2.0", "2"], ["groups[0].rule.value"]),
    ("The creatinine cutoff was below 2.0 mg/dL or 2 mg/dL", ["2.0", "2"], ["groups[0].rule.value"]),
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


@pytest.mark.parametrize("sentence, number", [
    ("We grouped creatinine by a cutoff of 2.0 mg/dL, and 2 sensitivity models were run", "2"),
    ("We used SOFA level 2, and 1 secondary endpoint was analyzed", "1"),
    ("The creatinine threshold was 2 mg/dL and 2 alternate fits converged", "2"),
    ("We compared SOFA code 1, and 2 validation folds completed", "2"),
    ("We compared SOFA code 1 or 2", "2"),
    ("We used SOFA level 1 and level 9", "9"),
])
def test_coordinated_numbers_need_their_own_definition_attachment(typed_definition_store, sentence, number):
    store, records = typed_definition_store
    text = sentence + " {evidence:research_context}."
    with pytest.raises(EvidenceEnforcementError) as error:
        bind_numeric_values(text, evidence=store, per_step_records=records)
    assert error.value.detail["untraced_occurrences"] == [
        {"start": sentence.rindex(number), "end": sentence.rindex(number) + len(number), "value": number},
    ]
    _, removed, _, _, _ = _chain(store, records, text)
    assert [row["untraced"] for row in removed] == [[number]]


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
    "The SOFA 6 points threshold defined the strata {evidence:research_context}.",
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


@pytest.fixture
def cohort_plan_store(tmp_path):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    _register_context(store, ResearchContext.model_validate({
        "research_question": "Study respiratory-support-free days by creatinine",
        "cohort": {"cohort_name": "synthetic", "database": "eicu", "n_stays": 100},
        "variables": [
            {"name": "age", "dtype": "float64", "unit": "years"},
            {"name": "free_days", "dtype": "int64", "description": "Respiratory-support-free days"},
            {"name": "crea", "dtype": "float64", "unit": "mg/dL"},
            {"name": "urine_creatinine", "dtype": "float64", "unit": "mg/dL"},
        ],
    }))
    plan = {
        "research_question": "Study respiratory-support-free days by creatinine",
        "cohort": {"name": "synthetic", "inclusion": [
            _predicate("age", ">=", 54), _predicate("crea", "<", 2.5),
        ]},
        "display_labels": {"crea": "Creatinine"},
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
    }
    return store, plan


def _predicate(name, operator, value):
    return {
        "concept_id": name, "op": operator, "value": value, "aggregation": "first",
        "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 1},
    }


def _register_cohort_plan(store, plan, evidence_id):
    plan = AnalysisPlan.model_validate(plan)
    if evidence_id == "approved_executable_plan_revision_1":
        from easyicu.research_agent.authority.plan_lifecycle import (
            ApprovedExecutablePlan, NormalizedPlan, ProposedPlan, persist_approved_executable_plan,
        )
        from easyicu.research_agent.authority.plan_review import PlanReviewAuthority

        normalized = NormalizedPlan.create(proposed=ProposedPlan.create(plan=plan, source="planner_llm"),
                                           transformation_receipts=(), plan=plan)
        approved = ApprovedExecutablePlan.create(
            normalized=normalized, plan_review_authority=PlanReviewAuthority.create(plan=plan),
            decision_set_sha256="a" * 64,
        )
        persist_approved_executable_plan(run_dir=store.root, evidence=store, approved=approved)
        return "plan_payload."
    store.register_json(kind="log", description="Registered synthetic plan",
                        payload=plan.model_dump(mode="json"), filename="plan.json",
                        evidence_id="analysis_plan", producer="planner", generation_mode="system")
    return ""


@pytest.mark.parametrize("citation, owner", [
    ("research_context", "analysis_plan"),
    ("analysis_plan", "analysis_plan"),
    ("approved_executable_plan_revision_1", "approved_executable_plan_revision_1"),
])
def test_cohort_threshold_and_outcome_window_use_the_typed_plan_fields(cohort_plan_store, citation, owner):
    store, plan = cohort_plan_store
    prefix = _register_cohort_plan(store, plan, owner)
    cite = "{evidence:" + citation + "}."
    sentences = [
        "Age at hospital entry was 54.0 years or older " + cite,
        "Respiratory-support-free days were assessed within 37 days " + cite,
    ]
    for sentence in sentences:
        assert repair_miscited_numeric_citations(sentence, evidence=store) == (sentence, [])
    _, removed, final, bindings, untraced = _chain(store, [], "\n\n".join(sentences))
    assert removed == [] and untraced == []
    assert [claim.source_field for claim in bindings.values()] == [
        prefix + "cohort.inclusion[0].value",
        prefix + "steps[0].model_requirements[0].outcome_definition.free_days.horizon_days",
    ]
    assert {claim.evidence_id for claim in bindings.values()} == {owner}
    assert final.count("authority=definition") == 2

    result = "Respiratory-support-free days were 37.0 days " + cite
    _, removed, _, _, _ = _chain(store, [], result)
    assert [row["untraced"] for row in removed] == [["37.0"]]


@pytest.mark.parametrize("predicate, number", [
    ("Creatinine was below 2.5 mmol/L at screening", "2.5"),
    ("Urine creatinine was below 2.5 mg/dL", "2.5"),
    ("We used creatinine < 2.5 mmol/L", "2.5"),
    ("We screened on urine creatinine < 2.5 mg/dL", "2.5"),
])
@pytest.mark.parametrize("owner", ["analysis_plan", "approved_executable_plan_revision_1"])
def test_a_plan_citation_preserves_unit_and_compound_variable_guards(cohort_plan_store, predicate, number, owner):
    store, plan = cohort_plan_store
    _register_cohort_plan(store, plan, owner)
    _, removed, _, _, _ = _chain(store, [], predicate + " {evidence:" + owner + "}.")
    assert [row["untraced"] for row in removed] == [[number]]


def test_an_uncited_literal_predicate_also_needs_the_right_unit_and_variable(cohort_plan_store):
    store, plan = cohort_plan_store
    _register_cohort_plan(store, plan, "analysis_plan")
    _, removed, _, bindings, _ = _chain(store, [], "We screened on creatinine < 2.5.")
    assert not removed and next(iter(bindings.values())).source_field == "cohort.inclusion[1].value"
    for sentence in ("We screened on creatinine < 2.5 mmol/L.", "We screened on urine creatinine < 2.5 mg/dL."):
        _, removed, _, _, _ = _chain(store, [], sentence)
        assert [row["untraced"] for row in removed] == [["2.5"]]


@pytest.mark.parametrize("sentence", [
    "Eligibility required urine creatinine < 2.5.",
    "Urine creatinine below 2.5 qualified at entry {evidence:analysis_plan}.",
])
def test_an_unverified_context_cannot_enable_unguarded_plan_binding(tmp_path, sentence):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    question = "Describe the synthetic screening cohort"
    _register_context(store, ResearchContext.model_validate({
        "research_question": question,
        "cohort": {"cohort_name": "synthetic", "database": "eicu", "n_stays": 100},
        "variables": [
            {"name": "crea", "dtype": "float64", "unit": "mg/dL"},
            {"name": "urine_creatinine", "dtype": "float64", "unit": "mg/dL"},
        ],
    }), producer="importer")
    _register_cohort_plan(store, {
        "research_question": question,
        "cohort": {"name": "synthetic", "inclusion": [_predicate("crea", "<", 2.5)]},
        "display_labels": {"crea": "Creatinine"},
        "steps": [],
    }, "analysis_plan")

    with pytest.raises(EvidenceEnforcementError) as error:
        bind_numeric_values(sentence, evidence=store, per_step_records=[])
    assert [row["value"] for row in error.value.detail["untraced_occurrences"]] == ["2.5"]
    _, removed, _, _, _ = _chain(store, [], sentence)
    assert [row["untraced"] for row in removed] == [["2.5"]]


@pytest.mark.parametrize("second_horizon", [37, 42])
def test_repeated_outcome_windows_bind_only_when_the_requirements_agree(cohort_plan_store, second_horizon):
    store, plan = cohort_plan_store
    requirements = plan["steps"][0]["model_requirements"]
    alternative = deepcopy(requirements[0])
    alternative.update(requirement_id="alternative", analysis_role="sensitivity")
    alternative["outcome_definition"]["free_days"]["horizon_days"] = second_horizon
    requirements.append(alternative)
    _register_cohort_plan(store, plan, "analysis_plan")
    for horizon in dict.fromkeys((37, second_horizon)):
        sentence = "Respiratory-support-free days were evaluated through " + str(horizon)
        _, removed, final, bindings, _ = _chain(store, [], sentence + " days {evidence:research_context}.")
        if second_horizon == 37:
            assert not removed and final.count("authority=definition") == 1
            assert next(iter(bindings.values())).source_field.endswith("model_requirements[0].outcome_definition.free_days.horizon_days")
        else:
            assert [row["untraced"] for row in removed] == [[str(horizon)]]


@pytest.mark.parametrize("concept", ["plt", "PLT"])
@pytest.mark.parametrize("cites, evidence_id, field", [
    ("{evidence:research_context}", "research_context",
     "cohort.provenance.exposure_group_labels[0].parameters.groups[1].rule.value"),
    ("{evidence:analysis_plan}", "analysis_plan", "cohort.inclusion[0].value"),
    ("{evidence:analysis_plan} {evidence:research_context}", "research_context",
     "cohort.provenance.exposure_group_labels[0].parameters.groups[1].rule.value"),
])
def test_shared_cohort_and_exposure_boundaries_bind_to_a_cited_source(tmp_path, concept, cites, evidence_id, field):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    grouping = _grouping(concept, 117, "cells/µL")
    grouping["parameters"]["source_columns"] = {"value": "plt_first"}
    _register_context(store, ResearchContext.model_validate({
        "research_question": "Group platelet counts by a synthetic boundary",
        "cohort": {"cohort_name": "synthetic", "database": "eicu", "n_stays": 100,
                   "provenance": {"exposure_group_labels": [grouping]}},
        "variables": [{"name": "plt", "dtype": "int64", "description": "Platelet count", "unit": "cells/µL"}],
    }))
    _register_cohort_plan(store, {
        "research_question": "Group platelet counts by a synthetic boundary",
        "cohort": {"name": "synthetic", "inclusion": [
            _predicate("plt", ">=", 117), _predicate("plt", ">=", 117),
        ]},
        "steps": [],
    }, "analysis_plan")
    _, removed, _, bindings, _ = _chain(store, [], f"PLT at admission was no less than 117 cells/µL {cites}.")
    assert not removed and len(bindings) == 1
    claim = next(iter(bindings.values()))
    assert (claim.evidence_id, claim.source_field) == (evidence_id, field)


def test_adult_inclusion_does_not_mask_an_age_group_boundary(tmp_path):
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    question = "Compare older and younger adult strata"
    _register_context(store, ResearchContext.model_validate({
        "research_question": question,
        "cohort": {"cohort_name": "synthetic", "database": "eicu", "n_stays": 100,
                   "provenance": {"exposure_group_labels": [_grouping("age", 80, "years")]}},
        "variables": [{"name": "age", "dtype": "float64", "unit": "years"}],
    }))
    _register_cohort_plan(store, {
        "research_question": question,
        "cohort": {"name": "synthetic", "inclusion": [_predicate("age", ">=", 18)]},
        "steps": [],
    }, "analysis_plan")

    text = (
        "Age was stratified as under 80 {evidence:research_context}.\n\n"
        "Age of 80 or older defined the second stratum {evidence:research_context}."
    )
    _, removed, final, bindings, untraced = _chain(store, [], text)
    assert removed == [] and untraced == []
    assert final.count("authority=definition") == 2
    assert [(claim.evidence_id, claim.source_field) for claim in bindings.values()] == [
        ("research_context", f"cohort.provenance.exposure_group_labels[0].parameters.groups[{pos}].rule.value")
        for pos in range(2)
    ]
    _, removed, _, _, _ = _chain(store, [], "Age was grouped at 18 years {evidence:research_context}.")
    assert [row["untraced"] for row in removed] == [["18"]]


def test_uncited_age_literal_activates_only_its_matching_predicate(cohort_plan_store):
    store, plan = cohort_plan_store
    plan["cohort"]["inclusion"][0] = _predicate("age", ">=", 18)
    plan["cohort"]["exclusion"] = [_predicate("age", ">=", 90)]
    _register_cohort_plan(store, plan, "analysis_plan")

    text = "The screening protocol required age >= 18."
    _, removed, _, bindings, untraced = _chain(store, [], text)
    assert removed == [] and untraced == []
    assert [(claim.value, claim.source_field) for claim in bindings.values()] == [
        ("18", "cohort.inclusion[0].value"),
    ]


def test_two_cohort_cutoffs_bind_to_their_own_values(cohort_plan_store):
    store, plan = cohort_plan_store
    plan["cohort"]["inclusion"] = [_predicate("age", ">=", 18), _predicate("age", "<=", 85)]
    _register_cohort_plan(store, plan, "analysis_plan")

    text = "For screening, the age cutoff was 18 years and 85 years {evidence:analysis_plan}."
    _, removed, _, bindings, untraced = _chain(store, [], text)
    assert removed == [] and untraced == []
    assert [(claim.value, claim.source_field) for claim in bindings.values()] == [
        ("18", "cohort.inclusion[0].value"), ("85", "cohort.inclusion[1].value"),
    ]


@pytest.mark.parametrize("sentence, value", [
    ("We screened for age above 73 years, and 73 years was the recorded median {evidence:research_context}.", "73"),
    ("We screened on creatinine below 2.5 mg/dL, and 2.5 mg/dL was the observed median {evidence:analysis_plan}.",
     "2.5"),
])
def test_an_unmarked_threshold_cannot_release_a_coordinated_result(cohort_plan_store, sentence, value):
    store, plan = cohort_plan_store
    plan["cohort"]["inclusion"][0] = _predicate("age", ">", 73)
    _register_cohort_plan(store, plan, "analysis_plan")

    with pytest.raises(EvidenceEnforcementError) as error:
        bind_numeric_values(sentence, evidence=store, per_step_records=[])
    start = sentence.rindex(value)
    assert error.value.detail["untraced_occurrences"] == [
        {"start": start, "end": start + len(value), "value": value},
    ]
    _, removed, _, _, _ = _chain(store, [], sentence)
    assert [row["untraced"] for row in removed] == [[value]]


def test_over_can_state_a_strict_boundary_or_a_duration(cohort_plan_store):
    store, plan = cohort_plan_store
    plan["cohort"]["inclusion"][0] = _predicate("age", ">", 73)
    _register_cohort_plan(store, plan, "analysis_plan")
    text = (
        "Age over 73 years qualified for screening {evidence:research_context}.\n\n"
        "Respiratory-support-free days were evaluated over 37 days {evidence:research_context}."
    )
    _, removed, final, bindings, _ = _chain(store, [], text)
    assert not removed and final.count("authority=definition") == 2
    assert [claim.value for claim in bindings.values()] == ["73", "37"]
    _, removed, _, _, _ = _chain(store, [], "Age of 73 or older qualified {evidence:research_context}.")
    assert [row["untraced"] for row in removed] == [["73"]]
