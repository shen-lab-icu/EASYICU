"""A formal rule outcome reaches the Results, Conclusion and Abstract it answers.

The strict Results grammar admits host claim tokens and neutral numeric facts.
A run whose owners all succeeded while their rules selected nothing had no
claim to place: its required subsections were empty or held only a figure
callout, the Conclusion kept only its validation caveat, and the quality audit
blocked the manuscript.  The same happened to a planned analysis the inputs
could not support.  Host rule-outcome claims now fill the subsection their
rule owns, and the Conclusion reads the family's primary claim.

Synthetic studies only: a lactate and mean arterial pressure trajectory over
0-48 h with no interior class solution, and a lactate-readmission association
with a non-executable sensitivity analysis.
"""

from __future__ import annotations

import re

from easyicu.research_agent.authority.manuscript_claim_policy import (
    expand_scientific_claim_tokens,
    filter_evidence_bound_scaffold,
    missing_scientific_claims_in_results,
    place_scientific_claim_tokens_in_results,
)
from easyicu.research_agent.authority.prespecified_rule_outcomes import (
    RULE_OUTCOME_SCHEMA_VERSION,
)
from easyicu.research_agent.authority.scientific_claims import (
    ScientificClaim,
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality,
    repair_reader_structure_from_existing_prose,
)
from easyicu.research_agent.reporting.manuscript_result_structure import (
    planned_result_roles,
    required_result_subsections,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

BLOCKING_CODES = {
    "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY",
    "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY",
    "MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY",
    "MANUSCRIPT_CONCLUSION_WITHOUT_INTERPRETATION",
}


def _plan(analysis_type: str, roles: dict[str, str]) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Synthetic question.",
        analysis_type=analysis_type,
        steps=[
            AnalysisStep(
                step_id=step_id,
                planned_analysis_role=role,
                intent="Run the prespecified step.",
                method="descriptive",
                inputs=["artifact:analysis_cohort"],
                expected_outputs=[f"table:{step_id}"],
            )
            for step_id, role in roles.items()
        ],
    )


TRAJECTORY_PLAN = _plan(
    "trajectory_clustering",
    {"00_panel": "auxiliary", "01_candidates": "primary", "05_description": "secondary"},
)


def _claims(step_id: str, summary: dict) -> list[ScientificClaim]:
    return bind_scientific_claim_drafts(
        [draft.model_dump(mode="json") for draft in derive_scientific_claim_drafts(summary)],
        step_id=step_id,
        evidence_id=f"{step_id}_summary",
    )


def _outcome(**payload) -> dict:
    return {"status": "ok", "reportable_rule_outcomes": [
        {"schema_version": RULE_OUTCOME_SCHEMA_VERSION, **payload}
    ]}


def _trajectory_claims() -> list[ScientificClaim]:
    return [
        *_claims("00_panel", _outcome(
            rule="minimum_observed_windows", anchor="icu_admission",
            window_start_hours=0, window_end_hours=48, window_width_hours=8,
            n_windows=6, minimum_observed_windows=3,
            input_n=1402, included_n=1240, excluded_n=162,
        )),
        *_claims("01_candidates", _outcome(
            rule="information_criterion_class_count", criterion="bic",
            candidate_class_counts=[2, 3, 4, 5], criterion_minimum_class_count=5,
            n_records=1240, smallest_class_fraction=0.081,
            minimum_class_fraction=0.05, disposition="minimum_at_upper_boundary",
        )),
        *_claims("05_description", _outcome(
            rule="frozen_class_description", disposition="no_frozen_solution",
        )),
    ]


def _manuscript(results: str, *, conclusion: str, abstract_conclusion: str = "") -> str:
    return "\n".join([
        "# Synthetic trajectory study",
        "",
        "**Keywords:** trajectory; intensive care",
        "",
        "## Abstract",
        "",
        "**Background:** Early physiology varies between ICU stays.",
        "",
        "**Methods:** We fitted prespecified latent class models to panel windows.",
        "",
        "**Results:** The cohort comprised 1,402 ICU stays {evidence:research_context}.",
        "",
        f"**Conclusions:** {abstract_conclusion}".rstrip(),
        "",
        "## Introduction",
        "",
        "Trajectory classes are often proposed for ICU physiology.",
        "",
        "## Methods",
        "",
        "### Study design and cohort",
        "",
        "This was a retrospective cohort study.",
        "",
        "### Variables",
        "",
        "Lactate and mean arterial pressure were the panel coordinates.",
        "",
        "### Statistical analysis",
        "",
        "Latent class models were compared by a prespecified criterion.",
        "",
        "### Software and reproducibility",
        "",
        "Analyses used Python.",
        "",
        "## Results",
        "",
        results,
        "",
        "## Discussion",
        "",
        "Interpretation stays in the Discussion.",
        "",
        "## Limitations",
        "",
        "The data come from one source.",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
    ])


# What the strict filter leaves of a Writer's trajectory Results when no rule
# outcome is a claim: the headings and the registered figure callout.
FILTERED_TRAJECTORY_RESULTS = "\n\n".join([
    "### Cohort characteristics",
    "### Cluster characteristics",
    "See Figure 1 {evidence:publication_figure_contract}.",
    "### Secondary analyses",
])


def _findings(text: str, plan: AnalysisPlan) -> dict[str, list]:
    audit = audit_manuscript_quality(
        text, analysis_plan=plan, require_administrative_sections=False,
    )
    found: dict[str, list] = {}
    for finding in audit.findings:
        if finding.code in BLOCKING_CODES:
            found.setdefault(finding.code, []).append(finding.excerpts)
    return found


def _section(text: str, heading: str) -> str:
    body = text.split(f"{heading}\n", 1)[1]
    return re.split(r"\n#{2,3} ", body, maxsplit=1)[0]


def _through_the_host(text: str, claims: list[ScientificClaim], plan: AnalysisPlan):
    placement = place_scientific_claim_tokens_in_results(
        text, claims=claims, planned_step_roles=planned_result_roles(plan),
    )
    repaired, repairs = repair_reader_structure_from_existing_prose(placement.scaffold)
    by_ref = {claim.claim_ref: claim for claim in claims}
    expanded = expand_scientific_claim_tokens(repaired, resolve_claim=by_ref.get)
    assert expanded.missing_claim_refs == ()
    assert expanded.malformed_sentences == ()
    return repaired, expanded.scaffold, {repair["code"] for repair in repairs}


def test_without_a_rule_claim_the_no_solution_manuscript_is_blocked() -> None:
    # The defect: every owner succeeded, yet nothing can be said.
    text = _manuscript(FILTERED_TRAJECTORY_RESULTS,
                       conclusion="Independent validation is required.")

    repaired, _ = repair_reader_structure_from_existing_prose(text)
    found = _findings(repaired, TRAJECTORY_PLAN)

    assert set(found) == BLOCKING_CODES


def test_the_rule_outcomes_fill_each_subsection_their_rules_own() -> None:
    claims = _trajectory_claims()
    eligibility, class_count, description = claims
    text = _manuscript(FILTERED_TRAJECTORY_RESULTS,
                       conclusion="Independent validation is required.")
    assert required_result_subsections(TRAJECTORY_PLAN) == (
        "Cohort characteristics", "Cluster characteristics", "Secondary analyses",
    )

    repaired, reader, repairs = _through_the_host(text, claims, TRAJECTORY_PLAN)

    assert eligibility.placeholder in _section(repaired, "### Cohort characteristics")
    assert class_count.placeholder in _section(repaired, "### Cluster characteristics")
    assert description.placeholder in _section(repaired, "### Secondary analyses")
    # The Conclusion is the primary rule's bounded reading, not the cohort
    # claim placed before it; the validation caveat stays.
    assert _section(repaired, "## Conclusion").split() == [
        class_count.placeholder, "Independent", "validation", "is", "required.",
    ]
    assert {"MANUSCRIPT_CONCLUSION_RESTORED",
            "MANUSCRIPT_ABSTRACT_CONCLUSIONS_RESTORED"} <= repairs
    assert class_count.render_reader_text(include_estimate=False) in _section(
        reader, "## Conclusion"
    )
    assert class_count.render_reader_text() in _section(
        reader, "### Cluster characteristics"
    )
    assert _findings(repaired, TRAJECTORY_PLAN) == {}
    assert _findings(reader, TRAJECTORY_PLAN) == {}
    assert missing_scientific_claims_in_results(reader, claims=claims) == ()


def test_the_strict_filter_keeps_the_token_and_drops_a_paraphrase() -> None:
    claims = _trajectory_claims()
    class_count = claims[1]
    by_ref = {claim.claim_ref: claim for claim in claims}
    results = "\n\n".join([
        "### Cohort characteristics",
        "### Cluster characteristics",
        class_count.placeholder,
        # What a Writer wrote from the criterion's minimum.
        "The selected solution contained 5 clusters {evidence:01_candidates_summary}.",
        "No interior solution existed, so no classes were described "
        "{evidence:01_candidates_summary}.",
        "### Secondary analyses",
    ])

    filtered = filter_evidence_bound_scaffold(
        _manuscript(results, conclusion="Independent validation is required."),
        resolve_claim=by_ref.get,
        resolve_evidence=lambda ref: True,
    )

    body = _section(filtered.scaffold, "### Cluster characteristics")
    assert class_count.placeholder in body
    assert "No interior solution existed" not in body
    assert "selected solution" not in body


def test_a_non_executable_sensitivity_analysis_is_stated_in_its_subsection() -> None:
    plan = _plan("association_study", {
        "02_lactate_readmission": "primary",
        "08_repeat_lactate_protocol": "sensitivity",
    })
    [primary] = _claims("02_lactate_readmission", {
        "interpretation_class": "adjusted_association",
        "exposure": "lactate_max",
        "outcome": "icu_readmission",
        "effect_scale": "odds_ratio",
        "primary_estimate": 1.04,
        "primary_estimate_interval": [0.82, 1.31],
        "analysis_set": "complete_case",
        "analysis_role": "primary",
        "adjustment_covariates": ["age", "sex"],
    })
    [feasibility] = _claims("08_repeat_lactate_protocol", _outcome(
        rule="planned_analysis_feasibility",
        disposition="not_executable_from_sealed_inputs",
        planned_analysis_role="sensitivity",
    ))
    results = "\n\n".join([
        "### Cohort characteristics",
        "The cohort comprised 1,402 ICU stays {evidence:research_context}.",
        "### Primary association",
        "### Sensitivity and subgroup analyses",
    ])
    text = _manuscript(results, conclusion="")
    assert _findings(text, plan)["MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY"]

    repaired, reader, _ = _through_the_host(text, [primary, feasibility], plan)

    assert feasibility.placeholder in _section(
        repaired, "### Sensitivity and subgroup analyses"
    )
    assert primary.placeholder in _section(repaired, "### Primary association")
    # A null primary association remains the conclusion; the protocol does not.
    assert _section(repaired, "## Conclusion").strip() == primary.placeholder
    assert "showed no clear association with" in _section(reader, "## Conclusion")
    assert _findings(reader, plan) == {}
