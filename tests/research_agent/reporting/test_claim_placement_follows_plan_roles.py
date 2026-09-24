"""A host claim reaches the Results subsection its reviewed plan role requires.

The plan-derived structure makes a Secondary or Sensitivity subsection
required when a step carries that role, and the strict Results grammar admits
only host claim tokens there.  Placement must therefore follow the same plan
roles, whatever role label a claim adapter fixed for its result kind.
"""

from __future__ import annotations

import ast
from pathlib import Path

from easyicu.research_agent.authority.absolute_risk_scientific_claims import (
    derive_absolute_risk_claim_payloads,
)
from easyicu.research_agent.authority.manuscript_claim_policy import (
    place_scientific_claim_tokens_in_results,
)
from easyicu.research_agent.authority.scientific_claims import (
    ScientificClaim,
    ScientificClaimDraft,
)
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.manuscript_quality import audit_manuscript_quality
from easyicu.research_agent.reporting.manuscript_result_structure import (
    planned_result_roles,
    required_result_subsections,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan(roles: dict[str, str]) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Is the first-day lactate maximum associated with ICU readmission?",
        analysis_type="association_study",
        steps=[
            AnalysisStep(
                step_id=step_id,
                planned_analysis_role=role,
                intent="Report the prespecified result.",
                method="descriptive",
                inputs=["artifact:analysis_cohort"],
                expected_outputs=[f"table:{step_id}"],
            )
            for step_id, role in roles.items()
        ],
    )


def _results(*, cohort: str = "", primary: str = "", secondary: str = "",
             sensitivity: str = "", secondary_heading: str = "Secondary analyses") -> str:
    parts = ["## Results", "### Cohort characteristics", cohort,
             "### Primary association", primary]
    if secondary_heading:
        parts += [f"### {secondary_heading}", secondary]
    parts += ["### Sensitivity and subgroup analyses", sensitivity, "## Discussion",
              "Interpretation stays in the Discussion."]
    return "\n\n".join(parts) + "\n"


def _subsection(text: str, heading: str) -> str:
    body = text.split(f"### {heading}\n", 1)[1]
    return body.split("\n#", 1)[0]


def _claim(**overrides) -> ScientificClaim:
    payload = {
        "claim_id": "adjusted_association",
        "claim_type": "association",
        "exposure": "vasopressor any",
        "outcome": "mort 28d",
        "direction": "no_clear_association",
        "estimand": "adjusted odds ratio",
        "population": "the complete case analysis set",
        "analysis_role": "primary",
        "status": "supported",
        "adjusted_for": ["age", "sex"],
        "step_id": "vasopressor_mortality",
        "evidence_id": "vasopressor_mortality_summary",
    }
    payload.update(overrides)
    return ScientificClaim(**payload)


def _absolute_risk_claim(step_id: str = "readmission_risk_context") -> ScientificClaim:
    """The real adapter output, which fixes ``analysis_role="auxiliary"``."""

    summary = {
        "analysis_family": "absolute_risk_context",
        "interpretation_class": "absolute_risk_context",
        "status": "ok",
        "adjusted_effect": None,
        "outcome": "icu_readmission",
        "n_total": 412,
        "outcome_missing_n": 0,
        "outcome_nonmissing_n": 412,
        "reportable_descriptive_results": {
            "schema_version": "easyicu.absolute_risk_reporting/1",
            "execution_owner": "absolute_risk_context_executor_v1",
            "interpretation_ceiling": "descriptive_not_causal",
            "overall_outcome": {
                "outcome": "icu_readmission",
                "n": 412,
                "event_n": 37,
                "risk_pct": 100.0 * 37 / 412,
            },
            "exposures": [],
        },
    }
    [payload] = derive_absolute_risk_claim_payloads(summary)
    draft = ScientificClaimDraft.model_validate(payload)
    assert draft.analysis_role == "auxiliary"
    return ScientificClaim(
        **draft.model_dump(), step_id=step_id, evidence_id=f"{step_id}_summary",
    )


def test_planned_result_roles_keep_only_roles_that_own_a_subsection() -> None:
    plan = _plan({
        "cohort": "auxiliary",
        "lactate_readmission": "primary",
        "readmission_risk_context": "secondary",
        "lactate_spline": "sensitivity",
    })

    assert planned_result_roles(plan) == {
        "readmission_risk_context": "secondary",
        "lactate_spline": "sensitivity",
    }
    assert planned_result_roles(None) == {}


def test_a_fixed_auxiliary_claim_reaches_its_planned_secondary_subsection() -> None:
    claim = _absolute_risk_claim()
    plan = _plan({"lactate_readmission": "primary", "readmission_risk_context": "secondary"})

    placement = place_scientific_claim_tokens_in_results(
        _results(), claims=[claim], planned_step_roles=planned_result_roles(plan),
    )

    assert claim.placeholder in _subsection(placement.scaffold, "Secondary analyses")
    assert claim.placeholder not in _subsection(placement.scaffold, "Cohort characteristics")
    assert placement.inserted_claim_refs == (claim.claim_ref,)
    assert placement.role_subsection_claim_refs == ()


def test_a_secondary_association_claim_does_not_fill_the_primary_subsection() -> None:
    claim = _claim(analysis_role="secondary")

    placement = place_scientific_claim_tokens_in_results(_results(), claims=[claim])

    assert claim.placeholder in _subsection(placement.scaffold, "Secondary analyses")
    assert claim.placeholder not in _subsection(placement.scaffold, "Primary association")


def test_the_step_role_decides_when_the_claim_role_owns_no_subsection() -> None:
    claim = _claim(analysis_role="primary", step_id="vasopressor_secondary_model")

    placement = place_scientific_claim_tokens_in_results(
        _results(),
        claims=[claim],
        planned_step_roles={"vasopressor_secondary_model": "secondary"},
    )

    assert claim.placeholder in _subsection(placement.scaffold, "Secondary analyses")


def test_a_claim_owned_sensitivity_role_wins_over_its_step_role() -> None:
    claim = _claim(analysis_role="sensitivity", step_id="vasopressor_models")

    placement = place_scientific_claim_tokens_in_results(
        _results(),
        claims=[claim],
        planned_step_roles={"vasopressor_models": "secondary"},
    )

    assert claim.placeholder in _subsection(
        placement.scaffold, "Sensitivity and subgroup analyses",
    )
    assert claim.placeholder not in _subsection(placement.scaffold, "Secondary analyses")


def test_a_claim_reported_only_elsewhere_is_repeated_in_its_role_subsection() -> None:
    claim = _absolute_risk_claim()
    scaffold = _results(cohort=claim.placeholder)

    placement = place_scientific_claim_tokens_in_results(
        scaffold,
        claims=[claim],
        planned_step_roles={"readmission_risk_context": "secondary"},
    )

    # The Writer's placement stays; removing it could empty that subsection.
    assert claim.placeholder in _subsection(placement.scaffold, "Cohort characteristics")
    assert claim.placeholder in _subsection(placement.scaffold, "Secondary analyses")
    assert placement.inserted_claim_refs == ()
    assert placement.role_subsection_claim_refs == (claim.claim_ref,)


def test_a_claim_already_in_its_role_subsection_is_left_unchanged() -> None:
    claim = _absolute_risk_claim()
    scaffold = _results(cohort=claim.placeholder, secondary=claim.placeholder)

    placement = place_scientific_claim_tokens_in_results(
        scaffold,
        claims=[claim],
        planned_step_roles={"readmission_risk_context": "secondary"},
    )

    assert placement.scaffold == scaffold
    assert placement.inserted_claim_refs == ()
    assert placement.role_subsection_claim_refs == ()


def test_without_plan_roles_the_claim_type_placement_is_unchanged() -> None:
    claim = _absolute_risk_claim()

    placement = place_scientific_claim_tokens_in_results(_results(), claims=[claim])

    assert claim.placeholder in _subsection(placement.scaffold, "Cohort characteristics")
    assert claim.placeholder not in _subsection(placement.scaffold, "Secondary analyses")


def test_a_missing_role_heading_falls_back_to_the_claim_type_placement() -> None:
    descriptive = _absolute_risk_claim()
    association = _claim(analysis_role="secondary")

    placement = place_scientific_claim_tokens_in_results(
        _results(secondary_heading=""),
        claims=[descriptive, association],
        planned_step_roles={"readmission_risk_context": "secondary"},
    )

    assert descriptive.placeholder in _subsection(placement.scaffold, "Cohort characteristics")
    assert association.placeholder in _subsection(placement.scaffold, "Primary association")


def test_the_plan_heading_wins_over_an_earlier_heading_that_mentions_the_role() -> None:
    claim = _claim(analysis_role="sensitivity")
    scaffold = _results().replace(
        "### Primary association", "### Primary association and robustness",
    )

    placement = place_scientific_claim_tokens_in_results(scaffold, claims=[claim])

    assert claim.placeholder in _subsection(
        placement.scaffold, "Sensitivity and subgroup analyses",
    )
    assert claim.placeholder not in _subsection(
        placement.scaffold, "Primary association and robustness",
    )


def test_a_writer_secondary_heading_variant_still_owns_secondary_claims() -> None:
    claim = _absolute_risk_claim()

    placement = place_scientific_claim_tokens_in_results(
        _results(secondary_heading="Secondary outcomes"),
        claims=[claim],
        planned_step_roles={"readmission_risk_context": "secondary"},
    )

    assert claim.placeholder in _subsection(placement.scaffold, "Secondary outcomes")


def test_the_placed_claim_satisfies_the_plan_required_secondary_subsection() -> None:
    plan = _plan({
        "lactate_readmission": "primary",
        "readmission_risk_context": "secondary",
        "lactate_spline": "sensitivity",
    })
    primary = _claim(
        exposure="lactate max", outcome="icu readmission",
        step_id="lactate_readmission", evidence_id="lactate_readmission_summary",
    )
    sensitivity = _claim(
        claim_id="sensitivity_spline", analysis_role="sensitivity",
        exposure="lactate max", outcome="icu readmission",
        step_id="lactate_spline", evidence_id="lactate_spline_summary",
    )
    secondary = _absolute_risk_claim()
    scaffold = _results(
        cohort="The source cohort included 412 ICU stays {evidence:table_one}.",
        primary=primary.placeholder,
        sensitivity=sensitivity.placeholder,
    )
    assert required_result_subsections(plan) == (
        "Cohort characteristics", "Primary association",
        "Secondary analyses", "Sensitivity and subgroup analyses",
    )

    def empty_result_subsections(text: str) -> list[tuple[str, ...]]:
        return [
            finding.excerpts
            for finding in audit_manuscript_quality(text, analysis_plan=plan).findings
            if finding.code == "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY"
            and finding.section == "Results"
        ]

    legacy = place_scientific_claim_tokens_in_results(
        scaffold, claims=[primary, secondary, sensitivity],
    )
    assert empty_result_subsections(legacy.scaffold) == [("Secondary analyses",)]

    placement = place_scientific_claim_tokens_in_results(
        scaffold,
        claims=[primary, secondary, sensitivity],
        planned_step_roles=planned_result_roles(plan),
    )
    assert empty_result_subsections(placement.scaffold) == []
    assert placement.inserted_claim_refs == (secondary.claim_ref,)


def test_the_writer_phase_passes_the_reviewed_plan_roles_to_claim_placement() -> None:
    tree = ast.parse(Path(write_phase.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "place_scientific_claim_tokens_in_results"
    ]

    assert len(calls) == 1
    roles = [kw.value for kw in calls[0].keywords if kw.arg == "planned_step_roles"]
    assert len(roles) == 1
    assert isinstance(roles[0], ast.Call)
    assert isinstance(roles[0].func, ast.Name)
    assert roles[0].func.id == "planned_result_roles"
