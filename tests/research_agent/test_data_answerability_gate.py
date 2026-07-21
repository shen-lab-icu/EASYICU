from __future__ import annotations

from easyicu.research_agent.gates.data_answerability import (
    primary_exposure_answerability_findings,
)
from easyicu.research_agent.literature import (
    HypothesisBlueprintAgent,
    LiteratureBundle,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    MissingnessProfile,
    ResearchContext,
)


def _context(
    *,
    domain: dict[str, object],
    missing_n: int,
    missingness_semantics: str | None = None,
) -> ResearchContext:
    return ResearchContext(
        research_question="Estimate the association of exposure with outcome.",
        cohort=CohortDescriptor(
            cohort_name="answerability",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[
            ConceptDescriptor(
                name="exposure_x",
                role="intervention",
                dtype="float64",
                observed_domain=domain,
                missingness_semantics=missingness_semantics,
                missingness=MissingnessProfile(
                    fraction_missing=missing_n / 100,
                    n_missing=missing_n,
                    n_total=100,
                ),
            ),
            ConceptDescriptor(name="outcome_y", role="outcome", dtype="int64"),
        ],
        primary_exposure="exposure_x",
        target_outcome="outcome_y",
    )


def test_single_observed_level_with_unknown_missing_semantics_blocks_before_planner():
    context = _context(
        domain={
            "n_unique": 1,
            "is_constant": True,
            "is_binary": True,
            "min": 1.0,
            "max": 1.0,
        },
        missing_n=84,
    )

    findings = primary_exposure_answerability_findings(context)

    assert len(findings) == 1
    assert (
        findings[0].detail["kind"] == "scientifically_infeasible_requires_data_contract"
    )
    assert findings[0].detail["required_action"] == (
        "supply_host_owned_source_absence_contract"
    )

    blueprint = HypothesisBlueprintAgent().run(
        context=context,
        literature=LiteratureBundle(
            research_question=context.research_question, citations=[]
        ),
    )
    assert blueprint.feasibility_status == "blocked"
    assert any(
        "only one observed level" in note for note in blueprint.domain_gate_notes
    )


def test_two_observed_levels_remain_answerable():
    context = _context(
        domain={
            "n_unique": 2,
            "is_constant": False,
            "is_binary": True,
            "min": 0.0,
            "max": 1.0,
        },
        missing_n=10,
    )

    assert primary_exposure_answerability_findings(context) == []


def test_explicit_missingness_semantics_are_left_for_source_status_gate():
    context = _context(
        domain={
            "n_unique": 1,
            "is_constant": True,
            "is_binary": True,
            "min": 1.0,
            "max": 1.0,
        },
        missing_n=84,
        missingness_semantics=(
            "Missing means verified event absence under complete source coverage."
        ),
    )

    assert primary_exposure_answerability_findings(context) == []


def test_complete_constant_exposure_is_scientifically_infeasible():
    context = _context(
        domain={"n_unique": 1, "is_constant": True, "min": 0.0, "max": 0.0},
        missing_n=0,
    )

    findings = primary_exposure_answerability_findings(context)

    assert len(findings) == 1
    assert (
        findings[0].detail["kind"] == "scientifically_infeasible_no_exposure_contrast"
    )
