from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.figures.publication import (
    make_figure_contract,
    save_publication_figure,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.display_suite import summarize_display_suite_status
from easyicu.research_agent.reporting.figure_claim_boundaries import (
    build_figure_claim_boundary_audit,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ResearchContext,
)


def _plan(*, with_selection: bool) -> AnalysisPlan:
    selection = None
    if with_selection:
        common = {
            "analysis_type": "association_study",
            "time_zero": "Start of the eligible ICU episode.",
            "observation_window": "The prespecified exposure-to-outcome window.",
            "required_variables": ["exposure", "outcome"],
            "assumptions": ["The declared timing and adjustment remain valid."],
            "literature_citation_keys": [],
        }
        selection = {
            "candidates": [
                {
                    **common,
                    "design_id": "adjusted_primary",
                    "estimand": "Adjusted exposure contrast for the outcome.",
                    "primary_method": "Adjusted generalized linear model",
                    "novelty_positioning": "Compare the estimand with prior ICU cohorts.",
                    "figure_role": "Lead with absolute risk and adjusted uncertainty.",
                    "supports": "A prespecified adjusted association estimate.",
                    "cannot_prove": "A causal effect without stronger identification.",
                    "disposition": "selected",
                    "decision_reason": "The exposure-outcome question requires adjustment.",
                },
                {
                    **common,
                    "design_id": "crude_alternative",
                    "estimand": "Unadjusted exposure contrast for the outcome.",
                    "primary_method": "Unadjusted descriptive comparison",
                    "novelty_positioning": "Provide a crude comparison with prior cohorts.",
                    "figure_role": "Show only the crude group contrast.",
                    "supports": "A descriptive difference between exposure groups.",
                    "cannot_prove": "An adjusted or causal exposure effect.",
                    "disposition": "rejected",
                    "decision_reason": "Reject because confounding remains unaddressed.",
                },
            ]
        }
    return AnalysisPlan(
        research_question="Question?",
        analysis_type="association_study",
        design_selection=selection,
        steps=[],
    )


def _write_contract(run_dir: Path) -> None:
    out_dir = run_dir / "publication_figures"
    out_dir.mkdir(parents=True)
    contract = make_figure_contract(
        figure_id="figure:primary",
        core_claim="Absolute risk and the adjusted estimate describe the association.",
        panels=[
            {
                "panel_id": "a",
                "title": "Absolute risk",
                "role": "descriptive_result",
                "claim": "Absolute outcome risks are shown by exposure group.",
                "evidence_ids": ["risk_table"],
            },
            {
                "panel_id": "b",
                "title": "Adjusted estimate",
                "role": "primary_estimand",
                "claim": "The adjusted estimate and interval are shown.",
                "evidence_ids": ["model_table"],
            },
        ],
        source_data=["primary_source_data.csv"],
    )
    save_publication_figure(
        contract,
        out_dir / "primary",
        formats=(),
    )


def test_selected_design_binds_every_figure_and_panel_claim_boundary(
    tmp_path: Path,
) -> None:
    _write_contract(tmp_path)

    audit = build_figure_claim_boundary_audit(
        plan=_plan(with_selection=True),
        run_dir=tmp_path,
    )

    assert audit.status == "complete"
    assert audit.boundary_ready is True
    assert audit.claim_ceiling == "analysis_only"
    assert len(audit.figures) == 1
    figure = audit.figures[0]
    assert figure.boundary_source == "selected_research_design"
    assert figure.supports.startswith("Absolute risk")
    assert figure.cannot_prove.startswith("A causal effect")
    assert [panel.panel_id for panel in figure.panels] == ["a", "b"]
    assert all(panel.cannot_prove == figure.cannot_prove for panel in figure.panels)
    assert audit.plan_sha256 and audit.design_selection_sha256


def test_legacy_figure_gets_explicit_analysis_only_boundary(tmp_path: Path) -> None:
    _write_contract(tmp_path)

    audit = build_figure_claim_boundary_audit(
        plan=_plan(with_selection=False),
        run_dir=tmp_path,
    )

    assert audit.status == "legacy_analysis_only"
    assert audit.boundary_ready is False
    assert audit.figures[0].boundary_source == "legacy_analysis_only"
    assert "cannot authorize a manuscript claim" in audit.figures[0].cannot_prove


def test_selected_design_primary_figure_boundary_fails_display_gate_closed(
    tmp_path: Path,
) -> None:
    _write_contract(tmp_path)
    contract_path = next((tmp_path / "publication_figures").glob("*.figure_contract.json"))
    raw = contract_path.read_text(encoding="utf-8")
    contract_path.write_text(raw.replace("The adjusted estimate and interval are shown.", ""), encoding="utf-8")
    context = ResearchContext(
        research_question="Question?",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )

    status = summarize_display_suite_status(
        context=context,
        plan=_plan(with_selection=True),
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        publication={"publication_figure_bundle_ready": True},
    )

    assert status["display_figure_claim_boundary_status"] == "incomplete"
    assert status["display_figure_claim_boundary_ready"] is False
    assert any(
        "supports/cannot-prove boundaries" in error
        for error in status["display_suite_errors"]
    )
