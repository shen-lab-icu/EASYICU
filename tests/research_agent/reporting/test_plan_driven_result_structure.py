"""Scientific families share a renderer, not a mandatory association template."""

import pytest

from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality, repair_registered_display_callouts,
)
from easyicu.research_agent.reporting.manuscript_sections import (
    render_manuscript_sections, repair_existing_manuscript_sections,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

from .test_manuscript_sections import _minimal_valid_section


def _plan(family="descriptive_epidemiology", roles=("primary",)):
    return AnalysisPlan(
        research_question="Describe recorded outcomes in the input cohort.",
        analysis_type=family,
        steps=[AnalysisStep(
            step_id=f"step_{index}", planned_analysis_role=role,
            intent="Summarize the prespecified endpoint.", method="descriptive",
            inputs=["artifact:analysis_cohort"], expected_outputs=[f"table:result_{index}"],
        ) for index, role in enumerate(roles)],
    )


def _results(heading, *, extra=""):
    return (
        "## Results\n\n### Cohort characteristics\n"
        "The cohort comprised 120 ICU stays {evidence:cohort}.\n\n"
        f"### {heading}\n"
        "Observed mortality was 12 of 120 observations {evidence:result}.\n"
        + extra
    )


@pytest.mark.parametrize(("family", "heading"), [
    ("descriptive_epidemiology", "Descriptive results"),
    ("prediction_model", "Model performance"),
    ("trajectory_clustering", "Cluster characteristics"),
    ("survival", "Survival results"),
])
def test_fresh_writer_and_final_audit_share_the_plan_structure(family, heading):
    plan = _plan(family)
    calls = []

    def call_section(**kwargs):
        calls.append(kwargs["section_name"])
        if kwargs["section_name"] == "Results":
            assert f"### {heading}" in kwargs["instruction"]
            assert "### Primary association" not in kwargs["instruction"]
            return _results(heading)
        return _minimal_valid_section(kwargs["section_name"])

    text = render_manuscript_sections(
        call_section=call_section, common={"analysis_plan": plan},
    )

    assert calls.count("Results") == 1
    assert "Sensitivity and subgroup" not in text
    assert not [f for f in audit_manuscript_quality(text, analysis_plan=plan).findings
                if f.section == "Results"]


def test_planned_secondary_and_sensitivity_results_cannot_disappear():
    plan = _plan(roles=("primary", "secondary", "sensitivity"))
    audit = audit_manuscript_quality(_results("Descriptive results"), analysis_plan=plan)
    missing = [f.excerpts for f in audit.findings if f.section == "Results"]
    assert ("Secondary analyses",) in missing
    assert ("Sensitivity and subgroup analyses",) in missing


def test_a_figure_callout_is_not_a_primary_result():
    plan = _plan()
    text = _results("Descriptive results").replace(
        "Observed mortality was 12 of 120 observations {evidence:result}.",
        "See Figure 1 {evidence:publication_figure_contract}.",
    )
    audit = audit_manuscript_quality(text, analysis_plan=plan)
    assert "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY" in {f.code for f in audit.findings}


def test_standalone_claim_tokens_are_not_mistaken_for_empty_figure_callouts():
    text = _results("Descriptive results").replace(
        "Observed mortality was 12 of 120 observations {evidence:result}.",
        "{claim:step_0.observed_risk}\n\nSee Figure 1 {evidence:publication_figure_contract}.",
    )
    audit = audit_manuscript_quality(text, analysis_plan=_plan())
    assert not [f for f in audit.findings if f.section == "Results"]


def test_registered_figure_callout_uses_the_actual_primary_heading():
    text = _results("Descriptive results")
    repaired, changes = repair_registered_display_callouts(
        text, expected_display_labels=("Figure 1",),
    )
    assert len(changes) == 1
    assert "See Figure 1" in repaired
    assert "Primary association" not in repaired
    assert repair_registered_display_callouts(
        repaired, expected_display_labels=("Figure 1",),
    ) == (repaired, ())


def test_saved_report_migration_uses_the_same_plan_without_redoing_other_sections():
    names = ("Title and Keywords", "Abstract", "Introduction", "Methods", "Results",
             "Discussion", "Limitations", "Conclusion")
    old = "\n\n".join(_minimal_valid_section(name) for name in names)
    calls = []

    def call_section(**kwargs):
        calls.append(kwargs["section_name"])
        assert "### Descriptive results" in kwargs["instruction"]
        return _results("Descriptive results")

    repaired, keys = repair_existing_manuscript_sections(
        old, call_section=call_section, common={"analysis_plan": _plan()},
    )
    assert calls == ["Results"] and keys == ("results",)
    assert "Primary association" not in repaired
