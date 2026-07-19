"""A deterministic SVG text-overlap 'spacing' warning is cosmetic and must not
block manuscript_ready (M3 subphenotype).

The step-level demotion (pipeline_execute._demote_cosmetic_visual_findings) runs
during execution, but the SAME finding is regenerated when the FINAL manuscript
SVG is audited -- after that pass -- so it leaked into analysis_errors and the
figure-bundle gate, blocking a run whose analysis + evidence were sound. Genuine
visual_qa errors (blank/absent figure) must still block.
"""

from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.pipeline_report import (
    _compute_readiness_gates,
    _is_cosmetic_visual_error,
)
from easyicu.research_agent.schema import ResearchContext, ValidationFinding


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Identify sepsis subphenotypes by unsupervised clustering.",
        cohort={
            "cohort_name": "c",
            "database": "miiv",
            "n_patients": 10,
            "n_stays": 10,
        },
        variables=[],
    )


def _overlap_finding() -> ValidationFinding:
    return ValidationFinding(
        validator="visual_qa",
        severity="error",
        message=(
            "SVG figure 'easyicu_phenotype_publication_figure.svg' has overlapping "
            "text elements; multi-panel labels, annotations or axis text need more "
            "spacing."
        ),
    )


def _blank_figure_finding() -> ValidationFinding:
    return ValidationFinding(
        validator="visual_qa",
        severity="error",
        message="SVG figure 'x.svg' contains no data-backed marks (blank figure).",
    )


def test_predicate_true_for_overlap_false_for_blank():
    assert _is_cosmetic_visual_error(_overlap_finding()) is True
    assert _is_cosmetic_visual_error(_blank_figure_finding()) is False
    # a warning (not error) is not what this predicate is about
    warn = _overlap_finding().model_copy(update={"severity": "warning"})
    assert _is_cosmetic_visual_error(warn) is False


def test_cosmetic_overlap_not_counted_as_analysis_error(tmp_path: Path):
    ev = EvidenceStore(tmp_path)
    gates = _compute_readiness_gates(
        context=_context(),
        plan=None,
        per_step_records=[],
        findings=[_overlap_finding()],
        evidence=ev,
        run_dir=tmp_path,
        manuscript_path=tmp_path / "m.md",
        stop_after_analysis=False,
    )
    assert not any("overlapping text" in m for m in gates["analysis_errors"]), gates[
        "analysis_errors"
    ]


def test_genuine_visual_error_still_blocks(tmp_path: Path):
    ev = EvidenceStore(tmp_path)
    gates = _compute_readiness_gates(
        context=_context(),
        plan=None,
        per_step_records=[],
        findings=[_blank_figure_finding()],
        evidence=ev,
        run_dir=tmp_path,
        manuscript_path=tmp_path / "m.md",
        stop_after_analysis=False,
    )
    assert any("blank figure" in m for m in gates["analysis_errors"]), gates[
        "analysis_errors"
    ]
