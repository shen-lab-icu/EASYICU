"""A single-panel *step-level* result figure must not fail-close a step when the
study-design family builds its PRIMARY publication figure deterministically in
the write phase (M3 subphenotype).

Deadlock this guards against: phenotyping (and any family in
``figures.FAMILY_RENDERERS``) has a deterministic multi-panel renderer, but it
only runs in the write phase, which is gated behind ``execution_complete``. When
the LLM's step figure is single-panel, the ``figure_contract_quality`` panel-
count ERROR marks the step ``contract_failed`` -> ``execution_complete`` stays
False -> the write phase (and thus the renderer that would produce the >=2-panel
primary) is skipped. Demoting that one shape error to a warning for renderer-
backed families breaks the deadlock; the write-phase display-suite gate stays
the fail-closed backstop. Association/descriptive (no deterministic renderer)
still block, and non-panel-count figure errors still block everywhere.
"""

from __future__ import annotations

from easyicu.research_agent.pipeline_execute import (
    _demote_result_figure_shape_for_family_renderer,
    _family_has_deterministic_figure_renderer,
    _is_too_few_panels_figure_finding,
)
from easyicu.research_agent.schema import ResearchContext, ValidationFinding


def _context(question: str) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort={
            "cohort_name": "c",
            "database": "miiv",
            "n_patients": 10,
            "n_stays": 10,
        },
        variables=[],
    )


_PHENOTYPING = "Identify sepsis subphenotypes by unsupervised clustering."
_ASSOCIATION = "Association between serum lactate and in-hospital mortality in sepsis."


def _panel_count_error(n: int = 1) -> ValidationFinding:
    return ValidationFinding(
        validator="figure_contract_quality",
        severity="error",
        message=f"cluster_profile_heatmap has only {n} panel(s); manuscript-facing "
        "result figures need at least two data-backed panels.",
        detail={"path": "x", "panel_count": n, "step_id": "01_x"},
    )


def _blank_title_error() -> ValidationFinding:
    return ValidationFinding(
        validator="figure_contract_quality",
        severity="error",
        message="fig has panel(s) without titles: A",
        detail={"path": "x", "panel_ids": ["A"]},
    )


# --- predicate --------------------------------------------------------------


def test_predicate_matches_only_the_panel_count_error():
    assert _is_too_few_panels_figure_finding(_panel_count_error(1)) is True
    # >=2 panels is not the too-few-panels error
    assert _is_too_few_panels_figure_finding(_panel_count_error(2)) is False
    # a different figure_contract_quality error (blank titles) is not matched
    assert _is_too_few_panels_figure_finding(_blank_title_error()) is False
    # warning severity is not matched (only the blocking error is)
    warn = _panel_count_error(1).model_copy(update={"severity": "warning"})
    assert _is_too_few_panels_figure_finding(warn) is False
    # a panel_count detail on a DIFFERENT validator is not matched
    other = _panel_count_error(1).model_copy(update={"validator": "figure_source_data"})
    assert _is_too_few_panels_figure_finding(other) is False


# --- family gate ------------------------------------------------------------


def test_family_gate_true_for_renderer_backed_false_for_association():
    assert _family_has_deterministic_figure_renderer(_context(_PHENOTYPING)) is True
    assert _family_has_deterministic_figure_renderer(_context(_ASSOCIATION)) is False


# --- demotion ---------------------------------------------------------------


def test_phenotyping_demotes_panel_count_error_to_warning():
    other_error = ValidationFinding(
        validator="exposure_integrity",
        severity="error",
        message="exposure leaked into features",
    )
    findings = [_panel_count_error(1), other_error]
    out = _demote_result_figure_shape_for_family_renderer(
        _context(_PHENOTYPING), findings
    )
    panel = next(f for f in out if f.validator == "figure_contract_quality")
    assert panel.severity == "warning"
    assert "advisory" in panel.message
    # an unrelated integrity error is left untouched -- it still blocks
    integ = next(f for f in out if f.validator == "exposure_integrity")
    assert integ.severity == "error"


def test_association_family_leaves_panel_count_error_blocking():
    findings = [_panel_count_error(1)]
    out = _demote_result_figure_shape_for_family_renderer(
        _context(_ASSOCIATION), findings
    )
    assert out[0].severity == "error"


def test_non_panel_count_error_is_untouched_even_for_phenotyping():
    # A different figure_contract_quality error (blank titles) still blocks; only
    # the panel-count shape rule is advisory for renderer-backed families.
    findings = [_blank_title_error()]
    out = _demote_result_figure_shape_for_family_renderer(
        _context(_PHENOTYPING), findings
    )
    assert out[0].severity == "error"


def test_passthrough_when_no_panel_count_finding():
    findings = [_blank_title_error()]
    out = _demote_result_figure_shape_for_family_renderer(
        _context(_PHENOTYPING), findings
    )
    assert [f.message for f in out] == [f.message for f in findings]
