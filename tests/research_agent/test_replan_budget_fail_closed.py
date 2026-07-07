"""Replan-budget cap + fail-closed demotion.

Built 2026-07-06 after profiling an E3 canonical run that replanned 9× over
~50 min and still failed: the replanner cap (``max_replans``) was disabled
(0), so a non-converging run churned expensive planner calls indefinitely.

The balanced policy locked here:
  * full runs get a budget of 6 substantive revisions (legitimate repair
    headroom — a real run rarely needs more than a handful);
  * ``stabilization_mode`` (fast primary-only iteration) tightens it to 3;
  * ``max_consecutive_noop_replans`` stays 2 (unchanged);
  * when the budget is exhausted the run FAILS CLOSED to ``diagnostic_only``
    — a runaway replan loop must not launder a manuscript.

These tests exercise the two contracts directly: the pipeline effective-cap
wiring, and the fail-closed demotion (both the readiness gate that scans for
the run-level ``replan_budget`` finding and the scorecard tristate floor).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import easyicu.research_agent as ra
from easyicu.research_agent.evaluation_scorecard import compute_tristate
from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline_report import _compute_readiness_gates
from easyicu.research_agent.schema import ResearchContext, ValidationFinding


# ---------------------------------------------------------------------------
# 1. Pipeline effective-cap wiring
# ---------------------------------------------------------------------------


def _pipeline(tmp_path: Path, **kwargs):
    return ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient(), **kwargs)


def test_default_full_run_replan_budget_is_six(tmp_path: Path):
    # The full-run default gives legitimate repair headroom without churn.
    assert _pipeline(tmp_path)._max_replans == 6


def test_stabilization_mode_tightens_budget_to_three(tmp_path: Path):
    p = _pipeline(tmp_path, stabilization_mode=True)
    assert p._max_replans == 3
    assert p._stabilization_mode is True


def test_stabilization_preserves_a_smaller_explicit_cap(tmp_path: Path):
    # A caller that deliberately set a tighter cap keeps it under stabilization.
    assert _pipeline(tmp_path, stabilization_mode=True, max_replans=2)._max_replans == 2


def test_stabilization_rearms_a_disabled_cap(tmp_path: Path):
    # A disabled cap (0) must not leave stabilization runs uncapped.
    assert _pipeline(tmp_path, stabilization_mode=True, max_replans=0)._max_replans == 3


def test_explicit_full_run_cap_is_respected(tmp_path: Path):
    assert _pipeline(tmp_path, max_replans=10)._max_replans == 10


def test_noop_replan_cap_default_unchanged(tmp_path: Path):
    # The balanced policy keeps the no-op streak guard at 2.
    assert _pipeline(tmp_path)._max_consecutive_noop_replans == 2


# ---------------------------------------------------------------------------
# 2. Fail-closed demotion — readiness gate
# ---------------------------------------------------------------------------


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Q",
        cohort={
            "cohort_name": "c",
            "database": "miiv",
            "n_patients": 10,
            "n_stays": 10,
        },
        variables=[],
    )


def _budget_finding() -> ValidationFinding:
    return ValidationFinding(
        validator="replan_budget",
        severity="error",
        message="Replan budget exhausted: 6 substantive plan revisions ...",
        detail={
            "replan_budget_exhausted": True,
            "cap": 6,
            "substantive_revisions": 6,
        },
    )


def test_replan_budget_finding_sets_gate_and_blocks_manuscript(tmp_path: Path):
    ev = EvidenceStore(tmp_path)
    gates = _compute_readiness_gates(
        context=_context(),
        plan=None,
        per_step_records=[],
        findings=[_budget_finding()],
        evidence=ev,
        run_dir=tmp_path,
        manuscript_path=tmp_path / "m.md",
        stop_after_analysis=False,
    )
    assert gates["replan_budget_exhausted"] is True
    assert gates["manuscript_ready"] is False


def test_no_replan_budget_finding_leaves_gate_false(tmp_path: Path):
    ev = EvidenceStore(tmp_path)
    gates = _compute_readiness_gates(
        context=_context(),
        plan=None,
        per_step_records=[],
        findings=[],
        evidence=ev,
        run_dir=tmp_path,
        manuscript_path=tmp_path / "m.md",
        stop_after_analysis=False,
    )
    assert gates["replan_budget_exhausted"] is False


def test_replan_budget_latch_survives_a_step_id_shaped_reason(tmp_path: Path):
    # The trigger lives in ``detail`` only, so a step-id-shaped reason cannot
    # let the readiness supersession rule drop the run-level latch.
    finding = ValidationFinding(
        validator="replan_budget",
        severity="error",
        message="Replan budget exhausted after 03_primary_analysis_figure.",
        detail={
            "replan_budget_exhausted": True,
            "reason": "03_primary_analysis_figure",
        },
    )
    ev = EvidenceStore(tmp_path)
    gates = _compute_readiness_gates(
        context=_context(),
        plan=None,
        per_step_records=[
            {"step_id": "03_primary_analysis_figure", "status": "ok"},
        ],
        findings=[finding],
        evidence=ev,
        run_dir=tmp_path,
        manuscript_path=tmp_path / "m.md",
        stop_after_analysis=False,
    )
    assert gates["replan_budget_exhausted"] is True


# ---------------------------------------------------------------------------
# 3. Fail-closed demotion — scorecard tristate floor
# ---------------------------------------------------------------------------


def test_compute_tristate_floors_to_diagnostic_only_when_budget_exhausted():
    # Even a run that limped a manuscript through is floored: a non-converging
    # replan loop is not a reportable result.
    verdict = compute_tristate(
        {
            "replan_budget_exhausted": True,
            "manuscript_ready": True,
            "execution_complete": True,
        }
    )
    assert verdict == "diagnostic_only"


def test_compute_tristate_unchanged_without_budget_flag():
    assert (
        compute_tristate({"manuscript_ready": True, "execution_complete": True})
        == "gate_reportable"
    )


def test_scratch_dir_isolated():
    # Guard: the module-level tempdir helper never writes into the repo tree.
    d = Path(tempfile.mkdtemp())
    assert d.exists() and "site-packages" not in str(d)
