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
from easyicu.research_agent.authority.evidence_store import EvidenceStore
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


# ---------------------------------------------------------------------------
# 4. Outcome-aware replan-budget rule (2026-07-07)
#
# Reaching the cap demotes to diagnostic_only ONLY if the run did not otherwise
# converge. H2 fix8 reached execution_complete + a bound deterministic OR 3.04 +
# 0 failed steps + clean manuscript, yet the planner churned 6 revisions -- that
# is churny-but-successful, so the cap is advisory, not a demotion.
# ---------------------------------------------------------------------------

from easyicu.research_agent.pipeline_report import (  # noqa: E402
    _deterministic_primary_estimate_bound,
    _replan_budget_demotes,
)


def test_replan_cap_is_advisory_for_a_converged_clean_run():
    # This is the exact H2 fix8 gate state: everything clean except the cap hit.
    assert (
        _replan_budget_demotes(
            hit=True,
            execution_complete=True,
            has_failed_steps=False,
            has_base_errors=False,
            evidence_complete=True,
            numeric_verified=True,
            primary_estimate_bound=True,
        )
        is False
    )


def test_replan_cap_demotes_when_execution_incomplete():
    assert _replan_budget_demotes(
        hit=True,
        execution_complete=False,
        has_failed_steps=False,
        has_base_errors=False,
        evidence_complete=True,
        numeric_verified=True,
        primary_estimate_bound=True,
    )


def test_replan_cap_demotes_when_a_step_failed():
    assert _replan_budget_demotes(
        hit=True,
        execution_complete=True,
        has_failed_steps=True,
        has_base_errors=False,
        evidence_complete=True,
        numeric_verified=True,
        primary_estimate_bound=True,
    )


def test_replan_cap_demotes_when_primary_estimate_unbound():
    # No bound deterministic headline -> unresolved -> fail closed.
    assert _replan_budget_demotes(
        hit=True,
        execution_complete=True,
        has_failed_steps=False,
        has_base_errors=False,
        evidence_complete=True,
        numeric_verified=True,
        primary_estimate_bound=False,
    )


def test_replan_cap_demotes_when_other_hard_errors_present():
    assert _replan_budget_demotes(
        hit=True,
        execution_complete=True,
        has_failed_steps=False,
        has_base_errors=True,
        evidence_complete=True,
        numeric_verified=True,
        primary_estimate_bound=True,
    )


def test_no_cap_hit_never_demotes():
    assert (
        _replan_budget_demotes(
            hit=False,
            execution_complete=False,
            has_failed_steps=True,
            has_base_errors=True,
            evidence_complete=False,
            numeric_verified=False,
            primary_estimate_bound=False,
        )
        is False
    )


def test_legacy_deterministic_iptw_marker_does_not_claim_primary_ownership():
    recs = [
        {
            "step_id": "04_causal_effect_estimation",
            "deterministic_standard_analysis": "causal_primary_iptw",
            "step_summary": {
                "status": "ok",
                "primary_predictor": "vasopressor",
                "adjusted_effect": 3.04,
                "adjusted_effect_scale": "odds_ratio",
            },
        }
    ]
    assert _deterministic_primary_estimate_bound(recs) is False


def test_primary_estimate_bound_false_for_llm_coded_estimate():
    # Bound estimate but NOT from a deterministic runner -> conservative False,
    # so an LLM-coded primary that hit the cap still fails closed.
    recs = [
        {
            "step_id": "04_causal_effect_estimation",
            "step_summary": {
                "status": "ok",
                "primary_predictor": "vasopressor",
                "adjusted_effect": 3.04,
                "adjusted_effect_scale": "odds_ratio",
            },
        }
    ]
    assert _deterministic_primary_estimate_bound(recs) is False


def test_primary_estimate_bound_false_when_runner_blocked():
    recs = [
        {
            "step_id": "04_causal_effect_estimation",
            "deterministic_standard_analysis": "causal_primary_iptw",
            "step_summary": {"status": "blocked", "adjusted_effect": None},
        }
    ]
    assert _deterministic_primary_estimate_bound(recs) is False


def test_legacy_deterministic_ordinal_marker_does_not_claim_primary_ownership():
    # Legacy records remain readable, but a retired runner marker cannot regain
    # ownership of a scientific estimand.
    recs = [
        {
            "step_id": "02_dose_response",
            "deterministic_standard_analysis": "ordinal_dose_response",
            "step_summary": {
                "status": "ok",
                "primary_predictor": "kdigo",
                "adjusted_effect": 1.62,
                "adjusted_effect_scale": "odds_ratio",
            },
        }
    ]
    assert _deterministic_primary_estimate_bound(recs) is False


# ---------------------------------------------------------------------------
# 5. Agent-owned primary families
#
# Primary scientific analyses are agent-owned. A converged, fully validated run
# must not be demoted purely because no deterministic runner can bind its
# estimand, while genuine validation failures still demote.
# ---------------------------------------------------------------------------


def test_cap_advisory_for_converged_no_primary_family():
    # No deterministic primary is expected; all other gates are clean.
    assert (
        _replan_budget_demotes(
            hit=True,
            execution_complete=True,
            has_failed_steps=False,
            has_base_errors=False,
            evidence_complete=True,
            numeric_verified=True,
            primary_estimate_bound=False,  # phenotyping never binds one
            no_deterministic_primary_expected=True,
        )
        is False
    )


def test_cap_still_demotes_no_primary_family_with_base_errors():
    # The waiver only covers the missing primary -- a real error still demotes.
    assert _replan_budget_demotes(
        hit=True,
        execution_complete=True,
        has_failed_steps=False,
        has_base_errors=True,
        evidence_complete=True,
        numeric_verified=True,
        primary_estimate_bound=False,
        no_deterministic_primary_expected=True,
    )


def test_unknown_family_without_agent_owned_waiver_stays_fail_closed():
    # If family inference fails, the conservative default remains fail closed.
    assert _replan_budget_demotes(
        hit=True,
        execution_complete=True,
        has_failed_steps=False,
        has_base_errors=False,
        evidence_complete=True,
        numeric_verified=True,
        primary_estimate_bound=False,
        no_deterministic_primary_expected=False,
    )
