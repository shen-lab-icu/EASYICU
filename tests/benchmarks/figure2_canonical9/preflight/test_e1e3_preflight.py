"""E1-E3 zero-Provider graph-level offline preflight (Figure 2 batch 1).

Each E-series Canonical9 task is driven through the real
``ResearchAgentPipeline`` graph fully offline (scripted ``MockLLMClient``,
synthetic in-memory cohort, host subprocess runner) and the nine graph-level
dimensions the owner specified are asserted per task:

  1. plan-contract validity            5. per-step timeout wiring
  2. expected graph stages             6. stop / resume (no double-execution)
  3. deterministic vs coder division   7. loop termination (bounded, no hang)
  4. repair/retry cap -> fail-closed   8. final tristate matches expectation
                                       9. no paper authority + provider_calls==0

These are graph-level runs, not pure predicate unit tests.  Run just this batch::

    PYTHONPATH="src:." pytest tests/benchmarks/figure2_canonical9/preflight/ -p no:randomly

The subprocess runner is the documented offline-diagnosis path; the ``auto``
runner's Docker source-SHA integrity gate is a production blocker that is NOT
bypassed here.  Cohorts are tiny; run serially if memory-constrained.
"""

from __future__ import annotations

import pytest

from benchmarks.figure2_canonical9.evaluator.suite import (
    easyicu_evaluation_protocol_suite,
)
from benchmarks.figure2_canonical9.preflight.fixtures import (
    DETERMINISTIC_STEP_ID,
    E1E3_CASES,
    PRIMARY_STEP_ID,
    PreflightCase,
)
from benchmarks.figure2_canonical9.preflight.harness import (
    ScriptedPreflightLLM,
    paper_acceptance_status,
    run_preflight,
)

_CASES = list(E1E3_CASES.values())
_IDS = [c.task_id for c in _CASES]

# A generous ceiling on the number of executed steps; any real graph for these
# minimal plans settles well under this, so exceeding it means a runaway loop.
_MAX_STEPS = 12


@pytest.fixture(scope="module", params=_CASES, ids=_IDS)
def normal_run(request, tmp_path_factory):
    """One offline normal run per case, shared across the dimension checks."""

    case: PreflightCase = request.param
    workdir = tmp_path_factory.mktemp(f"pf_{case.task_id}")
    return run_preflight(case, workdir=workdir, n_rows=80)


# ---------------------------------------------------------------------------
# Fixtures are bound to the formal task protocol (owner requirement 2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_fixture_is_bound_to_a_real_suite_task(case: PreflightCase):
    suite_ids = {t.task_id for t in easyicu_evaluation_protocol_suite().tasks}
    assert case.task_id in suite_ids
    assert case.diagnostic_only is True


# ---------------------------------------------------------------------------
# Dimension 1 — plan-contract validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_dim1_plan_contract_is_valid(case: PreflightCase):
    plan = case.build_plan()
    # Round-trips through the strict AnalysisPlan schema.
    reparsed = type(plan).model_validate_json(plan.model_dump_json())
    assert reparsed.analysis_type == "association_study"
    table_one = next(s for s in plan.steps if s.step_id == DETERMINISTIC_STEP_ID)
    assert table_one.table_one_spec is not None
    assert table_one.expected_outputs == ["table:table_one"]
    primary = next(s for s in plan.steps if s.step_id == PRIMARY_STEP_ID)
    assert primary.planned_analysis_role == "primary"


def test_dim1_injected_plan_is_accepted_once(normal_run):
    # The scripted plan was accepted by the planner on the first attempt (no
    # parse-retry churn, no deterministic-planner fallback).
    assert normal_run.llm.plan_calls == 1
    assert normal_run.raised is None


# ---------------------------------------------------------------------------
# Dimension 2 — expected graph stages
# ---------------------------------------------------------------------------


def test_dim2_expected_graph_stages(normal_run):
    step_ids = normal_run.step_ids
    # Probe + the typed Table 1 + the agent-owned primary always run; the plan
    # shaper adds the family figure/audit stages.
    assert "00_probe" in step_ids
    assert DETERMINISTIC_STEP_ID in step_ids
    assert PRIMARY_STEP_ID in step_ids
    # At least one shaper-added display/audit stage beyond the authored two.
    assert len(step_ids) >= 4


# ---------------------------------------------------------------------------
# Dimension 3 — deterministic executor/renderer vs coder division of labour
# ---------------------------------------------------------------------------


def test_dim3_table_one_is_host_deterministic(normal_run):
    rec = normal_run.record(DETERMINISTIC_STEP_ID)
    assert rec.get("deterministic_standard_analysis") == "grouped_table_one"
    assert rec.get("generation_mode") == "deterministic_standard"
    assert rec.get("status") == "ok"


def test_dim3_primary_is_agent_coded(normal_run):
    rec = normal_run.record(PRIMARY_STEP_ID)
    # The scientific estimand is agent-owned: no standard executor claims it.
    assert rec.get("deterministic_standard_analysis") is None
    assert rec.get("generation_mode") == "llm"
    assert rec.get("status") == "ok"


# ---------------------------------------------------------------------------
# Dimension 5 — per-step timeout wiring
# ---------------------------------------------------------------------------


def test_dim5_per_step_timeout_wiring(normal_run):
    # The deterministic standard executor gets the standard-executor timeout;
    # the agent-coded step gets the ordinary step timeout.  Distinct values
    # prove the timeout config flows per step.
    t1 = normal_run.record(DETERMINISTIC_STEP_ID).get("execution_timeout_seconds")
    primary = normal_run.record(PRIMARY_STEP_ID).get("execution_timeout_seconds")
    assert t1 == 900.0
    assert primary == 60.0
    assert t1 != primary


# ---------------------------------------------------------------------------
# Dimension 7 — loop termination (bounded, no hang)
# ---------------------------------------------------------------------------


def test_dim7_run_terminates_bounded(normal_run):
    # Reaching here at all means the graph did not hang.  The step count is far
    # below the runaway ceiling.
    assert 0 < len(normal_run.step_ids) <= _MAX_STEPS


# ---------------------------------------------------------------------------
# Dimension 8 — final tristate matches expectation
# ---------------------------------------------------------------------------


def test_dim8_final_tristate_is_diagnostic_only(normal_run):
    # A minimal diagnostic-only fixture never completes the full article
    # display contract, so the honest fail-closed verdict is diagnostic_only.
    assert normal_run.tristate == normal_run.case.expected_tristate == "diagnostic_only"
    assert normal_run.readiness.get("manuscript_ready") is False
    assert normal_run.readiness.get("publication_ready") is False


# ---------------------------------------------------------------------------
# Dimension 9 — no paper authority + zero external Provider
# ---------------------------------------------------------------------------


def test_dim9_zero_external_provider(normal_run):
    assert normal_run.llm_is_mock is True
    assert normal_run.external_provider_calls == 0
    assert isinstance(normal_run.llm, ScriptedPreflightLLM)


def test_dim9_mock_run_has_no_paper_authority(normal_run):
    # The production Figure 2 acceptance gate rejects a single diagnostic-only
    # mock run (needs the exact 9-task, aware-arm, replay-verified batch).
    assert paper_acceptance_status(normal_run) == "invalid"


# ---------------------------------------------------------------------------
# Dimension 4 — repair/retry cap under coder fault injection -> fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_dim4_repair_cap_bounds_and_fails_closed(case: PreflightCase, tmp_path):
    cap = 1
    run = run_preflight(
        case,
        workdir=tmp_path,
        n_rows=60,
        fault_step=PRIMARY_STEP_ID,
        max_code_repair_attempts=cap,
    )
    # Deterministic, bounded repair budget: one initial write + `cap` repair
    # rounds (two code calls each) — never an unbounded loop.
    assert run.llm.code_calls.get(PRIMARY_STEP_ID) == 2 * cap + 1
    assert run.raised is None
    assert run.record(PRIMARY_STEP_ID).get("status") in {
        "repair_failed",
        "execution_failed",
        "contract_failed",
    }
    # Fault at the primary fails the run closed.
    assert run.tristate == "diagnostic_only"
    assert run.external_provider_calls == 0


def test_dim4_zero_repair_cap_makes_a_single_attempt(tmp_path):
    # With the repair budget disabled the failing step is attempted exactly once.
    run = run_preflight(
        E1E3_CASES["e2_lactate_mortality"],
        workdir=tmp_path,
        n_rows=50,
        fault_step=PRIMARY_STEP_ID,
        max_code_repair_attempts=0,
    )
    assert run.llm.code_calls.get(PRIMARY_STEP_ID) == 1
    assert run.tristate == "diagnostic_only"


# ---------------------------------------------------------------------------
# Dimension 6 — stop / resume with no double-execution
# ---------------------------------------------------------------------------


def _step_dir(run_dir, step_id):
    """Locate a step's on-disk output directory under the authoritative run dir."""

    direct = run_dir / "steps" / step_id
    if direct.exists():
        return direct
    return next((p for p in run_dir.rglob(step_id) if p.is_dir()), None)


def _latest_mtime(path):
    if path is None or not path.exists():
        return None
    files = [p for p in path.rglob("*") if p.is_file()]
    return max((p.stat().st_mtime for p in files), default=None)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_dim6_stop_then_resume_without_double_execution(case: PreflightCase, tmp_path):
    # Stop right after the deterministic Table 1 step.
    stop = run_preflight(
        case, workdir=tmp_path, n_rows=40, stop_after_step_id=DETERMINISTIC_STEP_ID
    )
    assert stop.raised is None
    assert stop.step_ids == ["00_probe", DETERMINISTIC_STEP_ID]
    assert stop.record(DETERMINISTIC_STEP_ID).get("status") == "ok"

    run_dir = stop.run_dir
    step_dir = _step_dir(run_dir, DETERMINISTIC_STEP_ID)
    mtime_after_stop = _latest_mtime(step_dir)
    assert mtime_after_stop is not None
    assert _step_dir(run_dir, PRIMARY_STEP_ID) is None  # primary not run yet

    # Resume the same run.
    resume = run_preflight(case, workdir=tmp_path, n_rows=40, resume_run_id=stop.run_id)
    assert resume.raised is None
    assert resume.run_id == stop.run_id
    assert resume.run_dir == run_dir
    # Continued past the stop point.
    assert PRIMARY_STEP_ID in resume.step_ids
    assert _step_dir(run_dir, PRIMARY_STEP_ID) is not None
    # The already-completed Table 1 step was reused, not re-executed: its output
    # directory is byte-for-byte untouched (mtime unchanged).
    assert _latest_mtime(_step_dir(run_dir, DETERMINISTIC_STEP_ID)) == mtime_after_stop
    assert resume.external_provider_calls == 0
