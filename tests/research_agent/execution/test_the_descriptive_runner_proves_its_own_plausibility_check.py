"""The host's own sealed runner has to satisfy the host's own obligation.

``absolute_risk_context_code`` returned a four-line stub: import the host
function, call it.  Everything the step does happens inside imported code.

The flag-only plausibility gate reads the SCRIPT'S OWN SOURCE looking for a
comparison against bounds taken from the resolved contract.  A delegating stub
shows it nothing, so the gate answers ``plausibility_check_not_attributable``
however correctly the callee behaves -- the host refusing code the host wrote.

Nobody had seen it, because this owner had never claimed a step: its ownership
predicate carried a method allowlist the Planner never matched, 0 claims in 89
opportunities.  The first real run in which it claimed (e2, 2026-08-04) died
here, at ``03_absolute_risk_context``, with two concept repairs spent against a
script no model authored.

The fix renders the receipt into the script, exactly as the six sibling runners
that were reachable already do.  The stub is unchanged for a step under no
obligation, so a runner that owes nothing still emits four lines.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.execution.runners.deterministic_descriptive import (
    absolute_risk_context_code,
)
from easyicu.research_agent.gates.plausibility_obligation import (
    flag_only_plausibility_obligation_findings,
)
from easyicu.research_agent.schema import AnalysisStep

#: Taken verbatim from the run record of the step that died (e2, verify18).
_RECORDED_SCOPE = {
    "step_id": "03_absolute_risk_context",
    "expected_columns": ("lact_max",),
    "source_contracts_sha256": (
        "8cd0899da917f95a298155512ffe46582be9edf9c336199337ff5984db2f7196"
    ),
    "authority_kind": "resolved_raw_input_contracts",
}


def _scope(**overrides) -> FlagOnlyPlausibilityScope:
    return FlagOnlyPlausibilityScope(**{**_RECORDED_SCOPE, **overrides})


def _step(step_id: str = "03_absolute_risk_context") -> AnalysisStep:
    return AnalysisStep(
        step_id=step_id,
        intent="Descriptive exposure prevalence and absolute risk.",
        method="descriptive",
        planned_analysis_role="auxiliary",
        inputs=["artifact:analysis_cohort", "lact_max"],
        expected_outputs=["table:absolute_risk_context"],
    )


def _gate(code: str, scope: FlagOnlyPlausibilityScope, step: AnalysisStep):
    return flag_only_plausibility_obligation_findings(
        ast.parse(code), script_text=code, step=step, scope=scope
    )


# ---------------------------------------------------------------------------
# The production failure, reproduced and cleared
# ---------------------------------------------------------------------------


def test_the_delegating_stub_is_what_the_gate_refused():
    """Reproduces the recorded refusal from the runner's own former output."""

    scope, step = _scope(), _step()
    findings = _gate(absolute_risk_context_code(), scope, step)

    assert len(findings) == 1, findings
    message = str(findings[0].message)
    assert "no comparison against a bound read from it can be located" in message
    assert (
        findings[0].detail.get("reason") == "plausibility_check_not_attributable"
    ), findings[0].detail


def test_the_rendered_script_clears_the_same_gate():
    """The whole point: same scope, same step, same gate, no finding."""

    scope, step = _scope(), _step()
    code = absolute_risk_context_code(plausibility_scope=scope)

    assert _gate(code, scope, step) == []


def test_the_script_still_calls_the_host_runner():
    """The receipt is added beside the work, never instead of it.

    A render that satisfied the gate and stopped computing the table would pass
    every assertion above and produce nothing.
    """

    code = absolute_risk_context_code(plausibility_scope=_scope())

    assert "run_absolute_risk_context()" in code
    assert (
        "from easyicu.research_agent.execution.runners.deterministic_descriptive"
        in code
    )


def test_the_audit_is_written_under_the_key_the_gate_names():
    code = absolute_risk_context_code(plausibility_scope=_scope())

    assert 'summary["plausibility_audit"] = plausibility_audit' in code
    assert 'os.environ["STEP_OUT_DIR"]' in code
    # Read-modify-write, not a second canonical write of the summary.
    assert code.count("summary_path.write_text") == 1


def test_the_bounds_come_from_the_contract_not_from_a_literal():
    """The gate's actual demand, and the thing a shortcut would violate."""

    code = absolute_risk_context_code(plausibility_scope=_scope())

    assert "analysis_plausibility_range" in code
    assert 'plausibility_range.get("minimum")' in code
    assert 'plausibility_range.get("maximum")' in code
    # The declared contract digest is checked before any bound is trusted.
    assert _RECORDED_SCOPE["source_contracts_sha256"] in code


def test_the_rendered_script_is_valid_python():
    ast.parse(absolute_risk_context_code(plausibility_scope=_scope()))
    ast.parse(absolute_risk_context_code())


# ---------------------------------------------------------------------------
# A step under no obligation is untouched
# ---------------------------------------------------------------------------


def test_no_scope_still_returns_the_four_line_stub():
    """Most steps owe nothing here; they must not grow a receipt."""

    assert absolute_risk_context_code() == absolute_risk_context_code(
        plausibility_scope=None
    )
    assert "plausibility_audit" not in absolute_risk_context_code()


def test_a_scope_naming_no_column_adds_nothing():
    """An empty obligation is not an obligation."""

    code = absolute_risk_context_code(plausibility_scope=_scope(expected_columns=()))

    assert code == absolute_risk_context_code()


def test_the_scope_step_id_is_still_enforced_by_the_gate():
    """The renderer does not get to decide which step owes the obligation."""

    scope = _scope()
    with pytest.raises(Exception):
        _gate(
            absolute_risk_context_code(plausibility_scope=scope),
            scope,
            _step("07_some_other_step"),
        )


# ---------------------------------------------------------------------------
# Reachability: this owner really does get asked now
# ---------------------------------------------------------------------------


def test_the_dispatch_site_actually_hands_over_the_scope():
    """Without this the whole fix is dead and every other test still passes.

    A mutation that dropped the keyword at the call site survived the rest of
    this file: the renderer keeps working, and nothing ever asks it to render.
    Read as a syntax tree rather than as text, so commenting the call out or
    guarding it with ``if False`` cannot satisfy it.
    """

    import inspect

    from easyicu.research_agent.execution import phase_support as execution_phase_support

    tree = ast.parse(
        inspect.getsource(
            execution_phase_support._step_deterministic_absolute_risk_context_code
        )
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "absolute_risk_context_code"
    ]

    assert calls, "the dispatcher never calls the descriptive runner's code builder"
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "plausibility_scope" in keywords, ast.dump(call)


def test_the_recorded_run_shows_the_owner_claiming_and_then_dying_here():
    """Anchors the file in the run that exposed it, not in a hypothesis."""

    run = pathlib.Path(
        "/Volumes/外置硬盘/easyicu_data/canonical9_runs"
        "/batch_20260804_luna_miiv_FULL_2e7947f_verify18"
    )
    if not run.exists():
        pytest.skip("the verify18 run is not mounted")

    manifests = list(run.glob("*/aware/run_*/manifest.json"))
    if not manifests:
        pytest.skip("the recorded run carries no manifest")
    manifest = json.loads(manifests[0].read_text())

    records = {
        str(record.get("step_id")): record
        for record in manifest.get("per_step_records", [])
    }
    record = records.get("03_absolute_risk_context")
    if record is None:
        pytest.skip("that run's plan had no absolute-risk context step")

    # The owner DID claim it -- that is what made the gate reachable at all.
    assert record.get("deterministic_standard_analysis") == "absolute_risk_context"
    assert record.get("status") == "blocked_by_concept_audit"
