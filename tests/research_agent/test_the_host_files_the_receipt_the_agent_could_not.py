"""The largest single pre-execution blocker, handed back to the host.

``flag_only_plausibility_obligation`` is mechanical: read each declared
column's bounds from the sealed manifest, count what falls outside, file the
counts under one exact key in ``step_summary.json``. The host renders it
correctly for its own deterministic executors. Agent-authored steps had to
hand-write it.

MEASURED over every recorded run: 37 findings across 32 distinct steps in 8 of
the 9 tasks -- 53 % of every mechanical-preflight finding, by far the largest
single cause. h2's causal step spent BOTH of its LLM repairs on this one
message with five provider calls still unspent, and died anyway.

A host helper the agent calls was considered and rejected on the decisive
point: it still depends on the agent REMEMBERING to call it, which is the exact
thing that fails 37 times. The receipt module's own docstring gives the second
reason -- the comparisons are rendered into the source so the static gate can
verify the code that will actually run.

Replayed over every recorded blocked step whose script survives on disk:
19 of 21 clear. The 2 that remain fail for a different reason
(``out_of_range_record_not_in_declared_output``) and are not what this repairs.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.execution.runners.plausibility_receipt import (
    host_plausibility_receipt_injected,
)
from easyicu.research_agent.gates.plausibility_obligation import (
    flag_only_plausibility_obligation_findings,
)
from easyicu.research_agent.schema import AnalysisStep

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")
_BODY = "import pandas as pd\n\nframe = pd.read_parquet('/cohort.parquet')\nprint(len(frame))\n"


def _scope(*columns: str) -> FlagOnlyPlausibilityScope:
    return FlagOnlyPlausibilityScope(
        step_id="05_step",
        expected_columns=tuple(columns),
        source_contracts_sha256="a" * 64,
        authority_kind="resolved_raw_input_contracts",
    )


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="05_step",
        intent="Primary estimation.",
        method="adjusted_association_models",
        planned_analysis_role="primary",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
    )


# ---------------------------------------------------------------------------
# It does nothing unless it is owed
# ---------------------------------------------------------------------------


def test_no_scope_leaves_the_script_byte_identical():
    assert (
        host_plausibility_receipt_injected(_BODY, scope=None, already_satisfied=False)
        == _BODY
    )


def test_an_empty_scope_leaves_the_script_byte_identical():
    assert (
        host_plausibility_receipt_injected(
            _BODY, scope=_scope(), already_satisfied=False
        )
        == _BODY
    )


def test_a_script_that_already_satisfies_the_gate_is_untouched():
    """The deterministic executors already carry the receipt.

    Injecting again would render it twice, so the caller passes the gate's own
    verdict and a satisfied script must come back unchanged.
    """

    assert (
        host_plausibility_receipt_injected(
            _BODY, scope=_scope("age"), already_satisfied=True
        )
        == _BODY
    )


def test_an_empty_body_is_not_wrapped():
    for empty in ("", "   \n"):
        assert (
            host_plausibility_receipt_injected(
                empty, scope=_scope("age"), already_satisfied=False
            )
            == empty
        )


# ---------------------------------------------------------------------------
# What it produces
# ---------------------------------------------------------------------------


def test_the_agent_body_survives_verbatim():
    """The host adds; it never edits the agent's analysis."""

    out = host_plausibility_receipt_injected(
        _BODY, scope=_scope("age"), already_satisfied=False
    )

    assert _BODY.strip() in out
    ast.parse(out)


def test_the_receipt_runs_after_the_body_and_files_last():
    """CORRECTED. The first version asserted the receipt ran FIRST.

    A prologue would have been worse than useless: agent bodies routinely bind
    ``plausibility_audit`` themselves, and a body running after a prologue
    silently overwrites the host's value. Measured on the recorded scripts,
    three of them do exactly that. Running last makes the host's value win by
    construction.
    """

    out = host_plausibility_receipt_injected(
        _BODY, scope=_scope("age"), already_satisfied=False
    )

    body_at = out.index("print(len(frame))")
    receipt_at = out.index("EASYICU_RESOLVED_INPUTS_JSON")
    filing_at = out.index('_easyicu_host_summary["plausibility_audit"]')
    assert body_at < receipt_at < filing_at


def test_the_bounds_are_read_from_the_sealed_manifest():
    out = host_plausibility_receipt_injected(
        _BODY, scope=_scope("age"), already_satisfied=False
    )

    assert "analysis_plausibility_range" in out
    assert "EASYICU_RESOLVED_INPUTS_JSON" in out
    assert "a" * 64 in out  # the scope digest the receipt self-checks against


def test_a_body_that_binds_the_same_name_is_overridden_not_killed():
    """The case that forced the design, taken from the recorded scripts.

    Three recorded bodies bind ``plausibility_audit`` themselves --
    ``= {}`` and ``= build_plausibility_audit(...)``. A first version put the
    receipt in a prologue and added a runtime check that RAISED when the value
    changed; against those bodies it would have killed the very steps it was
    meant to save, from host-injected code. The host now simply computes last.
    """

    body = "plausibility_audit = {}\nprint(len(plausibility_audit))\n"
    out = host_plausibility_receipt_injected(
        body, scope=_scope("age"), already_satisfied=False
    )

    assert out.index("plausibility_audit = {}") < out.index(
        "EASYICU_RESOLVED_INPUTS_JSON"
    )
    assert 'raise RuntimeError(\n        "The host-computed' not in out
    # And the host's own name is not leaked into the tracked flow.
    assert "_easyicu_host_plausibility_audit" not in out


def test_the_filing_is_guarded_on_a_summary_the_body_actually_wrote():
    """Creating one would manufacture a record the step never produced."""

    out = host_plausibility_receipt_injected(
        _BODY, scope=_scope("age"), already_satisfied=False
    )

    assert "if _easyicu_host_summary_path.exists():" in out


# ---------------------------------------------------------------------------
# It clears the real gate, on the real recorded scripts
# ---------------------------------------------------------------------------


def _gate(code: str, scope: FlagOnlyPlausibilityScope, step: AnalysisStep):
    return flag_only_plausibility_obligation_findings(
        ast.parse(code), script_text=code, step=step, scope=scope
    )


def test_the_recorded_failure_is_cleared():
    """h2's causal step: the one that burned both repairs and died."""

    run = (
        _CORPUS
        / "batch_20260804_luna_miiv_FULL_13b0aa5_verify23"
        / "h2_vasopressor_causal"
        / "aware"
    )
    if not run.exists():
        pytest.skip("the verify23 run is not mounted")
    manifests = list(run.glob("run_*/manifest.json"))
    if not manifests:
        pytest.skip("the recorded run carries no manifest")

    manifest = json.loads(manifests[0].read_text())
    record = next(
        (
            item
            for item in manifest.get("per_step_records", [])
            if str(item.get("step_id", "")).startswith("05_")
        ),
        None,
    )
    if record is None or not record.get("flag_only_plausibility_scope"):
        pytest.skip("that run's plan had no scoped estimation step")

    recorded = record["flag_only_plausibility_scope"]
    script = (
        manifests[0].parent
        / "steps"
        / recorded["step_id"]
        / ".quarantine"
        / "concept_draft.py"
    )
    if not script.exists():
        pytest.skip("the recorded draft is not mounted")

    scope = FlagOnlyPlausibilityScope(
        step_id=recorded["step_id"],
        expected_columns=tuple(recorded["expected_columns"]),
        source_contracts_sha256=recorded["source_contracts_sha256"],
        authority_kind=recorded["authority_kind"],
    )
    step = AnalysisStep(
        step_id=recorded["step_id"],
        intent="Primary causal estimation.",
        method="causal_inference",
        planned_analysis_role="primary",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:primary_causal_contrast"],
    )
    source = script.read_text(encoding="utf-8", errors="replace")

    assert _gate(source, scope, step), "the recorded script no longer fails the gate"
    injected = host_plausibility_receipt_injected(
        source, scope=scope, already_satisfied=False
    )
    assert _gate(injected, scope, step) == []


def test_the_injection_runs_before_the_concept_audit():
    """Digest identity depends on the ORDER, so read it as a syntax tree.

    ``concept_approved_code_sha256`` hashes the ``code`` string and
    ``executed_code_sha256`` hashes the file written from it. Injecting after
    the audit would make the executed script differ from the approved one; a
    source-text test would survive commenting the call out, so this walks the
    module.
    """

    import inspect

    from easyicu.research_agent.execution import phase as execution_phase

    tree = ast.parse(inspect.getsource(execution_phase))
    # The assignment, not merely the call: a mutation reading
    # ``code = code or _host_plausibility_receipt_injected(...)`` leaves the
    # call in place and never runs it, and a call-only test survives that.
    inject_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "code"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_host_plausibility_receipt_injected"
    ]
    audit_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ConceptAuditCoordinator"
    ]

    assert inject_lines, "the host never injects the receipt"
    assert audit_lines, "the concept audit coordinator is no longer constructed"
    assert min(inject_lines) < min(audit_lines)
