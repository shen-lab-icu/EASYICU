"""A step one declaration away from a deterministic owner must be refused.

The gate exists because the alternative is silent: an owner declines over a
field the Planner never filled in, the selector records "contract declined",
and the paper's primary estimate is written by the coder instead of fitted by
the host.  Measured over 553 recorded real steps, that is 26 of them.

Reachability is proved from a recorded artifact rather than a hand-built
fixture -- a check that never fires on real input is worse than none, because
it reads as coverage.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.execution import phase as pipeline_execute
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
    ADJUSTED_ASSOCIATION_OUTPUT,
)
from easyicu.research_agent.execution.owner_declaration import (
    owner_declaration_plan_findings,
    owner_declaration_replan_directive,
)
from easyicu.research_agent.schema import AnalysisPlan

from tests.research_agent.core.test_adjusted_association_executor import _model_terms, _real_step_payload
_COVARIATES = ["age", "sex", "charlson_max"]


def _plan(*steps: dict) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Is the exposure associated with the outcome?",
            "steps": [json.loads(json.dumps(step)) for step in steps],
            "rationale": "owner-declaration gate regression",
        }
    )


def _phase_replan_call_order() -> dict[str, int]:
    """First call line of the plan gate vs the replan dispatcher helpers.

    The owner-declaration gate is called in ``run_execute_phase``; ``_maybe_replan``
    is invoked by the extracted replan helpers after the plan preflight block.
    Concatenate in logical order so the line numbers preserve the ordering.
    """
    import ast
    import inspect

    from easyicu.research_agent.execution import phase as pipeline_execute

    source = (
        inspect.getsource(pipeline_execute.run_execute_phase)
        + "\n"
        + inspect.getsource(pipeline_execute._step_run_initial_replan)
        + "\n"
        + inspect.getsource(pipeline_execute._step_resolve_run_transition)
        + "\n"
        + inspect.getsource(pipeline_execute._step_maybe_directed_model_replan)
    )
    wanted = {"owner_declaration_plan_findings", "_maybe_replan"}
    first: dict[str, int] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in wanted:
                first[name] = min(first.get(name, node.lineno), node.lineno)
    return first


# ---------------------------------------------------------------------------
# Reachability, from the artifact
# ---------------------------------------------------------------------------


def test_the_gate_fires_on_the_recorded_plan_step():
    """The fresh19 step as recorded: primary model declared, no adjustment set."""

    recorded = _real_step_payload()
    assert recorded["model_requirements"][0].get("covariates") is None

    findings = owner_declaration_plan_findings(plan=_plan(recorded))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "error"
    assert finding.detail["reason"] == "owner_declaration_incomplete"
    assert finding.detail["analysis_kind"] == ADJUSTED_ASSOCIATION_ANALYSIS_KIND
    assert finding.detail["missing_declarations"] == [
        "model_requirements[0].covariates"
    ]
    # Cause first: only ``message`` reaches a prompt, clipped from the tail.
    assert finding.message.startswith(f"Step {recorded['step_id']} does not declare")
    assert "model_requirements[0].covariates" in finding.message


# ---------------------------------------------------------------------------
# The two ways it must stay quiet
# ---------------------------------------------------------------------------


def test_a_complete_declaration_produces_no_finding():
    payload = json.loads(json.dumps(_real_step_payload()))
    payload["model_requirements"][0]["covariates"] = list(_COVARIATES)
    payload["model_requirements"][0]["model_terms"] = _model_terms(_COVARIATES)
    assert owner_declaration_plan_findings(plan=_plan(payload)) == []


def test_a_multi_model_step_produces_no_finding():
    """The 33-step case: a shape no owner covers, not a field anyone forgot.

    Reporting it would send the Planner to fix something that is not broken --
    and the fix it would reach for (dropping a model) changes the science.
    """

    payload = json.loads(json.dumps(_real_step_payload()))
    payload["model_requirements"] = [
        payload["model_requirements"][0],
        {
            **payload["model_requirements"][0],
            "requirement_id": "second_model",
            "analysis_role": "secondary",
        },
    ]
    assert owner_declaration_plan_findings(plan=_plan(payload)) == []


def test_a_step_no_owner_recognises_produces_no_finding():
    findings = owner_declaration_plan_findings(
        plan=_plan(
            {
                "step_id": "01_viz",
                "planned_analysis_role": "auxiliary",
                "intent": "Render a figure.",
                "inputs": ["stay_id"],
                "expected_outputs": ["figure:whatever"],
                "method": "visualization",
                "icu_rule_refs": [],
            }
        )
    )
    assert findings == []


# ---------------------------------------------------------------------------
# It must not answer in the permissive direction for a fact it lacks
# ---------------------------------------------------------------------------


def test_a_step_whose_selection_raises_is_reported_unevaluated(monkeypatch):
    """ "Could not check" must never be recorded as "checked, fine"."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("selector exploded")

    monkeypatch.setattr(
        "easyicu.research_agent.execution.owner_declaration.select_standard_executor",
        _boom,
    )
    findings = owner_declaration_plan_findings(plan=_plan(_real_step_payload()))

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["reason"] == "owner_declaration_gate_unevaluated"
    assert findings[0].detail["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# The directive must not invite a scientific edit
# ---------------------------------------------------------------------------


def test_the_directive_is_none_without_findings():
    assert owner_declaration_replan_directive([]) is None


def test_the_directive_forbids_changing_the_science():
    findings = owner_declaration_plan_findings(plan=_plan(_real_step_payload()))
    directive = owner_declaration_replan_directive(findings)

    assert directive is not None
    lowered = directive.lower()
    # A replanner that "fixes" a missing declaration by choosing the value, or
    # by removing the step, has changed the study to satisfy bookkeeping.
    for forbidden in (
        "exposure",
        "outcome",
        "cohort",
        "covariate",
        "estimand",
        "method",
    ):
        assert forbidden in lowered
    assert "do not split or merge steps" in lowered
    assert "do not delete a step" in lowered
    # The findings themselves must travel, or the replanner cannot act.
    assert "model_requirements[0].covariates" in directive


# ---------------------------------------------------------------------------
# Placement: before the repair dispatch, where the Planner can still act
# ---------------------------------------------------------------------------


def test_the_gate_runs_before_the_replan_dispatch():
    """Placement is the whole point of a plan-time gate.

    Evaluated after ``_maybe_replan`` it could not influence the plan it
    describes; evaluated after execution the only remaining move is repairing
    generated code. Read the call order from the AST, not from string indexes.
    """

    order = _phase_replan_call_order()
    assert "owner_declaration_plan_findings" in order, (
        "the gate is not called in run_execute_phase at all"
    )
    assert "_maybe_replan" in order
    assert order["owner_declaration_plan_findings"] < order["_maybe_replan"]


def test_the_gate_forces_the_replan_it_asked_for():
    """A directive nobody dispatches is a comment.

    ``_maybe_replan`` only runs when something forces it; if this gate's
    findings were left out of that condition, a plan whose *only* problem is an
    incomplete declaration would sail through with the directive unsent.
    """

    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(
        inspect.getsource(pipeline_execute._step_run_initial_replan)
    )
    tree = ast.parse(source)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_maybe_replan"
    )
    forced = next(kw for kw in call.keywords if kw.arg == "force")
    names = {node.id for node in ast.walk(forced.value) if isinstance(node, ast.Name)}
    assert "owner_declaration_preflight" in names

    directive = next(kw for kw in call.keywords if kw.arg == "directive")
    directive_names = {
        node.id for node in ast.walk(directive.value) if isinstance(node, ast.Name)
    }
    assert "owner_declaration_directive" in directive_names


@pytest.mark.parametrize("product", [ADJUSTED_ASSOCIATION_OUTPUT])
def test_the_gate_is_keyed_on_the_typed_product_not_a_case_name(product: str):
    """Case neutrality: nothing here may key on a benchmark, column or cohort.

    The owner is asked; the owner keys on the declared method and product. A
    grep for case identifiers in the gate is the cheap half of that claim, and
    the expensive half is that the gate never inspects the step itself.
    """

    from pathlib import Path

    source = Path(
        "src/easyicu/research_agent/execution/owner_declaration.py"
    ).read_text(encoding="utf-8")
    for case_token in ("sep3", "lactate", "mimic", "E1_", "death", "sofa"):
        assert case_token not in source, (
            f"case-specific token {case_token!r} in the gate"
        )
    assert product not in source
