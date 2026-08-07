"""The host wrote its own receipt only where the step could never reach.

``_write_host_input_binding_receipts`` exists because the host, not the
generated script, knows which typed inputs it resolved: a producer whose code
the host rendered cannot be asked to attest to its own input.

The execute layer called it from two places and spelled the eligibility rule
out by hand at each:

* line ~9229, BEFORE ``_step_deterministic_contract_findings`` -- registered
  standard executor or sealed renderer only;
* line ~10827, AFTER that gate -- the same two PLUS
  ``deterministic_fallback_used``.

The three deterministic fallback runners (robustness/sensitivity,
absolute-risk context, missingness audit) are in the second list and not the
first.  So their receipts were written at a point they could not reach: the
contract gate refuses the step first, for the absence of exactly those
receipts.  ``step_summary_integrity`` reports ``input_bindings_missing``, the
host dispatches a contract repair against its own rendered code, and the two
repair slots are gone.

Measured over every recorded run: 18 steps ran on a deterministic fallback
runner, 12 were refused for a missing host input-binding receipt (all 12 the
robustness/sensitivity runner), and 11 of those 12 died -- 7 contract_failed,
3 blocked_by_concept_audit, 1 execution_failed.  The single survivor is the
one the earlier comment at the post-gate site already named: a Coder rewrite
that hand-built the receipt block.

That comment shows the asymmetry was found once and the widening was applied
to the wrong one of the two sites.  The rule now has one owner and both sites
ask it, so a third caller cannot reintroduce a narrower copy.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from easyicu.research_agent.authority.typed_binding import (
    host_owns_input_binding_receipts,
)


def _call(**overrides: bool) -> bool:
    kwargs = {
        "deterministic_standard_executor_used": False,
        "deterministic_fallback_used": False,
        "sealed_renderer_repair": False,
    }
    kwargs.update(overrides)
    return host_owns_input_binding_receipts(**kwargs)


def test_a_deterministic_fallback_runner_is_owed_a_host_receipt() -> None:
    """The property that was false at the pre-gate site.

    This is the whole defect: the robustness/sensitivity runner is host-
    authored code, so the host owes it a receipt -- and owes it before the
    gate that demands one.
    """

    assert _call(deterministic_fallback_used=True) is True


@pytest.mark.parametrize(
    "flag",
    [
        "deterministic_standard_executor_used",
        "deterministic_fallback_used",
        "sealed_renderer_repair",
    ],
)
def test_every_host_authored_producer_is_owed_a_receipt(flag: str) -> None:
    assert _call(**{flag: True}) is True


def test_generated_code_is_not_owed_a_receipt() -> None:
    """A Coder script must still report its own bindings.

    If this flipped, the host would manufacture receipts for code it did not
    write -- attesting to an input it never resolved for that script.
    """

    assert _call() is False


# --- the rule has exactly one owner ------------------------------------------

_PHASE = Path(
    __import__("easyicu.research_agent.execution.phase", fromlist=["__file__"]).__file__
)


def _receipt_writer_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_host_input_binding_receipts"
    ]


def test_the_two_general_call_sites_ask_the_owner() -> None:
    """Both sites that gate on producer kind must call the shared predicate.

    Anchored on the number of ``host_owns_input_binding_receipts`` calls rather
    than on line numbers, which move.  The third writer call site is inside the
    publication-figure repair and is deliberately narrower -- it fires only for
    a sealed renderer repair -- so it is not counted here.
    """

    tree = ast.parse(_PHASE.read_text())
    owner_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "host_owns_input_binding_receipts"
    ]
    assert len(owner_calls) == 2, (
        "the pre-gate and post-execution receipt sites must both ask the owner; "
        f"found {len(owner_calls)} call(s)"
    )
    assert len(_receipt_writer_calls(tree)) == 3


def test_no_call_site_respells_the_rule() -> None:
    """The shape that let the two copies drift.

    A boolean test on ``deterministic_standard_executor_used`` written inline
    is how the narrower copy existed in the first place; the flags must reach
    the decision only as arguments to the owner.
    """

    tree = ast.parse(_PHASE.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp):
            continue
        rendered = ast.unparse(node)
        if (
            "deterministic_standard_executor_used" in rendered
            and "is_sealed_renderer_repair" in rendered
        ):
            offenders.append(rendered[:120])
    assert not offenders, (
        "a call site decides receipt ownership by hand instead of asking the "
        f"owner: {offenders}"
    )


def test_the_owner_lives_with_the_writer_it_governs() -> None:
    from easyicu.research_agent.authority import typed_binding

    assert "host_owns_input_binding_receipts" in typed_binding.__all__
    assert (
        host_owns_input_binding_receipts.__module__
        == typed_binding._write_host_input_binding_receipts.__module__
    )
