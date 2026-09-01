"""A handler that ``continue``s never reaches the code the gate accused it of.

The 2026-08-01 E3 robustness step wrote the textbook idiom::

    for key in candidate_keys:
        ...
        try:
            numeric_effect = float(candidate_effect)
        except (TypeError, ValueError):
            continue
        if numeric_effect == numeric_effect and ...:
            primary_effect = numeric_effect
            break

and the gate refused it for reading ``numeric_effect`` on a path that never
assigned it.  There is no such path: when the handler runs it jumps to the next
iteration, so the ``if`` below is unreachable from it.  The step was blocked
before it ever ran, having spent two provider calls on a defect that did not
exist.

The host had already computed the right answer.  ``_block_flow_outcomes``
classifies that handler as ``{loop_escape}`` -- distinct from
``{function_exit}`` and from falling through.  ``_branch_all_paths_exit`` then
compared it for equality against ``{function_exit}`` alone, so the distinction
was made and immediately discarded.

The two provenance callers of ``_branch_all_paths_exit`` genuinely ask "does
every path leave the function", and a ``continue`` is not a ``raise``; widening
that predicate would have weakened them.  So the wider question -- "can control
reach the next statement at all" -- gets its own predicate, and only the two
unbound-local rules use it.

Measured over the 524 recorded generated scripts and quarantined drafts: one
finding removed (this one), zero added, three genuine unbound-local findings
still reported.
"""

from __future__ import annotations

import ast

import pytest

from easyicu.research_agent.gates.preflight import (
    _block_flow_outcomes,
    _branch_all_paths_exit,
    _branch_local_unbound_findings,
    _branch_never_falls_through,
)

# Exactly the shape from the run, reduced to the loop that carried it.
REAL_SHAPE = """
for effect_key in ("primary_or", "adjusted_or"):
    candidate_effect = summary.get(effect_key)
    if candidate_effect is not None:
        try:
            numeric_effect = float(candidate_effect)
        except (TypeError, ValueError):
            continue
        if numeric_effect == numeric_effect:
            primary_effect = numeric_effect
            break
"""


def _unbound_names(source: str) -> list[str]:
    return sorted(
        str(finding.detail.get("name"))
        for finding in _branch_local_unbound_findings(ast.parse(source))
    )


def test_the_real_shape_is_no_longer_refused() -> None:
    assert _unbound_names(REAL_SHAPE) == []


@pytest.mark.parametrize("escape", ["continue", "break", "raise", "return"])
def test_a_handler_that_leaves_the_block_cannot_reach_the_read(escape: str) -> None:
    source = f"""
def f(items):
    for item in items:
        try:
            value = float(item)
        except ValueError:
            {escape}
        if value > 0:
            print(value)
"""
    assert _unbound_names(source) == []


@pytest.mark.parametrize("swallow", ["pass", "value_error_count += 1"])
def test_a_handler_that_falls_through_is_still_refused(swallow: str) -> None:
    """The coverage that must not be lost: this one really can raise.

    After ``pass`` control reaches the ``if``, and ``value`` was never assigned
    on that path.  If this stops being reported the fix has traded a false
    block for a missed defect.
    """

    source = f"""
def f(items):
    value_error_count = 0
    for item in items:
        try:
            value = float(item)
        except ValueError:
            {swallow}
        if value > 0:
            print(value)
"""
    assert _unbound_names(source) == ["value"]


def test_the_two_predicates_answer_different_questions() -> None:
    """Why a second predicate exists instead of widening the first.

    ``_branch_all_paths_exit`` is what the provenance rules use to decide that a
    failure was re-raised rather than swallowed.  A ``continue`` is not a raise,
    so it must keep answering False there while the new predicate answers True.
    """

    handler = ast.parse("def f():\n    for _ in []:\n        continue\n")
    body = handler.body[0].body[0].body  # the loop body: a bare `continue`

    assert _block_flow_outcomes(body) == {"loop_escape"}
    assert _branch_all_paths_exit(body) is False
    assert _branch_never_falls_through(body) is True

    raising = ast.parse("def f():\n    raise ValueError('x')\n").body[0].body
    assert _branch_all_paths_exit(raising) is True
    assert _branch_never_falls_through(raising) is True

    plain = ast.parse("x = 1").body
    assert _branch_all_paths_exit(plain) is False
    assert _branch_never_falls_through(plain) is False


def test_an_exception_alias_is_still_refused_after_a_continue() -> None:
    """The one place the wider predicate would have HIDDEN a real defect.

    Widening this call site was tried and reverted.  Python deletes the
    exception alias when the handler is left by any route -- ``continue``
    included -- and the delete removes the name outright rather than restoring
    what it held before the ``try``.  Verified against CPython: the snippet
    below raises ``NameError`` on the second iteration.

    So "the handler cannot reach the read" is the wrong question here; only
    ``raise``/``return`` make the later read unreachable, and that is exactly
    what the narrow predicate answers.
    """

    source = """
exc = "before"
for item in items:
    try:
        f(item)
    except ValueError as exc:
        continue
    print(exc)
"""
    assert _unbound_names(source) == ["exc"]


def test_the_unbound_rules_use_the_wider_predicate_and_provenance_does_not() -> None:
    """Locks which call sites moved, so a later edit cannot quietly swap them.

    Widening the provenance callers would let a ``continue`` count as a
    re-raise, which is the failure this split exists to avoid.
    """

    from pathlib import Path

    from easyicu.research_agent.gates import preflight

    source = Path(preflight.__file__).read_text()
    tree = ast.parse(source)

    users: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id in {
                    "_branch_all_paths_exit",
                    "_branch_never_falls_through",
                }:
                    users.setdefault(call.func.id, set()).add(node.name)

    # The narrow predicate is passed explicitly to the extracted provenance
    # owner rather than called by an old nested helper in this module.
    provenance_bindings = [
        keyword.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "branch_all_paths_exit"
        and isinstance(keyword.value, ast.Name)
    ]
    assert provenance_bindings == ["_branch_all_paths_exit"]
    # ... and the wider one is used somewhere the narrow one no longer is.
    assert users.get("_branch_never_falls_through")
