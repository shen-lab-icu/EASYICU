"""A monkeypatched double must accept every keyword production forwards.

THREE failures in one session, same seam, three different symptoms none of which
named the cause:

* ``study_endpoint`` was added to ``LLMConceptAuditor.audit``. Two doubles in
  ``test_resume.py`` kept the older signature, so every call raised ``TypeError``,
  the step aborted mid-run, and the pre-review checkpoint status was left behind.
  The failure read as ``executed_pending_review`` instead of ``repair_failed``.
* ``plan_step_roster`` was added to the same method, breaking the same doubles
  again.
* ``step`` was forwarded to ``deterministic_concept_audit_repair`` when the
  all-rows profile-roles repair was registered. The double there kept its older
  signature, the deterministic-repair path died, and the FIRST run of
  ``test_resume_reaudits_material_deterministic_quarantine_repair`` issued zero
  coder repairs instead of one. That symptom was read as a deliberate lifecycle
  change -- "the deterministic repair no longer gets an execution opportunity" --
  and a bisect was needed to show it was a crashed double. The commit that
  actually caused it was not the one suspected.

Every one of the three was a keyword the double could not accept, and in every
one the reported symptom was a status or a count several layers away. The seam is
cheap to check directly: for each host function tests replace, the replacement's
signature must accept what the production caller forwards.

This does not police behaviour, only arity. That is the whole failure mode.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Iterator

import pytest

from easyicu.research_agent.audits.validators import LLMConceptAuditor
from easyicu.research_agent.repairs.coordination import (
    deterministic_concept_audit_repair,
)

_TESTS = Path(__file__).parents[1]

#: (attribute name as monkeypatched, the real callable it stands in for).
#: Entries are added when a host function acquires a keyword argument, which is
#: exactly when the doubles go stale.
_PATCHED_HOST_CALLABLES = {
    "audit": LLMConceptAuditor.audit,
    "deterministic_concept_audit_repair": deterministic_concept_audit_repair,
}


def _keyword_only_parameters(func: object) -> set[str]:
    """Only the parameters that can ONLY be passed by name.

    A first draft compared the whole signature and produced a false positive: the
    production caller passes ``deterministic_concept_audit_repair``'s first two
    arguments positionally, so a double naming its second parameter ``messages``
    instead of ``audit_messages`` works perfectly. Flagging it would be the
    fail-closed check that blocks correct code -- the failure mode that costs more
    than the one it prevents.

    Keyword-only parameters are exactly the ones a rename or omission breaks, and
    exactly the ones every observed staleness involved.
    """

    return {
        name
        for name, parameter in inspect.signature(func).parameters.items()  # type: ignore[arg-type]
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }


def _local_function_defs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    defs: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node  # type: ignore[assignment]
    return defs


def _double_signatures() -> Iterator[tuple[str, str, str, set[str]]]:
    """Yield (file, patched attribute, double name, its parameter names).

    Finds ``monkeypatch.setattr(<target>, "<attr>", <name>)`` where ``<attr>`` is
    one this module tracks and ``<name>`` resolves to a function defined in the
    same file. A double built any other way is out of scope and is not silently
    counted as checked -- the reachability assertion below fails if the scan
    finds nothing at all.
    """

    for path in sorted(_TESTS.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken test file is its own failure
            continue
        defs = _local_function_defs(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if getattr(func, "attr", None) != "setattr":
                continue
            if len(node.args) != 3:
                continue
            attribute, replacement = node.args[1], node.args[2]
            if not isinstance(attribute, ast.Constant) or not isinstance(
                attribute.value, str
            ):
                continue
            if attribute.value not in _PATCHED_HOST_CALLABLES:
                continue
            if not isinstance(replacement, ast.Name):
                continue
            definition = defs.get(replacement.id)
            if definition is None:
                continue
            args = definition.args
            names = {
                arg.arg
                for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
            }
            if args.kwarg is not None:
                names.add("**")
            yield path.name, attribute.value, replacement.id, names


def test_the_scan_finds_the_doubles_it_claims_to_check() -> None:
    """Reachability first: a scan that matches nothing proves nothing."""

    found = list(_double_signatures())
    assert found, "no monkeypatched double of a tracked host callable was found"
    attributes = {attribute for _, attribute, _, _ in found}
    assert attributes == set(_PATCHED_HOST_CALLABLES), sorted(
        set(_PATCHED_HOST_CALLABLES) - attributes
    )


@pytest.mark.parametrize(
    ("file_name", "attribute", "double_name", "parameters"),
    list(_double_signatures()),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_double_accepts_the_production_signature(
    file_name: str, attribute: str, double_name: str, parameters: set[str]
) -> None:
    if "**" in parameters:
        # A double that accepts anything cannot go stale on arity. It also cannot
        # report a keyword the host stopped reading, which is why this is allowed
        # rather than preferred.
        return
    expected = _keyword_only_parameters(_PATCHED_HOST_CALLABLES[attribute])
    missing = sorted(expected - parameters)
    assert not missing, (
        f"{file_name}::{double_name} stands in for {attribute} but does not "
        f"accept {missing}. Production forwards these, so every call through this "
        "double raises TypeError and the failure surfaces somewhere else entirely."
    )
