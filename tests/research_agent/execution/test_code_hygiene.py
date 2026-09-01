"""Regression tests for easyicu.research_agent.execution.code_hygiene."""

from __future__ import annotations

import textwrap

import pytest

from easyicu.research_agent.execution.code_hygiene import (
    forward_reference_report,
    reorder_forward_references,
)


def _exec(source: str) -> dict:
    """Execute ``source`` in a fresh namespace and return the namespace."""
    ns: dict = {}
    exec(compile(source, "<rewritten>", "exec"), ns)
    return ns


def test_no_forward_reference_is_a_noop():
    source = textwrap.dedent("""
        import json

        def to_jsonable(x):
            return str(x)

        payload = {"a": 1}
        print(json.dumps(payload, default=to_jsonable))
        """).lstrip()

    assert reorder_forward_references(source) == source
    assert forward_reference_report(source) == []


def test_qwen30b_to_json_serializable_regression():
    """Reproduces the exact bug seen in the v15 t01_table_one_descriptive run.

    qwen3-coder-30b emitted ``json.dump(..., default=to_json_serializable)``
    at the top of the script but put ``def to_json_serializable(...)`` at
    the bottom, which failed with ``NameError`` at execution time. The
    hoisted rewrite must execute successfully.
    """
    buggy = textwrap.dedent("""
        import json

        payload = {"missing_pct": 0.5}
        with open("/tmp/ignore_me", "w") as f:
            pass  # stand-in for the real file write site
        serialised = json.dumps(payload, default=to_json_serializable)

        def to_json_serializable(obj):
            return str(obj)
        """).lstrip()

    # The un-rewritten version must actually trigger the bug.
    with pytest.raises(NameError):
        _exec(buggy)

    rewritten = reorder_forward_references(buggy)
    assert "easyicu code_hygiene" in rewritten
    # After hoisting, the script executes without NameError.
    ns = _exec(rewritten)
    assert "to_json_serializable" in ns
    assert ns["serialised"] == '{"missing_pct": 0.5}'


def test_hoist_respects_imports_and_docstring():
    source = textwrap.dedent('''
        """Module docstring."""
        import json
        import os

        data = helper({"k": 1})

        def helper(payload):
            return json.dumps(payload)
        ''').lstrip()

    rewritten = reorder_forward_references(source)
    lines = rewritten.splitlines()
    # Docstring and both imports must remain the first substantive lines.
    assert lines[0].startswith('"""')
    assert any(l.strip() == "import json" for l in lines[:5])
    assert any(l.strip() == "import os" for l in lines[:5])
    # The hoisted def must appear before the use of ``helper`` on RHS.
    def_idx = next(i for i, l in enumerate(lines) if l.startswith("def helper"))
    use_idx = next(i for i, l in enumerate(lines) if l.startswith("data ="))
    assert def_idx < use_idx

    ns = _exec(rewritten)
    assert ns["data"] == '{"k": 1}'


def test_multiple_forward_references_preserve_original_order():
    source = textwrap.dedent("""
        import json

        a = first({"n": 1})
        b = second({"n": 2})

        def second(x):
            return json.dumps(x)

        def first(x):
            return json.dumps(x)
        """).lstrip()

    rewritten = reorder_forward_references(source)

    # ``second`` is defined before ``first`` in the original source; the
    # rewrite must preserve that relative order to avoid accidentally
    # re-ordering defs that depend on each other.
    second_idx = rewritten.index("def second(")
    first_idx = rewritten.index("def first(")
    assert second_idx < first_idx

    ns = _exec(rewritten)
    assert ns["a"] == '{"n": 1}'
    assert ns["b"] == '{"n": 2}'


def test_nested_reference_inside_function_body_is_not_hoisted():
    """Names referenced only inside other functions are legal already."""
    source = textwrap.dedent("""
        def outer():
            return inner()

        def inner():
            return 42
        """).lstrip()

    # Both names are referenced only from function bodies; nothing to hoist.
    assert reorder_forward_references(source) == source
    assert forward_reference_report(source) == []
    ns = _exec(source)
    assert ns["outer"]() == 42


def test_syntax_error_is_returned_unchanged():
    bad = "def foo(:\n"
    assert reorder_forward_references(bad) == bad


def test_rewrite_is_idempotent():
    source = textwrap.dedent("""
        import json
        x = helper()

        def helper():
            return 1
        """).lstrip()

    once = reorder_forward_references(source)
    twice = reorder_forward_references(once)
    assert once == twice


def test_class_forward_reference_is_hoisted():
    source = textwrap.dedent("""
        instance = Thing(7)

        class Thing:
            def __init__(self, v):
                self.v = v
        """).lstrip()

    rewritten = reorder_forward_references(source)
    ns = _exec(rewritten)
    assert ns["instance"].v == 7
