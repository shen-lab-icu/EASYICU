"""AST fixture tests for the architecture-metrics governance tool.

Codex governance requirement (2026-07-17): the scope metrics in
``tools/arch_measure.py`` may only be called "own-scope" if they actually model
callable scope. These fixtures pin the exact semantics of ``own_bound_names`` and
``callable_closure_captured_names`` across the cases that used to be wrong:
nested / nested-nested functions, for/with/except/import bindings, comprehension
scope (targets do NOT leak, first iterable IS outer), class-body binding vs read,
walrus (targets DO leak), and nonlocal/global.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
BASELINE_PATH = TOOLS_DIR / "arch_baselines" / "execution_phase.json"
sys.path.insert(0, str(TOOLS_DIR))
try:
    import arch_measure  # type: ignore[import-not-found]
finally:
    sys.path.pop(0)


def _fn(src: str, name: str) -> ast.AST:
    node = arch_measure._find_function(ast.parse(src), name)
    assert node is not None, name
    return node


def _bound(src: str, name: str) -> set:
    return arch_measure._build_scope_tree(_fn(src, name)).bound


def _captured(src: str, name: str) -> set:
    return arch_measure._callable_closure_captured(
        arch_measure._build_scope_tree(_fn(src, name))
    )


def _captured_metric(src: str, name: str) -> int:
    return arch_measure._function_metrics(_fn(src, name))[
        "callable_closure_captured_names"
    ]


def test_measure_tracks_current_authority_boundaries() -> None:
    measured = arch_measure.measure()["files"]

    assert "execution/phase.py" in measured
    assert "authority/typed_binding.py" in measured
    assert "authority/plan_authority.py" in measured
    assert "schema.py" in measured
    assert measured["authority/plan_authority.py"].get("missing") is not True


def test_checked_in_architecture_baseline_has_no_regression() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert arch_measure.diff(baseline, arch_measure.measure()) == 0


def test_architecture_diff_blocks_function_growth() -> None:
    current = arch_measure.measure()
    baseline = copy.deepcopy(current)
    function = next(iter(current["functions"]))
    baseline["functions"][function]["own_bound_names"] -= 1

    assert arch_measure.diff(baseline, current) == 1


def test_architecture_diff_blocks_target_file_growth() -> None:
    current = arch_measure.measure()
    baseline = copy.deepcopy(current)
    file_name = next(iter(current["files"]))
    baseline["files"][file_name]["loc"] -= 1

    assert arch_measure.diff(baseline, current) == 1


def test_architecture_diff_blocks_missing_target() -> None:
    current = arch_measure.measure()
    baseline = copy.deepcopy(current)
    function = next(iter(current["functions"]))
    current["functions"][function] = {"missing": True}

    assert arch_measure.diff(baseline, current) == 1


# --------------------------------------------------------------------------- #
# own_bound_names: every binding form, not just Assign/AnnAssign/AugAssign.
# --------------------------------------------------------------------------- #


def test_own_bound_covers_all_statement_binding_forms() -> None:
    src = """
def god(a, b):
    x = 1
    y: int = 2
    z = 0
    z += 3
    for i in range(10):
        pass
    with open("f") as fh:
        pass
    try:
        pass
    except ValueError as err:
        pass
    import os
    from sys import argv as av
    total = (w := 5) + w
"""
    bound = _bound(src, "god")
    assert bound == {
        "a",
        "b",  # params
        "x",
        "y",
        "z",  # assign / annassign / augassign
        "i",  # for target
        "fh",  # with ... as
        "err",  # except ... as
        "os",
        "av",  # import / import-from-as
        "total",
        "w",  # assign + walrus
    }


def test_tuple_unpacking_targets_are_bound() -> None:
    src = """
def god():
    a, (b, c) = 1, (2, 3)
    [d, *rest] = [4, 5, 6]
"""
    assert _bound(src, "god") == {"a", "b", "c", "d", "rest"}


def test_attribute_and_subscript_targets_do_not_bind_locals() -> None:
    src = """
def god(obj, arr):
    obj.field = 1
    arr[0] = 2
"""
    # Only params bind; obj.field / arr[0] are not new local names.
    assert _bound(src, "god") == {"obj", "arr"}


# --------------------------------------------------------------------------- #
# Comprehension scope: Py3 loop targets are local to the comprehension.
# --------------------------------------------------------------------------- #


def test_comprehension_targets_do_not_leak_into_function_scope() -> None:
    src = """
def god(items):
    data = [1, 2, 3]
    squares = [n * n for n in data]
    lookup = {k: v for k, v in items}
    flat = [p for row in data for p in row]
"""
    # n, k, v, row, p are comprehension-local and must NOT appear.
    assert _bound(src, "god") == {"items", "data", "squares", "lookup", "flat"}


def test_walrus_inside_comprehension_leaks_to_function_scope() -> None:
    src = """
def god(values):
    filtered = [y for x in values if (y := x * 2) > 3]
"""
    # x stays comprehension-local; y (walrus) leaks to the function scope.
    assert _bound(src, "god") == {"values", "filtered", "y"}


def test_comprehension_first_iterable_is_read_in_enclosing_scope() -> None:
    # Codex counterexample #1: the FIRST generator's iterable is evaluated in the
    # ENCLOSING scope, so a nested function's ``[x for x in x]`` must capture the
    # god-scope ``x`` (the outer iterable), even though the comprehension shadows
    # ``x`` for the element.
    src = """
def god():
    x = [1]
    def child():
        return [x for x in x]
    return child
"""
    assert _captured(src, "god") == {"x"}


def test_comprehension_later_iterable_is_comprehension_local() -> None:
    # Only the FIRST iterable is enclosing-scope; a later iterable that reads a
    # prior comprehension target is comprehension-local, not a god capture.
    src = """
def god():
    rows = [[1], [2]]
    def child():
        return [c for row in rows for c in row]
    return child
"""
    # rows (first iterable) is captured; row/c never reach god.
    assert _captured(src, "god") == {"rows"}


# --------------------------------------------------------------------------- #
# Nested / nested-nested closure capture with LEGB shadowing.
# --------------------------------------------------------------------------- #


def test_closure_capture_respects_grandchild_shadowing() -> None:
    src = """
def god():
    shared = 1
    other = 2
    reused = 3
    def child():
        reused = 10          # child rebinds -> shadows god for its subtree
        def grand():
            return shared + reused + other
        return grand
    return child
"""
    # grand reads shared/reused/other; reused is shadowed by child, so only
    # shared and other resolve back to the god scope.
    assert _captured(src, "god") == {"shared", "other"}


def test_closure_capture_direct_child_reads() -> None:
    src = """
def god():
    used = 1
    unused = 2
    def child():
        return used + 5
    return child
"""
    assert _captured(src, "god") == {"used"}


def test_nested_counts_direct_vs_total() -> None:
    src = """
def god():
    def child():
        def grand():
            pass
        return grand
    def sibling():
        pass
    return child
"""
    metrics = arch_measure._function_metrics(_fn(src, "god"))
    assert metrics["direct_nested_funcs"] == 2  # child, sibling
    assert metrics["total_nested_funcs"] == 3  # child, sibling, grand


def test_grandchild_read_of_unshadowed_god_name_is_captured() -> None:
    src = """
def god():
    deep = 1
    def child():
        def grand():
            return deep
        return grand
    return child
"""
    # No intermediate rebinding of deep -> the grandchild capture reaches god.
    assert _captured(src, "god") == {"deep"}


# --------------------------------------------------------------------------- #
# nonlocal / global handling.
# --------------------------------------------------------------------------- #


def test_own_nonlocal_names_reported_and_excluded_from_bound() -> None:
    src = """
def outer():
    state = {}
    def measured():
        nonlocal state
        state["x"] = 1
        local_only = 2
        return local_only
    return measured
"""
    node = _fn(src, "measured")
    metrics = arch_measure._function_metrics(node)
    assert metrics["own_nonlocal_names"] == ["state"]
    # state is nonlocal (binds in outer), so it is NOT an own binding here;
    # local_only is.
    assert _bound(src, "measured") == {"local_only"}


def test_global_declared_names_are_not_own_bindings() -> None:
    src = """
def god():
    global counter
    counter = 5
    kept = 1
    return kept
"""
    assert _bound(src, "god") == {"kept"}


# --------------------------------------------------------------------------- #
# Nested-scope bindings never leak up into the god scope.
# --------------------------------------------------------------------------- #


def test_nested_function_locals_do_not_leak_into_parent_bound() -> None:
    src = """
def god(param):
    top = 1
    def child(carg):
        inner = 2
        return carg + inner
    return child
"""
    # child's params/locals (carg, inner) must not appear in god's own bound.
    assert _bound(src, "god") == {"param", "top", "child"}


# --------------------------------------------------------------------------- #
# ClassDef: the class name binds in the enclosing scope; class scope is
# transparent to method free-variable resolution (LEGB skips class scope).
# --------------------------------------------------------------------------- #


def test_classdef_binds_name_in_enclosing_function_scope() -> None:
    # Codex counterexample #2: ``class Child:`` binds Child in the function
    # scope; the class body's own bindings (attr) do NOT leak up.
    src = """
def god():
    class Child:
        attr = 1
        def method(self):
            pass
    return Child
"""
    assert _bound(src, "god") == {"Child"}


def test_class_bases_and_decorators_read_in_enclosing_scope() -> None:
    src = """
def god():
    base = object
    meta = type
    @decorate
    class Child(base, metaclass=meta):
        pass
    return Child
"""
    # base/meta reads happen in god itself (own loads), so they are not
    # "captured" — but Child must be bound and the class body must not leak.
    assert _bound(src, "god") == {"base", "meta", "Child"}


def test_method_captures_god_name_through_transparent_class_scope() -> None:
    src = """
def god():
    cfg = 1
    class Child:
        def method(self):
            return cfg
    return Child
"""
    # The method reads cfg; class scope is transparent, so it resolves to god.
    assert _captured(src, "god") == {"cfg"}


def test_class_body_binding_does_not_shadow_method_capture() -> None:
    src = """
def god():
    val = 1
    class Child:
        val = 2
        def method(self):
            return val
    return Child
"""
    # Python LEGB skips the class scope for methods: method's ``val`` resolves to
    # god.val, NOT the class attribute. The class-scope binding must not shadow.
    assert _captured(src, "god") == {"val"}


def test_class_body_read_of_enclosing_name_is_excluded_by_definition() -> None:
    # Codex final narrowing: ``class C: y = x`` DOES access the enclosing
    # function's x at class-definition time (LOAD_CLASSDEREF). This metric is
    # scoped to CALLABLE closure captures, so that class-body read is deliberately
    # NOT counted — a class body is not a callable being extracted to a module
    # function. (A method reading x still counts; see the test above.)
    src = """
def god():
    x = 1
    class C:
        y = x
    return C
"""
    assert _captured(src, "god") == set()
    assert _captured_metric(src, "god") == 0
    # x is still a god binding; only the class-body READ is out of metric scope.
    assert _bound(src, "god") == {"x", "C"}


# --------------------------------------------------------------------------- #
# Default args eval in the enclosing scope; annotations are excluded (PEP 563).
# --------------------------------------------------------------------------- #


def test_default_arg_evaluated_in_enclosing_scope_not_child() -> None:
    src = """
def god():
    base = 10
    def child(x=base):
        return x
    return child
"""
    # base is read while evaluating child's default — that read happens in god
    # itself, so it is god's own load, not a capture BY child.
    assert _captured(src, "god") == set()
    assert _bound(src, "god") == {"base", "child"}


def test_default_arg_of_grandchild_is_captured_by_intermediate() -> None:
    src = """
def god():
    g = 1
    def child():
        def grand(x=g):
            return x
        return grand
    return child
"""
    # grand's default ``g`` is evaluated in CHILD scope (grand's enclosing), so
    # child reads g and it resolves back to god -> captured.
    assert _captured(src, "god") == {"g"}


def test_argument_annotations_are_not_counted_pep563() -> None:
    src = """
def god():
    T = int
    def child(x: T) -> T:
        return x
    return child
"""
    # Under ``from __future__ import annotations`` T is never evaluated, so the
    # tool deliberately does NOT attribute the annotation as a read/capture.
    assert _captured(src, "god") == set()
    assert _bound(src, "god") == {"T", "child"}


# --- baseline provenance -----------------------------------------------------
#
# The ratchet drifted 166 commits before CI noticed, because ``--emit`` used to
# overwrite the file leaving no trace of what had been accepted. ``--reason``
# was added for that, and then the very next emit demonstrated the remaining
# hole: a first emit recorded +2,641 LOC across 12 files, a follow-up emit
# eleven lines later replaced it, and the file went on to claim the whole batch
# had grown the package by 11 lines. History has to accumulate, not replace.


def _emit(tmp_path, monkeypatch, *, locs, reason, with_reason=True):
    """Run ``--emit`` over a synthetic measurement, and return (rc, file)."""

    payload = {
        "tool_version": 1,
        "tool_sha256": "deadbeef",
        "shim_import_paths": [],
        "functions": {},
        "files": {name: {"loc": loc} for name, loc in locs.items()},
    }
    monkeypatch.setattr(arch_measure, "measure", lambda: copy.deepcopy(payload))
    out = tmp_path / "baseline.json"
    argv = ["arch_measure.py", "--emit", str(out)]
    if with_reason:
        argv += ["--reason", reason]
    monkeypatch.setattr(sys, "argv", argv)
    return arch_measure.main(), out


def test_emit_without_a_reason_writes_nothing(tmp_path, monkeypatch) -> None:
    rc, out = _emit(
        tmp_path, monkeypatch, locs={"a.py": 10}, reason="", with_reason=False
    )

    assert rc == 2
    assert not out.exists(), "a refused emit must not leave a half-moved ratchet"


def test_consecutive_emits_keep_every_move_in_the_history(
    tmp_path, monkeypatch
) -> None:
    """The exact sequence that erased the real record."""

    _emit(tmp_path, monkeypatch, locs={"a.py": 100, "b.py": 50}, reason="first move")
    rc, out = _emit(
        tmp_path,
        monkeypatch,
        locs={"a.py": 2_741, "b.py": 50},
        reason="second move",
    )
    assert rc == 0

    recorded = json.loads(out.read_text(encoding="utf-8"))
    history = recorded["baseline_history"]

    assert [item["reason"] for item in history] == ["first move", "second move"]
    # The growth the first move accepted is still legible after the second.
    assert history[1]["accepted_growth"] == {"a.py": {"loc_was": 100, "loc_now": 2_741}}
    # And the top-level keys still describe only the latest move, which is why
    # they cannot be the record on their own.
    assert recorded["baseline_reason"] == "second move"
    assert recorded["baseline_accepted_growth"] == history[1]["accepted_growth"]


def test_a_third_emit_appends_rather_than_truncating(tmp_path, monkeypatch) -> None:
    for step, loc in enumerate((100, 200, 300), start=1):
        rc, out = _emit(
            tmp_path, monkeypatch, locs={"a.py": loc}, reason=f"move {step}"
        )
        assert rc == 0

    history = json.loads(out.read_text(encoding="utf-8"))["baseline_history"]
    assert [item["reason"] for item in history] == ["move 1", "move 2", "move 3"]
    lifetime = sum(
        v["loc_now"] - v["loc_was"]
        for item in history
        for v in (item.get("accepted_growth") or {}).values()
    )
    assert lifetime == 200, "the whole accepted growth, not just the last step"


def test_the_history_keys_do_not_reach_the_gate(tmp_path, monkeypatch) -> None:
    """Provenance must not be able to change a pass/fail verdict.

    ``diff`` reads only ``functions`` and ``files``. Pinning that here keeps a
    later reader from moving a metric into the history block and quietly
    exempting it.
    """

    _emit(tmp_path, monkeypatch, locs={"a.py": 100}, reason="first move")
    _, out = _emit(tmp_path, monkeypatch, locs={"a.py": 100}, reason="second move")
    baseline = json.loads(out.read_text(encoding="utf-8"))

    assert baseline["baseline_history"], "precondition: history is populated"
    unchanged = {"functions": {}, "files": {"a.py": {"loc": 100}}}
    grown = {"functions": {}, "files": {"a.py": {"loc": 101}}}
    assert arch_measure.diff(baseline, unchanged) == 0
    assert arch_measure.diff(baseline, grown) == 1
