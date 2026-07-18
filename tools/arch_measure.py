#!/usr/bin/env python3
"""Deterministic architecture-decomposition metrics for the research_agent engine.

Codex governance requirement (2026-07-17): every god-function decomposition batch
must report NUMBERS, not "extracted a seam". Track A extracted five seams yet the
god function shrank 6,859 -> 6,693 (2.5%) — because the numbers were never gated.

This tool is the shared, re-runnable measurement harness. It measures ONLY what
it can measure exactly and precisely; it does not pretend to measure things it
does not (perf is measured by the real E3 run, not here).

What it measures
----------------
* per-function OWN-SCOPE metrics, computed from a lexical scope model of the
  CALLABLE scopes (fn + every nested def/async def/lambda; class bodies and
  comprehensions are modelled only where they affect bindings, not as callable
  closures) — a name "belongs to" a scope only if it is bound/read directly
  there, never in a nested scope inside it:
    - lines                = end_lineno - lineno + 1   (inclusive; no off-by-one)
    - direct_nested_funcs  = functions defined directly in fn.body
    - total_nested_funcs   = functions defined anywhere inside fn (complexity)
    - own_bound_names      = distinct names bound in fn's OWN scope: params +
      assign/annassign/augassign targets + for/with/except-as targets + import
      names + walrus targets + nested class/def names; excludes nonlocal/global-
      declared names and comprehension loop targets (Py3 comprehension scope,
      but the first generator's iterable IS evaluated in this scope). Does NOT
      descend into nested def/lambda/class/comprehension scopes.
    - own_nonlocal_names   = ``nonlocal`` declared in fn's OWN body
    - callable_closure_captured_names = distinct OWN-scope names captured by a
      nested CALLABLE (def / async def / lambda) as a resolved free variable,
      honouring LEGB shadowing among callable scopes (a name a nested callable
      rebinds is NOT counted). This is the extraction signal: how many god-scope
      locals become explicit closure parameters when a nested callable is pulled
      out to module level. Reads that happen in a nested CLASS BODY itself (the
      ``LOAD_CLASSDEREF`` case, e.g. ``class C: y = x``) are deliberately NOT
      counted — this metric is scoped to callables, not a full Python name
      resolver. Methods inside such a class ARE callables and ARE counted.
* whole-file LOC for the targeted modules;
* module dependency edges: the intra-package modules each target file imports;
* SHA-256 of every measured file AND of this tool itself, so a baseline is
  pinned to exact bytes of both the code and the measuring instrument;
* an optional shim-bypass check: given SHIM_IMPORT_PATHS (populated once a batch
  introduces a compatibility shim), assert the target files' canonical imports do
  not route through the shim.

Out of scope (measured elsewhere, by design)
--------------------------------------------
* runtime performance -> the real E3 Step02 run (1 call / 0 repair / 26.8s).
* whether behaviour is preserved -> characterization + shard regression tests.

Usage
-----
    python tools/arch_measure.py --emit tools/arch_baselines/pipeline_execute.json
    python tools/arch_measure.py --diff tools/arch_baselines/pipeline_execute.json
    python tools/arch_measure.py            # print current metrics as JSON
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

TOOL_VERSION = "1.4.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
RA = REPO_ROOT / "src" / "easyicu" / "research_agent"

# Whole-file LOC + dependency edges. Append-only across batches so baselines stay
# comparable.
TARGET_FILES: List[Path] = [
    RA / "pipeline_execute.py",
    RA / "pipeline.py",
    RA / "pipeline_report.py",
    RA / "authority" / "typed_binding.py",
]
# (file, function name) — first match by name (top-level or nested).
TARGET_FUNCTIONS: List[Tuple[str, str]] = [
    ("pipeline_execute.py", "_execute_one_step"),
    ("pipeline_execute.py", "run_execute_phase"),
]
# Compatibility-shim import paths a future batch may introduce. While empty the
# shim-bypass check is a no-op; populate with e.g. "easyicu.research_agent.pipeline_execute"
# once a seam moves and the OLD path becomes a forwarding shim that canonical code
# must not import.
SHIM_IMPORT_PATHS: Set[str] = set()

_FUNC_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMP_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _find_function(tree: ast.AST, name: str) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _param_names(fn: ast.AST) -> Set[str]:
    if not isinstance(fn, _FUNC_SCOPE_NODES):
        return set()
    args = fn.args
    names = {
        a.arg
        for a in list(getattr(args, "posonlyargs", []))
        + list(args.args)
        + list(args.kwonlyargs)
    }
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _target_store_names(target: ast.AST) -> Set[str]:
    """Names bound by a single assignment/for/comprehension target expression
    (handles tuple / list / starred unpacking)."""
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


class _ScanResult:
    __slots__ = ("bound", "loads", "children", "nonlocal_names", "global_names")

    def __init__(self) -> None:
        self.bound: Set[str] = set()
        self.loads: Set[str] = set()
        self.children: List[ast.AST] = []
        self.nonlocal_names: Set[str] = set()
        self.global_names: Set[str] = set()


def _scan_scope(scope_node: ast.AST) -> _ScanResult:
    """Collect the DIRECT (own-scope) bindings, loads, and child-scope nodes for
    ONE function/lambda scope.

    Own-scope bindings = params + every binding statement that lands in THIS
    scope: Assign / AnnAssign / AugAssign targets, ``for``/``with``/``except as``
    targets, ``import`` names, walrus (``:=``) targets, and ``class``/``def``
    names. The scan does NOT descend into nested ``def``/``async def``/``lambda``
    /``class`` bodies (those become child scopes).

    Py3 comprehension scope is honoured precisely: a comprehension's loop targets
    are local to it and excluded here, EXCEPT the FIRST generator's iterable,
    which is evaluated in THIS (enclosing) scope — so a name read only in that
    outermost iterable is an own-scope read, not comprehension-local. Walrus
    targets inside a comprehension still leak to this scope.

    Decorators, base classes, and default-argument expressions of a nested
    def/class are evaluated eagerly in THIS scope, so they are scanned here.
    Argument/return ANNOTATIONS are deliberately NOT attributed to any scope:
    under ``from __future__ import annotations`` (PEP 563) they are strings and
    never evaluated, so counting them would make the metric depend on that
    import. Names declared ``nonlocal``/``global`` are removed (they bind in
    another scope) and are reported separately.
    """
    res = _ScanResult()
    res.bound |= _param_names(scope_node)

    if isinstance(scope_node, ast.Lambda):
        body: List[ast.AST] = [scope_node.body]
    else:
        body = list(getattr(scope_node, "body", []))

    def handle(node: ast.AST, comp_locals: Set[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            res.bound.add(node.name)  # the def NAME binds in THIS scope
            for dec in node.decorator_list:  # decorators eval in THIS scope
                handle(dec, comp_locals)
            for default in list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]:
                handle(default, comp_locals)  # arg defaults eval in THIS scope
            res.children.append(node)
            return
        if isinstance(node, ast.Lambda):
            for default in list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]:
                handle(default, comp_locals)
            res.children.append(node)
            return
        if isinstance(node, ast.ClassDef):
            res.bound.add(node.name)  # the class NAME binds in THIS scope
            for dec in node.decorator_list:  # decorators eval in THIS scope
                handle(dec, comp_locals)
            for base in node.bases:  # base classes eval in THIS scope
                handle(base, comp_locals)
            for kw in node.keywords:  # e.g. metaclass= eval in THIS scope
                handle(kw.value, comp_locals)
            res.children.append(node)
            return
        if isinstance(node, _COMP_NODES):
            # Py3 comprehension scope. The FIRST generator's iterable is evaluated
            # in the ENCLOSING scope; every later iterable, all conditions, and the
            # element see the progressively-bound comprehension targets.
            running = set(comp_locals)
            for index, gen in enumerate(node.generators):
                handle(gen.iter, comp_locals if index == 0 else running)
                running |= _target_store_names(gen.target)
                for cond in gen.ifs:
                    handle(cond, running)
            if isinstance(node, ast.DictComp):
                handle(node.key, running)
                handle(node.value, running)
            else:
                handle(node.elt, running)
            return
        if isinstance(node, ast.NamedExpr):  # walrus binds in the function scope
            if isinstance(node.target, ast.Name):
                res.bound.add(node.target.id)
            handle(node.value, comp_locals)
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                if node.id not in comp_locals:
                    res.bound.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                if node.id not in comp_locals:
                    res.loads.add(node.id)
            return
        if isinstance(node, ast.ExceptHandler):
            if node.name:  # ``except E as name`` binds ``name``
                res.bound.add(node.name)
            if node.type is not None:
                handle(node.type, comp_locals)
            for stmt in node.body:
                handle(stmt, comp_locals)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                res.bound.add((alias.asname or alias.name).split(".")[0])
            return
        if isinstance(node, ast.Nonlocal):
            res.nonlocal_names.update(node.names)
            return
        if isinstance(node, ast.Global):
            res.global_names.update(node.names)
            return
        for child in ast.iter_child_nodes(node):
            handle(child, comp_locals)

    for stmt in body:
        handle(stmt, set())

    # Names declared nonlocal/global are NOT local bindings of this scope.
    res.bound -= res.nonlocal_names
    res.bound -= res.global_names
    return res


class _Scope:
    __slots__ = ("node", "parent", "children", "bound", "loads", "nonlocal_names", "is_class")

    def __init__(
        self, node: ast.AST, parent: Optional["_Scope"], is_class: bool = False
    ) -> None:
        self.node = node
        self.parent = parent
        self.children: List["_Scope"] = []
        self.bound: Set[str] = set()
        self.loads: Set[str] = set()
        self.nonlocal_names: Set[str] = set()
        self.is_class = is_class


def _build_scope_tree(fn: ast.AST) -> _Scope:
    """Build the lexical scope tree rooted at fn (fn + every nested
    def/async def/lambda/class). Comprehensions are folded into their enclosing
    function scope (see ``_scan_scope``), not modelled as separate nodes."""
    root = _Scope(fn, None)
    stack = [root]
    while stack:
        scope = stack.pop()
        res = _scan_scope(scope.node)
        scope.bound = res.bound
        scope.loads = res.loads
        scope.nonlocal_names = res.nonlocal_names
        for child_node in res.children:
            child = _Scope(
                child_node, scope, is_class=isinstance(child_node, ast.ClassDef)
            )
            scope.children.append(child)
            stack.append(child)
    return root


def _callable_closure_captured(root: _Scope) -> Set[str]:
    """God-scope bound names captured by a nested CALLABLE (def/async/lambda).

    This is a scoped architecture metric, NOT a full Python name resolver. It
    answers "when a nested callable is extracted to module level, how many
    god-scope locals must become explicit closure parameters?".

    LEGB shadowing is honoured among callable scopes: a load of ``n`` in a
    descendant callable resolves to the god scope only if neither that callable
    nor any intermediate callable between it and the god scope rebinds ``n`` —
    so a grandchild's read is not mis-attributed when a child rebinds the name.

    Class bodies are handled correctly for METHOD resolution — Python does not
    let a method see class-body names via LEGB, so a class scope contributes no
    shadowing (its bindings are skipped when accumulating ``ancestor_bound``) and
    methods inside it are walked as normal callables. But a class BODY's own
    reads (the ``LOAD_CLASSDEREF`` case, e.g. ``class C: y = x``) DO access the
    enclosing function's ``x`` at class-definition time; those reads are
    deliberately EXCLUDED from this metric because a class body is not a callable
    being extracted. This is a definitional narrowing, not a claim that the read
    does not happen.
    """
    god = root.bound
    captured: Set[str] = set()

    def walk(scope: _Scope, ancestor_bound: Set[str]) -> None:
        if scope.is_class:
            # Class body: not a callable. Its own reads are out of scope for this
            # metric; its bindings do not shadow for the methods it contains.
            deeper = ancestor_bound
        else:
            for name in scope.loads:
                if name in god and name not in scope.bound and name not in ancestor_bound:
                    captured.add(name)
            deeper = ancestor_bound | scope.bound
        for child in scope.children:
            walk(child, deeper)

    for child in root.children:
        walk(child, frozenset())
    return captured


def _function_metrics(fn: ast.AST) -> Dict[str, Any]:
    direct_nested = [n for n in fn.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    total_nested = [
        x
        for x in ast.walk(fn)
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x is not fn
    ]
    root = _build_scope_tree(fn)
    captured = _callable_closure_captured(root)
    return {
        "lines": fn.end_lineno - fn.lineno + 1,
        "line_start": fn.lineno,
        "direct_nested_funcs": len(direct_nested),
        "total_nested_funcs": len(total_nested),
        "own_bound_names": len(root.bound),
        "own_nonlocal_names": sorted(root.nonlocal_names),
        "own_nonlocal_count": len(root.nonlocal_names),
        "callable_closure_captured_names": len(captured),
    }


def _intra_package_imports(tree: ast.AST) -> List[str]:
    """Relative + absolute intra-easyicu module names this file imports."""
    edges: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                edges.add("." * node.level + node.module)
            elif node.module and node.module.startswith("easyicu"):
                edges.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("easyicu"):
                    edges.add(alias.name)
    return sorted(edges)


def _shim_bypass_violations(tree: ast.AST) -> List[str]:
    if not SHIM_IMPORT_PATHS:
        return []
    bad: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in SHIM_IMPORT_PATHS:
            bad.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in SHIM_IMPORT_PATHS:
                    bad.append(alias.name)
    return bad


def measure() -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for p in TARGET_FILES:
        if not p.exists():
            files[p.name] = {"missing": True}
            continue
        tree = ast.parse(p.read_text())
        files[p.name] = {
            "loc": len(p.read_text().splitlines()),
            "sha256": _sha256(p),
            "intra_package_import_edges": len(_intra_package_imports(tree)),
            "shim_bypass_violations": _shim_bypass_violations(tree),
        }
    functions: Dict[str, Any] = {}
    for fname, funcname in TARGET_FUNCTIONS:
        p = RA / fname
        key = f"{fname}::{funcname}"
        if not p.exists():
            functions[key] = {"missing": True}
            continue
        fn = _find_function(ast.parse(p.read_text()), funcname)
        functions[key] = {"missing": True} if fn is None else _function_metrics(fn)
    return {
        "tool_version": TOOL_VERSION,
        "tool_sha256": _sha256(Path(__file__)),
        "shim_import_paths": sorted(SHIM_IMPORT_PATHS),
        "files": files,
        "functions": functions,
    }


# Function metrics where SMALLER is better (a batch must not regress these).
_LOWER_IS_BETTER_FUNC = (
    "lines",
    "direct_nested_funcs",
    "total_nested_funcs",
    "own_bound_names",
    "own_nonlocal_count",
    "callable_closure_captured_names",
)


def diff(baseline: Dict[str, Any], current: Dict[str, Any]) -> int:
    regressions = 0
    print(
        f"tool baseline v{baseline.get('tool_version')} sha {str(baseline.get('tool_sha256'))[:12]}"
        f"  |  current v{current.get('tool_version')} sha {str(current.get('tool_sha256'))[:12]}"
    )
    if baseline.get("tool_sha256") != current.get("tool_sha256"):
        print("  NOTE: measuring tool changed since baseline — re-emit baseline if metric")
        print("        SEMANTICS changed (not just target code).")
    print()
    print(f"{'metric':<52} {'baseline':>10} {'current':>10} {'delta':>8}")
    print("-" * 84)
    for key, cur in current.get("functions", {}).items():
        base = baseline.get("functions", {}).get(key, {})
        for m in _LOWER_IS_BETTER_FUNC:
            b, c = base.get(m), cur.get(m)
            if b is None or c is None:
                continue
            delta = c - b
            flag = "  <-- REGRESSED" if delta > 0 else ("  improved" if delta < 0 else "")
            if delta > 0:
                regressions += 1
            print(f"{key + '.' + m:<52} {b:>10} {c:>10} {delta:>+8}{flag}")
    print()
    for name, cur in current.get("files", {}).items():
        for m in ("loc", "intra_package_import_edges"):
            b = baseline.get("files", {}).get(name, {}).get(m)
            c = cur.get(m)
            if b is None or c is None:
                continue
            print(f"{name + '.' + m:<52} {b:>10} {c:>10} {c - b:>+8}")
        for v in cur.get("shim_bypass_violations", []):
            print(f"  SHIM-BYPASS VIOLATION in {name}: imports shim {v}")
            regressions += 1
    print()
    if regressions:
        print(f"FAIL: {regressions} regression(s) vs baseline.")
    else:
        print("OK: no lower-is-better metric regressed vs baseline.")
    return regressions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", metavar="PATH", help="write current metrics as a baseline JSON")
    ap.add_argument("--diff", metavar="PATH", help="diff current metrics against a baseline JSON")
    args = ap.parse_args()

    current = measure()

    if args.emit:
        out = Path(args.emit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"wrote baseline -> {out}")
        return 0
    if args.diff:
        baseline = json.loads(Path(args.diff).read_text(encoding="utf-8"))
        return 1 if diff(baseline, current) else 0
    print(json.dumps(current, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
