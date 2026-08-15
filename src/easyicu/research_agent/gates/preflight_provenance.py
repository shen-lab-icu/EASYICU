"""Provenance fail-closed and exposure-authority statics from gates/preflight.py."""

from __future__ import annotations

import ast
from typing import Optional

from ..schema import ValidationFinding
from .ast_semantics import (
    call_name as _call_name,
)

_TRY_STAR_NODE_TYPES = (
    (try_star_type,) if (try_star_type := getattr(ast, "TryStar", None)) else ()
)
_TRY_NODE_TYPES = (ast.Try, *_TRY_STAR_NODE_TYPES)
_TYPE_PARAMETER_NODE_TYPES = tuple(
    filter(
        None,
        (getattr(ast, name, None) for name in ("TypeVar", "ParamSpec", "TypeVarTuple")),
    )
)
_STRUCTURAL_ACCOUNTING_PRODUCTS = frozenset(
    {
        "attrition",
        "cohort_accounting",
        "cohort_flow",
        "denominator_reconciliation",
        "source_availability",
        "source_availability_audit",
        "universe_count_reconciliation",
    }
)
_RENDER_METHODS = frozenset(
    {"figure", "publication_figure", "visualization", "descriptive_visualization"}
)

_PROVENANCE_FAILURE_KEYS = frozenset({"invalid_pair_n", "discordant_n"})
_PROVENANCE_FAILURE_DECISION_KEYS = frozenset({"fail_closed"})
_PROVENANCE_SUCCESS_DECISION_KEYS = frozenset(
    {"completed_step_allowed", "provenance_valid"}
)
_PROVENANCE_FULL_COVERAGE = frozenset(_PROVENANCE_FAILURE_KEYS)
_PROVENANCE_FAILURE = "failure"
_PROVENANCE_SUCCESS = "success"
_PROVENANCE_LOOP_SENTINEL = "_easyicu_provenance_loop_observed"
_FLOW_FALLTHROUGH = "fallthrough"
_FLOW_FUNCTION_EXIT = "function_exit"
_FLOW_LOOP_ESCAPE = "loop_escape"
def _literal_string_tokens(node: ast.AST) -> set[str]:
    return {
        str(candidate.value).strip().lower()
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    }
def _referenced_names(node: ast.AST) -> set[str]:
    return {
        candidate.id for candidate in ast.walk(node) if isinstance(candidate, ast.Name)
    }
_REFLECTION_MODULE_ROOTS = frozenset(
    {"__main__", "gc", "importlib", "inspect", "pkgutil", "pydoc", "unittest"}
)
def _expression_identity(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)
def _mapping_root_name(node: ast.AST) -> Optional[str]:
    current = node
    if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        current = current.func.value
    while isinstance(current, (ast.Subscript, ast.Attribute)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None
def _literal_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and node.value == 0
    )
def _literal_bool(node: ast.AST) -> Optional[bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None
def _swap_provenance_meaning(
    meaning: Optional[tuple[str, frozenset[str]]],
) -> Optional[tuple[str, frozenset[str]]]:
    if meaning is None:
        return None
    kind, coverage = meaning
    inverse = (
        _PROVENANCE_SUCCESS if kind == _PROVENANCE_FAILURE else _PROVENANCE_FAILURE
    )
    return inverse, coverage
def _has_unrelated_control_ancestor(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    current = parents.get(node)
    while current is not None and not isinstance(
        current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
    ):
        if isinstance(current, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            return True
        current = parents.get(current)
    return False
_PROVENANCE_RESULT_SINK_METHODS = frozenset(
    {
        "fit",
        "fit_regularized",
        "predict",
        "dump",
        "save",
        "savefig",
        "savetxt",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_json",
        "to_parquet",
        "to_pickle",
        "write_bytes",
        "write_text",
    }
)
def _is_provenance_result_sink_call(candidate: ast.Call) -> bool:
    call_name = _call_name(candidate.func).lower()
    method = call_name.rsplit(".", 1)[-1]
    return (
        method in _PROVENANCE_RESULT_SINK_METHODS
        or "write_success" in method
        or (method.startswith("write_") and not method.startswith("write_failed"))
        or method.startswith("publish")
    )
def _provenance_branch_contains_result_sink(statements: list[ast.stmt]) -> bool:
    """Return whether an executed guard branch can publish scientific output."""

    found = False

    class _SinkVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            nonlocal found
            if _is_provenance_result_sink_call(node):
                found = True
                return
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

    visitor = _SinkVisitor()
    for statement in statements:
        visitor.visit(statement)
        if found:
            return True
    return False
def _result_sink_precedes_guard(guard: ast.If, parents: dict[ast.AST, ast.AST]) -> bool:
    scope: ast.AST = guard
    while scope in parents and not isinstance(
        scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
    ):
        scope = parents[scope]
    guard_line = int(getattr(guard, "lineno", 0) or 0)
    for candidate in ast.walk(scope):
        if not isinstance(candidate, ast.Call):
            continue
        current: Optional[ast.AST] = candidate
        while current in parents and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            current = parents[current]
        if current is not scope:
            continue
        line = int(getattr(candidate, "lineno", 0) or 0)
        if not line or line >= guard_line:
            continue
        if _is_provenance_result_sink_call(candidate):
            return True
    return False
def _provenance_signal_source(value: ast.AST) -> Optional[tuple[str, str]]:
    current = value
    if (
        isinstance(current, ast.Call)
        and isinstance(current.func, ast.Name)
        and current.func.id in {"bool", "float", "int"}
        and len(current.args) == 1
    ):
        current = current.args[0]
    if isinstance(current, ast.Name):
        return current.id, _PROVENANCE_FAILURE
    if not (
        isinstance(current, ast.Call)
        and isinstance(current.func, ast.Attribute)
        and current.func.attr in {"any", "sum"}
    ):
        return None
    base = current.func.value
    if isinstance(base, ast.Name):
        return base.id, _PROVENANCE_FAILURE
    if (
        isinstance(base, ast.UnaryOp)
        and isinstance(base.op, (ast.Not, ast.Invert))
        and isinstance(base.operand, ast.Name)
    ):
        return base.operand.id, _PROVENANCE_SUCCESS
    return None
def _provenance_pair_scan_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject an explicit measured-only scan that cannot see count-only concepts."""

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tokens = _literal_string_tokens(node)
        if not (_PROVENANCE_FAILURE_KEYS <= tokens and "audit_only" in tokens):
            continue
        scanned_suffixes: set[str] = set()
        for candidate in ast.walk(node):
            if not isinstance(candidate, ast.Call):
                continue
            if _call_name(candidate.func).split(".")[-1] != "endswith":
                continue
            scanned_suffixes.update(
                token
                for token in _literal_string_tokens(candidate)
                if token in {"_measured", "_n"}
            )
        if "_measured" not in scanned_suffixes or "_n" in scanned_suffixes:
            continue
        return [
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "The measurement-provenance audit scans measured columns only "
                    "and cannot fail closed for count-only concepts."
                ),
                detail={"reason": "provenance_pair_scan_not_bidirectional"},
            )
        ]
    return []
def _handler_immediately_reraises(handler: ast.ExceptHandler) -> bool:
    """Prove that a handler immediately propagates the caught exception."""

    return (
        bool(handler.body)
        and isinstance(handler.body[0], ast.Raise)
        and handler.body[0].exc is None
    )
def _statements_call_reconciliation(
    statements: list[ast.stmt],
    *,
    helper_call_names: set[str],
) -> bool:
    """Detect helper calls executed in one lexical scope.

    Function and lambda bodies are deferred code, so merely defining one inside
    a ``try`` must not make that ``try`` look as though it called the helper.
    Simple aliases assigned inside the inspected statements are followed in
    source order as well as by the visible-scope fixed-point alias scan above.
    """

    aliases = set(helper_call_names)
    called = False

    class _CallVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Assign(self, node: ast.Assign) -> None:
            source_name = _call_name(node.value).split(".")[-1]
            if source_name in aliases:
                aliases.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            self.visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is None:
                return
            source_name = _call_name(node.value).split(".")[-1]
            if source_name in aliases and isinstance(node.target, ast.Name):
                aliases.add(node.target.id)
            self.visit(node.value)

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal called
            if _call_name(node.func).split(".")[-1] in aliases:
                called = True
            self.generic_visit(node)

    visitor = _CallVisitor()
    for statement in statements:
        visitor.visit(statement)
    return called
def _finally_exception_suppressor(finalbody: list[ast.stmt]) -> ast.AST | None:
    """Return control flow that can suppress an active ``try`` exception.

    A ``return`` anywhere in the current lexical scope suppresses an exception
    raised by the corresponding ``try``.  A ``break`` or ``continue`` does the
    same only when it targets a loop outside that ``try``; control transfers
    inside a loop created by the ``finally`` suite do not escape the suite and
    therefore do not suppress the exception.
    """

    suppressor: ast.AST | None = None

    class _FinallyVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.loop_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Return(self, node: ast.Return) -> None:
            nonlocal suppressor
            suppressor = suppressor or node

        def visit_Break(self, node: ast.Break) -> None:
            nonlocal suppressor
            if self.loop_depth == 0:
                suppressor = suppressor or node

        def visit_Continue(self, node: ast.Continue) -> None:
            nonlocal suppressor
            if self.loop_depth == 0:
                suppressor = suppressor or node

        def _visit_loop(self, node: ast.AST) -> None:
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        visit_For = _visit_loop
        visit_AsyncFor = _visit_loop
        visit_While = _visit_loop

    visitor = _FinallyVisitor()
    for statement in finalbody:
        visitor.visit(statement)
    return suppressor
