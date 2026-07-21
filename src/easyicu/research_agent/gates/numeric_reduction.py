"""Pure AST proofs for boolean-mask reduction placement.

This module owns syntax recognition only.  It neither selects an analysis nor
applies a repair, so preflight and repair can share one proof without importing
one another's control planes.
"""

from __future__ import annotations

import ast
from typing import Optional

_ARRAY_BOOLEAN_PREDICATE_METHODS = frozenset(
    {"between", "duplicated", "isna", "isin", "notna"}
)


def is_array_boolean_predicate(node: ast.AST) -> bool:
    """Return whether ``node`` is structurally an array boolean predicate."""

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return is_array_boolean_predicate(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in _ARRAY_BOOLEAN_PREDICATE_METHODS
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        return any(
            any(
                isinstance(candidate, ast.Subscript)
                or (
                    isinstance(candidate, ast.Call)
                    and isinstance(candidate.func, ast.Attribute)
                )
                for candidate in ast.walk(operand)
            )
            for operand in operands
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        return is_array_boolean_predicate(node.left) and is_array_boolean_predicate(
            node.right
        )
    return False


def _is_numpy_boolean_predicate(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"isfinite", "isinf", "isnan"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"np", "numpy"}
        and len(node.args) == 1
        and not node.keywords
    )


def unambiguous_boolean_predicate_aliases(tree: ast.Module) -> dict[str, ast.AST]:
    """Return uniquely bound names whose values are proven boolean predicates.

    This deliberately uses a whole-module uniqueness proof.  A name is eligible
    only when it has exactly one ``Store`` occurrence, that store is a simple
    assignment, and the assigned expression is itself structurally proven.  A
    parameter, loop target, rebinding, or assignment in another lexical scope
    therefore makes the optional repair decline rather than guess.
    """

    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"eval", "exec", "globals", "locals", "vars"}
        for node in ast.walk(tree)
    ):
        return {}

    store_counts: dict[str, int] = {}

    def record(name: Optional[str]) -> None:
        if name:
            store_counts[name] = store_counts.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            record(node.id)
        elif isinstance(node, ast.arg):
            record(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            record(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                record(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ExceptHandler):
            record(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            record(node.name)
        elif isinstance(node, ast.MatchMapping):
            record(node.rest)
        elif isinstance(node, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)):
            record(node.name)

    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        name = targets[0].id
        if store_counts.get(name) == 1:
            assignments[name] = node.value

    proven: dict[str, ast.AST] = {}
    pending = dict(assignments)
    changed = True
    while changed:
        changed = False
        for name, value in list(pending.items()):
            if _is_proven_array_boolean_predicate(value, aliases=proven):
                proven[name] = value
                pending.pop(name)
                changed = True
    return proven


def _is_proven_array_boolean_predicate(
    node: ast.AST,
    *,
    aliases: Optional[dict[str, ast.AST]] = None,
) -> bool:
    if isinstance(node, ast.Name) and aliases is not None:
        return node.id in aliases
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return _is_proven_array_boolean_predicate(node.operand, aliases=aliases)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        return _is_proven_array_boolean_predicate(
            node.left, aliases=aliases
        ) and _is_proven_array_boolean_predicate(node.right, aliases=aliases)
    return is_array_boolean_predicate(node) or _is_numpy_boolean_predicate(node)


def misnested_boolean_mask_reduction_expression(
    node: ast.AST,
    *,
    aliases: Optional[dict[str, ast.AST]] = None,
) -> Optional[ast.BinOp]:
    """Return the intended mask when exactly one operand was reduced early."""

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.BinOp)
        and isinstance(node.args[0].op, (ast.BitAnd, ast.BitOr))
    ):
        return None
    operation = node.args[0]

    def strip_reduction(operand: ast.AST) -> tuple[ast.AST, bool]:
        inverted = isinstance(operand, ast.UnaryOp) and isinstance(
            operand.op, ast.Invert
        )
        candidate = operand.operand if inverted else operand
        if not (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == "sum"
            and not candidate.args
            and not candidate.keywords
            and _is_proven_array_boolean_predicate(
                candidate.func.value, aliases=aliases
            )
        ):
            return operand, False
        predicate = candidate.func.value
        if inverted:
            predicate = ast.UnaryOp(op=ast.Invert(), operand=predicate)
        return predicate, True

    left, left_reduced = strip_reduction(operation.left)
    right, right_reduced = strip_reduction(operation.right)
    if left_reduced == right_reduced:
        return None
    if not (
        _is_proven_array_boolean_predicate(left, aliases=aliases)
        and _is_proven_array_boolean_predicate(right, aliases=aliases)
    ):
        return None
    return ast.BinOp(left=left, op=operation.op, right=right)


def patch_misnested_boolean_mask_reduction(code: str) -> Optional[str]:
    """Move a proven operand-local reduction after the combined mask."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    aliases = unambiguous_boolean_predicate_aliases(tree)
    lines = code.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        combined = misnested_boolean_mask_reduction_expression(node, aliases=aliases)
        if combined is None or node.end_lineno is None or node.end_col_offset is None:
            continue
        replacements.append(
            (
                absolute_offset(node.lineno, node.col_offset),
                absolute_offset(node.end_lineno, node.end_col_offset),
                f"int(({ast.unparse(combined)}).sum())",
            )
        )
    if not replacements:
        return None
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired if repaired != code else None


__all__ = [
    "is_array_boolean_predicate",
    "misnested_boolean_mask_reduction_expression",
    "patch_misnested_boolean_mask_reduction",
    "unambiguous_boolean_predicate_aliases",
]
