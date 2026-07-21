"""Pure AST proofs for numeric-coercion guard control flow."""

from __future__ import annotations

import ast
from typing import Optional

_FAILURE_SUPPRESSING_OWNERS = (ast.Try, ast.TryStar, ast.With, ast.AsyncWith)
_OUTPUT_WRITER_NAMES = frozenset(
    {
        "dump",
        "savefig",
        "save_publication_figure",
        "to_csv",
        "to_excel",
        "to_json",
        "to_parquet",
        "write_bytes",
        "write_text",
    }
)


def _subscript_key(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _subscript_base_and_key(node: ast.AST) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None, None
    return node.value.id, _subscript_key(node.slice)


def audit_record_assignment_for_count(
    statement: ast.stmt,
    *,
    count_name: str,
) -> bool:
    """Prove one intervening assignment only records a loss-count audit row."""

    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1:
            return False
        target = statement.targets[0]
        value = statement.value
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        target = statement.target
        value = statement.value
    else:
        return False
    if not (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and isinstance(target.slice, (ast.Name, ast.Constant))
        and isinstance(value, ast.Dict)
    ):
        return False
    if not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == count_name
        for node in ast.walk(value)
    ):
        return False
    allowed_calls = {"int", "float", "len", "isna", "isnull", "sum"}
    for node in ast.walk(value):
        if isinstance(node, (ast.Lambda, ast.NamedExpr, ast.Yield, ast.YieldFrom)):
            return False
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        else:
            return False
        if call_name not in allowed_calls:
            return False
    return True


def _summary_assignment(
    statement: ast.stmt,
) -> tuple[Optional[str], Optional[str], Optional[ast.AST]]:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None, None, None
    base, key = _subscript_base_and_key(statement.targets[0])
    return base, key, statement.value


def _handler_clears_output_claims(handler: ast.ExceptHandler) -> Optional[str]:
    assignments: dict[tuple[str, str], ast.AST] = {}
    for statement in handler.body:
        base, key, value = _summary_assignment(statement)
        if base is not None and key is not None and value is not None:
            assignments[(base, key)] = value
            continue
        for call in (
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ):
            if isinstance(call.func, ast.Name) and call.func.id == "str":
                continue
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(call.func.value, ast.Subscript)
                and _subscript_key(call.func.value.slice) == "errors"
            ):
                continue
            return None
    candidate_bases = {base for base, key in assignments if key == "status"}
    for base in candidate_bases:
        status = assignments.get((base, "status"))
        figures = assignments.get((base, "figure_files"))
        outputs = assignments.get((base, "output_files"))
        if (
            isinstance(status, ast.Constant)
            and status.value == "failed"
            and isinstance(figures, ast.List)
            and not figures.elts
            and isinstance(outputs, ast.Dict)
            and not outputs.keys
        ):
            return base
    return None


def _finally_persists_summary(try_node: ast.Try | ast.TryStar, base: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dump"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == base
        for statement in try_node.finalbody
        for node in ast.walk(statement)
    )


def _is_terminal_statement(
    statement: ast.stmt,
    *,
    scope: ast.AST,
    parents: dict[int, ast.AST],
) -> bool:
    parent = parents.get(id(statement))
    if parent is not scope:
        return False
    return any(
        isinstance(value, list)
        and value
        and value[-1] is statement
        and all(isinstance(item, ast.stmt) for item in value)
        for _, value in ast.iter_fields(parent)
    )


def _published_before_guard(
    try_node: ast.Try | ast.TryStar,
    *,
    guard_line: int,
) -> bool:
    for statement in try_node.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call) or int(node.lineno) >= guard_line:
                continue
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                continue
            if call_name in _OUTPUT_WRITER_NAMES:
                return True
    return False


def _terminal_output_failure_summary(
    guard: ast.stmt,
    try_node: ast.Try | ast.TryStar,
    *,
    scope: ast.AST,
    parents: dict[int, ast.AST],
) -> bool:
    if not try_node.handlers or not _is_terminal_statement(
        try_node, scope=scope, parents=parents
    ):
        return False
    bases = [_handler_clears_output_claims(handler) for handler in try_node.handlers]
    if not bases or any(base is None for base in bases) or len(set(bases)) != 1:
        return False
    base = bases[0]
    assert base is not None
    return bool(
        _finally_persists_summary(try_node, base)
        and not _published_before_guard(try_node, guard_line=int(guard.lineno))
    )


def guard_failure_is_terminal(
    guard: ast.stmt,
    *,
    scope: ast.AST,
    parents: dict[int, ast.AST],
) -> bool:
    """Prove a guard either escapes or records a terminal output failure."""

    current: Optional[ast.AST] = guard
    while current is not scope:
        parent = parents.get(id(current)) if current is not None else None
        if parent is None:
            return False
        if isinstance(parent, (ast.Try, ast.TryStar)):
            return _terminal_output_failure_summary(
                guard,
                parent,
                scope=scope,
                parents=parents,
            )
        if isinstance(parent, (ast.With, ast.AsyncWith)):
            return False
        current = parent
    return True


__all__ = ["audit_record_assignment_for_count", "guard_failure_is_terminal"]
