"""Deterministic source repair for a proven numeric-coercion loss count.

This module owns only the syntax-preserving guard insertion.  It does not
choose variables, domains, missing-data policy, or scientific semantics.
"""

from __future__ import annotations

import ast
from typing import Optional

_GUARD_SENTINEL = "_easyicu_lossy_numeric_coercion_guard_v1"
_RETURN_GUARD_SENTINEL = "_easyicu_returned_coercion_loss_guard_v1"


def _expression_key(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _call_tail(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _lexical_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.AST:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(
            current,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            return current
    return current


def _exact_loss_count_call(
    value: ast.AST,
    *,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> Optional[ast.Call]:
    """Return an exact loss count tied to its numeric-coercion definition."""

    candidate = value
    if (
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "int"
        and len(candidate.args) == 1
        and not candidate.keywords
    ):
        candidate = candidate.args[0]
    if not (
        isinstance(candidate, ast.Call)
        and not candidate.args
        and not candidate.keywords
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == "sum"
        and isinstance(candidate.func.value, ast.BinOp)
        and isinstance(candidate.func.value.op, ast.BitAnd)
    ):
        return None
    terms = (candidate.func.value.left, candidate.func.value.right)

    def _method_name(node: ast.AST) -> str:
        if not (
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and isinstance(node.func, ast.Attribute)
        ):
            return ""
        return node.func.attr

    present_terms = [
        term for term in terms if _method_name(term) in {"notna", "notnull"}
    ]
    missing_terms = [term for term in terms if _method_name(term) in {"isna", "isnull"}]
    if len(present_terms) != 1 or len(missing_terms) != 1:
        return None
    present_call = present_terms[0]
    missing_call = missing_terms[0]
    assert isinstance(present_call, ast.Call)
    assert isinstance(present_call.func, ast.Attribute)
    assert isinstance(missing_call, ast.Call)
    assert isinstance(missing_call.func, ast.Attribute)
    if not isinstance(missing_call.func.value, ast.Name):
        return None
    coerced_name = missing_call.func.value.id
    raw_identity = _expression_key(present_call.func.value)
    raw_names = {
        node.id
        for node in ast.walk(present_call.func.value)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    candidate_scope = _lexical_scope(candidate, parents)
    matching_coercions = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == coerced_name
            and isinstance(node.value, ast.Call)
            and _call_tail(node.value.func) == "to_numeric"
            and node.value.args
            and any(
                keyword.arg == "errors"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "coerce"
                for keyword in node.value.keywords
            )
            and _expression_key(node.value.args[0]) == raw_identity
            and _lexical_scope(node, parents) is candidate_scope
            and int(node.lineno) < int(candidate.lineno)
        ):
            continue
        matching_coercions.append(node)
    if len(matching_coercions) != 1:
        return None
    coercion = matching_coercions[0]
    protected_names = {coerced_name, *raw_names}
    for node in ast.walk(candidate_scope):
        if not isinstance(
            node,
            (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr),
        ):
            continue
        if _lexical_scope(node, parents) is not candidate_scope:
            continue
        if not (int(coercion.lineno) < int(node.lineno) < int(candidate.lineno)):
            continue
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        else:
            targets.append(node.target)
        assigned_names = {
            child.id
            for target in targets
            for child in ast.walk(target)
            if isinstance(child, ast.Name)
        }
        if assigned_names & protected_names:
            return None
    return candidate


def _standalone_statement_source(
    code: str,
    statement: ast.Assign | ast.AnnAssign,
    *,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> Optional[tuple[list[str], str]]:
    if statement.end_lineno is None or statement.lineno < 1:
        return None
    lines = code.splitlines(keepends=True)
    if statement.end_lineno > len(lines):
        return None
    for other in ast.walk(tree):
        if not isinstance(other, ast.stmt) or other is statement:
            continue
        if int(statement.lineno) <= int(other.lineno) <= int(statement.end_lineno):
            return None
    current: Optional[ast.AST] = statement
    while current is not None and current in parents:
        current = parents[current]
        if isinstance(current, (ast.Try, ast.TryStar, ast.With, ast.AsyncWith)):
            return None
    start_line = lines[statement.lineno - 1]
    indent = start_line[: len(start_line) - len(start_line.lstrip(" \t"))]
    return lines, indent


def patch_lossy_numeric_coercion_guard(
    code: str,
    *,
    finding_lines: frozenset[int],
) -> str:
    """Insert a guard for one structurally proven scalar or dict loss count."""

    if _GUARD_SENTINEL in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not finding_lines:
        return code

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    from ..gates.preflight import _builtin_int_binding_is_unmodified

    builtin_int_is_safe = _builtin_int_binding_is_unmodified(tree)
    candidates: list[tuple[ast.Assign | ast.AnnAssign, str, Optional[str]]] = []
    for node in ast.walk(tree):
        target: Optional[ast.AST] = None
        value: Optional[ast.AST] = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        scalar_loss_call = _exact_loss_count_call(
            value,
            tree=tree,
            parents=parents,
        )
        if (
            scalar_loss_call is not None
            and int(scalar_loss_call.lineno) in finding_lines
            and not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "int"
                and not builtin_int_is_safe
            )
        ):
            candidates.append((node, target.id, None))
            continue
        if not isinstance(value, ast.Dict):
            continue
        if any(key is None for key in value.keys) or any(
            not isinstance(key, ast.Constant) for key in value.keys
        ):
            continue
        matching_entries: list[tuple[str, ast.AST]] = []
        for key, candidate_value in zip(value.keys, value.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            loss_call = _exact_loss_count_call(
                candidate_value,
                tree=tree,
                parents=parents,
            )
            if loss_call is None or int(loss_call.lineno) not in finding_lines:
                continue
            matching_entries.append((key.value, candidate_value))
        if len(matching_entries) != 1:
            continue
        count_key, loss_value = matching_entries[0]
        if (
            sum(
                1
                for key in value.keys
                if isinstance(key, ast.Constant) and key.value == count_key
            )
            != 1
        ):
            continue
        if (
            isinstance(loss_value, ast.Call)
            and isinstance(loss_value.func, ast.Name)
            and loss_value.func.id == "int"
            and not builtin_int_is_safe
        ):
            continue
        candidates.append((node, target.id, count_key))
    if len(candidates) != 1:
        return code

    assignment, record_name, count_key = candidates[0]
    standalone = _standalone_statement_source(
        code,
        assignment,
        tree=tree,
        parents=parents,
    )
    if standalone is None or assignment.end_lineno is None:
        return code
    lines, indent = standalone
    body_indent = indent + ("\t" if "\t" in indent else "    ")
    if not lines[assignment.end_lineno - 1].endswith(("\n", "\r")):
        lines[assignment.end_lineno - 1] += "\n"
    count_expression = (
        record_name if count_key is None else f"{record_name}[{count_key!r}]"
    )
    guard = (
        f"{indent}# {_GUARD_SENTINEL}\n"
        f"{indent}if {count_expression} > 0:\n"
        f'{body_indent}raise ValueError("numeric coercion invalidated observed '
        'non-missing values")\n'
    )
    lines.insert(assignment.end_lineno, guard)
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _returned_loss_count(
    node: ast.AST,
    *,
    raw_name: str,
    coerced_name: str,
) -> bool:
    """Recognise ``int((raw.notna() & coerced.isna()).sum())`` exactly."""

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
        and not node.keywords
    ):
        return False
    reduction = node.args[0]
    if not (
        isinstance(reduction, ast.Call)
        and isinstance(reduction.func, ast.Attribute)
        and reduction.func.attr == "sum"
        and not reduction.args
        and not reduction.keywords
        and isinstance(reduction.func.value, ast.BinOp)
        and isinstance(reduction.func.value.op, ast.BitAnd)
    ):
        return False

    def _method_call(candidate: ast.AST, owner: str, method: str) -> bool:
        return bool(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == method
            and isinstance(candidate.func.value, ast.Name)
            and candidate.func.value.id == owner
            and not candidate.args
            and not candidate.keywords
        )

    mask = reduction.func.value
    return _method_call(mask.left, raw_name, "notna") and _method_call(
        mask.right, coerced_name, "isna"
    )


def patch_returned_coercion_loss_guard(code: str) -> str:
    """Fail closed when a helper merely returns its numeric coercion losses.

    The repair is authorized separately by a structured concept finding.  It
    only accepts one straight-line helper containing a proven
    ``to_numeric(errors='coerce')`` assignment and returning the exact loss
    count for that same raw/coerced pair.  Reporting the count is retained,
    but a positive count now terminates before the caller can treat those
    values as ordinary missingness.
    """

    if _RETURN_GUARD_SENTINEL in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    if any(
        (
            isinstance(node, ast.Name)
            and node.id == "RuntimeError"
            and isinstance(node.ctx, (ast.Store, ast.Del))
        )
        or (isinstance(node, ast.arg) and node.arg == "RuntimeError")
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == "RuntimeError"
        )
        for node in ast.walk(tree)
    ):
        return code

    candidates: list[tuple[ast.Return, ast.AST, str]] = []
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        assignments = {
            target.id: statement
            for statement in function.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance((target := statement.targets[0]), ast.Name)
        }
        coercions: list[tuple[str, str]] = []
        for coerced_name, statement in assignments.items():
            call = statement.value
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "to_numeric"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and any(
                    keyword.arg == "errors"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "coerce"
                    for keyword in call.keywords
                )
            ):
                continue
            raw_name = call.args[0].id
            if raw_name in assignments:
                coercions.append((raw_name, coerced_name))
        for statement in function.body:
            if not (
                isinstance(statement, ast.Return)
                and isinstance(statement.value, ast.Tuple)
            ):
                continue
            if any(
                isinstance(parent, (ast.Try, ast.TryStar))
                for parent in _ancestor_nodes(statement, parents)
            ):
                continue
            for item in statement.value.elts:
                matching = [
                    (raw_name, coerced_name)
                    for raw_name, coerced_name in coercions
                    if _returned_loss_count(
                        item,
                        raw_name=raw_name,
                        coerced_name=coerced_name,
                    )
                ]
                if len(matching) == 1:
                    candidates.append((statement, item, matching[0][1]))
    if len(candidates) != 1:
        return code
    return_statement, loss_expression, coerced_name = candidates[0]
    if (
        return_statement.end_lineno is None
        or loss_expression.end_lineno is None
        or loss_expression.end_col_offset is None
    ):
        return code

    count_name = f"_easyicu_{coerced_name}_coercion_loss_count_v1"
    if any(
        (isinstance(node, ast.Name) and node.id == count_name)
        or (isinstance(node, ast.arg) and node.arg == count_name)
        for node in ast.walk(tree)
    ):
        return code
    lines = code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    expression_start = offsets[loss_expression.lineno - 1] + loss_expression.col_offset
    expression_end = (
        offsets[loss_expression.end_lineno - 1] + loss_expression.end_col_offset
    )
    return_start = offsets[return_statement.lineno - 1]
    line = lines[return_statement.lineno - 1]
    indent = line[: len(line) - len(line.lstrip(" \t"))]
    body_indent = indent + ("\t" if "\t" in indent else "    ")
    expression_source = code[expression_start:expression_end]
    guard = (
        f"{indent}# {_RETURN_GUARD_SENTINEL}\n"
        f"{indent}{count_name} = {expression_source}\n"
        f"{indent}if {count_name} > 0:\n"
        f'{body_indent}raise RuntimeError("numeric coercion invalidated observed '
        'non-missing values")\n'
    )
    repaired = code[:expression_start] + count_name + code[expression_end:]
    repaired = repaired[:return_start] + guard + repaired[return_start:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _ancestor_nodes(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> list[ast.AST]:
    ancestors: list[ast.AST] = []
    current = node
    while current in parents:
        current = parents[current]
        ancestors.append(current)
    return ancestors


__all__ = [
    "patch_lossy_numeric_coercion_guard",
    "patch_returned_coercion_loss_guard",
]
