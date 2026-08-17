"""Runtime repair for a typed strict-numeric result used as raw values."""

from __future__ import annotations

import ast

_STRICT_RESULT_TYPE_ERROR = (
    "float() argument must be a string or a real number, "
    "not 'StrictNumericInput'"
)
_NUMERIC_CONSTRUCTORS = {
    ("np", "array"),
    ("np", "asarray"),
    ("pd", "DataFrame"),
    ("pd", "Series"),
    ("pd", "to_numeric"),
}


def _call_name(node: ast.Call) -> tuple[str, str] | None:
    function = node.func
    if (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
    ):
        return function.value.id, function.attr
    return None


def _owner(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    current: ast.AST | None = node
    while current in parents:
        current = parents[current]
        if isinstance(
            current,
            (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda),
        ):
            return current
    return None


def _numeric_sink_uses_name(
    tree: ast.Module,
    *,
    name: str,
    assignment: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    assignment_owner = _owner(assignment, parents)
    for call in ast.walk(tree):
        if (
            not isinstance(call, ast.Call)
            or _call_name(call) not in _NUMERIC_CONSTRUCTORS
            or _owner(call, parents) is not assignment_owner
        ):
            continue
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
            if any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == name
                for node in ast.walk(argument)
            ):
                return True
    return False


def _assignment_value(statement: ast.AST) -> ast.expr | None:
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return statement.value
    return None


def _single_name_target(statement: ast.AST) -> str | None:
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    else:
        return None
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return None
    return targets[0].id


def _is_projection_passthrough(value: ast.expr, *, source_name: str) -> bool:
    """Recognize only wrappers that preserve the typed result as their input."""

    if isinstance(value, ast.Name):
        return value.id == source_name
    if not isinstance(value, ast.Call):
        return False
    if (
        isinstance(value.func, ast.Attribute)
        and value.func.attr == "copy"
        and isinstance(value.func.value, ast.Name)
    ):
        return value.func.value.id == source_name and not value.args
    if _call_name(value) != ("pd", "Series") or not value.args:
        return False
    first_argument = value.args[0]
    return isinstance(first_argument, ast.Name) and first_argument.id == source_name


def _numeric_sink_uses_projection_alias(
    tree: ast.Module,
    *,
    name: str,
    assignment: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Follow one fail-closed alias layer into a numeric constructor.

    Generated code sometimes branches on the typed wrapper before assigning it
    to one Series-shaped alias.  Accept that shape only when every definition
    of the alias in the same function is a known projection passthrough.  An
    arbitrary transform or mixed-source reassignment remains ambiguous.
    """

    assignment_owner = _owner(assignment, parents)
    definitions: dict[str, list[ast.AST]] = {}
    for statement in ast.walk(tree):
        if _owner(statement, parents) is not assignment_owner:
            continue
        target_name = _single_name_target(statement)
        if target_name is not None:
            definitions.setdefault(target_name, []).append(statement)

    for alias, statements in definitions.items():
        if not statements or not all(
            (value := _assignment_value(statement)) is not None
            and _is_projection_passthrough(value, source_name=name)
            for statement in statements
        ):
            continue
        if _numeric_sink_uses_name(
            tree,
            name=alias,
            assignment=assignment,
            parents=parents,
        ):
            return True
    return False


def _replace_call(code: str, call: ast.Call) -> str | None:
    if call.end_lineno is None or call.end_col_offset is None:
        return None
    lines = code.splitlines(keepends=True)

    def offset(line_number: int, byte_column: int) -> int:
        line = lines[line_number - 1]
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        return sum(len(part) for part in lines[: line_number - 1]) + character_column

    end = offset(call.end_lineno, call.end_col_offset)
    return code[:end] + ".values" + code[end:]


def patch_strict_numeric_input_result_projection(code: str, run_log: str) -> str:
    """Project ``.values`` when one typed result reached a numeric constructor.

    The repair is authorized by the exact runtime type error and a single,
    attributable AST flow from ``strict_numeric_input`` into a known NumPy or
    pandas numeric constructor.  Multiple candidates remain an error instead
    of inviting a guess.
    """

    if _STRICT_RESULT_TYPE_ERROR.lower() not in (run_log or "").lower():
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
    candidates: list[ast.Call] = []
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "strict_numeric_input"
        ):
            continue
        parent = parents.get(call)
        if isinstance(parent, ast.Attribute) and parent.attr == "values":
            continue
        if isinstance(parent, ast.Call) and _call_name(parent) in _NUMERIC_CONSTRUCTORS:
            candidates.append(call)
            continue
        if not (
            isinstance(parent, (ast.Assign, ast.AnnAssign))
            and parent.value is call
        ):
            continue
        targets = list(parent.targets) if isinstance(parent, ast.Assign) else [parent.target]
        if (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and (
                _numeric_sink_uses_name(
                    tree,
                    name=targets[0].id,
                    assignment=parent,
                    parents=parents,
                )
                or _numeric_sink_uses_projection_alias(
                    tree,
                    name=targets[0].id,
                    assignment=parent,
                    parents=parents,
                )
            )
        ):
            candidates.append(call)
    if len(candidates) != 1:
        return code
    repaired = _replace_call(code, candidates[0])
    if repaired is None:
        return code
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_strict_numeric_input_result_projection"]
