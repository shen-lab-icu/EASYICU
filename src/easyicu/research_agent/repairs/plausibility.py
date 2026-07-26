"""Deterministic repairs for typed plausibility-range policy findings."""

from __future__ import annotations

import ast
import re
from typing import Optional, Sequence

from ..schema import ValidationFinding


_PLAUSIBILITY_RANGE_KEY_ERROR = re.compile(
    r"KeyError:\s*(?P<key>0|1|'lower'|'upper'|\"lower\"|\"upper\")",
    re.IGNORECASE,
)


def patch_plausibility_range_schema_keys(code: str, run_log: str) -> str:
    """Adapt legacy/list range access to the sealed minimum/maximum schema.

    The host manifest always represents ``analysis_plausibility_range`` as a
    JSON object with ``minimum`` and ``maximum`` keys. A generated script may
    still guess a two-item sequence or ``lower``/``upper`` aliases. Only an
    exact KeyError plus an AST-proven assignment from that host field permits
    rewriting the key tokens; bound values and policy logic are untouched.
    """

    match = _PLAUSIBILITY_RANGE_KEY_ERROR.search(str(run_log or ""))
    if match is None:
        return code
    raw_key = match.group("key").strip("\"'")
    failed_key: int | str = int(raw_key) if raw_key in {"0", "1"} else raw_key.lower()
    key_map: dict[int | str, str] = {
        0: "minimum",
        1: "maximum",
        "lower": "minimum",
        "upper": "maximum",
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    range_names = {
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "get"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "analysis_plausibility_range"
    }
    candidates: list[tuple[ast.AST, int | str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in range_names
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in key_map
        ):
            continue
        candidates.append((node.slice, node.slice.value))
    if not candidates or failed_key not in {key for _, key in candidates}:
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements = [
        (
            _absolute_offset(node.lineno, node.col_offset),
            _absolute_offset(node.end_lineno, node.end_col_offset),
            repr(key_map[key]),
        )
        for node, key in candidates
    ]
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_flag_only_plausibility_range_rejection(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Remove one auditor-proven hard rejection of a flag-only range.

    The ConceptDescriptor plausibility range is not an exclusion contract.  A
    deterministic repair is permitted only when the typed auditor identifies
    one variable and the script contains exactly one closed range-rejection
    shape.  Supported shapes are a direct adjacent
    ``range-mask = lower | upper`` / ``if range-mask.any(): raise`` pair, or a
    mask/count/terminal-failure chain where the mask and count remain available
    for audit reporting and only the failure guard is removed.  Any ambiguity,
    scientific filtering use, side effect, or non-literal boundary leaves the
    code unchanged for provider repair.
    """

    matching: list[set[str] | None] = []
    for finding in repair_findings:
        detail = finding.detail or {}
        if (
            finding.validator == "llm_concept_auditor"
            and detail.get("issue_code") == "plausibility_range_exclusion_required"
            and detail.get("value_class") == "finite_outside_plausibility_range"
        ):
            raw_variable = detail.get("variable")
            if isinstance(raw_variable, str) and raw_variable.isidentifier():
                matching.append({raw_variable})
            elif (
                isinstance(raw_variable, list)
                and raw_variable
                and all(isinstance(item, str) and item.strip() for item in raw_variable)
            ):
                # The auditor may report a semantic column family while the
                # generated code validates it through one loop-local Series.
                # In that case the structural match below must be globally
                # unique because no local Python name can be inferred safely.
                matching.append(None)
    if not matching:
        return code
    accepted_variables = (
        None if any(item is None for item in matching) else set().union(*matching)
    )

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def numeric_literal(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            )
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and numeric_literal(node.operand)
        )

    def bound_side(node: ast.AST) -> Optional[tuple[str, str]]:
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
        ):
            return None
        left, right = node.left, node.comparators[0]
        operator = node.ops[0]
        if isinstance(left, ast.Name) and numeric_literal(right):
            if isinstance(operator, (ast.Lt, ast.LtE)):
                return "lower", left.id
            if isinstance(operator, (ast.Gt, ast.GtE)):
                return "upper", left.id
        if numeric_literal(left) and isinstance(right, ast.Name):
            if isinstance(operator, (ast.Gt, ast.GtE)):
                return "lower", right.id
            if isinstance(operator, (ast.Lt, ast.LtE)):
                return "upper", right.id
        return None

    def range_variable(node: ast.AST) -> Optional[str]:
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)):
            return None
        bounds = (bound_side(node.left), bound_side(node.right))
        if any(bound is None for bound in bounds):
            return None
        assert bounds[0] is not None and bounds[1] is not None
        if {bounds[0][0], bounds[1][0]} != {"lower", "upper"}:
            return None
        if bounds[0][1] != bounds[1][1]:
            return None
        return bounds[0][1]

    def range_mask_variable(node: ast.AST) -> Optional[str]:
        direct = range_variable(node)
        if direct is not None:
            return direct
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)):
            return None
        for availability, bounds in ((node.left, node.right), (node.right, node.left)):
            variable = range_variable(bounds)
            if (
                variable is not None
                and isinstance(availability, ast.Call)
                and isinstance(availability.func, ast.Attribute)
                and availability.func.attr == "notna"
                and isinstance(availability.func.value, ast.Name)
                and availability.func.value.id == variable
                and not availability.args
                and not availability.keywords
            ):
                return variable
        return None

    semantic_aliases: dict[str, set[str]] = {}
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                continue
            target = node.targets[0].id
            inferred: set[str] = {target}
            value = node.value
            if (
                isinstance(value, ast.Subscript)
                and isinstance(value.slice, ast.Constant)
                and isinstance(value.slice.value, str)
                and value.slice.value.isidentifier()
            ):
                inferred.add(value.slice.value)
            elif isinstance(value, ast.Name):
                inferred.update(semantic_aliases.get(value.id, {value.id}))
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "pd"
                and value.func.attr == "to_numeric"
                and value.args
                and isinstance(value.args[0], ast.Name)
            ):
                source = value.args[0].id
                inferred.update(semantic_aliases.get(source, {source}))
            before = semantic_aliases.get(target, set())
            after = before | inferred
            if after != before:
                semantic_aliases[target] = after
                changed = True
        if not changed:
            break

    def variable_is_authorized(variable: str) -> bool:
        return (
            accepted_variables is None
            or variable in accepted_variables
            or bool(semantic_aliases.get(variable, set()) & accepted_variables)
        )

    terminating_helpers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Raise)
    }

    def is_terminal_failure_body(body: list[ast.stmt]) -> bool:
        if len(body) != 1:
            return False
        statement = body[0]
        if isinstance(statement, ast.Raise):
            return True
        return (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id in terminating_helpers
        )

    def is_raise_guard(node: ast.stmt, mask_name: str) -> bool:
        if not (
            isinstance(node, ast.If)
            and is_terminal_failure_body(node.body)
            and not node.orelse
        ):
            return False
        test = node.test
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "bool"
            and len(test.args) == 1
            and not test.keywords
        ):
            test = test.args[0]
        return (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and test.func.attr == "any"
            and isinstance(test.func.value, ast.Name)
            and test.func.value.id == mask_name
            and not test.args
            and not test.keywords
        )

    def count_assignment_mask(node: ast.stmt, mask_name: str) -> Optional[str]:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "int"
            and len(node.value.args) == 1
            and not node.value.keywords
        ):
            return None
        reduction = node.value.args[0]
        if not (
            isinstance(reduction, ast.Call)
            and isinstance(reduction.func, ast.Attribute)
            and reduction.func.attr == "sum"
            and isinstance(reduction.func.value, ast.Name)
            and reduction.func.value.id == mask_name
            and not reduction.args
            and not reduction.keywords
        ):
            return None
        return node.targets[0].id

    def is_positive_count_guard(node: ast.stmt, count_name: str) -> bool:
        if not (
            isinstance(node, ast.If)
            and not node.orelse
            and is_terminal_failure_body(node.body)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and len(node.test.comparators) == 1
        ):
            return False
        left, right = node.test.left, node.test.comparators[0]
        operator = node.test.ops[0]
        return (
            isinstance(left, ast.Name)
            and left.id == count_name
            and isinstance(right, ast.Constant)
            and right.value == 0
            and isinstance(operator, ast.Gt)
        ) or (
            isinstance(left, ast.Constant)
            and left.value == 0
            and isinstance(right, ast.Name)
            and right.id == count_name
            and isinstance(operator, ast.Lt)
        )

    parent_by_id = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    direct_candidates: list[tuple[ast.Assign, ast.If]] = []
    counted_candidates: list[tuple[ast.Assign, ast.Assign, ast.If]] = []
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(parent, field, None)
            if not isinstance(statements, list):
                continue
            for first, second in zip(statements, statements[1:]):
                range_variable_name = (
                    range_mask_variable(first.value)
                    if isinstance(first, ast.Assign)
                    else None
                )
                if not (
                    isinstance(first, ast.Assign)
                    and len(first.targets) == 1
                    and isinstance(first.targets[0], ast.Name)
                    and range_variable_name is not None
                    and variable_is_authorized(range_variable_name)
                ):
                    continue
                mask_name = first.targets[0].id
                if not is_raise_guard(second, mask_name):
                    continue
                loads = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == mask_name
                ]
                if len(loads) == 1:
                    direct_candidates.append((first, second))
            for first, second, third in zip(statements, statements[1:], statements[2:]):
                range_variable_name = (
                    range_mask_variable(first.value)
                    if isinstance(first, ast.Assign)
                    else None
                )
                if not (
                    isinstance(first, ast.Assign)
                    and len(first.targets) == 1
                    and isinstance(first.targets[0], ast.Name)
                    and range_variable_name is not None
                    and variable_is_authorized(range_variable_name)
                ):
                    continue
                mask_name = first.targets[0].id
                count_name = count_assignment_mask(second, mask_name)
                if count_name is None or not is_positive_count_guard(third, count_name):
                    continue
                mask_loads = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == mask_name
                ]
                if len(mask_loads) != 1 or mask_loads[0] not in set(ast.walk(second)):
                    continue
                guard_nodes = set(ast.walk(third))
                count_loads = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == count_name
                ]
                if not count_loads or any(
                    load not in guard_nodes
                    and not (
                        isinstance(parent_by_id.get(id(load)), ast.Dict)
                        and load in parent_by_id[id(load)].values
                    )
                    for load in count_loads
                ):
                    continue
                counted_candidates.append((first, second, third))
    if len(direct_candidates) + len(counted_candidates) != 1:
        return code
    if direct_candidates:
        assignment, guard = direct_candidates[0]
        replace_start = assignment.lineno
    else:
        _, _, guard = counted_candidates[0]
        replace_start = guard.lineno
    if guard.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    source_line = lines[replace_start - 1]
    indent = source_line[: len(source_line) - len(source_line.lstrip(" \t"))]
    replacement = f"{indent}pass  # _easyicu_flag_only_plausibility_range_retained_v1\n"
    lines[replace_start - 1 : guard.end_lineno] = [replacement]
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = [
    "patch_flag_only_plausibility_range_rejection",
    "patch_plausibility_range_schema_keys",
]
