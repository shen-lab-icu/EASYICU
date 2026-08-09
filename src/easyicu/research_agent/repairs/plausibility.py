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
    shape.  Supported shapes are a mask/count/terminal-failure chain where the
    mask and count remain available for audit reporting and only the failure
    guard is removed, or a unique pair of lower/upper terminal guards whose
    bounds are both read from the sealed ``analysis_plausibility_range``
    minimum/maximum schema, whether the bound is compared directly or narrowed
    once through ``float(...)``.  Any ambiguity, scientific filtering use, or
    side effect leaves the code unchanged for provider repair.

    This repair proves only the retention half of ``retain_and_flag``: it
    deletes a terminal guard, so no row is excluded.  It cannot supply the
    structured flag or count, which belongs to the generated script's declared
    outputs.  A step whose only evidence of policy compliance is this marker
    has satisfied retention, not flagging.

    **The marker is a breadcrumb, not a receipt: nothing reads it today.**  The
    flagging half is gated in ``audits/validators.py``, and that gate binds
    only while the auditor is still reporting the exclusion demand.  In the
    sealed-guard shape the repair neuters the guard bodies, so the script stops
    requesting exclusion, the auditor goes quiet, and the finding the gate
    attaches to never arrives -- leaving a script that has kept every row and
    recorded nothing.  Closing that needs the obligation to be checked from
    this marker (or from the step's declared ``plausibility_policy``) rather
    than from a finding.  Deliberately not done here: it would newly block any
    repaired step that does not flag, which is a scope and cost decision for
    the run's owner, not a side effect of this repair.

    That boundary is why the direct adjacent ``range-mask = lower | upper`` /
    ``if range-mask.any(): raise`` pair is **not** repaired here, though it
    once was.  In that shape the guard is the mask's only reader, so removing
    it removed the mask too, and the script was left computing nothing at all
    about the out-of-range rows: retention satisfied by destroying the record
    of what had been retained.  It also erased the evidence the audit
    downgrade in ``audits/validators.py`` looks for, so the step went on to
    pass having neither excluded nor flagged.  A shape a deterministic patch
    cannot leave better than it found belongs to provider repair, where the
    Coder is told to keep every row and record the count.
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

    plausibility_names = {
        node.targets[0].id
        for node in assignments
        if len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "get"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == "analysis_plausibility_range"
    }
    sealed_bound_names: dict[str, tuple[str, str]] = {}
    for node in assignments:
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id in plausibility_names
            and node.value.func.attr == "get"
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value in {"minimum", "maximum"}
        ):
            continue
        sealed_bound_names[node.targets[0].id] = (
            node.value.func.value.id,
            str(node.value.args[0].value),
        )

    def _non_null_bound_name(node: ast.AST) -> Optional[str]:
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.IsNot)
            and len(node.comparators) == 1
        ):
            return None
        left, right = node.left, node.comparators[0]
        if (
            isinstance(left, ast.Name)
            and isinstance(right, ast.Constant)
            and right.value is None
        ):
            return left.id
        if (
            isinstance(right, ast.Name)
            and isinstance(left, ast.Constant)
            and left.value is None
        ):
            return right.id
        return None

    def _compared_bound_name(node: ast.AST) -> Optional[str]:
        """Read the bound a comparison operand denotes, through one float().

        A host bound arrives as JSON, so a generated script commonly narrows it
        at the comparison itself (``numeric < float(lower)``). That is the same
        bound, not a computed threshold, so matching only a bare ``Name`` left
        this repair unable to fire on the exact shape the auditor blocks.
        Nothing else is unwrapped: a call with keywords, extra arguments, or any
        other callee is not a bound and returns None.
        """

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and len(node.args) == 1
            and not node.keywords
        ):
            node = node.args[0]
        return node.id if isinstance(node, ast.Name) else None

    def _sealed_comparison(
        node: ast.AST,
        *,
        bound_name: str,
        bound_kind: str,
    ) -> Optional[str]:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bool"
            and len(node.args) == 1
            and not node.keywords
        ):
            node = node.args[0]
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "any"
            and not node.args
            and not node.keywords
            and isinstance(node.func.value, ast.Compare)
            and len(node.func.value.ops) == 1
            and len(node.func.value.comparators) == 1
        ):
            return None
        comparison = node.func.value
        operator = comparison.ops[0]
        left_name = _compared_bound_name(comparison.left)
        right_name = _compared_bound_name(comparison.comparators[0])
        if left_name is None or right_name is None:
            return None
        # The series side must stay a plain name: only the bound may be
        # narrowed, so a comparison against float(series) is not this shape.
        if right_name == bound_name and isinstance(comparison.left, ast.Name):
            if (
                bound_kind == "minimum" and isinstance(operator, (ast.Lt, ast.LtE))
            ) or (bound_kind == "maximum" and isinstance(operator, (ast.Gt, ast.GtE))):
                return left_name
        if left_name == bound_name and isinstance(comparison.comparators[0], ast.Name):
            if (
                bound_kind == "minimum" and isinstance(operator, (ast.Gt, ast.GtE))
            ) or (bound_kind == "maximum" and isinstance(operator, (ast.Lt, ast.LtE))):
                return right_name
        return None

    def _sealed_guard(
        node: ast.stmt,
    ) -> Optional[tuple[str, str, str]]:
        if not (
            isinstance(node, ast.If)
            and not node.orelse
            and is_terminal_failure_body(node.body)
            and isinstance(node.test, ast.BoolOp)
            and isinstance(node.test.op, ast.And)
            and len(node.test.values) == 2
        ):
            return None
        for null_check, range_check in (
            (node.test.values[0], node.test.values[1]),
            (node.test.values[1], node.test.values[0]),
        ):
            bound_name = _non_null_bound_name(null_check)
            if bound_name is None or bound_name not in sealed_bound_names:
                continue
            plausibility_name, bound_kind = sealed_bound_names[bound_name]
            series_name = _sealed_comparison(
                range_check,
                bound_name=bound_name,
                bound_kind=bound_kind,
            )
            if series_name is not None:
                return plausibility_name, bound_kind, series_name
        return None

    sealed_pairs: list[tuple[ast.If, ast.If]] = []
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(parent, field, None)
            if not isinstance(statements, list):
                continue
            guards = [
                (statement, _sealed_guard(statement))
                for statement in statements
                if isinstance(statement, ast.If)
            ]
            guards = [
                (statement, detail)
                for statement, detail in guards
                if detail is not None
            ]
            for index, (lower_guard, lower_detail) in enumerate(guards):
                assert lower_detail is not None
                if lower_detail[1] != "minimum":
                    continue
                for upper_guard, upper_detail in guards[index + 1 :]:
                    assert upper_detail is not None
                    if (
                        upper_detail[1] == "maximum"
                        and upper_detail[0] == lower_detail[0]
                        and upper_detail[2] == lower_detail[2]
                    ):
                        sealed_pairs.append((lower_guard, upper_guard))
    if len(sealed_pairs) == 1:
        lines = code.splitlines(keepends=True)
        replacements: list[tuple[int, int, str]] = []
        for guard in sealed_pairs[0]:
            if not guard.body or guard.body[-1].end_lineno is None:
                return code
            start = guard.body[0].lineno
            end = guard.body[-1].end_lineno
            source_line = lines[start - 1]
            indent = source_line[: len(source_line) - len(source_line.lstrip(" \t"))]
            replacements.append(
                (
                    start,
                    end,
                    f"{indent}pass  # "
                    "_easyicu_flag_only_plausibility_range_retained_v1\n",
                )
            )
        for start, end, replacement in sorted(replacements, reverse=True):
            lines[start - 1 : end] = [replacement]
        repaired = "".join(lines)
        try:
            ast.parse(repaired)
        except SyntaxError:
            return code
        return repaired

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
    counted_candidates: list[tuple[ast.Assign, ast.Assign, ast.If]] = []
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(parent, field, None)
            if not isinstance(statements, list):
                continue
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
    if len(counted_candidates) != 1:
        return code
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
