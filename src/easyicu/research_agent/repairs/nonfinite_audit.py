"""Narrow repairs for self-declared non-finite value audits."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

from ..schema import ValidationFinding


_ORDERED_DOMAIN_RUNTIME_ERROR = re.compile(
    r"Invalid [^\r\n:]+ ordered-domain values:\s*"
    r"nonfinite=(?P<nonfinite>\d+),\s*"
    r"noninteger=(?P<noninteger>\d+),\s*"
    r"out_of_domain=(?P<out_of_domain>\d+)",
)


def _import_alias(tree: ast.Module, module: str) -> str | None:
    aliases = {
        alias.asname or module
        for statement in tree.body
        if isinstance(statement, ast.Import)
        for alias in statement.names
        if alias.name == module
    }
    if len(aliases) != 1:
        return None
    name = next(iter(aliases))
    if any(
        (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Store)
        )
        or (isinstance(node, ast.arg) and node.arg == name)
        for node in ast.walk(tree)
    ):
        return None
    return name


def _root_name(node: ast.AST) -> str | None:
    current = node
    while True:
        if isinstance(current, ast.Name):
            return current.id
        if isinstance(current, ast.Call):
            current = current.func
            continue
        if isinstance(current, ast.Attribute):
            current = current.value
            continue
        if isinstance(current, ast.Subscript):
            current = current.value
            continue
        return None


def _is_mask_sum(node: ast.AST, mask_name: str) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == "sum"
        and isinstance(candidate.func.value, ast.Name)
        and candidate.func.value.id == mask_name
        for candidate in ast.walk(node)
    )


def _declares_nonfinite_output(tree: ast.Module, mask_name: str) -> bool:
    count_names = {
        target.id
        for statement in ast.walk(tree)
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        for target in (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if isinstance(target, ast.Name)
        and statement.value is not None
        and _is_mask_sum(statement.value, mask_name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            key.value: value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        status = entries.get("status")
        count = entries.get("count")
        if not (
            isinstance(status, ast.Constant)
            and isinstance(status.value, str)
            and "nonfinite" in status.value.lower().replace("-", "")
            and count is not None
        ):
            continue
        if _is_mask_sum(count, mask_name) or (
            isinstance(count, ast.Name) and count.id in count_names
        ):
            return True
    return False


def _raise_reports_ordered_nonfinite_count(
    tree: ast.Module,
    count_name: str,
) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        joined_strings = [
            candidate
            for candidate in ast.walk(node.exc)
            if isinstance(candidate, ast.JoinedStr)
        ]
        for joined in joined_strings:
            literal_text = "".join(
                str(value.value)
                for value in joined.values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            )
            formatted_names = {
                candidate.value.id
                for candidate in joined.values
                if isinstance(candidate, ast.FormattedValue)
                and isinstance(candidate.value, ast.Name)
            }
            if (
                "ordered-domain values:" in literal_text
                and "nonfinite=" in literal_text
                and count_name in formatted_names
            ):
                return True
    return False


def patch_nonfinite_missing_mask_conflation(code: str, run_log: str) -> str:
    """Keep source-missing values out of an ordered-domain non-finite count.

    ``NaN`` represents source missingness in the analysis frame. Generated code
    can accidentally turn every missing value into a non-finite violation by
    evaluating ``~np.isfinite(values.fillna(np.nan))``. This repair activates
    only for the script's exact ordered-domain runtime error and one AST-proven
    count reported by that same error. It adds the missing observed-value mask;
    no rows, domains, thresholds, or measured values are changed.
    """

    error = _ORDERED_DOMAIN_RUNTIME_ERROR.search(str(run_log or ""))
    if error is None or int(error.group("nonfinite")) <= 0:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    numpy_alias = _import_alias(tree, "numpy")
    if numpy_alias is None:
        return code

    candidates: list[tuple[ast.UnaryOp, ast.Name]] = []
    for assignment in ast.walk(tree):
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and _raise_reports_ordered_nonfinite_count(
                tree,
                assignment.targets[0].id,
            )
            and isinstance(assignment.value, ast.Call)
            and isinstance(assignment.value.func, ast.Name)
            and assignment.value.func.id == "int"
            and len(assignment.value.args) == 1
        ):
            continue
        sum_call = assignment.value.args[0]
        if not (
            isinstance(sum_call, ast.Call)
            and isinstance(sum_call.func, ast.Attribute)
            and sum_call.func.attr == "sum"
            and isinstance(sum_call.func.value, ast.UnaryOp)
            and isinstance(sum_call.func.value.op, ast.Invert)
            and isinstance(sum_call.func.value.operand, ast.Call)
        ):
            continue
        inverted = sum_call.func.value
        finite_call = inverted.operand
        if not (
            isinstance(finite_call.func, ast.Attribute)
            and isinstance(finite_call.func.value, ast.Name)
            and finite_call.func.value.id == numpy_alias
            and finite_call.func.attr == "isfinite"
            and len(finite_call.args) == 1
            and isinstance(finite_call.args[0], ast.Call)
            and isinstance(finite_call.args[0].func, ast.Attribute)
            and finite_call.args[0].func.attr == "fillna"
            and isinstance(finite_call.args[0].func.value, ast.Name)
            and len(finite_call.args[0].args) == 1
            and isinstance(finite_call.args[0].args[0], ast.Attribute)
            and isinstance(finite_call.args[0].args[0].value, ast.Name)
            and finite_call.args[0].args[0].value.id == numpy_alias
            and finite_call.args[0].args[0].attr == "nan"
        ):
            continue
        candidates.append((inverted, finite_call.args[0].func.value))
    if len(candidates) != 1:
        return code

    inverted, values = candidates[0]
    if inverted.end_lineno is None or inverted.end_col_offset is None:
        return code
    values_text = ast.get_source_segment(code, values)
    if not values_text:
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

    start = _absolute_offset(inverted.lineno, inverted.col_offset)
    end = _absolute_offset(inverted.end_lineno, inverted.end_col_offset)
    replacement = (
        f"({values_text}.notna() & "
        f"~{numpy_alias}.isfinite({values_text}))"
    )
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_strict_numeric_nonfinite_audit_conflict(
    code: str,
    *,
    audit_messages: Sequence[str] = (),
    repair_findings: Sequence[ValidationFinding] = (),
) -> str:
    """Preserve non-finite values when code explicitly promises to audit them.

    The repair is deliberately structural and fail-closed.  It applies only
    when one blocking concept finding exists and exactly one code path both:

    * feeds a ``strict_numeric`` result into a derived series; and
    * publishes a named non-finite count that is currently hard-coded from an
      all-False mask.

    Other strict numeric inputs keep their fail-closed behavior.  The audited
    series still rejects lossy string-to-number coercion, while observed
    infinities remain available for the script's own exclusion and audit.
    """

    authorized = bool(audit_messages) or any(
        finding.validator == "llm_concept_auditor" and finding.severity == "error"
        for finding in repair_findings
    )
    if not authorized:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    pandas_alias = _import_alias(tree, "pandas")
    numpy_alias = _import_alias(tree, "numpy")
    if pandas_alias is None or numpy_alias is None:
        return code

    assignments: dict[str, ast.Assign] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments[node.targets[0].id] = node

    candidates: list[tuple[ast.Assign, ast.Assign, str, str, ast.AST]] = []
    for mask_name, mask_assignment in assignments.items():
        value = mask_assignment.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == pandas_alias
            and value.func.attr == "Series"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value is False
            and _declares_nonfinite_output(tree, mask_name)
        ):
            continue
        index_keyword = next(
            (keyword.value for keyword in value.keywords if keyword.arg == "index"),
            None,
        )
        if not (
            isinstance(index_keyword, ast.Attribute)
            and index_keyword.attr == "index"
            and isinstance(index_keyword.value, ast.Name)
        ):
            continue
        derived_name = index_keyword.value.id
        derived_assignment = assignments.get(derived_name)
        if derived_assignment is None:
            continue
        strict_name = _root_name(derived_assignment.value)
        strict_assignment = assignments.get(strict_name or "")
        if strict_assignment is None:
            continue
        strict_call = strict_assignment.value
        if not (
            isinstance(strict_call, ast.Call)
            and isinstance(strict_call.func, ast.Name)
            and strict_call.func.id == "strict_numeric"
            and len(strict_call.args) >= 2
            and isinstance(strict_call.args[1], ast.Constant)
            and isinstance(strict_call.args[1].value, str)
        ):
            continue
        candidates.append(
            (
                strict_assignment,
                mask_assignment,
                strict_name or "",
                derived_name,
                strict_call.args[0],
            )
        )
    if len(candidates) != 1:
        return code

    strict_assignment, mask_assignment, strict_name, derived_name, source = candidates[
        0
    ]
    if strict_assignment.end_lineno is None or mask_assignment.end_lineno is None:
        return code
    source_text = ast.get_source_segment(code, source)
    if not source_text:
        return code
    raw_name = f"_easyicu_{strict_name}_raw_nonfinite_audit_v1"
    loss_name = f"_easyicu_{strict_name}_coercion_loss_v1"
    numeric_name = f"_easyicu_{strict_name}_numeric_nonfinite_audit_v2"
    nonfinite_name = f"_easyicu_{strict_name}_nonfinite_observed_v2"
    all_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }
    if (
        any(
            name in all_names
            for name in {raw_name, loss_name, numeric_name, nonfinite_name}
        )
        or "strict_numeric_input" in all_names
    ):
        return code

    lines = code.splitlines(keepends=True)
    strict_indent = lines[strict_assignment.lineno - 1][: strict_assignment.col_offset]
    body_indent = strict_indent + "    "
    strict_replacement = (
        f"{strict_indent}{raw_name} = ({source_text}).copy()\n"
        f"{strict_indent}{numeric_name} = {pandas_alias}.to_numeric("
        f'{raw_name}, errors="coerce")\n'
        f"{strict_indent}{loss_name} = {raw_name}.notna() & "
        f"{numeric_name}.isna()\n"
        f"{strict_indent}if int({loss_name}.sum()) > 0:\n"
        f'{body_indent}raise ValueError("lossy numeric coercion in audited input")\n'
        f"{strict_indent}{nonfinite_name} = {numeric_name}.notna() & "
        f"~{numpy_alias}.isfinite({numeric_name})\n"
        f"{strict_indent}{strict_name} = strict_numeric_input("
        f"{numeric_name}.mask({nonfinite_name})).values\n"
    )
    mask_indent = lines[mask_assignment.lineno - 1][: mask_assignment.col_offset]
    mask_replacement = (
        f"{mask_indent}{mask_assignment.targets[0].id} = "
        f"{nonfinite_name}.reindex({derived_name}.index, fill_value=False)\n"
    )
    replacements = [
        (
            strict_assignment.lineno - 1,
            strict_assignment.end_lineno,
            strict_replacement,
        ),
        (
            mask_assignment.lineno - 1,
            mask_assignment.end_lineno,
            mask_replacement,
        ),
    ]
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    repaired = "".join(lines)
    repaired_tree = ast.parse(repaired)
    imports = [
        node
        for node in repaired_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if not imports or imports[-1].end_lineno is None:
        return code
    repaired_lines = repaired.splitlines(keepends=True)
    repaired_lines.insert(
        imports[-1].end_lineno,
        "from easyicu.research_agent.methods.descriptive_inputs "
        "import strict_numeric_input\n",
    )
    repaired = "".join(repaired_lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_nonfinite_audit_host_strict_boundary(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding] = (),
) -> str:
    """Upgrade the v1 audit-preservation patch to the host strict SDK."""

    variables = {
        variable
        for finding in repair_findings
        if finding.validator == "llm_concept_auditor"
        and finding.severity == "error"
        and (finding.detail or {}).get("issue_code")
        == "strict_numeric_nonfinite_guard_required"
        for variable in ((finding.detail or {}).get("variables") or [])
        if isinstance(variable, str) and variable.isidentifier()
    }
    if len(variables) != 1:
        return code
    variable = next(iter(variables))
    raw_name = f"_easyicu_{variable}_raw_nonfinite_audit_v1"
    loss_name = f"_easyicu_{variable}_coercion_loss_v1"
    full_mask_name = f"_easyicu_{variable}_nonfinite_observed_v2"
    if raw_name not in code or loss_name not in code or full_mask_name in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    pandas_alias = _import_alias(tree, "pandas")
    numpy_alias = _import_alias(tree, "numpy")
    if pandas_alias is None or numpy_alias is None:
        return code
    if any(
        (isinstance(node, ast.Name) and node.id == "strict_numeric_input")
        or (isinstance(node, ast.arg) and node.arg == "strict_numeric_input")
        for node in ast.walk(tree)
    ):
        return code

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    numeric_assignments = [
        node
        for node in assignments
        if node.targets[0].id == variable
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == pandas_alias
        and node.value.func.attr == "to_numeric"
        and node.value.args
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == raw_name
    ]
    loss_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(candidate, ast.Name) and candidate.id == loss_name
            for candidate in ast.walk(node.test)
        )
        and node.body
        and all(isinstance(statement, ast.Raise) for statement in node.body)
    ]
    mask_assignments = [
        node
        for node in assignments
        if isinstance(node.value, ast.BinOp)
        and any(
            isinstance(candidate, ast.Name)
            and any(
                assignment.targets[0].id == candidate.id
                and _root_name(assignment.value) == variable
                for assignment in assignments
            )
            for candidate in ast.walk(node.value)
        )
        and any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Attribute)
            and candidate.func.attr == "isfinite"
            for candidate in ast.walk(node.value)
        )
        and _declares_nonfinite_output(tree, node.targets[0].id)
    ]
    if not (
        len(numeric_assignments) == len(loss_guards) == len(mask_assignments) == 1
        and loss_guards[0].end_lineno is not None
        and mask_assignments[0].end_lineno is not None
    ):
        return code
    mask_assignment = mask_assignments[0]
    derived_names = {
        candidate.id
        for candidate in ast.walk(mask_assignment.value)
        if isinstance(candidate, ast.Name) and candidate.id != variable
    }
    derived_names -= {numpy_alias}
    if len(derived_names) != 1:
        return code
    derived_name = next(iter(derived_names))

    lines = code.splitlines(keepends=True)
    guard = loss_guards[0]
    guard_indent = lines[guard.lineno - 1][: guard.col_offset]
    upgrade = (
        f"{guard_indent}{full_mask_name} = {variable}.notna() & "
        f"~{numpy_alias}.isfinite({variable})\n"
        f"{guard_indent}{variable} = strict_numeric_input("
        f"{variable}.mask({full_mask_name})).values\n"
    )
    mask_indent = lines[mask_assignment.lineno - 1][: mask_assignment.col_offset]
    mask_replacement = (
        f"{mask_indent}{mask_assignment.targets[0].id} = "
        f"{full_mask_name}.reindex({derived_name}.index, fill_value=False)\n"
    )
    lines[mask_assignment.lineno - 1 : mask_assignment.end_lineno] = [mask_replacement]
    lines.insert(guard.end_lineno, upgrade)
    repaired = "".join(lines)
    repaired_tree = ast.parse(repaired)
    imports = [
        node
        for node in repaired_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if not imports or imports[-1].end_lineno is None:
        return code
    repaired_lines = repaired.splitlines(keepends=True)
    repaired_lines.insert(
        imports[-1].end_lineno,
        "from easyicu.research_agent.methods.descriptive_inputs "
        "import strict_numeric_input\n",
    )
    repaired = "".join(repaired_lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_strict_numeric_helper_nonfinite_guard(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding] = (),
) -> str:
    """Make one proven shared numeric-coercion helper reject infinities.

    The concept auditor supplies the exact affected model variables.  The
    source must route all of them through one literal-list call to one helper
    that already rejects lossy ``to_numeric(..., errors="coerce")`` conversion.
    This patch adds only the missing finite-value guard at that same boundary.
    """

    variables = {
        variable
        for finding in repair_findings
        if finding.validator == "llm_concept_auditor"
        and finding.severity == "error"
        and (finding.detail or {}).get("issue_code")
        == "strict_numeric_nonfinite_guard_required"
        for variable in ((finding.detail or {}).get("variables") or [])
        if isinstance(variable, str) and variable.isidentifier()
    }
    if not variables:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def top_level_import_alias(module: str) -> str | None:
        aliases = {
            alias.asname or module
            for statement in tree.body
            if isinstance(statement, ast.Import)
            for alias in statement.names
            if alias.name == module
        }
        return next(iter(aliases)) if len(aliases) == 1 else None

    pandas_alias = top_level_import_alias("pandas")
    numpy_alias = top_level_import_alias("numpy")
    if pandas_alias is None or numpy_alias is None:
        return code
    guard_name = "_easyicu_nonfinite_numeric_mask_v1"
    if any(
        (isinstance(node, ast.Name) and node.id in {guard_name, "RuntimeError"})
        and isinstance(node.ctx, (ast.Store, ast.Del))
        for node in ast.walk(tree)
    ):
        return code

    candidates: list[tuple[ast.FunctionDef, ast.Assign, str]] = []
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        assignments = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == pandas_alias
            and node.value.func.attr == "to_numeric"
            and any(
                keyword.arg == "errors"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "coerce"
                for keyword in node.value.keywords
            )
        ]
        for assignment in assignments:
            converted_name = assignment.targets[0].id
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "isfinite"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == numpy_alias
                and any(
                    isinstance(descendant, ast.Name) and descendant.id == converted_name
                    for descendant in ast.walk(node)
                )
                for node in ast.walk(function)
            ):
                continue
            has_loss_guard = any(
                isinstance(node, ast.If)
                and any(
                    isinstance(descendant, ast.Name) and "invalid" in descendant.id
                    for descendant in ast.walk(node.test)
                )
                and any(
                    isinstance(descendant, (ast.Raise, ast.Call))
                    for descendant in ast.walk(node)
                )
                for node in ast.walk(function)
            )
            if has_loss_guard:
                candidates.append((function, assignment, converted_name))
    if len(candidates) != 1:
        return code
    helper, assignment, converted_name = candidates[0]
    if any(
        (
            isinstance(node, ast.Name)
            and node.id in {pandas_alias, numpy_alias}
            and isinstance(node.ctx, (ast.Store, ast.Del))
        )
        or (isinstance(node, ast.arg) and node.arg in {pandas_alias, numpy_alias})
        for node in ast.walk(helper)
    ):
        return code

    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper.name
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Name)
    ]
    literal_lists: dict[str, list[set[str]]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.List)
            and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in node.value.elts
            )
        ):
            continue
        literal_lists.setdefault(node.targets[0].id, []).append(
            {str(item.value) for item in node.value.elts}
        )
    if len(helper_calls) != 1:
        return code
    routed_candidates = literal_lists.get(helper_calls[0].args[1].id, [])
    routed_variables = routed_candidates[0] if len(routed_candidates) == 1 else None
    if routed_variables is None or not variables <= routed_variables:
        return code
    if assignment.end_lineno is None:
        return code

    lines = code.splitlines(keepends=True)
    indent = lines[assignment.lineno - 1][: assignment.col_offset]
    body_indent = indent + "    "
    insertion = (
        f"{indent}{guard_name} = {converted_name}.notna() & "
        f"~{numpy_alias}.isfinite({converted_name}.to_numpy(dtype=float))\n"
        f"{indent}if int({guard_name}.sum()) > 0:\n"
        f'{body_indent}raise RuntimeError("non-finite numeric input")\n'
    )
    lines.insert(assignment.end_lineno, insertion)
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = [
    "patch_nonfinite_missing_mask_conflation",
    "patch_nonfinite_audit_host_strict_boundary",
    "patch_strict_numeric_nonfinite_audit_conflict",
    "patch_strict_numeric_helper_nonfinite_guard",
]
