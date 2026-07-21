"""Narrow deterministic repairs for host provenance summary envelopes."""

from __future__ import annotations

import ast
from typing import Any, Optional, Sequence

from .merge_collision import patch_table_one_left_provenance_source

_HOST_RECEIPT_FIELDS = frozenset(
    {
        "role",
        "comparison_n",
        "invalid_pair_n",
        "discordant_n",
        "status",
        "count_column",
    }
)


def _finding_parts(finding: Any) -> tuple[object, object, object, object]:
    if isinstance(finding, dict):
        return (
            finding.get("validator"),
            finding.get("severity"),
            finding.get("message"),
            finding.get("detail"),
        )
    return (
        getattr(finding, "validator", None),
        getattr(finding, "severity", None),
        getattr(finding, "message", None),
        getattr(finding, "detail", None),
    )


def _measurement_receipt_coordinates(
    findings: Sequence[Any],
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Return exact invalid and missing measured/count coordinates."""

    invalid: dict[str, str] = {}
    missing: dict[str, str] = {}
    for finding in findings:
        validator, severity, _message, detail = _finding_parts(finding)
        if (
            validator != "step_summary_integrity"
            or severity != "error"
            or not isinstance(detail, dict)
        ):
            continue
        issue = detail.get("issue")
        if issue not in {
            "measurement_provenance_check_invalid",
            "measurement_provenance_check_missing",
        }:
            continue
        measured = detail.get("measured_column")
        count = detail.get("expected_count_column")
        if not isinstance(measured, str) or not measured.strip():
            return None
        if not isinstance(count, str) or not count.strip() or measured == count:
            return None
        target = invalid if issue == "measurement_provenance_check_invalid" else missing
        if measured in target or measured in (
            missing if target is invalid else invalid
        ):
            return None
        if target is invalid:
            invalid_fields = detail.get("invalid_fields")
            if (
                detail.get("expected_status") != "checked"
                or not isinstance(invalid_fields, list)
                or not invalid_fields
                or not all(isinstance(field, str) for field in invalid_fields)
                or not set(invalid_fields) <= _HOST_RECEIPT_FIELDS
                or not set(invalid_fields)
                & {"role", "comparison_n", "invalid_pair_n", "discordant_n"}
            ):
                return None
        target[measured] = count
    if not invalid:
        return None
    return invalid, missing


def _source_offsets(code: str) -> tuple[list[str], list[int]]:
    lines = code.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    return lines, starts


def _node_span(
    node: ast.AST,
    *,
    lines: Sequence[str],
    starts: Sequence[int],
) -> tuple[int, int] | None:
    if node.end_lineno is None or node.end_col_offset is None:
        return None

    def position(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return starts[lineno - 1] + char_col

    return (
        position(node.lineno, node.col_offset),
        position(node.end_lineno, node.end_col_offset),
    )


def patch_superseded_manual_provenance_audit(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Drop one dead manual audit superseded by an exact host receipt.

    Generated code sometimes computes a provisional literal
    ``invalid_pair_n``/``discordant_n`` audit and later replaces that same
    variable with receipts from ``measurement_provenance_receipt`` before any
    output reads it.  The literal assignment is dead but still triggers the
    conservative module-level preflight.  Remove only that proven-dead
    assignment.  If the host call's frame name is unbound, replace it only
    when there is exactly one preceding top-level pandas table reader.
    """

    matching = []
    for finding in findings:
        validator, severity, _message, detail = _finding_parts(finding)
        if (
            validator == "mechanical_code_preflight"
            and severity == "error"
            and isinstance(detail, dict)
            and detail.get("reason") == "provenance_audit_not_fail_closed"
        ):
            matching.append(detail)
    if len(matching) != 1:
        return code
    issues = matching[0].get("issues")
    if not (
        isinstance(issues, list)
        and len(issues) == 1
        and issues[0]
        == {
            "failure_mode": "module_provenance_scope_not_proven_fail_closed",
            "helper_name": "<module>",
        }
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    exact_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "measurement_provenance_receipt" and alias.asname is None
            for alias in node.names
        )
    ]
    if len(exact_imports) != 1:
        return code

    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    manual_assignments = []
    for node in assignments:
        literals = {
            item.value
            for item in ast.walk(node.value)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        if {"audit_only", "invalid_pair_n", "discordant_n"} <= literals:
            manual_assignments.append(node)
    if len(manual_assignments) != 1:
        return code
    manual = manual_assignments[0]
    audit_name = manual.targets[0].id

    host_assignments: list[tuple[ast.Assign, str]] = []
    for node in assignments:
        if node.lineno <= manual.lineno or node.targets[0].id != audit_name:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        mapping = {
            key.value: value
            for key, value in zip(node.value.keys, node.value.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        checks = mapping.get("checks")
        if isinstance(checks, ast.Name):
            host_assignments.append((node, checks.id))
    if len(host_assignments) != 1:
        return code
    host_assignment, receipts_name = host_assignments[0]

    between = [
        statement
        for statement in tree.body
        if manual.lineno < statement.lineno < host_assignment.lineno
    ]
    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == audit_name
        for statement in between
        for node in ast.walk(statement)
    ):
        return code
    if not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == audit_name
        and int(getattr(node, "lineno", 0) or 0) > host_assignment.lineno
        for node in ast.walk(tree)
    ):
        return code

    receipt_initializers = [
        node
        for node in assignments
        if node.targets[0].id == receipts_name
        and isinstance(node.value, ast.List)
        and not node.value.elts
        and node.lineno < host_assignment.lineno
    ]
    if len(receipt_initializers) != 1:
        return code
    receipt_calls = [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == receipts_name
        and node.func.attr == "append"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Call)
    ]
    if not receipt_calls:
        return code
    if any(
        not (
            isinstance(call.func, ast.Name)
            and call.func.id == "measurement_provenance_receipt"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and {keyword.arg for keyword in call.keywords}
            == {"measured_column", "count_column"}
        )
        for call in receipt_calls
    ):
        return code

    direct_reader_assignments = [
        node
        for node in assignments
        if isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "pd"
        and node.value.func.attr
        in {"read_csv", "read_excel", "read_parquet", "read_table"}
    ]
    replacements: list[tuple[ast.AST, str]] = []
    for call in receipt_calls:
        frame = call.args[0]
        assert isinstance(frame, ast.Name)
        existing_bindings = [
            node
            for node in assignments
            if node.targets[0].id == frame.id and node.lineno < call.lineno
        ]
        if existing_bindings:
            continue
        candidates = [
            node
            for node in direct_reader_assignments
            if node.lineno < call.lineno
            and sum(other.targets[0].id == node.targets[0].id for other in assignments)
            == 1
        ]
        if len(candidates) != 1:
            return code
        replacements.append((frame, candidates[0].targets[0].id))

    lines, starts = _source_offsets(code)
    edits: list[tuple[int, int, str]] = []
    manual_span = _node_span(manual, lines=lines, starts=starts)
    if manual_span is None:
        return code
    edits.append((*manual_span, ""))
    for node, replacement in replacements:
        span = _node_span(node, lines=lines, starts=starts)
        if span is None:
            return code
        edits.append((*span, replacement))
    repaired = code
    for start, end, replacement in sorted(edits, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def repair_superseded_provenance(
    code: str,
    findings: Sequence[Any],
) -> tuple[str, list[str]]:
    """Return the exact patch plus its central-registry repair identifier."""

    repaired = patch_superseded_manual_provenance_audit(code, findings=findings)
    applied = ["superseded_manual_provenance_receipt_v1"] if repaired != code else []
    return repaired, applied


def patch_custom_measurement_provenance_receipts(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Replace one proven custom receipt list with host-owned receipts.

    Authorization comes only from structured summary-integrity coordinates.
    The generated source must expose one static three-column specification,
    one direct four-argument helper call, and one unique helper definition.
    Missing checks are appended from validator-owned measured/count pairs;
    no value column or scientific role is inferred.
    """

    coordinates = _measurement_receipt_coordinates(findings)
    if coordinates is None:
        return code
    invalid, missing = coordinates
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    receipt_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "measurement_provenance_receipt" and alias.asname is None
            for alias in node.names
        )
    ]
    receipt_name_bindings = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == "measurement_provenance_receipt"
        )
        or (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id == "measurement_provenance_receipt"
        )
        or (
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(
                (alias.asname or alias.name) == "measurement_provenance_receipt"
                and not (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                    == "easyicu.research_agent.methods.descriptive_inputs"
                    and alias.name == "measurement_provenance_receipt"
                    and alias.asname is None
                )
                for alias in node.names
            )
        )
    ]
    if len(receipt_imports) > 1 or receipt_name_bindings:
        return code

    candidates: list[
        tuple[ast.Assign, ast.ListComp, ast.Call, ast.Assign, ast.FunctionDef]
    ] = []
    top_assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    for checks_assignment in top_assignments:
        if (
            len(checks_assignment.targets) != 1
            or not isinstance(checks_assignment.targets[0], ast.Name)
            or not isinstance(checks_assignment.value, ast.ListComp)
        ):
            continue
        list_comp = checks_assignment.value
        if (
            not isinstance(list_comp.elt, ast.Call)
            or not isinstance(list_comp.elt.func, ast.Name)
            or len(list_comp.elt.args) != 4
            or list_comp.elt.keywords
            or len(list_comp.generators) != 1
        ):
            continue
        generator = list_comp.generators[0]
        if (
            generator.is_async
            or generator.ifs
            or not isinstance(generator.iter, ast.Name)
            or not isinstance(generator.target, (ast.Tuple, ast.List))
            or len(generator.target.elts) != 3
            or not all(isinstance(item, ast.Name) for item in generator.target.elts)
        ):
            continue
        value_name, measured_name, count_name = (
            item.id for item in generator.target.elts
        )
        call = list_comp.elt
        if not (
            isinstance(call.args[1], ast.Name)
            and call.args[1].id == measured_name
            and isinstance(call.args[2], ast.Name)
            and call.args[2].id == count_name
            and isinstance(call.args[3], ast.Name)
            and call.args[3].id == value_name
        ):
            continue
        spec_assignments = [
            node
            for node in top_assignments
            if len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == generator.iter.id
            and isinstance(node.value, ast.List)
        ]
        helper_defs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == call.func.id
            and not node.decorator_list
        ]
        if len(spec_assignments) != 1 or len(helper_defs) != 1:
            continue
        helper = helper_defs[0]
        if (
            len(helper.args.args) != 4
            or helper.args.posonlyargs
            or helper.args.kwonlyargs
            or helper.args.vararg is not None
            or helper.args.kwarg is not None
            or helper.args.defaults
        ):
            continue
        candidates.append(
            (checks_assignment, list_comp, call, spec_assignments[0], helper)
        )
    if len(candidates) != 1:
        return code
    checks_assignment, _list_comp, call, specs_assignment, helper = candidates[0]

    spec_pairs: dict[str, str] = {}
    for item in specs_assignment.value.elts:
        if (
            not isinstance(item, (ast.Tuple, ast.List))
            or len(item.elts) != 3
            or not all(
                isinstance(value, ast.Constant) and isinstance(value.value, str)
                for value in item.elts
            )
        ):
            return code
        measured = str(item.elts[1].value)
        count = str(item.elts[2].value)
        if measured in spec_pairs:
            return code
        spec_pairs[measured] = count
    if spec_pairs != invalid:
        return code

    helper_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == helper.name
    ]
    returns = [node for node in ast.walk(helper) if isinstance(node, ast.Return)]
    if (
        len(helper_loads) != 1
        or len(returns) != 1
        or not isinstance(returns[0].value, ast.Dict)
    ):
        return code
    return_keys = {
        str(key.value)
        for key in returns[0].value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    if not {"measured_column", "count_column", "status"} <= return_keys:
        return code

    frame_source = ast.get_source_segment(code, call.args[0])
    measured_source = ast.get_source_segment(code, call.args[1])
    count_source = ast.get_source_segment(code, call.args[2])
    if not frame_source or not measured_source or not count_source:
        return code
    replacement_call = (
        f"measurement_provenance_receipt({frame_source}, "
        f"measured_column={measured_source}, count_column={count_source})"
    )

    lines, starts = _source_offsets(code)
    call_span = _node_span(call, lines=lines, starts=starts)
    checks_span = _node_span(checks_assignment, lines=lines, starts=starts)
    if call_span is None or checks_span is None or helper.end_lineno is None:
        return code
    helper_start = starts[helper.lineno - 1]
    helper_end = (
        starts[helper.end_lineno] if helper.end_lineno < len(lines) else len(code)
    )
    while helper_end < len(code) and code[helper_end] == "\n":
        helper_end += 1

    edits: list[tuple[int, int, str]] = [
        (call_span[0], call_span[1], replacement_call),
        (helper_start, helper_end, ""),
    ]
    if missing:
        checks_name = checks_assignment.targets[0].id
        rendered = ",\n".join(
            "        measurement_provenance_receipt(\n"
            f"            {frame_source}, measured_column={measured!r}, "
            f"count_column={count!r}\n"
            "        )"
            for measured, count in sorted(missing.items())
        )
        extension = f"\n{checks_name}.extend(\n" "    [\n" f"{rendered}\n" "    ]\n" ")"
        edits.append((checks_span[1], checks_span[1], extension))

    if not receipt_imports:
        body_index = 0
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            body_index = 1
        leading_imports: list[ast.AST] = []
        while body_index < len(tree.body) and isinstance(
            tree.body[body_index], (ast.Import, ast.ImportFrom)
        ):
            leading_imports.append(tree.body[body_index])
            body_index += 1
        if leading_imports:
            last_import = leading_imports[-1]
            if last_import.end_lineno is None:
                return code
            import_at = (
                starts[last_import.end_lineno]
                if last_import.end_lineno < len(lines)
                else len(code)
            )
        elif tree.body:
            import_at = starts[tree.body[0].lineno - 1]
        else:
            import_at = 0
        edits.append(
            (
                import_at,
                import_at,
                "from easyicu.research_agent.methods.descriptive_inputs "
                "import measurement_provenance_receipt\n",
            )
        )

    repaired = code
    for start, end, replacement in sorted(edits, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        repaired_tree = ast.parse(repaired)
    except SyntaxError:
        return code
    if any(
        isinstance(node, ast.FunctionDef) and node.name == helper.name
        for node in repaired_tree.body
    ):
        return code
    return repaired


def _lexical_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.AST:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return current


def patch_direct_host_provenance_summary(code: str) -> str:
    """Wrap one direct host receipt in the required source/checks envelope.

    The candidate must import the exact host helper, bind its returned mapping
    once, and place that mapping directly in the unique step summary.  The
    transformation does not derive or alter any receipt value.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    exact_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "measurement_provenance_receipt" and alias.asname is None
            for alias in node.names
        )
    ]
    if len(exact_imports) != 1:
        return code
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    summary_values: list[ast.Name] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "step_summary"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "measurement_provenance_audit"
                and isinstance(value, ast.Name)
            ):
                summary_values.append(value)
    if len(summary_values) != 1:
        return code
    summary_value = summary_values[0]
    receipt_name = summary_value.id
    scope = _lexical_scope(summary_value, parents)
    assignments = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and _lexical_scope(node, parents) is scope
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == receipt_name
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "measurement_provenance_receipt"
        and len(node.value.args) == 1
        and not any(keyword.arg is None for keyword in node.value.keywords)
        and {keyword.arg for keyword in node.value.keywords}
        == {"measured_column", "count_column"}
    ]
    if len(assignments) != 1:
        return code
    loads = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == receipt_name
        and _lexical_scope(node, parents) is scope
    ]

    def _allowed_none_guard(load: ast.Name) -> bool:
        current: Optional[ast.AST] = load
        compare: Optional[ast.Compare] = None
        while current in parents and current is not scope:
            current = parents[current]
            if isinstance(current, ast.Compare):
                compare = current
            if isinstance(current, ast.If):
                return bool(
                    compare is not None
                    and current.test is compare
                    and len(compare.ops) == 1
                    and isinstance(compare.ops[0], ast.Is)
                    and len(compare.comparators) == 1
                    and isinstance(compare.comparators[0], ast.Constant)
                    and compare.comparators[0].value is None
                    and current.body
                    and isinstance(current.body[0], ast.Raise)
                )
        return False

    if any(
        load is not summary_value and not _allowed_none_guard(load) for load in loads
    ):
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

    if summary_value.end_lineno is None or summary_value.end_col_offset is None:
        return code
    start = _absolute_offset(summary_value.lineno, summary_value.col_offset)
    end = _absolute_offset(summary_value.end_lineno, summary_value.end_col_offset)
    replacement = '{"source": "COHORT_PARQUET", "checks": [' + receipt_name + "]}"
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_nested_host_provenance_summary(code: str) -> str:
    """Reference one already-authored closed provenance mapping in the summary."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    sources: list[str] = []
    summaries: list[ast.Dict] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            continue
        keys = {
            str(key.value): value
            for key, value in zip(node.value.keys, node.value.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if node.targets[0].id == "step_summary":
            if "measurement_provenance_audit" not in keys:
                summaries.append(node.value)
            continue
        provenance = keys.get("measurement_provenance_audit")
        if not isinstance(provenance, ast.Dict):
            continue
        envelope = {
            str(key.value): value
            for key, value in zip(
                provenance.keys,
                provenance.values,
                strict=True,
            )
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if (
            set(envelope) == {"source", "checks"}
            and isinstance(envelope["source"], ast.Constant)
            and envelope["source"].value == "COHORT_PARQUET"
            and isinstance(envelope["checks"], ast.Name)
        ):
            sources.append(node.targets[0].id)
    if len(sources) != 1 or len(summaries) != 1:
        return code
    summary = summaries[0]
    if summary.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    closing_index = int(summary.end_lineno) - 1
    closing = lines[closing_index]
    closing_indent = closing[: len(closing) - len(closing.lstrip())]
    if closing.strip() not in {"}", "},"}:
        return code
    field = (
        f'{closing_indent}    "measurement_provenance_audit": '
        f'{sources[0]}["measurement_provenance_audit"],\n'
    )
    repaired = "".join(lines[:closing_index] + [field] + lines[closing_index:])
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_measurement_provenance_summary(code: str) -> str:
    """Apply one proven provenance-envelope normalization, fail-closed."""

    for patcher in (
        patch_table_one_left_provenance_source,
        patch_direct_host_provenance_summary,
        patch_nested_host_provenance_summary,
    ):
        repaired = patcher(code)
        if repaired != code:
            return repaired
    return code


__all__ = [
    "patch_direct_host_provenance_summary",
    "patch_measurement_provenance_summary",
    "patch_nested_host_provenance_summary",
]
