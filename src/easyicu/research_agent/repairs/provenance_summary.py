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

    direct_candidates: list[tuple[ast.FunctionDef, list[ast.Call]]] = []
    for helper in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        returns = [node for node in ast.walk(helper) if isinstance(node, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Dict):
            continue
        return_keys = {
            str(key.value)
            for key in returns[0].value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if not {"measured_column", "count_column"} <= return_keys:
            continue
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper.name
        ]
        helper_loads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == helper.name
        ]
        if calls and len(helper_loads) == len(calls):
            direct_candidates.append((helper, calls))
    if len(direct_candidates) == 1 and not missing:
        helper, calls = direct_candidates[0]
        direct_pairs: dict[str, str] = {}
        rendered_calls: list[tuple[ast.Call, str]] = []
        for call in calls:
            if (
                len(call.args) < 3
                or call.keywords
                or not isinstance(call.args[1], ast.Constant)
                or not isinstance(call.args[1].value, str)
                or not isinstance(call.args[2], ast.Constant)
                or not isinstance(call.args[2].value, str)
            ):
                return code
            measured = str(call.args[1].value)
            count = str(call.args[2].value)
            if measured in direct_pairs:
                return code
            frame_source = ast.get_source_segment(code, call.args[0])
            if not frame_source:
                return code
            direct_pairs[measured] = count
            rendered_calls.append(
                (
                    call,
                    f"measurement_provenance_receipt({frame_source}, "
                    f"measured_column={measured!r}, count_column={count!r})",
                )
            )
        if direct_pairs == invalid:
            lines, starts = _source_offsets(code)
            edits: list[tuple[int, int, str]] = []
            helper_span = _node_span(helper, lines=lines, starts=starts)
            if helper_span is None:
                return code
            edits.append((*helper_span, ""))
            for call, replacement in rendered_calls:
                call_span = _node_span(call, lines=lines, starts=starts)
                if call_span is None:
                    return code
                edits.append((*call_span, replacement))
            if not receipt_imports:
                imports = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ]
                if not imports or imports[-1].end_lineno is None:
                    return code
                import_at = (
                    starts[imports[-1].end_lineno]
                    if imports[-1].end_lineno < len(lines)
                    else len(code)
                )
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
                ast.parse(repaired)
            except SyntaxError:
                return code
            return repaired

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
        extension = f"\n{checks_name}.extend(\n    [\n{rendered}\n    ]\n)"
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


def patch_unplanned_measurement_provenance_summary(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Replace one rejected hand-written audit with the exact host receipt.

    Authority comes from one validator-owned missing measured/count coordinate
    plus the rejected check paths.  The source must contain one literal
    ``measurement_provenance_audit`` envelope, one reader-bound frame in the
    same lexical scope, and no competing provenance-column literals.  The
    repair changes only the summary receipt; it does not alter cohort rows or
    scientific outputs.
    """

    missing: list[tuple[str, str]] = []
    unplanned_paths: set[str] = set()
    for finding in findings:
        validator, severity, _message, detail = _finding_parts(finding)
        if (
            validator != "step_summary_integrity"
            or severity != "error"
            or not isinstance(detail, dict)
        ):
            continue
        issue = detail.get("issue")
        if issue == "measurement_provenance_check_missing":
            measured = detail.get("measured_column")
            count = detail.get("expected_count_column")
            if not (
                isinstance(measured, str)
                and measured.strip()
                and isinstance(count, str)
                and count.strip()
                and measured != count
            ):
                return code
            missing.append((measured, count))
        elif issue == "measurement_provenance_check_unplanned":
            path = detail.get("summary_path")
            if not (
                isinstance(path, str)
                and path.startswith("measurement_provenance_audit.checks.")
            ):
                return code
            unplanned_paths.add(path)
    if len(missing) != 1 or not unplanned_paths:
        return code
    measured_column, count_column = missing[0]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    audit_nodes: list[ast.Dict] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "measurement_provenance_audit"
                and isinstance(value, ast.Dict)
            ):
                audit_nodes.append(value)
    if len(audit_nodes) != 1:
        return code
    audit = audit_nodes[0]
    envelope = {
        str(key.value): value
        for key, value in zip(audit.keys, audit.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    checks = envelope.get("checks")
    shadowed_host_helper = any(
        (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == "measurement_provenance_receipt"
        )
        or (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id == "measurement_provenance_receipt"
        )
        for node in ast.walk(tree)
    )
    referenced_receipt_name = None
    literal_checks = bool(
        isinstance(checks, ast.List)
        and len(checks.elts) == len(unplanned_paths)
        and all(isinstance(item, ast.Dict) for item in checks.elts)
    )
    if (
        shadowed_host_helper
        and isinstance(checks, ast.List)
        and len(checks.elts) == 1
        and isinstance(checks.elts[0], ast.Name)
        and len(unplanned_paths) == 1
    ):
        referenced_receipt_name = checks.elts[0].id
    if not (
        set(envelope) == {"source", "checks"}
        and isinstance(envelope["source"], ast.Constant)
        and envelope["source"].value == "COHORT_PARQUET"
        and (literal_checks or referenced_receipt_name is not None)
    ):
        return code
    provenance_literals: set[str] = set()
    for check in checks.elts if literal_checks else []:
        assert isinstance(check, ast.Dict)
        fields = {
            str(key.value): value
            for key, value in zip(check.keys, check.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        for field_name in {
            "measured_column",
            "measurement_column",
            "status_column",
            "count_column",
        }:
            value = fields.get(field_name)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if value.value.endswith("_measured") or value.value.endswith("_n"):
                    provenance_literals.add(str(value.value))
    if provenance_literals and not provenance_literals <= {
        measured_column,
        count_column,
    }:
        return code

    scope = _lexical_scope(audit, parents)
    reader_assignments = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and _lexical_scope(node, parents) is scope
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id in {"pd", "pl"}
        and node.value.func.attr in {"read_csv", "read_parquet", "read_table"}
        and node.lineno < audit.lineno
    ]
    if len(reader_assignments) != 1:
        return code
    frame_name = reader_assignments[0].targets[0].id
    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.id == frame_name
        and node is not reader_assignments[0].targets[0]
        and _lexical_scope(node, parents) is scope
        for node in ast.walk(scope)
    ):
        return code
    receipt_name = (
        "_easyicu_measurement_provenance_receipt_v1"
        if shadowed_host_helper
        else "measurement_provenance_receipt"
    )
    if shadowed_host_helper and any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.id == receipt_name
        for node in ast.walk(tree)
    ):
        return code

    lines, starts = _source_offsets(code)
    replacement = (
        '{"source": "COHORT_PARQUET", "checks": ['
        f"{receipt_name}({frame_name}, "
        f"measured_column={measured_column!r}, count_column={count_column!r})"
        "]}"
    )
    edits: list[tuple[int, int, str]] = []
    if referenced_receipt_name is None:
        audit_span = _node_span(audit, lines=lines, starts=starts)
        if audit_span is None:
            return code
        edits.append((*audit_span, replacement))
    else:
        receipt_assignments = [
            node
            for node in ast.walk(scope)
            if isinstance(node, ast.Assign)
            and _lexical_scope(node, parents) is scope
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == referenced_receipt_name
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "measurement_provenance_receipt"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == frame_name
            and not any(keyword.arg is None for keyword in node.value.keywords)
            and node.lineno < audit.lineno
        ]
        if len(receipt_assignments) != 1:
            return code
        call = receipt_assignments[0].value
        call_span = _node_span(call, lines=lines, starts=starts)
        if call_span is None:
            return code
        edits.append(
            (
                *call_span,
                f"{receipt_name}({frame_name}, "
                f"measured_column={measured_column!r}, "
                f"count_column={count_column!r})",
            )
        )
    exact_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "measurement_provenance_receipt"
            and (alias.asname or alias.name) == receipt_name
            for alias in node.names
        )
    ]
    if len(exact_imports) > 1:
        return code
    if not exact_imports:
        imports = [
            node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        if not imports or imports[-1].end_lineno is None:
            return code
        import_at = (
            starts[imports[-1].end_lineno]
            if imports[-1].end_lineno < len(lines)
            else len(code)
        )
        edits.append(
            (
                import_at,
                import_at,
                "from easyicu.research_agent.methods.descriptive_inputs "
                "import measurement_provenance_receipt"
                + (
                    " as _easyicu_measurement_provenance_receipt_v1"
                    if shadowed_host_helper
                    else ""
                )
                + "\n",
            )
        )
    repaired = code
    for start, end, rendered in sorted(edits, reverse=True):
        repaired = repaired[:start] + rendered + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_unplanned_measurement_provenance_specs(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Remove rejected rows from one proven host-receipt specification loop."""

    rejected: dict[str, int] = {}
    planned_sets: set[tuple[str, ...]] = set()
    for finding in findings:
        validator, severity, _message, detail = _finding_parts(finding)
        if (
            validator != "step_summary_integrity"
            or severity != "error"
            or not isinstance(detail, dict)
            or detail.get("issue") != "measurement_provenance_check_unplanned"
        ):
            continue
        measured_column = detail.get("measured_column")
        summary_path = detail.get("summary_path")
        raw_planned = detail.get("planned_measured_columns")
        if not (
            isinstance(measured_column, str)
            and measured_column.strip()
            and isinstance(summary_path, str)
            and summary_path.startswith("measurement_provenance_audit.checks.")
            and summary_path.rsplit(".", 1)[-1].isdigit()
            and isinstance(raw_planned, list)
            and raw_planned
            and all(isinstance(item, str) and item.strip() for item in raw_planned)
        ):
            return code
        rejected[measured_column] = int(summary_path.rsplit(".", 1)[-1])
        planned_sets.add(tuple(sorted(str(item) for item in raw_planned)))
    if not rejected or len(planned_sets) != 1:
        return code
    planned = set(next(iter(planned_sets)))
    if planned & set(rejected):
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
    candidates: list[tuple[ast.Assign, list[tuple[str, str]]]] = []
    for assignment in ast.walk(tree):
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and isinstance(assignment.value, ast.List)
        ):
            continue
        coordinates: list[tuple[str, str]] = []
        for item in assignment.value.elts:
            if not (
                isinstance(item, (ast.Tuple, ast.List))
                and len(item.elts) == 2
                and all(
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value
                    for value in item.elts
                )
            ):
                coordinates = []
                break
            coordinates.append((str(item.elts[0].value), str(item.elts[1].value)))
        measured = [column for column, _count in coordinates]
        if (
            coordinates
            and len(measured) == len(set(measured))
            and set(measured) == planned | set(rejected)
            and {
                column: index
                for index, column in enumerate(measured)
                if column in rejected
            }
            == rejected
        ):
            candidates.append((assignment, coordinates))
    if len(candidates) != 1:
        return code

    assignment, coordinates = candidates[0]
    spec_name = assignment.targets[0].id
    scope = _lexical_scope(assignment, parents)
    spec_loads = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == spec_name
    ]
    loops = [
        parent
        for node in spec_loads
        if isinstance((parent := parents.get(node)), ast.For)
        and parent.iter is node
    ]
    if len(spec_loads) != 1 or len(loops) != 1:
        return code
    loop = loops[0]
    if not (
        isinstance(loop.target, (ast.Tuple, ast.List))
        and len(loop.target.elts) == 2
        and all(isinstance(item, ast.Name) for item in loop.target.elts)
        and len(loop.body) == 2
        and isinstance(loop.body[0], ast.Assign)
        and len(loop.body[0].targets) == 1
        and isinstance(loop.body[0].targets[0], ast.Name)
        and isinstance(loop.body[0].value, ast.Call)
        and isinstance(loop.body[0].value.func, ast.Name)
        and loop.body[0].value.func.id == "measurement_provenance_receipt"
        and isinstance(loop.body[1], ast.Expr)
        and isinstance(loop.body[1].value, ast.Call)
        and isinstance(loop.body[1].value.func, ast.Attribute)
        and loop.body[1].value.func.attr == "append"
        and isinstance(loop.body[1].value.func.value, ast.Name)
        and len(loop.body[1].value.args) == 1
        and isinstance(loop.body[1].value.args[0], ast.Name)
        and loop.body[1].value.args[0].id == loop.body[0].targets[0].id
    ):
        return code
    measured_name, count_name = (item.id for item in loop.target.elts)
    keywords = {
        keyword.arg: keyword.value
        for keyword in loop.body[0].value.keywords
        if keyword.arg is not None
    }
    if not (
        set(keywords) == {"measured_column", "count_column"}
        and isinstance(keywords["measured_column"], ast.Name)
        and keywords["measured_column"].id == measured_name
        and isinstance(keywords["count_column"], ast.Name)
        and keywords["count_column"].id == count_name
    ):
        return code
    checks_name = loop.body[1].value.func.value.id
    checks_initializers = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == checks_name
        and isinstance(node.value, ast.List)
        and not node.value.elts
        and node.lineno < loop.lineno
    ]
    def _is_target_envelope(node: ast.AST) -> bool:
        if not isinstance(node, ast.Dict):
            return False
        fields = {
            str(key.value): value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        source = fields.get("source")
        checks = fields.get("checks")
        return bool(
            isinstance(source, ast.Constant)
            and source.value == "COHORT_PARQUET"
            and isinstance(checks, ast.Name)
            and checks.id == checks_name
        )

    envelope_count = sum(_is_target_envelope(node) for node in ast.walk(scope))
    if len(checks_initializers) != 1 or envelope_count != 1:
        return code

    kept = [
        item
        for item, (measured_column, _count_column) in zip(
            assignment.value.elts, coordinates, strict=True
        )
        if measured_column not in rejected
    ]
    lines, starts = _source_offsets(code)
    span = _node_span(assignment.value, lines=lines, starts=starts)
    if span is None:
        return code
    repaired = (
        code[: span[0]]
        + ast.unparse(ast.List(elts=kept, ctx=ast.Load()))
        + code[span[1] :]
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_late_measurement_provenance_receipt(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Move one exact host receipt ahead of all result-file writes.

    The LLM concept auditor owns the ordering finding.  The host only moves an
    already-authored, self-raising receipt into the same lexical scope and
    replaces the later inline expression with the bound mapping.  It does not
    alter the receipt arguments, dataframe, output rows, or scientific policy.
    """

    ordering_findings = []
    for finding in findings:
        validator, severity, _message, detail = _finding_parts(finding)
        if (
            validator == "llm_concept_auditor"
            and severity == "error"
            and isinstance(detail, dict)
            and detail.get("issue_code") == "audit_only_companion_row_gating_required"
        ):
            ordering_findings.append(detail)
    if len(ordering_findings) != 1:
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
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    receipt_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "measurement_provenance_receipt"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and not any(keyword.arg is None for keyword in node.keywords)
        and {keyword.arg for keyword in node.keywords}
        == {"measured_column", "count_column"}
    ]
    if len(receipt_calls) != 1:
        return code
    call = receipt_calls[0]
    current: ast.AST = call
    summary_assignment: ast.Assign | None = None
    while current in parents:
        current = parents[current]
        if (
            isinstance(current, ast.Assign)
            and len(current.targets) == 1
            and isinstance(current.targets[0], ast.Name)
            and current.targets[0].id == "step_summary"
        ):
            summary_assignment = current
            break
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            break
    if summary_assignment is None:
        return code
    scope = _lexical_scope(call, parents)

    output_methods = {
        "savefig",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_json",
        "to_parquet",
        "to_pickle",
    }
    output_statements: list[ast.stmt] = []
    for node in ast.walk(scope):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in output_methods
            and node.lineno < call.lineno
            and _lexical_scope(node, parents) is scope
        ):
            continue
        statement: ast.AST = node
        while statement in parents and parents[statement] is not scope:
            statement = parents[statement]
        if isinstance(statement, ast.stmt):
            output_statements.append(statement)
    if not output_statements:
        return code
    first_output = min(output_statements, key=lambda node: int(node.lineno))
    frame_name = call.args[0].id
    frame_bindings = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and _lexical_scope(node, parents) is scope
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == frame_name
        and node.lineno < first_output.lineno
    ]
    if len(frame_bindings) != 1:
        return code

    receipt_name = "_easyicu_measurement_provenance_receipt_v1"
    if any(
        (isinstance(node, ast.Name) and node.id == receipt_name)
        or (isinstance(node, ast.arg) and node.arg == receipt_name)
        for node in ast.walk(tree)
    ):
        return code
    lines, starts = _source_offsets(code)
    call_span = _node_span(call, lines=lines, starts=starts)
    if call_span is None:
        return code
    call_source = ast.get_source_segment(code, call)
    if not call_source:
        return code
    insert_at = starts[first_output.lineno - 1]
    indent = lines[first_output.lineno - 1][: first_output.col_offset]
    assignment = f"{indent}{receipt_name} = {call_source}\n\n"
    edits = [
        (*call_span, receipt_name),
        (insert_at, insert_at, assignment),
    ]
    repaired = code
    for start, end, replacement in sorted(edits, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_unsupported_measurement_receipt_keyword(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Remove one unsupported extra keyword from the exact host receipt.

    The stable helper needs only the frame plus measured/count column keys.
    This repair activates only when structured preflight findings name the
    precise call line, the direct host import is unshadowed, and the call
    already carries both required keyword arguments. It therefore removes an
    invalid adapter argument without choosing or changing any scientific data.
    """

    finding_lines = {
        int(detail["line"])
        for finding in findings
        for validator, severity, _message, detail in [_finding_parts(finding)]
        if validator == "mechanical_code_preflight"
        and severity == "error"
        and isinstance(detail, dict)
        and detail.get("reason") == "host_helper_call_signature_invalid"
        and detail.get("helper_name") == "measurement_provenance_receipt"
        and isinstance(detail.get("line"), int)
        and not isinstance(detail.get("line"), bool)
        and int(detail["line"]) > 0
    }
    if not finding_lines:
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
    shadowing_bindings = [
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
    ]
    if len(exact_imports) != 1 or shadowing_bindings:
        return code
    candidates: list[ast.Call] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        keyword_names = [keyword.arg for keyword in call.keywords]
        if (
            int(getattr(call, "lineno", 0) or 0) in finding_lines
            and isinstance(call.func, ast.Name)
            and call.func.id == "measurement_provenance_receipt"
            and len(call.args) == 1
            and all(name is not None for name in keyword_names)
            and {"measured_column", "count_column"} <= set(keyword_names)
            and set(keyword_names) - {"measured_column", "count_column"}
        ):
            candidates.append(call)
    if len(candidates) != 1:
        return code
    call = candidates[0]
    call_source = ast.get_source_segment(code, call)
    frame_source = ast.get_source_segment(code, call.args[0])
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    measured_source = ast.get_source_segment(code, keywords["measured_column"])
    count_source = ast.get_source_segment(code, keywords["count_column"])
    if not (
        call_source
        and frame_source
        and measured_source
        and count_source
        and code.count(call_source) == 1
    ):
        return code
    replacement = (
        f"measurement_provenance_receipt({frame_source}, "
        f"measured_column={measured_source}, count_column={count_source})"
    )
    repaired = code.replace(call_source, replacement, 1)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_companion_audit_frame_alignment(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Run audit-only companion receipts on the exact analyzed typed frame.

    The concept auditor supplies both identifiers after proving that a receipt
    was evaluated on a different frame from the result-bearing values.  This
    repair is deliberately narrow: the analysis frame must be the first value
    returned by ``load_bound_table(..., "artifact:analysis_cohort")`` and the
    affected call must be either the host receipt itself or a transparent local
    wrapper around that exact host helper.  No column, mask, or row policy is
    changed.
    """

    details = []
    for finding in findings:
        validator, severity, _message, detail = _finding_parts(finding)
        if (
            validator == "llm_concept_auditor"
            and severity == "error"
            and isinstance(detail, dict)
            and detail.get("issue_code") == "audit_only_companion_row_gating_required"
        ):
            details.append(detail)
    if len(details) != 1:
        return code
    audit_frame = details[0].get("audit_frame")
    analysis_frame = details[0].get("analysis_frame")
    if (
        not isinstance(audit_frame, str)
        or not audit_frame.isidentifier()
        or not isinstance(analysis_frame, str)
        or not analysis_frame.isidentifier()
        or audit_frame == analysis_frame
    ):
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

    transparent_wrappers: set[str] = set()
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        if not function.args.args:
            continue
        frame_parameter = function.args.args[0].arg
        receipt_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "measurement_provenance_receipt"
        ]
        if len(receipt_calls) != 1:
            continue
        receipt = receipt_calls[0]
        if (
            len(receipt.args) == 1
            and isinstance(receipt.args[0], ast.Name)
            and receipt.args[0].id == frame_parameter
            and not any(keyword.arg is None for keyword in receipt.keywords)
            and {keyword.arg for keyword in receipt.keywords}
            == {"measured_column", "count_column"}
        ):
            transparent_wrappers.add(function.name)

    safe_calls = {"measurement_provenance_receipt", *transparent_wrappers}
    candidates: list[ast.Name] = []
    candidate_scopes: set[ast.AST] = set()
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not (
            isinstance(call.func, ast.Name)
            and call.func.id in safe_calls
            and call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == audit_frame
        ):
            continue
        candidates.append(call.args[0])
        candidate_scopes.add(_lexical_scope(call, parents))
    if not candidates or len(candidate_scopes) != 1:
        return code
    scope = next(iter(candidate_scopes))

    analysis_bindings = []
    for assignment in (
        node for node in ast.walk(scope) if isinstance(node, ast.Assign)
    ):
        if _lexical_scope(assignment, parents) is not scope:
            continue
        target_names = {
            node.id
            for target in assignment.targets
            for node in ast.walk(target)
            if isinstance(node, ast.Name)
        }
        if analysis_frame not in target_names:
            continue
        value = assignment.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "load_bound_table"
            and any(
                isinstance(argument, ast.Constant)
                and argument.value == "artifact:analysis_cohort"
                for argument in value.args
            )
        ):
            continue
        analysis_bindings.append(assignment)
    if len(analysis_bindings) != 1:
        return code

    lines, starts = _source_offsets(code)
    spans = [
        span
        for node in candidates
        if (span := _node_span(node, lines=lines, starts=starts)) is not None
    ]
    if len(spans) != len(candidates):
        return code
    repaired = code
    for start, end in sorted(spans, reverse=True):
        repaired = repaired[:start] + analysis_frame + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_audit_only_companion_value_selector(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Detach one physiological-value selector from audit-only companions.

    The structured concept finding names a derived selector that combines a
    provenance-support mask with the value's own validity mask. This repair
    changes only one ``series.loc[selector]`` used for the value distribution,
    and only when exactly one direct conjunct is demonstrably derived from that
    same series. Source-status accounting remains intact. Ambiguous data flow
    is left for agent repair.
    """

    repaired = _patch_unsupported_measurement_receipt_keyword(
        code,
        findings=findings,
    )
    aligned = patch_companion_audit_frame_alignment(repaired, findings=findings)
    if aligned != repaired:
        repaired = aligned
    if repaired != code:
        return repaired
    issue_details = []
    for finding in findings:
        validator, severity, _message, detail = _finding_parts(finding)
        if (
            validator == "llm_concept_auditor"
            and severity == "error"
            and isinstance(detail, dict)
            and detail.get("issue_code") == "audit_only_companion_row_gating_required"
        ):
            issue_details.append(detail)
    if len(issue_details) != 1:
        return code
    raw_variables = issue_details[0].get("variables")
    if not isinstance(raw_variables, list):
        return code
    named_variables = {
        value
        for value in raw_variables
        if isinstance(value, str) and value.isidentifier()
    }
    if not named_variables:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    assignments: dict[str, list[ast.Assign]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assignments.setdefault(node.targets[0].id, []).append(node)

    def bitand_names(node: ast.AST) -> list[str] | None:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            left = bitand_names(node.left)
            right = bitand_names(node.right)
            if left is not None and right is not None:
                return [*left, *right]
        return None

    candidates: list[tuple[ast.Name, str]] = []
    for selector in sorted(named_variables):
        selector_assignments = assignments.get(selector, [])
        if len(selector_assignments) != 1:
            continue
        conjuncts = bitand_names(selector_assignments[0].value)
        if not conjuncts or len(set(conjuncts)) < 2:
            continue
        uses = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "loc"
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.slice, ast.Name)
            and node.slice.id == selector
        ]
        if len(uses) != 1:
            continue
        receiver = uses[0].value.value.id
        value_masks = []
        for conjunct in conjuncts:
            definitions = assignments.get(conjunct, [])
            if len(definitions) != 1:
                continue
            if any(
                isinstance(candidate, ast.Name)
                and candidate.id == receiver
                and isinstance(candidate.ctx, ast.Load)
                for candidate in ast.walk(definitions[0].value)
            ):
                value_masks.append(conjunct)
        if len(value_masks) == 1:
            candidates.append((uses[0].slice, value_masks[0]))
    if len(candidates) != 1:
        return code

    selector_node, replacement = candidates[0]
    lines, starts = _source_offsets(code)
    span = _node_span(selector_node, lines=lines, starts=starts)
    if span is None:
        return code
    repaired = code[: span[0]] + replacement + code[span[1] :]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


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


def _patch_named_provenance_summary(code: str) -> str:
    """Wrap one top-level named receipt collection in the closed envelope."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    summary_values: list[ast.Name] = []
    for node in tree.body:
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
    audit_name = summary_values[0].id
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == audit_name
    ]
    audit_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == audit_name
    ]
    if len(assignments) != 1 or len(audit_loads) != 1:
        return code
    assignment = assignments[0]
    value = assignment.value
    receipts_name: str | None = None
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "from_records"
        and isinstance(value.func.value, ast.Attribute)
        and value.func.value.attr == "DataFrame"
        and isinstance(value.func.value.value, ast.Name)
        and value.func.value.value.id == "pd"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and not value.keywords
    ):
        receipts_name = value.args[0].id
    elif isinstance(value, ast.Name) and value.id != audit_name:
        candidate_name = value.id
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
        receipt_assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == candidate_name
            and isinstance(node.value, ast.List)
        ]
        append_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == candidate_name
            and node.func.attr == "append"
        ]
        other_attribute_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == candidate_name
            and node.func.attr != "append"
        ]
        stored_names = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == candidate_name
        ]
        if (
            len(exact_imports) != 1
            or len(receipt_assignments) != 1
            or len(stored_names) != 1
            or other_attribute_calls
        ):
            return code
        receipt_calls = list(receipt_assignments[0].value.elts) + [
            node.args[0]
            for node in append_calls
            if len(node.args) == 1 and not node.keywords
        ]
        if not receipt_calls or len(receipt_calls) != (
            len(receipt_assignments[0].value.elts) + len(append_calls)
        ):
            return code
        if any(
            not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "measurement_provenance_receipt"
                and len(node.args) == 1
                and not any(keyword.arg is None for keyword in node.keywords)
                and {keyword.arg for keyword in node.keywords}
                == {"measured_column", "count_column"}
            )
            for node in receipt_calls
        ):
            return code
        receipts_name = candidate_name
    if receipts_name is None:
        return code
    assignment_source = ast.get_source_segment(code, assignment)
    if not assignment_source or code.count(assignment_source) != 1:
        return code
    replacement = (
        f'{audit_name} = {{"source": "COHORT_PARQUET", "checks": {receipts_name}}}'
    )
    repaired = code.replace(assignment_source, replacement, 1)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_direct_host_receipt_source_guard(code: str, run_log: str) -> str:
    """Remove one impossible direct-receipt ``source`` assertion and envelope it.

    ``measurement_provenance_receipt`` returns one self-validating check record;
    the ``source`` coordinate belongs to the step-summary envelope, not that
    record. This patch is limited to the exact runtime diagnostic, one direct
    host call, its redundant type/source guards, and one direct summary use.
    """

    diagnostic = "measurement provenance source is not cohort_parquet"
    if diagnostic not in str(run_log or "").lower():
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
    receipt_name = assignments[0].targets[0].id

    type_guards: list[ast.Expr] = []
    source_guards: list[ast.Expr] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "require"
            and len(node.value.args) >= 2
            and isinstance(node.value.args[1], ast.Constant)
            and isinstance(node.value.args[1].value, str)
        ):
            continue
        condition = node.value.args[0]
        message = node.value.args[1].value.lower()
        if (
            message == "measurement_provenance_receipt did not return a mapping"
            and isinstance(condition, ast.Call)
            and isinstance(condition.func, ast.Name)
            and condition.func.id == "isinstance"
            and len(condition.args) == 2
            and isinstance(condition.args[0], ast.Name)
            and condition.args[0].id == receipt_name
            and isinstance(condition.args[1], ast.Name)
            and condition.args[1].id == "dict"
        ):
            type_guards.append(node)
            continue
        if message != diagnostic or not isinstance(condition, ast.Compare):
            continue
        if (
            len(condition.ops) != 1
            or not isinstance(condition.ops[0], ast.Eq)
            or len(condition.comparators) != 1
        ):
            continue
        candidates = (condition.left, condition.comparators[0])
        source_get = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Attribute)
                and candidate.func.attr == "get"
                and isinstance(candidate.func.value, ast.Name)
                and candidate.func.value.id == receipt_name
                and len(candidate.args) == 1
                and isinstance(candidate.args[0], ast.Constant)
                and candidate.args[0].value == "source"
                and not candidate.keywords
            ),
            None,
        )
        source_literal = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, ast.Constant)
                and candidate.value == "COHORT_PARQUET"
            ),
            None,
        )
        if source_get is not None and source_literal is not None:
            source_guards.append(node)

    if len(type_guards) != 1 or len(source_guards) != 1:
        return code
    lines = code.splitlines(keepends=True)
    guards = sorted(
        (*type_guards, *source_guards),
        key=lambda item: item.lineno,
        reverse=True,
    )
    for guard in guards:
        if guard.end_lineno is None:
            return code
        del lines[guard.lineno - 1 : guard.end_lineno]
    without_false_guards = "".join(lines)
    try:
        ast.parse(without_false_guards)
    except SyntaxError:
        return code
    repaired = patch_direct_host_provenance_summary(without_false_guards)
    return repaired if repaired != without_false_guards else code


def patch_closed_provenance_envelope_alias(code: str) -> str:
    """Rename one proven closed provenance envelope to its canonical key."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def _closed_envelope(value: ast.AST, *, scope: ast.AST) -> bool:
        candidate = value
        if isinstance(candidate, ast.Name):
            assignments = [
                node
                for node in ast.walk(scope)
                if isinstance(node, ast.Assign)
                and _lexical_scope(node, parents) is scope
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == candidate.id
            ]
            if len(assignments) != 1:
                return False
            candidate = assignments[0].value
        if not isinstance(candidate, ast.Dict):
            return False
        fields = {
            str(key.value): item
            for key, item in zip(candidate.keys, candidate.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        return bool(
            len(fields) == len(candidate.keys) == 2
            and set(fields) == {"source", "checks"}
            and isinstance(fields["source"], ast.Constant)
            and fields["source"].value == "COHORT_PARQUET"
            and isinstance(fields["checks"], (ast.List, ast.Name, ast.Tuple))
        )

    aliases: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "step_summary"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        entries = [
            (key, value)
            for key, value in zip(node.value.keys, node.value.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        keys = [str(key.value) for key, _value in entries]
        if "measurement_provenance_audit" in keys:
            continue
        scope = _lexical_scope(node, parents)
        aliases.extend(
            key
            for key, value in entries
            if key.value == "measurement_provenance"
            and _closed_envelope(value, scope=scope)
        )
    if len(aliases) != 1:
        return code
    lines, starts = _source_offsets(code)
    span = _node_span(aliases[0], lines=lines, starts=starts)
    if span is None:
        return code
    repaired = (
        code[: span[0]]
        + '"measurement_provenance_audit"'
        + code[span[1] :]
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_measurement_provenance_summary(code: str) -> str:
    """Apply one proven provenance-envelope normalization, fail-closed."""

    for patcher in (
        patch_closed_provenance_envelope_alias,
        patch_table_one_left_provenance_source,
        patch_direct_host_provenance_summary,
        patch_nested_host_provenance_summary,
        _patch_named_provenance_summary,
    ):
        repaired = patcher(code)
        if repaired != code:
            return repaired
    return code


def patch_measurement_provenance_contract(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Normalize one validator-proven provenance contract representation."""

    repaired = patch_unplanned_measurement_provenance_specs(
        code,
        findings=findings,
    )
    if repaired != code:
        return repaired
    repaired = patch_unplanned_measurement_provenance_summary(
        code,
        findings=findings,
    )
    return repaired if repaired != code else patch_measurement_provenance_summary(code)


__all__ = [
    "patch_audit_only_companion_value_selector",
    "patch_closed_provenance_envelope_alias",
    "patch_companion_audit_frame_alignment",
    "patch_direct_host_provenance_summary",
    "patch_direct_host_receipt_source_guard",
    "patch_measurement_provenance_contract",
    "patch_measurement_provenance_summary",
    "patch_nested_host_provenance_summary",
    "patch_late_measurement_provenance_receipt",
    "patch_unplanned_measurement_provenance_summary",
    "patch_unplanned_measurement_provenance_specs",
]
