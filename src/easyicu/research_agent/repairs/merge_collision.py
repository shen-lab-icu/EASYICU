"""Fail-closed repair for one proven Pandas merge suffix collision."""

from __future__ import annotations

import ast
import re

_KEY_ERROR = re.compile(r"KeyError:\s*['\"](?P<name>[A-Za-z_]\w*)['\"]")
_TABLE_ONE_MISSING = re.compile(
    r"Table 1 input columns are missing:\s*(?P<columns>\[[^\n]+\])"
)
_SENTINELS = {
    "_easyicu_merge_source_v1",
    "_easyicu_merge_left_values_v1",
    "_easyicu_merge_right_values_v1",
}


def _replace_node_source(code: str, node: ast.AST, replacement: str) -> str | None:
    if node.end_lineno is None or node.end_col_offset is None:
        return None
    lines = code.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    def position(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return starts[lineno - 1] + char_col

    start = position(node.lineno, node.col_offset)
    end = position(node.end_lineno, node.end_col_offset)
    return code[:start] + replacement + code[end:]


def _canonicalize_table_one_left_provenance_source(
    code: str,
    *,
    left_name: str,
    result_name: str,
) -> str:
    """Canonicalize only host receipts proven to retain left-cohort columns.

    A Table One overlay may replace one validated value column from a secondary
    typed product while its measurement/count companions remain in the exact
    ``artifact:analysis_cohort`` left frame.  In that narrow shape the receipt
    is still independently replayed against ``COHORT_PARQUET`` by the host.
    Correct only the summary source label; never alter receipt values.
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

    left_bindings = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == left_name
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "loaded_products"
        and isinstance(node.value.slice, ast.Constant)
        and node.value.slice.value == "artifact:analysis_cohort"
    ]
    if len(left_bindings) != 1:
        return code

    pair_assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "measurement_pairs"
        and isinstance(node.value, (ast.List, ast.Tuple))
    ]
    if len(pair_assignments) != 1:
        return code
    pair_columns: set[str] = set()
    for element in pair_assignments[0].value.elts:
        if not (
            isinstance(element, (ast.List, ast.Tuple))
            and len(element.elts) == 2
            and all(
                isinstance(value, ast.Constant) and isinstance(value.value, str)
                for value in element.elts
            )
        ):
            return code
        pair_columns.update(str(value.value) for value in element.elts)
    if not pair_columns:
        return code

    required_left_assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "required_analysis_columns"
        and isinstance(node.value, (ast.List, ast.Tuple))
        and all(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in node.value.elts
        )
    ]
    if len(required_left_assignments) != 1:
        return code
    required_left_columns = {
        str(value.value) for value in required_left_assignments[0].value.elts
    }
    if not pair_columns <= required_left_columns:
        return code
    has_fail_closed_left_check = any(
        isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Name) and child.id == "required_analysis_columns"
            for child in ast.walk(node.test)
        )
        and any(
            isinstance(child, ast.Attribute)
            and child.attr == "columns"
            and isinstance(child.value, ast.Name)
            and child.value.id == left_name
            for child in ast.walk(node.test)
        )
        and any(isinstance(child, ast.Raise) for child in node.body)
        for node in tree.body
    )
    if not has_fail_closed_left_check:
        return code

    checks_assignments: list[ast.Assign] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "measurement_checks"
            and isinstance(node.value, ast.ListComp)
            and isinstance(node.value.elt, ast.Call)
        ):
            continue
        call = node.value.elt
        if (
            isinstance(call.func, ast.Name)
            and call.func.id == "measurement_provenance_receipt"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == result_name
            and {keyword.arg for keyword in call.keywords}
            == {"measured_column", "count_column"}
            and len(node.value.generators) == 1
            and isinstance(node.value.generators[0].iter, ast.Name)
            and node.value.generators[0].iter.id == "measurement_pairs"
        ):
            checks_assignments.append(node)
    if len(checks_assignments) != 1:
        return code

    source_nodes: list[ast.Constant] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"step_summary", "summary"}
            and isinstance(node.value, ast.Dict)
        ):
            continue
        summary = {
            key.value: value
            for key, value in zip(node.value.keys, node.value.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        audit = summary.get("measurement_provenance_audit")
        if not isinstance(audit, ast.Dict):
            continue
        envelope = {
            key.value: value
            for key, value in zip(audit.keys, audit.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        source = envelope.get("source")
        checks = envelope.get("checks")
        if (
            set(envelope) == {"source", "checks"}
            and isinstance(source, ast.Constant)
            and isinstance(source.value, str)
            and source.value != "COHORT_PARQUET"
            and isinstance(checks, ast.Name)
            and checks.id == "measurement_checks"
        ):
            source_nodes.append(source)
    if len(source_nodes) != 1:
        return code
    repaired = _replace_node_source(code, source_nodes[0], repr("COHORT_PARQUET"))
    if repaired is None:
        return code
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _right_projection(
    tree: ast.Module,
    *,
    right_name: str,
    key_name: str,
) -> str | None:
    matches: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == right_name
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "copy"
            and not node.value.args
            and not node.value.keywords
            and isinstance(node.value.func.value, ast.Subscript)
            and isinstance(node.value.func.value.slice, ast.List)
        ):
            continue
        elements = node.value.func.value.slice.elts
        if (
            len(elements) == 2
            and isinstance(elements[0], ast.Constant)
            and elements[0].value == key_name
            and isinstance(elements[1], ast.Name)
        ):
            matches.append(elements[1].id)
    return matches[0] if len(matches) == 1 else None


def patch_pandas_merge_dynamic_column_collision(code: str, run_log: str) -> str:
    """Guard and remove a duplicate dynamic column only after exact equality."""

    if _TABLE_ONE_MISSING.search(run_log or "") is not None:
        return patch_table_one_authored_secondary_overlay(code, run_log)
    if _KEY_ERROR.search(run_log or "") is None:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if names & _SENTINELS:
        return code
    candidates: list[tuple[ast.Assign, ast.Name, str, str, str, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "merge"
            and isinstance(node.value.func.value, ast.Name)
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
        ):
            continue
        on_values = [
            keyword.value.value
            for keyword in node.value.keywords
            if keyword.arg == "on"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ]
        if len(on_values) != 1 or not str(on_values[0]).isidentifier():
            continue
        left_name = node.value.func.value.id
        right_name = node.value.args[0].id
        result_name = node.targets[0].id
        key_name = str(on_values[0])
        column_name = _right_projection(
            tree,
            right_name=right_name,
            key_name=key_name,
        )
        if column_name is None:
            continue
        dynamic_reads = [
            subscript
            for subscript in ast.walk(tree)
            if isinstance(subscript, ast.Subscript)
            and isinstance(subscript.value, ast.Name)
            and subscript.value.id == result_name
            and isinstance(subscript.slice, ast.Name)
            and subscript.slice.id == column_name
            and int(subscript.lineno) > int(node.lineno)
        ]
        if not dynamic_reads:
            continue
        candidates.append(
            (node, node.value.func.value, left_name, right_name, key_name, column_name)
        )
    if len(candidates) != 1:
        return code
    statement, receiver, left_name, right_name, key_name, column_name = candidates[0]
    if statement.end_lineno is None or receiver.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    source_line = lines[int(receiver.lineno) - 1]
    receiver_start = len(source_line.encode("utf-8")[: receiver.col_offset].decode())
    receiver_end = len(source_line.encode("utf-8")[: receiver.end_col_offset].decode())
    if source_line[receiver_start:receiver_end] != left_name:
        return code
    lines[int(receiver.lineno) - 1] = (
        source_line[:receiver_start]
        + "_easyicu_merge_source_v1"
        + source_line[receiver_end:]
    )
    indent = " " * int(statement.col_offset)
    guard = (
        f"{indent}_easyicu_merge_source_v1 = {left_name}\n"
        f"{indent}if {column_name} in {left_name}.columns and "
        f"{column_name} in {right_name}.columns:\n"
        f"{indent}    _easyicu_merge_left_values_v1 = "
        f"{left_name}.set_index({key_name!r})[{column_name}].sort_index()\n"
        f"{indent}    _easyicu_merge_right_values_v1 = "
        f"{right_name}.set_index({key_name!r})[{column_name}].sort_index()\n"
        f"{indent}    if not _easyicu_merge_left_values_v1.equals("
        "_easyicu_merge_right_values_v1):\n"
        f"{indent}        raise RuntimeError("
        "'Duplicate merge column disagrees across typed inputs')\n"
        f"{indent}    _easyicu_merge_source_v1 = "
        f"{left_name}.drop(columns=[{column_name}])\n"
    )
    insertion = int(statement.lineno) - 1
    repaired = "".join(lines[:insertion] + [guard] + lines[insertion:])
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_table_one_authored_secondary_overlay(code: str, run_log: str) -> str:
    """Preserve an authored secondary-table overlay after Pandas suffixing.

    The repair is deliberately narrower than a general merge heuristic.  It
    requires the runner to prove that Table One lost a declared column, the
    generated script to load the right-hand frame from a Planner-resolved
    typed product, and that same script to assert the missing column in the
    right-hand frame before a unique one-to-one left join.  Dropping the
    colliding left copy then realizes the source selection the authored script
    already made; it does not choose a variable or input on the model's behalf.
    """

    match = _TABLE_ONE_MISSING.search(run_log or "")
    if match is None:
        return code
    try:
        missing_value = ast.literal_eval(match.group("columns"))
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    if not (
        isinstance(missing_value, list)
        and missing_value
        and all(
            isinstance(value, str) and value.isidentifier() for value in missing_value
        )
    ):
        return code
    missing_columns = tuple(dict.fromkeys(missing_value))

    literal_lists: dict[str, set[str]] = {}
    resolved_right_names: set[str] = set()
    table_one_results: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, (ast.List, ast.Tuple))
            and all(
                isinstance(element, ast.Constant) and isinstance(element.value, str)
                for element in node.value.elts
            )
        ):
            literal_lists[node.targets[0].id] = {
                str(element.value) for element in node.value.elts
            }
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "loaded_products"
            and isinstance(node.value.slice, ast.Constant)
            and isinstance(node.value.slice.value, str)
        ):
            resolved_right_names.add(node.targets[0].id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_grouped_table_one"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            table_one_results.add(node.args[0].id)

    candidates: list[tuple[ast.Assign, ast.Name, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in table_one_results
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "merge"
            and isinstance(node.value.func.value, ast.Name)
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id in resolved_right_names
        ):
            continue
        keyword_values = {
            keyword.arg: keyword.value.value
            for keyword in node.value.keywords
            if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
        }
        if (
            keyword_values.get("how") != "left"
            or keyword_values.get("validate") != "one_to_one"
            or not isinstance(keyword_values.get("on"), str)
        ):
            continue
        right_name = node.value.args[0].id
        right_contracts = [
            columns
            for name, columns in literal_lists.items()
            if set(missing_columns) <= columns
            and any(
                isinstance(child, ast.Name) and child.id == name
                for child in ast.walk(tree)
            )
        ]
        if len(right_contracts) != 1:
            continue
        # The right frame itself must be referenced by a fail-closed column
        # membership check, not merely loaded and ignored.
        right_membership_checks = [
            child
            for child in ast.walk(tree)
            if isinstance(child, ast.Attribute)
            and child.attr == "columns"
            and isinstance(child.value, ast.Name)
            and child.value.id == right_name
        ]
        if not right_membership_checks:
            continue
        candidates.append((node, node.value.func.value, node.value.func.value.id))
    if len(candidates) != 1:
        return code

    statement, receiver, left_name = candidates[0]
    result_name = statement.targets[0].id
    if statement.end_lineno is None or receiver.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    source_line = lines[int(receiver.lineno) - 1]
    receiver_start = len(source_line.encode("utf-8")[: receiver.col_offset].decode())
    receiver_end = len(source_line.encode("utf-8")[: receiver.end_col_offset].decode())
    if source_line[receiver_start:receiver_end] != left_name:
        return code
    replacement = f"{left_name}.drop(columns={list(missing_columns)!r})"
    lines[int(receiver.lineno) - 1] = (
        source_line[:receiver_start] + replacement + source_line[receiver_end:]
    )
    repaired = "".join(lines)
    repaired = _canonicalize_table_one_left_provenance_source(
        repaired,
        left_name=left_name,
        result_name=result_name,
    )
    try:
        ast.parse(repaired)
    except SyntaxError:  # pragma: no cover - defensive
        return code
    return repaired


__all__ = [
    "patch_pandas_merge_dynamic_column_collision",
    "patch_table_one_authored_secondary_overlay",
]
