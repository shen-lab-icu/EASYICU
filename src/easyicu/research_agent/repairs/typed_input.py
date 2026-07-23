"""Science-neutral source repairs for host-issued typed-input bindings."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Sequence

from ..gates.typed_input import resolved_input_shadowed_by_cohort_env_findings
from ..schema import ValidationFinding


def patch_all_rows_outcome_coordinate_filter(
    code: str,
    run_log: str,
    *,
    resolved_input_bindings: Mapping[str, Any] | None,
) -> str:
    """Keep every row of a target-bound outcome-incidence product.

    A producer may use a reader-facing label in its ``outcome`` column while
    the ResearchContext uses the canonical source coordinate.  A rendering
    consumer must not equate those two namespaces and silently remove every
    row.  This repair is intentionally closed to one exact runtime failure,
    one exact AST filter/guard pair, and a host binding that proves the input
    is the complete ``table:outcome_incidence`` product in ``all_rows`` mode.
    It does not choose an outcome: the unique-label runtime guard verifies
    that the already bound table represents exactly one outcome coordinate.
    """

    if (
        "runtimeerror: no bound outcome-incidence rows for target outcome"
        not in str(run_log or "").lower()
    ):
        return code
    binding = dict((resolved_input_bindings or {}).get("table:outcome_incidence") or {})
    consumption = binding.get("consumption_contract")
    product_contract = binding.get("product_contract")
    identity = binding.get("identity_row")
    if (
        not isinstance(consumption, Mapping)
        or consumption.get("mode") != "all_rows"
        or not isinstance(product_contract, Mapping)
        or "outcome" not in list(product_contract.get("columns") or [])
        or not isinstance(identity, Mapping)
        or identity.get("declared_kind") != "table"
        or identity.get("product") != "outcome_incidence"
        or identity.get("input_key") != "table:outcome_incidence"
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def _matches_filter(node: ast.Assign) -> str | None:
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or not isinstance(node.value, ast.Call)
            or node.value.args
            or node.value.keywords
            or not isinstance(node.value.func, ast.Attribute)
            or node.value.func.attr != "copy"
            or not isinstance(node.value.func.value, ast.Subscript)
            or not isinstance(node.value.func.value.value, ast.Attribute)
            or node.value.func.value.value.attr != "loc"
        ):
            return None
        frame_name = node.targets[0].id
        loc_owner = node.value.func.value.value.value
        predicate = node.value.func.value.slice
        expected = ast.parse(
            f'{frame_name}["outcome"].astype(str).eq(str(target_outcome))',
            mode="eval",
        ).body
        if (
            not isinstance(loc_owner, ast.Name)
            or loc_owner.id != frame_name
            or ast.dump(predicate, include_attributes=False)
            != ast.dump(expected, include_attributes=False)
        ):
            return None
        return frame_name

    def _matches_empty_guard(node: ast.stmt, frame_name: str) -> bool:
        return bool(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == frame_name
            and node.test.attr == "empty"
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Raise)
            and isinstance(node.body[0].exc, ast.Call)
            and isinstance(node.body[0].exc.func, ast.Name)
            and node.body[0].exc.func.id == "RuntimeError"
            and len(node.body[0].exc.args) == 1
            and isinstance(node.body[0].exc.args[0], ast.Constant)
            and node.body[0].exc.args[0].value
            == "No bound outcome-incidence rows for target outcome"
        )

    candidates: list[tuple[ast.Assign, ast.If, str]] = []
    for owner in ast.walk(tree):
        body = getattr(owner, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body[:-1]):
            if not isinstance(statement, ast.Assign):
                continue
            frame_name = _matches_filter(statement)
            guard = body[index + 1]
            if frame_name is not None and _matches_empty_guard(guard, frame_name):
                candidates.append((statement, guard, frame_name))
    if len(candidates) != 1:
        return code
    assignment, guard, frame_name = candidates[0]
    label_name = "_easyicu_bound_outcome_labels"
    if any(
        isinstance(node, ast.Name) and node.id == label_name for node in ast.walk(tree)
    ):
        return code
    if not all(
        isinstance(value, int)
        for value in (
            assignment.lineno,
            assignment.col_offset,
            guard.end_lineno,
            guard.end_col_offset,
        )
    ):
        return code

    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    cursor = 0
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line)

    def _offset(lineno: int, utf8_column: int) -> int:
        line = lines[lineno - 1]
        char_column = len(line.encode("utf-8")[:utf8_column].decode("utf-8"))
        return line_starts[lineno - 1] + char_column

    start = _offset(assignment.lineno, assignment.col_offset)
    end = _offset(guard.end_lineno, guard.end_col_offset)
    indent = " " * int(assignment.col_offset)
    replacement = (
        f'{label_name} = {frame_name}["outcome"].dropna().astype(str).unique().tolist()\n'
        f"{indent}if len({label_name}) != 1:\n"
        f'{indent}    raise RuntimeError("Bound outcome-incidence table must contain exactly one outcome coordinate")\n'
        f"{indent}{frame_name} = {frame_name}.copy()"
    )
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_resolved_input_manifest_env(code: str, run_log: str) -> str:
    """Use the host-issued resolved-input manifest path after an exact failure.

    The generated script sometimes derives ``resolved_inputs.json`` from the
    run directory even though the host provides its authoritative path through
    ``EASYICU_RESOLVED_INPUTS_JSON``.  Another narrow failure mode treats that
    path string as the JSON document itself.  Both repairs are deliberately
    limited to exact runtime failures and structurally unique AST shapes.
    """

    lowered_log = (run_log or "").lower()
    missing_manifest_failure = "resolved input manifest not found:" in lowered_log
    path_parsed_as_json_failure = (
        "easyicu_resolved_inputs_json must contain valid json" in lowered_log
        and "jsondecodeerror:" in lowered_log
    )
    if not missing_manifest_failure and not path_parsed_as_json_failure:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def env_subscript(node: ast.AST, env_name: str) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == env_name
        )

    if path_parsed_as_json_failure:
        path_imported = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "pathlib"
            and any(
                alias.name == "Path" and alias.asname is None for alias in node.names
            )
            for node in tree.body
        )
        if not path_imported:
            return code
        env_bindings = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and env_subscript(node.value, "EASYICU_RESOLVED_INPUTS_JSON")
        ]
        if len(env_bindings) != 1:
            return code
        binding_name = env_bindings[0].targets[0].id
        loads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.func.attr == "loads"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == binding_name
            and node.args[0].end_lineno is not None
            and node.args[0].end_col_offset is not None
        ]
        binding_reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == binding_name
        ]
        if len(loads) != 1 or len(binding_reads) != 1:
            return code
        candidate = loads[0].args[0]
        lines = code.splitlines(keepends=True)
        line_starts: list[int] = []
        offset = 0
        for line in lines:
            line_starts.append(offset)
            offset += len(line)

        def absolute_offset(lineno: int, utf8_col: int) -> int:
            line = lines[lineno - 1]
            char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
            return line_starts[lineno - 1] + char_col

        start = absolute_offset(candidate.lineno, candidate.col_offset)
        end = absolute_offset(candidate.end_lineno, candidate.end_col_offset)
        replacement = f'Path({binding_name}).read_text(encoding="utf-8")'
        repaired = code[:start] + replacement + code[end:]
        try:
            ast.parse(repaired)
        except SyntaxError:
            return code
        return repaired

    def env_path_name(node: ast.AST, env_name: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Subscript)
            and isinstance(node.args[0].value, ast.Attribute)
            and isinstance(node.args[0].value.value, ast.Name)
            and node.args[0].value.value.id == "os"
            and node.args[0].value.attr == "environ"
            and isinstance(node.args[0].slice, ast.Constant)
            and node.args[0].slice.value == env_name
        )

    run_dir_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and env_path_name(node.value, "EASYICU_RUN_DIR")
    }
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.left, ast.Name)
        and node.left.id in run_dir_names
        and isinstance(node.right, ast.Constant)
        and node.right.value == "resolved_inputs.json"
        and node.end_lineno is not None
        and node.end_col_offset is not None
    ]
    if len(candidates) != 1:
        return code
    candidate = candidates[0]
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    start = absolute_offset(candidate.lineno, candidate.col_offset)
    end = absolute_offset(candidate.end_lineno, candidate.end_col_offset)
    replacement = 'Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])'
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_resolved_input_relative_path_root(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Join host-issued run-relative input paths to ``EASYICU_RUN_DIR``."""

    coordinates: list[tuple[int, int, int, int]] = []
    matching_findings = 0
    for finding in repair_findings:
        detail = finding.detail or {}
        if detail.get("reason") != "resolved_input_relative_path_wrong_root":
            continue
        matching_findings += 1
        occurrences = detail.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            return code
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                return code
            values = (
                occurrence.get("line"),
                occurrence.get("column"),
                occurrence.get("end_line"),
                occurrence.get("end_column"),
            )
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            ):
                return code
            coordinates.append(tuple(int(value) for value in values))
    if (
        matching_findings != 1
        or not coordinates
        or len(coordinates) != len(set(coordinates))
    ):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    constants = {
        (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        ): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and node.value == "EASYICU_EVIDENCE_DIR"
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int]] = []
    for coordinate in coordinates:
        node = constants.get(coordinate)
        if node is None:
            return code
        replacements.append(
            (
                absolute_offset(coordinate[0], coordinate[1]),
                absolute_offset(coordinate[2], coordinate[3]),
            )
        )
    repaired = code
    replacement = repr("EASYICU_RUN_DIR")
    for start, end in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def patch_resolved_input_cohort_env_shadow(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Keep an exact manifest-bound artifact as the physical typed input."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    authoritative = resolved_input_shadowed_by_cohort_env_findings(tree)
    supplied = [
        finding
        for finding in repair_findings
        if (finding.detail or {}).get("reason")
        == "resolved_typed_input_shadowed_by_cohort_env"
    ]
    if (
        len(authoritative) != 1
        or len(supplied) != 1
        or supplied[0].detail != authoritative[0].detail
    ):
        return code
    occurrences = authoritative[0].detail.get("occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        return code
    calls = {
        (
            int(node.lineno),
            int(node.col_offset),
            int(node.end_lineno),
            int(node.end_col_offset),
        ): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.end_lineno is not None
        and node.end_col_offset is not None
    }
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for occurrence in occurrences:
        if not isinstance(occurrence, dict):
            return code
        values = (
            occurrence.get("line"),
            occurrence.get("column"),
            occurrence.get("end_line"),
            occurrence.get("end_column"),
        )
        replacement_name = occurrence.get("replacement_name")
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            )
            or not isinstance(replacement_name, str)
            or not replacement_name.isidentifier()
        ):
            return code
        coordinate = tuple(int(value) for value in values)
        call = calls.get(coordinate)
        if (
            call is None
            or not isinstance(call.func, ast.Name)
            or call.func.id != "Path"
        ):
            return code
        replacements.append(
            (
                absolute_offset(coordinate[0], coordinate[1]),
                absolute_offset(coordinate[2], coordinate[3]),
                replacement_name,
            )
        )
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = [
    "patch_all_rows_outcome_coordinate_filter",
    "patch_resolved_input_manifest_env",
    "patch_resolved_input_cohort_env_shadow",
    "patch_resolved_input_relative_path_root",
]
