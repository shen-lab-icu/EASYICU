"""Science-neutral source repairs for host-issued typed-input bindings."""

from __future__ import annotations

import ast
from typing import Sequence

from ..gates.typed_input import resolved_input_shadowed_by_cohort_env_findings
from ..schema import ValidationFinding


def patch_resolved_input_manifest_env(code: str, run_log: str) -> str:
    """Use the host-issued resolved-input manifest path after an exact failure.

    The generated script sometimes derives ``resolved_inputs.json`` from the
    run directory even though the host provides its authoritative path through
    ``EASYICU_RESOLVED_INPUTS_JSON``.  This repair is deliberately limited to
    that exact runtime failure and to a script which proves both the run-dir
    binding and one literal ``run_dir / "resolved_inputs.json"`` expression.
    """

    if "resolved input manifest not found:" not in (run_log or "").lower():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

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
    "patch_resolved_input_manifest_env",
    "patch_resolved_input_cohort_env_shadow",
    "patch_resolved_input_relative_path_root",
]
