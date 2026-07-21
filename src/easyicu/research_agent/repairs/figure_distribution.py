"""Narrow repairs for generated categorical-distribution renderers."""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping, Optional

_EMPTY_DISTRIBUTION_MARKERS = (
    "no supported ",
    "categorical distribution rows were found",
    "explicitly supported distribution statistic",
)
_KNOWN_DISTRIBUTION_ROLES = {
    "distribution",
    "category_distribution",
    "level_distribution",
    "frequency",
    "count",
    "percentage",
    "prevalence",
}


def _subscript_key(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Subscript):
        return None
    value = node.slice
    return (
        value.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
        else None
    )


def _has_category_nonnull_guard(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "notna"
        and _subscript_key(node.func.value) == "category"
        for node in ast.walk(tree)
    )


def _used_as_statistic_role_set(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "isin"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == name
        ):
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Subscript):
            key = _subscript_key(receiver)
            if key is not None and "statistic" in key.lower():
                return True
    return False


def _set_assignment_candidates(tree: ast.AST) -> list[tuple[ast.Name, ast.Set]]:
    candidates: list[tuple[ast.Name, ast.Set]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Set)
        ):
            continue
        values = {
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        if len(values) != len(node.value.elts):
            continue
        if "clinical_bin" in values or len(values & _KNOWN_DISTRIBUTION_ROLES) < 3:
            continue
        candidates.append((node.targets[0], node.value))
    return candidates


def _append_set_literal(code: str, node: ast.Set) -> Optional[str]:
    if node.end_lineno is None or node.end_col_offset is None:
        return None
    lines = code.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
    end = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    segment = code[start:end]
    segment_lines = segment.splitlines()
    if len(segment_lines) >= 2 and segment_lines[-1].strip() == "}":
        item_indent = re.match(r"\s*", segment_lines[1]).group(0)
        replacement = "\n".join(
            [*segment_lines[:-1], f'{item_indent}"clinical_bin",', segment_lines[-1]]
        )
    elif segment.endswith("}"):
        replacement = segment[:-1].rstrip() + ', "clinical_bin"}'
    else:
        return None
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


def patch_categorical_distribution_clinical_bin_role(
    code: str,
    step_summary: Mapping[str, Any],
) -> Optional[str]:
    """Admit the host's closed ``clinical_bin`` role in one failed renderer.

    The patch is authorized only when the renderer itself reported that its
    categorical role allowlist selected no rows, the source contains one
    unambiguous literal role set used against a statistic column, and category
    rows remain protected by a non-null guard.  It changes no values, bins,
    labels, denominators, or reconciliation checks.
    """

    summary_text = str(step_summary).lower()
    if not all(marker in summary_text for marker in _EMPTY_DISTRIBUTION_MARKERS):
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not _has_category_nonnull_guard(tree):
        return None
    candidates = [
        set_node
        for name_node, set_node in _set_assignment_candidates(tree)
        if _used_as_statistic_role_set(tree, name_node.id)
    ]
    if len(candidates) != 1:
        return None
    return _append_set_literal(code, candidates[0])


def patch_text_distribution_denominator_from_counts(
    code: str,
    run_log: str,
) -> Optional[str]:
    """Reconcile one semantic denominator label from verified category counts.

    Some tidy source tables use a text ``denominator`` column to name the
    population (for example, ``valid observed values``) while carrying the
    numeric denominator in the mutually exclusive category counts.  Generated
    renderers occasionally coerce that semantic label to numeric and fail.

    This repair is intentionally closed: the exact runtime failure, one
    ``pd.to_numeric(..., errors="coerce")`` assignment, and a prior numeric
    conversion of the same frame's ``count`` column must all be present.  The
    fallback is allowed only when every denominator value is non-null and the
    same text label.  It derives no scientific value beyond ``sum(count)``;
    downstream count/percentage reconciliation remains authoritative.
    """

    if (
        "distribution denominator must be numeric for figure reconciliation"
        not in str(run_log).lower()
        or "_easyicu_text_denominator_from_counts_v1" in code
    ):
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    candidates: list[tuple[ast.Assign, str, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and _subscript_key(node.targets[0]) == "denominator_numeric"
            and isinstance(node.targets[0].value, ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "pd"
            and node.value.func.attr == "to_numeric"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
            and len(node.value.keywords) == 1
            and node.value.keywords[0].arg == "errors"
            and isinstance(node.value.keywords[0].value, ast.Constant)
            and node.value.keywords[0].value.value == "coerce"
        ):
            continue
        candidates.append((node, node.targets[0].value.id, node.value.args[0].id))
    if len(candidates) != 1:
        return None
    assignment, frame_name, original_name = candidates[0]
    if assignment.end_lineno is None:
        return None

    prior_count_conversions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and node.lineno < assignment.lineno
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == frame_name
            and _subscript_key(target) == "count"
            for target in node.targets
        )
        and any(
            isinstance(call, ast.Call)
            and (
                (
                    isinstance(call.func, ast.Name)
                    and call.func.id == "strict_numeric_input"
                )
                or (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "pd"
                    and call.func.attr == "to_numeric"
                )
            )
            for call in ast.walk(node.value)
        )
    ]
    if len(prior_count_conversions) != 1:
        return None

    lines = code.splitlines(keepends=True)
    first_line = lines[assignment.lineno - 1]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    body = indent + ("\t" if "\t" in indent else "    ")
    patch = (
        f"{indent}# _easyicu_text_denominator_from_counts_v1\n"
        f"{indent}_easyicu_denominator_numeric_v1 = pd.to_numeric(\n"
        f'{body}{original_name}, errors="coerce"\n'
        f"{indent})\n"
        f"{indent}if _easyicu_denominator_numeric_v1.isna().all():\n"
        f"{body}_easyicu_denominator_labels_v1 = (\n"
        f"{body}    {original_name}.dropna().astype(str).str.strip()\n"
        f"{body})\n"
        f"{body}if (\n"
        f"{body}    _easyicu_denominator_labels_v1.empty\n"
        f"{body}    or _easyicu_denominator_labels_v1.nunique() != 1\n"
        f"{body}    or len(_easyicu_denominator_labels_v1) != len({original_name})\n"
        f"{body}):\n"
        f"{body}    raise ValueError(\n"
        f'{body}        "Distribution denominator labels are not one complete semantic role"\n'
        f"{body}    )\n"
        f"{body}_easyicu_denominator_from_counts_v1 = float(\n"
        f'{body}    {frame_name}["count"].sum()\n'
        f"{body})\n"
        f"{body}if (\n"
        f"{body}    not np.isfinite(_easyicu_denominator_from_counts_v1)\n"
        f"{body}    or _easyicu_denominator_from_counts_v1 <= 0\n"
        f"{body}):\n"
        f"{body}    raise ValueError(\n"
        f'{body}        "Category counts cannot define a positive denominator"\n'
        f"{body}    )\n"
        f"{body}_easyicu_denominator_numeric_v1 = pd.Series(\n"
        f"{body}    _easyicu_denominator_from_counts_v1,\n"
        f"{body}    index={frame_name}.index,\n"
        f"{body}    dtype=float,\n"
        f"{body})\n"
        f'{indent}{frame_name}["denominator_numeric"] = '
        f"_easyicu_denominator_numeric_v1\n"
    )
    lines[assignment.lineno - 1 : assignment.end_lineno] = [patch]
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


__all__ = [
    "patch_categorical_distribution_clinical_bin_role",
    "patch_text_distribution_denominator_from_counts",
]
