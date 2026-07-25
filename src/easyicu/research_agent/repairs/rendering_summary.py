"""Closed repair for effect echoes in rendering-only step summaries."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Sequence


def _finding_parts(finding: Any) -> tuple[str, Mapping[str, Any] | None]:
    if isinstance(finding, Mapping):
        validator = str(finding.get("validator") or "")
        detail = finding.get("detail")
    else:
        validator = str(getattr(finding, "validator", "") or "")
        detail = getattr(finding, "detail", None)
    return validator, detail if isinstance(detail, Mapping) else None


def _dict_items(node: ast.AST | None) -> dict[str, ast.AST] | None:
    if not isinstance(node, ast.Dict) or any(key is None for key in node.keys):
        return None
    if any(
        not isinstance(key, ast.Constant) or not isinstance(key.value, str)
        for key in node.keys
    ):
        return None
    keys = [str(key.value) for key in node.keys]
    if len(keys) != len(set(keys)):
        return None
    return dict(zip(keys, node.values, strict=True))


def _span(code: str, node: ast.AST) -> tuple[int, int] | None:
    if not all(
        hasattr(node, attribute)
        for attribute in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        return None
    lines = code.splitlines(keepends=True)

    def character_offset(line_number: int, byte_column: int) -> int:
        line = lines[line_number - 1]
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        return sum(len(part) for part in lines[: line_number - 1]) + character_column

    return (
        character_offset(node.lineno, node.col_offset),
        character_offset(node.end_lineno, node.end_col_offset),
    )


def patch_render_only_effect_echo(
    code: str,
    *,
    findings: Sequence[Any],
) -> str | None:
    """Remove parent-effect echoes from one rendering-only summary mapping."""

    matching_details: list[Mapping[str, Any]] = []
    for finding in findings:
        validator, detail = _finding_parts(finding)
        if (
            validator == "declared_product_contract"
            and detail is not None
            and detail.get("kind") == "unauthorized_effect_product"
        ):
            matching_details.append(detail)
    if len(matching_details) != 1:
        return None
    detail = matching_details[0]
    if str(
        detail.get("planned_method") or ""
    ).strip().lower() != "visualization" or list(
        detail.get("declared_effect_products") or []
    ):
        return None

    summary_paths = detail.get("summary_effect_paths")
    registered_products = detail.get("registered_effect_products")
    if (
        not isinstance(summary_paths, list)
        or not summary_paths
        or not all(
            isinstance(path, str)
            and path.startswith("numeric_summary.")
            and path.count(".") == 1
            for path in summary_paths
        )
        or not isinstance(registered_products, list)
    ):
        return None
    echo_keys = {path.split(".", 1)[1] for path in summary_paths}
    expected_registered = {
        f"{kind}:{key}" for key in echo_keys for kind in ("log", "statistic")
    }
    if set(registered_products) != expected_registered:
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    candidates: list[ast.Dict] = []
    for node in ast.walk(tree):
        if (
            not isinstance(node, (ast.Assign, ast.AnnAssign))
            or (
                isinstance(node, ast.Assign)
                and (
                    len(node.targets) != 1
                    or not isinstance(node.targets[0], ast.Name)
                    or node.targets[0].id != "step_summary"
                )
            )
            or (
                isinstance(node, ast.AnnAssign)
                and (
                    not isinstance(node.target, ast.Name)
                    or node.target.id != "step_summary"
                )
            )
        ):
            continue
        value = node.value
        items = _dict_items(value)
        if items is None:
            continue
        status = items.get("status")
        numeric_summary = items.get("numeric_summary")
        if (
            not isinstance(status, ast.Constant)
            or status.value not in {"completed", "ok"}
            or "figure_files" not in items
            or "output_files" not in items
            or not isinstance(numeric_summary, ast.Dict)
        ):
            continue
        candidates.append(numeric_summary)
    if len(candidates) != 1:
        return None

    numeric_summary = candidates[0]
    numeric_items = _dict_items(numeric_summary)
    if (
        numeric_items is None
        or not echo_keys.issubset(numeric_items)
        or len(numeric_items) <= len(echo_keys)
    ):
        return None
    remaining_keys: list[ast.expr] = []
    remaining_values: list[ast.expr] = []
    for key, value in zip(numeric_summary.keys, numeric_summary.values, strict=True):
        assert isinstance(key, ast.Constant)
        if key.value in echo_keys:
            continue
        remaining_keys.append(key)
        remaining_values.append(value)
    replacement = ast.unparse(
        ast.Dict(keys=remaining_keys, values=remaining_values),
    )
    offsets = _span(code, numeric_summary)
    if offsets is None:
        return None
    start, end = offsets
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


__all__ = ["patch_render_only_effect_echo"]
