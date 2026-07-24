"""Mechanical repair for explicitly named typed-statistic JSON sidecars."""

from __future__ import annotations

import ast
from typing import Any, Mapping


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _output_basename(node: ast.AST) -> str | None:
    """Return the literal leaf in ``OUT_DIR / "name.json"``-style paths."""

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _literal_string(node.right)
    return _literal_string(node)


def _dict_has_explicit_name(node: ast.Dict) -> bool:
    return any(
        _literal_string(key) in {"name", "statistic"}
        for key in node.keys
        if key is not None
    )


def _character_span(code: str, node: ast.AST) -> tuple[int, int] | None:
    coordinates = (
        getattr(node, "lineno", None),
        getattr(node, "col_offset", None),
        getattr(node, "end_lineno", None),
        getattr(node, "end_col_offset", None),
    )
    if any(value is None for value in coordinates):
        return None
    lines = code.splitlines(keepends=True)

    def offset(line_number: int, utf8_column: int) -> int:
        prefix = lines[line_number - 1].encode("utf-8")[:utf8_column]
        return sum(len(line) for line in lines[: line_number - 1]) + len(
            prefix.decode("utf-8")
        )

    return (
        offset(int(coordinates[0]), int(coordinates[1])),
        offset(int(coordinates[2]), int(coordinates[3])),
    )


def patch_typed_statistic_sidecar_names(
    code: str,
    *,
    step_summary: Mapping[str, Any],
) -> str | None:
    """Add only the missing typed product name to exact JSON write payloads.

    The repair is deliberately all-or-nothing.  It requires a typed
    ``output_files`` registry, one literal JSON basename per statistic, one
    matching ``write_json(path, payload)`` call, and one uniquely owned dict
    literal.  It never creates or changes a numeric value.
    """

    output_files = step_summary.get("output_files")
    if not isinstance(output_files, Mapping):
        return None
    statistics: dict[str, str] = {}
    for raw_product, raw_path in output_files.items():
        product = str(raw_product or "").strip()
        path = str(raw_path or "").strip()
        if (
            product.startswith("statistic:")
            and product.count(":") == 1
            and product.split(":", 1)[1]
            and path
            and "/" not in path
            and "\\" not in path
            and path.endswith(".json")
        ):
            statistics[product.split(":", 1)[1]] = path
    if not statistics:
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    assignments: dict[str, list[ast.Dict]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            assignments.setdefault(node.targets[0].id, []).append(node.value)

    payloads: dict[str, ast.Dict] = {}
    for statistic_name, basename in statistics.items():
        matches: list[ast.Dict] = []
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id != "write_json"
                or len(node.args) != 2
                or _output_basename(node.args[0]) != basename
            ):
                continue
            payload = node.args[1]
            if isinstance(payload, ast.Dict):
                matches.append(payload)
            elif isinstance(payload, ast.Name):
                candidates = assignments.get(payload.id, [])
                if len(candidates) == 1:
                    matches.append(candidates[0])
        if len(matches) != 1:
            return None
        payloads[statistic_name] = matches[0]

    owners: dict[int, list[str]] = {}
    for statistic_name, payload in payloads.items():
        owners.setdefault(id(payload), []).append(statistic_name)
    if any(len(names) != 1 for names in owners.values()):
        return None

    replacements: list[tuple[int, int, str]] = []
    for statistic_name, payload in payloads.items():
        if _dict_has_explicit_name(payload):
            continue
        span = _character_span(code, payload)
        if span is None:
            return None
        repaired_payload = ast.Dict(
            keys=[ast.Constant(value="name"), *payload.keys],
            values=[ast.Constant(value=statistic_name), *payload.values],
        )
        replacements.append((span[0], span[1], ast.unparse(repaired_payload)))

    if not replacements:
        return None
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired if repaired != code else None


__all__ = ["patch_typed_statistic_sidecar_names"]
