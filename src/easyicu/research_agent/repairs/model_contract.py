"""Science-neutral source repairs for model-contract observability."""

from __future__ import annotations

import ast
from typing import Any, Mapping, Sequence


def _finding_payload(finding: Any) -> tuple[str, Mapping[str, Any]]:
    validator = getattr(finding, "validator", None)
    detail = getattr(finding, "detail", None)
    if isinstance(finding, Mapping):
        validator = finding.get("validator")
        detail = finding.get("detail")
    return str(validator or ""), detail if isinstance(detail, Mapping) else {}


def _penalized_convergence_model_ids(findings: Sequence[Any]) -> tuple[str, ...]:
    model_ids: list[str] = []
    for finding in findings:
        validator, detail = _finding_payload(finding)
        if validator != "primary_model_contract":
            continue
        issues = detail.get("issues")
        if not isinstance(issues, list):
            continue
        for issue in issues:
            if not isinstance(issue, Mapping):
                continue
            if issue.get("issue") != "penalized_convergence_not_verified":
                continue
            model_id = str(issue.get("model_id") or "").strip()
            if model_id:
                model_ids.append(model_id)
    return tuple(dict.fromkeys(model_ids))


def _top_level_string_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    duplicates: set[str] = set()
    for node in tree.body:
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id in bindings:
                duplicates.add(target.id)
            bindings[target.id] = node.value.value
    for name in duplicates:
        bindings.pop(name, None)
    return bindings


def _resolved_string(node: ast.AST, bindings: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    return None


def patch_penalized_convergence_contract(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Publish an already-computed convergence flag under controlled fields.

    The primary-model gate requires penalized fits to state both how optimizer
    convergence was checked and whether that check passed. Generated scripts
    sometimes compute a trustworthy ``converged`` boolean from the fitted
    estimator but omit the two machine-readable aliases. This repair only
    copies that existing boolean into the contract; it does not fit a model,
    choose a method, alter rows, or introduce a result.
    """

    target_model_ids = _penalized_convergence_model_ids(findings)
    if len(target_model_ids) != 1:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    string_bindings = _top_level_string_bindings(tree)
    candidates: list[ast.Dict] = []
    convergence_sources: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or len(node.keys) != len(node.values):
            continue
        keys: dict[str, ast.AST] = {}
        malformed = False
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                malformed = True
                break
            if key.value in keys:
                malformed = True
                break
            keys[key.value] = value
        if malformed:
            continue
        if not {
            "model_id",
            "fit_method",
            "penalized",
            "converged",
        }.issubset(keys):
            continue
        if {"convergence_method", "optimizer_success"} & set(keys):
            continue
        if _resolved_string(keys["model_id"], string_bindings) != target_model_ids[0]:
            continue
        converged = keys["converged"]
        if not isinstance(converged, ast.Name):
            continue
        candidates.append(node)
        convergence_sources[id(node)] = converged.id

    if len(candidates) != 1:
        return code
    candidate = candidates[0]
    if candidate.end_lineno is None or candidate.end_col_offset is None:
        return code
    lines = code.splitlines(keepends=True)
    closing_line = lines[candidate.end_lineno - 1]
    if closing_line.strip() != "}":
        return code
    closing_offset = sum(len(line) for line in lines[: candidate.end_lineno - 1])
    before_closing = code[:closing_offset].rstrip()
    if not before_closing.endswith(","):
        return code

    key_columns = [
        key.col_offset
        for key in candidate.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    if not key_columns or len(set(key_columns)) != 1:
        return code
    entry_indent = " " * key_columns[0]
    converged_name = convergence_sources[id(candidate)]
    insertion = (
        f'{entry_indent}"convergence_method": "optimizer_success",\n'
        f'{entry_indent}"optimizer_success": bool({converged_name}),\n'
    )
    repaired = code[:closing_offset] + insertion + code[closing_offset:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


__all__ = ["patch_penalized_convergence_contract"]
