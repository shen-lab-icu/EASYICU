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


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _runtime_name_write_count(tree: ast.Module, name: str) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ) + sum(
        1 for node in ast.walk(tree) if isinstance(node, ast.arg) and node.arg == name
    )


def _optimizer_path_is_unmodified(
    tree: ast.Module,
    path: tuple[str, ...],
) -> bool:
    root = path[0]
    if _runtime_name_write_count(tree, root):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or isinstance(node.ctx, ast.Load):
            continue
        target_path = _attribute_path(node)
        if target_path is not None and target_path[: len(path)] == path:
            return False
    return True


def _scipy_optimizer_bindings(tree: ast.Module) -> set[str]:
    """Return names bound to reviewed scipy.optimize entry points."""

    bindings: set[str] = set()
    scipy_aliases = {"scipy"}
    optimize_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scipy":
                    scipy_aliases.add(alias.asname or "scipy")
                elif alias.name == "scipy.optimize":
                    optimize_aliases.add(alias.asname or "scipy.optimize")
        elif isinstance(node, ast.ImportFrom) and node.module == "scipy.optimize":
            for alias in node.names:
                if alias.name in {"least_squares", "minimize", "root"}:
                    bindings.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _attribute_path(node.func)
        if path is None:
            continue
        if (
            len(path) == 3
            and path[0] in scipy_aliases
            and path[1] == "optimize"
            and path[2] in {"least_squares", "minimize", "root"}
        ):
            bindings.add(".".join(path))
        elif (
            len(path) == 2
            and path[0] in optimize_aliases
            and path[1] in {"least_squares", "minimize", "root"}
        ):
            bindings.add(".".join(path))
    return bindings


def _controlled_optimizer_success_names(tree: ast.Module) -> set[str]:
    """Trace booleans to the ``success`` field of a reviewed optimizer call."""

    optimizer_call_names = _scipy_optimizer_bindings(tree)
    optimizer_results: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        func_path = _attribute_path(value.func)
        if (
            func_path is None
            or ".".join(func_path) not in optimizer_call_names
            or not _optimizer_path_is_unmodified(tree, func_path)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        optimizer_results.update(
            target.id
            for target in targets
            if isinstance(target, ast.Name)
            and _runtime_name_write_count(tree, target.id) == 1
        )

    success_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "bool"
            and len(value.args) == 1
            and not value.keywords
        ):
            value = value.args[0]
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "success"
            and isinstance(value.value, ast.Name)
            and value.value.id in optimizer_results
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        success_names.update(
            target.id
            for target in targets
            if isinstance(target, ast.Name)
            and _runtime_name_write_count(tree, target.id) == 1
        )
    return success_names


def patch_penalized_convergence_contract(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Publish a convergence flag proven to come from optimizer status.

    The primary-model gate requires penalized fits to state both how optimizer
    convergence was checked and whether that check passed.  Only a boolean
    traced to the ``success`` field of a reviewed ``scipy.optimize`` call may
    receive the machine-readable aliases.  Arbitrary names, constants, finite
    coefficients, and iteration-limit heuristics remain unverified.
    """

    target_model_ids = _penalized_convergence_model_ids(findings)
    if len(target_model_ids) != 1:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    string_bindings = _top_level_string_bindings(tree)
    controlled_success_names = _controlled_optimizer_success_names(tree)
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
        if (
            not isinstance(converged, ast.Name)
            or converged.id not in controlled_success_names
        ):
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
