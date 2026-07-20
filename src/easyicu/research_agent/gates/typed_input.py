"""Mechanical gates for host-issued typed-input bindings."""

from __future__ import annotations

import ast

from ..schema import ValidationFinding

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _subscript_key(node: ast.AST) -> object:
    return node.value if isinstance(node, ast.Constant) else None


def _parents(tree: ast.Module) -> dict[int, ast.AST]:
    return {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _scope_id(
    node: ast.AST,
    *,
    parents: dict[int, ast.AST],
    tree: ast.Module,
) -> int:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, _SCOPE_NODES):
            return id(current)
        current = parents.get(id(current))
    return id(tree)


def resolved_input_relative_path_root_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Require run-root resolution for host-issued ``relative_path`` values.

    Resolved-input paths already include the ``evidence/`` component. Joining
    one to ``EASYICU_EVIDENCE_DIR`` produces ``evidence/evidence/...``. The
    finding is limited to bindings proven to descend from the resolved-input
    manifest; arbitrary dictionaries remain outside host repair authority.
    """

    parents = _parents(tree)
    manifest_keys: dict[tuple[int, str], set[str]] = {}
    for node in ast.walk(tree):
        scope_id = _scope_id(node, parents=parents, tree=tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            key = _subscript_key(node.slice)
            if isinstance(key, str):
                manifest_keys.setdefault((scope_id, node.value.id), set()).add(key)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.args
            and isinstance(_subscript_key(node.args[0]), str)
        ):
            manifest_keys.setdefault((scope_id, node.func.value.id), set()).add(
                str(_subscript_key(node.args[0]))
            )
    resolved_manifests = {
        coordinate
        for coordinate, keys in manifest_keys.items()
        if {"planner_declared_inputs", "inputs"} <= keys
    }

    input_mappings: set[tuple[int, str]] = set()
    binding_names: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        target_name = targets[0].id
        value = node.value
        scope = _scope_id(node, parents=parents, tree=tree)
        manifest_name: str | None = None
        if (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and _subscript_key(value.slice) == "inputs"
        ):
            manifest_name = value.value.id
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get"
            and isinstance(value.func.value, ast.Name)
            and value.args
            and _subscript_key(value.args[0]) == "inputs"
        ):
            manifest_name = value.func.value.id
        if manifest_name is not None and (scope, manifest_name) in resolved_manifests:
            input_mappings.add((scope, target_name))
            continue

        if (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Subscript)
            and isinstance(value.value.value, ast.Name)
            and _subscript_key(value.value.slice) == "inputs"
            and (scope, value.value.value.id) in resolved_manifests
        ):
            binding_names.add((scope, target_name))
        elif (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and (scope, value.value.id) in input_mappings
        ):
            binding_names.add((scope, target_name))

    def environment_key(node: ast.AST) -> ast.Constant | None:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "EASYICU_EVIDENCE_DIR"
        ):
            return node.slice
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
            and node.func.value.attr == "environ"
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "EASYICU_EVIDENCE_DIR"
        ):
            return node.args[0]
        return None

    def is_binding_relative_path(node: ast.AST, *, scope: int) -> bool:
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return (scope, node.value.id) in binding_names and _subscript_key(
                node.slice
            ) == "relative_path"
        return bool(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and (scope, node.func.value.id) in binding_names
            and node.args
            and _subscript_key(node.args[0]) == "relative_path"
        )

    occurrences: list[dict[str, int]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "Path"
            and len(node.left.args) == 1
        ):
            continue
        scope = _scope_id(node, parents=parents, tree=tree)
        if not is_binding_relative_path(node.right, scope=scope):
            continue
        key = environment_key(node.left.args[0])
        if key is None or key.end_lineno is None or key.end_col_offset is None:
            continue
        occurrences.append(
            {
                "line": int(key.lineno),
                "column": int(key.col_offset),
                "end_line": int(key.end_lineno),
                "end_column": int(key.end_col_offset),
            }
        )
    if not occurrences:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "Resolved typed-input relative_path values are run-root-relative; "
                "joining them to EASYICU_EVIDENCE_DIR duplicates the evidence "
                "directory. Join them to EASYICU_RUN_DIR."
            ),
            detail={
                "reason": "resolved_input_relative_path_wrong_root",
                "occurrences": occurrences,
            },
        )
    ]


__all__ = ["resolved_input_relative_path_root_findings"]
