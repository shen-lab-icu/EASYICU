"""Direct-scope checks for host-issued typed-input identity rows."""

from __future__ import annotations

import ast

from ..schema import ValidationFinding


def _literal_key(node: ast.AST) -> str | None:
    return (
        str(node.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _module_scope_nodes(tree: ast.Module):
    stack = list(reversed(tree.body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _target_binds_name(target: ast.AST, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.List, ast.Tuple)):
        return any(_target_binds_name(item, name) for item in target.elts)
    return False


def _inside_for_target_shadow(
    node: ast.AST,
    *,
    name: str,
    parents: dict[int, ast.AST],
) -> bool:
    """Return whether a loop target locally rebinds ``name`` around ``node``."""

    current = node
    while (parent := parents.get(id(current))) is not None:
        if (
            isinstance(parent, (ast.For, ast.AsyncFor))
            and _target_binds_name(parent.target, name)
            and current in parent.body
        ):
            return True
        current = parent
    return False


def direct_resolved_input_key_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject module-scope ``binding.input_key`` reads from resolved inputs."""

    nodes = list(_module_scope_nodes(tree))
    node_ids = {id(node) for node in nodes}
    parents = {
        id(child): node
        for node in nodes
        for child in ast.iter_child_nodes(node)
        if id(child) in node_ids
    }
    keys_by_name: dict[str, set[str]] = {}
    for node in nodes:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.args
            and (key := _literal_key(node.args[0])) is not None
        ):
            keys_by_name.setdefault(node.func.value.id, set()).add(key)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if (key := _literal_key(node.slice)) is not None:
                keys_by_name.setdefault(node.value.id, set()).add(key)
    manifest_names = {
        name
        for name, keys in keys_by_name.items()
        if {"planner_declared_inputs", "inputs"} <= keys
    }

    input_mapping_names: set[str] = set()
    for node in nodes:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id in manifest_names
            and value.args
            and _literal_key(value.args[0]) == "inputs"
        ) or (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id in manifest_names
            and _literal_key(value.slice) == "inputs"
        ):
            input_mapping_names.add(node.targets[0].id)

    binding_names = {
        node.targets[0].id
        for node in nodes
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id in input_mapping_names
    }

    findings: list[ValidationFinding] = []
    for binding_name in sorted(binding_names):
        occurrences: list[dict[str, int]] = []
        for node in nodes:
            direct_subscript = (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == binding_name
                and _literal_key(node.slice) == "input_key"
            )
            direct_get = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == binding_name
                and node.args
                and _literal_key(node.args[0]) == "input_key"
            )
            if (
                not (direct_subscript or direct_get)
                or _inside_for_target_shadow(
                    node,
                    name=binding_name,
                    parents=parents,
                )
                or node.end_lineno is None
                or node.end_col_offset is None
            ):
                continue
            occurrences.append(
                {
                    "line": int(node.lineno),
                    "column": int(node.col_offset),
                    "end_line": int(node.end_lineno),
                    "end_column": int(node.end_col_offset),
                }
            )
        if occurrences:
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "A resolved typed-input binding stores its authoritative "
                        "input key in identity_row; the top-level binding row does "
                        "not expose binding['input_key']."
                    ),
                    detail={
                        "reason": "resolved_input_key_not_materialized",
                        "binding_name": binding_name,
                        "access_occurrences": occurrences,
                    },
                )
            )
    return findings


__all__ = ["direct_resolved_input_key_findings"]
