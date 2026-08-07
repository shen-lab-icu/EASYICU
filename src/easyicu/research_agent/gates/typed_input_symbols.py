"""Prove local symbols descended from the host resolved-input manifest."""

from __future__ import annotations

import ast
from collections import Counter

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def subscript_key(node: ast.AST) -> object:
    return node.value if isinstance(node, ast.Constant) else None


def parents(tree: ast.Module) -> dict[int, ast.AST]:
    return {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def scope_id(
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


def _environment_key(node: ast.AST, expected: str) -> bool:
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and node.value.attr == "environ"
    ):
        return subscript_key(node.slice) == expected
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "os"
        and node.func.value.attr == "environ"
        and node.func.attr == "get"
        and node.args
        and subscript_key(node.args[0]) == expected
    )


def _resolved_manifest_loads(
    tree: ast.Module,
    *,
    parents_by_id: dict[int, ast.AST],
) -> set[tuple[int, str]]:
    """Return names loaded directly from the host manifest path."""

    def local_json_loader_names() -> set[str]:
        loaders: set[str] = set()
        for function in tree.body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            positional = [*function.args.posonlyargs, *function.args.args]
            if (
                function.decorator_list
                or len(positional) != 1
                or function.args.vararg is not None
                or function.args.kwarg is not None
            ):
                continue
            body = [
                statement
                for statement in function.body
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            ]
            if len(body) != 1 or not isinstance(body[0], (ast.With, ast.AsyncWith)):
                continue
            path_name = positional[0].arg
            returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
            if len(returns) != 1:
                continue
            returned = returns[0].value
            if not (
                isinstance(returned, ast.Call)
                and isinstance(returned.func, ast.Attribute)
                and isinstance(returned.func.value, ast.Name)
                and returned.func.value.id == "json"
                and returned.func.attr == "load"
                and len(returned.args) == 1
                and isinstance(returned.args[0], ast.Name)
            ):
                continue
            handle_name = returned.args[0].id
            opens_parameter = any(
                isinstance(node, (ast.With, ast.AsyncWith))
                and any(
                    isinstance(item.optional_vars, ast.Name)
                    and item.optional_vars.id == handle_name
                    and isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Name)
                    and item.context_expr.func.id == "open"
                    and item.context_expr.args
                    and isinstance(item.context_expr.args[0], ast.Name)
                    and item.context_expr.args[0].id == path_name
                    for item in node.items
                )
                for node in ast.walk(function)
            )
            if opens_parameter:
                loaders.add(function.name)
        return loaders

    manifest_paths: set[tuple[int, str]] = set()
    manifest_handles: set[tuple[int, str]] = set()
    assignments: list[tuple[int, str, ast.AST]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        scope = scope_id(node, parents=parents_by_id, tree=tree)
        assignments.append((scope, targets[0].id, node.value))
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Path"
            and len(value.args) == 1
            and _environment_key(value.args[0], "EASYICU_RESOLVED_INPUTS_JSON")
        ):
            manifest_paths.add((scope, targets[0].id))

    assignment_counts = Counter(
        (scope, target) for scope, target, _value in assignments
    )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        scope = scope_id(node, parents=parents_by_id, tree=tree)
        for item in node.items:
            if not isinstance(item.optional_vars, ast.Name):
                continue
            context = item.context_expr
            if not (
                isinstance(context, ast.Call)
                and isinstance(context.func, ast.Attribute)
                and context.func.attr == "open"
                and isinstance(context.func.value, ast.Name)
                and (scope, context.func.value.id) in manifest_paths
            ):
                continue
            manifest_handles.add((scope, item.optional_vars.id))

    loader_names = local_json_loader_names()
    resolved: set[tuple[int, str]] = set()
    for scope, target, value in assignments:
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in loader_names
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and (scope, value.args[0].id) in manifest_paths
            and assignment_counts[(scope, value.args[0].id)] == 1
            and assignment_counts[(scope, target)] == 1
        ):
            resolved.add((scope, target))
            continue
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "json"
            and value.func.attr in {"load", "loads"}
            and len(value.args) == 1
        ):
            continue
        source = value.args[0]
        if (
            value.func.attr == "load"
            and isinstance(source, ast.Name)
            and (scope, source.id) in manifest_handles
        ):
            resolved.add((scope, target))
            continue
        if (
            value.func.attr == "loads"
            and isinstance(source, ast.Call)
            and isinstance(source.func, ast.Attribute)
            and source.func.attr == "read_text"
            and isinstance(source.func.value, ast.Name)
            and (scope, source.func.value.id) in manifest_paths
        ):
            resolved.add((scope, target))

    changed = True
    while changed:
        changed = False
        for scope, target, value in assignments:
            source_name: str | None = None
            if isinstance(value, ast.Name):
                source_name = value.id
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "get"
                and isinstance(value.func.value, ast.Name)
                and len(value.args) >= 2
                and subscript_key(value.args[0]) == "manifest"
                and isinstance(value.args[1], ast.Name)
                and value.args[1].id == value.func.value.id
            ):
                source_name = value.func.value.id
            coordinate = (scope, target)
            if (
                source_name is not None
                and (scope, source_name) in resolved
                and assignment_counts[(scope, source_name)] == 1
                and assignment_counts[(scope, target)] == 1
                and coordinate not in resolved
            ):
                resolved.add(coordinate)
                changed = True
    return resolved


def resolved_input_symbols(
    tree: ast.Module,
) -> tuple[
    dict[int, ast.AST],
    set[tuple[int, str]],
    set[tuple[int, str]],
    set[tuple[int, str]],
]:
    """Return parent links and manifest, binding, and relative-path symbols."""

    parents_by_id = parents(tree)
    manifest_keys: dict[tuple[int, str], set[str]] = {}
    for node in ast.walk(tree):
        scope = scope_id(node, parents=parents_by_id, tree=tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            key = subscript_key(node.slice)
            if isinstance(key, str):
                manifest_keys.setdefault((scope, node.value.id), set()).add(key)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.args
            and isinstance(subscript_key(node.args[0]), str)
        ):
            manifest_keys.setdefault((scope, node.func.value.id), set()).add(
                str(subscript_key(node.args[0]))
            )
    resolved_manifests = {
        coordinate
        for coordinate, keys in manifest_keys.items()
        if {"planner_declared_inputs", "inputs"} <= keys
    }
    resolved_manifests.update(
        _resolved_manifest_loads(tree, parents_by_id=parents_by_id)
    )

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
        scope = scope_id(node, parents=parents_by_id, tree=tree)
        manifest_name: str | None = None
        if (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and subscript_key(value.slice) == "inputs"
        ):
            manifest_name = value.value.id
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get"
            and isinstance(value.func.value, ast.Name)
            and value.args
            and subscript_key(value.args[0]) == "inputs"
        ):
            manifest_name = value.func.value.id
        if manifest_name is not None and (scope, manifest_name) in resolved_manifests:
            input_mappings.add((scope, target_name))
            continue
        if (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Subscript)
            and isinstance(value.value.value, ast.Name)
            and subscript_key(value.value.slice) == "inputs"
            and (scope, value.value.value.id) in resolved_manifests
        ):
            binding_names.add((scope, target_name))
        elif (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and (scope, value.value.id) in input_mappings
        ):
            binding_names.add((scope, target_name))
    relative_path_names: set[tuple[int, str]] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not (
                isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            scope = scope_id(node, parents=parents_by_id, tree=tree)
            value = node.value
            is_relative = bool(
                (
                    isinstance(value, ast.Name)
                    and (scope, value.id) in relative_path_names
                )
                or (
                    isinstance(value, ast.Subscript)
                    and isinstance(value.value, ast.Name)
                    and (scope, value.value.id) in binding_names
                    and subscript_key(value.slice) == "relative_path"
                )
                or (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "get"
                    and isinstance(value.func.value, ast.Name)
                    and (scope, value.func.value.id) in binding_names
                    and value.args
                    and subscript_key(value.args[0]) == "relative_path"
                )
            )
            coordinate = (scope, targets[0].id)
            if is_relative and coordinate not in relative_path_names:
                relative_path_names.add(coordinate)
                changed = True
    return parents_by_id, input_mappings, binding_names, relative_path_names


__all__ = ["resolved_input_symbols", "scope_id", "subscript_key"]
