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


def _resolved_input_symbols(
    tree: ast.Module,
) -> tuple[
    dict[int, ast.AST],
    set[tuple[int, str]],
    set[tuple[int, str]],
]:
    """Return parent links, resolved-input mappings, and binding variables."""

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
    return parents, input_mappings, binding_names


def resolved_input_relative_path_root_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Require run-root resolution for host-issued ``relative_path`` values.

    Resolved-input paths already include the ``evidence/`` component. Joining
    one to ``EASYICU_EVIDENCE_DIR`` produces ``evidence/evidence/...``. The
    finding is limited to bindings proven to descend from the resolved-input
    manifest; arbitrary dictionaries remain outside host repair authority.
    """

    parents, _input_mappings, binding_names = _resolved_input_symbols(tree)

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


def resolved_input_shadowed_by_cohort_env_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject replacing an explicit typed product with ``COHORT_PARQUET``.

    ``COHORT_PARQUET`` is the step's generic execution cohort.  A Planner-
    declared typed artifact may instead be a transformed, digest-bound child
    produced by an upstream step.  When code has already verified that exact
    binding, demanding path equality with the generic cohort discards the
    declared product and breaks development projections.
    """

    parents, _input_mappings, binding_names = _resolved_input_symbols(tree)

    def environment_key(node: ast.AST, key: str) -> bool:
        return bool(
            (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "os"
                and node.value.attr == "environ"
                and _subscript_key(node.slice) == key
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os"
                and node.func.value.attr == "environ"
                and node.func.attr == "get"
                and node.args
                and _subscript_key(node.args[0]) == key
            )
        )

    def binding_relative_path(node: ast.AST, scope: int) -> bool:
        return bool(
            (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and (scope, node.value.id) in binding_names
                and _subscript_key(node.slice) == "relative_path"
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and (scope, node.func.value.id) in binding_names
                and node.args
                and _subscript_key(node.args[0]) == "relative_path"
            )
        )

    cohort_env_names: set[tuple[int, str]] = set()
    run_root_env_names: set[tuple[int, str]] = set()
    binding_relative_names: set[tuple[int, str]] = set()
    bound_paths: set[tuple[int, str]] = set()
    cohort_path_assignments: dict[tuple[int, str], ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        scope = _scope_id(node, parents=parents, tree=tree)
        if binding_relative_path(node.value, scope):
            binding_relative_names.add((scope, target.id))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        scope = _scope_id(node, parents=parents, tree=tree)
        if environment_key(node.value, "COHORT_PARQUET"):
            cohort_env_names.add((scope, target.id))
            continue
        if environment_key(node.value, "EASYICU_RUN_DIR"):
            run_root_env_names.add((scope, target.id))
            continue
        if (
            isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Div)
            and isinstance(node.value.left, ast.Call)
            and isinstance(node.value.left.func, ast.Name)
            and node.value.left.func.id == "Path"
            and len(node.value.left.args) == 1
            and (
                environment_key(node.value.left.args[0], "EASYICU_RUN_DIR")
                or (
                    isinstance(node.value.left.args[0], ast.Name)
                    and (scope, node.value.left.args[0].id) in run_root_env_names
                )
            )
            and (
                binding_relative_path(node.value.right, scope)
                or (
                    isinstance(node.value.right, ast.Name)
                    and (scope, node.value.right.id) in binding_relative_names
                )
            )
        ):
            bound_paths.add((scope, target.id))
            continue
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "Path"
            and len(node.value.args) == 1
            and (
                (
                    isinstance(node.value.args[0], ast.Name)
                    and (scope, node.value.args[0].id) in cohort_env_names
                )
                or environment_key(node.value.args[0], "COHORT_PARQUET")
            )
        ):
            cohort_path_assignments[(scope, target.id)] = node.value

    def resolved_name(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resolve"
            and isinstance(node.func.value, ast.Name)
        ):
            return node.func.value.id
        return None

    occurrences: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], (ast.NotEq, ast.IsNot))
            and len(node.test.comparators) == 1
            and any(isinstance(child, ast.Raise) for child in ast.walk(node))
        ):
            continue
        scope = _scope_id(node, parents=parents, tree=tree)
        left = resolved_name(node.test.left)
        right = resolved_name(node.test.comparators[0])
        pair = next(
            (
                (cohort_name, bound_name)
                for cohort_name, bound_name in ((left, right), (right, left))
                if cohort_name is not None
                and bound_name is not None
                and (scope, cohort_name) in cohort_path_assignments
                and (scope, bound_name) in bound_paths
            ),
            None,
        )
        if pair is None:
            continue
        assignment = cohort_path_assignments[(scope, pair[0])]
        if assignment.end_lineno is None or assignment.end_col_offset is None:
            continue
        occurrences.append(
            {
                "line": int(assignment.lineno),
                "column": int(assignment.col_offset),
                "end_line": int(assignment.end_lineno),
                "end_column": int(assignment.end_col_offset),
                "replacement_name": pair[1],
            }
        )
    if not occurrences:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "A resolved typed artifact is the declared physical input; "
                "do not replace it with or require path equality to the generic "
                "COHORT_PARQUET execution cohort."
            ),
            detail={
                "reason": "resolved_typed_input_shadowed_by_cohort_env",
                "occurrences": occurrences,
            },
        )
    ]


__all__ = [
    "resolved_input_relative_path_root_findings",
    "resolved_input_shadowed_by_cohort_env_findings",
]
