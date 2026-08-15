"""Mechanical gates for host-issued typed-input bindings."""

from __future__ import annotations

import ast
from collections import Counter

from ..schema import ValidationFinding
from .typed_input_symbols import (
    environment_key_node,
    resolved_input_symbols as _resolved_input_symbols,
    scope_id as _scope_id,
    subscript_key as _subscript_key,
)


def resolved_input_relative_path_root_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Require run-root resolution for host-issued ``relative_path`` values.

    Resolved paths include ``evidence/``; joining them to the evidence directory
    duplicates it. Only proven manifest bindings receive host repair authority.
    """

    parents, _, binding_names, relative_path_names = _resolved_input_symbols(tree)

    def environment_key(node: ast.AST) -> ast.AST | None:
        return environment_key_node(node, "EASYICU_EVIDENCE_DIR")

    def is_binding_relative_path(node: ast.AST, *, scope: int) -> bool:
        if isinstance(node, ast.Name):
            return (scope, node.id) in relative_path_names
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

    def environment_value_is(node: ast.AST, expected: str) -> bool:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            return _subscript_key(node.slice) == expected
        return bool(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
            and node.func.value.attr == "environ"
            and node.func.attr == "get"
            and node.args
            and _subscript_key(node.args[0]) == expected
        )

    def path_from_environment(node: ast.AST, expected: str) -> bool:
        return bool(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and len(node.args) == 1
            and not node.keywords
            and environment_value_is(node.args[0], expected)
        )

    occurrences: list[dict[str, object]] = []
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
                "kind": "environment_key",
            }
        )

    assignment_counts: Counter[tuple[int, str]] = Counter()
    evidence_root_candidates: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        scope = _scope_id(node, parents=parents, tree=tree)
        coordinate = (scope, targets[0].id)
        assignment_counts[coordinate] += 1
        if path_from_environment(node.value, "EASYICU_EVIDENCE_DIR"):
            evidence_root_candidates.add(coordinate)
    evidence_root_names = {
        coordinate
        for coordinate in evidence_root_candidates
        if assignment_counts[coordinate] == 1
    }

    function_names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    unique_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and function_names.count(node.name) == 1
    }

    def call_argument(
        call: ast.Call,
        *,
        parameter: str,
        position: int,
    ) -> ast.AST | None:
        if position < len(call.args):
            return call.args[position]
        for keyword in call.keywords:
            if keyword.arg == parameter:
                return keyword.value
        return None

    def helper_relative_names(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        entry_parameter: str,
    ) -> set[str]:
        names: set[str] = set()
        changed = True
        while changed:
            changed = False
            for assignment in ast.walk(function):
                if not (
                    isinstance(assignment, (ast.Assign, ast.AnnAssign))
                    and assignment.value is not None
                ):
                    continue
                targets = (
                    assignment.targets
                    if isinstance(assignment, ast.Assign)
                    else [assignment.target]
                )
                if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                    continue
                value = assignment.value
                derived = bool(
                    isinstance(value, ast.Name)
                    and value.id in names
                    or isinstance(value, ast.Subscript)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == entry_parameter
                    and _subscript_key(value.slice) == "relative_path"
                    or isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "get"
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == entry_parameter
                    and value.args
                    and _subscript_key(value.args[0]) == "relative_path"
                )
                if derived and targets[0].id not in names:
                    names.add(targets[0].id)
                    changed = True
        return names

    calls_by_name: dict[str, list[ast.Call]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls_by_name.setdefault(node.func.id, []).append(node)

    for function_name, function in unique_functions.items():
        positional = [*function.args.posonlyargs, *function.args.args]
        parameter_positions = {
            parameter.arg: index for index, parameter in enumerate(positional)
        }
        calls = calls_by_name.get(function_name, [])
        if not calls:
            continue
        for entry_parameter in parameter_positions:
            relative_names = helper_relative_names(
                function,
                entry_parameter=entry_parameter,
            )
            if not relative_names:
                continue
            for root_parameter in parameter_positions:
                if root_parameter == entry_parameter:
                    continue
                root_nodes: list[ast.Name] = []
                for node in ast.walk(function):
                    if not (
                        isinstance(node, ast.BinOp)
                        and isinstance(node.op, ast.Div)
                        and isinstance(node.left, ast.Call)
                        and isinstance(node.left.func, ast.Name)
                        and node.left.func.id == "Path"
                        and len(node.left.args) == 1
                        and isinstance(node.left.args[0], ast.Name)
                        and node.left.args[0].id == root_parameter
                        and isinstance(node.right, ast.Name)
                        and node.right.id in relative_names
                    ):
                        continue
                    root_nodes.append(node.left.args[0])
                if not root_nodes:
                    continue
                loaded_roots = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == root_parameter
                ]
                if {id(node) for node in loaded_roots} != {
                    id(node) for node in root_nodes
                }:
                    continue
                calls_proven = True
                for call in calls:
                    call_scope = _scope_id(call, parents=parents, tree=tree)
                    entry_argument = call_argument(
                        call,
                        parameter=entry_parameter,
                        position=parameter_positions[entry_parameter],
                    )
                    root_argument = call_argument(
                        call,
                        parameter=root_parameter,
                        position=parameter_positions[root_parameter],
                    )
                    binding_proven = bool(
                        isinstance(entry_argument, ast.Name)
                        and (call_scope, entry_argument.id) in binding_names
                        and assignment_counts[(call_scope, entry_argument.id)] == 1
                    )
                    evidence_root_proven = bool(
                        path_from_environment(
                            root_argument,
                            "EASYICU_EVIDENCE_DIR",
                        )
                        if root_argument is not None
                        else False
                    ) or bool(
                        isinstance(root_argument, ast.Name)
                        and (call_scope, root_argument.id) in evidence_root_names
                    )
                    if not binding_proven or not evidence_root_proven:
                        calls_proven = False
                        break
                if not calls_proven:
                    continue
                for root_node in root_nodes:
                    if root_node.end_lineno is None or root_node.end_col_offset is None:
                        continue
                    occurrences.append(
                        {
                            "line": int(root_node.lineno),
                            "column": int(root_node.col_offset),
                            "end_line": int(root_node.end_lineno),
                            "end_column": int(root_node.end_col_offset),
                            "kind": "root_parameter",
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

    parents, _, binding_names, _ = _resolved_input_symbols(tree)

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
