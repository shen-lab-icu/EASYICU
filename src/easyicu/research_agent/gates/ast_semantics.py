"""Small, case-neutral AST facts shared by mechanical gates."""

from __future__ import annotations

import ast
from collections.abc import Collection, Mapping

DYNAMIC_NAMESPACE_PRIMITIVES = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "import_module",
        "locals",
        "vars",
    }
)
DYNAMIC_NAMESPACE_MUTATORS = frozenset(
    {"__delattr__", "__setattr__", "delattr", "setattr"}
)


def literal_observational_getattr(
    node: ast.AST,
    *,
    protected_names: Collection[str] = (),
) -> bool:
    """Recognize a fixed, non-magic attribute read without granting reflection."""

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) in {2, 3}
        and not node.keywords
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return False
    attribute = node.args[1].value
    return bool(
        attribute
        and attribute not in protected_names
        and attribute not in DYNAMIC_NAMESPACE_PRIMITIVES
        and attribute not in DYNAMIC_NAMESPACE_MUTATORS
        and not (attribute.startswith("__") and attribute.endswith("__"))
    )


def contains_literal_provenance_audit_row(
    scope: ast.AST,
    *,
    tree: ast.Module,
    parents: Mapping[ast.AST, ast.AST],
    failure_keys: Collection[str],
) -> bool:
    """Return whether one scope contains a concrete numeric provenance row."""

    expected_function = None if scope is tree else scope

    def nearest_function(node: ast.AST) -> ast.AST | None:
        current = parents.get(node)
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            current = parents.get(current)
        return current

    for node in ast.walk(scope):
        if (
            not isinstance(node, ast.Dict)
            or nearest_function(node) is not expected_function
        ):
            continue
        fields = {
            str(key.value): value
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if all(
            isinstance(fields.get(key), ast.Constant) and fields[key].value is None
            for key in failure_keys
        ):
            continue
        role = fields.get("role")
        if (
            set(failure_keys) <= fields.keys()
            and isinstance(role, ast.Constant)
            and str(role.value).strip().lower() == "audit_only"
        ):
            return True
    return False


__all__ = [
    "DYNAMIC_NAMESPACE_MUTATORS",
    "DYNAMIC_NAMESPACE_PRIMITIVES",
    "contains_literal_provenance_audit_row",
    "literal_observational_getattr",
]
