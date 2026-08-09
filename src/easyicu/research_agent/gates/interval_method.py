"""Mechanical validation of confidence-interval method labels.

The owner of this contract is the generated-code preflight layer.  It only
checks that a statsmodels result's default ``conf_int()`` is not described as
a profile-likelihood interval; it does not choose or refit a model.
"""

from __future__ import annotations

import ast
from typing import Iterable

from ..schema import ValidationFinding

_STATSMODELS_MODEL_CLASSES = frozenset(
    {
        "GLM",
        "Logit",
        "MNLogit",
        "OLS",
        "Poisson",
        "Probit",
        "RLM",
        "WLS",
    }
)


def _assigned_names(targets: Iterable[ast.expr]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(_assigned_names(target.elts))
    return names


def _assignment_parts(node: ast.AST) -> tuple[list[ast.expr], ast.AST | None]:
    if isinstance(node, ast.Assign):
        return list(node.targets), node.value
    if isinstance(node, ast.AnnAssign):
        return [node.target], node.value
    return [], None


def _statsmodels_symbols(tree: ast.Module) -> tuple[set[str], set[str]]:
    module_aliases: set[str] = set()
    constructor_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"statsmodels", "statsmodels.api"}:
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "statsmodels",
            "statsmodels.api",
        }:
            for alias in node.names:
                if alias.name in _STATSMODELS_MODEL_CLASSES:
                    constructor_aliases.add(alias.asname or alias.name)
    return module_aliases, constructor_aliases


def _is_statsmodels_constructor(
    node: ast.AST,
    *,
    module_aliases: set[str],
    constructor_aliases: set[str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Name):
        return function.id in constructor_aliases
    return (
        isinstance(function, ast.Attribute)
        and function.attr in _STATSMODELS_MODEL_CLASSES
        and isinstance(function.value, ast.Name)
        and function.value.id in module_aliases
    )


def _is_fit_of_model(node: ast.AST, model_names: set[str]) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"fit", "fit_regularized"}
    ):
        return False
    owner = node.func.value
    return (
        isinstance(owner, ast.Name)
        and owner.id in model_names
        or isinstance(owner, ast.Call)
    )


def _result_names(
    tree: ast.Module,
    *,
    module_aliases: set[str],
    constructor_aliases: set[str],
) -> set[str]:
    model_names: set[str] = set()
    result_names: set[str] = set()
    for node in ast.walk(tree):
        targets, value = _assignment_parts(node)
        if value is None:
            continue
        names = _assigned_names(targets)
        if _is_statsmodels_constructor(
            value,
            module_aliases=module_aliases,
            constructor_aliases=constructor_aliases,
        ):
            model_names.update(names)
            continue
        if _is_fit_of_model(value, model_names):
            owner = value.func.value
            if isinstance(owner, ast.Call) and not _is_statsmodels_constructor(
                owner,
                module_aliases=module_aliases,
                constructor_aliases=constructor_aliases,
            ):
                continue
            result_names.update(names)
    return result_names


def _uses_default_conf_int(tree: ast.Module, result_names: set[str]) -> bool:
    return any(
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "conf_int"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in result_names
        for node in ast.walk(tree)
    )


def _profile_label_nodes(tree: ast.Module) -> list[ast.Constant]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (
            node.value == "profile_normal"
            or node.value.endswith("_profile_normal")
        )
    ]


def _wald_label(value: str) -> str:
    if value == "profile_normal":
        return "wald_95_percent"
    return value[: -len("profile_normal")] + "wald_95_percent"


def confidence_interval_method_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Return a finding when default statsmodels intervals are mislabeled."""

    module_aliases, constructor_aliases = _statsmodels_symbols(tree)
    if not module_aliases and not constructor_aliases:
        return []
    result_names = _result_names(
        tree,
        module_aliases=module_aliases,
        constructor_aliases=constructor_aliases,
    )
    if not result_names or not _uses_default_conf_int(tree, result_names):
        return []
    labels = _profile_label_nodes(tree)
    if not labels:
        return []
    occurrences = [
        {
            "line": int(node.lineno),
            "column": int(node.col_offset),
            "end_line": int(node.end_lineno or node.lineno),
            "end_column": int(node.end_col_offset or node.col_offset),
            "reported": str(node.value),
            "expected": _wald_label(str(node.value)),
        }
        for node in labels
    ]
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "Default statsmodels conf_int() intervals are Wald/asymptotic "
                "normal intervals, but the generated metadata labels them as "
                "profile likelihood."
            ),
            detail={
                "reason": "confidence_interval_method_mislabeled",
                "occurrence_count": len(occurrences),
                "occurrences": occurrences,
                "repair_safe": True,
            },
        )
    ]


__all__ = ["confidence_interval_method_findings"]
