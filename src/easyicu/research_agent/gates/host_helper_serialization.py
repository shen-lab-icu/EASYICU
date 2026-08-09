"""Serializing a host helper's typed result with ``json.dump``.

fresh17, step ``05_missingness_event_timing_audit_figure``. The figure itself
rendered completely -- PDF, PNG, SVG, TIFF and source data all written -- and
the step still failed, because the metadata sidecar was zero bytes::

    TypeError: Object of type FigureContract is not JSON serializable

``json.dump`` had opened the file, then raised while serializing, leaving an
empty file the contract reader could not parse.

The host's ``make_figure_contract`` returns a pydantic ``FigureContract`` which
carries ``to_json()`` and ``to_dict()``, and the host serializes it with
``model_dump_json`` in its own code. Its docstring documents only what it
*accepts*, so generated code hand-writes a ``default=`` hook each time and the
outcome depends on whether that hook happens to cover pydantic. Two figure
steps in the same run prove it is a coin flip: ``08``'s hook included
``to_dict`` and passed, ``05``'s handled numpy/pandas only and failed.

A sweep of 288 real generated scripts found 10 such call sites across
``make_figure_contract`` and ``run_cohort_summary_from_env``.

The check resolves the helper's declared return type from the host itself
rather than matching helper names, so a new helper returning a typed model is
covered the day it is added and nothing has to be re-listed here.
"""

from __future__ import annotations

import ast
import importlib
import typing
from typing import Dict, List, Optional, Set

from ..schema import ValidationFinding

_VALIDATOR = "host_helper_result_serialization"
_HOST_PREFIX = "easyicu.research_agent"

# Ways the value can legitimately reach ``json.dump``.
_SANCTIONED = ("to_json", "to_dict", "model_dump", "model_dump_json")


def _typed_model_helpers(tree: ast.Module) -> Dict[str, str]:
    """Imported host helpers whose declared return type is a pydantic model."""

    try:
        from pydantic import BaseModel
    except ModuleNotFoundError:  # pragma: no cover - pydantic is a hard dep
        return {}

    helpers: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith(_HOST_PREFIX):
            continue
        try:
            module = importlib.import_module(node.module)
        except Exception:  # noqa: BLE001 - an unimportable module is not our finding
            continue
        for alias in node.names:
            attribute = getattr(module, alias.name, None)
            if attribute is None or not callable(attribute):
                continue
            try:
                returns = typing.get_type_hints(attribute).get("return")
            except Exception:  # noqa: BLE001 - unresolvable hints are not our finding
                continue
            if isinstance(returns, type) and issubclass(returns, BaseModel):
                helpers[alias.asname or alias.name] = returns.__name__
    return helpers


def _names_bound_to_helper_results(
    tree: ast.Module,
    helpers: Dict[str, str],
) -> Dict[str, str]:
    """Single-name assignments whose value is a direct call to such a helper."""

    bound: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        helper = getattr(value.func, "id", None)
        if helper in helpers:
            bound[target.id] = helper
    return bound


def _rebound_elsewhere(tree: ast.Module, names: Set[str]) -> Set[str]:
    """Names reassigned from anything else -- their type is no longer known."""

    rebound: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        value = node.value
        if not (isinstance(value, ast.Call) and getattr(value.func, "id", None)):
            rebound.add(target.id)
    return rebound


def _default_hook_reaches_the_model(
    call: ast.Call,
    tree: ast.Module,
) -> bool:
    """Whether this ``json.dump`` supplies a hook that can reach the model.

    Deliberately coarse, and deliberately biased toward *not* reporting. A
    ``default=`` hook that mentions a sanctioned accessor is how the passing
    step in the same run serialized the very same type, so treating it as a
    defect would block working code and spend a repair on it -- the failure
    mode this gate exists to avoid paying for. A hook that never names one
    cannot produce anything but the observed TypeError.
    """

    hook = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "default"),
        None,
    )
    if hook is None:
        return False
    if isinstance(hook, ast.Attribute) and hook.attr in _SANCTIONED:
        return True
    if not isinstance(hook, ast.Name):
        # A lambda or call expression supplied inline: read it where it stands.
        return any(
            getattr(inner, "attr", None) in _SANCTIONED for inner in ast.walk(hook)
        )
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == hook.id
        ):
            if any(
                getattr(inner, "attr", None) in _SANCTIONED for inner in ast.walk(node)
            ):
                return True
    return False


def host_helper_result_serialization_findings(
    tree: Optional[ast.Module],
    *,
    script_text: str,
) -> List[ValidationFinding]:
    """Report ``json.dump(<typed helper result>)`` before it runs."""

    if tree is None:
        try:
            tree = ast.parse(str(script_text or ""))
        except SyntaxError:
            return []

    helpers = _typed_model_helpers(tree)
    if not helpers:
        return []
    bound = _names_bound_to_helper_results(tree, helpers)
    if not bound:
        return []
    # A name reassigned from a non-call expression may hold anything by now.
    for name in _rebound_elsewhere(tree, set(bound)):
        bound.pop(name, None)

    findings: List[ValidationFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr in {"dump", "dumps"}
            and getattr(function.value, "id", None) == "json"
        ):
            continue
        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        name = node.args[0].id
        helper = bound.get(name)
        if helper is None:
            continue
        if _default_hook_reaches_the_model(node, tree):
            continue
        model = helpers[helper]
        findings.append(
            ValidationFinding(
                validator=_VALIDATOR,
                severity="error",
                message=(
                    f"`json.dump({name})` cannot serialize the {model} that "
                    f"{helper}() returns, and fails only after opening the "
                    f"file -- leaving it empty. Write `{name}.to_json()` "
                    f"(or dump `{name}.to_dict()`) instead of relying on a "
                    "hand-written `default=` hook."
                ),
                detail={
                    "reason": "typed_helper_result_json_dumped",
                    "line": node.lineno,
                    "binding": name,
                    "helper": helper,
                    "returns": model,
                    "sanctioned_accessors": list(_SANCTIONED),
                },
            )
        )
    return findings


__all__ = ["host_helper_result_serialization_findings"]
