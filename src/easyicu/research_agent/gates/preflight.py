"""Deterministic mechanical checks that run before semantic LLM review.

These checks reject implementation shortcuts only.  They do not select or
rewrite the planner-owned exposure, outcome, cohort, method, or estimand.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..icu_rules import companion_count_column_for_measured
from ..research_context.prompt_scope import normalised_method_head
from ..schema import AnalysisStep, ValidationFinding
from .ast_semantics import (
    DYNAMIC_NAMESPACE_MUTATORS as _DYNAMIC_NAMESPACE_MUTATORS,
    DYNAMIC_NAMESPACE_PRIMITIVES as _DYNAMIC_NAMESPACE_PRIMITIVES,
    call_name as _call_name,
    contains_literal_provenance_audit_row,
    execution_cohort_row_count_findings as _cohort_count_findings,
    literal_observational_getattr,
    pre312_fstring_subscript_quote_occurrences,
    runtime_context_opaque_level_findings as _runtime_context_level_findings,
)
from .binary_feasibility import binary_feasibility_guard_findings
from .host_helper_result import (
    host_helper_result_findings,
    table_one_spec_binding_findings,
)
from .numeric_reduction import is_array_boolean_predicate as _is_array_boolean_predicate
from .numeric_reduction import is_proven_array_boolean_predicate
from .numeric_reduction import misnested_boolean_mask_reduction_expression
from .numeric_reduction import unambiguous_array_predicate_aliases
from . import coercion_guard
from .typed_input import (
    resolved_input_relative_path_root_findings,
    resolved_input_shadowed_by_cohort_env_findings,
)
from .typed_binding_identity import direct_resolved_input_key_findings

_TRY_STAR_NODE_TYPES = (
    (try_star_type,) if (try_star_type := getattr(ast, "TryStar", None)) else ()
)
_TRY_NODE_TYPES = (ast.Try, *_TRY_STAR_NODE_TYPES)
_TYPE_PARAMETER_NODE_TYPES = tuple(
    node_type
    for name in ("TypeVar", "ParamSpec", "TypeVarTuple")
    if (node_type := getattr(ast, name, None)) is not None
)
_STRUCTURAL_ACCOUNTING_PRODUCTS = frozenset(
    {
        "attrition",
        "cohort_accounting",
        "cohort_flow",
        "denominator_reconciliation",
        "source_availability",
        "source_availability_audit",
        "universe_count_reconciliation",
    }
)
_RENDER_METHODS = frozenset(
    {"figure", "publication_figure", "visualization", "descriptive_visualization"}
)


def _is_frame_columns(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "columns"


def _function_arbitrary_column_fallback(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Optional[tuple[int, str]]:
    """Find a fallback that returns a dtype-compatible frame-order column."""

    candidate_return_seen = False
    for node in ast.walk(function):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            loop_name = node.target.id
            if not _is_frame_columns(node.iter):
                for nested in ast.walk(node):
                    if isinstance(nested, ast.Return):
                        candidate_return_seen = True
                continue
            returns_loop_column = any(
                isinstance(nested, ast.Return)
                and isinstance(nested.value, ast.Name)
                and nested.value.id == loop_name
                for nested in ast.walk(node)
            )
            if returns_loop_column and candidate_return_seen:
                return int(node.lineno), function.name

    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript):
            continue
        base_name = _call_name(node.value)
        index = node.slice
        is_first = isinstance(index, ast.Constant) and index.value == 0
        if is_first and ("select_dtypes" in base_name or base_name.endswith("columns")):
            return int(node.lineno), function.name
    return None


def _typed_input_products(step: AnalysisStep) -> set[str]:
    products = set()
    for raw in step.inputs or []:
        kind, separator, name = str(raw or "").strip().lower().partition(":")
        if separator and kind == "table" and name:
            products.add(name)
    return products


def _mask_name_from_slice(node: ast.AST) -> Optional[str]:
    target = node
    if isinstance(target, ast.Tuple) and target.elts:
        target = target.elts[0]
    if isinstance(target, ast.Name):
        return target.id
    return None


def _subscript_frame_name(node: ast.AST) -> str:
    """Return the DataFrame name addressed by ``frame[mask]`` or ``frame.loc``."""

    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and node.attr in {"loc", "iloc"}
        and isinstance(node.value, ast.Name)
    ):
        return node.value.id
    return ""


def _is_mask_method_call(node: ast.AST, mask_name: str, method: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == mask_name
    )


def _is_len_call(node: ast.AST, frame_names: set[str]) -> bool:
    """Return whether ``len`` measures the same frame that will be filtered."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in frame_names
    )


def _mask_incomplete_test(
    test: ast.AST,
    mask_name: str,
    frame_names: set[str],
) -> bool:
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _is_mask_method_call(test.operand, mask_name, "all")
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Invert):
        return _is_mask_method_call(test.operand, mask_name, "all")

    if (
        isinstance(test, ast.Call)
        and not test.args
        and not test.keywords
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "any"
        and isinstance(test.func.value, ast.UnaryOp)
        and isinstance(test.func.value.op, ast.Invert)
        and isinstance(test.func.value.operand, ast.Name)
        and test.func.value.operand.id == mask_name
    ):
        return True
    if (
        isinstance(test, ast.Call)
        and not test.args
        and not test.keywords
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "any"
        and isinstance(test.func.value, ast.Call)
        and isinstance(test.func.value.func, ast.Attribute)
        and test.func.value.func.attr == "eq"
        and isinstance(test.func.value.func.value, ast.Name)
        and test.func.value.func.value.id == mask_name
        and len(test.func.value.args) == 1
        and isinstance(test.func.value.args[0], ast.Constant)
        and test.func.value.args[0].value is False
    ):
        return True

    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    operator = test.ops[0]
    if isinstance(operator, (ast.Eq, ast.Is)):
        return (
            _is_mask_method_call(left, mask_name, "all")
            and isinstance(right, ast.Constant)
            and right.value is False
        ) or (
            _is_mask_method_call(right, mask_name, "all")
            and isinstance(left, ast.Constant)
            and left.value is False
        )
    if isinstance(operator, (ast.NotEq, ast.IsNot)):
        return (
            _is_mask_method_call(left, mask_name, "sum")
            and _is_len_call(right, frame_names)
        ) or (
            _is_mask_method_call(right, mask_name, "sum")
            and _is_len_call(left, frame_names)
        )
    return False


def _mask_complete_test(
    test: ast.AST,
    mask_name: str,
    frame_names: set[str],
) -> bool:
    if _is_mask_method_call(test, mask_name, "all"):
        return True
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    if not isinstance(test.ops[0], (ast.Eq, ast.Is)):
        return False
    return (
        _is_mask_method_call(left, mask_name, "sum")
        and _is_len_call(right, frame_names)
    ) or (
        _is_mask_method_call(right, mask_name, "sum")
        and _is_len_call(left, frame_names)
    )


def _is_raise_only_guard(
    statement: ast.stmt,
    mask_name: str,
    frame_names: set[str],
) -> bool:
    if isinstance(statement, ast.Assert):
        return _mask_complete_test(statement.test, mask_name, frame_names)
    if not isinstance(statement, ast.If) or not statement.body:
        return False
    if not all(isinstance(item, (ast.Raise, ast.Return)) for item in statement.body):
        return False
    return _mask_incomplete_test(statement.test, mask_name, frame_names)


def _is_boolean_mask_expression(node: ast.AST) -> bool:
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        return True
    if isinstance(node, ast.Call):
        return _call_name(node.func).split(".")[-1] in {
            "between",
            "eq",
            "ge",
            "gt",
            "isna",
            "isin",
            "le",
            "lt",
            "ne",
            "notna",
        }
    return False


def _structural_filter_findings(
    tree: ast.Module, step: AnalysisStep
) -> list[ValidationFinding]:
    if normalised_method_head(step.method) not in _RENDER_METHODS:
        return []
    accounting_products = _typed_input_products(step) & _STRUCTURAL_ACCOUNTING_PRODUCTS
    if not accounting_products:
        return []

    findings = []
    for owner in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        body = getattr(owner, "body", [])
        prior_guards: dict[str, list[ast.stmt]] = {}
        mask_names: set[str] = set()
        mask_sources: dict[str, set[str]] = {}
        frame_aliases: dict[str, set[str]] = {}
        for statement in body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                alias_source: Optional[str] = None
                if isinstance(value, ast.Name):
                    alias_source = value.id
                elif (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "copy"
                    and isinstance(value.func.value, ast.Name)
                ):
                    alias_source = value.func.value.id
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    frame_aliases.pop(target.id, None)
                    if alias_source:
                        frame_aliases[target.id] = {
                            alias_source,
                            *frame_aliases.get(alias_source, set()),
                        }
                if value is not None and _is_boolean_mask_expression(value):
                    for target in targets:
                        if not isinstance(target, ast.Name):
                            continue
                        mask_names.add(target.id)
                        mask_sources[target.id] = _referenced_names(value)
            for node in ast.walk(statement):
                if not isinstance(node, ast.Subscript):
                    continue
                mask_name = _mask_name_from_slice(node.slice)
                value_name = _call_name(node.value)
                source_names = mask_sources.get(mask_name or "", set())
                direct_name = _subscript_frame_name(node.value)
                direct_sources = {
                    direct_name,
                    *frame_aliases.get(direct_name, set()),
                }
                equivalent_frame_names = {
                    *direct_sources,
                    *{
                        alias
                        for alias, sources in frame_aliases.items()
                        if direct_name in sources
                    },
                }
                is_row_filter = bool(
                    mask_name
                    and (
                        value_name.endswith(".loc")
                        or (direct_name and source_names & direct_sources)
                    )
                )
                if not is_row_filter:
                    continue
                if (
                    not mask_name
                    or mask_name not in mask_names
                    or any(
                        _is_raise_only_guard(
                            guard,
                            mask_name,
                            equivalent_frame_names,
                        )
                        for guard in prior_guards.get(mask_name, [])
                    )
                ):
                    continue
                findings.append(
                    ValidationFinding(
                        validator="mechanical_code_preflight",
                        severity="error",
                        message=(
                            "A rendering-only structural-accounting step filters "
                            "rows before plotting without first failing closed when "
                            "the validation mask is incomplete."
                        ),
                        detail={
                            "reason": "structural_accounting_filter",
                            "line": int(node.lineno),
                            "mask": mask_name,
                            "typed_products": sorted(accounting_products),
                        },
                    )
                )
            for possible_mask in {
                node.id for node in ast.walk(statement) if isinstance(node, ast.Name)
            }:
                if isinstance(statement, (ast.Assert, ast.If)):
                    prior_guards.setdefault(possible_mask, []).append(statement)
    return findings


def _uses_zero_decimal_count_rendering(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FormattedValue) or node.format_spec is None:
            continue
        try:
            format_spec = ast.unparse(node.format_spec).lower()
        except Exception:
            continue
        if ".0f" in format_spec:
            return True
    return False


def _has_integer_like_accounting_guard(tree: ast.Module) -> bool:
    def _has_integer_operation(test: ast.AST) -> bool:
        for candidate in ast.walk(test):
            if isinstance(candidate, ast.Call):
                function_name = _call_name(candidate.func).split(".")[-1].lower()
                if function_name in {"round", "rint", "floor", "mod", "is_integer"}:
                    return True
            if isinstance(candidate, ast.BinOp) and isinstance(candidate.op, ast.Mod):
                return True
        return False

    def _accounting_identifier(value: object) -> bool:
        token = str(value or "").strip().lower()
        if token == "n":
            return True
        parts = {part for part in re.split(r"[^a-z0-9]+", token) if part}
        return bool(parts & {"count", "counts", "numerator", "denominator"})

    def _guard_references_accounting_value(test: ast.AST) -> bool:
        call_functions = {
            id(node.func) for node in ast.walk(test) if isinstance(node, ast.Call)
        }
        for candidate in ast.walk(test):
            if id(candidate) in call_functions:
                continue
            if isinstance(candidate, ast.Name) and _accounting_identifier(candidate.id):
                return True
            if isinstance(candidate, ast.Subscript):
                key = _subscript_key(candidate.slice)
                if key is not None and _accounting_identifier(key):
                    return True
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not _has_integer_operation(node.test):
            continue
        if not _guard_references_accounting_value(node.test):
            continue
        body_text = "\n".join(ast.unparse(statement).lower() for statement in node.body)
        if "raise " in body_text or ".append(" in body_text or "return " in body_text:
            return True
    return False


def _structural_integer_findings(
    tree: ast.Module, step: AnalysisStep
) -> list[ValidationFinding]:
    if normalised_method_head(step.method) not in _RENDER_METHODS:
        return []
    accounting_products = _typed_input_products(step) & _STRUCTURAL_ACCOUNTING_PRODUCTS
    if not accounting_products or not _uses_zero_decimal_count_rendering(tree):
        return []
    if _has_integer_like_accounting_guard(tree):
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "A rendering-only structural-accounting step formats counts as "
                "whole numbers without first failing closed on fractional "
                "count values."
            ),
            detail={
                "reason": "structural_accounting_integer_validation",
                "typed_products": sorted(accounting_products),
            },
        )
    ]


def _subscript_key(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _binding_metadata_findings(tree: ast.Module) -> list[ValidationFinding]:
    required_keys: set[str] = set()
    literal_keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            literal_keys.update(
                key
                for key in (_subscript_key(item) for item in node.keys)
                if key is not None
            )
        if not isinstance(node, ast.Subscript):
            continue
        outer_key = _subscript_key(node.slice)
        if outer_key is None or not isinstance(node.value, ast.Subscript):
            continue
        inner_key = _subscript_key(node.value.slice)
        if inner_key == "binding":
            required_keys.add(outer_key)
    missing = sorted(required_keys - literal_keys)
    if not missing:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "Typed-input metadata is read later from a local binding record "
                "but is never persisted into any constructed binding record."
            ),
            detail={
                "reason": "unpersisted_binding_metadata",
                "missing_keys": missing,
            },
        )
    ]


_PROVENANCE_FAILURE_KEYS = frozenset({"invalid_pair_n", "discordant_n"})
_PROVENANCE_FAILURE_DECISION_KEYS = frozenset({"fail_closed"})
_PROVENANCE_SUCCESS_DECISION_KEYS = frozenset(
    {"completed_step_allowed", "provenance_valid"}
)
_PROVENANCE_FULL_COVERAGE = frozenset(_PROVENANCE_FAILURE_KEYS)
_PROVENANCE_FAILURE = "failure"
_PROVENANCE_SUCCESS = "success"
_PROVENANCE_LOOP_SENTINEL = "_easyicu_provenance_loop_observed"
_FLOW_FALLTHROUGH = "fallthrough"
_FLOW_FUNCTION_EXIT = "function_exit"
_FLOW_LOOP_ESCAPE = "loop_escape"


def _literal_string_tokens(node: ast.AST) -> set[str]:
    return {
        str(candidate.value).strip().lower()
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    }


def _referenced_names(node: ast.AST) -> set[str]:
    return {
        candidate.id for candidate in ast.walk(node) if isinstance(candidate, ast.Name)
    }


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return {name for item in node.elts for name in _target_names(item)}
    return set()


_REFLECTION_MODULE_ROOTS = frozenset(
    {"__main__", "gc", "importlib", "inspect", "pkgutil", "pydoc", "unittest"}
)


def _has_dynamic_namespace_indirection(tree: ast.Module) -> bool:
    """Reject optional proofs when dynamic namespace tools escape direct calls.

    A direct ``exec(...)`` or ``globals(...)`` call is easy for each proof to
    reject.  Assigning the callable first (``runner = exec``), or retrieving it
    from ``builtins``, otherwise hides the same operation behind an arbitrary
    name.  These preflight proofs are optional, so declining under that
    ambiguity is safer than trying to interpret dynamically generated code.
    """

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    builtin_roots = {"builtins", "__builtins__"}
    operator_roots = {"operator"}
    reflection_roots = set(_REFLECTION_MODULE_ROOTS)
    sys_roots = {"sys"}
    operator_accessors = {"delitem", "getitem", "setitem"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.split(".", 1)[0] in _REFLECTION_MODULE_ROOTS
                for alias in node.names
            ):
                return True
            builtin_roots.update(
                alias.asname or "builtins"
                for alias in node.names
                if alias.name == "builtins"
            )
            operator_roots.update(
                alias.asname or "operator"
                for alias in node.names
                if alias.name == "operator"
            )
            sys_roots.update(
                alias.asname or "sys" for alias in node.names if alias.name == "sys"
            )
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".", 1)[
            0
        ] in (_REFLECTION_MODULE_ROOTS | {"builtins", "operator"}):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "sys"
            and any(alias.name in {"_getframe", "modules"} for alias in node.names)
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "inspect"
            and any(alias.name == "currentframe" for alias in node.names)
        ):
            return True

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            if _mapping_root_name(node.value) not in builtin_roots:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            aliases = {name for target in targets for name in _target_names(target)}
            if not aliases <= builtin_roots:
                builtin_roots.update(aliases)
                changed = True

    protected = _DYNAMIC_NAMESPACE_PRIMITIVES | _DYNAMIC_NAMESPACE_MUTATORS
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id
            in (builtin_roots | operator_roots | reflection_roots | sys_roots)
            and isinstance(node.ctx, (ast.Load, ast.Store, ast.Del))
        ):
            return True
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and (
                (
                    node.attr.startswith("__")
                    and node.attr.endswith("__")
                    and node.attr != "__name__"
                )
                or node.attr in {"f_builtins", "f_globals", "f_locals"}
                or (node.attr == "modules" and _mapping_root_name(node) in sys_roots)
            )
        ):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            root = _mapping_root_name(node.func.value)
            method = node.func.attr
            if root in operator_roots and method in operator_accessors:
                return True
            if method in {"__delattr__", "__setattr__"}:
                return True
            if root in builtin_roots and method in {
                "__getattribute__",
                "__getitem__",
                "get",
                "pop",
                "setdefault",
            }:
                key = _subscript_key(node.args[0]) if node.args else None
                if key is None or key in protected:
                    return True
        if isinstance(node, ast.Call) and (
            _call_name(node.func).rsplit(".", 1)[-1]
            in {"__getattribute__", "_getframe", "currentframe"}
        ):
            return True
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in protected:
                continue
            parent = parents.get(node)
            if not (isinstance(parent, ast.Call) and parent.func is node):
                return True
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and (
                (_mapping_root_name(node) in builtin_roots and node.attr in protected)
                or (
                    _mapping_root_name(node) in operator_roots
                    and node.attr in operator_accessors
                )
            )
        ):
            return True
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and _mapping_root_name(node) in builtin_roots
            and _subscript_key(node.slice) in protected
        ):
            return True
    return False


def _expression_identity(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _mapping_access_key(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Subscript):
        return _subscript_key(node.slice)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
    ):
        return _subscript_key(node.args[0])
    return None


def _mapping_root_name(node: ast.AST) -> Optional[str]:
    current = node
    if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        current = current.func.value
    while isinstance(current, (ast.Subscript, ast.Attribute)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _literal_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and node.value == 0
    )


def _literal_bool(node: ast.AST) -> Optional[bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _swap_provenance_meaning(
    meaning: Optional[tuple[str, frozenset[str]]],
) -> Optional[tuple[str, frozenset[str]]]:
    if meaning is None:
        return None
    kind, coverage = meaning
    inverse = (
        _PROVENANCE_SUCCESS if kind == _PROVENANCE_FAILURE else _PROVENANCE_FAILURE
    )
    return inverse, coverage


def _provenance_predicate_meaning(
    node: ast.AST,
    *,
    expression_roles: dict[str, frozenset[str]],
    signal_meanings: dict[str, tuple[str, frozenset[str]]],
    assignments: dict[str, ast.AST],
    audit_containers: set[str],
    seen_names: Optional[set[str]] = None,
) -> Optional[tuple[str, frozenset[str]]]:
    """Classify provenance failure/success polarity without name heuristics."""

    seen_names = set(seen_names or set())
    expression_coverage = expression_roles.get(_expression_identity(node))
    if expression_coverage:
        return _PROVENANCE_FAILURE, expression_coverage

    if isinstance(node, ast.Name):
        if node.id in seen_names or node.id not in assignments:
            return None
        seen_names.add(node.id)
        return _provenance_predicate_meaning(
            assignments[node.id],
            expression_roles=expression_roles,
            signal_meanings=signal_meanings,
            assignments=assignments,
            audit_containers=audit_containers,
            seen_names=seen_names,
        )

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
        return _swap_provenance_meaning(
            _provenance_predicate_meaning(
                node.operand,
                expression_roles=expression_roles,
                signal_meanings=signal_meanings,
                assignments=assignments,
                audit_containers=audit_containers,
                seen_names=seen_names,
            )
        )

    access_key = _mapping_access_key(node)
    access_root = _mapping_root_name(node)
    if access_root in audit_containers:
        if access_key in _PROVENANCE_FAILURE_KEYS:
            return _PROVENANCE_FAILURE, frozenset({access_key})
        if access_key in _PROVENANCE_FAILURE_DECISION_KEYS:
            return _PROVENANCE_FAILURE, _PROVENANCE_FULL_COVERAGE
        if access_key in _PROVENANCE_SUCCESS_DECISION_KEYS:
            return _PROVENANCE_SUCCESS, _PROVENANCE_FULL_COVERAGE

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        method = node.func.attr
        base = node.func.value
        inverted = isinstance(base, ast.UnaryOp) and isinstance(
            base.op, (ast.Not, ast.Invert)
        )
        if inverted:
            base = base.operand
        if isinstance(base, ast.Name) and base.id in signal_meanings:
            meaning = signal_meanings[base.id]
            if inverted:
                meaning = _swap_provenance_meaning(meaning)
            if method == "all" and meaning and meaning[0] == _PROVENANCE_SUCCESS:
                return meaning
            if (
                method in {"any", "sum"}
                and meaning
                and meaning[0] == _PROVENANCE_FAILURE
            ):
                return meaning

    if isinstance(node, ast.BoolOp):
        meanings = [
            _provenance_predicate_meaning(
                value,
                expression_roles=expression_roles,
                signal_meanings=signal_meanings,
                assignments=assignments,
                audit_containers=audit_containers,
                seen_names=seen_names,
            )
            for value in node.values
        ]
        wanted = (
            _PROVENANCE_FAILURE
            if isinstance(node.op, ast.Or)
            else _PROVENANCE_SUCCESS if isinstance(node.op, ast.And) else None
        )
        selected = [item for item in meanings if item and item[0] == wanted]
        if wanted and selected:
            coverage = frozenset().union(*(item[1] for item in selected))
            return wanted, coverage
        return None

    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    left = node.left
    right = node.comparators[0]
    operator = node.ops[0]

    left_key = _mapping_access_key(left)
    left_root = _mapping_root_name(left)
    if (
        left_root in audit_containers
        and left_key == "status"
        and isinstance(right, ast.Constant)
        and str(right.value).strip().lower() == "checked"
    ):
        if isinstance(operator, (ast.NotEq, ast.IsNot)):
            return _PROVENANCE_FAILURE, _PROVENANCE_FULL_COVERAGE
        if isinstance(operator, (ast.Eq, ast.Is)):
            return _PROVENANCE_SUCCESS, _PROVENANCE_FULL_COVERAGE

    left_meaning = _provenance_predicate_meaning(
        left,
        expression_roles=expression_roles,
        signal_meanings=signal_meanings,
        assignments=assignments,
        audit_containers=audit_containers,
        seen_names=seen_names,
    )
    right_meaning = _provenance_predicate_meaning(
        right,
        expression_roles=expression_roles,
        signal_meanings=signal_meanings,
        assignments=assignments,
        audit_containers=audit_containers,
        seen_names=seen_names,
    )

    if left_meaning and _literal_zero(right):
        kind, coverage = left_meaning
        if kind != _PROVENANCE_FAILURE:
            return None
        if isinstance(operator, (ast.Eq, ast.Is, ast.LtE)):
            return _PROVENANCE_SUCCESS, coverage
        if isinstance(operator, (ast.NotEq, ast.IsNot, ast.Gt, ast.GtE)):
            return _PROVENANCE_FAILURE, coverage
    if right_meaning and _literal_zero(left):
        kind, coverage = right_meaning
        if kind != _PROVENANCE_FAILURE:
            return None
        if isinstance(operator, (ast.Eq, ast.Is, ast.GtE)):
            return _PROVENANCE_SUCCESS, coverage
        if isinstance(operator, (ast.NotEq, ast.IsNot, ast.Lt, ast.LtE)):
            return _PROVENANCE_FAILURE, coverage

    bool_value = _literal_bool(right)
    if (
        left_meaning
        and bool_value is not None
        and isinstance(operator, (ast.Eq, ast.Is, ast.NotEq, ast.IsNot))
    ):
        same = isinstance(operator, (ast.Eq, ast.Is)) == bool_value
        return left_meaning if same else _swap_provenance_meaning(left_meaning)
    return None


def _statement_flow_outcomes(statement: ast.stmt) -> set[str]:
    if isinstance(statement, (ast.Raise, ast.Return)):
        return {_FLOW_FUNCTION_EXIT}
    if isinstance(statement, (ast.Break, ast.Continue)):
        return {_FLOW_LOOP_ESCAPE}
    if isinstance(statement, ast.If):
        body = _block_flow_outcomes(statement.body)
        orelse = (
            _block_flow_outcomes(statement.orelse)
            if statement.orelse
            else {_FLOW_FALLTHROUGH}
        )
        return body | orelse
    if isinstance(
        statement,
        (ast.For, ast.AsyncFor, ast.While, *_TRY_NODE_TYPES, ast.Match),
    ):
        return {_FLOW_FALLTHROUGH, _FLOW_LOOP_ESCAPE}
    return {_FLOW_FALLTHROUGH}


def _block_flow_outcomes(statements: list[ast.stmt]) -> set[str]:
    outcomes = {_FLOW_FALLTHROUGH}
    for statement in statements:
        if _FLOW_FALLTHROUGH not in outcomes:
            break
        outcomes.remove(_FLOW_FALLTHROUGH)
        outcomes.update(_statement_flow_outcomes(statement))
    return outcomes


def _branch_all_paths_exit(statements: list[ast.stmt]) -> bool:
    return _block_flow_outcomes(statements) == {_FLOW_FUNCTION_EXIT}


def _has_unrelated_control_ancestor(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    current = parents.get(node)
    while current is not None and not isinstance(
        current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
    ):
        if isinstance(current, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            return True
        current = parents.get(current)
    return False


_PROVENANCE_RESULT_SINK_METHODS = frozenset(
    {
        "fit",
        "fit_regularized",
        "predict",
        "dump",
        "save",
        "savefig",
        "savetxt",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_json",
        "to_parquet",
        "to_pickle",
        "write_bytes",
        "write_text",
    }
)


def _is_provenance_result_sink_call(candidate: ast.Call) -> bool:
    call_name = _call_name(candidate.func).lower()
    method = call_name.rsplit(".", 1)[-1]
    return (
        method in _PROVENANCE_RESULT_SINK_METHODS
        or "write_success" in method
        or (method.startswith("write_") and not method.startswith("write_failed"))
        or method.startswith("publish")
    )


def _provenance_branch_contains_result_sink(statements: list[ast.stmt]) -> bool:
    """Return whether an executed guard branch can publish scientific output."""

    found = False

    class _SinkVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            nonlocal found
            if _is_provenance_result_sink_call(node):
                found = True
                return
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

    visitor = _SinkVisitor()
    for statement in statements:
        visitor.visit(statement)
        if found:
            return True
    return False


def _result_sink_precedes_guard(guard: ast.If, parents: dict[ast.AST, ast.AST]) -> bool:
    scope: ast.AST = guard
    while scope in parents and not isinstance(
        scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
    ):
        scope = parents[scope]
    guard_line = int(getattr(guard, "lineno", 0) or 0)
    for candidate in ast.walk(scope):
        if not isinstance(candidate, ast.Call):
            continue
        current: Optional[ast.AST] = candidate
        while current in parents and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            current = parents[current]
        if current is not scope:
            continue
        line = int(getattr(candidate, "lineno", 0) or 0)
        if not line or line >= guard_line:
            continue
        if _is_provenance_result_sink_call(candidate):
            return True
    return False


def _provenance_signal_source(value: ast.AST) -> Optional[tuple[str, str]]:
    current = value
    if (
        isinstance(current, ast.Call)
        and isinstance(current.func, ast.Name)
        and current.func.id in {"bool", "float", "int"}
        and len(current.args) == 1
    ):
        current = current.args[0]
    if isinstance(current, ast.Name):
        return current.id, _PROVENANCE_FAILURE
    if not (
        isinstance(current, ast.Call)
        and isinstance(current.func, ast.Attribute)
        and current.func.attr in {"any", "sum"}
    ):
        return None
    base = current.func.value
    if isinstance(base, ast.Name):
        return base.id, _PROVENANCE_FAILURE
    if (
        isinstance(base, ast.UnaryOp)
        and isinstance(base.op, (ast.Not, ast.Invert))
        and isinstance(base.operand, ast.Name)
    ):
        return base.operand.id, _PROVENANCE_SUCCESS
    return None


def _provenance_fail_closed_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Require a terminating guard for an implemented provenance failure audit."""

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    id_parents, statement_positions = _ast_parent_and_statement_positions(tree)

    def _nearest_function(node: ast.AST) -> Optional[ast.AST]:
        current = parents.get(node)
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            current = parents.get(current)
        return current

    def _contains_literal_audit_row(scope: ast.AST) -> bool:
        return contains_literal_provenance_audit_row(
            scope,
            tree=tree,
            parents=parents,
            failure_keys=_PROVENANCE_FAILURE_KEYS,
        )

    def _uses_host_provenance_receipt(scope: ast.AST) -> bool:
        """Leave fail-closed semantics of the host receipt to its own gate."""

        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "measurement_provenance_receipt"
            and _nearest_function(node) is scope
            for node in ast.walk(scope)
        )

    marker_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _contains_literal_audit_row(node)
        and not _uses_host_provenance_receipt(node)
    ]
    module_is_marker = _contains_literal_audit_row(tree)
    if not marker_nodes and not module_is_marker:
        return []
    marker_names = {node.name for node in marker_nodes}
    all_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    ambiguous_names = {
        name
        for name in marker_names
        if sum(node.name == name for node in all_functions) != 1
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in marker_names
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            ambiguous_names.add(node.id)
        if isinstance(node, ast.ClassDef) and node.name in marker_names:
            ambiguous_names.add(node.name)
        if isinstance(node, ast.ExceptHandler) and node.name in marker_names:
            ambiguous_names.add(str(node.name))
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in marker_names:
            ambiguous_names.add(str(node.name))
        if isinstance(node, ast.MatchMapping) and node.rest in marker_names:
            ambiguous_names.add(str(node.rest))
        if isinstance(node, _TYPE_PARAMETER_NODE_TYPES) and (
            node.name in marker_names
        ):
            ambiguous_names.add(node.name)
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        bound_names = {name for target in targets for name in _target_names(target)}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound_names.update(
                argument.arg
                for argument in [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
            )
            if node.args.vararg is not None:
                bound_names.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                bound_names.add(node.args.kwarg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound_names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        ambiguous_names.update(marker_names & bound_names)
    marker_functions = {
        node.name: node for node in marker_nodes if node.name not in ambiguous_names
    }
    module_scope_key = "<easyicu-module-provenance-scope>"
    if module_is_marker:
        marker_functions[module_scope_key] = tree

    def _scope(node: ast.AST) -> ast.AST:
        current: Optional[ast.AST] = node
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            current = parents.get(current)
        return current or tree

    def _local_nodes(scope: ast.AST) -> list[ast.AST]:
        return [
            node for node in ast.walk(scope) if node is scope or _scope(node) is scope
        ]

    def _environment(
        scope: ast.AST,
    ) -> tuple[
        dict[str, frozenset[str]],
        dict[str, tuple[str, frozenset[str]]],
        dict[str, ast.AST],
        set[str],
    ]:
        expression_roles: dict[str, frozenset[str]] = {}
        signal_meanings: dict[str, tuple[str, frozenset[str]]] = {}
        assignments: dict[str, ast.AST] = {}
        audit_containers: set[str] = set()
        for node in _local_nodes(scope):
            if isinstance(node, ast.Dict):
                for key_node, value_node in zip(node.keys, node.values):
                    key = _subscript_key(key_node)
                    if key not in _PROVENANCE_FAILURE_KEYS or isinstance(
                        value_node, ast.Constant
                    ):
                        continue
                    identity = _expression_identity(value_node)
                    expression_roles[identity] = frozenset(
                        set(expression_roles.get(identity, frozenset())) | {key}
                    )
                    source = _provenance_signal_source(value_node)
                    if source is not None:
                        name, kind = source
                        existing = signal_meanings.get(name)
                        if existing is None or existing[0] == kind:
                            coverage = set(existing[1] if existing else ()) | {key}
                            signal_meanings[name] = kind, frozenset(coverage)
                parent = parents.get(node)
                if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        parent.targets
                        if isinstance(parent, ast.Assign)
                        else [parent.target]
                    )
                    audit_containers.update(
                        name for target in targets for name in _target_names(target)
                    )
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    for name in _target_names(target):
                        assignments.setdefault(name, node.value)
        return expression_roles, signal_meanings, assignments, audit_containers

    def _next_statement(statement: ast.stmt) -> Optional[ast.stmt]:
        parent = parents.get(statement)
        if parent is None:
            return None
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list) or statement not in value:
                continue
            index = value.index(statement)
            if index + 1 < len(value) and isinstance(value[index + 1], ast.stmt):
                return value[index + 1]
        return None

    def _preceding_direct_assignments(statement: ast.stmt) -> dict[str, ast.AST]:
        """Return straight-line bindings in the exact suite before a guard."""

        parent = parents.get(statement)
        if parent is None:
            return {}
        preceding: list[ast.stmt] = []
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list) or statement not in value:
                continue
            preceding = [
                item
                for item in value[: value.index(statement)]
                if isinstance(item, ast.stmt)
            ]
            break
        assignments: dict[str, ast.AST] = {}
        for candidate in preceding:
            if not isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                continue
            value = candidate.value
            if value is None:
                continue
            targets = (
                candidate.targets
                if isinstance(candidate, ast.Assign)
                else [candidate.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = value
        return assignments

    def _direct_audit_row(
        statement: ast.stmt,
    ) -> Optional[tuple[dict[str, ast.AST], set[str]]]:
        """Return one audit row only when the statement must materialize it."""

        payload: Optional[ast.AST] = None
        containers: set[str] = set()
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            payload = statement.value
            containers.add(statement.targets[0].id)
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr in {"add", "append"}
            and isinstance(statement.value.func.value, ast.Name)
            and len(statement.value.args) == 1
            and not statement.value.keywords
        ):
            payload = statement.value.args[0]
            containers.add(statement.value.func.value.id)
        else:
            return None

        if isinstance(payload, (ast.List, ast.Tuple)):
            if len(payload.elts) != 1 or not isinstance(payload.elts[0], ast.Dict):
                return None
            payload = payload.elts[0]
        if not isinstance(payload, ast.Dict):
            return None
        if any(
            key is None
            or not isinstance(key, ast.Constant)
            or not isinstance(key.value, str)
            for key in payload.keys
        ):
            return None
        keys = [str(key.value) for key in payload.keys if isinstance(key, ast.Constant)]
        if len(keys) != len(set(keys)):
            return None
        fields = dict(zip(keys, payload.values))
        checks = fields.get("checks")
        if checks is not None:
            if not (
                isinstance(checks, (ast.List, ast.Tuple))
                and len(checks.elts) == 1
                and isinstance(checks.elts[0], ast.Dict)
            ):
                return None
            payload = checks.elts[0]
            if any(
                key is None
                or not isinstance(key, ast.Constant)
                or not isinstance(key.value, str)
                for key in payload.keys
            ):
                return None
            keys = [
                str(key.value) for key in payload.keys if isinstance(key, ast.Constant)
            ]
            if len(keys) != len(set(keys)):
                return None
            fields = dict(zip(keys, payload.values))
        role = fields.get("role")
        if not (
            _PROVENANCE_FAILURE_KEYS <= fields.keys()
            and isinstance(role, ast.Constant)
            and str(role.value).strip().lower() == "audit_only"
        ):
            return None
        return fields, containers

    def _immediate_returned_audit_row(
        guard: ast.If,
    ) -> Optional[dict[str, ast.AST]]:
        """Read one literal audit row returned immediately after a guard."""

        following = _next_statement(guard)
        if not (
            isinstance(following, ast.Return)
            and isinstance(following.value, ast.Dict)
            and parents.get(following) is parents.get(guard)
        ):
            return None
        outer = following.value
        outer_fields = {
            _subscript_key(key): value
            for key, value in zip(outer.keys, outer.values)
            if key is not None
        }
        checks = outer_fields.get("checks")
        if not (
            isinstance(checks, (ast.List, ast.Tuple))
            and len(checks.elts) == 1
            and isinstance(checks.elts[0], ast.Dict)
        ):
            return None
        row = checks.elts[0]
        if any(
            key is None
            or not isinstance(key, ast.Constant)
            or not isinstance(key.value, str)
            for key in row.keys
        ):
            return None
        keys = [str(key.value) for key in row.keys if isinstance(key, ast.Constant)]
        if len(keys) != len(set(keys)):
            return None
        fields = dict(zip(keys, row.values))
        role = fields.get("role")
        if not (
            _PROVENANCE_FAILURE_KEYS <= fields.keys()
            and isinstance(role, ast.Constant)
            and str(role.value).strip().lower() == "audit_only"
        ):
            return None
        return fields

    def _post_audit_alias_path_is_pure(
        statements: list[ast.stmt],
        *,
        count_names: set[str],
        audit_containers: set[str],
    ) -> bool:
        """Allow only side-effect-free boolean aliases before the guard."""

        trusted = set(count_names)

        def _pure(node: ast.AST) -> bool:
            if isinstance(node, ast.Name):
                return node.id in trusted
            if isinstance(node, ast.Constant):
                return isinstance(node.value, (bool, int, float, type(None)))
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                return bool(node.values) and all(_pure(value) for value in node.values)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return _pure(node.operand)
            if isinstance(node, ast.Compare):
                return _pure(node.left) and all(
                    _pure(comparator) for comparator in node.comparators
                )
            return False

        for statement in statements:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id not in trusted
                and statement.targets[0].id not in audit_containers
                and _pure(statement.value)
            ):
                return False
            trusted.add(statement.targets[0].id)
        return True

    def _module_pre_guard_has_indirect_effects(
        statements: list[ast.stmt],
    ) -> bool:
        """Reject module helpers that can publish or rewrite audit authority."""

        helpers = {
            statement.name: statement
            for statement in tree.body
            if isinstance(statement, ast.FunctionDef)
        }
        helper_aliases = {name: name for name in helpers}
        ambiguous_aliases: set[str] = set()
        simple_aliases: dict[str, str] = {}
        helper_escape = False
        for statement in tree.body:
            value = (
                statement.value
                if isinstance(statement, (ast.Assign, ast.AnnAssign))
                else None
            )
            canonical_alias = (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Name)
                and statement.value.id in helper_aliases
            )
            if (
                value is not None
                and (_referenced_names(value) & set(helper_aliases))
                and not canonical_alias
            ):
                helper_escape = True
            if not canonical_alias and not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                continue
            target = statement.targets[0].id
            if isinstance(statement.value, ast.Name) and (
                statement.value.id in helper_aliases
            ):
                simple_aliases[target] = statement.value.id
                resolved = helper_aliases[statement.value.id]
                previous = helper_aliases.get(target)
                if previous is not None and previous != resolved:
                    ambiguous_aliases.add(target)
                helper_aliases[target] = resolved
            elif target in helper_aliases and target not in helpers:
                ambiguous_aliases.add(target)
            elif isinstance(statement.value, ast.Name):
                simple_aliases[target] = statement.value.id

        if ambiguous_aliases or helper_escape:
            return True
        for start in simple_aliases:
            seen: set[str] = set()
            current = start
            while current in simple_aliases:
                if current in seen:
                    return True
                seen.add(current)
                current = simple_aliases[current]

        def _definition_runtime_expressions(
            function: ast.FunctionDef,
        ) -> list[ast.AST]:
            arguments = function.args
            expressions: list[ast.AST] = [
                *function.decorator_list,
                *arguments.defaults,
                *(value for value in arguments.kw_defaults if value is not None),
            ]
            if function.returns is not None:
                expressions.append(function.returns)
            expressions.extend(
                argument.annotation
                for argument in [
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                ]
                if argument.annotation is not None
            )
            if arguments.vararg is not None and arguments.vararg.annotation is not None:
                expressions.append(arguments.vararg.annotation)
            if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
                expressions.append(arguments.kwarg.annotation)
            return expressions

        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                return True
            if isinstance(statement, ast.FunctionDef):
                if any(
                    isinstance(candidate, ast.Call)
                    for expression in _definition_runtime_expressions(statement)
                    for candidate in ast.walk(expression)
                ):
                    return True
                continue
            for candidate in ast.walk(statement):
                if not isinstance(candidate, ast.Call):
                    continue
                call_name = _call_name(candidate.func).rsplit(".", 1)[-1]
                if (
                    _is_provenance_result_sink_call(candidate)
                    or (
                        call_name in _DYNAMIC_NAMESPACE_PRIMITIVES
                        and not literal_observational_getattr(
                            candidate,
                            protected_names=set(helper_aliases),
                        )
                    )
                    or call_name in _DYNAMIC_NAMESPACE_MUTATORS
                ):
                    return True
                if (
                    isinstance(candidate.func, ast.Name)
                    and candidate.func.id in helper_aliases
                ):
                    return True
                if any(
                    _referenced_names(argument) & set(helper_aliases)
                    for argument in [
                        *candidate.args,
                        *(keyword.value for keyword in candidate.keywords),
                    ]
                ):
                    return True
        return False

    def _module_direct_guard_is_bound(guard: ast.If) -> bool:
        """Prove a module guard is bound to one exact, immutable audit row."""

        if parents.get(guard) is not tree or guard not in tree.body:
            return False
        guard_index = tree.body.index(guard)
        preceding = tree.body[:guard_index]
        audit_rows: list[tuple[int, dict[str, ast.AST], set[str]]] = []
        for index, statement in enumerate(preceding):
            if not isinstance(statement, ast.Assign):
                continue
            row = _direct_audit_row(statement)
            if row is not None:
                fields, containers = row
                audit_rows.append((index, fields, containers))
        audit_follows_guard = False
        if not audit_rows:
            following = _next_statement(guard)
            following_audit_row = (
                _direct_audit_row(following)
                if isinstance(following, ast.stmt)
                else None
            )
            if following_audit_row is not None:
                fields, containers = following_audit_row
                audit_rows.append((len(preceding), fields, containers))
                audit_follows_guard = True
        if len(audit_rows) != 1:
            return False

        audit_index, fields, audit_containers = audit_rows[0]
        bound_names = {
            value.id
            for key, value in fields.items()
            if key in _PROVENANCE_FAILURE_KEYS and isinstance(value, ast.Name)
        }
        if len(bound_names) != len(_PROVENANCE_FAILURE_KEYS):
            return False

        direct_bindings: dict[str, ast.AST] = {}
        for statement in preceding[:audit_index]:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            if value is None:
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                if isinstance(target, ast.Name) and target.id in bound_names:
                    direct_bindings[target.id] = value
        if direct_bindings.keys() != bound_names or any(
            isinstance(value, ast.Constant) for value in direct_bindings.values()
        ):
            return False
        # When the immutable count bindings and terminal guard precede the
        # audit row, earlier helper calls cannot mutate audit authority that
        # does not exist yet.  Stable-count, direct-raise, dynamic-namespace,
        # and pre-result-sink proofs are enforced by the caller.  The stricter
        # helper-effect scan remains necessary when the row already exists.
        if not audit_follows_guard and _module_pre_guard_has_indirect_effects(
            preceding
        ):
            return False
        return _post_audit_alias_path_is_pure(
            preceding[audit_index + 1 :],
            count_names=bound_names,
            audit_containers=audit_containers,
        )

    def _canonical_entrypoint_guard(node: ast.AST) -> bool:
        if not (
            isinstance(node, ast.If)
            and not node.orelse
            and parents.get(node) is tree
            and tree.body
            and tree.body[-1] is node
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
        ):
            return False
        operands = (node.test.left, node.test.comparators[0])
        return any(
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
            for left, right in (operands, operands[::-1])
        )

    def _direct_function_runtime_binding(function: ast.AST) -> bool:
        if not (
            isinstance(function, ast.FunctionDef)
            and not function.decorator_list
            and sum(candidate.name == function.name for candidate in all_functions) == 1
            and not any(
                isinstance(candidate, (ast.Yield, ast.YieldFrom))
                and _nearest_function(candidate) is function
                for candidate in ast.walk(function)
            )
        ):
            return False
        if _has_dynamic_namespace_indirection(tree):
            return False
        dynamic_namespace_calls = set(_DYNAMIC_NAMESPACE_PRIMITIVES)
        for candidate in ast.walk(tree):
            if (
                isinstance(candidate, ast.Name)
                and candidate.id == function.name
                and isinstance(candidate.ctx, (ast.Store, ast.Del))
            ):
                return False
            if isinstance(candidate, ast.ClassDef) and candidate.name == function.name:
                return False
            if isinstance(candidate, ast.arg) and candidate.arg == function.name:
                return False
            if isinstance(candidate, ast.ExceptHandler) and (
                candidate.name == function.name
            ):
                return False
            if isinstance(candidate, (ast.MatchAs, ast.MatchStar)) and (
                candidate.name == function.name
            ):
                return False
            if isinstance(candidate, ast.MatchMapping) and (
                candidate.rest == function.name
            ):
                return False
            if (
                isinstance(candidate, _TYPE_PARAMETER_NODE_TYPES)
                and candidate.name == function.name
            ):
                return False
            if isinstance(candidate, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split(".")[0]) == function.name
                for alias in candidate.names
            ):
                return False
            if (
                isinstance(candidate, ast.Attribute)
                and isinstance(candidate.ctx, (ast.Store, ast.Del))
                and (
                    candidate.attr == function.name
                    or _mapping_root_name(candidate) == function.name
                )
            ):
                return False
            if (
                isinstance(candidate, ast.Subscript)
                and isinstance(candidate.ctx, (ast.Store, ast.Del))
                and (
                    _subscript_key(candidate.slice) == function.name
                    or _mapping_root_name(candidate) == function.name
                )
            ):
                return False
            if not isinstance(candidate, ast.Call):
                continue
            call_name = _call_name(candidate.func).rsplit(".", 1)[-1]
            if call_name in dynamic_namespace_calls and not (
                literal_observational_getattr(
                    candidate,
                    protected_names={function.name},
                )
            ):
                return False
            if (
                call_name in _DYNAMIC_NAMESPACE_MUTATORS
                and len(candidate.args) >= 2
                and (
                    _subscript_key(candidate.args[1]) is None
                    or _subscript_key(candidate.args[1]) == function.name
                    or _mapping_root_name(candidate.args[0]) == function.name
                )
            ):
                return False
        return True

    def _dynamic_namespace_execution_present() -> bool:
        if _has_dynamic_namespace_indirection(tree):
            return True
        for candidate in ast.walk(tree):
            if not isinstance(candidate, ast.Call):
                continue
            call_name = _call_name(candidate.func).rsplit(".", 1)[-1]
            if call_name in _DYNAMIC_NAMESPACE_PRIMITIVES and not (
                literal_observational_getattr(
                    candidate,
                    protected_names=marker_names,
                )
            ):
                return True
            if call_name in _DYNAMIC_NAMESPACE_MUTATORS:
                return True
        return False

    def _terminal_entry_function(function: ast.AST) -> bool:
        if not _direct_function_runtime_binding(function):
            return False

        calls = [
            candidate
            for candidate in ast.walk(tree)
            if isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == function.name
        ]
        if not calls:
            return False
        for call in calls:
            statement = parents.get(call)
            if not (
                isinstance(statement, ast.Expr)
                and statement.value is call
                and (
                    (parents.get(statement) is tree and tree.body[-1] is statement)
                    or _canonical_entrypoint_guard(parents.get(statement))
                )
            ):
                return False
        return True

    def _direct_execution_statement(statement: ast.stmt) -> bool:
        current: ast.AST = statement
        while True:
            owner = parents.get(current)
            if owner is tree or _canonical_entrypoint_guard(owner):
                return True
            if isinstance(owner, ast.FunctionDef):
                return _terminal_entry_function(owner)
            # A ``try`` body is entered directly when its owning statement is
            # entered.  Handler/finally safety is evaluated separately by
            # ``_failure_exit_may_be_swallowed``; branches, loops, handlers,
            # ``else`` suites, and context managers remain non-direct here.
            if isinstance(owner, ast.Try) and current in owner.body:
                current = owner
                continue
            return False

    def _result_sink_precedes_call(call: ast.Call) -> bool:
        call_scope = _scope(call)
        call_line = int(getattr(call, "lineno", 0) or 0)
        return any(
            isinstance(candidate, ast.Call)
            and int(getattr(candidate, "lineno", 0) or 0) < call_line
            and _is_provenance_result_sink_call(candidate)
            for candidate in _local_nodes(call_scope)
        )

    def _eager_outer_call_statement(call: ast.Call) -> ast.stmt | None:
        """Return the statement for a nested, eagerly evaluated call argument.

        Python evaluates positional and keyword arguments before invoking the
        outer callable, so an exception raised by ``outer(audit(...))`` cannot
        be swallowed by ``outer``.  Keep this proof deliberately narrow: only
        calls, keyword wrappers, and starred call arguments may occur between
        the provenance helper and its statement.  Lazy or conditional
        constructs (lambda/comprehensions/bool ops/conditional expressions)
        therefore remain fail-closed.
        """

        current: ast.AST = call
        crossed_outer_call = False
        while True:
            owner = parents.get(current)
            if isinstance(owner, ast.stmt):
                if crossed_outer_call and isinstance(
                    owner, (ast.Expr, ast.Assign, ast.AnnAssign)
                ):
                    return owner
                return None
            if isinstance(owner, ast.Call):
                crossed_outer_call = True
                current = owner
                continue
            if isinstance(owner, (ast.keyword, ast.Starred)):
                current = owner
                continue
            return None

    def _loop_eager_argument_is_fail_closed(
        call: ast.Call, statement: ast.stmt
    ) -> bool:
        """Prove an eager helper call covers every non-empty loop execution.

        The supported grammar is intentionally small: a direct ``for`` loop
        appends the helper result to an empty collection, only assignments or
        unconditional failure guards may precede that append, and the next
        statement rejects an empty collection.  Thus an empty iterable fails,
        while every non-empty iteration either raises before the helper or
        eagerly evaluates the self-raising helper before appending its result.
        """

        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr in {"append", "add"}
            and isinstance(statement.value.func.value, ast.Name)
        ):
            return False
        collection = statement.value.func.value.id
        loop = parents.get(statement)
        call_scope = _scope(call)
        if not (
            isinstance(loop, ast.For)
            and statement in loop.body
            and not loop.orelse
            and isinstance(
                call_scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and _direct_scope_statement(loop, call_scope)
        ):
            return False
        forbidden_loop_control = (
            ast.AsyncFor,
            ast.Break,
            ast.Continue,
            ast.Match,
            ast.Return,
            *_TRY_NODE_TYPES,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.Yield,
            ast.YieldFrom,
        )
        if any(
            isinstance(candidate, forbidden_loop_control)
            for candidate in ast.walk(loop)
        ) or any(
            isinstance(candidate, ast.For) and candidate is not loop
            for candidate in ast.walk(loop)
        ):
            return False
        if _provenance_branch_contains_result_sink(loop.body):
            return False

        owner = parents.get(loop)
        preceding_suite: list[ast.stmt] = []
        if owner is not None:
            for _, value in ast.iter_fields(owner):
                if isinstance(value, list) and loop in value:
                    preceding_suite = [
                        item
                        for item in value[: value.index(loop)]
                        if isinstance(item, ast.stmt)
                    ]
                    break
        initialization_indexes = [
            index
            for index, candidate in enumerate(preceding_suite)
            if isinstance(candidate, (ast.Assign, ast.AnnAssign))
            and candidate.value is not None
            and any(
                isinstance(target, ast.Name) and target.id == collection
                for target in (
                    candidate.targets
                    if isinstance(candidate, ast.Assign)
                    else [candidate.target]
                )
            )
            and isinstance(candidate.value, ast.List)
            and not candidate.value.elts
        ]
        if not initialization_indexes:
            return False
        initialization_index = initialization_indexes[-1]
        if any(
            any(
                isinstance(candidate, ast.Name) and candidate.id == collection
                for candidate in ast.walk(later)
            )
            for later in preceding_suite[initialization_index + 1 :]
        ):
            return False
        empty_guard = _next_statement(loop)
        if not (
            isinstance(empty_guard, ast.If)
            and isinstance(empty_guard.test, ast.UnaryOp)
            and isinstance(empty_guard.test.op, ast.Not)
            and isinstance(empty_guard.test.operand, ast.Name)
            and empty_guard.test.operand.id == collection
            and _branch_all_paths_raise(empty_guard.body)
            and not empty_guard.orelse
            and _direct_scope_statement(empty_guard, call_scope)
            and not _failure_exit_may_be_swallowed(empty_guard)
            and not _provenance_branch_contains_result_sink(empty_guard.body)
            and not _result_sink_precedes_guard(empty_guard, parents)
        ):
            return False

        for preceding in loop.body[: loop.body.index(statement)]:
            if isinstance(preceding, (ast.Assign, ast.AnnAssign)):
                if any(
                    isinstance(candidate, ast.Name) and candidate.id == collection
                    for candidate in ast.walk(preceding)
                ):
                    return False
                continue
            if (
                isinstance(preceding, ast.If)
                and not preceding.orelse
                and _branch_all_paths_raise(preceding.body)
                and not _failure_exit_may_be_swallowed(preceding)
                and not _provenance_branch_contains_result_sink(preceding.body)
            ):
                continue
            return False
        return True

    def _exact_collection_test(node: ast.AST, name: str) -> bool:
        if isinstance(node, ast.Name):
            return node.id == name
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return False
        left, right = node.left, node.comparators[0]
        if not (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "len"
            and len(left.args) == 1
            and isinstance(left.args[0], ast.Name)
            and left.args[0].id == name
            and isinstance(right, ast.Constant)
            and right.value == 0
        ):
            return False
        return isinstance(node.ops[0], (ast.Gt, ast.NotEq))

    def _branch_all_paths_raise(statements: list[ast.stmt]) -> bool:
        direct_raise = _branch_all_paths_exit(statements) and not any(
            isinstance(node, ast.Return)
            for statement in statements
            for node in ast.walk(statement)
        )
        if direct_raise:
            return True
        if not statements or any(
            isinstance(node, ast.Return)
            for statement in statements
            for node in ast.walk(statement)
        ):
            return False
        terminal = statements[-1]
        terminal_position = statement_positions.get(id(terminal))
        return bool(
            terminal_position is not None
            and _block_flow_outcomes(statements[:-1]) == {_FLOW_FALLTHROUGH}
            and _stable_raise_only_helper_call(
                terminal,
                position=terminal_position,
                tree=tree,
                parents=id_parents,
                positions=statement_positions,
            )
        )

    swallowed_exit_issues: dict[tuple[int, int, int], dict[str, object]] = {}

    def _failure_exit_may_be_swallowed(node: ast.AST) -> bool:
        """Reject a raise site whose caller-side ``try`` can consume failure."""

        exit_line = int(getattr(node, "lineno", 0) or 0)
        current: Optional[ast.AST] = node
        while current is not None:
            parent = parents.get(current)
            if parent is None:
                return False
            if isinstance(parent, ast.Try):
                # A failure raised from the try body is proven to escape when
                # every possible handler immediately raises again.  ``else``
                # and ``finally`` remain outside this narrow proof because a
                # later control-flow exit could suppress or replace failure.
                if (
                    current in parent.body
                    and not parent.orelse
                    and not parent.finalbody
                    and bool(parent.handlers)
                    and all(
                        _handler_immediately_reraises(handler)
                        for handler in parent.handlers
                    )
                ):
                    current = parent
                    continue
                unsafe_handlers = [
                    handler
                    for handler in parent.handlers
                    if not _handler_immediately_reraises(handler)
                ]
                handler_lines = [
                    int(getattr(handler, "lineno", 0) or 0)
                    for handler in unsafe_handlers
                ] or [0]
                for handler_line in handler_lines:
                    issue = {
                        "failure_mode": "provenance_guard_swallowed_by_handler",
                        "exit_line": exit_line,
                        "try_line": int(getattr(parent, "lineno", 0) or 0),
                        "handler_line": handler_line or None,
                    }
                    swallowed_exit_issues[
                        (
                            int(issue["exit_line"] or 0),
                            int(issue["try_line"] or 0),
                            int(issue["handler_line"] or 0),
                        )
                    ] = issue
                return True
            if isinstance(parent, (*_TRY_STAR_NODE_TYPES, ast.With, ast.AsyncWith)):
                return True
            current = parent
        return False

    environments = {
        scope: _environment(scope) for scope in [tree, *marker_functions.values()]
    }

    def _direct_scope_statement(statement: ast.stmt, scope: ast.AST) -> bool:
        """Prove a statement lies on a direct, failure-propagating suite path."""

        current: ast.AST = statement
        while True:
            owner = parents.get(current)
            if owner is scope:
                return True
            if isinstance(owner, ast.Try) and current in owner.body:
                if _failure_exit_may_be_swallowed(current):
                    return False
                current = owner
                continue
            return False

    def _stable_local_failure_signals(guard: ast.If, scope: ast.AST) -> bool:
        """Bind both audit counts to immutable built-in ``int`` values."""

        if not _builtin_int_binding_is_unmodified(tree):
            return False
        owner = parents.get(guard)
        if owner is None:
            return False
        preceding: list[ast.stmt] = []
        for _, value in ast.iter_fields(owner):
            if not isinstance(value, list) or guard not in value:
                continue
            preceding = [
                statement
                for statement in value[: value.index(guard)]
                if isinstance(statement, ast.stmt)
            ]
            break
        if not preceding:
            return False

        def _builtin_int_call(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "int"
                and len(node.args) == 1
                and not node.keywords
            )

        audit_rows: list[tuple[int, dict[str, ast.AST], set[str]]] = []
        for index, statement in enumerate(preceding):
            row = _direct_audit_row(statement)
            if row is not None:
                fields, containers = row
                audit_rows.append((index, fields, containers))
        returned_audit_row = None
        if not audit_rows:
            following = _next_statement(guard)
            following_audit_row = (
                _direct_audit_row(following)
                if isinstance(following, ast.stmt)
                else None
            )
            if following_audit_row is not None:
                fields, containers = following_audit_row
                audit_rows.append((len(preceding), fields, containers))
            else:
                returned_audit_row = _immediate_returned_audit_row(guard)
                if returned_audit_row is not None:
                    audit_rows.append((len(preceding), returned_audit_row, set()))
        if len(audit_rows) != 1:
            return False

        for audit_index, fields, audit_containers in audit_rows:
            bound_names: set[str] = set()
            for key in _PROVENANCE_FAILURE_KEYS:
                value = fields[key]
                if _builtin_int_call(value):
                    continue
                if not isinstance(value, ast.Name):
                    return False
                bound_names.add(value.id)
                binding: Optional[ast.AST] = None
                binding_index: Optional[int] = None
                for index, statement in enumerate(preceding[:audit_index]):
                    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        continue
                    assignment_value = statement.value
                    if assignment_value is None:
                        continue
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    if any(
                        isinstance(target, ast.Name) and target.id == value.id
                        for target in targets
                    ):
                        binding = assignment_value
                        binding_index = index
                if binding is None or not _builtin_int_call(binding):
                    return False
                if returned_audit_row is not None and binding_index is not None:
                    for statement in preceding[binding_index + 1 :]:
                        if any(
                            isinstance(candidate, ast.Name)
                            and candidate.id == value.id
                            and isinstance(candidate.ctx, (ast.Store, ast.Del))
                            for candidate in ast.walk(statement)
                        ):
                            return False
            post_audit = preceding[audit_index + 1 :]
            if scope is tree:
                if not _post_audit_alias_path_is_pure(
                    post_audit,
                    count_names=bound_names,
                    audit_containers=audit_containers,
                ):
                    return False
                continue
            for statement in post_audit:
                for candidate in ast.walk(statement):
                    if (
                        isinstance(candidate, ast.Name)
                        and candidate.id in bound_names
                        and isinstance(candidate.ctx, (ast.Store, ast.Del))
                    ):
                        return False
                    if (
                        isinstance(candidate, (ast.Attribute, ast.Subscript))
                        and isinstance(candidate.ctx, (ast.Store, ast.Del))
                        and _mapping_root_name(candidate) in bound_names
                    ):
                        return False
                    if (
                        isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Attribute)
                        and _mapping_root_name(candidate.func.value)
                        in (bound_names | audit_containers)
                    ):
                        return False
        return True

    def _full_failure_test(
        node: ast.AST,
        scope: ast.AST,
        *,
        require_stable_signals: bool = False,
    ) -> bool:
        roles, signals, assignments, containers = environments.setdefault(
            scope, _environment(scope)
        )
        if scope is tree:
            guard = parents.get(node)
            if isinstance(guard, ast.If):
                assignments = {
                    **assignments,
                    **_preceding_direct_assignments(guard),
                }
        meaning = _provenance_predicate_meaning(
            node,
            expression_roles=roles,
            signal_meanings=signals,
            assignments=assignments,
            audit_containers=containers,
        )
        if meaning != (_PROVENANCE_FAILURE, _PROVENANCE_FULL_COVERAGE):
            return False
        guard = parents.get(node)
        return not require_stable_signals or (
            isinstance(guard, ast.If) and _stable_local_failure_signals(guard, scope)
        )

    def _separate_direct_failure_guards(
        scope: ast.AST,
    ) -> set[ast.If]:
        """Prove complete fail-close coverage split across direct guards.

        A generated entry point may reject the two stable provenance counts in
        separate ``if`` statements before publishing outputs.  Treat that as
        equivalent to one ``invalid_pair_n or discordant_n`` guard only when a
        single literal audit row binds both roles to immutable built-in ``int``
        locals, every role is covered by a direct raising branch, and no result
        sink precedes either branch.  This is deliberately narrower than
        general control-flow analysis and does not infer semantics from names.
        """

        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) or not (
            _builtin_int_binding_is_unmodified(tree)
        ):
            return set()

        audit_rows = [
            (statement, row)
            for statement in scope.body
            if (row := _direct_audit_row(statement)) is not None
        ]
        if len(audit_rows) != 1:
            return set()
        audit_statement, (fields, _) = audit_rows[0]

        signal_coverage: dict[str, set[str]] = {}
        for key in _PROVENANCE_FAILURE_KEYS:
            source = _provenance_signal_source(fields[key])
            if source is None or source[1] != _PROVENANCE_FAILURE:
                return set()
            signal_coverage.setdefault(source[0], set()).add(key)
        signal_names = set(signal_coverage)

        def _builtin_int_call(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "int"
                and len(node.args) == 1
                and not node.keywords
            )

        bindings: dict[str, ast.stmt] = {}
        for signal_name in signal_names:
            candidates: list[ast.stmt] = []
            for statement in scope.body:
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                value = statement.value
                if value is None:
                    continue
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                if any(signal_name in _target_names(target) for target in targets):
                    candidates.append(statement)
            if len(candidates) != 1 or not _builtin_int_call(candidates[0].value):
                return set()
            binding = candidates[0]
            if scope.body.index(binding) >= scope.body.index(audit_statement):
                return set()
            bindings[signal_name] = binding

            stores = [
                candidate
                for candidate in _local_nodes(scope)
                if isinstance(candidate, ast.Name)
                and candidate.id == signal_name
                and isinstance(candidate.ctx, (ast.Store, ast.Del))
            ]
            if len(stores) != 1:
                return set()

        roles, signals, assignments, containers = environments.setdefault(
            scope, _environment(scope)
        )

        def _direct_count_failure_coverage(node: ast.AST) -> set[str]:
            """Read one non-zero comparison against an audit-bound count."""

            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                return set()
            left, right = node.left, node.comparators[0]
            operator = node.ops[0]
            if (
                isinstance(left, ast.Name)
                and left.id in signal_coverage
                and _literal_zero(right)
                and isinstance(operator, (ast.NotEq, ast.IsNot, ast.Gt, ast.GtE))
            ):
                return set(signal_coverage[left.id])
            if (
                isinstance(right, ast.Name)
                and right.id in signal_coverage
                and _literal_zero(left)
                and isinstance(operator, (ast.NotEq, ast.IsNot, ast.Lt, ast.LtE))
            ):
                return set(signal_coverage[right.id])
            return set()

        guards: set[ast.If] = set()
        coverage: set[str] = set()
        for guard in scope.body:
            if not isinstance(guard, ast.If):
                continue
            meaning = _provenance_predicate_meaning(
                guard.test,
                expression_roles=roles,
                signal_meanings=signals,
                assignments=assignments,
                audit_containers=containers,
            )
            role_coverage = (
                set(meaning[1]) & set(_PROVENANCE_FAILURE_KEYS)
                if meaning is not None and meaning[0] == _PROVENANCE_FAILURE
                else _direct_count_failure_coverage(guard.test)
            )
            if not role_coverage or role_coverage == set(_PROVENANCE_FAILURE_KEYS):
                continue
            if not (
                _branch_all_paths_raise(guard.body)
                and not guard.orelse
                and not _provenance_branch_contains_result_sink(guard.body)
                and not _result_sink_precedes_guard(guard, parents)
                and not _failure_exit_may_be_swallowed(guard)
            ):
                continue
            guard_index = scope.body.index(guard)
            if any(
                scope.body.index(binding) >= guard_index
                for binding in bindings.values()
            ):
                continue
            guards.add(guard)
            coverage.update(role_coverage)

        return (
            guards
            if len(guards) >= 2 and coverage == set(_PROVENANCE_FAILURE_KEYS)
            else set()
        )

    returned_slots: dict[str, Optional[int]] = {}
    self_guarded: set[str] = set()
    self_raising: set[str] = set()
    self_raising_guards: dict[str, set[ast.If]] = {}
    for name, function in marker_functions.items():
        local_nodes = _local_nodes(function)

        separate_guards = _separate_direct_failure_guards(function)
        if separate_guards:
            self_guarded.add(name)
            self_raising.add(name)
            self_raising_guards.setdefault(name, set()).update(separate_guards)

        def _direct_append_collection(
            statement: ast.stmt,
            *,
            require_audit_row: bool = False,
        ) -> Optional[str]:
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr in {"append", "add"}
                and isinstance(statement.value.func.value, ast.Name)
            ):
                return None
            if not require_audit_row:
                return statement.value.func.value.id
            if len(statement.value.args) != 1 or statement.value.keywords:
                return None
            payload = statement.value.args[0]
            if not isinstance(payload, ast.Dict):
                return None
            fields = {
                _subscript_key(key): value
                for key, value in zip(payload.keys, payload.values)
                if key is not None
            }
            role = fields.get("role")
            if not (
                _PROVENANCE_FAILURE_KEYS <= fields.keys()
                and isinstance(role, ast.Constant)
                and str(role.value).strip().lower() == "audit_only"
            ):
                return None
            return statement.value.func.value.id

        def _exact_bool_assignment(
            statement: Optional[ast.stmt], name: str, value: bool
        ) -> bool:
            return (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == name
                and isinstance(statement.value, ast.Constant)
                and statement.value.value is value
            )

        def _previous_statement(statement: ast.stmt) -> Optional[ast.stmt]:
            parent = parents.get(statement)
            if parent is None:
                return None
            for _, value in ast.iter_fields(parent):
                if not isinstance(value, list) or statement not in value:
                    continue
                index = value.index(statement)
                if index > 0 and isinstance(value[index - 1], ast.stmt):
                    return value[index - 1]
            return None

        def _negative_name_test(node: ast.AST, name: str) -> bool:
            return (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.Not)
                and isinstance(node.operand, ast.Name)
                and node.operand.id == name
            )

        def _direct_loop_aggregate_proves_coverage(
            loop: ast.For,
            full_guard: ast.If,
            failure_collection: str,
        ) -> bool:
            """Prove the narrow loop grammar used for aggregate audit failures."""

            if (
                not _direct_scope_statement(loop, function)
                or parents.get(full_guard) is not loop
                or full_guard not in loop.body
                or loop.orelse
                or not loop.body
            ):
                return False
            initialization = _previous_statement(loop)
            first_statement = loop.body[0]
            empty_guard = _next_statement(loop)
            if not (
                _exact_bool_assignment(initialization, _PROVENANCE_LOOP_SENTINEL, False)
                and _exact_bool_assignment(
                    first_statement, _PROVENANCE_LOOP_SENTINEL, True
                )
                and isinstance(empty_guard, ast.If)
                and _negative_name_test(empty_guard.test, _PROVENANCE_LOOP_SENTINEL)
                and _branch_all_paths_raise(empty_guard.body)
                and _direct_scope_statement(empty_guard, function)
                and not _provenance_branch_contains_result_sink(empty_guard.body)
                and not _result_sink_precedes_guard(empty_guard, parents)
            ):
                return False

            sentinel_names = [
                candidate
                for candidate in local_nodes
                if isinstance(candidate, ast.Name)
                and candidate.id == _PROVENANCE_LOOP_SENTINEL
            ]
            if len(sentinel_names) != 3:
                return False

            forbidden = (
                ast.AsyncFor,
                ast.Break,
                ast.Match,
                ast.Return,
                *_TRY_NODE_TYPES,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.Yield,
                ast.YieldFrom,
            )
            if any(isinstance(candidate, forbidden) for candidate in ast.walk(loop)):
                return False
            if any(
                isinstance(candidate, ast.For) and candidate is not loop
                for candidate in ast.walk(loop)
            ):
                return False
            if _result_sink_precedes_guard(full_guard, parents):
                return False

            guard_index = loop.body.index(full_guard)
            audit_collections = {
                collection
                for statement in loop.body[:guard_index]
                if (
                    collection := _direct_append_collection(
                        statement, require_audit_row=True
                    )
                )
            }
            if len(audit_collections) != 1:
                return False
            audit_collection = next(iter(audit_collections))

            for continuation in (
                candidate
                for candidate in ast.walk(loop)
                if isinstance(candidate, ast.Continue)
            ):
                branch = parents.get(continuation)
                if not isinstance(branch, ast.If) or parents.get(branch) is not loop:
                    return False
                if loop.body.index(branch) >= guard_index:
                    return False
                suite = branch.body if continuation in branch.body else branch.orelse
                if continuation not in suite:
                    return False
                preceding = suite[: suite.index(continuation)]
                collections = {
                    collection
                    for statement in preceding
                    if (collection := _direct_append_collection(statement))
                }
                audit_rows = {
                    collection
                    for statement in preceding
                    if (
                        collection := _direct_append_collection(
                            statement, require_audit_row=True
                        )
                    )
                }
                if (
                    failure_collection not in collections
                    or audit_collection not in audit_rows
                ):
                    return False
            return True

        def _direct_inline_loop_guard_proves_failure(
            loop: ast.For,
            full_guard: ast.If,
        ) -> bool:
            """Accept one audit row immediately protected inside each iteration.

            This is the inline counterpart of a self-raising provenance helper:
            the row and its two stable failure counts are authored in the loop,
            then the direct guard raises before execution can reach a result
            sink. Any control-flow construct before that guard could bypass it
            and therefore keeps the script fail-closed.
            """

            if (
                not _direct_scope_statement(loop, function)
                or parents.get(full_guard) is not loop
                or full_guard not in loop.body
                or loop.orelse
                or not _branch_all_paths_raise(full_guard.body)
                or _failure_exit_may_be_swallowed(full_guard)
            ):
                return False
            guard_index = loop.body.index(full_guard)
            preceding = loop.body[:guard_index]
            if (
                len(
                    [
                        row
                        for statement in preceding
                        if (row := _direct_audit_row(statement)) is not None
                    ]
                )
                != 1
            ):
                return False
            forbidden = (
                ast.AsyncFor,
                ast.Break,
                ast.Continue,
                ast.For,
                ast.Match,
                ast.Raise,
                ast.Return,
                *_TRY_NODE_TYPES,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.Yield,
                ast.YieldFrom,
            )
            return not any(
                isinstance(candidate, forbidden)
                for statement in preceding
                for candidate in ast.walk(statement)
            )

        collection_events: dict[str, set[ast.Call]] = {}
        for guard in local_nodes:
            if not isinstance(guard, ast.If) or not _full_failure_test(
                guard.test,
                function,
                require_stable_signals=True,
            ):
                continue
            owner = parents.get(guard)
            direct_guard = (
                (function is not tree and owner is function)
                or (function is tree and _module_direct_guard_is_bound(guard))
                or (
                    isinstance(owner, ast.For)
                    and _direct_inline_loop_guard_proves_failure(owner, guard)
                )
            )
            for statement in guard.body:
                if isinstance(statement, (ast.Raise, ast.Return)):
                    break
                collection = _direct_append_collection(statement)
                if collection is None:
                    continue
                if not direct_guard:
                    if not (
                        isinstance(owner, ast.For)
                        and _direct_loop_aggregate_proves_coverage(
                            owner, guard, collection
                        )
                    ):
                        continue
                collection_events.setdefault(collection, set()).add(statement.value)
            if (
                direct_guard
                and (
                    _branch_all_paths_exit(guard.body)
                    or _branch_all_paths_raise(guard.body)
                )
                and not _provenance_branch_contains_result_sink(guard.body)
                and not _result_sink_precedes_guard(guard, parents)
                and not _failure_exit_may_be_swallowed(guard)
            ):
                self_guarded.add(name)
                if _branch_all_paths_raise(guard.body):
                    self_raising.add(name)
                    self_raising_guards.setdefault(name, set()).add(guard)

        def _empty_initialization(node: ast.AST) -> bool:
            return isinstance(node, (ast.List, ast.Set)) and not node.elts

        returns = [node for node in local_nodes if isinstance(node, ast.Return)]
        valid_collections: set[str] = set()
        for collection, allowed_calls in collection_events.items():

            def _mutates_collection(target: ast.AST) -> bool:
                if isinstance(target, ast.Name):
                    return target.id == collection
                if isinstance(target, (ast.Tuple, ast.List)):
                    return any(_mutates_collection(item) for item in target.elts)
                if isinstance(target, (ast.Subscript, ast.Attribute)):
                    return collection in _referenced_names(target)
                return False

            initializations = 0
            invalid_mutation = False
            boundary_lines = [int(call.lineno) for call in allowed_calls] + [
                int(statement.lineno) for statement in returns
            ]
            for candidate in local_nodes:
                targets: list[ast.AST] = []
                value: Optional[ast.AST] = None
                if isinstance(candidate, ast.Assign):
                    targets = list(candidate.targets)
                    value = candidate.value
                elif isinstance(candidate, ast.AnnAssign):
                    targets = [candidate.target]
                    value = candidate.value
                elif isinstance(candidate, (ast.AugAssign, ast.NamedExpr, ast.Delete)):
                    targets = (
                        list(candidate.targets)
                        if isinstance(candidate, ast.Delete)
                        else [candidate.target]
                    )
                if any(_mutates_collection(target) for target in targets):
                    if (
                        value is not None
                        and len(targets) == 1
                        and isinstance(targets[0], ast.Name)
                        and _empty_initialization(value)
                        and _direct_scope_statement(candidate, function)
                        and boundary_lines
                        and int(candidate.lineno) < min(boundary_lines)
                    ):
                        initializations += 1
                    else:
                        invalid_mutation = True
                if value is not None and collection in _referenced_names(value):
                    invalid_mutation = True
                if (
                    isinstance(candidate, ast.Call)
                    and isinstance(candidate.func, ast.Attribute)
                    and isinstance(candidate.func.value, ast.Name)
                    and candidate.func.value.id == collection
                    and candidate not in allowed_calls
                    and candidate.func.attr not in {"append", "add"}
                ):
                    invalid_mutation = True
                if isinstance(candidate, ast.Call) and any(
                    collection in _referenced_names(argument)
                    for argument in [
                        *candidate.args,
                        *(keyword.value for keyword in candidate.keywords),
                    ]
                ):
                    invalid_mutation = True
            if (
                initializations == 1
                and not invalid_mutation
                and all(parents.get(statement) is function for statement in returns)
            ):
                valid_collections.add(collection)

            # An entry-point function may implement the provenance audit
            # inline instead of returning the failure collection to a caller.
            # Accept that form only when the same exact collection grammar is
            # preserved and a direct, post-collection guard raises on every
            # path before any scientific result sink.  Reads inside that
            # terminal branch (for example, joining failures into an error
            # message before writing a failed summary) cannot weaken the
            # already-taken branch, so they are excluded from the mutation
            # proof.  Mutations before or after the guard remain disqualifying.
            terminal_guards = [
                candidate
                for candidate in local_nodes
                if isinstance(candidate, ast.If)
                and _exact_collection_test(candidate.test, collection)
                and _branch_all_paths_raise(candidate.body)
                and _direct_scope_statement(candidate, function)
                and not _provenance_branch_contains_result_sink(candidate.body)
                and not _result_sink_precedes_guard(candidate, parents)
                and not _failure_exit_may_be_swallowed(candidate)
                and all(
                    int(call.lineno) < int(candidate.lineno) for call in allowed_calls
                )
            ]
            for terminal_guard in terminal_guards:

                def _inside_terminal_branch(candidate: ast.AST) -> bool:
                    current: Optional[ast.AST] = candidate
                    while current is not None and current is not function:
                        if current is terminal_guard:
                            return True
                        current = parents.get(current)
                    return False

                terminal_initializations = 0
                terminal_invalid_mutation = False
                event_lines = [int(call.lineno) for call in allowed_calls]
                if not event_lines:
                    continue
                for candidate in local_nodes:
                    if _inside_terminal_branch(candidate):
                        continue
                    targets: list[ast.AST] = []
                    value: Optional[ast.AST] = None
                    if isinstance(candidate, ast.Assign):
                        targets = list(candidate.targets)
                        value = candidate.value
                    elif isinstance(candidate, ast.AnnAssign):
                        targets = [candidate.target]
                        value = candidate.value
                    elif isinstance(
                        candidate, (ast.AugAssign, ast.NamedExpr, ast.Delete)
                    ):
                        targets = (
                            list(candidate.targets)
                            if isinstance(candidate, ast.Delete)
                            else [candidate.target]
                        )
                    if any(_mutates_collection(target) for target in targets):
                        if (
                            value is not None
                            and len(targets) == 1
                            and isinstance(targets[0], ast.Name)
                            and _empty_initialization(value)
                            and _direct_scope_statement(candidate, function)
                            and int(candidate.lineno) < min(event_lines)
                        ):
                            terminal_initializations += 1
                        else:
                            terminal_invalid_mutation = True
                    if value is not None and collection in _referenced_names(value):
                        terminal_invalid_mutation = True
                    if not (
                        isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Attribute)
                        and isinstance(candidate.func.value, ast.Name)
                        and candidate.func.value.id == collection
                    ):
                        if isinstance(candidate, ast.Call) and any(
                            collection in _referenced_names(argument)
                            for argument in [
                                *candidate.args,
                                *(keyword.value for keyword in candidate.keywords),
                            ]
                        ):
                            terminal_invalid_mutation = True
                        continue
                    if candidate in allowed_calls:
                        continue
                    if candidate.func.attr not in {"append", "add"} or int(
                        candidate.lineno
                    ) >= int(terminal_guard.lineno):
                        terminal_invalid_mutation = True
                if terminal_initializations == 1 and not terminal_invalid_mutation:
                    self_raising.add(name)
                    self_raising_guards.setdefault(name, set()).add(terminal_guard)
                    break

        positions: set[int] = set()
        for statement in returns:
            if statement.value is None:
                positions.clear()
                break
            values = (
                statement.value.elts
                if isinstance(statement.value, (ast.Tuple, ast.List))
                else [statement.value]
            )
            matches = {
                index
                for index, value in enumerate(values)
                if isinstance(value, ast.Name) and value.id in valid_collections
            }
            if len(matches) != 1:
                positions.clear()
                break
            positions.update(matches)
        returned_slots[name] = next(iter(positions)) if len(positions) == 1 else None

    called_functions: set[str] = set()
    unsafe_call = False
    provenance_call_issues: list[dict[str, object]] = []

    def _proven_unavailable_audit_return(
        scope: ast.AST,
        statement: ast.Return,
    ) -> bool:
        """Accept an explicit audit-only return when a source column is absent."""

        branch = parents.get(statement)
        if not (
            isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            and isinstance(branch, ast.If)
            and parents.get(branch) is scope
            and not branch.orelse
            and len(branch.body) >= 2
            and branch.body[-1] is statement
            and isinstance(branch.test, ast.UnaryOp)
            and isinstance(branch.test.op, ast.Not)
            and isinstance(branch.test.operand, ast.Name)
        ):
            return False

        check_statement = branch.body[0]
        audit_row = _direct_audit_row(check_statement)
        if not (
            audit_row is not None
            and isinstance(check_statement, ast.Assign)
            and len(check_statement.targets) == 1
            and isinstance(check_statement.targets[0], ast.Name)
            and not _provenance_branch_contains_result_sink(branch.body)
            and not _result_sink_precedes_guard(branch, parents)
        ):
            return False

        check_name = check_statement.targets[0].id
        check_fields, _ = audit_row
        status = check_fields.get("status")
        if not (
            isinstance(status, ast.Constant)
            and str(status.value).strip().lower() == "unavailable"
            and all(
                isinstance(check_fields[key], ast.Constant)
                and check_fields[key].value is None
                for key in _PROVENANCE_FAILURE_KEYS
            )
            and "checks" in _literal_string_tokens(statement)
            and check_name in _referenced_names(statement)
        ):
            return False

        assignments = _preceding_direct_assignments(branch)
        count_exists = assignments.get(branch.test.operand.id)
        if not (
            isinstance(count_exists, ast.Compare)
            and len(count_exists.ops) == 1
            and isinstance(count_exists.ops[0], ast.In)
            and len(count_exists.comparators) == 1
            and isinstance(count_exists.comparators[0], ast.Attribute)
            and count_exists.comparators[0].attr == "columns"
        ):
            return False
        return True

    def _record_call_issue(
        call: ast.Call,
        called: str,
        failure_mode: str,
        *,
        following: ast.stmt | None = None,
    ) -> None:
        issue = {
            "failure_mode": failure_mode,
            "helper_name": called,
            "call_line": int(getattr(call, "lineno", 0) or 0),
            "helper_proves_self_raising": called in self_raising,
            "returned_failure_slot": returned_slots.get(called),
            "following_guard_line": (
                int(getattr(following, "lineno", 0) or 0)
                if following is not None
                else None
            ),
        }
        if issue not in provenance_call_issues:
            provenance_call_issues.append(issue)

    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        called = call.func.id
        if called not in marker_functions:
            continue
        called_functions.add(called)
        marker_function = marker_functions[called]
        failure_may_be_swallowed = _failure_exit_may_be_swallowed(call)
        if not _direct_function_runtime_binding(marker_function):
            unsafe_call = True
            _record_call_issue(
                call,
                called,
                "provenance_helper_runtime_binding_ambiguous",
            )
            continue
        if called in self_raising:
            statement = parents.get(call)
            direct_self_raising_call = bool(
                isinstance(statement, ast.Expr)
                and statement.value is call
                or isinstance(statement, ast.Assign)
                and statement.value is call
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                or isinstance(statement, ast.AnnAssign)
                and statement.value is call
                and isinstance(statement.target, ast.Name)
            )
            if not direct_self_raising_call:
                eager_statement = _eager_outer_call_statement(call)
                if eager_statement is not None:
                    statement = eager_statement
                    direct_self_raising_call = True
            direct_call_execution = bool(
                direct_self_raising_call
                and isinstance(statement, ast.stmt)
                and (
                    _direct_execution_statement(statement)
                    or _loop_eager_argument_is_fail_closed(call, statement)
                )
            )
            returns = [
                candidate
                for candidate in _local_nodes(marker_function)
                if isinstance(candidate, ast.Return)
            ]
            direct_body = (
                marker_function.body
                if isinstance(marker_function, ast.FunctionDef)
                else []
            )
            direct_guard_indexes = [
                direct_body.index(guard)
                for guard in self_raising_guards.get(called, set())
                if guard in direct_body
            ]
            returns_follow_guard = not returns or (
                bool(direct_guard_indexes)
                and all(
                    (
                        parents.get(candidate) is marker_function
                        and direct_body.index(candidate) > min(direct_guard_indexes)
                    )
                    or _proven_unavailable_audit_return(marker_function, candidate)
                    for candidate in returns
                )
            )
            if (
                not (
                    direct_call_execution
                    and returns_follow_guard
                    and not _result_sink_precedes_call(call)
                )
                or failure_may_be_swallowed
            ):
                unsafe_call = True
                _record_call_issue(
                    call,
                    called,
                    "provenance_self_raising_call_not_directly_propagated",
                )
            continue
        node = parents.get(call)
        if not (isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is call):
            unsafe_call = True
            _record_call_issue(
                call,
                called,
                "provenance_helper_result_not_bound",
            )
            continue
        following = _next_statement(node)
        if not (
            isinstance(following, ast.If)
            and parents.get(node) is parents.get(following)
            and _direct_execution_statement(node)
        ):
            unsafe_call = True
            _record_call_issue(
                call,
                called,
                "provenance_helper_result_not_immediately_guarded",
                following=following,
            )
            continue

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        slot = returned_slots.get(called)
        guarded = False
        if (
            slot is not None
            and len(targets) == 1
            and isinstance(targets[0], (ast.Tuple, ast.List))
        ):
            items = targets[0].elts
            if slot < len(items) and isinstance(items[slot], ast.Name):
                guarded = _exact_collection_test(
                    following.test, items[slot].id
                ) and _branch_all_paths_raise(following.body)
        if not guarded and len(targets) == 1 and isinstance(targets[0], ast.Name):
            result_name = targets[0].id
            environment = environments.setdefault(
                _scope(node), _environment(_scope(node))
            )
            environment[3].add(result_name)
            guarded = _branch_all_paths_raise(following.body) and _full_failure_test(
                following.test, _scope(node)
            )
        if guarded:
            guarded = not (
                _failure_exit_may_be_swallowed(following)
                or _provenance_branch_contains_result_sink(following.body)
                or _result_sink_precedes_guard(following, parents)
            )
        if not guarded:
            _record_call_issue(
                call,
                called,
                "provenance_helper_result_guard_not_fail_closed",
                following=following,
            )
        unsafe_call = unsafe_call or not guarded

    unsafe_module = module_is_marker and (
        module_scope_key not in self_raising or _dynamic_namespace_execution_present()
    )
    unsafe_definition = bool(ambiguous_names) or any(
        name not in called_functions
        for name in marker_functions
        if name != module_scope_key
    )
    if not unsafe_call and not unsafe_definition and not unsafe_module:
        return []

    detail: dict[str, object] = {"reason": "provenance_audit_not_fail_closed"}
    issue_details = [
        *swallowed_exit_issues.values(),
        *provenance_call_issues,
    ]
    if unsafe_module:
        issue_details.append(
            {
                "failure_mode": "module_provenance_scope_not_proven_fail_closed",
                "helper_name": "<module>",
            }
        )
    if issue_details:
        detail["issues"] = issue_details

    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "A measurement-provenance audit records invalid or discordant "
                "pairs but does not fail the completed step before scientific "
                "outputs can be published."
            ),
            detail=detail,
        )
    ]


def _provenance_pair_scan_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject an explicit measured-only scan that cannot see count-only concepts."""

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tokens = _literal_string_tokens(node)
        if not (_PROVENANCE_FAILURE_KEYS <= tokens and "audit_only" in tokens):
            continue
        scanned_suffixes: set[str] = set()
        for candidate in ast.walk(node):
            if not isinstance(candidate, ast.Call):
                continue
            if _call_name(candidate.func).split(".")[-1] != "endswith":
                continue
            scanned_suffixes.update(
                token
                for token in _literal_string_tokens(candidate)
                if token in {"_measured", "_n"}
            )
        if "_measured" not in scanned_suffixes or "_n" in scanned_suffixes:
            continue
        return [
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "The measurement-provenance audit scans measured columns only "
                    "and cannot fail closed for count-only concepts."
                ),
                detail={"reason": "provenance_pair_scan_not_bidirectional"},
            )
        ]
    return []


def _resolved_context_payload_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject treating the resolved-input context binding as its JSON payload."""

    findings: list[ValidationFinding] = []

    def _nodes_in_scope(owner: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef):
        nodes: list[ast.AST] = []

        class _Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                nodes.append(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                nodes.append(node)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                nodes.append(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                nodes.append(node)

            def generic_visit(self, node: ast.AST) -> None:
                nodes.append(node)
                super().generic_visit(node)

        visitor = _Visitor()
        for statement in owner.body:
            visitor.visit(statement)
        return nodes

    scope_owners: list[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef] = [tree]
    scope_owners.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    for owner in scope_owners:
        scope_nodes = _nodes_in_scope(owner)
        keys_by_name: dict[str, set[str]] = {}
        for node in scope_nodes:
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and isinstance(_subscript_key(node.slice), str)
            ):
                keys_by_name.setdefault(node.value.id, set()).add(
                    str(_subscript_key(node.slice))
                )
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            keys_by_name.setdefault(node.func.value.id, set()).add(node.args[0].value)
        resolved_manifest_names = {
            name
            for name, keys in keys_by_name.items()
            if {"planner_declared_inputs", "inputs"} <= keys
        }
        for node in scope_nodes:
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript)
                and _subscript_key(node.value.slice) == "variables"
                and isinstance(node.value.value, ast.Subscript)
                and _subscript_key(node.value.value.slice) == "context"
                and isinstance(node.value.value.value, ast.Name)
                and node.value.value.value.id in resolved_manifest_names
            ):
                continue
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "The resolved-input manifest context entry is a digest-bound "
                        "file reference, not an inline ResearchContext payload."
                    ),
                    detail={
                        "reason": "resolved_context_payload_not_loaded",
                        "line": int(node.lineno),
                        "manifest_name": node.value.value.value.id,
                        "target_name": node.targets[0].id,
                    },
                )
            )
    return sorted(findings, key=lambda finding: int(finding.detail["line"]))


def _resolved_input_binding_key_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject reading ``input_key`` from the wrong typed-binding schema level."""

    parents, _ = _ast_parent_and_statement_positions(tree)
    manifest_keys: dict[tuple[int, str], set[str]] = {}
    for node in ast.walk(tree):
        scope_id = _scope_id_for_node(node, parents=parents, tree=tree)
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

    input_mappings: dict[tuple[int, str], int] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        value = node.value
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
        if manifest_name is None:
            continue
        scope_id = _scope_id_for_node(node, parents=parents, tree=tree)
        if (scope_id, manifest_name) in resolved_manifests:
            input_mappings[(scope_id, targets[0].id)] = int(node.lineno)

    binding_origins: dict[tuple[int, str], tuple[str, int]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and isinstance(value.slice, ast.Name)
        ):
            continue
        scope_id = _scope_id_for_node(node, parents=parents, tree=tree)
        mapping_line = input_mappings.get((scope_id, value.value.id))
        if mapping_line is not None and mapping_line < int(node.lineno):
            binding_origins[(scope_id, targets[0].id)] = (
                value.slice.id,
                int(node.lineno),
            )

    findings: list[ValidationFinding] = []
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        nodes = _scope_nodes(function.body)
        keys_by_parameter: dict[str, set[str]] = {}
        input_key_accesses: dict[str, list[int]] = {}
        for node in nodes:
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                key = _subscript_key(node.slice)
                if node.value.id in parameters and isinstance(key, str):
                    keys_by_parameter.setdefault(node.value.id, set()).add(key)
                    if key == "input_key":
                        input_key_accesses.setdefault(node.value.id, []).append(
                            int(node.lineno)
                        )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in parameters
                and node.args
                and isinstance(_subscript_key(node.args[0]), str)
            ):
                parameter_name = node.func.value.id
                key = str(_subscript_key(node.args[0]))
                keys_by_parameter.setdefault(parameter_name, set()).add(key)
                if key == "input_key":
                    input_key_accesses.setdefault(parameter_name, []).append(
                        int(node.lineno)
                    )
        candidate_parameters = {
            name
            for name, keys in keys_by_parameter.items()
            if {"input_key", "relative_path", "sha256", "product_contract"} <= keys
        }
        for parameter_name in sorted(candidate_parameters):
            positional_parameters = [
                argument.arg
                for argument in (*function.args.posonlyargs, *function.args.args)
            ]
            if parameter_name not in positional_parameters:
                continue
            parameter_index = positional_parameters.index(parameter_name)
            helper_loads = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == function.name
            ]
            calls: list[ast.Call] = []
            valid_calls = bool(helper_loads)
            for load in helper_loads:
                parent = parents.get(id(load))
                if not (
                    isinstance(parent, ast.Call)
                    and parent.func is load
                    and len(parent.args) > parameter_index
                    and isinstance(parent.args[parameter_index], ast.Name)
                ):
                    valid_calls = False
                    break
                argument_name = parent.args[parameter_index].id
                scope_id = _scope_id_for_node(parent, parents=parents, tree=tree)
                origin = binding_origins.get((scope_id, argument_name))
                if origin is None or origin[1] >= int(parent.lineno):
                    valid_calls = False
                    break
                calls.append(parent)
            if not valid_calls or not calls:
                continue
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
                        "helper_name": function.name,
                        "binding_parameter": parameter_name,
                        "access_lines": sorted(
                            input_key_accesses.get(parameter_name, [])
                        ),
                    },
                )
            )
    return findings


def _pre312_fstring_subscript_quote_findings(
    code: str,
    tree: ast.Module | None,
) -> list[ValidationFinding]:
    """Find PEP-701-only subscript quotes rejected by the Python 3.11 runner."""

    occurrences = pre312_fstring_subscript_quote_occurrences(code, tree)
    if not occurrences:
        return []
    occurrences = [
        {
            **occurrence,
            "outer_quote": (
                "double" if occurrence["outer_quote"] == '"' else "single"
            ),
        }
        for occurrence in occurrences
    ]
    occurrences.sort(
        key=lambda occurrence: (
            int(occurrence["line"]),
            int(occurrence["column"]),
            int(occurrence["end_line"]),
            int(occurrence["end_column"]),
        )
    )
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "A generated f-string uses its outer quote inside a subscript "
                "expression; the sandbox Python runtime rejects this PEP-701-only "
                "syntax."
            ),
            detail={
                "reason": "fstring_runtime_quote_incompatible",
                "occurrences": occurrences,
            },
        )
    ]


_HOST_VALIDATION_FAILURE_EXCEPTIONS = frozenset(
    {
        "BaseException",
        "DescriptiveInputError",
        "Exception",
        "TypeError",
        "ValueError",
    }
)
_RECONCILIATION_HELPER_NAME = "reconcile_binary_event_presence"
_DESCRIPTIVE_INPUT_HELPER_NAMES = frozenset(
    {
        "closed_categorical_counts",
        "measurement_provenance_receipt",
        "strict_numeric_input",
    }
)


def _caught_exception_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Tuple):
        return {
            name for element in node.elts for name in _caught_exception_names(element)
        }
    name = _call_name(node).split(".")[-1]
    return {name} if name else set()


def _handler_catches_reconciliation_failure(handler: ast.ExceptHandler) -> bool:
    """Return whether *handler* can swallow a host validation failure."""

    if handler.type is None:
        return True
    return bool(
        _caught_exception_names(handler.type) & _HOST_VALIDATION_FAILURE_EXCEPTIONS
    )


def _handler_immediately_reraises(handler: ast.ExceptHandler) -> bool:
    """Prove that a handler immediately propagates the caught exception."""

    return (
        bool(handler.body)
        and isinstance(handler.body[0], ast.Raise)
        and handler.body[0].exc is None
    )


def _reconciliation_helper_call_names(
    *,
    target: ast.AST,
    parents: dict[int, ast.AST],
    helper_name: str = _RECONCILIATION_HELPER_NAME,
    require_authoritative_import: bool = False,
) -> set[str]:
    """Return helper aliases visible from *target*'s lexical scopes.

    ``from ... import reconcile_binary_event_presence as reconcile`` retains
    the helper's semantics even though the call-site identifier changes.  The
    imported symbol, rather than a loose name substring, is the authority for
    adding an alias to the closed call-name set. Simple name assignments are
    also followed, but definitions nested below the target's scopes are not
    borrowed into the current execution path.
    """

    names = set() if require_authoritative_import else {helper_name}
    alias_pairs: list[tuple[str, str]] = []

    class _AliasVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = str(node.module or "")
            for imported in node.names:
                if imported.name != helper_name:
                    continue
                if require_authoritative_import and not module.endswith(
                    "easyicu.research_agent.methods.descriptive_inputs"
                ):
                    continue
                if imported.name == helper_name:
                    names.add(imported.asname or imported.name)

        def visit_Assign(self, node: ast.Assign) -> None:
            source_name = _call_name(node.value).split(".")[-1]
            for assignment_target in node.targets:
                if isinstance(assignment_target, ast.Name) and source_name:
                    alias_pairs.append((assignment_target.id, source_name))

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is None:
                return
            source_name = _call_name(node.value).split(".")[-1]
            if isinstance(node.target, ast.Name) and source_name:
                alias_pairs.append((node.target.id, source_name))

    scopes: list[ast.AST] = []
    current: ast.AST | None = target
    while current is not None:
        current = parents.get(id(current))
        if isinstance(current, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(current)
    for scope in reversed(scopes):
        visitor = _AliasVisitor()
        for statement in getattr(scope, "body", []):
            visitor.visit(statement)

    changed = True
    while changed:
        changed = False
        for alias, source_name in alias_pairs:
            if source_name in names and alias not in names:
                names.add(alias)
                changed = True
    return names


def _statements_call_reconciliation(
    statements: list[ast.stmt],
    *,
    helper_call_names: set[str],
) -> bool:
    """Detect helper calls executed in one lexical scope.

    Function and lambda bodies are deferred code, so merely defining one inside
    a ``try`` must not make that ``try`` look as though it called the helper.
    Simple aliases assigned inside the inspected statements are followed in
    source order as well as by the visible-scope fixed-point alias scan above.
    """

    aliases = set(helper_call_names)
    called = False

    class _CallVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Assign(self, node: ast.Assign) -> None:
            source_name = _call_name(node.value).split(".")[-1]
            if source_name in aliases:
                aliases.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            self.visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is None:
                return
            source_name = _call_name(node.value).split(".")[-1]
            if source_name in aliases and isinstance(node.target, ast.Name):
                aliases.add(node.target.id)
            self.visit(node.value)

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal called
            if _call_name(node.func).split(".")[-1] in aliases:
                called = True
            self.generic_visit(node)

    visitor = _CallVisitor()
    for statement in statements:
        visitor.visit(statement)
    return called


def _finally_exception_suppressor(finalbody: list[ast.stmt]) -> ast.AST | None:
    """Return control flow that can suppress an active ``try`` exception.

    A ``return`` anywhere in the current lexical scope suppresses an exception
    raised by the corresponding ``try``.  A ``break`` or ``continue`` does the
    same only when it targets a loop outside that ``try``; control transfers
    inside a loop created by the ``finally`` suite do not escape the suite and
    therefore do not suppress the exception.
    """

    suppressor: ast.AST | None = None

    class _FinallyVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.loop_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Return(self, node: ast.Return) -> None:
            nonlocal suppressor
            suppressor = suppressor or node

        def visit_Break(self, node: ast.Break) -> None:
            nonlocal suppressor
            if self.loop_depth == 0:
                suppressor = suppressor or node

        def visit_Continue(self, node: ast.Continue) -> None:
            nonlocal suppressor
            if self.loop_depth == 0:
                suppressor = suppressor or node

        def _visit_loop(self, node: ast.AST) -> None:
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        visit_For = _visit_loop
        visit_AsyncFor = _visit_loop
        visit_While = _visit_loop

    visitor = _FinallyVisitor()
    for statement in finalbody:
        visitor.visit(statement)
    return suppressor


def _swallowed_reconciliation_error_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject caught host-owned validation failures that continue.

    ``reconcile_binary_event_presence`` is itself a deterministic validation
    boundary.  Once an Agent elects to call it, converting its exception into
    an ``unavailable`` audit and continuing can publish outputs after a known
    contradictory triad.  The descriptive-input helpers provide the same
    fail-closed boundary for coercion, closed accounting, and measurement
    provenance.  This is mechanical enforcement; it does not select variables,
    category levels, grouping, or any scientific method.
    """

    findings: list[ValidationFinding] = []
    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    helper_names = {
        _RECONCILIATION_HELPER_NAME,
        *_DESCRIPTIVE_INPUT_HELPER_NAMES,
    }

    def _aliases_for(target: ast.AST) -> dict[str, set[str]]:
        return {
            helper_name: _reconciliation_helper_call_names(
                target=target,
                parents=parents,
                helper_name=helper_name,
                require_authoritative_import=(
                    helper_name in _DESCRIPTIVE_INPUT_HELPER_NAMES
                ),
            )
            for helper_name in helper_names
        }

    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        suppresses_validation_failure = False
        for item in node.items:
            context = item.context_expr
            if not isinstance(context, ast.Call):
                continue
            if _call_name(context.func).split(".")[-1] != "suppress":
                continue
            caught = {
                name
                for argument in context.args
                for name in _caught_exception_names(argument)
            }
            if caught & _HOST_VALIDATION_FAILURE_EXCEPTIONS:
                suppresses_validation_failure = True
                break
        if not suppresses_validation_failure:
            continue
        aliases_by_helper = _aliases_for(node)
        suppressed_helpers = {
            helper_name
            for helper_name, call_names in aliases_by_helper.items()
            if call_names
            and _statements_call_reconciliation(
                node.body,
                helper_call_names=call_names,
            )
        }
        if not suppressed_helpers:
            continue
        provenance_only = suppressed_helpers == {_RECONCILIATION_HELPER_NAME}
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "A context manager suppresses errors from a host-owned "
                    "validation helper, so invalid declared inputs could "
                    "continue as usable."
                ),
                detail={
                    "reason": (
                        "provenance_helper_error_swallowed"
                        if provenance_only
                        else "host_validation_helper_error_swallowed"
                    ),
                    "helper_names": sorted(suppressed_helpers),
                    "line": int(node.lineno),
                },
            )
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        aliases_by_helper = _aliases_for(node)
        calls_in_body = {
            helper_name
            for helper_name, call_names in aliases_by_helper.items()
            if _statements_call_reconciliation(
                node.body,
                helper_call_names=call_names,
            )
        }
        before_finally = [
            *node.body,
            *node.orelse,
            *(statement for handler in node.handlers for statement in handler.body),
        ]
        calls_before_finally = {
            helper_name
            for helper_name, call_names in aliases_by_helper.items()
            if _statements_call_reconciliation(
                before_finally,
                helper_call_names=call_names,
            )
        }
        finally_suppressor = _finally_exception_suppressor(node.finalbody)
        if calls_before_finally and finally_suppressor is not None:
            provenance_only = calls_before_finally == {_RECONCILIATION_HELPER_NAME}
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "Errors from a host-owned validation helper can be "
                        "suppressed by control flow in the finally suite, so "
                        "invalid declared inputs could continue."
                    ),
                    detail={
                        "reason": (
                            "provenance_helper_error_swallowed"
                            if provenance_only
                            else "host_validation_helper_error_swallowed"
                        ),
                        "helper_names": sorted(calls_before_finally),
                        "line": getattr(finally_suppressor, "lineno", node.lineno),
                    },
                )
            )
            continue
        if not calls_in_body:
            continue
        for handler in node.handlers:
            if not _handler_catches_reconciliation_failure(handler):
                continue
            if _handler_immediately_reraises(handler):
                continue
            provenance_only = calls_in_body == {_RECONCILIATION_HELPER_NAME}
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "Errors from a host-owned validation helper enter control "
                        "flow that can continue without propagating the failure, "
                        "so invalid declared inputs could be recorded as usable."
                    ),
                    detail={
                        "reason": (
                            "provenance_helper_error_swallowed"
                            if provenance_only
                            else "host_validation_helper_error_swallowed"
                        ),
                        "helper_names": sorted(calls_in_body),
                        "line": handler.lineno,
                    },
                )
            )
    return findings


def _scope_nodes(statements: list[ast.stmt]) -> list[ast.AST]:
    """Walk one lexical scope without borrowing uses from nested functions."""

    collected: list[ast.AST] = []

    class _ScopeVisitor(ast.NodeVisitor):
        def generic_visit(self, node: ast.AST) -> None:
            collected.append(node)
            super().generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            collected.append(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            collected.append(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            collected.append(node)

    visitor = _ScopeVisitor()
    for statement in statements:
        visitor.visit(statement)
    return collected


_EXPOSURE_RESULT_CALLS = frozenset(
    {
        "agg",
        "aggregate",
        "average",
        "bar",
        "boxplot",
        "count",
        "describe",
        "dump",
        "fit",
        "fit_predict",
        "fit_regularized",
        "groupby",
        "hist",
        "kdeplot",
        "lineplot",
        "mean",
        "median",
        "plot",
        "predict",
        "quantile",
        "save",
        "savefig",
        "scatter",
        "sum",
        "to_csv",
        "to_json",
        "to_parquet",
        "value_counts",
        "violinplot",
        "write",
        "write_bytes",
        "write_text",
    }
)


def _assignment_target_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: set[str] = set()
    pending = list(targets)
    while pending:
        target = pending.pop()
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.List, ast.Tuple)):
            pending.extend(target.elts)
    return names


def _contains_bound_exposure_selection(
    node: ast.AST,
    *,
    definition_names: set[str],
    column_binding_names: set[str],
) -> bool:
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Subscript):
            continue
        if _referenced_names(candidate.value) & (
            definition_names | column_binding_names
        ):
            return True
        if _referenced_names(candidate.slice) & column_binding_names:
            return True
    return False


def _authoritative_exposure_binding_findings(
    tree: ast.Module, step: AnalysisStep
) -> list[ValidationFinding]:
    authoritative_product = "artifact:primary_exposure_definition"
    if authoritative_product not in {
        str(value or "").strip().lower() for value in step.inputs or []
    }:
        return []

    scopes = [tree.body]
    scopes.extend(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    for statements in scopes:
        nodes = _scope_nodes(statements)
        definition_names: set[str] = set()
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            if authoritative_product not in _literal_string_tokens(node.value):
                continue
            is_binding_lookup = (
                isinstance(node.value, ast.Subscript)
                or isinstance(node.value, ast.Call)
                and _call_name(node.value.func).split(".")[-1] in {"get", "pop"}
            )
            if not is_binding_lookup:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            definition_names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        if not definition_names:
            continue

        assignments = [
            node
            for node in nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        ]
        column_binding_names: set[str] = set()
        for assignment in assignments:
            if not isinstance(assignment.value, ast.Call):
                continue
            call_inputs = [
                *assignment.value.args,
                *[keyword.value for keyword in assignment.value.keywords],
            ]
            if any(
                _referenced_names(value) & definition_names for value in call_inputs
            ):
                column_binding_names.update(_assignment_target_names(assignment))

        selected_value_names: set[str] = set()
        changed = True
        while changed:
            changed = False
            for assignment in assignments:
                target_names = _assignment_target_names(assignment)
                if not target_names or target_names & definition_names:
                    continue
                value = assignment.value
                if not (
                    _contains_bound_exposure_selection(
                        value,
                        definition_names=definition_names,
                        column_binding_names=column_binding_names,
                    )
                    or _referenced_names(value) & selected_value_names
                ):
                    continue
                new_names = target_names - selected_value_names
                if new_names:
                    selected_value_names.update(new_names)
                    changed = True

        for node in nodes:
            if isinstance(node, ast.Return) and node.value is not None:
                if (
                    _contains_bound_exposure_selection(
                        node.value,
                        definition_names=definition_names,
                        column_binding_names=column_binding_names,
                    )
                    or _referenced_names(node.value) & selected_value_names
                ):
                    return []
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func).split(".")[-1].lower()
            if call_name not in _EXPOSURE_RESULT_CALLS:
                continue
            if (
                _contains_bound_exposure_selection(
                    node,
                    definition_names=definition_names,
                    column_binding_names=column_binding_names,
                )
                or _referenced_names(node) & selected_value_names
            ):
                return []

        return [
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "The authoritative primary-exposure definition is loaded but "
                    "never consumed to bind the executable exposure column."
                ),
                detail={"reason": "authoritative_primary_exposure_unused"},
            )
        ]
    return []


def _authoritative_exposure_fallback_findings(
    tree: ast.Module, step: AnalysisStep
) -> list[ValidationFinding]:
    authoritative_product = "artifact:primary_exposure_definition"
    if authoritative_product not in {
        str(value or "").strip().lower() for value in step.inputs or []
    }:
        return []

    binding_keys = {"exposure_column", "source_concept", "role"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body_text = "\n".join(ast.unparse(statement) for statement in node.body)
        if "exposure_definition" not in body_text:
            continue
        for handler in node.handlers:
            if binding_keys <= _literal_string_tokens(handler):
                return [
                    ValidationFinding(
                        validator="mechanical_code_preflight",
                        severity="error",
                        message=(
                            "A failed authoritative exposure binding is replaced "
                            "with constructed fallback metadata instead of failing "
                            "closed."
                        ),
                        detail={"reason": "authoritative_primary_exposure_fallback"},
                    )
                ]
    return []


def _finalized_exposure_reconciliation_findings(
    tree: ast.Module, step: AnalysisStep
) -> list[ValidationFinding]:
    """Reject re-derivation of a row-aligned finalized exposure table.

    A tabular ``primary_exposure_definition`` can be the producer-finalized,
    row-aligned exposure itself.  Once a DataFrame branch reads values directly
    from that typed artifact, raw-event reconciliation must not replace or
    reinterpret those values.  This is an implementation boundary only; the
    check does not choose the exposure or its scientific semantics.
    """

    authoritative_product = "artifact:primary_exposure_definition"
    if authoritative_product not in {
        str(value or "").strip().lower() for value in step.inputs or []
    }:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        dataframe_names: set[str] = set()
        for call in ast.walk(node.test):
            if not isinstance(call, ast.Call) or _call_name(call.func) != "isinstance":
                continue
            if len(call.args) < 2 or not _call_name(call.args[1]).endswith("DataFrame"):
                continue
            dataframe_names.update(_referenced_names(call.args[0]))
        if not dataframe_names:
            continue

        body_nodes = [
            candidate for statement in node.body for candidate in ast.walk(statement)
        ]
        reads_finalized_values = any(
            isinstance(candidate, ast.Subscript)
            and _referenced_names(candidate.value) & dataframe_names
            for candidate in body_nodes
        )
        repeats_raw_event_reconciliation = any(
            isinstance(candidate, ast.Call)
            and _call_name(candidate.func).split(".")[-1]
            == "reconcile_binary_event_presence"
            for candidate in body_nodes
        )
        if reads_finalized_values and repeats_raw_event_reconciliation:
            return [
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "A row-aligned finalized primary-exposure table is read "
                        "directly and then reinterpreted through raw binary-event "
                        "reconciliation."
                    ),
                    detail={
                        "reason": "finalized_exposure_reconciliation_fallback",
                        "line": int(node.lineno),
                    },
                )
            ]
    return []


def _typed_dataframe_erasure_findings(
    tree: ast.Module, step: AnalysisStep
) -> list[ValidationFinding]:
    """Reject type cleanup that makes a later typed-DataFrame branch unreachable."""

    if "artifact:primary_exposure_definition" not in {
        str(value or "").strip().lower() for value in step.inputs or []
    }:
        return []

    dataframe_branches: set[str] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or _call_name(call.func) != "isinstance":
            continue
        if len(call.args) < 2 or not _call_name(call.args[1]).endswith("DataFrame"):
            continue
        if isinstance(call.args[0], ast.Name):
            dataframe_branches.add(call.args[0].id)

    if not dataframe_branches:
        return []

    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.If)
            or not isinstance(node.test, ast.UnaryOp)
            or not isinstance(node.test.op, ast.Not)
            or not isinstance(node.test.operand, ast.Call)
            or _call_name(node.test.operand.func) != "isinstance"
            or len(node.test.operand.args) < 2
            or not isinstance(node.test.operand.args[0], ast.Name)
        ):
            continue
        variable = node.test.operand.args[0].id
        if variable not in dataframe_branches:
            continue
        accepted_types = {
            _call_name(candidate)
            for candidate in ast.walk(node.test.operand.args[1])
            if isinstance(candidate, (ast.Name, ast.Attribute))
        }
        if any(name.endswith("DataFrame") for name in accepted_types):
            continue
        erases_value = False
        for statement in node.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if not any(
                isinstance(target, ast.Name) and target.id == variable
                for target in targets
            ):
                continue
            value = statement.value
            erases_value = (isinstance(value, ast.Dict) and not value.keys) or (
                isinstance(value, (ast.List, ast.Tuple)) and not value.elts
            )
            if erases_value:
                break
        if erases_value:
            return [
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "A supported typed DataFrame artifact is replaced by an "
                        "empty fallback before its DataFrame resolver can consume it."
                    ),
                    detail={
                        "reason": "typed_dataframe_artifact_erased",
                        "line": int(node.lineno),
                        "variable": variable,
                    },
                )
            ]
    return []


def _undefined_direct_call_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject direct calls whose Python name has no lexical binding or import."""

    known_names = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known_names.add(node.name)
        elif isinstance(node, ast.Import):
            known_names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            known_names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            known_names.add(node.id)
        elif isinstance(node, ast.arg):
            known_names.add(node.arg)

    unresolved: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in known_names:
            unresolved.setdefault(node.func.id, int(node.lineno))
    if not unresolved:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "The script directly calls helper names that are neither defined "
                "nor imported in the generated program."
            ),
            detail={
                "reason": "undefined_helper_call",
                "calls": [
                    {"name": name, "line": line}
                    for name, line in sorted(unresolved.items())
                ],
            },
        )
    ]


def _local_call_signature_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject direct local-helper calls that Python can prove are invalid."""

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    invalid_calls: list[dict[str, object]] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        function = functions.get(call.func.id)
        if function is None:
            continue

        arguments = function.args
        positional = [*arguments.posonlyargs, *arguments.args]
        positional_names = [argument.arg for argument in positional]
        positional_only_names = {argument.arg for argument in arguments.posonlyargs}
        keyword_only_names = {argument.arg for argument in arguments.kwonlyargs}
        accepted_keywords = (
            set(positional_names) - positional_only_names
        ) | keyword_only_names
        explicit_keywords = [
            keyword.arg for keyword in call.keywords if keyword.arg is not None
        ]
        has_star_args = any(isinstance(argument, ast.Starred) for argument in call.args)
        has_star_keywords = any(keyword.arg is None for keyword in call.keywords)

        reasons: list[str] = []
        explicit_positional_count = sum(
            not isinstance(argument, ast.Starred) for argument in call.args
        )
        if (
            not has_star_args
            and arguments.vararg is None
            and explicit_positional_count > len(positional)
        ):
            reasons.append("too_many_positional_arguments")

        if not has_star_keywords and arguments.kwarg is None:
            unexpected = sorted(
                name for name in explicit_keywords if name not in accepted_keywords
            )
            if unexpected:
                reasons.append("unexpected_keyword_arguments=" + ",".join(unexpected))

        duplicate = sorted(
            name
            for index, name in enumerate(positional_names)
            if index < explicit_positional_count and name in explicit_keywords
        )
        if duplicate:
            reasons.append("multiple_values_for=" + ",".join(duplicate))

        if not has_star_args and not has_star_keywords:
            required_positional_count = len(positional) - len(arguments.defaults)
            supplied_names = set(explicit_keywords)
            missing_positional = [
                name
                for index, name in enumerate(
                    positional_names[:required_positional_count]
                )
                if index >= explicit_positional_count and name not in supplied_names
            ]
            missing_keyword_only = [
                argument.arg
                for argument, default in zip(
                    arguments.kwonlyargs,
                    arguments.kw_defaults,
                )
                if default is None and argument.arg not in supplied_names
            ]
            missing = [*missing_positional, *missing_keyword_only]
            if missing:
                reasons.append("missing_required_arguments=" + ",".join(missing))

        if reasons:
            invalid_calls.append(
                {
                    "name": call.func.id,
                    "line": int(call.lineno),
                    "reasons": reasons,
                }
            )

    if not invalid_calls:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "A direct call to a locally defined helper cannot satisfy that "
                "helper's Python signature and would fail before analysis."
            ),
            detail={
                "reason": "invalid_local_helper_call",
                "calls": sorted(
                    invalid_calls,
                    key=lambda item: (int(item["line"]), str(item["name"])),
                ),
            },
        )
    ]


def _branch_local_unbound_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Find locals assigned in only one branch and read after the merge.

    This deliberately handles only the mechanically provable straight-line
    case.  Ambiguous nested control flow is left to Python/runtime validation.
    """

    def _scope_nodes_without_nested_functions(node: ast.AST) -> list[ast.AST]:
        collected: list[ast.AST] = []

        class _Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
                return None

            def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
                return None

            def visit_Lambda(self, child: ast.Lambda) -> None:
                return None

            def visit_ListComp(self, child: ast.ListComp) -> None:
                return None

            def visit_SetComp(self, child: ast.SetComp) -> None:
                return None

            def visit_DictComp(self, child: ast.DictComp) -> None:
                return None

            def visit_GeneratorExp(self, child: ast.GeneratorExp) -> None:
                return None

            def generic_visit(self, child: ast.AST) -> None:
                collected.append(child)
                super().generic_visit(child)

        _Visitor().visit(node)
        return collected

    def _names(statements: list[ast.stmt], context: type[ast.expr_context]) -> set[str]:
        names: set[str] = set()
        wrapper = ast.Module(body=statements, type_ignores=[])
        for node in _scope_nodes_without_nested_functions(wrapper):
            if isinstance(node, ast.Name) and isinstance(node.ctx, context):
                names.add(node.id)
        return names

    def _branch_terminates(statements: list[ast.stmt]) -> bool:
        return bool(statements) and isinstance(statements[-1], (ast.Raise, ast.Return))

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def _scope_label(node: ast.AST) -> str:
        names: list[str] = []
        current: ast.AST | None = node
        while current is not None:
            if isinstance(
                current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                names.append(current.name)
            current = parents.get(id(current))
        return ".".join(reversed(names)) or "<module>"

    def _occurrence_id(
        *,
        scope: str,
        name: str,
        phase: str,
        lexical_path: tuple[str, ...],
    ) -> str:
        path = "/".join(lexical_path) or "root"
        return f"branch_local_unbound:{scope}:{path}:{phase}:{name}"

    findings: list[ValidationFinding] = []

    def _target_names(target: ast.AST | None) -> set[str]:
        if target is None:
            return set()
        return {
            node.id
            for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

    def _direct_target_names(statement: ast.stmt) -> set[str]:
        targets: list[ast.AST] = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets = [statement.target]
        elif isinstance(statement, ast.AugAssign):
            targets = [statement.target]
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            return {
                alias.asname or alias.name.split(".")[0] for alias in statement.names
            }
        return {name for target in targets for name in _target_names(target)}

    def _is_hashable_literal(value: ast.AST) -> bool:
        if isinstance(value, ast.Constant):
            try:
                hash(value.value)
            except TypeError:
                return False
            return True
        if isinstance(value, ast.Tuple):
            return all(_is_hashable_literal(item) for item in value.elts)
        return False

    def _is_nonthrowing_literal(value: ast.AST) -> bool:
        if isinstance(value, ast.Constant):
            return True
        if isinstance(value, (ast.Tuple, ast.List)):
            return all(_is_nonthrowing_literal(item) for item in value.elts)
        if isinstance(value, ast.Set):
            return all(_is_hashable_literal(item) for item in value.elts)
        if isinstance(value, ast.Dict):
            return all(
                key is not None and _is_hashable_literal(key) for key in value.keys
            ) and all(_is_nonthrowing_literal(item) for item in value.values)
        return False

    def _safe_prefix_assignments(statements: list[ast.stmt]) -> set[str]:
        """Return direct local names assigned before any operation can fail."""

        assigned: set[str] = set()
        for statement in statements:
            if isinstance(statement, ast.Pass) or (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if isinstance(statement, ast.Assign) and _is_nonthrowing_literal(
                statement.value
            ):
                if all(isinstance(target, ast.Name) for target in statement.targets):
                    assigned.update(target.id for target in statement.targets)
                    continue
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
                and _is_nonthrowing_literal(statement.value)
            ):
                assigned.add(statement.target.id)
                continue
            break
        return assigned

    def _load_lines(statement: ast.AST, name: str) -> list[int]:
        lines = [
            int(node.lineno)
            for node in _scope_nodes_without_nested_functions(statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == name
        ]
        lines.extend(
            int(node.lineno)
            for node in _scope_nodes_without_nested_functions(statement)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        )
        return sorted(set(lines))

    def _first_unassigned_load(
        statements: list[ast.stmt],
        *,
        name: str,
        initially_assigned: bool,
    ) -> tuple[Optional[int], bool, bool]:
        """Return first definite read-before-store plus post-block state."""

        assigned = initially_assigned
        for statement in statements:
            if isinstance(statement, ast.If):
                test_lines = _load_lines(statement.test, name)
                if test_lines and not assigned:
                    return min(test_lines), assigned, True
                body_line, body_assigned, body_continues = _first_unassigned_load(
                    statement.body,
                    name=name,
                    initially_assigned=assigned,
                )
                else_line, else_assigned, else_continues = _first_unassigned_load(
                    statement.orelse,
                    name=name,
                    initially_assigned=assigned,
                )
                branch_lines = [
                    line for line in (body_line, else_line) if line is not None
                ]
                if branch_lines:
                    return min(branch_lines), assigned, True
                if body_continues and else_continues:
                    assigned = body_assigned and else_assigned
                elif body_continues:
                    assigned = body_assigned
                elif else_continues:
                    assigned = else_assigned
                else:
                    return None, assigned, False
                continue

            if isinstance(statement, (ast.For, ast.AsyncFor)):
                iterator_lines = _load_lines(statement.iter, name)
                if iterator_lines and not assigned:
                    return min(iterator_lines), assigned, True
                target_assigned = name in _target_names(statement.target)
                body_line, _, _ = _first_unassigned_load(
                    statement.body,
                    name=name,
                    initially_assigned=assigned or target_assigned,
                )
                else_line, _, _ = _first_unassigned_load(
                    statement.orelse,
                    name=name,
                    initially_assigned=assigned,
                )
                branch_lines = [
                    line for line in (body_line, else_line) if line is not None
                ]
                if branch_lines:
                    return min(branch_lines), assigned, True
                # A loop may be empty, so its target is not a post-loop
                # definite assignment.
                continue

            if isinstance(statement, (ast.With, ast.AsyncWith)):
                context_lines = [
                    line
                    for item in statement.items
                    for line in _load_lines(item.context_expr, name)
                ]
                if context_lines and not assigned:
                    return min(context_lines), assigned, True
                target_assigned = any(
                    name in _target_names(item.optional_vars)
                    for item in statement.items
                )
                body_line, body_assigned, body_continues = _first_unassigned_load(
                    statement.body,
                    name=name,
                    initially_assigned=assigned or target_assigned,
                )
                if body_line is not None:
                    return body_line, assigned, True
                if body_continues:
                    assigned = body_assigned or target_assigned
                else:
                    return None, assigned, False
                continue

            lines = _load_lines(statement, name)
            if lines and not assigned:
                return min(lines), assigned, True
            if name in _direct_target_names(statement):
                assigned = True
            if isinstance(statement, (ast.Raise, ast.Return)):
                return None, assigned, False
        return None, assigned, True

    def _analyze_block(
        statements: list[ast.stmt],
        initially_assigned: set[str],
        scope: str,
        block_path: tuple[str, ...],
    ) -> None:
        assigned_before = set(initially_assigned)
        for index, statement in enumerate(statements):
            statement_kind = type(statement).__name__.lower()
            statement_ordinal = sum(
                isinstance(prior, type(statement)) for prior in statements[:index]
            )
            statement_path = (
                *block_path,
                f"{statement_kind}[{statement_ordinal}]",
            )
            if isinstance(statement, ast.If):
                body_stores = _names(statement.body, ast.Store)
                else_stores = _names(statement.orelse, ast.Store)
                body_only = body_stores - else_stores - assigned_before
                else_only = else_stores - body_stores - assigned_before
                candidates = set()
                if not _branch_terminates(statement.orelse):
                    candidates.update(body_only)
                if not _branch_terminates(statement.body):
                    candidates.update(else_only)
                following = statements[index + 1 :]
                for name in sorted(candidates):
                    load_lines = [
                        int(node.lineno)
                        for later in following
                        for node in _scope_nodes_without_nested_functions(later)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id == name
                    ]
                    later_store_lines = [
                        int(node.lineno)
                        for later in following
                        for node in _scope_nodes_without_nested_functions(later)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                        and node.id == name
                    ]
                    if not load_lines:
                        continue
                    first_load = min(load_lines)
                    if later_store_lines and min(later_store_lines) < first_load:
                        continue
                    assignment_lines = [
                        int(node.lineno)
                        for branch in (statement.body, statement.orelse)
                        for node in _scope_nodes_without_nested_functions(
                            ast.Module(body=branch, type_ignores=[])
                        )
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                        and node.id == name
                    ]
                    findings.append(
                        ValidationFinding(
                            validator="mechanical_code_preflight",
                            severity="error",
                            message=(
                                "A local variable is assigned in only one branch "
                                "and then read after control-flow merges, so a "
                                "valid input form can raise UnboundLocalError."
                            ),
                            detail={
                                "reason": "branch_local_unbound",
                                "occurrence_id": _occurrence_id(
                                    scope=scope,
                                    name=name,
                                    phase="if_merge",
                                    lexical_path=statement_path,
                                ),
                                "scope": scope,
                                "name": name,
                                "branch_line": int(statement.lineno),
                                "assignment_lines": sorted(assignment_lines),
                                "first_use_line": first_load,
                            },
                        )
                    )

                guaranteed = body_stores & else_stores
                if _branch_terminates(statement.orelse):
                    guaranteed.update(body_stores)
                if _branch_terminates(statement.body):
                    guaranteed.update(else_stores)
                _analyze_block(
                    statement.body,
                    set(assigned_before),
                    scope,
                    (*statement_path, "body"),
                )
                _analyze_block(
                    statement.orelse,
                    set(assigned_before),
                    scope,
                    (*statement_path, "else"),
                )
                assigned_before.update(guaranteed)
                continue

            if isinstance(statement, _TRY_NODE_TYPES):
                following = statements[index + 1 :]

                def _top_level_stores(block: list[ast.stmt]) -> set[str]:
                    return {
                        name
                        for block_statement in block
                        for name in _direct_target_names(block_statement)
                    }

                def _top_level_unbound_annotations(
                    block: list[ast.stmt],
                ) -> set[str]:
                    return {
                        name
                        for block_statement in block
                        if isinstance(block_statement, ast.AnnAssign)
                        and block_statement.value is None
                        for name in _target_names(block_statement.target)
                    }

                normal_guaranteed = _top_level_stores(
                    [*statement.body, *statement.orelse]
                )
                final_guaranteed = _top_level_stores(statement.finalbody)
                safe_exception_prefix = _safe_prefix_assignments(statement.body)
                continuing_handlers = [
                    handler
                    for handler in statement.handlers
                    if not _branch_all_paths_exit(handler.body)
                ]
                handler_guaranteed = [
                    _top_level_stores(handler.body) for handler in continuing_handlers
                ]
                # Limit this proof to straight-line stores at the try/handler
                # level. Nested conditionals and loops need their own path
                # proof; recursively treating any nested store as guaranteed
                # would create false positives for valid all-branch code.
                all_try_stores = set().union(
                    _top_level_stores(statement.body),
                    _top_level_stores(statement.orelse),
                    _top_level_stores(statement.finalbody),
                    _top_level_unbound_annotations(statement.body),
                    _top_level_unbound_annotations(statement.orelse),
                    _top_level_unbound_annotations(statement.finalbody),
                    *(
                        _top_level_stores(handler.body)
                        for handler in statement.handlers
                    ),
                    *(
                        _top_level_unbound_annotations(handler.body)
                        for handler in statement.handlers
                    ),
                )
                internal_candidates = all_try_stores - assigned_before
                assignment_lines_by_name = {
                    name: sorted(
                        int(node.lineno)
                        for node in _scope_nodes_without_nested_functions(statement)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                        and node.id == name
                    )
                    for name in internal_candidates
                }
                for handler_index, handler in enumerate(statement.handlers):
                    for name in sorted(internal_candidates):
                        handler_initial = (
                            name in assigned_before
                            or name in safe_exception_prefix
                            or handler.name == name
                        )
                        first_load, _, _ = _first_unassigned_load(
                            handler.body,
                            name=name,
                            initially_assigned=handler_initial,
                        )
                        if first_load is None:
                            continue
                        findings.append(
                            ValidationFinding(
                                validator="mechanical_code_preflight",
                                severity="error",
                                message=(
                                    "A try-local variable can be read inside an "
                                    "exception handler before the failing path "
                                    "has assigned it."
                                ),
                                detail={
                                    "reason": "branch_local_unbound",
                                    "occurrence_id": _occurrence_id(
                                        scope=scope,
                                        name=name,
                                        phase=f"handler_{handler_index}",
                                        lexical_path=statement_path,
                                    ),
                                    "scope": scope,
                                    "name": name,
                                    "branch_line": int(handler.lineno),
                                    "assignment_lines": assignment_lines_by_name[name],
                                    "first_use_line": first_load,
                                },
                            )
                        )

                for name in sorted(internal_candidates):
                    first_load, _, _ = _first_unassigned_load(
                        statement.finalbody,
                        name=name,
                        initially_assigned=(
                            name in assigned_before or name in safe_exception_prefix
                        ),
                    )
                    if first_load is None:
                        continue
                    findings.append(
                        ValidationFinding(
                            validator="mechanical_code_preflight",
                            severity="error",
                            message=(
                                "A try-local variable can be read in a finally "
                                "suite before an exceptional path has assigned it."
                            ),
                            detail={
                                "reason": "branch_local_unbound",
                                "occurrence_id": _occurrence_id(
                                    scope=scope,
                                    name=name,
                                    phase="finally",
                                    lexical_path=statement_path,
                                ),
                                "scope": scope,
                                "name": name,
                                "branch_line": int(statement.lineno),
                                "assignment_lines": assignment_lines_by_name[name],
                                "first_use_line": first_load,
                            },
                        )
                    )

                candidates = (
                    all_try_stores
                    - assigned_before
                    - final_guaranteed
                    - safe_exception_prefix
                )
                for name in sorted(candidates):
                    if name in normal_guaranteed and all(
                        name in names for names in handler_guaranteed
                    ):
                        continue
                    load_lines = [
                        int(node.lineno)
                        for later in following
                        for node in _scope_nodes_without_nested_functions(later)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id == name
                    ]
                    later_store_lines = [
                        int(node.lineno)
                        for later in following
                        for node in _scope_nodes_without_nested_functions(later)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                        and node.id == name
                    ]
                    if not load_lines:
                        continue
                    first_load = min(load_lines)
                    if later_store_lines and min(later_store_lines) < first_load:
                        continue
                    assignment_lines = [
                        int(node.lineno)
                        for node in _scope_nodes_without_nested_functions(statement)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                        and node.id == name
                    ]
                    findings.append(
                        ValidationFinding(
                            validator="mechanical_code_preflight",
                            severity="error",
                            message=(
                                "A local variable is not assigned on every "
                                "continuing try/except path before it is read, "
                                "so an exception can raise UnboundLocalError."
                            ),
                            detail={
                                "reason": "branch_local_unbound",
                                "occurrence_id": _occurrence_id(
                                    scope=scope,
                                    name=name,
                                    phase="after_try",
                                    lexical_path=statement_path,
                                ),
                                "scope": scope,
                                "name": name,
                                "branch_line": int(statement.lineno),
                                "assignment_lines": sorted(assignment_lines),
                                "first_use_line": first_load,
                            },
                        )
                    )

                for handler_index, handler in enumerate(statement.handlers):
                    alias = str(handler.name or "").strip()
                    if not alias:
                        continue

                    alias_internal_loads: list[tuple[str, int]] = []
                    if handler.type is not None:
                        type_loads = [
                            int(node.lineno)
                            for node in ast.walk(handler.type)
                            if isinstance(node, ast.Name)
                            and isinstance(node.ctx, ast.Load)
                            and node.id == alias
                        ]
                        if type_loads:
                            alias_internal_loads.append(
                                ("handler_type", min(type_loads))
                            )
                    for phase, block in (
                        ("else", statement.orelse),
                        ("finally", statement.finalbody),
                    ):
                        first_load, _, _ = _first_unassigned_load(
                            block,
                            name=alias,
                            initially_assigned=False,
                        )
                        if first_load is not None:
                            alias_internal_loads.append((phase, first_load))
                    for phase, first_load in alias_internal_loads:
                        findings.append(
                            ValidationFinding(
                                validator="mechanical_code_preflight",
                                severity="error",
                                message=(
                                    "A Python exception alias is unavailable outside "
                                    "the body of the handler that binds it."
                                ),
                                detail={
                                    "reason": "branch_local_unbound",
                                    "occurrence_id": _occurrence_id(
                                        scope=scope,
                                        name=alias,
                                        phase=f"exception_alias_{phase}_{handler_index}",
                                        lexical_path=statement_path,
                                    ),
                                    "scope": scope,
                                    "name": alias,
                                    "branch_line": int(handler.lineno),
                                    "assignment_lines": [int(handler.lineno)],
                                    "first_use_line": first_load,
                                },
                            )
                        )

                    if alias in final_guaranteed:
                        continue

                    def _inside_rebinding_handler(node: ast.AST) -> bool:
                        child: ast.AST = node
                        current = parents.get(id(node))
                        while current is not None and not isinstance(
                            current,
                            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module),
                        ):
                            if isinstance(current, ast.ExceptHandler):
                                if current.name == alias:
                                    return child in current.body
                            child = current
                            current = parents.get(id(current))
                        return False

                    load_lines = [
                        int(node.lineno)
                        for later in following
                        for node in _scope_nodes_without_nested_functions(later)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id == alias
                        and not _inside_rebinding_handler(node)
                    ]
                    later_store_lines = [
                        int(node.lineno)
                        for later in following
                        for node in _scope_nodes_without_nested_functions(later)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                        and node.id == alias
                    ]
                    if not load_lines:
                        continue
                    first_load = min(load_lines)
                    if later_store_lines and min(later_store_lines) < first_load:
                        continue
                    if alias in assigned_before and _branch_all_paths_exit(
                        handler.body
                    ):
                        continue
                    findings.append(
                        ValidationFinding(
                            validator="mechanical_code_preflight",
                            severity="error",
                            message=(
                                "A Python exception alias is cleared when its "
                                "handler exits and cannot be read afterward "
                                "without a new assignment."
                            ),
                            detail={
                                "reason": "branch_local_unbound",
                                "occurrence_id": _occurrence_id(
                                    scope=scope,
                                    name=alias,
                                    phase=f"exception_alias_{handler_index}",
                                    lexical_path=statement_path,
                                ),
                                "scope": scope,
                                "name": alias,
                                "branch_line": int(handler.lineno),
                                "assignment_lines": [int(handler.lineno)],
                                "first_use_line": first_load,
                            },
                        )
                    )

                nested_try_blocks = [
                    ("body", statement.body),
                    ("else", statement.orelse),
                    ("finally", statement.finalbody),
                    *(
                        (f"handler[{handler_index}]", handler.body)
                        for handler_index, handler in enumerate(statement.handlers)
                    ),
                ]
                for role, block in nested_try_blocks:
                    _analyze_block(
                        block,
                        set(assigned_before),
                        scope,
                        (*statement_path, role),
                    )
                if final_guaranteed:
                    assigned_before.update(final_guaranteed)
                elif normal_guaranteed and all(handler_guaranteed):
                    assigned_before.update(
                        normal_guaranteed.intersection(*handler_guaranteed)
                    )
                continue

            nested_blocks: list[tuple[str, list[ast.stmt]]] = []
            if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                nested_blocks.extend(
                    [("body", statement.body), ("else", statement.orelse)]
                )
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                nested_blocks.append(("body", statement.body))
            for role, block in nested_blocks:
                _analyze_block(
                    block,
                    set(assigned_before),
                    scope,
                    (*statement_path, role),
                )
            assigned_before.update(_direct_target_names(statement))

    _analyze_block(tree.body, set(), "<module>", ())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = {
            argument.arg
            for argument in [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
        }
        if node.args.vararg is not None:
            arguments.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            arguments.add(node.args.kwarg.arg)
        _analyze_block(node.body, arguments, _scope_label(node), ())
    return findings


def _local_read_before_assignment_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Find a provable local read that lexically precedes its first assignment."""

    scopes: list[tuple[str, list[ast.stmt], set[str]]] = [
        ("<module>", tree.body, set())
    ]
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = {
            argument.arg
            for argument in [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            ]
        }
        if function.args.vararg is not None:
            arguments.add(function.args.vararg.arg)
        if function.args.kwarg is not None:
            arguments.add(function.args.kwarg.arg)
        scopes.append((function.name, function.body, arguments))

    findings: list[ValidationFinding] = []
    for scope_name, scope_body, arguments in scopes:
        wrapper = ast.Module(body=scope_body, type_ignores=[])
        occurrences: list[ast.Name] = []
        excluded_names: set[str] = set()

        class _ScopeVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return None

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return None

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return None

            def visit_ListComp(self, node: ast.ListComp) -> None:
                return None

            def visit_SetComp(self, node: ast.SetComp) -> None:
                return None

            def visit_DictComp(self, node: ast.DictComp) -> None:
                return None

            def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
                return None

            def visit_Global(self, node: ast.Global) -> None:
                excluded_names.update(node.names)

            def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
                excluded_names.update(node.names)

            def visit_Name(self, node: ast.Name) -> None:
                occurrences.append(node)

        _ScopeVisitor().visit(wrapper)

        local_names = (
            {node.id for node in occurrences if isinstance(node.ctx, ast.Store)}
            - arguments
            - excluded_names
        )
        for name in sorted(local_names):
            store_lines = [
                int(node.lineno)
                for node in occurrences
                if node.id == name and isinstance(node.ctx, ast.Store)
            ]
            load_lines = [
                int(node.lineno)
                for node in occurrences
                if node.id == name and isinstance(node.ctx, ast.Load)
            ]
            if not store_lines or not load_lines or min(load_lines) >= min(store_lines):
                continue
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "A scope-local variable is read before its first "
                        "assignment and can raise NameError or UnboundLocalError "
                        "before the intended fail-closed branch completes."
                    ),
                    detail={
                        "reason": "local_read_before_assignment",
                        "function": scope_name,
                        "name": name,
                        "first_use_line": min(load_lines),
                        "first_assignment_line": min(store_lines),
                    },
                )
            )
    return findings


def _ordinal_rounding_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject lossy round-to-integer coercion inside explicit ordinal branches."""

    unsafe_lines: list[int] = []
    for branch in ast.walk(tree):
        if not isinstance(branch, ast.If):
            continue
        test_tokens = {
            candidate.id.lower()
            for candidate in ast.walk(branch.test)
            if isinstance(candidate, ast.Name)
        }
        if not any("ordinal" in token for token in test_tokens):
            continue
        for node in ast.walk(ast.Module(body=branch.body, type_ignores=[])):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "astype"
                and isinstance(node.func.value, ast.Call)
                and _call_name(node.func.value.func).split(".")[-1] == "round"
            ):
                continue
            dtype = (
                node.args[0]
                if node.args
                else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "dtype"
                    ),
                    None,
                )
            )
            dtype_name = _call_name(dtype).lower() if dtype is not None else ""
            if isinstance(dtype, ast.Constant) and isinstance(dtype.value, str):
                dtype_name = dtype.value.lower()
            if dtype_name == "int" or dtype_name.startswith(("int", "uint")):
                unsafe_lines.append(int(node.lineno))

    if not unsafe_lines:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "An ordinal-analysis branch rounds observed numeric values before "
                "integer conversion; validate exact registered levels instead of "
                "silently changing fractional or out-of-domain values."
            ),
            detail={
                "reason": "lossy_ordinal_rounding",
                "lines": sorted(set(unsafe_lines)),
            },
        )
    ]


def _builtin_int_binding_is_unmodified(tree: ast.Module) -> bool:
    """Prove conservatively that ``int(...)`` still resolves to the built-in."""

    if _has_dynamic_namespace_indirection(tree):
        return False

    builtin_roots = {"builtins", "__builtins__"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        builtin_roots.update(
            alias.asname or "builtins"
            for alias in node.names
            if alias.name == "builtins"
        )

    # Follow simple aliases of the builtins object or its mapping.  The repair
    # is optional, so ambiguous aliases are a reason to decline it rather than
    # guess whether a later bare ``int`` still has standard semantics.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            if _mapping_root_name(node.value) not in builtin_roots:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            aliases = {name for target in targets for name in _target_names(target)}
            if not aliases <= builtin_roots:
                builtin_roots.update(aliases)
                changed = True

    mutating_methods = {
        "__delitem__",
        "__setitem__",
        "clear",
        "pop",
        "popitem",
        "setdefault",
        "update",
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in {"int", "__builtins__"}
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            return False
        if isinstance(node, ast.arg) and node.arg == "int":
            return False
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == "int"
        ):
            return False
        if isinstance(node, ast.MatchAs) and node.name == "int":
            return False
        if isinstance(node, ast.MatchStar) and node.name == "int":
            return False
        if isinstance(node, ast.MatchMapping) and node.rest == "int":
            return False
        if isinstance(node, _TYPE_PARAMETER_NODE_TYPES) and (
            node.name == "int"
        ):
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name == "*" for alias in node.names):
                return False
            if isinstance(node, ast.ImportFrom) and node.module == "builtins":
                return False
            if any(
                (alias.asname or alias.name.split(".")[0]) == "int"
                for alias in node.names
            ):
                return False
        if isinstance(node, ast.ExceptHandler) and node.name == "int":
            return False
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "int"
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            return False
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and _subscript_key(node.slice) == "int"
        ):
            return False
        if (
            isinstance(node, ast.AugAssign)
            and _mapping_root_name(node.target) in builtin_roots
        ):
            return False
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if call_name.rsplit(".", 1)[-1] in _DYNAMIC_NAMESPACE_PRIMITIVES and not (
            literal_observational_getattr(
                node,
                protected_names={"int"},
            )
        ):
            return False
        if (
            call_name in _DYNAMIC_NAMESPACE_MUTATORS
            and len(node.args) >= 2
            and (
                _subscript_key(node.args[1]) is None
                or _subscript_key(node.args[1]) == "int"
            )
        ):
            return False
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in mutating_methods
            and _mapping_root_name(node.func.value) in builtin_roots
        ):
            return False
    return True


def _scalar_cast_before_reduction_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject mechanically invalid integer casts around array-like counts."""

    if not _builtin_int_binding_is_unmodified(tree):
        return []

    unsafe_lines: list[int] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sum"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "int"
            and len(node.func.value.args) == 1
            and not node.func.value.keywords
        ):
            continue
        unsafe_lines.append(int(node.lineno))

    unsafe_lines.extend(
        int(node.lineno) for node in _unreduced_boolean_mask_count_casts(tree)
    )
    aliases = unambiguous_array_predicate_aliases(tree)
    unsafe_lines.extend(
        int(node.lineno)
        for node in ast.walk(tree)
        if misnested_boolean_mask_reduction_expression(
            node,
            aliases=aliases.boolean,
            array_aliases=aliases.values,
        )
        is not None
    )

    if not unsafe_lines:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "A built-in integer cast is applied to an unreduced array-like "
                "count; reduce the boolean mask before converting the resulting "
                "scalar."
            ),
            detail={
                "reason": "scalar_cast_before_reduction",
                "lines": sorted(set(unsafe_lines)),
            },
        )
    ]


def _is_numeric_zero(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.Constant)
        and isinstance(value.value, (int, float))
        and not isinstance(value.value, bool)
        and float(value.value) == 0.0
    )


def _name_has_numeric_zero_guard(
    name: str,
    *,
    after_line: int,
    nodes: Sequence[ast.AST],
) -> bool:
    for node in nodes:
        if (
            not isinstance(node, ast.Compare)
            or int(getattr(node, "lineno", 0)) <= after_line
            or len(node.ops) != 1
            or len(node.comparators) != 1
        ):
            continue
        if isinstance(node.left, ast.Name) and node.left.id == name:
            if _is_numeric_zero(node.comparators[0]):
                return True
        if isinstance(node.comparators[0], ast.Name) and node.comparators[0].id == name:
            if _is_numeric_zero(node.left):
                return True
    return False


def _unreduced_boolean_mask_count_casts(tree: ast.Module) -> list[ast.Call]:
    """Find ``count = int(mask)`` only when later control flow proves count use.

    The deliberately narrow proof requires a simple, uniquely assigned name,
    an array-like boolean expression joined by ``&``/``|``, and a later
    comparison of that name with numeric zero in the same lexical scope.  This
    avoids rewriting scalar bitwise expressions or guessing from variable
    names.
    """

    alias_proof = unambiguous_array_predicate_aliases(tree)
    scopes: list[list[ast.stmt]] = [tree.body]
    scopes.extend(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    candidates: list[ast.Call] = []
    for statements in scopes:
        nodes = _scope_nodes(statements)
        assignments_by_name: dict[str, list[ast.AST]] = {}
        candidate_by_name: dict[str, tuple[ast.Assign | ast.AnnAssign, ast.Call]] = {}
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments_by_name.setdefault(target.id, []).append(node)
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "int"
                and len(value.args) == 1
                and not value.keywords
                and isinstance(value.args[0], ast.BinOp)
                and isinstance(value.args[0].op, (ast.BitAnd, ast.BitOr))
                and is_proven_array_boolean_predicate(
                    value.args[0], aliases=alias_proof
                )
            ):
                continue
            candidate_by_name[targets[0].id] = (node, value)

        for name, (assignment, cast) in candidate_by_name.items():
            if len(assignments_by_name.get(name, ())) != 1:
                continue
            if not _name_has_numeric_zero_guard(
                name,
                after_line=int(assignment.lineno),
                nodes=nodes,
            ):
                continue
            candidates.append(cast)
    return candidates


def _first_time_companion_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject ``value_first`` + ``_first_time`` double-suffix composition."""

    assignments: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(node.value)

    def origin_literals(name: str, seen: set[str]) -> set[str]:
        if name in seen:
            return set()
        seen = {*seen, name}
        values: set[str] = set()
        for expression in assignments.get(name, []):
            values.update(
                str(candidate.value)
                for candidate in ast.walk(expression)
                if isinstance(candidate, ast.Constant)
                and isinstance(candidate.value, str)
            )
            for candidate in ast.walk(expression):
                if isinstance(candidate, ast.Name) and candidate.id != name:
                    values.update(origin_literals(candidate.id, seen))
        return values

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name, function in functions.items():
        parameters = [argument.arg for argument in function.args.args]
        risky_parameters: list[tuple[int, str, int]] = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.JoinedStr
            ):
                continue
            values = node.value.values
            if len(values) != 2:
                continue
            formatted, suffix = values
            if not (
                isinstance(formatted, ast.FormattedValue)
                and isinstance(formatted.value, ast.Name)
                and isinstance(suffix, ast.Constant)
                and suffix.value == "_first_time"
            ):
                continue
            item_name = formatted.value.id
            iterator_name = next(
                (
                    loop.iter.id
                    for loop in ast.walk(function)
                    if isinstance(loop, ast.For)
                    and isinstance(loop.target, ast.Name)
                    and loop.target.id == item_name
                    and isinstance(loop.iter, ast.Name)
                    and loop.iter.id in parameters
                ),
                "",
            )
            if iterator_name:
                risky_parameters.append(
                    (parameters.index(iterator_name), item_name, int(node.lineno))
                )

        for parameter_index, item_name, line in risky_parameters:
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                if _call_name(call.func).split(".")[-1] != function_name:
                    continue
                argument: ast.AST | None = None
                if parameter_index < len(call.args):
                    argument = call.args[parameter_index]
                elif parameter_index < len(parameters):
                    parameter_name = parameters[parameter_index]
                    argument = next(
                        (
                            keyword.value
                            for keyword in call.keywords
                            if keyword.arg == parameter_name
                        ),
                        None,
                    )
                if argument is None:
                    continue
                literals = {
                    str(candidate.value)
                    for candidate in ast.walk(argument)
                    if isinstance(candidate, ast.Constant)
                    and isinstance(candidate.value, str)
                }
                for candidate in ast.walk(argument):
                    if isinstance(candidate, ast.Name):
                        literals.update(origin_literals(candidate.id, set()))
                first_aggregates = sorted(
                    value
                    for value in literals
                    if value.endswith("_first") and not value.endswith("_first_time")
                )
                if not first_aggregates:
                    continue
                return [
                    ValidationFinding(
                        validator="mechanical_code_preflight",
                        severity="error",
                        message=(
                            "A first-value column name is concatenated with "
                            "'_first_time' without removing its existing "
                            "'_first' suffix, so valid companion timestamps "
                            "would be looked up as '*_first_first_time'."
                        ),
                        detail={
                            "reason": "double_first_time_companion_suffix",
                            "function": function_name,
                            "line": line,
                            "loop_variable": item_name,
                            "first_value_examples": first_aggregates[:8],
                        },
                    )
                ]
    return []


def _flatten_bitand_terms(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        return _flatten_bitand_terms(node.left) + _flatten_bitand_terms(node.right)
    return [node]


def _series_method_root(node: ast.AST, methods: set[str]) -> Optional[str]:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in methods
    ):
        return _expression_identity(node.func.value)
    return None


@dataclass
class _StatementPosition:
    scope: ast.AST
    scope_id: int
    owner: ast.AST
    field_name: str
    index: int
    block: list[ast.stmt] = field(repr=False)


@dataclass(frozen=True)
class _NumericCoercionSite:
    scope_id: int
    statement_id: int
    root: str
    line: int
    scope: ast.AST = field(compare=False, hash=False, repr=False)
    statement: ast.stmt = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class _GuardBinding:
    kind: str
    name: str
    base: Optional[str]
    scope_id: int
    runtime_stable: bool = True


@dataclass(frozen=True)
class _CoercionLossBinding:
    coercion: _NumericCoercionSite
    guard_binding: _GuardBinding
    count_line: int
    statement_id: int
    statement: ast.stmt = field(compare=False, hash=False, repr=False)


_LEXICAL_SCOPE_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def _ast_parent_and_statement_positions(
    tree: ast.Module,
) -> tuple[dict[int, ast.AST], dict[int, _StatementPosition]]:
    """Index lexical scopes and ordered statement-list membership."""

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def _scope_for(node: ast.AST) -> ast.AST:
        current = parents.get(id(node))
        while current is not None:
            if isinstance(current, _LEXICAL_SCOPE_NODES):
                return current
            current = parents.get(id(current))
        return tree

    positions: dict[int, _StatementPosition] = {}
    for owner in ast.walk(tree):
        for field_name, value in ast.iter_fields(owner):
            if not (
                isinstance(value, list)
                and value
                and all(isinstance(item, ast.stmt) for item in value)
            ):
                continue
            block = value
            for index, statement in enumerate(block):
                scope = _scope_for(statement)
                positions[id(statement)] = _StatementPosition(
                    scope=scope,
                    scope_id=id(scope),
                    owner=owner,
                    field_name=field_name,
                    index=index,
                    block=block,
                )
    return parents, positions


def _scope_id_for_node(
    node: ast.AST,
    *,
    parents: dict[int, ast.AST],
    tree: ast.Module,
) -> int:
    current: Optional[ast.AST] = node
    while current is not None:
        current = parents.get(id(current))
        if isinstance(current, _LEXICAL_SCOPE_NODES):
            return id(current)
    return id(tree)


def _numeric_coercion_sites(
    tree: ast.Module,
    positions: dict[int, _StatementPosition],
) -> list[_NumericCoercionSite]:
    """Return definition-site identities for ``to_numeric(..., coerce)``."""

    sites: list[_NumericCoercionSite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = [node.target]
        else:
            continue
        if not (
            isinstance(value, ast.Call)
            and _call_name(value.func).split(".")[-1] == "to_numeric"
            and any(
                keyword.arg == "errors"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "coerce"
                for keyword in value.keywords
            )
        ):
            continue
        position = positions.get(id(node))
        if position is None:
            continue
        for target in targets:
            sites.extend(
                _NumericCoercionSite(
                    scope_id=position.scope_id,
                    statement_id=id(node),
                    root=_expression_identity(ast.Name(id=name, ctx=ast.Load())),
                    line=int(node.lineno),
                    scope=position.scope,
                    statement=node,
                )
                for name in _target_names(target)
            )
    return sites


def _coercion_loss_site(
    node: ast.AST,
    coercion_sites: list[_NumericCoercionSite],
    *,
    scope_id: int,
) -> Optional[_NumericCoercionSite]:
    """Resolve a loss-count expression to one preceding coercion definition."""

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sum"
    ):
        return None
    terms = _flatten_bitand_terms(node.func.value)
    if not any(_series_method_root(term, {"notna", "notnull"}) for term in terms):
        return None
    line = int(node.lineno)
    for term in terms:
        root = _series_method_root(term, {"isna", "isnull"})
        candidates = [
            site
            for site in coercion_sites
            if site.scope_id == scope_id and site.root == root and site.line <= line
        ]
        if candidates:
            return max(candidates, key=lambda site: (site.line, site.statement_id))
    return None


def _coercion_loss_bindings(
    tree: ast.Module,
    coercion_sites: list[_NumericCoercionSite],
    positions: dict[int, _StatementPosition],
    *,
    builtin_int_unmodified: bool,
) -> list[_CoercionLossBinding]:
    """Return scope-bound scalar/dict-key loss-count assignments."""

    bindings: list[_CoercionLossBinding] = []

    def _contains_loss_count(
        node: ast.AST,
        *,
        scope_id: int,
    ) -> Optional[tuple[_NumericCoercionSite, int, bool]]:
        matches: list[tuple[_NumericCoercionSite, ast.Call]] = []
        for candidate in ast.walk(node):
            site = _coercion_loss_site(
                candidate,
                coercion_sites,
                scope_id=scope_id,
            )
            if site is not None:
                assert isinstance(candidate, ast.Call)
                matches.append((site, candidate))
        if not matches:
            return None
        site, loss_call = matches[0]
        exact = len(matches) == 1 and (
            node is loss_call
            or bool(
                builtin_int_unmodified
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "int"
                and len(node.args) == 1
                and not node.keywords
                and node.args[0] is loss_call
            )
        )
        return site, int(loss_call.lineno), exact

    def _straight_line_from_coercion(
        coercion: _NumericCoercionSite,
        loss_position: _StatementPosition,
    ) -> bool:
        coercion_position = positions.get(coercion.statement_id)
        return bool(
            coercion_position is not None
            and coercion_position.scope_id == loss_position.scope_id
            and coercion_position.owner is loss_position.owner
            and coercion_position.field_name == loss_position.field_name
            and coercion_position.index < loss_position.index
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = [node.target]
        else:
            continue
        position = positions.get(id(node))
        if position is None:
            continue

        if isinstance(value, ast.Dict):
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            base = targets[0].id
            literal_keys = [
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            has_unpack = any(key is None for key in value.keys)
            has_dynamic_key = any(
                key is not None and not isinstance(key, ast.Constant)
                for key in value.keys
            )
            for key, candidate_value in zip(value.keys, value.values):
                if not (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and candidate_value is not None
                ):
                    continue
                matched = _contains_loss_count(
                    candidate_value,
                    scope_id=position.scope_id,
                )
                if matched is None:
                    continue
                coercion, line, exact = matched
                bindings.append(
                    _CoercionLossBinding(
                        coercion=coercion,
                        guard_binding=_GuardBinding(
                            kind="dict_key",
                            name=key.value,
                            base=base,
                            scope_id=position.scope_id,
                            runtime_stable=(
                                exact
                                and _straight_line_from_coercion(
                                    coercion,
                                    position,
                                )
                                and not has_unpack
                                and not has_dynamic_key
                                and literal_keys.count(key.value) == 1
                            ),
                        ),
                        count_line=line,
                        statement_id=id(node),
                        statement=node,
                    )
                )
            continue

        matched = _contains_loss_count(value, scope_id=position.scope_id)
        if matched is None:
            continue
        coercion, line, exact = matched
        for target in targets:
            bindings.extend(
                _CoercionLossBinding(
                    coercion=coercion,
                    guard_binding=_GuardBinding(
                        kind="name",
                        name=name,
                        base=None,
                        scope_id=position.scope_id,
                        runtime_stable=(
                            exact
                            and _straight_line_from_coercion(
                                coercion,
                                position,
                            )
                        ),
                    ),
                    count_line=line,
                    statement_id=id(node),
                    statement=node,
                )
                for name in _target_names(target)
            )
    return bindings


def _binding_expression_matches(
    node: ast.AST,
    binding: _GuardBinding,
    *,
    scope_id: int,
    builtin_int_unmodified: bool,
) -> bool:
    if not binding.runtime_stable:
        return False
    if scope_id != binding.scope_id:
        return False
    current = node
    if binding.kind == "name":
        return isinstance(current, ast.Name) and current.id == binding.name
    return bool(
        binding.kind == "dict_key"
        and isinstance(current, ast.Subscript)
        and isinstance(current.value, ast.Name)
        and current.value.id == binding.base
        and isinstance(current.slice, ast.Constant)
        and current.slice.value == binding.name
    )


def _literal_int(node: ast.AST, value: int) -> bool:
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and node.value == value
    )


def _positive_count_test(
    node: ast.AST,
    binding: _GuardBinding,
    *,
    scope_id: int,
    builtin_int_unmodified: bool,
) -> bool:
    """Return whether an ``if`` condition is true exactly when count > 0."""

    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # ``a_loss > 0 or b_loss > 0`` fail-closes for each named loss count:
        # either positive operand is sufficient to enter the terminating body.
        # ``and`` is deliberately excluded because the other operand could keep
        # the guard false while this binding is positive.
        return any(
            _positive_count_test(
                value,
                binding,
                scope_id=scope_id,
                builtin_int_unmodified=builtin_int_unmodified,
            )
            for value in node.values
        )
    if _binding_expression_matches(
        node,
        binding,
        scope_id=scope_id,
        builtin_int_unmodified=builtin_int_unmodified,
    ):
        return True
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False
    left = node.left
    right = node.comparators[0]
    op = node.ops[0]
    if _binding_expression_matches(
        left,
        binding,
        scope_id=scope_id,
        builtin_int_unmodified=builtin_int_unmodified,
    ):
        return (
            isinstance(op, ast.Gt)
            and _literal_int(right, 0)
            or isinstance(op, ast.GtE)
            and _literal_int(right, 1)
            or isinstance(op, ast.NotEq)
            and _literal_int(right, 0)
        )
    if _binding_expression_matches(
        right,
        binding,
        scope_id=scope_id,
        builtin_int_unmodified=builtin_int_unmodified,
    ):
        return (
            isinstance(op, ast.Lt)
            and _literal_int(left, 0)
            or isinstance(op, ast.LtE)
            and _literal_int(left, 1)
            or isinstance(op, ast.NotEq)
            and _literal_int(left, 0)
        )
    return False


def _zero_count_assertion(
    node: ast.AST,
    binding: _GuardBinding,
    *,
    scope_id: int,
    builtin_int_unmodified: bool,
) -> bool:
    """Return whether an assertion fails whenever the count is positive."""

    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        # ``assert a_loss == 0 and b_loss == 0`` fails if either audited count
        # is positive.  An ``or`` assertion would not provide that guarantee.
        return any(
            _zero_count_assertion(
                value,
                binding,
                scope_id=scope_id,
                builtin_int_unmodified=builtin_int_unmodified,
            )
            for value in node.values
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _binding_expression_matches(
            node.operand,
            binding,
            scope_id=scope_id,
            builtin_int_unmodified=builtin_int_unmodified,
        )
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False
    left = node.left
    right = node.comparators[0]
    op = node.ops[0]
    if _binding_expression_matches(
        left,
        binding,
        scope_id=scope_id,
        builtin_int_unmodified=builtin_int_unmodified,
    ):
        return (
            isinstance(op, ast.Eq)
            and _literal_int(right, 0)
            or isinstance(op, ast.LtE)
            and _literal_int(right, 0)
        )
    if _binding_expression_matches(
        right,
        binding,
        scope_id=scope_id,
        builtin_int_unmodified=builtin_int_unmodified,
    ):
        return (
            isinstance(op, ast.Eq)
            and _literal_int(left, 0)
            or isinstance(op, ast.GtE)
            and _literal_int(left, 0)
        )
    return False


def _stable_raise_only_helper_call(
    statement: ast.stmt,
    *,
    position: _StatementPosition,
    tree: ast.Module,
    parents: dict[int, ast.AST],
    positions: dict[int, _StatementPosition],
) -> bool:
    """Prove a direct local helper call cannot return successfully.

    Generated scripts often centralize terminal errors in a tiny helper such as
    ``def stop(message): raise RuntimeError(message)``.  Treating every helper
    call as terminating would be fail-open, so this proof deliberately accepts
    only an undecorated, unconditionally defined same-scope function whose sole
    executable statement is ``raise`` and whose binding is never replaced.
    The helper name is irrelevant.
    """

    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ):
        return False
    helper_name = statement.value.func.id

    def _scope_binds_name(scope: ast.AST, name: str) -> bool:
        scope_id = id(scope)
        for node in ast.walk(scope):
            if (
                node is scope
                or _scope_id_for_node(
                    node,
                    parents=parents,
                    tree=tree,
                )
                != scope_id
            ):
                continue
            if (
                isinstance(node, ast.Name)
                and node.id == name
                and isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                return True
            if isinstance(node, ast.arg) and node.arg == name:
                return True
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name
            ):
                return True
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split(".", 1)[0]) == name
                for alias in node.names
            ):
                return True
            if isinstance(node, ast.ExceptHandler) and node.name == name:
                return True
            if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
                return True
            if isinstance(node, ast.MatchMapping) and node.rest == name:
                return True
        return False

    def _visible_candidate(node: ast.FunctionDef) -> bool:
        helper_position = positions.get(id(node))
        if helper_position is None or helper_position.field_name != "body":
            return False
        same_scope = bool(
            helper_position.scope_id == position.scope_id
            and helper_position.owner is position.scope
            and int(node.lineno) < int(statement.lineno)
        )
        if same_scope:
            return True
        caller = position.scope
        caller_position = positions.get(id(caller))
        return bool(
            helper_position.scope is tree
            and helper_position.owner is tree
            and isinstance(caller, (ast.FunctionDef, ast.AsyncFunctionDef))
            and caller_position is not None
            and caller_position.owner is tree
            and int(node.lineno) < int(caller.lineno)
            and not _scope_binds_name(caller, helper_name)
        )

    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == helper_name
        and _visible_candidate(node)
    ]
    if len(candidates) != 1:
        return False
    helper = candidates[0]
    executable_body = list(helper.body)
    if (
        executable_body
        and isinstance(executable_body[0], ast.Expr)
        and isinstance(executable_body[0].value, ast.Constant)
        and isinstance(executable_body[0].value.value, str)
    ):
        executable_body = executable_body[1:]
    return bool(
        not helper.decorator_list
        and len(executable_body) == 1
        and isinstance(executable_body[0], ast.Raise)
        and _function_binding_is_stable(
            tree,
            helper,
            parents=parents,
            defining_scope_id=position.scope_id,
        )
    )


def _statement_is_fail_closed_guard(
    statement: ast.stmt,
    binding: _GuardBinding,
    *,
    position: _StatementPosition,
    tree: ast.Module,
    parents: dict[int, ast.AST],
    positions: dict[int, _StatementPosition],
    builtin_int_unmodified: bool,
) -> bool:
    if not coercion_guard.guard_failure_is_terminal(
        statement,
        scope=position.scope,
        parents=parents,
    ):
        return False
    if isinstance(statement, ast.If):
        return bool(
            _positive_count_test(
                statement.test,
                binding,
                scope_id=position.scope_id,
                builtin_int_unmodified=builtin_int_unmodified,
            )
            and statement.body
            and (
                isinstance(statement.body[0], ast.Raise)
                or _stable_raise_only_helper_call(
                    statement.body[0],
                    position=position,
                    tree=tree,
                    parents=parents,
                    positions=positions,
                )
            )
        )
    return bool(
        isinstance(statement, ast.Assert)
        and _zero_count_assertion(
            statement.test,
            binding,
            scope_id=position.scope_id,
            builtin_int_unmodified=builtin_int_unmodified,
        )
    )


def _immediate_guard_for_statement(
    statement: ast.stmt,
    binding: _GuardBinding,
    *,
    tree: ast.Module,
    positions: dict[int, _StatementPosition],
    parents: dict[int, ast.AST],
    builtin_int_unmodified: bool,
) -> bool:
    position = positions.get(id(statement))
    if position is None or position.index + 1 >= len(position.block):
        return False
    following = position.block[position.index + 1]
    following_position = positions.get(id(following))
    if following_position is None or following_position.scope_id != binding.scope_id:
        return False
    return _statement_is_fail_closed_guard(
        following,
        binding,
        position=following_position,
        tree=tree,
        parents=parents,
        positions=positions,
        builtin_int_unmodified=builtin_int_unmodified,
    )


def _grouped_loss_guard_for_statement(
    statement: ast.stmt,
    binding: _GuardBinding,
    *,
    audit_statement_ids: set[int],
    tree: ast.Module,
    positions: dict[int, _StatementPosition],
    parents: dict[int, ast.AST],
    builtin_int_unmodified: bool,
) -> bool:
    """Accept one shared guard after only consecutive coercion-audit statements.

    This admits the common host-audit shape where two returned loss counts are
    unpacked around adjacent proven numeric-coercion/loss assignments and one
    ``or`` guard terminates for either.  No other statement may be crossed, so
    scientific work or a side effect cannot occur between computing a loss
    count and enforcing its guard.
    """

    position = positions.get(id(statement))
    if position is None:
        return False
    next_index = position.index + 1
    while next_index < len(position.block) and (
        id(position.block[next_index]) in audit_statement_ids
        or (
            binding.kind == "name"
            and coercion_guard.audit_record_assignment_for_count(
                position.block[next_index], count_name=binding.name
            )
        )
    ):
        next_index += 1
    if next_index >= len(position.block):
        return False
    guard = position.block[next_index]
    guard_position = positions.get(id(guard))
    if guard_position is None or guard_position.scope_id != binding.scope_id:
        return False
    return _statement_is_fail_closed_guard(
        guard,
        binding,
        position=guard_position,
        tree=tree,
        parents=parents,
        positions=positions,
        builtin_int_unmodified=builtin_int_unmodified,
    )


def _function_returns(
    function: ast.FunctionDef,
) -> list[ast.Return]:
    returns: list[ast.Return] = []

    class _ReturnVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Return(self, node: ast.Return) -> None:
            returns.append(node)

    visitor = _ReturnVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return returns


def _returned_name_slot(
    value: Optional[ast.AST],
    name: str,
) -> Optional[tuple[int, int]]:
    if isinstance(value, ast.Name) and value.id == name:
        return 0, 1
    if not isinstance(value, (ast.Tuple, ast.List)):
        return None
    slots = [
        index
        for index, element in enumerate(value.elts)
        if isinstance(element, ast.Name) and element.id == name
    ]
    if len(slots) != 1:
        return None
    return slots[0], len(value.elts)


def _assigned_name_for_slot(
    target: ast.AST,
    *,
    slot: int,
    width: int,
) -> Optional[str]:
    if width == 1 and isinstance(target, ast.Name):
        return target.id
    if not (
        isinstance(target, (ast.Tuple, ast.List))
        and len(target.elts) == width
        and isinstance(target.elts[slot], ast.Name)
    ):
        return None
    return target.elts[slot].id


def _function_binding_is_stable(
    tree: ast.Module,
    function: ast.FunctionDef,
    *,
    parents: dict[int, ast.AST],
    defining_scope_id: int,
) -> bool:
    """Reject decorators or same-scope rebinding of a receipt helper."""

    if function.decorator_list:
        return False
    for node in ast.walk(tree):
        if node is function:
            continue
        if _scope_id_for_node(node, parents=parents, tree=tree) != defining_scope_id:
            continue
        if (
            isinstance(node, ast.Name)
            and node.id == function.name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            return False
        if isinstance(node, ast.arg) and node.arg == function.name:
            return False
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == function.name
        ):
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name.split(".")[0]) == function.name
            for alias in node.names
        ):
            return False
        if isinstance(node, ast.ExceptHandler) and node.name == function.name:
            return False
        if isinstance(node, ast.MatchAs) and node.name == function.name:
            return False
        if isinstance(node, ast.MatchStar) and node.name == function.name:
            return False
        if isinstance(node, ast.MatchMapping) and node.rest == function.name:
            return False
    return True


def _exported_receipt_guard_proves_failure(
    tree: ast.Module,
    binding: _CoercionLossBinding,
    *,
    parents: dict[int, ast.AST],
    positions: dict[int, _StatementPosition],
    builtin_int_unmodified: bool,
) -> bool:
    """Prove every direct call immediately guards one returned audit dict."""

    loss_guard = binding.guard_binding
    function = binding.coercion.scope
    loss_position = positions.get(binding.statement_id)
    if not (
        loss_guard.kind == "dict_key"
        and loss_guard.base
        and isinstance(function, ast.FunctionDef)
        and not function.decorator_list
        and loss_position is not None
        and loss_position.owner is function
        and loss_position.field_name == "body"
        and loss_position.index + 1 < len(loss_position.block)
    ):
        return False
    returned = loss_position.block[loss_position.index + 1]
    returns = _function_returns(function)
    if not (
        isinstance(returned, ast.Return)
        and returns == [returned]
        and (slot_info := _returned_name_slot(returned.value, loss_guard.base))
        is not None
    ):
        return False
    if _has_dynamic_namespace_indirection(tree):
        return False

    function_position = positions.get(id(function))
    if function_position is None or not isinstance(function_position.owner, ast.Module):
        return False
    if not _function_binding_is_stable(
        tree,
        function,
        parents=parents,
        defining_scope_id=function_position.scope_id,
    ):
        return False
    if (
        sum(
            1
            for statement in function_position.block
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == function.name
        )
        != 1
    ):
        return False

    name_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == function.name
    ]
    calls: list[ast.Call] = []
    for name_load in name_loads:
        parent = parents.get(id(name_load))
        if not (
            isinstance(parent, ast.Call)
            and parent.func is name_load
            and parent not in calls
        ):
            return False
        calls.append(parent)
    if not calls:
        return False

    slot, width = slot_info
    for call in calls:
        assignment = parents.get(id(call))
        if isinstance(assignment, ast.Assign):
            if assignment.value is not call or len(assignment.targets) != 1:
                return False
            target = assignment.targets[0]
        elif isinstance(assignment, ast.AnnAssign):
            if assignment.value is not call:
                return False
            target = assignment.target
        else:
            return False
        assignment_position = positions.get(id(assignment))
        if not (
            assignment_position is not None
            and assignment_position.owner is function_position.owner
            and assignment_position.field_name == function_position.field_name
            and assignment_position.index > function_position.index
        ):
            return False
        audit_name = _assigned_name_for_slot(target, slot=slot, width=width)
        if audit_name is None:
            return False
        caller_binding = _GuardBinding(
            kind="dict_key",
            name=loss_guard.name,
            base=audit_name,
            scope_id=assignment_position.scope_id,
            runtime_stable=loss_guard.runtime_stable,
        )
        if not _immediate_guard_for_statement(
            assignment,
            caller_binding,
            tree=tree,
            positions=positions,
            parents=parents,
            builtin_int_unmodified=builtin_int_unmodified,
        ):
            return False
    return True


def _guarded_coercion_roots(
    tree: ast.Module,
    bindings: list[_CoercionLossBinding],
    *,
    parents: dict[int, ast.AST],
    positions: dict[int, _StatementPosition],
) -> set[_NumericCoercionSite]:
    guarded: set[_NumericCoercionSite] = set()
    builtin_int_unmodified = _builtin_int_binding_is_unmodified(tree)
    audit_statement_ids = {
        statement_id
        for binding in bindings
        for statement_id in (binding.statement_id, binding.coercion.statement_id)
    }
    for binding in bindings:
        if (
            _immediate_guard_for_statement(
                binding.statement,
                binding.guard_binding,
                tree=tree,
                positions=positions,
                parents=parents,
                builtin_int_unmodified=builtin_int_unmodified,
            )
            or _grouped_loss_guard_for_statement(
                binding.statement,
                binding.guard_binding,
                audit_statement_ids=audit_statement_ids,
                tree=tree,
                positions=positions,
                parents=parents,
                builtin_int_unmodified=builtin_int_unmodified,
            )
            or _exported_receipt_guard_proves_failure(
                tree,
                binding,
                parents=parents,
                positions=positions,
                builtin_int_unmodified=builtin_int_unmodified,
            )
        ):
            guarded.add(binding.coercion)
    return guarded


def _notna_gated_domain_checks(
    tree: ast.Module,
    coercion_sites: list[_NumericCoercionSite],
    *,
    parents: dict[int, ast.AST],
) -> list[tuple[_NumericCoercionSite, int]]:
    """Return coerced roots whose domain checks exclude null values."""

    checks: list[tuple[_NumericCoercionSite, int]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)):
            continue
        scope_id = _scope_id_for_node(node, parents=parents, tree=tree)
        terms = _flatten_bitand_terms(node)
        notna_roots = {
            root
            for term in terms
            for root in [_series_method_root(term, {"notna", "notnull"})]
            if root
        }
        for term in terms:
            root = _series_method_root(term, {"isin", "between"})
            candidates = [
                site
                for site in coercion_sites
                if site.scope_id == scope_id
                and site.root == root
                and site.line <= int(term.lineno)
            ]
            if root and root in notna_roots and candidates:
                checks.append(
                    (
                        max(
                            candidates,
                            key=lambda site: (site.line, site.statement_id),
                        ),
                        int(term.lineno),
                    )
                )
    return list(dict.fromkeys(checks))


_HOST_HELPER_CALL_CONTRACTS: dict[tuple[str, str], dict[str, object]] = {
    (
        "easyicu.research_agent.figures.publication",
        "audit_publication_exports",
    ): {
        "max_positional": 1,
        "positional_parameter": "paths",
        "required_keywords": (),
        "allowed_keywords": ("paths", "min_bytes", "require_svg_text"),
    },
    (
        "easyicu.research_agent.methods.descriptive_inputs",
        "closed_categorical_counts",
    ): {
        "max_positional": 1,
        "positional_parameter": "series",
        "required_keywords": ("declared_levels",),
        "allowed_keywords": ("series", "declared_levels"),
    },
    (
        "easyicu.research_agent.methods.descriptive_inputs",
        "measurement_provenance_receipt",
    ): {
        "max_positional": 1,
        "positional_parameter": "frame",
        "required_keywords": ("measured_column", "count_column"),
        "allowed_keywords": ("frame", "measured_column", "count_column"),
    },
}


def _host_helper_call_signature_findings(
    tree: ast.Module,
    step: AnalysisStep,
) -> list[ValidationFinding]:
    """Validate calls to imported, stable host helpers before execution.

    This is a small explicit host-API registry, not runtime introspection and
    not a guess based on a helper-like name. Only an exact import from a
    registered module grants host authority. Calls through locally shadowed
    bindings are ignored here and remain ordinary generated code.
    """

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    imported_calls: dict[
        int,
        dict[str, list[tuple[int, str, dict[str, object], ast.AST]]],
    ] = {}
    for node in ast.walk(tree):
        scope_id = _scope_id_for_node(node, parents=parents, tree=tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                contract = _HOST_HELPER_CALL_CONTRACTS.get((node.module, alias.name))
                if contract is not None:
                    call_name = alias.asname or alias.name
                    imported_calls.setdefault(scope_id, {}).setdefault(
                        call_name, []
                    ).append((int(node.lineno), alias.name, contract, node))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                matching = [
                    (symbol, contract)
                    for (
                        module,
                        symbol,
                    ), contract in _HOST_HELPER_CALL_CONTRACTS.items()
                    if module == alias.name
                ]
                for symbol, contract in matching:
                    module_name = alias.asname or alias.name
                    call_name = f"{module_name}.{symbol}"
                    imported_calls.setdefault(scope_id, {}).setdefault(
                        call_name, []
                    ).append((int(node.lineno), symbol, contract, node))
    if not imported_calls:
        return []

    def _scope_chain(node: ast.AST) -> list[int]:
        chain: list[int] = []
        current: Optional[ast.AST] = node
        while current is not None:
            current = parents.get(id(current))
            if isinstance(current, _LEXICAL_SCOPE_NODES):
                chain.append(id(current))
        if id(tree) not in chain:
            chain.append(id(tree))
        return chain

    def _scope_binds_root(scope_id: int, root: str) -> bool:
        for node in ast.walk(tree):
            if _scope_id_for_node(node, parents=parents, tree=tree) != scope_id:
                continue
            if isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                if node.id == root:
                    return True
            if isinstance(node, ast.arg) and node.arg == root:
                return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == root:
                    return True
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(
                    (alias.asname or alias.name.split(".", 1)[0]) == root
                    for alias in node.names
                ):
                    return True
        return False

    def _binding_for_call(
        call: ast.Call,
        call_name: str,
    ) -> tuple[str, dict[str, object]] | None:
        root = call_name.split(".", 1)[0]
        call_line = int(call.lineno)
        chain = _scope_chain(call)
        for index, scope_id in enumerate(chain):
            candidates = [
                item
                for item in imported_calls.get(scope_id, {}).get(call_name, [])
                if item[0] <= call_line
            ]
            if candidates:
                import_line, helper_name, contract, import_node = max(
                    candidates,
                    key=lambda item: item[0],
                )
                rebound = False
                for node in ast.walk(tree):
                    if node is import_node or not hasattr(node, "lineno"):
                        continue
                    if _scope_id_for_node(
                        node, parents=parents, tree=tree
                    ) != scope_id or not (import_line < int(node.lineno) <= call_line):
                        continue
                    if (
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, (ast.Store, ast.Del))
                        and node.id == root
                    ):
                        rebound = True
                        break
                    if (
                        isinstance(
                            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                        )
                        and node.name == root
                    ):
                        rebound = True
                        break
                    if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                        (alias.asname or alias.name.split(".", 1)[0]) == root
                        for alias in node.names
                    ):
                        rebound = True
                        break
                if not rebound:
                    return helper_name, contract
                return None
            if index + 1 < len(chain) and _scope_binds_root(scope_id, root):
                return None
        return None

    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        imported = _binding_for_call(node, call_name)
        if imported is None:
            continue
        helper_name, contract = imported
        max_positional = int(contract["max_positional"])
        positional_parameter = str(contract["positional_parameter"])
        required_keywords = tuple(contract["required_keywords"])
        allowed_keywords = set(contract["allowed_keywords"])
        keyword_names = [keyword.arg for keyword in node.keywords]
        violations: list[str] = []
        detail_additions: dict[str, object] = {}
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            violations.append("starred_positional_arguments_unverifiable")
        if len(node.args) > max_positional:
            violations.append("keyword_only_parameters_passed_positionally")
        if any(name is None for name in keyword_names):
            violations.append("expanded_keyword_arguments_unverifiable")
        explicit_keywords = [name for name in keyword_names if name is not None]
        if len(explicit_keywords) != len(set(explicit_keywords)):
            violations.append("duplicate_keyword_arguments")
        if any(name not in allowed_keywords for name in explicit_keywords):
            violations.append("unknown_keyword_argument")
        if node.args and positional_parameter in explicit_keywords:
            violations.append(f"{positional_parameter}_bound_more_than_once")
        if not node.args and positional_parameter not in explicit_keywords:
            violations.append(f"{positional_parameter}_argument_missing")
        if not set(required_keywords) <= set(explicit_keywords):
            violations.append("required_keyword_only_argument_missing")
        if helper_name == "measurement_provenance_receipt":
            keyword_values = {
                str(keyword.arg): keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
            measured_node = keyword_values.get("measured_column")
            count_node = keyword_values.get("count_column")
            measured_column = (
                str(measured_node.value)
                if isinstance(measured_node, ast.Constant)
                and isinstance(measured_node.value, str)
                else None
            )
            count_column = (
                str(count_node.value)
                if isinstance(count_node, ast.Constant)
                and isinstance(count_node.value, str)
                else None
            )
            declared_inputs = {
                str(value).strip()
                for value in step.inputs
                if ":" not in str(value) and str(value).strip()
            }
            expected_count = (
                companion_count_column_for_measured(measured_column)
                if measured_column is not None
                else None
            )
            measured_candidates = sorted(
                value
                for value in declared_inputs
                if count_column is not None
                and companion_count_column_for_measured(value) == count_column
            )
            role_contract_relevant = bool(
                (measured_column is not None and measured_column in declared_inputs)
                or (count_column is not None and count_column in declared_inputs)
                or (expected_count is not None and expected_count in declared_inputs)
                or measured_candidates
            )
            if (
                role_contract_relevant
                and measured_column is not None
                and expected_count is None
            ):
                violations.append("measured_column_role_invalid")
                detail_additions["observed_measured_column"] = measured_column
                if len(measured_candidates) == 1:
                    detail_additions["expected_measured_column"] = measured_candidates[
                        0
                    ]
            if (
                role_contract_relevant
                and count_column is not None
                and not measured_candidates
            ):
                violations.append("count_column_role_invalid")
                detail_additions["observed_count_column"] = count_column
                if expected_count is not None and expected_count in declared_inputs:
                    detail_additions["expected_count_column"] = expected_count
            if (
                role_contract_relevant
                and measured_column is not None
                and count_column is not None
                and expected_count is not None
                and expected_count != count_column
            ):
                violations.append("measurement_companion_columns_mismatch")
                detail_additions.setdefault(
                    "observed_measured_column",
                    measured_column,
                )
                detail_additions.setdefault("observed_count_column", count_column)
                if expected_count in declared_inputs:
                    detail_additions.setdefault("expected_count_column", expected_count)
                if len(measured_candidates) == 1:
                    detail_additions.setdefault(
                        "expected_measured_column",
                        measured_candidates[0],
                    )
        if not violations:
            continue
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "A call to a registered host-owned helper violates its stable "
                    "argument contract and would fail only after sandbox launch."
                ),
                detail={
                    "reason": "host_helper_call_signature_invalid",
                    "helper_name": helper_name,
                    "line": int(node.lineno),
                    "max_positional": max_positional,
                    "required_keywords": list(required_keywords),
                    "violations": sorted(set(violations)),
                    **detail_additions,
                },
            )
        )
    return findings


_BOOLEAN_REDUCTION_METHODS = frozenset({"all", "any"})
_PANDAS_SERIES_METHODS = frozenset(
    {
        "astype",
        "between",
        "dropna",
        "eq",
        "ge",
        "gt",
        "isna",
        "isin",
        "le",
        "lt",
        "ne",
        "notna",
    }
)
_NUMPY_ARRAY_CONSTRUCTORS = frozenset(
    {"array", "asarray", "empty", "full", "ones", "zeros"}
)


def _boolean_reduction_identity_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject identity comparison against pandas/numpy boolean reductions.

    Python's ``True``/``False`` singletons are not identical to
    ``numpy.bool_``.  Detection is intentionally broader than automatic
    repair: unresolved ``.all()``/``.any()`` receivers remain blocking, while
    only reductions proven to return a total scalar boolean are repairable.
    Locally defined custom classes are outside this pandas/numpy contract.
    """

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    import_bindings: dict[int, dict[str, list[tuple[int, str, ast.AST]]]] = {}
    custom_classes = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    for node in ast.walk(tree):
        scope_id = _scope_id_for_node(node, parents=parents, tree=tree)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"numpy", "pandas"}:
                    continue
                local_name = alias.asname or alias.name
                import_bindings.setdefault(scope_id, {}).setdefault(
                    local_name, []
                ).append((int(node.lineno), f"{alias.name}_module", node))
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module not in {"numpy", "pandas"}:
                continue
            for alias in node.names:
                if node.module == "numpy" and alias.name in _BOOLEAN_REDUCTION_METHODS:
                    kind = f"numpy_{alias.name}"
                elif node.module == "pandas" and alias.name in {
                    "DataFrame",
                    "Series",
                }:
                    kind = f"pandas_{alias.name.lower()}_constructor"
                else:
                    continue
                local_name = alias.asname or alias.name
                import_bindings.setdefault(scope_id, {}).setdefault(
                    local_name, []
                ).append((int(node.lineno), kind, node))

    def _scope_chain(node: ast.AST) -> list[int]:
        chain: list[int] = []
        current: Optional[ast.AST] = node
        while current is not None:
            current = parents.get(id(current))
            if isinstance(current, _LEXICAL_SCOPE_NODES):
                chain.append(id(current))
        if id(tree) not in chain:
            chain.append(id(tree))
        return chain

    def _scope_binds_name(scope_id: int, name: str, *, ignore: ast.AST) -> bool:
        for candidate in ast.walk(tree):
            if (
                candidate is ignore
                or _scope_id_for_node(candidate, parents=parents, tree=tree) != scope_id
            ):
                continue
            if (
                isinstance(candidate, ast.Name)
                and isinstance(candidate.ctx, (ast.Store, ast.Del))
                and candidate.id == name
            ):
                return True
            if isinstance(candidate, ast.arg) and candidate.arg == name:
                return True
            if (
                isinstance(
                    candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and candidate.name == name
            ):
                return True
        return False

    def _import_kind(name: str, at_node: ast.AST) -> Optional[str]:
        line = int(getattr(at_node, "lineno", 0))
        for scope_id in _scope_chain(at_node):
            candidates = [
                item
                for item in import_bindings.get(scope_id, {}).get(name, [])
                if item[0] <= line
            ]
            if candidates:
                _, kind, import_node = max(candidates, key=lambda item: item[0])
                if not _scope_binds_name(scope_id, name, ignore=import_node):
                    return kind
                return None
            # Any local binding shadows an outer import for the whole scope.
            sentinel = ast.Pass()
            if _scope_binds_name(scope_id, name, ignore=sentinel):
                return None
        return None

    def _latest_assignment(name: str, at_node: ast.AST) -> Optional[ast.AST]:
        scope_id = _scope_id_for_node(at_node, parents=parents, tree=tree)
        line = int(getattr(at_node, "lineno", 0))
        candidates: list[tuple[int, ast.AST]] = []
        for candidate in ast.walk(tree):
            if (
                _scope_id_for_node(candidate, parents=parents, tree=tree) != scope_id
                or int(getattr(candidate, "lineno", 0)) >= line
            ):
                continue
            if isinstance(candidate, (ast.Assign, ast.NamedExpr)):
                targets = (
                    candidate.targets
                    if isinstance(candidate, ast.Assign)
                    else [candidate.target]
                )
                if any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in targets
                ):
                    candidates.append((int(candidate.lineno), candidate.value))
            elif (
                isinstance(candidate, ast.AnnAssign)
                and isinstance(candidate.target, ast.Name)
                and candidate.target.id == name
                and candidate.value is not None
            ):
                candidates.append((int(candidate.lineno), candidate.value))
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    def _expression_kind(
        expression: ast.AST,
        *,
        at_node: ast.AST,
        seen_names: frozenset[str] = frozenset(),
    ) -> str:
        if isinstance(expression, ast.Name):
            if expression.id in seen_names:
                return "unknown"
            assigned = _latest_assignment(expression.id, at_node)
            if assigned is None:
                return "unknown"
            return _expression_kind(
                assigned,
                at_node=at_node,
                seen_names=seen_names | {expression.id},
            )
        if isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name):
                if expression.func.id in custom_classes:
                    return "custom"
                imported = _import_kind(expression.func.id, expression)
                if imported == "pandas_series_constructor":
                    return "pandas_series"
                if imported == "pandas_dataframe_constructor":
                    return "pandas_dataframe"
            if isinstance(expression.func, ast.Attribute):
                base = expression.func.value
                method = expression.func.attr
                if isinstance(base, ast.Name):
                    imported = _import_kind(base.id, expression)
                    if imported == "pandas_module":
                        if method == "Series":
                            return "pandas_series"
                        if method == "DataFrame" or method.startswith("read_"):
                            return "pandas_dataframe"
                    if imported == "numpy_module" and method in (
                        _NUMPY_ARRAY_CONSTRUCTORS
                    ):
                        return "numpy_array"
                receiver_kind = _expression_kind(
                    base,
                    at_node=at_node,
                    seen_names=seen_names,
                )
                if method in _PANDAS_SERIES_METHODS and receiver_kind in {
                    "pandas_series",
                    "pandas_dataframe",
                }:
                    return receiver_kind
                if receiver_kind == "numpy_array" and method in {
                    "astype",
                    "copy",
                    "reshape",
                    "ravel",
                }:
                    return receiver_kind
            return "unknown"
        if isinstance(expression, ast.Subscript):
            owner_kind = _expression_kind(
                expression.value,
                at_node=at_node,
                seen_names=seen_names,
            )
            if owner_kind == "pandas_dataframe":
                if isinstance(expression.slice, ast.Constant) and isinstance(
                    expression.slice.value, str
                ):
                    return "pandas_series"
                return "pandas_dataframe"
            if owner_kind == "numpy_array":
                return "numpy_array"
        return "unknown"

    def _literal_keyword(call: ast.Call, name: str) -> tuple[bool, object]:
        matches = [keyword for keyword in call.keywords if keyword.arg == name]
        if len(matches) != 1 or not isinstance(matches[0].value, ast.Constant):
            return False, None
        return True, matches[0].value.value

    def _reduction_info(expression: ast.AST) -> Optional[tuple[str, str, bool]]:
        if not isinstance(expression, ast.Call):
            return None

        reduction: Optional[str] = None
        provenance = "unknown"
        if isinstance(expression.func, ast.Attribute) and isinstance(
            expression.func.value, ast.Name
        ):
            imported = _import_kind(expression.func.value.id, expression)
            if (
                imported == "numpy_module"
                and expression.func.attr in _BOOLEAN_REDUCTION_METHODS
            ):
                reduction = expression.func.attr
                provenance = "numpy_function"
        elif isinstance(expression.func, ast.Name):
            imported = _import_kind(expression.func.id, expression)
            if imported in {"numpy_all", "numpy_any"}:
                reduction = imported.removeprefix("numpy_")
                provenance = "numpy_function"
        if reduction is not None:
            if any(
                isinstance(argument, ast.Starred) for argument in expression.args
            ) or any(keyword.arg is None for keyword in expression.keywords):
                return "dynamic", provenance, False
            scalar = len(expression.args) == 1
            if any(keyword.arg != "axis" for keyword in expression.keywords):
                scalar = False
            axis_present, axis = _literal_keyword(expression, "axis")
            if any(keyword.arg == "axis" for keyword in expression.keywords) and (
                not axis_present or axis is not None
            ):
                scalar = False
            return reduction, provenance, scalar

        if isinstance(expression.func, ast.Attribute) and expression.func.attr in (
            _BOOLEAN_REDUCTION_METHODS
        ):
            reduction = expression.func.attr
            provenance = _expression_kind(
                expression.func.value,
                at_node=expression,
            )
            if provenance in {"custom", "unknown"}:
                return None
            if any(
                isinstance(argument, ast.Starred) for argument in expression.args
            ) or any(keyword.arg is None for keyword in expression.keywords):
                return "dynamic", provenance, False
            scalar = provenance in {"pandas_series", "numpy_array"}
            if expression.args:
                scalar = False
            allowed_keywords = (
                {"axis", "skipna"} if provenance == "pandas_series" else {"axis"}
            )
            if any(
                keyword.arg not in allowed_keywords for keyword in expression.keywords
            ):
                scalar = False
            axis_present, axis = _literal_keyword(expression, "axis")
            if any(keyword.arg == "axis" for keyword in expression.keywords) and (
                not axis_present or axis is not None
            ):
                scalar = False
            skipna_present, skipna = _literal_keyword(expression, "skipna")
            if any(keyword.arg == "skipna" for keyword in expression.keywords) and (
                not skipna_present or skipna is not True
            ):
                scalar = False
            return reduction, provenance, scalar

        return None

    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for index, operator in enumerate(node.ops):
            if not isinstance(operator, (ast.Is, ast.IsNot)):
                continue
            left, right = operands[index], operands[index + 1]
            if (
                isinstance(left, ast.Constant)
                and isinstance(left.value, bool)
                and (info := _reduction_info(right)) is not None
            ):
                boolean_literal = left.value
            elif (
                isinstance(right, ast.Constant)
                and isinstance(right.value, bool)
                and (info := _reduction_info(left)) is not None
            ):
                boolean_literal = right.value
            else:
                continue
            reduction, provenance, scalar = info
            repair_safe = len(node.ops) == 1 and scalar
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "A pandas/numpy boolean reduction is compared to a "
                        "Python boolean singleton by identity; use value truth "
                        "semantics only when the reduction is a proven scalar."
                    ),
                    detail={
                        "reason": "boolean_reduction_identity_comparison",
                        "line": int(node.lineno),
                        "operator": (
                            "is_not" if isinstance(operator, ast.IsNot) else "is"
                        ),
                        "boolean_literal": bool(boolean_literal),
                        "reduction": reduction,
                        "provenance": provenance,
                        "repair_safe": repair_safe,
                    },
                )
            )
    return findings


def _local_helper_unpack_arity_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject statically provable local return/unpack arity mismatches.

    Only module-level helpers whose every direct return is a fixed tuple/list
    of the same width are claimed. Dynamic, branching-width, starred, nested,
    attribute, and indirect calls remain outside this mechanical contract.
    """

    def _direct_body_nodes(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[ast.AST]:
        nodes: list[ast.AST] = []
        pending: list[ast.AST] = list(reversed(function.body))
        while pending:
            node = pending.pop()
            nodes.append(node)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                continue
            pending.extend(reversed(list(ast.iter_child_nodes(node))))
        return nodes

    helpers: dict[str, tuple[int, tuple[int, ...]]] = {}
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        returns = [
            node
            for node in _direct_body_nodes(function)
            if isinstance(node, ast.Return)
        ]
        if not returns or any(
            not isinstance(node.value, (ast.Tuple, ast.List)) for node in returns
        ):
            continue
        widths = {len(node.value.elts) for node in returns}
        if len(widths) != 1:
            continue
        helpers[function.name] = (
            next(iter(widths)),
            tuple(sorted(int(node.lineno) for node in returns)),
        )
    if not helpers:
        return []

    findings: list[ValidationFinding] = []
    for statement in tree.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in _direct_body_nodes(statement):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in helpers
                and len(targets) == 1
                and isinstance(targets[0], (ast.Tuple, ast.List))
                and not any(isinstance(item, ast.Starred) for item in targets[0].elts)
            ):
                continue
            return_arity, return_lines = helpers[value.func.id]
            target_arity = len(targets[0].elts)
            if return_arity == target_arity:
                continue
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "A statically defined local helper returns a fixed number "
                        "of values that does not match its direct unpack target."
                    ),
                    detail={
                        "reason": "local_helper_unpack_arity_mismatch",
                        "function_name": value.func.id,
                        "call_line": int(value.lineno),
                        "return_lines": list(return_lines),
                        "return_arity": return_arity,
                        "target_arity": target_arity,
                    },
                )
            )
    return findings


def _host_helper_runtime_introspection_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Identify runtime adaptation of stable host-owned helper APIs."""

    inspect_modules: set[str] = set()
    inspect_signature_names: set[str] = set()
    helper_references: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "inspect":
                    inspect_modules.add(alias.asname or "inspect")
                elif (
                    alias.name == "easyicu.research_agent.figures.publication"
                    and alias.asname
                ):
                    helper_references[f"{alias.asname}.save_publication_figure"] = (
                        "save_publication_figure"
                    )
                elif (
                    alias.name == "easyicu.research_agent.methods.descriptive_inputs"
                    and alias.asname
                ):
                    helper_references[f"{alias.asname}.closed_categorical_counts"] = (
                        "closed_categorical_counts"
                    )
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == "inspect":
                inspect_signature_names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "signature"
                )
            elif node.module == "easyicu.research_agent.figures.publication":
                helper_references.update(
                    {
                        alias.asname or alias.name: "save_publication_figure"
                        for alias in node.names
                        if alias.name == "save_publication_figure"
                    }
                )
            elif node.module == "easyicu.research_agent.methods.descriptive_inputs":
                helper_references.update(
                    {
                        alias.asname or alias.name: "closed_categorical_counts"
                        for alias in node.names
                        if alias.name == "closed_categorical_counts"
                    }
                )
    if not helper_references:
        return []

    signature_call_names = {
        *(f"{name}.signature" for name in inspect_modules),
        *inspect_signature_names,
    }
    introspection_nodes: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _call_name(node.func) in signature_call_names
            and node.args
            and _call_name(node.args[0]) in helper_references
        ):
            reference = _call_name(node.args[0])
            introspection_nodes.append((node, helper_references[reference]))
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "__signature__"
            and _call_name(node.value) in helper_references
        ):
            reference = _call_name(node.value)
            introspection_nodes.append((node, helper_references[reference]))
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "Generated code must call the stable host-owned "
                "helper API directly instead of adapting its runtime signature."
            ),
            detail={
                "reason": "host_helper_runtime_introspection",
                "helper_name": helper_name,
                "line": int(node.lineno),
            },
        )
        for node, helper_name in introspection_nodes
    ]


def _lossy_numeric_coercion_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Detect numeric coercion whose losses never fail closed (A1-1).

    Two structural gaps, both requiring a ``pd.to_numeric(errors="coerce")``
    source and the absence of any loss fail-close guard (a raise/assert on the
    computed loss count, or the host helper ``strict_numeric_input``):

    1. ``unchecked_coercion_loss_count`` — the script computes
       ``(original.notna() & coerced.isna()).sum()`` but never raises on it.
    2. ``domain_check_gated_on_notna`` — domain validation is conjoined with
       ``notna()``, so values nulled by coercion silently become missingness.
    """

    parents, positions = _ast_parent_and_statement_positions(tree)
    coercion_sites = _numeric_coercion_sites(tree, positions)
    if not coercion_sites:
        return []
    builtin_int_unmodified = _builtin_int_binding_is_unmodified(tree)
    loss_bindings = _coercion_loss_bindings(
        tree,
        coercion_sites,
        positions,
        builtin_int_unmodified=builtin_int_unmodified,
    )
    guarded_roots = _guarded_coercion_roots(
        tree,
        loss_bindings,
        parents=parents,
        positions=positions,
    )
    issues: list[dict[str, object]] = []
    unguarded_loss_lines = sorted(
        {
            binding.count_line
            for binding in loss_bindings
            if binding.coercion not in guarded_roots
        }
    )
    if unguarded_loss_lines:
        issues.append(
            {
                "gap": "unchecked_coercion_loss_count",
                "lines": unguarded_loss_lines,
            }
        )
    issues.extend(coercion_guard.returned_coercion_loss_issues(tree))
    domain_lines = sorted(
        {
            line
            for root, line in _notna_gated_domain_checks(
                tree,
                coercion_sites,
                parents=parents,
            )
            if root not in guarded_roots
        }
    )
    if domain_lines:
        issues.append(
            {
                "gap": "domain_check_gated_on_notna",
                "lines": domain_lines,
            }
        )
    if not issues:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "Numeric coercion can silently invalidate observed values: "
                "the script computes or implies a coercion-loss count but "
                "never fails closed on it. Add "
                "`if newly_invalid > 0: raise ValueError(...)` after "
                "`pd.to_numeric(..., errors='coerce')` (or use the host "
                "helper `strict_numeric_input`) before any notna()-gated "
                "domain validation."
            ),
            detail={
                "reason": "lossy_numeric_coercion",
                "issues": issues,
            },
        )
    ]


def _conditional_nonfinite_guard_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject a non-finite guard narrowed by an unrelated variable branch.

    Generated cohort code sometimes computes one generic non-finite mask inside
    a loop, but only raises for one named variable.  The other variables then
    fall through into a missingness/eligibility mask.  This is a mechanical
    control-flow defect: the host neither chooses the variables nor their
    scientific domains, it only requires the already-authored non-finite error
    condition to terminate for every value series to which it is applied.

    The repairable grammar is deliberately narrow: an assignment containing a
    negated ``isfinite`` call, immediately followed by ``if int(mask.sum()) >
    0`` whose sole body statement is another conditional around one proven
    raise-only statement.  More complicated control flow remains fail-closed
    for agent repair.
    """

    parents, positions = _ast_parent_and_statement_positions(tree)

    def _negates_isfinite(node: ast.AST) -> bool:
        return any(
            isinstance(candidate, ast.UnaryOp)
            and isinstance(candidate.op, (ast.Invert, ast.Not))
            and isinstance(candidate.operand, ast.Call)
            and _call_name(candidate.operand.func).split(".")[-1] == "isfinite"
            for candidate in ast.walk(node)
        )

    def _positive_mask_sum(test: ast.AST, mask_name: str) -> bool:
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Gt)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == 0
        ):
            return False
        left = test.left
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "int"
            and len(left.args) == 1
            and not left.keywords
        ):
            left = left.args[0]
        return bool(
            isinstance(left, ast.Call)
            and not left.args
            and not left.keywords
            and isinstance(left.func, ast.Attribute)
            and left.func.attr == "sum"
            and isinstance(left.func.value, ast.Name)
            and left.func.value.id == mask_name
        )

    findings: list[ValidationFinding] = []
    for assignment in ast.walk(tree):
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and _negates_isfinite(assignment.value)
        ):
            continue
        position = positions.get(id(assignment))
        if position is None or position.index + 1 >= len(position.block):
            continue
        guard = position.block[position.index + 1]
        mask_name = assignment.targets[0].id
        if not (
            isinstance(guard, ast.If)
            and not guard.orelse
            and _positive_mask_sum(guard.test, mask_name)
            and len(guard.body) == 1
            and isinstance(guard.body[0], ast.If)
        ):
            continue
        inner = guard.body[0]
        if inner.orelse or len(inner.body) != 1:
            continue
        terminal = inner.body[0]
        terminal_position = positions.get(id(terminal))
        if terminal_position is None or not (
            isinstance(terminal, ast.Raise)
            or _stable_raise_only_helper_call(
                terminal,
                position=terminal_position,
                tree=tree,
                parents=parents,
                positions=positions,
            )
        ):
            continue
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "A generic non-finite numeric guard terminates only for a "
                    "conditional subset of the value series it validates."
                ),
                detail={
                    "reason": "conditional_nonfinite_guard",
                    "assignment_line": int(assignment.lineno),
                    "guard_line": int(guard.lineno),
                    "inner_guard_line": int(inner.lineno),
                },
            )
        )
    return findings


def _strict_numeric_nonfinite_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Reject strict coercion helpers that return non-finite numeric values.

    This is deliberately narrower than a blanket ``to_numeric`` policy.  A
    candidate is claimed only when the same lexical function already proves
    coercion loss fail-closed and then returns the coerced series.  The host
    therefore adds no range, domain, variable, or missing-data decision; it
    merely requires the helper's existing *strict* contract to reject infinities
    before downstream summaries can silently turn them into nulls.
    """

    parents, positions = _ast_parent_and_statement_positions(tree)
    coercion_sites = _numeric_coercion_sites(tree, positions)
    if not coercion_sites:
        return []
    loss_bindings = _coercion_loss_bindings(
        tree,
        coercion_sites,
        positions,
        builtin_int_unmodified=_builtin_int_binding_is_unmodified(tree),
    )
    guarded_sites = _guarded_coercion_roots(
        tree,
        loss_bindings,
        parents=parents,
        positions=positions,
    )

    def _returns_name(function: ast.FunctionDef, name: str) -> bool:
        return any(
            isinstance(node, ast.Return)
            and any(
                isinstance(candidate, ast.Name) and candidate.id == name
                for candidate in ast.walk(node.value)
            )
            for node in _scope_nodes(function.body)
            if isinstance(node, ast.Return) and node.value is not None
        )

    def _negated_isfinite_for(node: ast.AST, name: str) -> bool:
        return any(
            isinstance(candidate, ast.UnaryOp)
            and isinstance(candidate.op, (ast.Invert, ast.Not))
            and isinstance(candidate.operand, ast.Call)
            and _call_name(candidate.operand.func).split(".")[-1] == "isfinite"
            and any(
                isinstance(argument_node, ast.Name) and argument_node.id == name
                for argument in candidate.operand.args
                for argument_node in ast.walk(argument)
            )
            for candidate in ast.walk(node)
        )

    def _has_guard(function: ast.FunctionDef, name: str, *, after_line: int) -> bool:
        body = function.body
        masks: set[str] = set()
        for statement in body:
            if int(getattr(statement, "lineno", -1)) <= after_line:
                continue
            if isinstance(statement, ast.Return):
                return False
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and _negated_isfinite_for(statement.value, name)
            ):
                masks.add(statement.targets[0].id)
                continue
            if not (
                isinstance(statement, ast.If)
                and statement.body
                and all(
                    isinstance(item, (ast.Raise, ast.Return)) for item in statement.body
                )
            ):
                continue
            if _positive_boolean_mask_test(
                statement.test,
                mask_names=masks,
                inline_match=lambda candidate: _negated_isfinite_for(candidate, name),
            ):
                return True
        return False

    def _coerced_name(site: _NumericCoercionSite) -> str:
        statement = site.statement
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return statement.targets[0].id
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            return statement.target.id
        return ""

    findings: list[ValidationFinding] = []
    for site in guarded_sites:
        coerced_name = _coerced_name(site)
        if not (
            isinstance(site.scope, ast.FunctionDef)
            and coerced_name
            and _returns_name(site.scope, coerced_name)
            and not _has_guard(site.scope, coerced_name, after_line=site.line)
        ):
            continue
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "A strict numeric coercion helper rejects lossy conversion "
                    "but can still return non-finite observed values. Reject "
                    "non-finite non-missing values before returning the coerced "
                    "series."
                ),
                detail={
                    "reason": "strict_numeric_nonfinite_unchecked",
                    "coercion_line": int(site.line),
                    "function_line": int(site.scope.lineno),
                },
            )
        )
    return sorted(findings, key=lambda finding: int(finding.detail["coercion_line"]))


def _positive_boolean_mask_test(
    test: ast.AST,
    *,
    mask_names: set[str],
    inline_match: Callable[[ast.AST], bool],
) -> bool:
    """Return whether a condition triggers when an invalid mask has any rows."""

    def _matches_mask(node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id in mask_names) or inline_match(
            node
        )

    if (
        isinstance(test, ast.Call)
        and not test.args
        and not test.keywords
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "any"
        and _matches_mask(test.func.value)
    ):
        return True
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and len(test.comparators) == 1
    ):
        return False

    def _sum_source(node: ast.AST) -> ast.AST | None:
        current = node
        if (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Name)
            and current.func.id == "int"
            and len(current.args) == 1
            and not current.keywords
        ):
            current = current.args[0]
        if (
            isinstance(current, ast.Call)
            and not current.args
            and not current.keywords
            and isinstance(current.func, ast.Attribute)
            and current.func.attr == "sum"
        ):
            return current.func.value
        return None

    left_source = _sum_source(test.left)
    right = test.comparators[0]
    op = test.ops[0]
    if left_source is not None and _matches_mask(left_source):
        return bool(
            isinstance(op, ast.Gt)
            and _literal_int(right, 0)
            or isinstance(op, ast.GtE)
            and _literal_int(right, 1)
            or isinstance(op, ast.NotEq)
            and _literal_int(right, 0)
        )
    right_source = _sum_source(right)
    if right_source is not None and _matches_mask(right_source):
        return bool(
            isinstance(op, ast.Lt)
            and _literal_int(test.left, 0)
            or isinstance(op, ast.LtE)
            and _literal_int(test.left, 1)
            or isinstance(op, ast.NotEq)
            and _literal_int(test.left, 0)
        )
    return False


def _categorical_level_reconciliation_findings(
    tree: ast.Module,
) -> list[ValidationFinding]:
    """Require categorical summaries to cover their declared level set.

    The detector never invents or normalizes categories.  It claims only a
    straight-line helper that drops missing values, computes ``value_counts``,
    and emits rows by iterating an Agent-authored ``levels`` argument.  Such a
    helper must fail closed when a non-missing value is absent from that same
    declared set, otherwise its reported counts cannot reconcile to its own
    denominator.
    """

    def _assignment_name(statement: ast.stmt) -> tuple[str, ast.AST] | None:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return statement.targets[0].id, statement.value
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            return statement.target.id, statement.value
        return None

    def _dropna_source(value: ast.AST) -> str:
        if (
            isinstance(value, ast.Call)
            and not value.args
            and not value.keywords
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "dropna"
            and isinstance(value.func.value, ast.Name)
        ):
            return value.func.value.id
        return ""

    def _value_counts_source(value: ast.AST) -> str:
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "value_counts"
            and isinstance(value.func.value, ast.Name)
        ):
            return value.func.value.id
        return ""

    def _is_uncovered_mask(node: ast.AST, values_name: str, levels_name: str) -> bool:
        return bool(
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.Invert, ast.Not))
            and isinstance(node.operand, ast.Call)
            and isinstance(node.operand.func, ast.Attribute)
            and node.operand.func.attr == "isin"
            and isinstance(node.operand.func.value, ast.Name)
            and node.operand.func.value.id == values_name
            and len(node.operand.args) == 1
            and isinstance(node.operand.args[0], ast.Name)
            and node.operand.args[0].id == levels_name
        )

    def _guarded(
        statements: list[ast.stmt],
        *,
        before_index: int,
        values_name: str,
        levels_name: str,
    ) -> bool:
        mask_names: set[str] = set()
        for statement in statements[:before_index]:
            assignment = _assignment_name(statement)
            if assignment is not None and _is_uncovered_mask(
                assignment[1], values_name, levels_name
            ):
                mask_names.add(assignment[0])
                continue
            if not (
                isinstance(statement, ast.If)
                and statement.body
                and all(
                    isinstance(item, (ast.Raise, ast.Return)) for item in statement.body
                )
            ):
                continue
            if _positive_boolean_mask_test(
                statement.test,
                mask_names=mask_names,
                inline_match=lambda candidate: _is_uncovered_mask(
                    candidate, values_name, levels_name
                ),
            ):
                return True
            if (
                isinstance(statement.test, ast.UnaryOp)
                and isinstance(statement.test.op, ast.Not)
                and isinstance(statement.test.operand, ast.Call)
                and isinstance(statement.test.operand.func, ast.Attribute)
                and statement.test.operand.func.attr == "all"
                and isinstance(statement.test.operand.func.value, ast.Call)
                and isinstance(statement.test.operand.func.value.func, ast.Attribute)
                and statement.test.operand.func.value.func.attr == "isin"
                and isinstance(statement.test.operand.func.value.func.value, ast.Name)
                and statement.test.operand.func.value.func.value.id == values_name
                and len(statement.test.operand.func.value.args) == 1
                and isinstance(statement.test.operand.func.value.args[0], ast.Name)
                and statement.test.operand.func.value.args[0].id == levels_name
            ):
                return True
        return False

    findings: list[ValidationFinding] = []
    for function in [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]:
        parameter_names = {argument.arg for argument in function.args.args}
        statements = function.body
        nonmissing_bindings: dict[str, str] = {}
        for index, statement in enumerate(statements):
            assignment = _assignment_name(statement)
            if assignment is None:
                continue
            target_name, value = assignment
            source_name = _dropna_source(value)
            if source_name in parameter_names:
                nonmissing_bindings[target_name] = source_name
                continue
            values_name = _value_counts_source(value)
            if values_name not in nonmissing_bindings:
                continue
            counts_name = target_name
            matching_loops = [
                later
                for later in statements[index + 1 :]
                if isinstance(later, ast.For)
                and isinstance(later.target, ast.Name)
                and isinstance(later.iter, ast.Name)
                and later.iter.id in parameter_names
                and any(
                    isinstance(candidate, ast.Call)
                    and isinstance(candidate.func, ast.Attribute)
                    and candidate.func.attr == "get"
                    and isinstance(candidate.func.value, ast.Name)
                    and candidate.func.value.id == counts_name
                    and candidate.args
                    and isinstance(candidate.args[0], ast.Name)
                    and candidate.args[0].id == later.target.id
                    for candidate in ast.walk(later)
                )
            ]
            if len(matching_loops) != 1:
                continue
            levels_name = matching_loops[0].iter.id
            if _guarded(
                statements,
                before_index=index,
                values_name=values_name,
                levels_name=levels_name,
            ):
                continue
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "Categorical rows iterate declared levels without "
                        "proving that every non-missing value belongs to those "
                        "levels, so counts can disagree with the denominator."
                    ),
                    detail={
                        "reason": "categorical_level_accounting_unverified",
                        "counts_line": int(statement.lineno),
                        "function_line": int(function.lineno),
                    },
                )
            )
    return sorted(findings, key=lambda finding: int(finding.detail["counts_line"]))


def audit_mechanical_code_contracts(
    script_text: str,
    step: AnalysisStep,
) -> list[ValidationFinding]:
    """Return implementation-only findings before any LLM concept audit."""

    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        return _pre312_fstring_subscript_quote_findings(
            str(script_text or ""),
            None,
        )
    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        fallback = _function_arbitrary_column_fallback(node)
        if fallback is None:
            continue
        line, function_name = fallback
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "Column discovery falls back to an arbitrary frame-order "
                    "column after named candidates fail; fail closed on the "
                    "missing declared schema field instead."
                ),
                detail={
                    "reason": "arbitrary_column_fallback",
                    "line": line,
                    "function": function_name,
                },
            )
        )
    findings.extend(_structural_filter_findings(tree, step))
    findings.extend(_structural_integer_findings(tree, step))
    findings.extend(_binding_metadata_findings(tree))
    findings.extend(_provenance_fail_closed_findings(tree))
    findings.extend(_provenance_pair_scan_findings(tree))
    findings.extend(_resolved_context_payload_findings(tree))
    findings.extend(_resolved_input_binding_key_findings(tree))
    findings.extend(direct_resolved_input_key_findings(tree))
    findings.extend(resolved_input_relative_path_root_findings(tree))
    findings.extend(resolved_input_shadowed_by_cohort_env_findings(tree))
    findings.extend(_pre312_fstring_subscript_quote_findings(script_text, tree))
    findings.extend(_swallowed_reconciliation_error_findings(tree))
    findings.extend(_authoritative_exposure_binding_findings(tree, step))
    findings.extend(_authoritative_exposure_fallback_findings(tree, step))
    findings.extend(_finalized_exposure_reconciliation_findings(tree, step))
    findings.extend(_typed_dataframe_erasure_findings(tree, step))
    findings.extend(_undefined_direct_call_findings(tree))
    findings.extend(_local_call_signature_findings(tree))
    findings.extend(_local_read_before_assignment_findings(tree))
    findings.extend(_branch_local_unbound_findings(tree))
    findings.extend(_ordinal_rounding_findings(tree))
    findings.extend(_scalar_cast_before_reduction_findings(tree))
    findings.extend(_first_time_companion_findings(tree))
    findings.extend(_host_helper_call_signature_findings(tree, step))
    findings.extend(host_helper_result_findings(tree, step))
    findings.extend(table_one_spec_binding_findings(tree, step))
    findings.extend(_boolean_reduction_identity_findings(tree))
    findings.extend(_local_helper_unpack_arity_findings(tree))
    findings.extend(_host_helper_runtime_introspection_findings(tree))
    findings.extend(_lossy_numeric_coercion_findings(tree))
    findings.extend(_conditional_nonfinite_guard_findings(tree))
    findings.extend(_strict_numeric_nonfinite_findings(tree))
    findings.extend(_categorical_level_reconciliation_findings(tree))
    findings.extend(
        binary_feasibility_guard_findings(tree)
        + _cohort_count_findings(tree)
        + _runtime_context_level_findings(tree)
    )
    return findings


__all__ = ["audit_mechanical_code_contracts"]
