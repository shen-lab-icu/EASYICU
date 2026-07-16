"""Deterministic mechanical checks that run before semantic LLM review.

These checks reject implementation shortcuts only.  They do not select or
rewrite the planner-owned exposure, outcome, cohort, method, or estimand.
"""

from __future__ import annotations

import ast
import builtins
import re
from typing import Optional

from .coder_context import normalised_method_head
from .schema import AnalysisStep, ValidationFinding

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


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


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


_DYNAMIC_NAMESPACE_PRIMITIVES = frozenset(
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
_DYNAMIC_NAMESPACE_MUTATORS = frozenset(
    {"__delattr__", "__setattr__", "delattr", "setattr"}
)
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
                (node.attr.startswith("__") and node.attr.endswith("__"))
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
        (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar, ast.Match),
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

    def _nearest_function(node: ast.AST) -> Optional[ast.AST]:
        current = parents.get(node)
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            current = parents.get(current)
        return current

    def _function_tokens(function: ast.AST) -> set[str]:
        return {
            str(node.value)
            for node in ast.walk(function)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _nearest_function(node) is function
        }

    marker_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _PROVENANCE_FAILURE_KEYS <= _function_tokens(node)
        and "audit_only" in _function_tokens(node)
    ]
    module_tokens = {
        str(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _nearest_function(node) is None
    }
    module_is_marker = (
        _PROVENANCE_FAILURE_KEYS <= module_tokens and "audit_only" in module_tokens
    )
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
        if isinstance(node, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)) and (
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
        role = fields.get("role")
        if not (
            _PROVENANCE_FAILURE_KEYS <= fields.keys()
            and isinstance(role, ast.Constant)
            and str(role.value).strip().lower() == "audit_only"
        ):
            return None
        return fields, containers

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
                    or call_name in _DYNAMIC_NAMESPACE_PRIMITIVES
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
        if _module_pre_guard_has_indirect_effects(preceding):
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
                isinstance(candidate, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple))
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
            if call_name in dynamic_namespace_calls:
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
            if call_name in _DYNAMIC_NAMESPACE_PRIMITIVES:
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
        return _branch_all_paths_exit(statements) and not any(
            isinstance(node, ast.Return)
            for statement in statements
            for node in ast.walk(statement)
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
            if isinstance(parent, (ast.TryStar, ast.With, ast.AsyncWith)):
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
                for statement in preceding[:audit_index]:
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
                if binding is None or not _builtin_int_call(binding):
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

    returned_slots: dict[str, Optional[int]] = {}
    self_guarded: set[str] = set()
    self_raising: set[str] = set()
    self_raising_guards: dict[str, set[ast.If]] = {}
    for name, function in marker_functions.items():
        local_nodes = _local_nodes(function)

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
                ast.Try,
                ast.TryStar,
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

        collection_events: dict[str, set[ast.Call]] = {}
        for guard in local_nodes:
            if not isinstance(guard, ast.If) or not _full_failure_test(
                guard.test,
                function,
                require_stable_signals=True,
            ):
                continue
            owner = parents.get(guard)
            direct_guard = (function is not tree and owner is function) or (
                function is tree and _module_direct_guard_is_bound(guard)
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
                and _branch_all_paths_exit(guard.body)
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
                    parents.get(candidate) is marker_function
                    and direct_body.index(candidate) > min(direct_guard_indexes)
                    for candidate in returns
                )
            )
            if (
                not (
                    isinstance(statement, ast.Expr)
                    and statement.value is call
                    and _direct_execution_statement(statement)
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

            if isinstance(statement, (ast.Try, ast.TryStar)):
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
        if isinstance(node, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)) and (
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
        if call_name.rsplit(".", 1)[-1] in _DYNAMIC_NAMESPACE_PRIMITIVES:
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
    """Reject the mechanically invalid built-in ``int(value).sum()`` form."""

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

    if not unsafe_lines:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "A built-in integer cast is applied before a zero-argument sum "
                "reduction; reduce the array-like expression before converting "
                "the resulting scalar."
            ),
            detail={
                "reason": "scalar_cast_before_reduction",
                "lines": sorted(set(unsafe_lines)),
            },
        )
    ]


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


def audit_mechanical_code_contracts(
    script_text: str,
    step: AnalysisStep,
) -> list[ValidationFinding]:
    """Return implementation-only findings before any LLM concept audit."""

    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        return []  # Runtime/code syntax handling owns this existing failure class.
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
    return findings


__all__ = ["audit_mechanical_code_contracts"]
