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
    {"fit", "fit_regularized", "predict", "savefig"}
)


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
        line = int(getattr(candidate, "lineno", 0) or 0)
        if not line or line >= guard_line:
            continue
        call_name = _call_name(candidate.func).lower()
        method = call_name.rsplit(".", 1)[-1]
        if method in _PROVENANCE_RESULT_SINK_METHODS:
            return True
        if "write_success" in method or method.startswith("publish_"):
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

    marker_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _PROVENANCE_FAILURE_KEYS <= _literal_string_tokens(node)
        and "audit_only" in _literal_string_tokens(node)
    }
    if not marker_functions:
        return []

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    expression_roles: dict[str, frozenset[str]] = {}
    signal_meanings: dict[str, tuple[str, frozenset[str]]] = {}
    audit_containers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
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
            signal_source = _provenance_signal_source(value_node)
            if signal_source is not None:
                signal_name, signal_kind = signal_source
                existing = signal_meanings.get(signal_name)
                if existing is None or existing[0] == signal_kind:
                    coverage = set(existing[1] if existing else frozenset()) | {key}
                    signal_meanings[signal_name] = signal_kind, frozenset(coverage)
        if not (_PROVENANCE_FAILURE_KEYS & set(_literal_string_tokens(node))):
            continue
        parent = parents.get(node)
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            targets = (
                parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            )
            audit_containers.update(
                name for target in targets for name in _target_names(target)
            )

    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = {name for target in targets for name in _target_names(target)}
        for name in target_names:
            assignments.setdefault(name, node.value)
        if (
            isinstance(node.value, ast.Call)
            and _call_name(node.value.func) in marker_functions
        ):
            audit_containers.update(target_names)

    failure_collections: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not (_literal_string_tokens(node.test) & _PROVENANCE_FAILURE_KEYS):
            continue
        for statement in [*node.body, *node.orelse]:
            for candidate in ast.walk(statement):
                if not isinstance(candidate, ast.Call):
                    continue
                if not isinstance(candidate.func, ast.Attribute):
                    continue
                if candidate.func.attr not in {"add", "append", "extend"}:
                    continue
                if isinstance(candidate.func.value, ast.Name):
                    failure_collections.add(candidate.func.value.id)
    for name in failure_collections:
        expression_roles[_expression_identity(ast.Name(id=name, ctx=ast.Load()))] = (
            _PROVENANCE_FULL_COVERAGE
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _branch_all_paths_exit(node.body):
            continue
        if _has_unrelated_control_ancestor(node, parents):
            continue
        if _result_sink_precedes_guard(node, parents):
            continue
        meaning = _provenance_predicate_meaning(
            node.test,
            expression_roles=expression_roles,
            signal_meanings=signal_meanings,
            assignments=assignments,
            audit_containers=audit_containers,
        )
        if meaning == (_PROVENANCE_FAILURE, _PROVENANCE_FULL_COVERAGE):
            return []

    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "A measurement-provenance audit records invalid or discordant "
                "pairs but does not fail the completed step before scientific "
                "outputs can be published."
            ),
            detail={"reason": "provenance_audit_not_fail_closed"},
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


_RECONCILIATION_FAILURE_EXCEPTIONS = frozenset(
    {"BaseException", "Exception", "TypeError", "ValueError"}
)
_RECONCILIATION_HELPER_NAME = "reconcile_binary_event_presence"


def _caught_exception_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Tuple):
        return {
            name for element in node.elts for name in _caught_exception_names(element)
        }
    name = _call_name(node).split(".")[-1]
    return {name} if name else set()


def _handler_catches_reconciliation_failure(handler: ast.ExceptHandler) -> bool:
    """Return whether *handler* can swallow the standard helper's failures."""

    if handler.type is None:
        return True
    return bool(
        _caught_exception_names(handler.type) & _RECONCILIATION_FAILURE_EXCEPTIONS
    )


def _handler_immediately_raises(handler: ast.ExceptHandler) -> bool:
    """Conservatively prove that no caught failure can continue execution."""

    return bool(handler.body) and isinstance(handler.body[0], ast.Raise)


def _reconciliation_helper_call_names(
    *,
    target: ast.Try,
    parents: dict[int, ast.AST],
) -> set[str]:
    """Return helper aliases visible from *target*'s lexical scopes.

    ``from ... import reconcile_binary_event_presence as reconcile`` retains
    the helper's semantics even though the call-site identifier changes.  The
    imported symbol, rather than a loose name substring, is the authority for
    adding an alias to the closed call-name set. Simple name assignments are
    also followed, but definitions nested below the target's scopes are not
    borrowed into the current execution path.
    """

    names = {_RECONCILIATION_HELPER_NAME}
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
            for imported in node.names:
                if imported.name == _RECONCILIATION_HELPER_NAME:
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
    """Reject caught sparse-event reconciliation failures that continue.

    ``reconcile_binary_event_presence`` is itself a deterministic validation
    boundary.  Once an Agent elects to call it, converting its exception into
    an ``unavailable`` audit and continuing can publish outputs after a known
    contradictory triad.  This is mechanical fail-close enforcement; it does
    not select the exposure or require the optional helper to be called.
    """

    findings: list[ValidationFinding] = []
    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        helper_call_names = _reconciliation_helper_call_names(
            target=node,
            parents=parents,
        )
        calls_reconciliation = _statements_call_reconciliation(
            node.body,
            helper_call_names=helper_call_names,
        )
        calls_before_finally = _statements_call_reconciliation(
            [
                *node.body,
                *node.orelse,
                *(statement for handler in node.handlers for statement in handler.body),
            ],
            helper_call_names=helper_call_names,
        )
        finally_suppressor = _finally_exception_suppressor(node.finalbody)
        if calls_before_finally and finally_suppressor is not None:
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "Errors from reconcile_binary_event_presence can be "
                        "suppressed by control flow in the finally suite, so "
                        "an invalid declared provenance triad could continue."
                    ),
                    detail={
                        "reason": "provenance_helper_error_swallowed",
                        "line": getattr(finally_suppressor, "lineno", node.lineno),
                    },
                )
            )
            continue
        if not calls_reconciliation:
            continue
        for handler in node.handlers:
            if not _handler_catches_reconciliation_failure(handler):
                continue
            if _handler_immediately_raises(handler):
                continue
            findings.append(
                ValidationFinding(
                    validator="mechanical_code_preflight",
                    severity="error",
                    message=(
                        "Errors from reconcile_binary_event_presence enter "
                        "control flow that can continue without propagating the "
                        "failure, so an invalid declared provenance triad could "
                        "be recorded as unavailable."
                    ),
                    detail={
                        "reason": "provenance_helper_error_swallowed",
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
        positional_only_names = {
            argument.arg for argument in arguments.posonlyargs
        }
        keyword_only_names = {
            argument.arg for argument in arguments.kwonlyargs
        }
        accepted_keywords = (
            set(positional_names) - positional_only_names
        ) | keyword_only_names
        explicit_keywords = [
            keyword.arg for keyword in call.keywords if keyword.arg is not None
        ]
        has_star_args = any(
            isinstance(argument, ast.Starred) for argument in call.args
        )
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
                reasons.append(
                    "unexpected_keyword_arguments=" + ",".join(unexpected)
                )

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
                for index, name in enumerate(positional_names[:required_positional_count])
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

    findings: list[ValidationFinding] = []

    def _direct_target_names(statement: ast.stmt) -> set[str]:
        targets: list[ast.AST] = []
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            targets = [statement.target]
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            return {
                alias.asname or alias.name.split(".")[0]
                for alias in statement.names
            }
        return {
            node.id
            for target in targets
            for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

    def _analyze_block(
        statements: list[ast.stmt], initially_assigned: set[str]
    ) -> None:
        assigned_before = set(initially_assigned)
        for index, statement in enumerate(statements):
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
                _analyze_block(statement.body, set(assigned_before))
                _analyze_block(statement.orelse, set(assigned_before))
                assigned_before.update(guaranteed)
                continue

            nested_blocks: list[list[ast.stmt]] = []
            if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                nested_blocks.extend([statement.body, statement.orelse])
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                nested_blocks.append(statement.body)
            elif isinstance(statement, ast.Try):
                nested_blocks.extend(
                    [statement.body, statement.orelse, statement.finalbody]
                )
                nested_blocks.extend(handler.body for handler in statement.handlers)
            for block in nested_blocks:
                _analyze_block(block, set(assigned_before))
            assigned_before.update(_direct_target_names(statement))

    _analyze_block(tree.body, set())
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
        _analyze_block(node.body, arguments)
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
            dtype = node.args[0] if node.args else next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "dtype"
                ),
                None,
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
    findings.extend(_first_time_companion_findings(tree))
    return findings


__all__ = ["audit_mechanical_code_contracts"]
