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
    if not marker_nodes:
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
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        bound_names = {
            name for target in targets for name in _target_names(target)
        }
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

    environments = {
        scope: _environment(scope) for scope in [tree, *marker_functions.values()]
    }

    def _full_failure_test(node: ast.AST, scope: ast.AST) -> bool:
        roles, signals, assignments, containers = environments.setdefault(
            scope, _environment(scope)
        )
        return _provenance_predicate_meaning(
            node,
            expression_roles=roles,
            signal_meanings=signals,
            assignments=assignments,
            audit_containers=containers,
        ) == (_PROVENANCE_FAILURE, _PROVENANCE_FULL_COVERAGE)

    returned_slots: dict[str, Optional[int]] = {}
    self_guarded: set[str] = set()
    self_raising: set[str] = set()
    for name, function in marker_functions.items():
        local_nodes = _local_nodes(function)
        collection_events: dict[str, set[ast.Call]] = {}
        for guard in local_nodes:
            if not isinstance(guard, ast.If) or not _full_failure_test(
                guard.test, function
            ):
                continue
            for statement in guard.body:
                if isinstance(statement, (ast.Raise, ast.Return)):
                    break
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr in {"append", "add"}
                    and isinstance(statement.value.func.value, ast.Name)
                ):
                    continue
                collection_events.setdefault(statement.value.func.value.id, set()).add(
                    statement.value
                )
            if (
                _branch_all_paths_exit(guard.body)
                and not _has_unrelated_control_ancestor(guard, parents)
                and not _result_sink_precedes_guard(guard, parents)
            ):
                self_guarded.add(name)
                if guard.body and isinstance(guard.body[0], ast.Raise):
                    self_raising.add(name)

        def _empty_initialization(node: ast.AST) -> bool:
            return (isinstance(node, (ast.List, ast.Set)) and not node.elts) or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"list", "set"}
                and not node.args
                and not node.keywords
            )

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
            boundary_lines = [
                int(call.lineno) for call in allowed_calls
            ] + [int(statement.lineno) for statement in returns]
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
                        and parents.get(candidate) is function
                        and boundary_lines
                        and int(candidate.lineno) < min(boundary_lines)
                    ):
                        initializations += 1
                    else:
                        invalid_mutation = True
                if (
                    value is not None
                    and collection in _referenced_names(value)
                ):
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
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        called = call.func.id
        if called not in marker_functions:
            continue
        called_functions.add(called)
        if called in self_raising:
            continue
        node = parents.get(call)
        if not (isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is call):
            unsafe_call = True
            continue
        following = _next_statement(node)
        if not isinstance(following, ast.If):
            unsafe_call = True
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
            guarded = _branch_all_paths_raise(
                following.body
            ) and _full_failure_test(following.test, _scope(node))
        unsafe_call = unsafe_call or not guarded

    unsafe_definition = bool(ambiguous_names) or any(
        name not in called_functions for name in marker_functions
    )
    if not unsafe_call and not unsafe_definition:
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


def _handler_immediately_raises(handler: ast.ExceptHandler) -> bool:
    """Conservatively prove that no caught failure can continue execution."""

    return bool(handler.body) and isinstance(handler.body[0], ast.Raise)


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
            if _handler_immediately_raises(handler):
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
