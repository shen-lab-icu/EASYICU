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


def _is_len_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
    )


def _mask_incomplete_test(test: ast.AST, mask_name: str) -> bool:
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
            _is_mask_method_call(left, mask_name, "sum") and _is_len_call(right)
        ) or (
            _is_mask_method_call(right, mask_name, "sum") and _is_len_call(left)
        )
    return False


def _mask_complete_test(test: ast.AST, mask_name: str) -> bool:
    if _is_mask_method_call(test, mask_name, "all"):
        return True
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    if not isinstance(test.ops[0], (ast.Eq, ast.Is)):
        return False
    return (
        _is_mask_method_call(left, mask_name, "sum") and _is_len_call(right)
    ) or (
        _is_mask_method_call(right, mask_name, "sum") and _is_len_call(left)
    )


def _is_raise_only_guard(statement: ast.stmt, mask_name: str) -> bool:
    if isinstance(statement, ast.Assert):
        return _mask_complete_test(statement.test, mask_name)
    if not isinstance(statement, ast.If) or not statement.body:
        return False
    if not all(isinstance(item, (ast.Raise, ast.Return)) for item in statement.body):
        return False
    return _mask_incomplete_test(statement.test, mask_name)


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


def _structural_filter_findings(tree: ast.Module, step: AnalysisStep) -> list[ValidationFinding]:
    if normalised_method_head(step.method) not in _RENDER_METHODS:
        return []
    accounting_products = _typed_input_products(step) & _STRUCTURAL_ACCOUNTING_PRODUCTS
    if not accounting_products:
        return []

    findings = []
    for owner in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        body = getattr(owner, "body", [])
        prior_guards: set[str] = set()
        mask_names: set[str] = set()
        mask_sources: dict[str, set[str]] = {}
        for statement in body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                if value is not None and _is_boolean_mask_expression(value):
                    for target in targets:
                        if not isinstance(target, ast.Name):
                            continue
                        mask_names.add(target.id)
                        mask_sources[target.id] = _referenced_names(value)
            for possible_mask in {
                node.id for node in ast.walk(statement) if isinstance(node, ast.Name)
            }:
                if _is_raise_only_guard(statement, possible_mask):
                    prior_guards.add(possible_mask)
            for node in ast.walk(statement):
                if not isinstance(node, ast.Subscript):
                    continue
                mask_name = _mask_name_from_slice(node.slice)
                value_name = _call_name(node.value)
                source_names = mask_sources.get(mask_name or "", set())
                filtered_names = _referenced_names(node.value)
                is_row_filter = bool(
                    mask_name
                    and source_names & filtered_names
                    and (
                        value_name.endswith(".loc")
                        or isinstance(node.value, ast.Name)
                    )
                )
                if not is_row_filter:
                    continue
                if (
                    not mask_name
                    or mask_name not in mask_names
                    or mask_name in prior_guards
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
            id(node.func)
            for node in ast.walk(test)
            if isinstance(node, ast.Call)
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
_PROVENANCE_DECISION_KEYS = frozenset(
    {"fail_closed", "completed_step_allowed", "provenance_valid"}
)


def _literal_string_tokens(node: ast.AST) -> set[str]:
    return {
        str(candidate.value).strip().lower()
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    }


def _referenced_names(node: ast.AST) -> set[str]:
    return {
        candidate.id
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Name)
    }


def _directly_raises(statements: list[ast.stmt]) -> bool:
    return any(isinstance(statement, ast.Raise) for statement in statements)


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

    derived_names: set[str] = set()
    result_names: set[str] = set()
    assignments: list[tuple[set[str], ast.AST]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = {
            target.id for target in targets if isinstance(target, ast.Name)
        }
        if not target_names:
            continue
        assignments.append((target_names, node.value))
        if isinstance(node.value, ast.Call) and _call_name(node.value.func) in marker_functions:
            result_names.update(target_names)

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
                    derived_names.add(candidate.func.value.id)

    changed = True
    while changed:
        changed = False
        for target_names, value in assignments:
            value_tokens = _literal_string_tokens(value)
            value_names = _referenced_names(value)
            if not (
                value_tokens & (_PROVENANCE_FAILURE_KEYS | _PROVENANCE_DECISION_KEYS)
                or value_names & derived_names
            ):
                continue
            new_names = target_names - derived_names
            if new_names:
                derived_names.update(new_names)
                changed = True

    guard_names = derived_names | result_names
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _directly_raises(node.body):
            continue
        test_tokens = _literal_string_tokens(node.test)
        test_names = _referenced_names(node.test)
        if test_names & guard_names and (
            test_names & derived_names
            or test_tokens & _PROVENANCE_DECISION_KEYS
        ):
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

        for node in nodes:
            if isinstance(node, ast.Call):
                call_inputs = [*node.args, *[keyword.value for keyword in node.keywords]]
                if any(
                    _referenced_names(value) & definition_names for value in call_inputs
                ):
                    return []
            if isinstance(node, ast.Subscript):
                if _referenced_names(node.value) & definition_names:
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
                        detail={
                            "reason": "authoritative_primary_exposure_fallback"
                        },
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
            candidate
            for statement in node.body
            for candidate in ast.walk(statement)
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


def _undefined_direct_call_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject direct calls whose Python name has no lexical binding or import."""

    known_names = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known_names.add(node.name)
        elif isinstance(node, ast.Import):
            known_names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
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
    findings.extend(_authoritative_exposure_binding_findings(tree, step))
    findings.extend(_authoritative_exposure_fallback_findings(tree, step))
    findings.extend(_finalized_exposure_reconciliation_findings(tree, step))
    findings.extend(_undefined_direct_call_findings(tree))
    return findings


__all__ = ["audit_mechanical_code_contracts"]
