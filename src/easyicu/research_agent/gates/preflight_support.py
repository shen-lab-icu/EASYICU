"""Host-owned preflight leaf helpers extracted from preflight.py.

These checks are deterministic and import no orchestrator authority.
They remain importable from :mod:`preflight` through re-exports.
"""

from __future__ import annotations

from __future__ import annotations
import ast
import builtins
from typing import Optional
from ..icu_rules import companion_count_column_for_measured
from ..schema import AnalysisStep, ValidationFinding
from .ast_semantics import (
    call_name as _call_name,
    pre312_fstring_subscript_quote_occurrences,
)
from .preflight_provenance import (  # noqa: F401 — owner module
    _FLOW_FALLTHROUGH,
    _FLOW_FUNCTION_EXIT,
    _FLOW_LOOP_ESCAPE,
    _PROVENANCE_FAILURE,
    _PROVENANCE_FAILURE_DECISION_KEYS,
    _PROVENANCE_FAILURE_KEYS,
    _PROVENANCE_FULL_COVERAGE,
    _PROVENANCE_LOOP_SENTINEL,
    _PROVENANCE_RESULT_SINK_METHODS,
    _PROVENANCE_SUCCESS,
    _PROVENANCE_SUCCESS_DECISION_KEYS,
    _REFLECTION_MODULE_ROOTS,
    _expression_identity,
    _finally_exception_suppressor,
    _handler_immediately_reraises,
    _has_unrelated_control_ancestor,
    _is_provenance_result_sink_call,
    _literal_bool,
    _literal_string_tokens,
    _literal_zero,
    _mapping_root_name,
    _provenance_branch_contains_result_sink,
    _provenance_pair_scan_findings,
    _provenance_signal_source,
    _referenced_names,
    _result_sink_precedes_guard,
    _statements_call_reconciliation,
    _swap_provenance_meaning,
)
from .preflight_statics import (  # noqa: F401 — owner module
    _EXPOSURE_RESULT_CALLS,
    _MODULE_DUNDERS,
    _assignment_target_names,
    _authoritative_exposure_binding_findings,
    _authoritative_exposure_fallback_findings,
    _contains_bound_exposure_selection,
    _finalized_exposure_reconciliation_findings,
    _names_bound_in_scope,
    _scope_nodes,
    unresolvable_names,
)

def _is_frame_columns(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "columns"


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


def _subscript_key(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return {name for item in node.elts for name in _target_names(item)}
    return set()


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
            "outer_quote": ("double" if occurrence["outer_quote"] == '"' else "single"),
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


def _caught_exception_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Tuple):
        return {
            name for element in node.elts for name in _caught_exception_names(element)
        }
    name = _call_name(node).split(".")[-1]
    return {name} if name else set()


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


def module_level_unbound_names(tree: ast.Module) -> list[tuple[str, int]]:
    """The module-scope answer, kept because the host's own fragments use it.

    A rendered fragment is not a whole module, so asking about function scopes
    it does not contain would be meaningless.  Delegates rather than repeating
    the walk.
    """

    module_scope = ast.Module(
        body=[
            statement
            for statement in tree.body
            if not isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        ],
        type_ignores=[],
    )
    defined = _names_bound_in_scope(tree)
    return [
        (name, line)
        for name, line in unresolvable_names(module_scope)
        if name not in defined
    ]


def _unresolvable_name_findings(tree: ast.Module) -> list[ValidationFinding]:
    """Reject a read Python's scope rules cannot resolve.

    fresh22 died on ``hashlib`` at module level; the H1 canary died twice in one
    step on ``manifest`` then ``table_one_spec``; canary4 died on
    ``predicate_flow``, a local of one function read as a global by another.
    Each cost an execution slot, and the last one cost the three steps behind
    it.  All six instances in the recorded corpus are real defects.

    ``_undefined_direct_call_findings`` overlaps on one shape -- a bare call to
    a name nothing defines -- and keeps its own reasoning.  A name both reject
    is reported twice, and both reports are true; neither is allowed to assume
    the other ran.
    """

    unbound = unresolvable_names(tree)
    if not unbound:
        return []
    return [
        ValidationFinding(
            validator="mechanical_code_preflight",
            severity="error",
            message=(
                "The script reads names Python cannot resolve where they are "
                "used, so it raises NameError at run time: "
                + ", ".join(f"{name} (line {line})" for name, line in unbound)
                + ". A name assigned inside another function is not visible "
                "here. Bind each one in the scope that reads it -- import it, "
                "pass it as an argument, or return it from the function that "
                "computes it -- instead of assuming it is already in scope."
            ),
            detail={
                "reason": "unresolvable_name",
                "names": [{"name": name, "line": line} for name, line in unbound],
            },
        )
    ]


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


def _is_numeric_zero(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.Constant)
        and isinstance(value.value, (int, float))
        and not isinstance(value.value, bool)
        and float(value.value) == 0.0
    )


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


def _literal_int(node: ast.AST, value: int) -> bool:
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and node.value == value
    )


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


def _measurement_provenance_scope_findings(
    tree: ast.Module,
    step: AnalysisStep,
) -> list[ValidationFinding]:
    """Reject provenance calls that widen the Planner's raw-input scope.

    ResearchContext may describe related provenance coordinates for semantic
    interpretation, while executable measured/count pairs remain owned by the
    step's exact inputs.  A host cohort-execution receipt can separately bind
    predicate columns, but it never grants sibling-column access.
    """

    declared_inputs = {
        str(value).strip()
        for value in step.inputs
        if ":" not in str(value) and str(value).strip()
    }
    declared_pairs = {
        (measured_column, count_column)
        for measured_column in declared_inputs
        if (count_column := companion_count_column_for_measured(measured_column))
        and count_column in declared_inputs
    }
    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and _call_name(node.func).rsplit(".", 1)[-1]
            == "measurement_provenance_receipt"
        ):
            continue
        keyword_values = {
            str(keyword.arg): keyword.value
            for keyword in node.keywords
            if keyword.arg is not None
        }
        measured_node = keyword_values.get("measured_column")
        count_node = keyword_values.get("count_column")
        observed_pair = (
            (
                str(measured_node.value),
                str(count_node.value),
            )
            if isinstance(measured_node, ast.Constant)
            and isinstance(measured_node.value, str)
            and isinstance(count_node, ast.Constant)
            and isinstance(count_node.value, str)
            else None
        )
        if declared_pairs and (
            observed_pair is None or observed_pair in declared_pairs
        ):
            continue
        findings.append(
            ValidationFinding(
                validator="mechanical_code_preflight",
                severity="error",
                message=(
                    "Generated code invokes measurement provenance outside an "
                    "exact Planner-declared measured/count input pair."
                ),
                detail={
                    "reason": "measurement_provenance_pair_undeclared",
                    "helper_name": "measurement_provenance_receipt",
                    "line": int(node.lineno),
                    "declared_pairs": [list(pair) for pair in sorted(declared_pairs)],
                    **(
                        {"observed_pair": list(observed_pair)}
                        if observed_pair is not None
                        else {}
                    ),
                },
            )
        )
    return findings

