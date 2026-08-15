"""Static name/binding analysis extracted from gates/preflight.py."""

from __future__ import annotations

import ast
import builtins

from ..schema import AnalysisStep, ValidationFinding
from .ast_semantics import (
    call_name as _call_name,
)

_TRY_STAR_NODE_TYPES = (
    (try_star_type,) if (try_star_type := getattr(ast, "TryStar", None)) else ()
)
_TRY_NODE_TYPES = (ast.Try, *_TRY_STAR_NODE_TYPES)
_TYPE_PARAMETER_NODE_TYPES = tuple(
    filter(
        None,
        (getattr(ast, name, None) for name in ("TypeVar", "ParamSpec", "TypeVarTuple")),
    )
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

from .preflight_provenance import _literal_string_tokens, _referenced_names

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
_MODULE_DUNDERS = frozenset(
    {
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__spec__",
        "__loader__",
        "__builtins__",
    }
)
def _names_bound_in_scope(scope: ast.AST) -> set[str]:
    """Every name this one scope binds, not descending into nested definitions."""

    bound: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = scope.args
        for group in (
            arguments.posonlyargs,
            arguments.args,
            arguments.kwonlyargs,
        ):
            bound.update(argument.arg for argument in group)
        for solo in (arguments.vararg, arguments.kwarg):
            if solo is not None:
                bound.add(solo.arg)

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(child.name)
                continue
            if isinstance(child, ast.Lambda):
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    bound.add(alias.asname or alias.name.split(".", 1)[0])
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    bound.add(alias.asname or alias.name)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                bound.add(child.id)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                bound.add(child.name)
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                bound.update(child.names)
            _walk(child)

    _walk(scope)
    return bound
def unresolvable_names(tree: ast.Module) -> list[tuple[str, int]]:
    """Names read where Python's scope rules cannot resolve them.

    ``compile()`` accepts every one of these -- a ``NameError`` is a runtime
    event -- so the syntax check is happy and the container is not.  A name is
    reported with the first line that reads it.

    A read resolves if the name is bound in its own scope, in an enclosing
    function scope, at module level, or is a builtin.  Nothing else counts, and
    that is the whole point: an earlier version of this check collected
    bindings from the *whole program*, so a name that was only ever a local of
    some other function looked bound.  canary4 died on exactly that --
    ``predicate_flow`` is a local of ``validate_receipt`` and ``main`` reads it
    as a global at line 133.  Both the module-only and whole-tree versions
    returned nothing for that script.

    Measured over the 409 recorded generated scripts: 4 flagged (1.0%), and
    every one is a real defect --

    * ``predicate_flow``  -- canary4, the death above
    * ``cohort_df``       -- an earlier run, same shape, also fatal
    * ``provenance_audit``-- unverifiable, that container never started
    * ``source_index``    -- a typo for ``source_row_index`` inside a ``raise``
      branch.  It has never executed, so no run has died of it; if the branch
      ever fires it replaces a written diagnostic with a ``NameError``, exactly
      when something has already gone wrong.

    A ``match`` statement binds names that are not ``Name`` stores, so a module
    containing one is abstained on rather than guessed at.  There is not one in
    the corpus, and a wrong flag costs a healthy step a repair -- the expensive
    direction to be wrong in.
    """

    if any(isinstance(node, ast.Match) for node in ast.walk(tree)):
        return []

    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    scopes: list[ast.AST] = [tree]
    scopes.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
    )
    bound_by_scope = {id(scope): _names_bound_in_scope(scope) for scope in scopes}
    # ``global x`` inside a function binds x at MODULE level, not in the
    # function that declares it -- so a sibling reading x afterwards is legal
    # Python.  Attributing it to the declaring scope reported that sibling,
    # which is the false positive this check must not produce.
    declared_global: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            declared_global.update(node.names)
    module_names = (
        bound_by_scope[id(tree)]
        | declared_global
        | set(dir(builtins))
        | set(_MODULE_DUNDERS)
    )

    loaded: dict[str, int] = {}
    for scope in scopes:
        visible = set(module_names) | bound_by_scope[id(scope)]
        enclosing = parents.get(id(scope))
        while enclosing is not None:
            if isinstance(
                enclosing, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                visible |= bound_by_scope[id(enclosing)]
            enclosing = parents.get(id(enclosing))

        def _reads(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Lambda,
                    ),
                ):
                    continue
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if child.id not in visible:
                        loaded.setdefault(child.id, int(child.lineno))
                _reads(child)

        _reads(scope)

    return sorted(loaded.items())
