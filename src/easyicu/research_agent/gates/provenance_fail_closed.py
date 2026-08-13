"""Fail-closed proof engine for generated measurement-provenance audits.

The caller owns generic preflight AST primitives and passes them through one
immutable service boundary.  This module owns only provenance proof ordering.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Optional

from ..schema import ValidationFinding
from .ast_semantics import (
    DYNAMIC_NAMESPACE_MUTATORS as _DYNAMIC_NAMESPACE_MUTATORS,
    DYNAMIC_NAMESPACE_PRIMITIVES as _DYNAMIC_NAMESPACE_PRIMITIVES,
    call_name as _call_name,
    contains_literal_provenance_audit_row,
    literal_observational_getattr,
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
_PROVENANCE_FAILURE_KEYS = frozenset({"invalid_pair_n", "discordant_n"})
_PROVENANCE_FULL_COVERAGE = frozenset(_PROVENANCE_FAILURE_KEYS)
_PROVENANCE_FAILURE = "failure"
_PROVENANCE_LOOP_SENTINEL = "_easyicu_provenance_loop_observed"
_FLOW_FALLTHROUGH = "fallthrough"


@dataclass(frozen=True)
class ProvenanceAnalysisServices:
    """Generic AST proofs supplied by the mechanical-preflight owner."""

    expression_identity: Callable[..., object]
    provenance_signal_source: Callable[..., object]
    subscript_key: Callable[..., object]
    target_names: Callable[..., object]
    has_dynamic_namespace_indirection: Callable[..., object]
    is_provenance_result_sink_call: Callable[..., object]
    mapping_root_name: Callable[..., object]
    referenced_names: Callable[..., object]
    block_flow_outcomes: Callable[..., object]
    branch_all_paths_exit: Callable[..., object]
    builtin_int_binding_is_unmodified: Callable[..., object]
    handler_immediately_reraises: Callable[..., object]
    literal_zero: Callable[..., object]
    provenance_branch_contains_result_sink: Callable[..., object]
    provenance_predicate_meaning: Callable[..., object]
    result_sink_precedes_guard: Callable[..., object]
    stable_raise_only_helper_call: Callable[..., object]
    ast_parent_and_statement_positions: Callable[..., object]
    literal_string_tokens: Callable[..., object]


def _build_provenance_scope_helpers(
    *,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    services: ProvenanceAnalysisServices,
) -> tuple[Callable[..., object], ...]:
    """Build scope-local provenance AST readers around one immutable tree."""

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
                    key = services.subscript_key(key_node)
                    if key not in _PROVENANCE_FAILURE_KEYS or isinstance(
                        value_node, ast.Constant
                    ):
                        continue
                    identity = services.expression_identity(value_node)
                    expression_roles[identity] = frozenset(
                        set(expression_roles.get(identity, frozenset())) | {key}
                    )
                    source = services.provenance_signal_source(value_node)
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
                        name for target in targets for name in services.target_names(target)
                    )
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    for name in services.target_names(target):
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
            services.subscript_key(key): value
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

    return (
        _nearest_function, _contains_literal_audit_row,
        _uses_host_provenance_receipt, _scope, _local_nodes, _environment,
        _next_statement, _preceding_direct_assignments, _direct_audit_row,
        _immediate_returned_audit_row, _post_audit_alias_path_is_pure,
    )

def _build_provenance_execution_helpers(
    *,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    all_functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
    marker_names: set[str],
    _nearest_function: Callable[[ast.AST], Optional[ast.AST]],
    _scope: Callable[[ast.AST], ast.AST],
    _local_nodes: Callable[[ast.AST], list[ast.AST]],
    _next_statement: Callable[[ast.stmt], Optional[ast.stmt]],
    _direct_audit_row: Callable[..., object],
    _post_audit_alias_path_is_pure: Callable[..., bool],
    services: ProvenanceAnalysisServices,
) -> tuple[Callable[..., object], ...]:
    """Build runtime-binding proofs for provenance audit helpers."""

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
                and (services.referenced_names(value) & set(helper_aliases))
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
                    services.is_provenance_result_sink_call(candidate)
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
                    services.referenced_names(argument) & set(helper_aliases)
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
        if services.has_dynamic_namespace_indirection(tree):
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
                    or services.mapping_root_name(candidate) == function.name
                )
            ):
                return False
            if (
                isinstance(candidate, ast.Subscript)
                and isinstance(candidate.ctx, (ast.Store, ast.Del))
                and (
                    services.subscript_key(candidate.slice) == function.name
                    or services.mapping_root_name(candidate) == function.name
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
                    services.subscript_key(candidate.args[1]) is None
                    or services.subscript_key(candidate.args[1]) == function.name
                    or services.mapping_root_name(candidate.args[0]) == function.name
                )
            ):
                return False
        return True
    def _dynamic_namespace_execution_present() -> bool:
        if services.has_dynamic_namespace_indirection(tree):
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
            and services.is_provenance_result_sink_call(candidate)
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

    return (
        _module_pre_guard_has_indirect_effects, _module_direct_guard_is_bound,
        _canonical_entrypoint_guard, _direct_function_runtime_binding,
        _dynamic_namespace_execution_present, _terminal_entry_function,
        _direct_execution_statement, _result_sink_precedes_call,
        _eager_outer_call_statement,
    )

def _build_provenance_guard_helpers(
    *,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    id_parents: dict[int, ast.AST],
    statement_positions: dict[int, object],
    swallowed_exit_issues: dict[tuple[int, int, int], dict[str, object]],
    environments: dict[ast.AST, object],
    _scope: Callable[[ast.AST], ast.AST],
    _local_nodes: Callable[[ast.AST], list[ast.AST]],
    _environment: Callable[..., object],
    _next_statement: Callable[[ast.stmt], Optional[ast.stmt]],
    _preceding_direct_assignments: Callable[[ast.stmt], dict[str, ast.AST]],
    _direct_audit_row: Callable[..., object],
    _immediate_returned_audit_row: Callable[..., object],
    _post_audit_alias_path_is_pure: Callable[..., bool],
    services: ProvenanceAnalysisServices,
) -> tuple[Callable[..., object], ...]:
    """Build terminating-guard proofs and attributable diagnostics."""

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
        if services.provenance_branch_contains_result_sink(loop.body):
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
            and not services.provenance_branch_contains_result_sink(empty_guard.body)
            and not services.result_sink_precedes_guard(empty_guard, parents)
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
                and not services.provenance_branch_contains_result_sink(preceding.body)
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
        direct_raise = services.branch_all_paths_exit(statements) and not any(
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
            and services.block_flow_outcomes(statements[:-1]) == {_FLOW_FALLTHROUGH}
            and services.stable_raise_only_helper_call(
                terminal,
                position=terminal_position,
                tree=tree,
                parents=id_parents,
                positions=statement_positions,
            )
        )
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
                        services.handler_immediately_reraises(handler)
                        for handler in parent.handlers
                    )
                ):
                    current = parent
                    continue
                unsafe_handlers = [
                    handler
                    for handler in parent.handlers
                    if not services.handler_immediately_reraises(handler)
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

        if not services.builtin_int_binding_is_unmodified(tree):
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
                        and services.mapping_root_name(candidate) in bound_names
                    ):
                        return False
                    if (
                        isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Attribute)
                        and services.mapping_root_name(candidate.func.value)
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
        meaning = services.provenance_predicate_meaning(
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
            services.builtin_int_binding_is_unmodified(tree)
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
            source = services.provenance_signal_source(fields[key])
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
                if any(signal_name in services.target_names(target) for target in targets):
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
                and services.literal_zero(right)
                and isinstance(operator, (ast.NotEq, ast.IsNot, ast.Gt, ast.GtE))
            ):
                return set(signal_coverage[left.id])
            if (
                isinstance(right, ast.Name)
                and right.id in signal_coverage
                and services.literal_zero(left)
                and isinstance(operator, (ast.NotEq, ast.IsNot, ast.Lt, ast.LtE))
            ):
                return set(signal_coverage[right.id])
            return set()

        guards: set[ast.If] = set()
        coverage: set[str] = set()
        for guard in scope.body:
            if not isinstance(guard, ast.If):
                continue
            meaning = services.provenance_predicate_meaning(
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
                and not services.provenance_branch_contains_result_sink(guard.body)
                and not services.result_sink_precedes_guard(guard, parents)
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

    return (
        _loop_eager_argument_is_fail_closed, _exact_collection_test,
        _branch_all_paths_raise, _failure_exit_may_be_swallowed,
        _direct_scope_statement, _stable_local_failure_signals,
        _full_failure_test, _separate_direct_failure_guards,
    )

def provenance_fail_closed_findings(
    tree: ast.Module, *, services: ProvenanceAnalysisServices
) -> list[ValidationFinding]:
    """Require a terminating guard for an implemented provenance failure audit."""

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    id_parents, statement_positions = services.ast_parent_and_statement_positions(tree)
    (
        _nearest_function, _contains_literal_audit_row,
        _uses_host_provenance_receipt, _scope, _local_nodes, _environment,
        _next_statement, _preceding_direct_assignments, _direct_audit_row,
        _immediate_returned_audit_row, _post_audit_alias_path_is_pure,
    ) = _build_provenance_scope_helpers(tree=tree, parents=parents, services=services)




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
        if isinstance(node, _TYPE_PARAMETER_NODE_TYPES) and (node.name in marker_names):
            ambiguous_names.add(node.name)
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        bound_names = {name for target in targets for name in services.target_names(target)}
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

    (
        _module_pre_guard_has_indirect_effects, _module_direct_guard_is_bound,
        _canonical_entrypoint_guard, _direct_function_runtime_binding,
        _dynamic_namespace_execution_present, _terminal_entry_function,
        _direct_execution_statement, _result_sink_precedes_call,
        _eager_outer_call_statement,
    ) = _build_provenance_execution_helpers(
        tree=tree, parents=parents, all_functions=all_functions,
        marker_names=marker_names, _nearest_function=_nearest_function,
        _scope=_scope, _local_nodes=_local_nodes, _next_statement=_next_statement,
        _direct_audit_row=_direct_audit_row,
        _post_audit_alias_path_is_pure=_post_audit_alias_path_is_pure,
        services=services,
    )





















    swallowed_exit_issues: dict[tuple[int, int, int], dict[str, object]] = {}


    environments = {
        scope: _environment(scope) for scope in [tree, *marker_functions.values()]
    }

    (
        _loop_eager_argument_is_fail_closed, _exact_collection_test,
        _branch_all_paths_raise, _failure_exit_may_be_swallowed,
        _direct_scope_statement, _stable_local_failure_signals,
        _full_failure_test, _separate_direct_failure_guards,
    ) = _build_provenance_guard_helpers(
        tree=tree, parents=parents, id_parents=id_parents,
        statement_positions=statement_positions,
        swallowed_exit_issues=swallowed_exit_issues, environments=environments,
        _scope=_scope, _local_nodes=_local_nodes, _environment=_environment,
        _next_statement=_next_statement,
        _preceding_direct_assignments=_preceding_direct_assignments,
        _direct_audit_row=_direct_audit_row,
        _immediate_returned_audit_row=_immediate_returned_audit_row,
        _post_audit_alias_path_is_pure=_post_audit_alias_path_is_pure,
        services=services,
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
                services.subscript_key(key): value
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
                and not services.provenance_branch_contains_result_sink(empty_guard.body)
                and not services.result_sink_precedes_guard(empty_guard, parents)
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
            if services.result_sink_precedes_guard(full_guard, parents):
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
                    services.branch_all_paths_exit(guard.body)
                    or _branch_all_paths_raise(guard.body)
                )
                and not services.provenance_branch_contains_result_sink(guard.body)
                and not services.result_sink_precedes_guard(guard, parents)
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
                    return collection in services.referenced_names(target)
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
                if value is not None and collection in services.referenced_names(value):
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
                    collection in services.referenced_names(argument)
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
                and not services.provenance_branch_contains_result_sink(candidate.body)
                and not services.result_sink_precedes_guard(candidate, parents)
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
                    if value is not None and collection in services.referenced_names(value):
                        terminal_invalid_mutation = True
                    if not (
                        isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Attribute)
                        and isinstance(candidate.func.value, ast.Name)
                        and candidate.func.value.id == collection
                    ):
                        if isinstance(candidate, ast.Call) and any(
                            collection in services.referenced_names(argument)
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
            and not services.provenance_branch_contains_result_sink(branch.body)
            and not services.result_sink_precedes_guard(branch, parents)
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
            and "checks" in services.literal_string_tokens(statement)
            and check_name in services.referenced_names(statement)
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
                or services.provenance_branch_contains_result_sink(following.body)
                or services.result_sink_precedes_guard(following, parents)
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


__all__ = [
    "ProvenanceAnalysisServices",
    "provenance_fail_closed_findings",
]
