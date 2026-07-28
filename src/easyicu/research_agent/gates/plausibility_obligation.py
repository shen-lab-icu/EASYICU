"""Host-owned obligation gate for the flag-only plausibility-range policy.

``plausibility_policy.out_of_range_action = "retain_and_flag"`` is two
obligations, and the earlier gate could only check them while an LLM auditor
was still complaining.  That made the check strictly weaker than the policy:
the deterministic repair in ``repairs/plausibility.py`` neuters the rejecting
guard, so the script stops requesting exclusion, the auditor goes quiet, and a
script that kept every row and recorded *nothing* passed with no finding to
attach to.

This module owes nothing to a finding.  It is triggered by the host's own
typed policy -- the ResearchContext declares a plausibility range, and the
generated script exercises it -- or by the repair's own receipt, and it asks
three questions of the script itself:

1. **Retention.**  No plausibility comparison may gate a terminal failure or
   filter rows out.
2. **Flagging into the artifact the host reads.**  A structured count or
   per-row indicator must reach the ``plausibility_audit`` key of the
   ``step_summary.json`` written into the output directory the host handed the
   step.  Two earlier drafts each stopped one step short of that.  The first
   asked only whether some serializer had been called, which proved that the
   script had touched something that writes and nothing more:
   ``json.dump(audit, sys.stdout)``, a scratch file under ``/tmp``, and
   ``DataFrame(...).to_json()`` with no destination at all each satisfied it.
   The second did decide on the destination but compared only the last path
   component, so ``/tmp/step_summary.json`` still passed for the canonical
   artifact and the counts the host reads could be anything.  A destination is
   a directory, a filename **and** a key.
3. **An unconditional receipt.**  The count must be computed and delivered on
   every path, not only when it is positive.  "No out-of-range rows" and "we
   never looked" are different facts, and a receipt that only appears when the
   count is nonzero cannot tell them apart.

Being shaped to record a count is still not the same as having recorded one --
a script can be shaped correctly and never run, or run and write an empty
summary.  ``plausibility_receipt`` closes that half by reading the sealed
artifact after execution; this module and that one share the same trigger and
the same published contract.

A script the gate cannot attribute is **blocked**, not passed.  Silence from a
structural check means nobody looked, which is exactly the reading this module
exists to refuse -- so ``not_attributable`` costs a provider repair rather than
buying a free pass.

Case neutrality: every variable name comes from the exact sealed step scope.
Nothing here knows which study, benchmark, column or bound is in play.
"""

from __future__ import annotations

import ast
import re
from typing import Iterator, Mapping, Optional, Sequence, Set

from ..authority.plausibility import FlagOnlyPlausibilityScope
from ..schema import AnalysisStep, ValidationFinding
from .plausibility_receipt import (
    CANONICAL_STEP_SUMMARY_FILENAME,
    HOST_OUTPUT_DIR_ENV_KEYS,
    POLICY_CONTRACT_KEY,
    RECEIPT_CONTRACT_SENTENCE,
    RECEIPT_POLICY_VALUE,
    RECEIPT_SUMMARY_KEY,
    REPAIR_RECEIPT_MARKER,
    step_is_under_the_flag_only_obligation,
)

#: The bound keys inside that contract, in both the sealed mapping spelling
#: and the two-item sequence a script sometimes still guesses.
_BOUND_KEYS: frozenset[object] = frozenset({"minimum", "maximum", 0, 1})

_VALIDATOR = "mechanical_code_preflight"

#: Reductions that turn the out-of-range mask into a number worth keeping.
_RECORDING_REDUCTIONS = frozenset(
    {"sum", "count", "mean", "size", "nunique", "value_counts"}
)

#: Library calls that put bytes somewhere.  This set is a *necessary* condition
#: only -- it says a call could write, never that the write is declared -- and
#: it is deliberately not growing.  An earlier draft used it as the whole test,
#: which proved only that a script had touched something that serializes:
#: ``json.dump(audit, sys.stdout)``, a scratch file under ``/tmp``, and
#: ``DataFrame(...).to_json()`` with no destination at all all satisfied it.
#: The destination is what the gate actually decides on now; keeping this set
#: alongside it is what stops a log line that merely *names* the summary file
#: from counting as a write.  ``write_json`` was removed from it: locally
#: defined helpers are resolved structurally below, by what their body does
#: rather than by what they were named.
_WRITER_ATTRIBUTES = frozenset(
    {
        "dump",
        "savez",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_html",
        "to_json",
        "to_markdown",
        "to_parquet",
        "to_pickle",
        "write",
        "write_text",
        "writerow",
        "writerows",
    }
)

#: Names whose call terminates the step.
_TERMINATING_CALLS = frozenset({"exit", "_exit", "sys_exit"})

#: Wrappers that do not change which file a path expression names.
_PATH_CONSTRUCTORS = frozenset({"Path", "PosixPath", "PurePath", "PurePosixPath"})
_PATH_IDENTITY = frozenset({"absolute", "expanduser", "resolve"})
_CALLBACK_DISPATCHERS = frozenset(
    {"agg", "aggregate", "apply", "applymap", "filter", "map", "transform"}
)


def _active_nodes(
    tree: ast.AST,
    active_node_ids: Optional[Set[int]],
) -> Iterator[ast.AST]:
    """Walk only code that can run from the module entry point."""

    return (
        node
        for node in ast.walk(tree)
        if active_node_ids is None or id(node) in active_node_ids
    )


class _RuntimeReachability:
    """Conservative module-entry reachability for locally defined helpers.

    A function body is not evidence merely because it exists in the file.
    Earlier versions walked the whole AST, so an uncalled helper containing a
    perfect comparison-to-receipt chain could certify literal zero receipts
    written by unrelated module code.  This index enters a helper only after a
    reachable direct call (or a small, explicit callback dispatcher) names its
    unique definition.

    Dynamic dispatch is intentionally not guessed.  If the generated script
    hides the obligation behind an unresolvable callback, the gate asks for a
    repair instead of treating dead-looking code as executed evidence.
    """

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        definitions: dict[str, list[ast.AST]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.setdefault(node.name, []).append(node)
        self.definitions = {
            name: nodes[0] for name, nodes in definitions.items() if len(nodes) == 1
        }
        self.active_node_ids: Set[int] = {id(tree)}
        self.reachable_function_ids: Set[int] = set()
        self._activate_statements(tree.body)
        self._close_calls()

    def _activate_statements(self, statements: Sequence[ast.stmt]) -> None:
        pending: list[ast.AST] = list(statements)
        while pending:
            node = pending.pop()
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                # Creating the object is active; its body is not.
                self.active_node_ids.add(id(node))
                continue
            if id(node) in self.active_node_ids:
                continue
            self.active_node_ids.add(id(node))
            pending.extend(ast.iter_child_nodes(node))

    def _activate_function(self, function: ast.AST) -> bool:
        if id(function) in self.reachable_function_ids:
            return False
        self.reachable_function_ids.add(id(function))
        self.active_node_ids.add(id(function))
        self._activate_statements(function.body)
        return True

    def _close_calls(self) -> None:
        while True:
            grew = False
            for node in ast.walk(self.tree):
                if id(node) not in self.active_node_ids or not isinstance(node, ast.Call):
                    continue
                callee = node.func
                if isinstance(callee, ast.Name):
                    function = self.definitions.get(callee.id)
                    if function is not None:
                        grew = self._activate_function(function) or grew
                dispatcher = (
                    callee.attr
                    if isinstance(callee, ast.Attribute)
                    else callee.id if isinstance(callee, ast.Name) else ""
                )
                if dispatcher not in _CALLBACK_DISPATCHERS:
                    continue
                for argument in _call_arguments(node):
                    if isinstance(argument, ast.Name):
                        function = self.definitions.get(argument.id)
                        if function is not None:
                            grew = self._activate_function(function) or grew
            if not grew:
                break


def _strip_path_wrappers(node: ast.AST) -> ast.AST:
    """Drop ``Path(...)``, ``str(...)`` and the no-op path methods."""

    while isinstance(node, ast.Call):
        callee = node.func
        if (
            isinstance(callee, ast.Name)
            and callee.id in (_PATH_CONSTRUCTORS | {"str"})
            and len(node.args) == 1
        ):
            node = node.args[0]
            continue
        if (
            isinstance(callee, ast.Attribute)
            and callee.attr in _PATH_IDENTITY
            and not node.args
        ):
            node = callee.value
            continue
        break
    return node


def _is_os_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _reads_the_host_output_directory(node: ast.AST) -> bool:
    """Whether an expression is the output directory the host handed the step.

    The host passes it in the environment, under every alias generated code has
    been observed to invent.  That makes it the one directory a static check can
    recognise without guessing -- and recognising the directory, not just the
    filename, is the whole point: ``/tmp/step_summary.json`` ends in the
    canonical name and is not the canonical artifact.
    """

    node = _strip_path_wrappers(node)
    if isinstance(node, ast.Subscript):
        return (
            _is_os_environ(node.value)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in HOST_OUTPUT_DIR_ENV_KEYS
        )
    if not isinstance(node, ast.Call) or not node.args:
        return False
    key = node.args[0]
    if not isinstance(key, ast.Constant) or key.value not in HOST_OUTPUT_DIR_ENV_KEYS:
        return False
    callee = node.func
    if isinstance(callee, ast.Attribute):
        return callee.attr == "getenv" or (
            callee.attr == "get" and _is_os_environ(callee.value)
        )
    return isinstance(callee, ast.Name) and callee.id == "getenv"


def _single_name_assignments(tree: ast.AST) -> list[ast.Assign]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]


def _function_definitions(tree: ast.AST) -> list[ast.AST]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _parameter_names(function: ast.AST) -> Set[str]:
    arguments = function.args
    return {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }


#: The node kinds that open a new lexical scope.  Comprehensions are included
#: so their targets bind inside them rather than leaking a name outwards.
_SCOPE_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _stored_names(node: ast.AST) -> Set[str]:
    """The names an assignment target rebinds.

    A subscript target does not rebind its root -- ``summary["k"] = v`` leaves
    ``summary`` bound to the same object -- and the ``Load`` context of the
    root is what says so.
    """

    return {
        inner.id
        for inner in ast.walk(node)
        if isinstance(inner, ast.Name) and isinstance(inner.ctx, (ast.Store, ast.Del))
    }


def _declared_elsewhere(tree: ast.AST) -> Set[str]:
    """Names some scope declares ``global`` or ``nonlocal``.

    Such a name is written in one scope and read in another, so no per-scope
    binding list is the whole story for it.  Rather than model that, the
    destination checks refuse the name everywhere: generated analysis scripts
    do not route an output path through a module-level rebind, and one that
    did would cost a repair instead of buying a pass.
    """

    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


class _Scopes:
    """Where each name is written, and which of those writes a use can see.

    Every destination question in this module is a name-level one -- is *this*
    name the directory the host handed the step, is *that* one the canonical
    summary -- and the first draft answered it from a single flat set per
    script.  A flat set cannot say that a name means one thing here and
    another there, so one trusted binding anywhere marked the name trusted
    everywhere, in both directions::

        out_dir = STEP_OUT_DIR
        out_dir = Path("/tmp")
        write_json(out_dir / "step_summary.json", real_audit)

    and, across two functions, an ``out_dir`` parameter that every caller
    feeds the host directory lending its name to an unrelated local in the
    next function down.  Either way the real counts went to a scratch file
    while the artifact the host opens carried whatever the script chose to put
    there.

    Bindings are therefore collected per lexical scope, and a name is trusted
    only when *every* binding of it in its own scope is trusted.  Mixed is not
    trusted: a name that can be two different files cannot be proven to be the
    right one, and refusing it costs a provider repair rather than buying a
    pass.  Ways of binding a name that carry no readable expression -- a loop
    target, an ``except`` alias, an import, tuple unpacking, ``*args`` -- are
    ``opaque`` and are never trusted, for the same reason.

    One binding form is an exception to the must-rule, and a real generated
    script proved it: ``with open(...) as handle`` scopes ``handle`` to that
    block.  Scripts routinely reuse the name -- one ``with`` reads a host input,
    a later one writes the summary -- and the two never coexist, so requiring
    every ``handle`` binding to be the summary made a compliant write
    invisible.  A use of such a name is therefore resolved to the innermost
    ``with`` that binds it (:meth:`governing_context`), which is exactly the
    extent the language gives it.
    """

    def __init__(self, tree: ast.Module) -> None:
        self.module = tree
        self.scope_of: dict[int, ast.AST] = {id(tree): tree}
        self.parent: dict[int, Optional[ast.AST]] = {}
        self.bindings: dict[int, dict[str, list[tuple]]] = {}
        self.parent_of: dict[int, ast.AST] = {}
        self._build(tree, None)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self.parent_of[id(child)] = parent
        for name in _declared_elsewhere(tree):
            for scope_id, names in self.bindings.items():
                names.setdefault(name, []).append(("opaque", tree))
            self.bindings[id(tree)].setdefault(name, []).append(("opaque", tree))

    def _bind(self, scope: ast.AST, name: str, binding: tuple) -> None:
        self.bindings.setdefault(id(scope), {}).setdefault(name, []).append(binding)

    def _build(self, scope: ast.AST, parent: Optional[ast.AST]) -> None:
        self.parent[id(scope)] = parent
        self.bindings.setdefault(id(scope), {})
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = scope.args
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ):
                self._bind(scope, argument.arg, ("param", scope))
            for collected in (arguments.vararg, arguments.kwarg):
                if collected is not None:
                    self._bind(scope, collected.arg, ("opaque", scope))
        pending = list(ast.iter_child_nodes(scope))
        while pending:
            node = pending.pop()
            self.scope_of[id(node)] = scope
            if isinstance(node, _SCOPE_NODES):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    self._bind(scope, node.name, ("opaque", node))
                self._build(node, scope)
                continue
            self._record(scope, node)
            pending.extend(ast.iter_child_nodes(node))

    def _record(self, scope: ast.AST, node: ast.AST) -> None:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self._bind(scope, node.targets[0].id, ("value", node.value))
                return
            for target in node.targets:
                for name in _stored_names(target):
                    self._bind(scope, name, ("opaque", node))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                self._bind(scope, node.target.id, ("value", node.value))
            else:
                for name in _stored_names(node.target):
                    self._bind(scope, name, ("opaque", node))
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            self._bind(scope, node.target.id, ("value", node.value))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is None:
                    continue
                if isinstance(item.optional_vars, ast.Name):
                    self._bind(
                        scope, item.optional_vars.id, ("open", item.context_expr)
                    )
                else:
                    for name in _stored_names(item.optional_vars):
                        self._bind(scope, name, ("opaque", node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                self._bind(
                    scope, (alias.asname or alias.name).split(".")[0], ("opaque", node)
                )
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                self._bind(scope, node.name, ("opaque", node))
        elif isinstance(
            node, (ast.AugAssign, ast.For, ast.AsyncFor, ast.comprehension)
        ):
            for name in _stored_names(node.target):
                self._bind(scope, name, ("opaque", node))

    def owner(self, node: ast.AST, name: str) -> Optional[tuple[int, list[tuple]]]:
        """The scope a use of ``name`` at ``node`` reads, and its bindings."""

        scope = self.scope_of.get(id(node))
        while scope is not None:
            bindings = self.bindings.get(id(scope), {}).get(name)
            if bindings:
                return id(scope), bindings
            scope = self.parent.get(id(scope))
        return None

    def resolves_into(self, node: ast.AST, trusted: Set[tuple]) -> bool:
        """Whether a bare name at ``node`` resolves to a trusted binding."""

        if not isinstance(node, ast.Name):
            return False
        found = self.owner(node, node.id)
        return found is not None and (found[0], node.id) in trusted

    def governing_context(self, node: ast.AST) -> Optional[ast.AST]:
        """The ``with`` expression whose ``as`` name a use at ``node`` sees.

        ``None`` when no enclosing ``with`` binds it, in which case the ordinary
        must-rule over the scope's bindings decides.
        """

        if not isinstance(node, ast.Name):
            return None
        current: Optional[ast.AST] = self.parent_of.get(id(node))
        while current is not None:
            if isinstance(current, (ast.With, ast.AsyncWith)):
                for item in current.items:
                    if (
                        isinstance(item.optional_vars, ast.Name)
                        and item.optional_vars.id == node.id
                    ):
                        return item.context_expr
            current = self.parent_of.get(id(current))
        return None


def _parameter_arguments(
    tree: ast.AST,
) -> dict[tuple[int, str], list[Optional[ast.AST]]]:
    """What every call site passes for each parameter of each function.

    Generated scripts routinely funnel the write through a helper that takes
    the output directory (or the summary mapping) as a parameter -- the corpus
    has ``def write_summary(summary, out_dir)`` writing
    ``out_dir / "step_summary.json"`` inside.  A recogniser that only follows
    assignments cannot read that, and in a fail-closed gate a legal spelling it
    cannot read is a wrong block, not a missed one.

    A parameter therefore inherits what its call sites give it, but only when
    *every* call gives it the same kind of thing.  A ``None`` in the list means
    one call did not bind it; an absent entry means the call sites cannot be
    read positionally at all, or that nothing calls the function.
    """

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    arguments: dict[tuple[int, str], list[Optional[ast.AST]]] = {}
    for function in _function_definitions(tree):
        assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        positional = [*function.args.posonlyargs, *function.args.args]
        parameters = {
            argument.arg for argument in (*positional, *function.args.kwonlyargs)
        }
        index_of = {argument.arg: index for index, argument in enumerate(positional)}
        bound: dict[str, list[Optional[ast.AST]]] = {name: [] for name in parameters}
        called = True
        for call in calls:
            callee = call.func
            name = (
                callee.id
                if isinstance(callee, ast.Name)
                else callee.attr if isinstance(callee, ast.Attribute) else None
            )
            if name != function.name:
                continue
            if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
                keyword.arg is None for keyword in call.keywords
            ):
                # `f(*args)` / `f(**kwargs)` -- positions are unknowable.
                bound = {}
                called = False
                break
            for parameter in parameters:
                index = index_of.get(parameter)
                value = (
                    call.args[index]
                    if index is not None and index < len(call.args)
                    else None
                )
                for keyword in call.keywords:
                    if keyword.arg == parameter:
                        value = keyword.value
                bound[parameter].append(value)
        if not called:
            continue
        for parameter, values in bound.items():
            if values:
                arguments[(id(function), parameter)] = values
    return arguments


class _Destinations:
    """Which names denote the host's output directory and canonical summary.

    Four name sets, each resolved by the same must-rule over the scope tree:
    the directory the host handed the step, the canonical filename when it is
    kept in a variable, the summary path itself, and an open handle on it.
    Every one of them was a flat script-wide set before, and every one of them
    had the same hole.
    """

    def __init__(self, tree: ast.Module) -> None:
        self.scopes = _Scopes(tree)
        self.arguments = _parameter_arguments(tree)
        self.literals = self._resolve_literals()
        self.directories = self._resolve(self._binding_is_the_directory)
        self.summaries = self._resolve(self._binding_denotes_the_summary)
        self.handles = self._resolve(self._binding_is_a_handle)

    # -- the fixpoint ---------------------------------------------------------

    def _resolve(self, predicate) -> Set[tuple]:
        """Every ``(scope, name)`` whose *every* binding satisfies ``predicate``."""

        total = sum(len(names) for names in self.scopes.bindings.values())
        trusted: Set[tuple] = set()
        for _ in range(total + 1):
            grew = False
            for scope_id, names in self.scopes.bindings.items():
                for name, bindings in names.items():
                    key = (scope_id, name)
                    if key in trusted or not bindings:
                        continue
                    if all(predicate(name, binding, trusted) for binding in bindings):
                        trusted.add(key)
                        grew = True
            if not grew:
                break
        return trusted

    def _from_call_sites(self, function: ast.AST, name: str, reads) -> bool:
        values = self.arguments.get((id(function), name))
        return bool(values) and all(
            value is not None and reads(value) for value in values
        )

    # -- the host's output directory -----------------------------------------

    def _binding_is_the_directory(self, name, binding, trusted) -> bool:
        kind, payload = binding
        if kind == "value":
            return self._expression_is_the_directory(payload, trusted)
        if kind == "param":
            return self._from_call_sites(
                payload,
                name,
                lambda node: self._expression_is_the_directory(node, trusted),
            )
        return False

    def _expression_is_the_directory(self, node: ast.AST, trusted: Set[tuple]) -> bool:
        node = _strip_path_wrappers(node)
        if isinstance(node, ast.Name):
            return self.scopes.resolves_into(node, trusted)
        return _reads_the_host_output_directory(node)

    def is_the_output_directory(self, node: ast.AST) -> bool:
        return self._expression_is_the_directory(node, self.directories)

    # -- the canonical filename ----------------------------------------------

    def _resolve_literals(self) -> dict[tuple, Set[str]]:
        """Every ``(scope, name)`` bound only to string literals, and to which.

        Absent means the name has a binding this cannot read, so it is not
        provably the canonical filename -- the same must-rule as the rest.
        """

        total = sum(len(names) for names in self.scopes.bindings.values())
        literals: dict[tuple, Set[str]] = {}
        for _ in range(total + 1):
            grew = False
            for scope_id, names in self.scopes.bindings.items():
                for name, bindings in names.items():
                    key = (scope_id, name)
                    if key in literals or not bindings:
                        continue
                    found: Set[str] = set()
                    for kind, payload in bindings:
                        if kind != "value":
                            found = set()
                            break
                        if isinstance(payload, ast.Constant) and isinstance(
                            payload.value, str
                        ):
                            found.add(payload.value)
                            continue
                        seen = None
                        if isinstance(payload, ast.Name):
                            inner = self.scopes.owner(payload, payload.id)
                            if inner is not None:
                                seen = literals.get((inner[0], payload.id))
                        if seen is None:
                            found = set()
                            break
                        found |= seen
                    if found:
                        literals[key] = found
                        grew = True
            if not grew:
                break
        return literals

    def names_the_canonical_summary(self, node: ast.AST) -> bool:
        """Whether an expression can only be the canonical summary filename.

        Exactly, not by last component: ``out_dir / "audit/step_summary.json"``
        puts the file somewhere the host does not look, and a check that took
        the basename could not tell the two apart.
        """

        node = _strip_path_wrappers(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value == CANONICAL_STEP_SUMMARY_FILENAME
        if not isinstance(node, ast.Name):
            return False
        found = self.scopes.owner(node, node.id)
        if found is None:
            return False
        return self.literals.get((found[0], node.id)) == {
            CANONICAL_STEP_SUMMARY_FILENAME
        }

    # -- the canonical summary path ------------------------------------------

    def _binding_denotes_the_summary(self, name, binding, trusted) -> bool:
        kind, payload = binding
        if kind == "value":
            return self._expression_denotes_the_summary(payload, trusted)
        if kind == "param":
            return self._from_call_sites(
                payload,
                name,
                lambda node: self._expression_denotes_the_summary(node, trusted),
            )
        return False

    def _expression_denotes_the_summary(
        self, node: ast.AST, trusted: Set[tuple]
    ) -> bool:
        """Whether an expression names the summary artifact the host opens.

        That is ``<host output directory>/step_summary.json`` and nothing
        else: the directory must be the one the host handed the step, and the
        filename must be its direct child.  Anything this cannot read is
        refused, so a spelling it does not know costs a repair rather than
        buying a pass -- which is why every spelling in the real corpus is
        locked by test.
        """

        node = _strip_path_wrappers(node)
        if isinstance(node, ast.Name):
            return self.scopes.resolves_into(node, trusted)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return self.is_the_output_directory(
                node.left
            ) and self.names_the_canonical_summary(node.right)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return False
        callee = node.func
        if callee.attr == "joinpath" and len(node.args) == 1:
            return self.is_the_output_directory(
                callee.value
            ) and self.names_the_canonical_summary(node.args[0])
        if callee.attr == "join" and len(node.args) == 2:
            return self.is_the_output_directory(
                node.args[0]
            ) and self.names_the_canonical_summary(node.args[1])
        return False

    def denotes_the_summary(self, node: ast.AST) -> bool:
        return self._expression_denotes_the_summary(node, self.summaries)

    # -- an open handle on it -------------------------------------------------

    def _binding_is_a_handle(self, name, binding, trusted) -> bool:
        kind, payload = binding
        if kind in {"value", "open"}:
            return self._expression_is_a_handle(payload, trusted)
        if kind == "param":
            return self._from_call_sites(
                payload, name, lambda node: self._expression_is_a_handle(node, trusted)
            )
        return False

    def _expression_is_a_handle(self, node: ast.AST, trusted: Set[tuple]) -> bool:
        if isinstance(node, ast.Name):
            # A `with ... as` name means whatever the innermost enclosing
            # `with` opened, and nothing else -- reusing one name across two
            # blocks is ordinary Python, and reading it as a mixed binding is
            # what made a real compliant write invisible.
            governing = self.scopes.governing_context(node)
            if governing is not None:
                return _opens_the_canonical_summary(governing, self.denotes_the_summary)
            return self.scopes.resolves_into(node, trusted)
        return _opens_the_canonical_summary(node, self.denotes_the_summary)

    def addresses_the_summary(self, node: ast.AST) -> bool:
        """The file itself, or a handle already open on it."""

        return self.denotes_the_summary(node) or self._expression_is_a_handle(
            node, self.handles
        )


def _accessed_key(node: ast.AST) -> Optional[tuple[ast.AST, object]]:
    """The container and key of ``x.get(k)`` / ``x[k]``, if the key is literal."""

    if (
        isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.Or)
        and len(node.values) == 2
        and (
            isinstance(node.values[1], ast.Dict)
            and not node.values[1].keys
            or isinstance(node.values[1], ast.Call)
            and isinstance(node.values[1].func, ast.Name)
            and node.values[1].func.id == "dict"
            and not node.values[1].args
            and not node.values[1].keywords
        )
    ):
        # Generated code commonly normalises an absent optional mapping with
        # ``contract.get("analysis_plausibility_range") or {}``.  The empty
        # fallback cannot introduce a bound, so the accessed host key remains
        # authoritative.  A populated or computed fallback is deliberately
        # not unwrapped: it could replace the sealed bounds with source
        # literals and must therefore remain unattributable.
        return _accessed_key(node.values[0])
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.func.value is not None
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ):
        return node.func.value, node.args[0].value
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return node.value, node.slice.value
    return None


def _unwrap_numeric_narrowing(node: ast.AST) -> ast.AST:
    """Strip one ``float(...)`` / ``int(...)`` around a host bound."""

    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"float", "int"}
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    return node


def _unwrapped_bound_name(node: ast.AST) -> Optional[str]:
    """Return the local name under supported numeric narrowing calls."""

    unwrapped = _unwrap_numeric_narrowing(node)
    return unwrapped.id if isinstance(unwrapped, ast.Name) else None


def _branch_values(node: ast.AST) -> list[ast.AST]:
    """Every value a (possibly conditional) expression can take.

    A script that has to cope with both the mapping and the two-item spellings
    of the range writes the choice inline, so the bound arrives through one arm
    of a conditional expression.  Either arm reads the same contract.
    """

    if isinstance(node, ast.IfExp):
        return [*_branch_values(node.body), *_branch_values(node.orelse)]
    return [node]


def _plausibility_range_and_bound_names(
    tree: ast.AST,
    *,
    active_node_ids: Optional[Set[int]] = None,
) -> tuple[Set[str], Set[str]]:
    """Local names holding a bound read out of a declared plausibility range.

    The bound, not the series, is what identifies a plausibility check.  An
    earlier draft also anchored on the ranged variable, and a sweep over the
    real generated scripts showed why that cannot work: a cohort eligibility
    threshold (``coerced_age >= threshold``) is an ordering comparison on the
    very same column, so it was read as a plausibility test and dragged the
    whole cohort-construction chain in with it.  A bound comes from the
    contract; an inclusion threshold comes from a receipt, and only the
    contract is this policy's business.
    """

    assignments = [
        node
        for node in _active_nodes(tree, active_node_ids)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    ]
    single = [node for node in assignments if isinstance(node.targets[0], ast.Name)]
    unpacked = [
        node
        for node in assignments
        if isinstance(node.targets[0], (ast.Tuple, ast.List))
    ]
    ranges: Set[str] = set()
    for node in single:
        access = _accessed_key(_unwrap_numeric_narrowing(node.value))
        if access is not None and str(access[1]).endswith("plausibility_range"):
            assert isinstance(node.targets[0], ast.Name)
            ranges.add(node.targets[0].id)

    def _reads_a_bound(value: ast.AST) -> bool:
        access = _accessed_key(_unwrap_numeric_narrowing(value))
        if access is None:
            return False
        container, key = access
        return (
            key in _BOUND_KEYS
            and isinstance(container, ast.Name)
            and container.id in ranges
        )

    bounds: Set[str] = set()
    for _ in range(len(assignments) + 1):
        grew = False
        for node in single:
            target = node.targets[0]
            assert isinstance(target, ast.Name)
            if target.id not in bounds and _reads_a_bound(node.value):
                bounds.add(target.id)
                grew = True
        for node in unpacked:
            # `minimum, maximum = plausibility` and the element-wise spelling.
            # Real generated code uses both, and a bound this misses is not a
            # missing block but a wrong one: the check downstream reports that
            # nothing could be attributed.
            targets = [
                element
                for element in node.targets[0].elts
                if isinstance(element, ast.Name)
            ]
            if not targets:
                continue
            for value in _branch_values(node.value):
                if isinstance(value, ast.Name) and value.id in ranges:
                    sources: Sequence[Optional[ast.AST]] = [None] * len(targets)
                elif isinstance(value, (ast.Tuple, ast.List)) and len(
                    value.elts
                ) == len(node.targets[0].elts):
                    sources = [
                        element
                        for element, target in zip(value.elts, node.targets[0].elts)
                        if isinstance(target, ast.Name)
                    ]
                else:
                    continue
                for target, source in zip(targets, sources):
                    if target.id in bounds:
                        continue
                    if source is None or _reads_a_bound(source):
                        bounds.add(target.id)
                        grew = True
        if not grew:
            break
    return ranges, bounds


def _declared_policy_action_names(
    tree: ast.AST,
    *,
    active_node_ids: Optional[Set[int]] = None,
) -> Set[str]:
    """Names whose every binding reads the sealed out-of-range action."""

    assignments: dict[str, list[ast.AST]] = {}
    for node in _active_nodes(tree, active_node_ids):
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and getattr(node, "value", None) is not None
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(node.value)

    def _accesses(value: ast.AST, key: str, containers: Set[str]) -> bool:
        access = _accessed_key(value)
        if access is None or access[1] != key:
            return False
        container = access[0]
        if not containers:
            return True
        if isinstance(container, ast.Name) and container.id in containers:
            return True
        nested = _accessed_key(container)
        return nested is not None and nested[1] == "plausibility_policy"

    policy_mappings: Set[str] = set()
    for _ in range(len(assignments) + 1):
        grew = False
        for name, values in assignments.items():
            if name in policy_mappings or not values:
                continue
            if all(
                _accesses(value, "plausibility_policy", set())
                or isinstance(value, ast.Name)
                and value.id in policy_mappings
                for value in values
            ):
                policy_mappings.add(name)
                grew = True
        if not grew:
            break

    actions: Set[str] = set()
    for _ in range(len(assignments) + 1):
        grew = False
        for name, values in assignments.items():
            if name in actions or not values:
                continue
            if all(
                _accesses(value, "out_of_range_action", policy_mappings)
                or isinstance(value, ast.Name)
                and value.id in actions
                for value in values
            ):
                actions.add(name)
                grew = True
        if not grew:
            break
    return actions


def _denotes_a_bound(node: ast.AST, ranges: Set[str], bounds: Set[str]) -> bool:
    """Whether a comparison operand is a declared plausibility bound."""

    if _unwrapped_bound_name(node) in bounds:
        return True
    access = _accessed_key(_unwrap_numeric_narrowing(node))
    if access is None:
        return False
    container, key = access
    if key not in _BOUND_KEYS:
        return False
    if isinstance(container, ast.Name) and container.id in ranges:
        return True
    inner = _accessed_key(container)
    return inner is not None and str(inner[1]).endswith("plausibility_range")


def _subscript_root(node: ast.AST) -> Optional[str]:
    """The name a (possibly chained) subscript target is rooted at."""

    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute) and node.attr in {"at", "loc", "iloc"}:
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _call_arguments(node: ast.Call) -> list[ast.AST]:
    return [*node.args, *(keyword.value for keyword in node.keywords)]


def _opens_the_canonical_summary(node: ast.AST, denotes) -> bool:
    """Whether an expression opens the summary artifact the host reads."""

    if not isinstance(node, ast.Call):
        return False
    callee = node.func
    candidates = _call_arguments(node)
    if isinstance(callee, ast.Name) and callee.id == "open":
        pass
    elif isinstance(callee, ast.Attribute) and callee.attr == "open":
        candidates.append(callee.value)
    else:
        return False
    return any(denotes(candidate) for candidate in candidates)


#: Constructors that produce an empty container regardless of how they are
#: spelled.  ``defaultdict(dict)`` takes an argument and is still empty.
_EMPTY_CONSTRUCTORS = frozenset({"dict", "list", "set", "tuple", "DataFrame", "Series"})
_EMPTY_FACTORIES = frozenset({"defaultdict", "OrderedDict", "Counter"})


def _initialises_an_accumulator(node: ast.AST) -> bool:
    """Whether a binding seeds an empty accumulator rather than replacing one.

    ``audit = {}`` before ``audit[column] = ...`` is the shape nearly every
    compliant script writes, and ``n_below = 0`` before the conditional count
    is the same idea one level down.  Neither says the name cannot carry the
    computed record.  A *populated* literal does.
    """

    if isinstance(node, ast.Constant):
        return node.value in (None, 0, 0.0, "", False)
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return not node.elts
    if isinstance(node, ast.Call):
        callee = node.func
        name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr if isinstance(callee, ast.Attribute) else None
        )
        if name in _EMPTY_FACTORIES:
            return True
        return name in _EMPTY_CONSTRUCTORS and not node.args
    return False


def _writer_functions(
    tree: ast.AST,
    *,
    active_node_ids: Optional[Set[int]] = None,
) -> Set[str]:
    """Locally defined helpers that write, recognised by body rather than name.

    Generated scripts routinely funnel every artifact through one small helper.
    Recognising it structurally is what lets the writer-name set stay put
    instead of growing a new entry each time a script invents a name for it.
    """

    return {
        function.name
        for function in _active_nodes(tree, active_node_ids)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr in _WRITER_ATTRIBUTES
            for inner in ast.walk(function)
            if active_node_ids is None or id(inner) in active_node_ids
        )
    }


def _is_terminal_failure(statements: Sequence[ast.stmt]) -> bool:
    """Whether a block can only end the step."""

    for statement in statements:
        if isinstance(statement, ast.Raise):
            return True
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and (
                (
                    isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr in _TERMINATING_CALLS
                )
                or (
                    isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id in _TERMINATING_CALLS
                )
            )
        ):
            return True
    return False


def _plausibility_comparisons(
    tree: ast.Module,
    *,
    active_node_ids: Optional[Set[int]] = None,
) -> list[ast.Compare]:
    """Ordering comparisons that test a value against a declared bound.

    Anchored on the bound rather than on the column, which is also what makes
    it work on the shape real generated code writes: a per-column helper takes
    the series as a parameter, so the variable's name appears nowhere near the
    comparison and only the bound identifies the check.
    """

    ranges, bounds = _plausibility_range_and_bound_names(
        tree,
        active_node_ids=active_node_ids,
    )
    if not ranges and not bounds:
        return []
    found: list[ast.Compare] = []
    for node in _active_nodes(tree, active_node_ids):
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
        ):
            continue
        if _denotes_a_bound(node.left, ranges, bounds) or _denotes_a_bound(
            node.comparators[0], ranges, bounds
        ):
            found.append(node)
    return found


def _expected_column_literals(
    node: ast.AST,
    expected_columns: Set[str],
) -> Set[str]:
    return {
        str(inner.value)
        for inner in ast.walk(node)
        if isinstance(inner, ast.Constant)
        and isinstance(inner.value, str)
        and inner.value in expected_columns
    }


def _expected_data_column_literals(
    node: ast.AST,
    expected_columns: Set[str],
) -> Set[str]:
    return {
        str(inner.slice.value)
        for inner in ast.walk(node)
        if isinstance(inner, ast.Subscript)
        and isinstance(inner.slice, ast.Constant)
        and isinstance(inner.slice.value, str)
        and inner.slice.value in expected_columns
    }


def _target_names(node: ast.AST) -> Set[str]:
    return {
        inner.id
        for inner in ast.walk(node)
        if isinstance(inner, ast.Name)
        and isinstance(inner.ctx, (ast.Store, ast.Del))
    }


def _reads_the_raw_contract_mapping(
    node: ast.AST,
    assignments: dict[str, list[ast.AST]],
    *,
    seen: Optional[Set[str]] = None,
) -> bool:
    """Whether an expression iterates the host-sealed raw contract mapping."""

    seen = set() if seen is None else set(seen)
    constants = {
        inner.value
        for inner in ast.walk(node)
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
    }
    if {"raw_input_contracts", "contracts"} <= constants:
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"items", "keys", "values"}:
            return _reads_the_raw_contract_mapping(
                node.func.value,
                assignments,
                seen=seen,
            )
    if not isinstance(node, ast.Name) or node.id in seen:
        return False
    seen.add(node.id)
    values = assignments.get(node.id, ())
    return bool(values) and all(
        _reads_the_raw_contract_mapping(value, assignments, seen=seen)
        for value in values
    )


def _comparison_scope_coverage(
    tree: ast.Module,
    comparisons: Sequence[ast.Compare],
    *,
    active_node_ids: Set[int],
    expected_columns: Sequence[str],
) -> Set[str]:
    """Columns whose comparison is connected to reachable step code.

    The receipt gate checks the exact output-key set after execution.  This
    preflight proves the other half: every expected column is connected to a
    reachable comparison, either by a literal call argument/direct data access
    or by a loop over the sealed raw-contract mapping.  A single invoked helper
    can no longer lend credibility to hand-written receipts for other columns.
    """

    expected = set(expected_columns)
    if not expected:
        return set()
    parent = {
        id(child): outer
        for outer in ast.walk(tree)
        for child in ast.iter_child_nodes(outer)
    }
    assignments: dict[str, list[ast.AST]] = {}
    for node in _active_nodes(tree, active_node_ids):
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and getattr(node, "value", None) is not None
        ):
            targets = (
                list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            )
            for target in targets:
                for name in _target_names(target):
                    assignments.setdefault(name, []).append(node.value)

    def _owner(node: ast.AST) -> Optional[ast.AST]:
        current = parent.get(id(node))
        while current is not None and not isinstance(
            current,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            current = parent.get(id(current))
        return current

    calls = [
        node
        for node in _active_nodes(tree, active_node_ids)
        if isinstance(node, ast.Call)
    ]

    def _upstream_names(node: ast.AST) -> Set[str]:
        pending = [
            inner.id
            for inner in ast.walk(node)
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load)
        ]
        seen_names: Set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            for value in assignments.get(name, ()):
                pending.extend(
                    inner.id
                    for inner in ast.walk(value)
                    if isinstance(inner, ast.Name)
                    and isinstance(inner.ctx, ast.Load)
                )
        return seen_names

    def _upstream_literals(node: ast.AST) -> Set[str]:
        literals = _expected_column_literals(node, expected)
        for name in _upstream_names(node):
            for value in assignments.get(name, ()):
                literals.update(_expected_column_literals(value, expected))
        return literals

    range_names, bound_names = _plausibility_range_and_bound_names(
        tree,
        active_node_ids=active_node_ids,
    )

    def _data_operand(comparison: ast.Compare) -> Optional[ast.AST]:
        left_is_bound = _denotes_a_bound(
            comparison.left,
            range_names,
            bound_names,
        )
        right = comparison.comparators[0]
        right_is_bound = _denotes_a_bound(right, range_names, bound_names)
        if left_is_bound == right_is_bound:
            return None
        return right if left_is_bound else comparison.left

    def _loop_contract_key_names(loop: ast.For | ast.AsyncFor) -> Set[str]:
        """Names that identify a contract key in this mapping iteration."""

        iterator = loop.iter
        if isinstance(iterator, ast.Call) and isinstance(
            iterator.func,
            ast.Attribute,
        ):
            if iterator.func.attr == "values":
                return set()
            if iterator.func.attr == "items":
                if isinstance(loop.target, (ast.Tuple, ast.List)) and loop.target.elts:
                    return _target_names(loop.target.elts[0])
                return set()
        return _target_names(loop.target)

    covered: Set[str] = set()
    for comparison in comparisons:
        owner = _owner(comparison)
        if owner is None:
            statement: ast.AST = comparison
            current = parent.get(id(comparison))
            while current is not None and not isinstance(current, ast.stmt):
                current = parent.get(id(current))
            if current is not None:
                statement = current
            covered.update(_upstream_literals(statement))
            data_operand = _data_operand(comparison)
            current = parent.get(id(comparison))
            while current is not None:
                if isinstance(current, (ast.For, ast.AsyncFor)):
                    key_names = _loop_contract_key_names(current)
                    if (
                        data_operand is not None
                        and key_names.intersection(_upstream_names(data_operand))
                        and _reads_the_raw_contract_mapping(
                            current.iter,
                            assignments,
                        )
                    ):
                        covered.update(expected)
                    break
                current = parent.get(id(current))
            continue

        covered.update(_expected_data_column_literals(owner, expected))
        owner_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Name) and call.func.id == owner.name
        ]
        for call in owner_calls:
            covered.update(_expected_column_literals(call, expected))
            current = parent.get(id(call))
            while current is not None and current is not owner:
                if isinstance(current, (ast.For, ast.AsyncFor)):
                    loop_names = _target_names(current.target)
                    call_names = {
                        inner.id
                        for argument in _call_arguments(call)
                        for inner in ast.walk(argument)
                        if isinstance(inner, ast.Name)
                        and isinstance(inner.ctx, ast.Load)
                    }
                    if loop_names.intersection(call_names):
                        covered.update(
                            _expected_column_literals(current.iter, expected)
                        )
                        if _reads_the_raw_contract_mapping(
                            current.iter,
                            assignments,
                        ):
                            covered.update(expected)
                    break
                current = parent.get(id(current))
    return covered


class _FlagFlow:
    """Where the out-of-range record is computed, and where it ends up.

    Name-level rather than value-level: a fixpoint over assignments, container
    literals and function returns.  That is enough for every compliant shape
    generated code actually writes, and everything it cannot follow ends in a
    block, so its limits cost provider repairs instead of buying passes.
    """

    def __init__(
        self,
        tree: ast.Module,
        comparisons: Sequence[ast.Compare],
        *,
        active_node_ids: Optional[Set[int]] = None,
    ) -> None:
        self.tree = tree
        self.active_node_ids = active_node_ids
        self.compared = {id(node) for node in comparisons}
        self.parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        #: Names, per lexical scope, the computed record *can* reach.
        self.carriers: Set[tuple] = set()
        self.returning_functions: Set[str] = set()
        #: How this script reaches the one artifact the host opens afterwards.
        #: A delivery is a write *to that file*; see `receipt_deliveries`.
        #: Scope-aware and binding-complete: a name that is the host's output
        #: directory in one place and something else in another is neither.
        self.destinations = _Destinations(tree)
        self.scopes = self.destinations.scopes
        self.writer_functions = _writer_functions(
            tree,
            active_node_ids=active_node_ids,
        )
        self._resolve()
        #: Names no later binding replaces with something that cannot carry.
        self.certified: Set[tuple] = self._resolve_certified()
        self.certainly_returning: Set[str] = self._resolve_certainly_returning()
        #: Names holding a mapping that carries the record under the receipt
        #: key.  Reaching the file is not enough on its own: the host reads one
        #: key of it, so a count filed anywhere else in the same summary is a
        #: count the host never sees.
        self.receipt_carriers: Set[tuple] = self._resolve_receipt_carriers()
        #: The mask and count themselves, without the containers they are
        #: later filed into. A guard on `mask.any()` or on `n_out > 0` is a
        #: rejection; a guard on the summary dict that happens to hold the
        #: count is an unrelated check, and reading the wide carrier set for
        #: both is how an ordinary contract assertion gets reported as a
        #: plausibility rejection.
        self.direct_values: Set[str] = self._resolve_direct_values()

    def _nodes(self) -> Iterator[ast.AST]:
        return _active_nodes(self.tree, self.active_node_ids)

    def _function_nodes(self, function: ast.AST) -> Iterator[ast.AST]:
        return (
            node
            for node in ast.walk(function)
            if self.active_node_ids is None or id(node) in self.active_node_ids
        )

    def _reaches(self, node: ast.AST, names: Set[tuple], functions: Set[str]) -> bool:
        for inner in ast.walk(node):
            if id(inner) in self.compared:
                return True
            if (
                isinstance(inner, ast.Name)
                and isinstance(inner.ctx, ast.Load)
                and self.scopes.resolves_into(inner, names)
            ):
                return True
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in functions
            ):
                return True
        return False

    def _carries(self, node: ast.AST) -> bool:
        """Whether the computed record *can* reach here."""

        return self._reaches(node, self.carriers, self.returning_functions)

    def _certainly_carries(self, node: ast.AST) -> bool:
        """Whether it can reach here and nothing later replaces it."""

        return self._reaches(node, self.certified, self.certainly_returning)

    def _scoped(self, node: ast.AST, name: str) -> Optional[tuple]:
        """Which scope's ``name`` a write at ``node`` touches.

        Resolved outward, not taken from the enclosing scope: filling a
        module-level accumulator from inside a helper -- ``plausibility_audit
        [column] = ...`` in the per-column function, which is the shape the
        real corpus writes -- is a write to the *module's* name.  A subscript
        does not create a local one.
        """

        found = self.scopes.owner(node, name)
        if found is not None:
            return (found[0], name)
        return (id(self.scopes.module), name)

    def _resolve(self) -> None:
        assignments = [
            node
            for node in self._nodes()
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        ]
        functions = [
            node
            for node in self._nodes()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        # Each pass adds at least one name or stops, so the number of
        # assignments plus functions bounds the iteration.
        for _ in range(len(assignments) + len(functions) + 1):
            grew = False
            for node in assignments:
                value = node.value
                if value is None or not self._carries(value):
                    continue
                for name in self._assigned_names(node):
                    key = self._scoped(node, name)
                    if key is not None and key not in self.carriers:
                        self.carriers.add(key)
                        grew = True
            for function in functions:
                if function.name in self.returning_functions:
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and inner.value is not None
                    and self._carries(inner.value)
                    for inner in self._function_nodes(function)
                ):
                    self.returning_functions.add(function.name)
                    grew = True
            if not grew:
                break

    def _resolve_certified(self) -> Set[tuple]:
        """Carriers no other binding of the same name replaces.

        ``carriers`` answers "can the computed record reach this name" -- a
        *may* question, and the right one for finding the computation.
        Certifying a delivery is the opposite question, and the same set
        answers it wrongly::

            plausibility_audit = {...computed...}
            plausibility_audit = {"marker": {"out_of_range_n": 0}}

        leaves the name in ``carriers`` for good, so writing the second value
        to the real artifact reads as a delivery of the first -- the same
        false green as the scratch-directory one, moved from the path to the
        payload.  A whole-name rebinding to something that cannot carry
        therefore disqualifies the name; seeding an empty accumulator does
        not, because it replaces nothing.

        Each binding is judged against the converged ``carriers`` set, not
        against this one, so a name rebound from itself (``audit =
        dict(audit)``) does not disqualify itself.
        """

        certified: Set[tuple] = set()
        for key in self.carriers:
            scope_id, name = key
            bindings = self.scopes.bindings.get(scope_id, {}).get(name)
            if bindings is None:
                # Bound only by a subscript write into an object it never
                # rebinds, so there is nothing that could have replaced it.
                certified.add(key)
                continue
            if all(self._binding_survives(name, binding) for binding in bindings):
                certified.add(key)
        return certified

    def _binding_survives(self, name: str, binding: tuple) -> bool:
        kind, payload = binding
        if kind == "value":
            return self._carries(payload) or _initialises_an_accumulator(payload)
        if kind == "param":
            values = self.destinations.arguments.get((id(payload), name))
            return bool(values) and all(
                value is not None
                and (self._carries(value) or _initialises_an_accumulator(value))
                for value in values
            )
        return False

    def _resolve_certainly_returning(self) -> Set[str]:
        functions = [
            node
            for node in self._nodes()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        certain: Set[str] = set()
        for _ in range(len(functions) + 1):
            grew = False
            for function in functions:
                if function.name in certain:
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and inner.value is not None
                    and self._reaches(inner.value, self.certified, certain)
                    for inner in self._function_nodes(function)
                ):
                    certain.add(function.name)
                    grew = True
            if not grew:
                break
        return certain

    def _resolve_direct_values(self) -> Set[tuple]:
        direct: Set[tuple] = set()
        assignments = [
            node
            for node in self._nodes()
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and not isinstance(node.value, (ast.Dict, ast.List, ast.Set, ast.Tuple))
        ]
        for _ in range(len(assignments) + 1):
            grew = False
            for node in assignments:
                key = self._scoped(node, node.targets[0].id)
                if key is None or key in direct:
                    continue
                if any(
                    id(inner) in self.compared
                    or (
                        isinstance(inner, ast.Name)
                        and isinstance(inner.ctx, ast.Load)
                        and self.scopes.resolves_into(inner, direct)
                    )
                    for inner in ast.walk(node.value)
                ):
                    direct.add(key)
                    grew = True
            if not grew:
                break
        return direct

    def rejects_on(self, node: ast.AST) -> bool:
        """Whether a test is decided by the out-of-range mask or its count."""

        return any(
            id(inner) in self.compared
            or (
                isinstance(inner, ast.Name)
                and isinstance(inner.ctx, ast.Load)
                and self.scopes.resolves_into(inner, self.direct_values)
            )
            for inner in ast.walk(node)
        )

    @staticmethod
    def _assigned_names(node: ast.AST) -> Set[str]:
        targets: list[ast.AST]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:  # pragma: no cover - defensive
            targets = []
        names: Set[str] = set()
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            else:
                root = _subscript_root(target)
                if root is not None:
                    names.add(root)
        return names

    def carries(self, node: ast.AST) -> bool:
        return self._carries(node)

    def records_a_structured_value(self) -> bool:
        """Whether the mask is turned into a kept indicator or number."""

        for node in self._nodes():
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Subscript) for target in node.targets)
                and self._carries(node.value)
            ):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RECORDING_REDUCTIONS
                and self._carries(node.func.value)
            ):
                for outer in self.ancestors(node):
                    if isinstance(outer, (ast.Assign, ast.Dict, ast.keyword)):
                        return True
                    if isinstance(outer, (ast.If, ast.Assert, ast.Raise)):
                        break
        return False

    def ancestors(self, node: ast.AST) -> Iterator[ast.AST]:
        current = self.parents.get(id(node))
        while current is not None:
            yield current
            current = self.parents.get(id(current))

    def statement_of(self, node: ast.AST) -> Optional[ast.stmt]:
        for outer in self.ancestors(node):
            if isinstance(outer, ast.stmt):
                return outer
        return None

    def denotes_the_summary(self, node: ast.AST) -> bool:
        """Whether an expression names the summary artifact the host opens."""

        return self.destinations.denotes_the_summary(node)

    def _addresses_the_summary(self, node: ast.AST) -> bool:
        return self.destinations.addresses_the_summary(node)

    def _resolve_receipt_carriers(self) -> Set[tuple]:
        """Names holding a mapping whose receipt key is the out-of-range record.

        Seeded by the two ways a script files something under a literal key --
        ``summary[key] = record`` and a dict literal -- then propagated the same
        way the carrier set is: through plain assignment and through helpers
        *proven* to return such a mapping.  It deliberately does not propagate
        through nesting: ``{"quality": summary}`` moves the receipt one level
        down, where the host does not look for it.

        This is the set that certifies a delivery, so it is built from
        ``certified`` rather than ``carriers`` and is keyed by scope.  Both
        matter: a script that files the real record under one function's
        ``payload`` and hands a literal to another function's ``payload``
        satisfied the flat spelling of this set with two writes that never
        met.
        """

        names: Set[tuple] = set()
        for node in self._nodes():
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == RECEIPT_SUMMARY_KEY
                        and self._certainly_carries(node.value)
                    ):
                        root = _subscript_root(target)
                        key = None if root is None else self._scoped(node, root)
                        if key is not None:
                            names.add(key)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == RECEIPT_SUMMARY_KEY
                and self._certainly_carries(node.args[1])
            ):
                root = _subscript_root(node.func.value)
                key = None if root is None else self._scoped(node, root)
                if key is not None:
                    names.add(key)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and any(
                    self._is_receipt_mapping(argument, names) for argument in node.args
                )
            ):
                root = _subscript_root(node.func.value)
                key = None if root is None else self._scoped(node, root)
                if key is not None:
                    names.add(key)

        assignments = [
            node
            for node in self._nodes()
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ]
        functions = [
            node
            for node in self._nodes()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        returning: Set[str] = set()
        for _ in range(len(assignments) + len(functions) + 1):
            grew = False
            for node in assignments:
                target = node.targets[0]
                assert isinstance(target, ast.Name)
                key = self._scoped(node, target.id)
                if key is None or key in names:
                    continue
                value = node.value
                if self._is_receipt_mapping(value, names) or (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in returning
                ):
                    names.add(key)
                    grew = True
            for function in functions:
                # A helper the script hands the summary to carries it inside,
                # the same way one it hands the output directory to does.  Only
                # when every call site agrees: one call that does not bind the
                # parameter is enough to leave it unproven -- and the parameter
                # is the one belonging to *this* function, not every parameter
                # of that name in the script.
                for parameter in _parameter_names(function):
                    key = (id(function), parameter)
                    values = self.destinations.arguments.get(key)
                    if key in names or not values:
                        continue
                    if all(
                        value is not None and self._is_receipt_mapping(value, names)
                        for value in values
                    ):
                        names.add(key)
                        grew = True
                if function.name in returning:
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and inner.value is not None
                    and self._is_receipt_mapping(inner.value, names)
                    for inner in self._function_nodes(function)
                ):
                    returning.add(function.name)
                    grew = True
            if not grew:
                break
        return names

    def _is_receipt_mapping(self, node: ast.AST, names: Set[tuple]) -> bool:
        """Whether an expression *is* a mapping carrying the record at its key."""

        for value in _branch_values(node):
            if self.scopes.resolves_into(value, names):
                return True
            if isinstance(value, ast.Dict):
                for key, item in zip(value.keys, value.values):
                    if key is None:  # `{**summary}`
                        if self.scopes.resolves_into(item, names):
                            return True
                    elif (
                        isinstance(key, ast.Constant)
                        and key.value == RECEIPT_SUMMARY_KEY
                        and self._certainly_carries(item)
                    ):
                        return True
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "dict"
            ):
                for keyword in value.keywords:
                    if keyword.arg is None:
                        if self.scopes.resolves_into(keyword.value, names):
                            return True
                    elif keyword.arg == RECEIPT_SUMMARY_KEY and self._certainly_carries(
                        keyword.value
                    ):
                        return True
        return False

    def delivers_the_receipt(self, node: ast.AST) -> bool:
        """Whether an expression handed to a write carries the receipt.

        One step looser than :meth:`_is_receipt_mapping`, and only here: the
        serialization a script wraps the mapping in on the way to the file
        (``json.dumps(summary)``) preserves it, so the argument of a call is
        followed.  A mapping nested under another key is still refused, because
        the containers themselves are read strictly.
        """

        if self._is_receipt_mapping(node, self.receipt_carriers):
            return True
        return isinstance(node, ast.Call) and any(
            self.delivers_the_receipt(argument) for argument in _call_arguments(node)
        )

    def receipt_deliveries(self) -> list[ast.AST]:
        """Expressions that put the receipt into the artifact the host reads.

        Three things have to line up, and dropping any one of them has already
        been shown to open a bypass: the call must *write* (or a log line that
        merely names the file would count), it must write to the canonical
        summary *in the host's own output directory* (or a scratch file with
        the right basename would count), and what it writes must carry the
        record *under the receipt key* (or a summary that mentions it anywhere
        would count).
        """

        deliveries: list[ast.AST] = []
        for node in self._nodes():
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            arguments = _call_arguments(node)
            receiver = callee.value if isinstance(callee, ast.Attribute) else None
            if not (
                any(self._addresses_the_summary(argument) for argument in arguments)
                or (receiver is not None and self._addresses_the_summary(receiver))
            ):
                continue
            if isinstance(callee, ast.Attribute):
                if callee.attr not in _WRITER_ATTRIBUTES:
                    continue
                candidates = [callee.value, *arguments]
            elif isinstance(callee, ast.Name) and callee.id in self.writer_functions:
                candidates = list(arguments)
            else:
                continue
            deliveries.extend(
                candidate
                for candidate in candidates
                if self.delivers_the_receipt(candidate)
            )
        return deliveries

    def reaches_the_step_summary(self) -> bool:
        return bool(self.receipt_deliveries())

    def guarded_by_its_own_count(self, node: ast.AST) -> bool:
        """Whether a node only runs when the out-of-range count is positive."""

        for outer in self.ancestors(node):
            if isinstance(outer, ast.If) and self.rejects_on(outer.test):
                return True
            if isinstance(outer, ast.IfExp) and self.rejects_on(outer.test):
                return True
        return False

    def has_unconditional_computation(self) -> bool:
        for node in self._nodes():
            if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            value = getattr(node, "value", None)
            if value is None:
                continue
            if not any(id(inner) in self.compared for inner in ast.walk(value)):
                continue
            if not self.guarded_by_its_own_count(node):
                return True
        return False

    def has_unconditional_delivery(self) -> bool:
        for expression in self.receipt_deliveries():
            statement = self.statement_of(expression)
            if statement is None or not self.guarded_by_its_own_count(statement):
                return True
        return False


def _tests_the_declared_action(
    node: ast.AST,
    policy_action_names: Set[str],
) -> Optional[bool]:
    """Whether a test decides on the policy being the declared flag-only action.

    Returns ``True`` for a test that is taken *when the policy is* that action,
    ``False`` for one taken when it is not, and ``None`` when the test says
    nothing about the policy.  ``and``/``or`` are searched, because the corpus
    writes the defensive check inline as
    ``if n_out > 0 and policy != "retain_and_flag": raise``.
    """

    if isinstance(node, ast.BoolOp):
        for value in node.values:
            decided = _tests_the_declared_action(value, policy_action_names)
            if decided is not None:
                return decided
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _tests_the_declared_action(node.operand, policy_action_names)
        return None if inner is None else not inner
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
        return None
    operands = [node.left, node.comparators[0]]
    literal_positions = [
        index
        for index, side in enumerate(operands)
        if isinstance(side, ast.Constant) and side.value == RECEIPT_POLICY_VALUE
    ]
    if len(literal_positions) != 1:
        return None
    other = operands[1 - literal_positions[0]]
    if not (
        isinstance(other, ast.Name) and other.id in policy_action_names
    ):
        return None
    return isinstance(node.ops[0], ast.Eq)


def _guarded_by_a_different_policy(
    flow: _FlagFlow,
    node: ast.AST,
    policy_action_names: Set[str],
) -> bool:
    """Whether a terminal branch is only reachable under a *different* policy.

    Generated scripts guard their own fatal fallback on the policy they were
    handed -- ``if policy == "retain_and_flag": pass`` / ``elif n_out: raise``,
    or the same thing inline with ``and policy != ...``.  Under the declared
    flag-only policy that ``raise`` cannot run, so reporting it as a rejection
    is a wrong block, and in a fail-closed gate a wrong block is the expensive
    kind: it cost a real canary its whole run at step 01.

    Being defensive about a policy the host did not declare is good practice,
    not a violation of the one it did.
    """

    child = node
    for outer in flow.ancestors(node):
        if isinstance(outer, ast.If):
            decided = _tests_the_declared_action(
                outer.test,
                policy_action_names,
            )
            if decided is not None:
                in_body = any(child is statement for statement in outer.body)
                # Reached when the policy IS the declared action only if the
                # test says so and we are on the branch it guards.
                if decided is not in_body:
                    return True
        child = outer
    return False


def _query_expression_reads_a_declared_bound(
    node: ast.AST,
    *,
    bound_names: Set[str],
    assignments: Mapping[str, Sequence[ast.AST]],
    seen: Optional[Set[str]] = None,
) -> bool:
    """Whether a pandas-query expression is derived from a contract bound.

    ``DataFrame.query`` hides its comparisons inside a string, so the normal
    AST comparison walk cannot see them.  Generated code uses either an
    f-string/``format`` value or pandas' ``@lower`` spelling.  Follow simple
    aliases as well; a query string assigned one line earlier is not weaker
    evidence merely because it has a name.
    """

    seen = set() if seen is None else set(seen)
    if any(
        isinstance(inner, ast.Name)
        and isinstance(inner.ctx, ast.Load)
        and inner.id in bound_names
        for inner in ast.walk(node)
    ):
        return True
    for inner in ast.walk(node):
        if not (
            isinstance(inner, ast.Constant) and isinstance(inner.value, str)
        ):
            continue
        if any(
            re.search(
                rf"(?<![A-Za-z0-9_])@{re.escape(name)}(?![A-Za-z0-9_])",
                inner.value,
            )
            for name in bound_names
        ):
            return True
    if not isinstance(node, ast.Name) or node.id in seen:
        return False
    seen.add(node.id)
    return any(
        _query_expression_reads_a_declared_bound(
            value,
            bound_names=bound_names,
            assignments=assignments,
            seen=seen,
        )
        for value in assignments.get(node.id, ())
    )


def _flag_only_range_transform(
    node: ast.Call,
    *,
    flow: _FlagFlow,
    range_names: Set[str],
    bound_names: Set[str],
    assignments: Mapping[str, Sequence[ast.AST]],
) -> Optional[str]:
    """Return the mutating/filtering operation tied to the flag-only range.

    The obligation is about preserving rows and values, not merely avoiding
    ``df[mask]``.  Pandas exposes equivalent spellings through method calls;
    those are blocked only when their arguments are connected to the
    out-of-range mask or a bound read from the sealed contract.
    """

    if not isinstance(node.func, ast.Attribute):
        return None
    operation = node.func.attr
    positional = list(node.args)
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}

    if operation == "drop":
        selectors = [
            *positional,
            *(
                value
                for key, value in keywords.items()
                if key in {"index", "labels"}
            ),
        ]
        return operation if any(flow.rejects_on(value) for value in selectors) else None

    if operation in {"where", "mask"}:
        condition = positional[0] if positional else keywords.get("cond")
        return (
            operation
            if condition is not None and flow.rejects_on(condition)
            else None
        )

    if operation == "clip":
        candidates = [
            *positional[:2],
            *(
                value
                for key, value in keywords.items()
                if key in {"lower", "upper"}
            ),
        ]
        return (
            operation
            if any(
                _denotes_a_bound(value, range_names, bound_names)
                for value in candidates
            )
            else None
        )

    if operation == "query":
        expression = positional[0] if positional else keywords.get("expr")
        return (
            operation
            if expression is not None
            and _query_expression_reads_a_declared_bound(
                expression,
                bound_names=bound_names,
                assignments=assignments,
            )
            else None
        )
    return None


def _rejection_findings(
    tree: ast.Module,
    flow: _FlagFlow,
    *,
    step_id: str,
    detail_base: dict,
    active_node_ids: Optional[Set[int]] = None,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    range_names, bound_names = _plausibility_range_and_bound_names(
        tree,
        active_node_ids=active_node_ids,
    )
    policy_action_names = _declared_policy_action_names(
        tree,
        active_node_ids=active_node_ids,
    )
    assignments: dict[str, list[ast.AST]] = {}
    for active in _active_nodes(tree, active_node_ids):
        if not (
            isinstance(active, (ast.Assign, ast.AnnAssign))
            and getattr(active, "value", None) is not None
        ):
            continue
        targets = (
            list(active.targets) if isinstance(active, ast.Assign) else [active.target]
        )
        for target in targets:
            for name in _target_names(target):
                assignments.setdefault(name, []).append(active.value)

    for node in _active_nodes(tree, active_node_ids):
        if (
            isinstance(node, ast.If)
            and flow.rejects_on(node.test)
            and _is_terminal_failure(node.body)
            and not _guarded_by_a_different_policy(
                flow,
                node,
                policy_action_names,
            )
            and _tests_the_declared_action(
                node.test,
                policy_action_names,
            )
            is not False
        ):
            findings.append(
                ValidationFinding(
                    validator=_VALIDATOR,
                    severity="error",
                    message=(
                        f"Step {step_id} ends the analysis when a value falls "
                        "outside its declared plausibility range. That range is "
                        "flag-only: keep every such row and record the count."
                    ),
                    detail={
                        **detail_base,
                        "reason": "flag_only_plausibility_range_rejected",
                        "line": node.lineno,
                    },
                )
            )
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Subscript)
            and flow.rejects_on(node.value.slice)
        ):
            findings.append(
                ValidationFinding(
                    validator=_VALIDATOR,
                    severity="error",
                    message=(
                        f"Step {step_id} filters rows by their declared "
                        "plausibility range. That range is flag-only: keep "
                        "every such row and record the count."
                    ),
                    detail={
                        **detail_base,
                        "reason": "flag_only_plausibility_range_filtered",
                        "line": node.lineno,
                    },
                )
            )
        if isinstance(node, ast.Call):
            operation = _flag_only_range_transform(
                node,
                flow=flow,
                range_names=range_names,
                bound_names=bound_names,
                assignments=assignments,
            )
            if operation is not None:
                findings.append(
                    ValidationFinding(
                        validator=_VALIDATOR,
                        severity="error",
                        message=(
                            f"Step {step_id} uses pandas `{operation}` to remove "
                            "or replace values based on its declared plausibility "
                            "range. That range is flag-only: preserve the original "
                            "rows and values and record the count."
                        ),
                        detail={
                            **detail_base,
                            "reason": "flag_only_plausibility_range_transformed",
                            "operation": operation,
                            "line": node.lineno,
                        },
                    )
                )
    return findings


def flag_only_plausibility_obligation_findings(
    tree: Optional[ast.Module],
    *,
    script_text: str,
    step: AnalysisStep,
    scope: FlagOnlyPlausibilityScope,
) -> list[ValidationFinding]:
    """Check both halves of ``retain_and_flag`` without consulting an auditor."""

    scope.require_step(step.step_id)
    text = str(script_text or "")
    if tree is None:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
    # The host-owned step scope is shared with the post-execution receipt check,
    # so generated source cannot make the two halves disagree about which step
    # owes the obligation.
    trigger = step_is_under_the_flag_only_obligation(
        script_text=text,
        tree=tree,
        scope=scope,
    )
    if trigger is None:
        return []

    step_id = str(step.step_id)
    detail_base = {
        "step_id": step_id,
        "issue_code": "flag_only_plausibility_obligation",
        "policy": "retain_and_flag",
        "policy_authority": scope.authority_kind,
        "trigger": trigger,
        "expected_columns": list(scope.expected_columns),
        "source_contracts_sha256": scope.source_contracts_sha256,
        "scope_sha256": scope.scope_sha256,
    }

    reachability = _RuntimeReachability(tree)
    active_node_ids = reachability.active_node_ids
    comparisons = _plausibility_comparisons(
        tree,
        active_node_ids=active_node_ids,
    )
    if not comparisons:
        return [
            ValidationFinding(
                validator=_VALIDATOR,
                severity="error",
                message=(
                    f"Step {step_id} names a plausibility range but no "
                    "comparison against a bound read from it can be located, "
                    "so nothing proves the out-of-range rows were counted. "
                    "Compare each ranged value against the minimum/maximum "
                    "taken from the contract -- not against a bound written "
                    f"into the source as a literal. {RECEIPT_CONTRACT_SENTENCE}"
                ),
                detail={
                    **detail_base,
                    "reason": "plausibility_check_not_attributable",
                    "flag_evidence": "not_attributable",
                },
            )
        ]

    covered_columns = _comparison_scope_coverage(
        tree,
        comparisons,
        active_node_ids=active_node_ids,
        expected_columns=scope.expected_columns,
    )
    missing_columns = sorted(set(scope.expected_columns) - covered_columns)
    if missing_columns:
        return [
            ValidationFinding(
                validator=_VALIDATOR,
                severity="error",
                message=(
                    f"Step {step_id} does not connect every exact flag-only "
                    "plausibility column to reachable comparison code. Call "
                    "the validator for each listed column or iterate the "
                    "host-sealed raw-input contract mapping before writing "
                    f"its receipt. {RECEIPT_CONTRACT_SENTENCE}"
                ),
                detail={
                    **detail_base,
                    "reason": "plausibility_scope_column_not_attributable",
                    "flag_evidence": "partial_scope",
                    "covered_columns": sorted(covered_columns),
                    "missing_columns": missing_columns,
                },
            )
        ]

    flow = _FlagFlow(
        tree,
        comparisons,
        active_node_ids=active_node_ids,
    )
    findings = _rejection_findings(
        tree,
        flow,
        step_id=step_id,
        detail_base=detail_base,
        active_node_ids=active_node_ids,
    )

    records = flow.records_a_structured_value()
    delivered = records and flow.reaches_the_step_summary()
    if not records:
        findings.append(
            ValidationFinding(
                validator=_VALIDATOR,
                severity="error",
                message=(
                    f"Step {step_id} compares values against their declared "
                    "plausibility range and keeps no record of the result. "
                    "`retain_and_flag` owes a structured count or per-row "
                    f"indicator, not only the retention. {RECEIPT_CONTRACT_SENTENCE}"
                ),
                detail={
                    **detail_base,
                    "reason": "out_of_range_record_absent",
                    "flag_evidence": "absent",
                },
            )
        )
    elif not delivered:
        findings.append(
            ValidationFinding(
                validator=_VALIDATOR,
                severity="error",
                message=(
                    f"Step {step_id} counts the out-of-range rows but never "
                    f"files them under {RECEIPT_SUMMARY_KEY!r} in the "
                    f"{CANONICAL_STEP_SUMMARY_FILENAME} it writes into the "
                    "host's step output directory. A value left in a local, "
                    "printed to the console, written to a scratch path that "
                    "merely ends in the same filename, or filed under some "
                    "other key is not a record the host reads. "
                    f"{RECEIPT_CONTRACT_SENTENCE}"
                ),
                detail={
                    **detail_base,
                    "reason": "out_of_range_record_not_in_declared_output",
                    "flag_evidence": "local_only",
                },
            )
        )
    elif not (
        flow.has_unconditional_computation() and flow.has_unconditional_delivery()
    ):
        findings.append(
            ValidationFinding(
                validator=_VALIDATOR,
                severity="error",
                message=(
                    f"Step {step_id} records the out-of-range count only when "
                    "it is positive. Write the receipt on every path: a count "
                    "of zero is a result, and its absence cannot be told apart "
                    "from never having looked."
                ),
                detail={
                    **detail_base,
                    "reason": "out_of_range_receipt_conditional_on_count",
                    "flag_evidence": "conditional",
                },
            )
        )
    return findings


__all__ = [
    "CANONICAL_STEP_SUMMARY_FILENAME",
    "POLICY_CONTRACT_KEY",
    "REPAIR_RECEIPT_MARKER",
    "flag_only_plausibility_obligation_findings",
]
