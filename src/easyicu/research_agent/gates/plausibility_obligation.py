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

Case neutrality: every variable name comes from ``context``.  Nothing here
knows which study, benchmark, column or bound is in play.
"""

from __future__ import annotations

import ast
from typing import Iterator, Optional, Sequence, Set

from ..audits.validators import _unwrapped_bound_name
from ..schema import AnalysisStep, ResearchContext, ValidationFinding
from .plausibility_receipt import (
    CANONICAL_STEP_SUMMARY_FILENAME,
    HOST_OUTPUT_DIR_ENV_KEYS,
    POLICY_CONTRACT_KEY,
    RECEIPT_CONTRACT_SENTENCE,
    RECEIPT_SUMMARY_KEY,
    REPAIR_RECEIPT_MARKER,
    ranged_variable_names,
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


def _host_output_directory_names(tree: ast.AST) -> Set[str]:
    """Local names bound to the host's own step-output directory."""

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    names: Set[str] = set()
    for _ in range(len(assignments) + 1):
        grew = False
        for node in assignments:
            target = node.targets[0]
            assert isinstance(target, ast.Name)
            if target.id in names:
                continue
            value = _strip_path_wrappers(node.value)
            if _reads_the_host_output_directory(value) or (
                isinstance(value, ast.Name) and value.id in names
            ):
                names.add(target.id)
                grew = True
        if not grew:
            break
    return names


def _is_the_host_output_directory(node: ast.AST, directories: Set[str]) -> bool:
    node = _strip_path_wrappers(node)
    if isinstance(node, ast.Name):
        return node.id in directories
    return _reads_the_host_output_directory(node)


def _literal_string_names(tree: ast.AST) -> dict[str, Set[str]]:
    """Local names bound to string literals, for a filename kept in a variable."""

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    literals: dict[str, Set[str]] = {}
    for _ in range(len(assignments) + 1):
        grew = False
        for node in assignments:
            target = node.targets[0]
            assert isinstance(target, ast.Name)
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found = {value.value}
            elif isinstance(value, ast.Name):
                found = set(literals.get(value.id, ()))
            else:
                continue
            if found - literals.get(target.id, set()):
                literals.setdefault(target.id, set()).update(found)
                grew = True
        if not grew:
            break
    return literals


def _names_the_canonical_summary(node: ast.AST, literals: dict[str, Set[str]]) -> bool:
    """Whether an expression is exactly the canonical summary filename.

    Exactly, not by last component: ``out_dir / "audit/step_summary.json"``
    puts the file somewhere the host does not look, and a check that took the
    basename could not tell the two apart.
    """

    node = _strip_path_wrappers(node)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value == CANONICAL_STEP_SUMMARY_FILENAME
    if isinstance(node, ast.Name):
        return CANONICAL_STEP_SUMMARY_FILENAME in literals.get(node.id, ())
    return False


def _denotes_the_canonical_summary(
    node: ast.AST,
    *,
    directories: Set[str],
    literals: dict[str, Set[str]],
    summaries: Set[str],
) -> bool:
    """Whether an expression names the summary artifact the host itself opens.

    That is ``<host output directory>/step_summary.json`` and nothing else:
    the directory must be the one the host handed the step, and the filename
    must be its direct child.  Anything the check cannot read this way is
    refused, so a spelling it does not know costs a repair rather than buying
    a pass -- which is why every spelling in the real corpus is locked by test.
    """

    node = _strip_path_wrappers(node)
    if isinstance(node, ast.Name):
        return node.id in summaries
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _is_the_host_output_directory(
            node.left, directories
        ) and _names_the_canonical_summary(node.right, literals)
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    callee = node.func
    if callee.attr == "joinpath" and len(node.args) == 1:
        return _is_the_host_output_directory(
            callee.value, directories
        ) and _names_the_canonical_summary(node.args[0], literals)
    if callee.attr == "join" and len(node.args) == 2:
        return _is_the_host_output_directory(
            node.args[0], directories
        ) and _names_the_canonical_summary(node.args[1], literals)
    return False


def _canonical_summary_names(
    tree: ast.AST,
    directories: Set[str],
    literals: dict[str, Set[str]],
) -> Set[str]:
    """Local names bound to the canonical summary path."""

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    summaries: Set[str] = set()
    for _ in range(len(assignments) + 1):
        grew = False
        for node in assignments:
            target = node.targets[0]
            assert isinstance(target, ast.Name)
            if target.id in summaries:
                continue
            if _denotes_the_canonical_summary(
                node.value,
                directories=directories,
                literals=literals,
                summaries=summaries,
            ):
                summaries.add(target.id)
                grew = True
        if not grew:
            break
    return summaries


def _accessed_key(node: ast.AST) -> Optional[tuple[ast.AST, object]]:
    """The container and key of ``x.get(k)`` / ``x[k]``, if the key is literal."""

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


def _branch_values(node: ast.AST) -> list[ast.AST]:
    """Every value a (possibly conditional) expression can take.

    A script that has to cope with both the mapping and the two-item spellings
    of the range writes the choice inline, so the bound arrives through one arm
    of a conditional expression.  Either arm reads the same contract.
    """

    if isinstance(node, ast.IfExp):
        return [*_branch_values(node.body), *_branch_values(node.orelse)]
    return [node]


def _plausibility_range_and_bound_names(tree: ast.AST) -> tuple[Set[str], Set[str]]:
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
        for node in ast.walk(tree)
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


def _canonical_summary_handles(tree: ast.AST, denotes) -> Set[str]:
    """Names bound to an open handle on the canonical summary."""

    handles: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if isinstance(
                    item.optional_vars, ast.Name
                ) and _opens_the_canonical_summary(item.context_expr, denotes):
                    handles.add(item.optional_vars.id)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _opens_the_canonical_summary(node.value, denotes)
        ):
            handles.add(node.targets[0].id)
    return handles


def _writer_functions(tree: ast.AST) -> Set[str]:
    """Locally defined helpers that write, recognised by body rather than name.

    Generated scripts routinely funnel every artifact through one small helper.
    Recognising it structurally is what lets the writer-name set stay put
    instead of growing a new entry each time a script invents a name for it.
    """

    return {
        function.name
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr in _WRITER_ATTRIBUTES
            for inner in ast.walk(function)
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


def _plausibility_comparisons(tree: ast.Module) -> list[ast.Compare]:
    """Ordering comparisons that test a value against a declared bound.

    Anchored on the bound rather than on the column, which is also what makes
    it work on the shape real generated code writes: a per-column helper takes
    the series as a parameter, so the variable's name appears nowhere near the
    comparison and only the bound identifies the check.
    """

    ranges, bounds = _plausibility_range_and_bound_names(tree)
    if not ranges and not bounds:
        return []
    found: list[ast.Compare] = []
    for node in ast.walk(tree):
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


class _FlagFlow:
    """Where the out-of-range record is computed, and where it ends up.

    Name-level rather than value-level: a fixpoint over assignments, container
    literals and function returns.  That is enough for every compliant shape
    generated code actually writes, and everything it cannot follow ends in a
    block, so its limits cost provider repairs instead of buying passes.
    """

    def __init__(self, tree: ast.Module, comparisons: Sequence[ast.Compare]) -> None:
        self.tree = tree
        self.compared = {id(node) for node in comparisons}
        self.parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        self.carriers: Set[str] = set()
        self.returning_functions: Set[str] = set()
        #: How this script reaches the one artifact the host opens afterwards.
        #: A delivery is a write *to that file*; see `receipt_deliveries`.
        self.output_directories = _host_output_directory_names(tree)
        self.literals = _literal_string_names(tree)
        self.summary_paths = _canonical_summary_names(
            tree, self.output_directories, self.literals
        )
        self.handles = _canonical_summary_handles(tree, self.denotes_the_summary)
        self.writer_functions = _writer_functions(tree)
        self._resolve()
        #: Names holding a mapping that carries the record under the receipt
        #: key.  Reaching the file is not enough on its own: the host reads one
        #: key of it, so a count filed anywhere else in the same summary is a
        #: count the host never sees.
        self.receipt_carriers: Set[str] = self._resolve_receipt_carriers()
        #: The mask and count themselves, without the containers they are
        #: later filed into. A guard on `mask.any()` or on `n_out > 0` is a
        #: rejection; a guard on the summary dict that happens to hold the
        #: count is an unrelated check, and reading the wide carrier set for
        #: both is how an ordinary contract assertion gets reported as a
        #: plausibility rejection.
        self.direct_values: Set[str] = self._resolve_direct_values()

    def _carries(self, node: ast.AST) -> bool:
        for inner in ast.walk(node):
            if id(inner) in self.compared:
                return True
            if (
                isinstance(inner, ast.Name)
                and isinstance(inner.ctx, ast.Load)
                and inner.id in self.carriers
            ):
                return True
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in self.returning_functions
            ):
                return True
        return False

    def _resolve(self) -> None:
        assignments = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        ]
        functions = [
            node
            for node in ast.walk(self.tree)
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
                    if name not in self.carriers:
                        self.carriers.add(name)
                        grew = True
            for function in functions:
                if function.name in self.returning_functions:
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and inner.value is not None
                    and self._carries(inner.value)
                    for inner in ast.walk(function)
                ):
                    self.returning_functions.add(function.name)
                    grew = True
            if not grew:
                break

    def _resolve_direct_values(self) -> Set[str]:
        direct: Set[str] = set()
        assignments = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and not isinstance(node.value, (ast.Dict, ast.List, ast.Set, ast.Tuple))
        ]
        for _ in range(len(assignments) + 1):
            grew = False
            for node in assignments:
                target = node.targets[0].id
                if target in direct:
                    continue
                if any(
                    id(inner) in self.compared
                    or (
                        isinstance(inner, ast.Name)
                        and isinstance(inner.ctx, ast.Load)
                        and inner.id in direct
                    )
                    for inner in ast.walk(node.value)
                ):
                    direct.add(target)
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
                and inner.id in self.direct_values
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

        for node in ast.walk(self.tree):
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

        return _denotes_the_canonical_summary(
            node,
            directories=self.output_directories,
            literals=self.literals,
            summaries=self.summary_paths,
        )

    def _addresses_the_summary(self, node: ast.AST) -> bool:
        return self.denotes_the_summary(node) or (
            isinstance(node, ast.Name) and node.id in self.handles
        )

    def _resolve_receipt_carriers(self) -> Set[str]:
        """Names holding a mapping whose receipt key is the out-of-range record.

        Seeded by the two ways a script files something under a literal key --
        ``summary[key] = record`` and a dict literal -- then propagated the same
        way the carrier set is: through plain assignment and through helpers
        *proven* to return such a mapping.  It deliberately does not propagate
        through nesting: ``{"quality": summary}`` moves the receipt one level
        down, where the host does not look for it.
        """

        names: Set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == RECEIPT_SUMMARY_KEY
                        and self._carries(node.value)
                    ):
                        root = _subscript_root(target)
                        if root is not None:
                            names.add(root)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == RECEIPT_SUMMARY_KEY
                and self._carries(node.args[1])
            ):
                root = _subscript_root(node.func.value)
                if root is not None:
                    names.add(root)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and any(
                    self._is_receipt_mapping(argument, names) for argument in node.args
                )
            ):
                root = _subscript_root(node.func.value)
                if root is not None:
                    names.add(root)

        assignments = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ]
        functions = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        returning: Set[str] = set()
        for _ in range(len(assignments) + len(functions) + 1):
            grew = False
            for node in assignments:
                target = node.targets[0]
                assert isinstance(target, ast.Name)
                if target.id in names:
                    continue
                value = node.value
                if self._is_receipt_mapping(value, names) or (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in returning
                ):
                    names.add(target.id)
                    grew = True
            for function in functions:
                if function.name in returning:
                    continue
                if any(
                    isinstance(inner, ast.Return)
                    and inner.value is not None
                    and self._is_receipt_mapping(inner.value, names)
                    for inner in ast.walk(function)
                ):
                    returning.add(function.name)
                    grew = True
            if not grew:
                break
        return names

    def _is_receipt_mapping(self, node: ast.AST, names: Set[str]) -> bool:
        """Whether an expression *is* a mapping carrying the record at its key."""

        for value in _branch_values(node):
            if isinstance(value, ast.Name) and value.id in names:
                return True
            if isinstance(value, ast.Dict):
                for key, item in zip(value.keys, value.values):
                    if key is None:  # `{**summary}`
                        if isinstance(item, ast.Name) and item.id in names:
                            return True
                    elif (
                        isinstance(key, ast.Constant)
                        and key.value == RECEIPT_SUMMARY_KEY
                        and self._carries(item)
                    ):
                        return True
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "dict"
            ):
                for keyword in value.keywords:
                    if keyword.arg is None:
                        if (
                            isinstance(keyword.value, ast.Name)
                            and keyword.value.id in names
                        ):
                            return True
                    elif keyword.arg == RECEIPT_SUMMARY_KEY and self._carries(
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
        for node in ast.walk(self.tree):
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
        for node in ast.walk(self.tree):
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


def _rejection_findings(
    tree: ast.Module,
    flow: _FlagFlow,
    *,
    step_id: str,
    detail_base: dict,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and flow.rejects_on(node.test)
            and _is_terminal_failure(node.body)
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
    return findings


def flag_only_plausibility_obligation_findings(
    tree: Optional[ast.Module],
    *,
    script_text: str,
    context: ResearchContext,
    step: AnalysisStep,
) -> list[ValidationFinding]:
    """Check both halves of ``retain_and_flag`` without consulting an auditor."""

    text = str(script_text or "")
    if tree is None:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
    # The trigger is the host's typed policy plus the script's use of it, and
    # it is shared with the post-execution receipt check so the two halves of
    # the obligation can never disagree about which steps owe one.
    trigger = step_is_under_the_flag_only_obligation(
        script_text=text,
        tree=tree,
        context=context,
    )
    if trigger is None:
        return []

    step_id = str(step.step_id)
    detail_base = {
        "step_id": step_id,
        "issue_code": "flag_only_plausibility_obligation",
        "policy": "retain_and_flag",
        "policy_authority": "typed_research_context_plausibility_policy",
        "trigger": trigger,
        "ranged_variables": ranged_variable_names(context),
    }

    comparisons = _plausibility_comparisons(tree)
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

    flow = _FlagFlow(tree, comparisons)
    findings = _rejection_findings(
        tree,
        flow,
        step_id=step_id,
        detail_base=detail_base,
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
