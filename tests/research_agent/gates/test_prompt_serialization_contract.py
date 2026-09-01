"""The host may not ask the Coder to serialize a value its own format cannot hold.

On the 2026-07-30 E1 run two of eleven steps died here, and neither was a science
failure: both had finished their analysis and died writing their own report.  The
Coder prompt named ``audit_publication_exports`` and, in the same sentence, said
to save its findings in ``step_summary.json`` where "JSON values must be Python
primitives" -- but that helper returns ``ValidationFinding`` models.  The host's
own mock provider carried a private ``finding_to_dict`` workaround, which is how
the host proved it knew the conversion was required while never publishing it.
Both Coders invented the converter and both got it wrong, differently.

The rule below is the general form: **a guidance line that routes a value into**
``step_summary.json`` **may only name host helpers whose return annotation is
JSON-primitive-safe.**  It is deliberately anchored on the prompt text plus the
helpers' real annotations rather than on a list of helper names, so a helper
added later cannot re-open the hole without failing this test.
"""

from __future__ import annotations

import ast
import collections.abc
import importlib
import re
import types
import typing
from pathlib import Path
from typing import Any, get_args, get_origin

import pytest


PROMPT_SCOPE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "research_agent"
    / "research_context"
    / "prompt_scope.py"
)

_JSON_PRIMITIVES = (str, int, float, bool, type(None))
_SEQUENCE_ORIGINS = (
    list,
    set,
    frozenset,
    tuple,
    collections.abc.Sequence,
    collections.abc.Iterable,
)
_MAPPING_ORIGINS = (dict, collections.abc.Mapping)


def _guidance_blocks() -> dict[str, str]:
    """Each module-level guidance string in prompt_scope.py, kept separate.

    Blocks must stay separate, not concatenated.  The defect this file exists
    for spans two bullets of one block -- "call this helper" and "put it in
    step_summary.json" -- so a check that scans the file as one blob lets a
    healthy block satisfy the guard for a broken one.  That is exactly how the
    first version of this test passed its own mutation.
    """
    tree = ast.parse(
        PROMPT_SCOPE.read_text(encoding="utf-8"), filename=str(PROMPT_SCOPE)
    )
    blocks: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    blocks[target.id] = value.value
    assert blocks, "prompt_scope.py exposes no module-level guidance strings"
    return blocks


def _guidance_text() -> str:
    return "\n".join(_guidance_blocks().values())


def _named_helper_modules(text: str) -> list[types.ModuleType]:
    """Import every ``easyicu.research_agent.*`` module the guidance names.

    Deriving the module set from the prompt itself is what keeps this test
    honest: pointing the Coder at a new host module automatically brings that
    module's helpers under the rule.
    """
    modules: dict[str, types.ModuleType] = {}
    for dotted in sorted(set(re.findall(r"easyicu\.research_agent[\w\.]*", text))):
        candidate = dotted.rstrip(".")
        try:
            modules[candidate] = importlib.import_module(candidate)
        except ImportError:
            # The prompt also names symbols as ``module.function``; drop the
            # trailing attribute and retry the module itself.
            head, _, _tail = candidate.rpartition(".")
            if head and head not in modules:
                try:
                    modules[head] = importlib.import_module(head)
                except ImportError:
                    continue
    assert modules, "no host helper module is named anywhere in the Coder guidance"
    return list(modules.values())


def _is_json_primitive_safe(annotation: Any) -> bool:
    if annotation is Any or annotation is Ellipsis:
        return True
    if annotation in _JSON_PRIMITIVES:
        return True

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (typing.Union, types.UnionType):
        return all(_is_json_primitive_safe(arg) for arg in args)
    if origin in _MAPPING_ORIGINS:
        return bool(args) and all(_is_json_primitive_safe(arg) for arg in args)
    if origin in _SEQUENCE_ORIGINS:
        return bool(args) and all(_is_json_primitive_safe(arg) for arg in args)
    return False


def _host_helpers_named(
    line: str, modules: list[types.ModuleType]
) -> list[tuple[str, Any]]:
    resolved: list[tuple[str, Any]] = []
    for name in re.findall(r"`([A-Za-z_]\w*)\(?", line):
        for module in modules:
            helper = getattr(module, name, None)
            if callable(helper) and getattr(helper, "__module__", "").startswith(
                "easyicu."
            ):
                resolved.append((f"{module.__name__}.{name}", helper))
                break
    return resolved


def _summary_routing_helpers() -> list[tuple[str, str, Any]]:
    """(block, qualified name, helper) for helpers named on a step_summary line."""
    modules = _named_helper_modules(_guidance_text())
    resolved: list[tuple[str, str, Any]] = []
    for block, text in _guidance_blocks().items():
        for line in text.splitlines():
            if "step_summary" not in line:
                continue
            for qualified_name, helper in _host_helpers_named(line, modules):
                resolved.append((block, qualified_name, helper))
    return resolved


def test_every_step_summary_routing_block_names_its_producing_helper() -> None:
    """A block may not route a host-produced value into step_summary.json by prose.

    This is the assertion the first version of this file was missing, and the
    reason its mutation run passed a wording that had already cost two steps.
    The fresh28 prompt described the value on the routing bullet ("publication
    export-QA findings") while naming the helper that produces it three bullets
    earlier, so no single line carried both facts and there was nothing left to
    check.  Requiring the producing call on the routing line is what makes the
    return-type check below able to see anything at all.
    """
    modules = _named_helper_modules(_guidance_text())

    offenders: list[str] = []
    for block, text in _guidance_blocks().items():
        lines = text.splitlines()
        routing = [line for line in lines if "step_summary" in line]
        if not routing:
            continue
        # A block that hands the Coder no host helper at all routes only its own
        # literals and has no producer to name.
        if not any(_host_helpers_named(line, modules) for line in lines):
            continue
        if not any(_host_helpers_named(line, modules) for line in routing):
            offenders.append(block)

    assert offenders == [], (
        "These guidance blocks tell the Coder to put a host-produced value in "
        "step_summary.json but name the producing helper only on some other "
        "line, so the Coder is left to guess how to serialize it: "
        + ", ".join(offenders)
    )


def test_summary_routing_helpers_return_json_primitives() -> None:
    helpers = _summary_routing_helpers()

    # Guard against the check quietly becoming vacuous: if a future prompt edit
    # stops naming any host helper next to step_summary.json, that is itself a
    # change worth failing on rather than silently passing zero cases.
    assert helpers, (
        "no host helper is named on a step_summary.json guidance line -- either "
        "the prompt changed shape or this test stopped resolving names"
    )

    offenders: list[str] = []
    for block, qualified_name, helper in helpers:
        hints = typing.get_type_hints(helper)
        annotation = hints.get("return", Any)
        if not _is_json_primitive_safe(annotation):
            offenders.append(f"{block}: {qualified_name} -> {annotation!r}")

    assert offenders == [], (
        "The Coder prompt routes these helpers' output into step_summary.json "
        "under a Python-primitives contract, but they return host types the "
        "Coder is never taught to serialize: " + "; ".join(offenders)
    )


def test_publication_export_audit_has_a_published_json_form() -> None:
    """The typed audit keeps its typed return; the Coder gets a primitive one.

    ``audit_publication_exports`` cannot simply start returning dicts: five host
    call sites in ``figures/skill.py`` merge its findings into the typed
    findings pipeline.  The fix is a second published entry point, not a repair
    rule that patches whatever the Coder invented.
    """
    publication = importlib.import_module("easyicu.research_agent.figures.publication")

    typed_return = typing.get_type_hints(publication.audit_publication_exports)[
        "return"
    ]
    json_return = typing.get_type_hints(publication.audit_publication_exports_json)[
        "return"
    ]

    assert not _is_json_primitive_safe(typed_return)
    assert _is_json_primitive_safe(json_return)
    assert "audit_publication_exports_json" in publication.__all__


def test_json_safety_checker_separates_the_two_real_return_types() -> None:
    """Mutation guard for the checker itself, using the two live annotations."""
    from easyicu.research_agent.schema import ValidationFinding

    assert _is_json_primitive_safe(list[dict[str, Any]])
    assert _is_json_primitive_safe(dict[str, Any])
    assert not _is_json_primitive_safe(list[ValidationFinding])
    assert not _is_json_primitive_safe(ValidationFinding)
    assert not _is_json_primitive_safe(Path)
    assert not _is_json_primitive_safe(dict[str, Path])


@pytest.mark.parametrize(
    "forbidden",
    ["finding_to_dict"],
)
def test_the_private_mock_workaround_is_gone(forbidden: str) -> None:
    """The converter must live in published API, not inside the host's own mock.

    A workaround in the mock provider is the tell that the host knows its
    published contract is unsatisfiable; leaving it there lets the hole reopen
    while every mock-backed test keeps passing.
    """
    mocks = (PROMPT_SCOPE.parents[1] / "providers" / "mocks.py").read_text(
        encoding="utf-8"
    )
    assert forbidden not in mocks
