"""No statistical kernel may exist without a route that reaches it.

Six modules under ``research_agent/methods`` -- 1,430 lines, 49 passing tests --
had zero importers outside their own test files, for months, while five of them
were simultaneously declared ``implementation="planned"`` to the Planner.

The cause was that every existing drift guard reads in one direction.
``test_analysis_method_suite.py`` iterates the *map* and checks the code backs
it up, which catches claiming MORE than we have. Nothing iterated the *code* and
checked the map mentions it, so claiming LESS -- dead code -- could not be
caught by construction.

This file is the other direction. Every module under ``methods/`` must be
reachable by exactly one declared route:

* the host imports it from non-test code in ``src/``; or
* it is offered to the Coder via ``CURATED_METHOD_KERNELS``; or
* it is explicitly declared unreachable with a written reason and a named
  pending decision.

Anything else fails here, naming the module. That is the maintenance mechanism:
not a document to remember, a test that goes red.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from easyicu.research_agent.contracts.method_kernels import (
    CURATED_METHOD_KERNELS,
    DECLARED_UNREACHABLE_KERNELS,
    KERNEL_MODULE_NAMES,
    UNREACHABLE_MODULE_NAMES,
)

_SRC = Path(__file__).resolve().parents[2] / "src"
_METHODS_DIR = _SRC / "easyicu" / "research_agent" / "methods"


def _kernel_modules() -> tuple[str, ...]:
    return tuple(
        sorted(
            path.stem for path in _METHODS_DIR.glob("*.py") if path.stem != "__init__"
        )
    )


def _modules_imported_from_src() -> frozenset[str]:
    """Kernel module names imported by non-test code under ``src/``.

    Read from the AST rather than by grepping text: a module name appearing in
    a docstring, a comment, or a string literal is not an import, and a guard
    that counts those would report a dead module as live -- the exact direction
    of error this file exists to prevent.
    """

    live: set[str] = set()
    kernels = set(_kernel_modules())
    for path in _SRC.rglob("*.py"):
        if path.is_relative_to(_METHODS_DIR):
            continue  # a kernel importing a sibling does not make it reachable
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # ``node.module`` carries no leading dots for a relative import:
                # ``from ..methods.table_one import x`` arrives as
                # "methods.table_one" with level=2. Matching on a dotted
                # substring therefore misses every relative import in the
                # package -- which is most of them. Match on path COMPONENTS.
                parts = (node.module or "").split(".")
                if "methods" not in parts:
                    continue
                index = parts.index("methods")
                if index + 1 < len(parts) and parts[index + 1] in kernels:
                    live.add(parts[index + 1])  # from ...methods.<kernel> import x
                elif index + 1 == len(parts):
                    for alias in node.names:  # from ...methods import <kernel>
                        if alias.name in kernels:
                            live.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if "methods" not in parts:
                        continue
                    index = parts.index("methods")
                    if index + 1 < len(parts) and parts[index + 1] in kernels:
                        live.add(parts[index + 1])
    return frozenset(live)


# ---------------------------------------------------------------------------
# code -> map: nothing may be dead
# ---------------------------------------------------------------------------


def test_every_kernel_module_is_reachable_by_a_declared_route():
    host_imported = _modules_imported_from_src()
    dead = [
        module
        for module in _kernel_modules()
        if module not in host_imported
        and module not in KERNEL_MODULE_NAMES
        and module not in UNREACHABLE_MODULE_NAMES
    ]
    assert not dead, (
        "these methods/ modules are reachable by nothing -- the host does not "
        "import them, they are not offered to the Coder, and they are not "
        f"declared unreachable: {dead}. Wire one, offer it in "
        "CURATED_METHOD_KERNELS, or declare it in DECLARED_UNREACHABLE_KERNELS "
        "with a reason."
    )


def test_the_guard_actually_sees_the_dead_modules_it_was_written_for():
    """Reachability of the guard itself.

    If ``_modules_imported_from_src`` over-reported, every module would look
    live and the guard above would pass vacuously forever. These six are the
    measured dead set on 2026-07-30; they must be invisible to the host-import
    scan, or this file is checking nothing.
    """

    host_imported = _modules_imported_from_src()
    for module in (
        "delong_auc",
        "decision_curve",
        "ph_schoenfeld",
        "rmst",
        "conformal",
        "evalue",
        # Found by this guard, not by the hand survey that preceded it: a
        # loose grep reported it as live. It is the most consequential of the
        # set -- it exists because generated code kept misreading a
        # "measured but event-absent" row as an event onset.
        "temporal_features",
    ):
        assert module not in host_imported, (
            f"{module} now has a host importer -- if that is a real wiring, "
            "move it out of this list; if the scan is over-reporting, the "
            "reachability guard has stopped protecting anything"
        )


def test_the_scan_does_see_a_module_the_host_really_imports():
    """The negative control for the test above.

    A scan that returned the empty set would also pass every assertion in
    ``test_the_guard_actually_sees_the_dead_modules_it_was_written_for``.
    """

    host_imported = _modules_imported_from_src()
    for module in ("multiple_testing", "sensitivity", "table_one", "fairness"):
        assert module in host_imported, (
            f"{module} is imported by src/ but the scan missed it -- the "
            "reachability scan under-reports and would flag live code as dead"
        )


# ---------------------------------------------------------------------------
# map -> code: nothing offered may be a lie
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", CURATED_METHOD_KERNELS, ids=lambda k: k.module)
def test_every_offered_kernel_imports_and_has_its_entrypoints(kernel):
    module = importlib.import_module(kernel.import_path)
    missing = [name for name in kernel.entrypoints if not hasattr(module, name)]
    assert not missing, (
        f"{kernel.import_path} is offered to the Coder but is missing "
        f"{missing}; the Coder would be told to call something that does not "
        "exist"
    )


@pytest.mark.parametrize("kernel", CURATED_METHOD_KERNELS, ids=lambda k: k.module)
def test_every_offered_kernel_declares_a_capability_and_a_fallback(kernel):
    assert kernel.entrypoints, f"{kernel.module} offers no entrypoint"
    assert kernel.capability.strip(), f"{kernel.module} declares no capability"
    assert kernel.families, f"{kernel.module} declares no analysis family"
    # A kernel with no stated fallback silently becomes a hard dependency: the
    # Coder cannot tell whether losing it degrades the analysis or ends it.
    assert kernel.fallback.strip(), f"{kernel.module} declares no fallback"


def test_offered_kernels_are_distinct_and_real_files():
    modules = [k.module for k in CURATED_METHOD_KERNELS]
    assert len(modules) == len(set(modules)), f"duplicate kernel entries: {modules}"
    for module in modules:
        assert (_METHODS_DIR / f"{module}.py").is_file(), f"no such kernel: {module}"


# ---------------------------------------------------------------------------
# the third state must stay expensive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", DECLARED_UNREACHABLE_KERNELS, ids=lambda e: e.module)
def test_declared_unreachable_kernels_carry_a_reason_and_a_decision(entry):
    assert (_METHODS_DIR / f"{entry.module}.py").is_file()
    # Without both fields this list degrades into "wire it later", which is the
    # state that produced 1,430 lines of dead code in the first place.
    assert len(entry.reason.strip()) > 40, f"{entry.module}: reason is not a reason"
    assert (
        len(entry.pending_decision.strip()) > 40
    ), f"{entry.module}: no named pending decision"


def test_a_module_cannot_be_both_offered_and_declared_dead():
    overlap = KERNEL_MODULE_NAMES & UNREACHABLE_MODULE_NAMES
    assert not overlap, f"declared both reachable and unreachable: {sorted(overlap)}"


# ---------------------------------------------------------------------------
# being offered is not the same as being SELECTED: max_software is 3
# ---------------------------------------------------------------------------

_RUNTIME = (
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "sklearn",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "lifelines",
    "shap",
    "xgboost",
)


def _coder_prompt(*, analysis_family: str, intent: str, method: str) -> str:
    """Run the host's OWN bundle builder, so the real policy is exercised.

    Rebuilding the policy here would test a policy nobody uses -- a first
    attempt at this measurement invented one whose review statuses excluded
    every descriptor, and it returned an empty selection that would have read
    as "the kernels are never chosen".
    """

    from easyicu.research_agent.resources.coder import build_coder_resource_bundle

    bundle = build_coder_resource_bundle(
        step_id="01_probe",
        profile_ref="test/profile@1",
        analysis_family=analysis_family,
        step_role="primary",
        question="Does the exposure change the outcome?",
        intent=intent,
        method=method,
        planner_inputs=("cohort",),
        expected_outputs=("table:result",),
        resolved_input_bindings={},
        runtime_import_names=_RUNTIME,
    )
    return bundle.prompt_projection


def _required_packages_of(prompt: str, import_suffix: str) -> list[str] | None:
    """The parsed ``requires`` of one selected resource, or None if absent.

    The prompt is newline-separated canonical JSON sections. Reading the field
    is the point: a substring search cannot tell a declared dependency from the
    same name mentioned in neighbouring prose.
    """

    import json

    for section in prompt.splitlines():
        section = section.strip()
        if not section.startswith("{"):
            continue
        try:
            payload = json.loads(section)
        except json.JSONDecodeError:
            continue
        for resource in payload.get("resources") or []:
            if str(resource.get("import_name", "")).endswith(import_suffix):
                requires = resource.get("requires")
                return list(requires) if requires is not None else None
    return None


def test_a_survival_step_is_offered_the_ph_kernel_and_the_library_it_wraps():
    """Only three software resources are selected per step.

    Measured before ``requires`` travelled inside the projection: a Cox step
    selected ph_schoenfeld / rmst / temporal_features and ranked *lifelines*
    out entirely -- the Coder got the assumption-check wrapper but not the
    library needed to fit the model at all. Offering a wrapper while hiding
    what it wraps is worse than offering neither.
    """

    prompt = _coder_prompt(
        analysis_family="time_to_event",
        intent=(
            "Fit a Cox proportional hazards model and check the proportional "
            "hazards assumption using Schoenfeld residuals."
        ),
        method="cox_proportional_hazards",
    )
    assert "methods.ph_schoenfeld" in prompt
    # NOT `"lifelines" in prompt`. That first version SURVIVED its own
    # mutation: deleting the requires projection entirely left the test green,
    # because "lifelines" also appears inside the kernel's fallback prose. A
    # substring shared with an incidental mention proves nothing about the
    # field under test. Anchor on what is EXCLUSIVE to it -- the parsed
    # requires list of the selected kernel.
    required = _required_packages_of(prompt, "methods.ph_schoenfeld")
    assert required is not None, (
        "ph_schoenfeld reached the prompt without a requires field -- the Coder "
        "is told to use a lifelines wrapper without being told about lifelines"
    )
    assert "lifelines" in required, (
        f"the Cox step's kernel declares requires={required}, so the library it "
        "wraps is not travelling with it"
    )


def test_a_plain_association_step_is_not_handed_the_trajectory_kernel():
    """A family list is a relevance claim, and over-claiming costs a slot.

    Measured with ``association`` in temporal_features' family list: it ranked
    FIRST for "fit an adjusted logistic regression", above statsmodels, on a
    step where trajectory timing is not the question.
    """

    prompt = _coder_prompt(
        analysis_family="association",
        intent="Fit an adjusted logistic regression of the exposure on the outcome.",
        method="adjusted_association_models",
    )
    assert "methods.temporal_features" not in prompt
    assert (
        "statsmodels" in prompt
    ), "the tool this step actually needs was ranked out of its three slots"


# ---------------------------------------------------------------------------
# the offer must actually reach the Coder
# ---------------------------------------------------------------------------


def test_offered_kernels_reach_the_coder_resource_selector():
    """A declaration nobody projects is a comment.

    ``_software_resources`` is what becomes the Coder's software authority; if
    the kernels were declared but never projected there, the Planner-facing
    registry would claim availability the Coder never sees.
    """

    from easyicu.research_agent.resources import coder as coder_resources

    descriptors = coder_resources._software_resources(runtime_import_names=())
    offered = {
        str(d.prompt_projection)
        for d in descriptors
        if "research_agent.methods." in str(d.prompt_projection)
    }
    for kernel in CURATED_METHOD_KERNELS:
        assert any(kernel.import_path in text for text in offered), (
            f"{kernel.import_path} is declared in CURATED_METHOD_KERNELS but "
            "never reaches the Coder resource projection"
        )
