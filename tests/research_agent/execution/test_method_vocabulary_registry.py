"""Cross-registry drift gate for the deterministic-capability method strings.

One capability chain spans four registries: the standard-executor selection
contract, the execution-phase compact prompt set, the sealed-renderer planner
methods in ``repair_registry``, and the renderer's own CONTROLLED_METHOD.
Historically each retyped the literal, and one copy drifting produced the
recurring per-family gate blockers. ``planning/method_vocabulary.py`` is now
the single definition point; this test fails when a registry retypes a
literal or the layers disagree.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import easyicu.research_agent as research_agent
from easyicu.research_agent import repair_registry
from easyicu.research_agent.execution import phase
from easyicu.research_agent.execution.runners.deterministic_missingness import (
    source_availability_audit_executor_owns_step,
)
from easyicu.research_agent.planning import method_vocabulary
from easyicu.research_agent.schema import AnalysisStep

PACKAGE_ROOT = Path(research_agent.__file__).resolve().parent

_VOCABULARY = {
    name: getattr(method_vocabulary, name) for name in method_vocabulary.__all__
}


def test_vocabulary_constants_are_normalised_method_heads() -> None:
    for name, value in _VOCABULARY.items():
        assert re.fullmatch(r"[a-z0-9_]+", value), (name, value)
        assert (
            "_with_" not in value
        ), f"{name} must hold the method head without a rider: {value!r}"


def test_sealed_renderer_planner_methods_come_from_the_vocabulary() -> None:
    known = set(_VOCABULARY.values())
    for repair_id, methods in repair_registry._SEALED_RENDERER_PLANNER_METHODS.items():
        for method in methods:
            assert method in known, (
                f"{repair_id} names Planner method {method!r} that is not in "
                "planning/method_vocabulary.py — add the constant there and "
                "reference it instead of retyping the literal."
            )


def test_renderer_controlled_methods_match_their_registry_entry() -> None:
    for (
        repair_id,
        modules,
    ) in repair_registry._SEALED_RENDERER_IMPLEMENTATION_MODULES.items():
        declared = repair_registry._SEALED_RENDERER_PLANNER_METHODS.get(repair_id, ())
        for module_name in modules:
            if not module_name.startswith("easyicu.research_agent.figures."):
                continue
            module = importlib.import_module(module_name)
            controlled = getattr(module, "CONTROLLED_METHOD", None)
            if controlled is None:
                continue
            assert controlled in declared or not declared, (
                f"{module_name}.CONTROLLED_METHOD={controlled!r} is not in "
                f"repair_registry planner methods for {repair_id}: {declared}"
            )
            compact = getattr(module, "COMPACT_CONTROLLED_METHOD", None)
            if compact is not None:
                assert compact in declared, (
                    f"{module_name}.COMPACT_CONTROLLED_METHOD={compact!r} is "
                    f"not registered for {repair_id}: {declared}"
                )


def test_missingness_chain_agrees_end_to_end() -> None:
    method = method_vocabulary.MISSINGNESS_SOURCE_AVAILABILITY_AUDIT
    assert method in phase._COMPACT_MISSINGNESS_METHODS
    step = AnalysisStep(
        step_id="02_audit",
        planned_analysis_role="auxiliary",
        intent="Audit measurement availability for every declared input.",
        inputs=["artifact:analysis_cohort", "lact_first", "lact_measured"],
        expected_outputs=[
            "table:missingness_audit",
            "table:measurement_source_audit",
        ],
        method=method,
        icu_rule_refs=[],
    )
    assert source_availability_audit_executor_owns_step(step)


def test_registries_do_not_retype_vocabulary_literals() -> None:
    """The four capability-chain registries must reference the constants."""

    guarded_files = (
        "repair_registry.py",
        "figures/missingness_source.py",
        "figures/distribution_availability.py",
        "figures/absolute_risk.py",
        "figures/continuous_measurement_audit.py",
        "execution/runners/deterministic_missingness.py",
    )
    for relative in guarded_files:
        source = (PACKAGE_ROOT / relative).read_text(encoding="utf-8")
        for name, value in _VOCABULARY.items():
            for quoted in (f'"{value}"', f"'{value}'"):
                assert quoted not in source, (
                    f"{relative} retypes {name} as a literal; import it from "
                    "planning/method_vocabulary.py instead."
                )
