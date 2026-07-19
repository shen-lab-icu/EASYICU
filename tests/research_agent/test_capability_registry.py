"""The capability registry must match the code that is actually wired.

The registry (``easyicu.research_agent.capability_registry``) is only useful if
it cannot lie. These tests cross-check every claim against the live pipeline:
the deterministic-runner names against ``_PRIMARY_DETERMINISTIC_RUNNERS`` in
BOTH pipeline modules, the figure-renderer keys against
``figures.FAMILY_RENDERERS``, the declared runner entrypoints against the
importable modules, and family coverage against the ``StudyDesignFamily`` enum.
Add or remove a runner without updating the registry and one of these fails —
which is the point.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import textwrap
from typing import get_args

from easyicu.research_agent import capability_registry as cr
from easyicu.research_agent import pipeline_execute
from easyicu.research_agent.figures import FAMILY_RENDERERS
from easyicu.research_agent.pipeline_execute import (
    _PRIMARY_DETERMINISTIC_RUNNERS as EXEC_RUNNERS,
)
from easyicu.research_agent.pipeline_report import (
    _PRIMARY_DETERMINISTIC_RUNNERS as REPORT_RUNNERS,
)
from easyicu.research_agent.study_design_playbook import StudyDesignFamily

# runner name -> (module, code-string entrypoint) for the importability check
_RUNNER_ENTRYPOINTS = {
    "survival_primary_cox": (
        "execution.runners.deterministic_survival",
        "survival_primary_analysis_code",
    ),
    "causal_primary_iptw": (
        "execution.runners.deterministic_causal",
        "causal_primary_analysis_code",
    ),
    "ordinal_dose_response": (
        "execution.runners.deterministic_ordinal",
        "ordinal_dose_response_analysis_code",
    ),
}


def _registry_primary_runners() -> set:
    return {
        c.primary_runner
        for c in cr.CAPABILITY_REGISTRY
        if c.primary_analysis == "deterministic" and c.primary_runner
    }


# --- deterministic primary runners: registry <-> wired sets ----------------


def test_every_registry_runner_is_wired_in_both_pipeline_modules():
    for name in _registry_primary_runners():
        assert name in EXEC_RUNNERS, f"{name} not wired in pipeline_execute"
        assert name in REPORT_RUNNERS, f"{name} not wired in pipeline_report"


def test_every_wired_runner_is_documented_in_the_registry():
    # No wired deterministic primary runner may be undocumented.
    documented = _registry_primary_runners()
    for name in EXEC_RUNNERS:
        assert name in documented, f"wired runner {name} missing from the registry"


def test_the_two_pipeline_modules_agree_on_the_runner_set():
    assert set(EXEC_RUNNERS) == set(REPORT_RUNNERS)


def test_registry_runner_entrypoints_are_importable_and_callable():
    for name in _registry_primary_runners():
        assert name in _RUNNER_ENTRYPOINTS, f"no entrypoint mapping for {name}"
        mod_name, fn_name = _RUNNER_ENTRYPOINTS[name]
        mod = importlib.import_module(f"easyicu.research_agent.{mod_name}")
        fn = getattr(mod, fn_name)
        code = fn()
        assert isinstance(code, str) and len(code) > 200


# --- deterministic figure renderers ----------------------------------------


def test_registry_figure_renderers_exist_in_family_renderers():
    for c in cr.CAPABILITY_REGISTRY:
        if c.figure != "deterministic" or not c.figure_renderer:
            continue
        # the base association skill is rendered outside FAMILY_RENDERERS
        if c.figure_renderer == "base_association_skill":
            continue
        assert (
            c.figure_renderer in FAMILY_RENDERERS
        ), f"{c.figure_renderer} not in FAMILY_RENDERERS"


# --- auxiliary runners are importable --------------------------------------


def test_auxiliary_runner_entrypoints_are_importable():
    for a in cr.AUXILIARY_DETERMINISTIC_RUNNERS:
        mod = importlib.import_module(f"easyicu.research_agent.{a.module}")
        fn = getattr(mod, a.entrypoint)
        assert callable(fn)


# --- family coverage --------------------------------------------------------


def test_every_study_design_family_is_covered():
    families = set(get_args(StudyDesignFamily))
    covered = {c.family for c in cr.CAPABILITY_REGISTRY}
    missing = families - covered
    assert not missing, f"families with no capability record: {missing}"


def test_partition_helpers_are_consistent():
    det = set(cr.deterministic_primary_families())
    llm = set(cr.llm_coded_primary_families())
    assert det == set(), "primary scientific analyses must remain agent-owned"
    assert llm
    assert det.isdisjoint(llm)
    assert len(det) + len(llm) == len(cr.CAPABILITY_REGISTRY)


def test_every_family_is_without_a_deterministic_primary_owner():
    fams = cr.families_without_deterministic_primary()
    assert fams == set(get_args(StudyDesignFamily))


# --- renderer ---------------------------------------------------------------


def test_markdown_matrix_renders_every_family_and_the_ladder():
    md = cr.render_capability_matrix_markdown()
    for c in cr.CAPABILITY_REGISTRY:
        assert c.label in md
    for name in EXEC_RUNNERS:
        assert name in md
    assert "Fail-closed / gap-report ladder" in md
    # the invariant sentence must be present
    assert "never silently filled" in md


def test_known_unsupported_boundary_is_recorded_and_rendered():
    # An explicit "not supported" boundary (competing-risks CIF) must be
    # first-class in the registry, not only a benchmark probe.
    assert cr.KNOWN_UNSUPPORTED_ESTIMANDS
    md = cr.render_capability_matrix_markdown()
    assert "Known unsupported estimands" in md
    assert "Competing-risks" in md


def test_get_capability_disambiguates_association():
    dose = cr.get_capability("association", dose_response=True)
    general = cr.get_capability("association", dose_response=False)
    assert dose is not None and dose.primary_runner is None
    assert general is not None and general.primary_runner is None
    assert "graded ordinal" in dose.label.lower()
    assert "general" in general.label.lower()


def test_live_auxiliary_dispatch_matches_registry_in_both_directions():
    """Inspect actual execute assignments, not a second hand-maintained set."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(pipeline_execute.run_execute_phase)))
    active: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            key = target.slice
            if isinstance(key, ast.Constant) and key.value == "deterministic_standard_analysis":
                active.add(node.value.value)
    documented = {runner.name for runner in cr.AUXILIARY_DETERMINISTIC_RUNNERS}
    assert active == documented


# --- generated doc stays in sync -------------------------------------------


def test_committed_docs_matrix_matches_the_registry_render():
    # docs/capability_matrix.md is generated from the registry; if it drifts,
    # regenerate with:
    #   python -m easyicu.research_agent.capability_registry > docs/capability_matrix.md
    repo_root = Path(__file__).resolve().parents[2]
    doc = repo_root / "docs" / "capability_matrix.md"
    assert doc.exists(), "docs/capability_matrix.md missing — regenerate it"
    committed = doc.read_text(encoding="utf-8").rstrip("\n")
    rendered = cr.render_capability_matrix_markdown().rstrip("\n")
    assert committed == rendered, "docs/capability_matrix.md is stale — regenerate it"
