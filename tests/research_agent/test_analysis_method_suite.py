"""Drift + consistency tests for the per-family analysis-method suite registry.

These lock the contract that makes the suite registry a trustworthy single source
of truth for "which methods does EasyICU run per family, and how are they
produced": every family is real and known to capability_registry; tiers and
implementation statuses use the closed vocabulary; a ``planned`` method is honest
(no runner, must fail closed); a ``deterministic`` method names a runner that
actually exists; and the generated doc does not drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from easyicu.research_agent import analysis_method_suite as ams
from easyicu.research_agent import capability_registry as cr
from easyicu.research_agent.figures import FAMILY_RENDERERS
from easyicu.research_agent.study_design_playbook import StudyDesignFamily

try:  # StudyDesignFamily is a typing.Literal — recover its allowed values
    _VALID_FAMILIES = frozenset(StudyDesignFamily.__args__)  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - defensive
    _VALID_FAMILIES = frozenset(
        {
            "association",
            "prediction",
            "time_to_event",
            "phenotyping",
            "causal_emulation",
            "descriptive",
        }
    )

_ALL_METHODS = [
    (suite, m) for suite in ams.METHOD_SUITE_REGISTRY for m in suite.methods
]


def test_every_suite_family_is_valid_and_in_capability_registry():
    reg_families = {c.family for c in cr.CAPABILITY_REGISTRY}
    for suite in ams.METHOD_SUITE_REGISTRY:
        assert suite.family in _VALID_FAMILIES, suite.family
        assert suite.family in reg_families, (
            f"{suite.family} has a method suite but no capability_registry record"
        )


def test_no_duplicate_families():
    families = [s.family for s in ams.METHOD_SUITE_REGISTRY]
    assert len(families) == len(set(families)), families


def test_tiers_and_implementations_use_closed_vocabulary():
    for suite, m in _ALL_METHODS:
        assert m.tier in ams.METHOD_TIERS, (suite.family, m.key, m.tier)
        assert m.implementation in ams.METHOD_IMPLEMENTATIONS, (
            suite.family,
            m.key,
            m.implementation,
        )


def test_method_keys_unique_within_suite():
    for suite in ams.METHOD_SUITE_REGISTRY:
        keys = [m.key for m in suite.methods]
        assert len(keys) == len(set(keys)), (suite.family, keys)


def test_every_suite_has_a_primary_method():
    for suite in ams.METHOD_SUITE_REGISTRY:
        primaries = [m for m in suite.methods if m.tier == "primary"]
        assert primaries, f"{suite.family} suite has no primary method"


def test_planned_methods_carry_no_runner():
    # A planned method is recognised but not implemented; claiming a runner would
    # imply it exists. It must fail closed, never be silently approximated.
    for suite, m in _ALL_METHODS:
        if m.tier == "planned" or m.implementation == "planned":
            assert m.runner is None, (
                f"{suite.family}.{m.key} is planned but names a runner {m.runner!r}"
            )


def test_deterministic_methods_name_a_real_runner():
    renderer_keys = set(FAMILY_RENDERERS.keys())
    for suite, m in _ALL_METHODS:
        if m.implementation != "deterministic":
            continue
        assert m.runner, f"{suite.family}.{m.key} is deterministic but has no runner"
        known = (
            m.runner in ams.KNOWN_PRIMARY_RUNNER_NAMES
            or m.runner in renderer_keys
            or importlib.util.find_spec(
                f"easyicu.research_agent.{m.runner}"
            )
            is not None
            or importlib.util.find_spec(
                f"easyicu.research_agent.methods.{m.runner}"
            )
            is not None
        )
        assert known, (
            f"{suite.family}.{m.key} deterministic runner {m.runner!r} is neither a "
            f"capability_registry primary/auxiliary runner, a FAMILY_RENDERERS key, "
            f"nor an importable research_agent methods module"
        )


def test_deterministic_primary_runner_agrees_with_capability_registry():
    # A PRIMARY method claimed deterministic must map to a runner the capability
    # registry actually wires (survival_primary_cox / causal_primary_iptw /
    # ordinal_dose_response), so the two registries cannot disagree on what is
    # deterministically produced as the headline.
    for suite, m in _ALL_METHODS:
        if m.tier == "primary" and m.implementation == "deterministic":
            assert m.runner in ams.KNOWN_PRIMARY_RUNNER_NAMES, (
                f"{suite.family}.{m.key} claims a deterministic primary but runner "
                f"{m.runner!r} is not a capability_registry primary/auxiliary runner"
            )


def test_competing_risks_stays_a_declared_planned_boundary():
    # The single most dangerous "nearby estimand" substitution (a Cox HR sold as a
    # competing-risks CIF) must remain honest: present, planned, no runner, and
    # consistent with the capability_registry KNOWN_UNSUPPORTED boundary.
    survival = ams.get_suite("time_to_event")
    assert survival is not None
    cif = [m for m in survival.methods if "competing_risks" in m.key]
    assert cif, "competing-risks CIF must be declared in the survival suite"
    for m in cif:
        assert m.tier == "planned" and m.implementation == "planned"
        assert m.runner is None
    assert any(
        "competing" in name.lower() for name, _why in cr.KNOWN_UNSUPPORTED_ESTIMANDS
    ), "competing-risks CIF must also be a capability_registry KNOWN_UNSUPPORTED estimand"


def test_trajectory_is_present_and_not_mislabelled_lcga():
    # WS3: the deterministic trajectory path is trajectory-FEATURE clustering and
    # must NOT be labelled LCGA; LCGA/GBTM is a separate PLANNED method.
    pheno = ams.get_suite("phenotyping")
    assert pheno is not None
    keys = {m.key for m in pheno.methods}
    assert "trajectory_feature_clustering" in keys
    assert "lcga_gbtm" in keys
    traj = next(m for m in pheno.methods if m.key == "trajectory_feature_clustering")
    assert "lcga" not in traj.name.lower(), "feature clustering must not be called LCGA"
    lcga = next(m for m in pheno.methods if m.key == "lcga_gbtm")
    assert lcga.tier == "planned" and lcga.implementation == "planned"


def test_accessors_and_roadmap_nonempty():
    assert ams.get_suite("time_to_event") is not None
    assert ams.get_suite("not_a_family") is None  # type: ignore[arg-type]
    assert ams.methods_by_tier("prediction", "primary")
    assert ams.supporting_methods("prediction")  # reviewer-expected depth set
    assert ams.planned_methods(), "there should be a declared planned roadmap"
    assert ams.deterministic_methods(), "some methods are deterministic today"


def test_docs_analysis_method_suite_matches_registry():
    # docs/analysis_method_suite.md is generated from the registry; regenerate with:
    #   python -m easyicu.research_agent.analysis_method_suite > docs/analysis_method_suite.md
    repo_root = Path(__file__).resolve().parents[2]
    doc = repo_root / "docs" / "analysis_method_suite.md"
    assert doc.exists(), "docs/analysis_method_suite.md missing — regenerate it"
    committed = doc.read_text(encoding="utf-8").rstrip("\n")
    rendered = ams.render_method_suite_markdown().rstrip("\n")
    assert committed == rendered, "docs/analysis_method_suite.md is stale — regenerate it"
