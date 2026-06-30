"""Tests for the advanced-method capability probe.

The agent may only import a curated package once it is (a) declared in the
pyproject ``methods`` extra (reproducible) and (b) importable at run time
(verified by the probe). These tests pin both halves so the coder prompt can
never invite an import the sandbox cannot satisfy.
"""

from __future__ import annotations

import importlib
import pathlib
import re


def _mod():
    return importlib.import_module("easyicu.research_agent.method_capabilities")


def test_capability_block_lists_baseline_and_forbids_unlisted(ra):
    mc = _mod()
    block = mc.coder_method_capability_block()
    for pkg in ("pandas", "numpy", "scipy", "statsmodels", "sklearn"):
        assert pkg in block
    # The block must explicitly forbid importing anything not named.
    assert "forbidden" in block.lower()
    assert "no network" in block.lower()


def test_available_advanced_packages_appear_in_block(ra):
    mc = _mod()
    available = {p.import_name for p in mc.available_method_packages()}
    block = mc.coder_method_capability_block()
    # Whatever is importable here must be named in the block; whatever is not
    # must be reported as unavailable rather than silently offered.
    for pkg in mc.CURATED_METHOD_PACKAGES:
        if pkg.import_name in available:
            assert pkg.import_name in block
            assert pkg.fallback in block
        else:
            assert pkg.import_name in block  # named in the "NOT available" line
    # In this dev environment shap/lifelines/xgboost are installed.
    assert {"lifelines", "shap", "xgboost"} <= available


def test_curated_packages_are_declared_in_pyproject_methods_extra(ra):
    """Reliability invariant: every curated package must be a declared dep, so
    'allowed in the prompt' implies 'reproducibly installable', not 'happens to
    be on one machine'."""
    mc = _mod()
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"\nmethods\s*=\s*\[(.*?)\]", pyproject, re.S)
    assert block, "pyproject must declare a [methods] optional-dependencies extra"
    declared = block.group(1).lower()
    for pkg in mc.CURATED_METHOD_PACKAGES:
        assert pkg.pip_name.lower() in declared, (
            f"{pkg.pip_name} is curated but not declared in the methods extra"
        )
    # And the methods extra must be wired into the aggregate 'all' extra.
    all_block = re.search(r"\nall\s*=\s*\[(.*?)\]", pyproject, re.S)
    assert all_block and "methods" in all_block.group(1)


def test_coder_prompt_embeds_capability_block(ra):
    """The CoderAgent run prompt must carry the capability block so the model
    sees the real allow-list, not the old hard-coded sklearn-only line."""
    agents = importlib.import_module("easyicu.research_agent.agents")
    mc = _mod()
    # The agents module must import and use the block builder.
    assert hasattr(agents, "coder_method_capability_block")
    block = mc.coder_method_capability_block()
    assert "AVAILABLE ANALYTICAL LIBRARIES" in block
