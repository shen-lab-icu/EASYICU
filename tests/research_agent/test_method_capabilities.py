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

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _mod():
    module = importlib.import_module(
        "easyicu.research_agent.execution.method_capabilities"
    )
    module.set_runtime_capability_snapshot_provider(None)
    return module


def test_capability_block_lists_baseline_and_forbids_unlisted(ra):
    mc = _mod()
    block = mc.coder_method_capability_block()
    for pkg in ("pandas", "numpy", "scipy", "statsmodels", "sklearn"):
        assert pkg in block
    # The block must explicitly forbid importing anything not named.
    assert "forbidden" in block.lower()
    assert "no network" in block.lower()
    assert "easyicu.research_agent.methods.*" in block
    assert "explicitly named by the code contract" in block
    assert "All other project-local imports" in block


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
    # Host installations may omit the optional methods extra; the prompt must
    # truthfully report that state rather than assuming this dev environment.
    assert available <= {p.import_name for p in mc.CURATED_METHOD_PACKAGES}


def test_explicit_runtime_snapshot_overrides_host_packages(ra, monkeypatch):
    mc = _mod()
    monkeypatch.setattr(mc, "_importable", lambda _name: True)

    block = mc.coder_method_capability_block(
        snapshot={*mc.BASELINE_PACKAGES, "seaborn", "lifelines"}
    )

    assert "* lifelines" in block
    assert "* shap" not in block
    assert "* xgboost" not in block


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
        assert (
            pkg.pip_name.lower() in declared
        ), f"{pkg.pip_name} is curated but not declared in the methods extra"
    # And the methods extra must be wired into the aggregate 'all' extra.
    all_block = re.search(r"\nall\s*=\s*\[(.*?)\]", pyproject, re.S)
    assert all_block and "methods" in all_block.group(1)


def test_coder_prompt_embeds_capability_block(ra):
    """The CoderAgent run prompt must carry the capability block so the model
    sees the real allow-list, not the old hard-coded sklearn-only line."""
    agents = importlib.import_module("easyicu.research_agent.agents.core")
    mc = _mod()
    # The agents module must import and use the block builder.
    assert hasattr(agents, "coder_method_capability_block")
    block = mc.coder_method_capability_block()
    assert "AVAILABLE ANALYTICAL LIBRARIES" in block


def test_reference_docker_image_matches_advertised_capabilities(ra):
    mc = _mod()
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    dockerfile = (
        (
            repo_root
            / "src"
            / "easyicu"
            / "research_agent"
            / "runner_image"
            / "Dockerfile"
        )
        .read_text(encoding="utf-8")
        .lower()
    )
    requirements_lock = (
        repo_root
        / "src"
        / "easyicu"
        / "research_agent"
        / "runner_image"
        / "requirements.lock"
    ).read_text(encoding="utf-8")
    locked_packages = {
        line.split("==", 1)[0].strip().lower()
        for line in requirements_lock.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project_dependencies = {
        re.split(r"[<>=!~;\[\s]", str(dependency), maxsplit=1)[0].lower()
        for dependency in pyproject["project"]["dependencies"]
    }
    assert project_dependencies <= locked_packages, (
        "runner lock is missing direct EasyICU dependencies: "
        + ", ".join(sorted(project_dependencies - locked_packages))
    )
    pip_names = {
        "sklearn": "scikit-learn",
        **{name: name for name in mc.BASELINE_PACKAGES if name != "sklearn"},
        **{name: name for name in mc.OPTIONAL_BASELINE_PACKAGES},
    }
    for package in mc.CURATED_METHOD_PACKAGES:
        pip_names[package.import_name] = package.pip_name

    for import_name, pip_name in pip_names.items():
        assert pip_name.lower() in locked_packages, (
            f"{import_name} is advertised but {pip_name} is absent from "
            "runner_image/requirements.lock"
        )
    assert all(
        "==" in line
        for line in requirements_lock.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert "requirements.lock" in dockerfile
    assert "pip install --no-cache-dir --no-deps /opt/easyicu" in dockerfile

    dockerignore = (repo_root / ".dockerignore").read_text(encoding="utf-8")
    for excluded in (".git", ".venv", ".env.*", "research_output", "output"):
        assert excluded in dockerignore
