from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _load_research_agent() -> Any:
    """Load ``easyicu.research_agent`` without importing the heavy parent."""

    if "easyicu.research_agent" in sys.modules:
        mod = sys.modules["easyicu.research_agent"]
        parent = sys.modules.get("easyicu")
        if parent is not None:
            setattr(parent, "research_agent", mod)
        return mod

    if "easyicu" not in sys.modules:
        stub = types.ModuleType("easyicu")
        stub.__path__ = [str((SRC_ROOT / "easyicu").resolve())]
        sys.modules["easyicu"] = stub

    ra_path = SRC_ROOT / "easyicu" / "research_agent" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "easyicu.research_agent",
        ra_path,
        submodule_search_locations=[str(ra_path.parent)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["easyicu.research_agent"] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules["easyicu"], "research_agent", mod)
    return mod


@pytest.fixture(scope="session")
def ra():
    """The ``easyicu.research_agent`` module."""

    return _load_research_agent()


@pytest.fixture
def research_agent_runner_backend(request: pytest.FixtureRequest) -> str:
    """Return the runner backend selected explicitly for one test."""

    parametrized = getattr(request, "param", None)
    requested = (
        str(parametrized)
        if parametrized is not None
        else os.environ.get("EASYICU_TEST_RUNNER_KIND", "")
    )
    requested = requested.strip().lower() or "subprocess"
    if requested not in {"auto", "docker", "subprocess"}:
        pytest.fail(
            "research-agent test runner backend must be one of "
            "'auto', 'docker', or 'subprocess'"
        )
    return requested


@pytest.fixture(autouse=True)
def _explicit_test_runner_backend(
    monkeypatch: pytest.MonkeyPatch,
    research_agent_runner_backend: str,
):
    """Make Research Agent tests independent of ambient Docker availability."""

    if research_agent_runner_backend == "auto":
        yield
        return

    if research_agent_runner_backend == "subprocess":
        monkeypatch.setenv("EASYICU_ALLOW_UNSAFE_HOST_FALLBACK", "1")

    pipeline = importlib.import_module("easyicu.research_agent.pipeline")
    monkeypatch.setattr(
        pipeline,
        "select_safe_runner_kind",
        lambda **_kwargs: research_agent_runner_backend,
    )
    yield


def _clear_runtime_capability_test_context() -> None:
    """Clear a runner capability publication without importing Agent code."""

    module = sys.modules.get("easyicu.research_agent.execution.method_capabilities")
    if module is not None:
        module.set_runtime_capability_snapshot_provider(None)


@pytest.fixture(autouse=True)
def _isolate_runtime_capability_provider():
    """Keep direct runner tests from contaminating later prompt tests."""

    _clear_runtime_capability_test_context()
    try:
        yield
    finally:
        _clear_runtime_capability_test_context()


@pytest.fixture(scope="session")
def synthetic_cohort():
    """Small synthetic cohort with a composite-score completeness signal."""

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 800
    age = rng.normal(65, 15, n).clip(18, 95)
    base = rng.integers(1, 14, size=n, endpoint=False)
    miss = rng.random(n) < 0.10
    truly_low = rng.random(n) < 0.05
    sofa2 = np.where(truly_low, 0, base)
    logit = -3.5 + 0.18 * sofa2 + 0.012 * (age - 65) + np.where(miss, 1.5, 0.0)
    p = 1.0 / (1.0 + np.exp(-logit))
    death = (rng.random(n) < p).astype(int)
    los = rng.gamma(2.0, 1.5 + 0.15 * sofa2, size=n).clip(0.1, 60)
    lact = rng.lognormal(0.4 + 0.08 * sofa2, 0.6, size=n).clip(0.5, 25)
    creat = rng.lognormal(0.05 + 0.04 * sofa2, 0.4, size=n).clip(0.1, 12)
    map_v = rng.normal(85 - 1.6 * sofa2, 12, size=n).clip(40, 130)
    vaso = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-1.5 + 0.20 * sofa2)))).astype(int)
    return pd.DataFrame(
        {
            "stay_id": np.arange(1, n + 1),
            "age": age,
            "sex": rng.choice(["M", "F"], size=n),
            "sofa2": sofa2,
            "sofa2_n_components": np.where(miss, 0, 6),
            "lact": lact,
            "creat": creat,
            "map": map_v,
            "vaso": vaso,
            "los_icu": los,
            "death": death,
        }
    )
