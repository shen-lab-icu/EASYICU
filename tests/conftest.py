from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
import types
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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


@pytest.fixture(autouse=True)
def _explicit_test_runner_backend(monkeypatch):
    """Select a host backend only when the test command explicitly requests it.

    Production ``runner_kind="auto"`` must stay fail-closed on Linux hosts
    without the pinned Docker image.  The CI integration lane exercises only
    repository-owned deterministic scripts, so it may opt into the host runner
    with both ``EASYICU_TEST_RUNNER_KIND=subprocess`` and the CodeRunner's
    separate unsafe-host authorization flag.  Keeping this switch in test
    plumbing prevents CI configuration from weakening the product default.
    """

    requested = os.environ.get("EASYICU_TEST_RUNNER_KIND", "").strip().lower()
    if not requested:
        yield
        return
    if requested != "subprocess":
        pytest.fail(
            "EASYICU_TEST_RUNNER_KIND only supports the explicit "
            "'subprocess' test backend"
        )

    pipeline = importlib.import_module("easyicu.research_agent.pipeline")
    monkeypatch.setattr(
        pipeline,
        "select_safe_runner_kind",
        lambda **_kwargs: "subprocess",
    )
    yield


@pytest.fixture(scope="session")
def synthetic_cohort():
    """Small synthetic cohort with a composite-score completeness signal."""

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 800
    age = rng.normal(65, 15, n).clip(18, 95)
    base = rng.integers(1, 14, size=n, endpoint=False)
    # Under-measurement is carried by the component count; a score of zero
    # remains a genuine low score rather than an imputed missing value.
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


def pytest_addoption(parser):
    parser.addoption(
        "--run-real",
        action="store_true",
        default=False,
        help="Run tests that require a local real ICU database path.",
    )
    parser.addoption(
        "--run-real-llm",
        action="store_true",
        default=False,
        help="Run tests that hit a real LLM provider (OpenAI / OpenRouter / Anthropic).",
    )
    parser.addoption(
        "--run-packaging",
        action="store_true",
        default=False,
        help="Run tests that build and install a real wheel (slow, ~2 min).",
    )


_REAL_LLM_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
)


def pytest_collection_modifyitems(config, items):
    run_real = config.getoption("--run-real", default=False)
    real_data_path = os.environ.get("EASYICU_DATA_PATH", "")
    real_data_ready = bool(real_data_path) and Path(real_data_path).exists()
    skip_real_data = pytest.mark.skip(
        reason="Need --run-real and an existing EASYICU_DATA_PATH"
    )

    run_real_llm = config.getoption("--run-real-llm", default=False)
    real_llm_ready = any(os.environ.get(k) for k in _REAL_LLM_KEY_ENV_VARS)
    skip_real_llm = pytest.mark.skip(
        reason="Need --run-real-llm and at least one of "
        + ", ".join(_REAL_LLM_KEY_ENV_VARS)
    )

    for item in items:
        if "needs_real_data" in item.keywords and not (run_real and real_data_ready):
            item.add_marker(skip_real_data)
        if "needs_real_llm" in item.keywords and not (run_real_llm and real_llm_ready):
            item.add_marker(skip_real_llm)
