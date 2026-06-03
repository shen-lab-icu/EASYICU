"""Shared fixtures for research_agent tests.

The research_agent sub-package depends only on a small subset of the
parent ``easyicu`` package (mostly ``get_concept_info`` for optional
metadata enrichment). Importing the heavy ``easyicu`` top level
unconditionally would force tests to set up data sources, etc., so
the fixtures here load ``easyicu.research_agent`` directly under a
stub parent package — exactly the pattern used by ``examples/`` and
the manual smoke tests during development.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types
from typing import Any

import pytest


def _load_research_agent() -> Any:
    """Load ``easyicu.research_agent`` without importing the heavy parent.

    We synthesize a stub ``easyicu`` package with the right
    ``__path__`` so relative imports inside ``research_agent`` work.
    Subsequent calls return the already-loaded module.
    """
    if "easyicu.research_agent" in sys.modules:
        return sys.modules["easyicu.research_agent"]

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    if "easyicu" not in sys.modules:
        stub = types.ModuleType("easyicu")
        stub.__path__ = [str((src_path / "easyicu").resolve())]
        sys.modules["easyicu"] = stub

    ra_path = src_path / "easyicu" / "research_agent" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "easyicu.research_agent",
        ra_path,
        submodule_search_locations=[str(ra_path.parent)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["easyicu.research_agent"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def ra():
    """The ``easyicu.research_agent`` module."""
    return _load_research_agent()


@pytest.fixture(scope="session")
def synthetic_cohort():
    """Small synthetic cohort with a composite-score completeness signal.

    Built as a self-contained closed-form generator so this fixture
    has no dependency on the demo script in ``examples/``.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 800
    age = rng.normal(65, 15, n).clip(18, 95)
    base = rng.integers(1, 14, size=n, endpoint=False)
    miss = rng.random(n) < 0.10
    truly_low = rng.random(n) < 0.05
    sofa2 = np.where(miss, 0, np.where(truly_low, 0, base))
    logit = -3.5 + 0.18 * sofa2 + 0.012 * (age - 65) + np.where(miss, 1.5, 0.0)
    p = 1.0 / (1.0 + np.exp(-logit))
    death = (rng.random(n) < p).astype(int)
    los = rng.gamma(2.0, 1.5 + 0.15 * sofa2, size=n).clip(0.1, 60)
    lact = rng.lognormal(0.4 + 0.08 * sofa2, 0.6, size=n).clip(0.5, 25)
    creat = rng.lognormal(0.05 + 0.04 * sofa2, 0.4, size=n).clip(0.1, 12)
    map_v = rng.normal(85 - 1.6 * sofa2, 12, size=n).clip(40, 130)
    vaso = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-1.5 + 0.20 * sofa2)))).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age, "sex": rng.choice(["M", "F"], size=n),
        "sofa2": sofa2,
        "sofa2_n_components": np.where(miss, 0, 6),
        "lact": lact, "creat": creat,
        "map": map_v, "vaso": vaso, "los_icu": los, "death": death,
    })
