"""Runtime probe of the advanced analytical packages available to the sandbox.

Agent-generated analysis code runs in a subprocess launched with the SAME
interpreter as this process (``CodeRunner`` uses ``sys.executable``), so a
package importable here is importable in the sandbox. The sandbox has no network,
so it cannot ``pip install`` anything at run time.

The reliability rule this module enforces: inject into the coder prompt ONLY the
advanced packages that are actually importable, so the model never writes an
``import`` that would hard-fail in the sandbox. Declaring a curated package in
``pyproject`` (the ``methods`` extra) and installing it automatically unlocks it
for the agent — no prompt edit needed. If a curated package is absent, the agent
is told the reliable baseline fallback instead, so a study still produces a
defensible result (the degradation ladder).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import List, Tuple

# Declared core scientific stack — always assumed present (hard deps / webapp
# extra). The agent may always use these.
BASELINE_PACKAGES: Tuple[str, ...] = (
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "sklearn",
    "pyarrow",
)


@dataclass(frozen=True)
class MethodPackage:
    """A curated advanced analytical package and its reliable fallback."""

    import_name: str
    pip_name: str
    capability: str  # one-line description injected into the coder prompt
    families: Tuple[str, ...]  # analysis families it primarily serves
    fallback: str  # what to use instead if the package is unavailable


# Curated advanced packages. Mirrored by the ``methods`` extra in pyproject.toml.
# Only add a package here once it is a declared, install-tested dependency — the
# point is that "listed here" implies "reliably installable", not "happens to be
# on one dev machine".
CURATED_METHOD_PACKAGES: Tuple[MethodPackage, ...] = (
    MethodPackage(
        import_name="lifelines",
        pip_name="lifelines",
        capability=(
            "survival analysis — KaplanMeierFitter, CoxPHFitter, "
            "logrank_test, concordance_index"
        ),
        families=("survival",),
        fallback="statsmodels.duration (PHReg, SurvfuncRight)",
    ),
    MethodPackage(
        import_name="shap",
        pip_name="shap",
        capability=(
            "model-agnostic feature attribution — TreeExplainer/Explainer, "
            "beeswarm and waterfall summaries of per-feature contributions"
        ),
        families=("prediction_model", "dynamic_prediction"),
        fallback="sklearn.inspection.permutation_importance or model coefficients",
    ),
    MethodPackage(
        import_name="xgboost",
        pip_name="xgboost",
        capability=(
            "gradient-boosted trees for tabular prediction "
            "(XGBClassifier / XGBRegressor)"
        ),
        families=("prediction_model", "dynamic_prediction"),
        fallback="sklearn HistGradientBoostingClassifier / GradientBoostingClassifier",
    ),
)


def _importable(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def available_method_packages() -> List[MethodPackage]:
    """Curated advanced packages that are importable in this environment."""
    return [pkg for pkg in CURATED_METHOD_PACKAGES if _importable(pkg.import_name)]


def coder_method_capability_block() -> str:
    """Prompt block naming exactly the libraries the coder may import.

    Self-truthing: reflects what is actually installed, so the model is never
    invited to import a package the no-network sandbox cannot provide.
    """
    available = available_method_packages()
    baseline = ", ".join(BASELINE_PACKAGES)
    lines = [
        "AVAILABLE ANALYTICAL LIBRARIES (import ONLY from this list; the sandbox "
        "has no network and cannot install anything):",
        f"- Always available: the Python standard library, {baseline}.",
    ]
    if available:
        lines.append(
            "- Advanced libraries installed and verified for THIS run — prefer "
            "the family-appropriate one over a hand-rolled approximation:"
        )
        for pkg in available:
            lines.append(
                f"    * {pkg.import_name} — {pkg.capability}. "
                f"If importing or fitting it fails, fall back to {pkg.fallback}."
            )
    missing = [
        pkg for pkg in CURATED_METHOD_PACKAGES if pkg not in set(available)
    ]
    if missing:
        names = ", ".join(pkg.import_name for pkg in missing)
        lines.append(
            f"- NOT available this run ({names}): do not import them; use the "
            "baseline-library fallback for that method instead."
        )
    lines.append(
        "Importing any package not named above is forbidden — it will fail in "
        "the sandbox and waste a repair attempt."
    )
    return "\n".join(lines)


__all__ = [
    "BASELINE_PACKAGES",
    "MethodPackage",
    "CURATED_METHOD_PACKAGES",
    "available_method_packages",
    "coder_method_capability_block",
]
