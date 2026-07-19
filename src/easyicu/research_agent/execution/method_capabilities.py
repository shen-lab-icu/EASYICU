"""Runtime probe of the advanced analytical packages available to the runner.

``CodeRunner`` uses this interpreter. The reference ``DockerRunner`` image is
built from the same baseline and curated lists and captures its own image digest
plus ``pip freeze`` at execution time. Custom images must preserve that contract.
Neither runner can install packages at analysis time.

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
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Collection, List, Optional, Tuple

# Declared core scientific stack — always assumed present (hard deps / webapp
# extra). The agent may always use these.
BASELINE_PACKAGES: Tuple[str, ...] = (
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "statsmodels",
    "sklearn",
    "pyarrow",
)

# Useful plotting convenience, but not a core project dependency. Probe it
# rather than advertising it unconditionally to minimal host installations.
OPTIONAL_BASELINE_PACKAGES: Tuple[str, ...] = ("seaborn",)


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


_RUNTIME_SNAPSHOT_PROVIDER: ContextVar[Optional[Callable[[], Collection[str]]]] = (
    ContextVar("easyicu_method_capability_snapshot_provider", default=None)
)


def set_runtime_capability_snapshot_provider(
    provider: Optional[Callable[[], Collection[str]]],
) -> None:
    """Select the execution-runtime capability source for this context.

    ``DockerRunner`` installs a lazy provider backed by its immutable image
    snapshot. ``CodeRunner`` clears it so host execution continues to probe the
    active interpreter. A ContextVar prevents concurrent runner contexts from
    leaking one image's allow-list into another.
    """

    _RUNTIME_SNAPSHOT_PROVIDER.set(provider)


def runtime_capability_snapshot() -> Optional[frozenset[str]]:
    provider = _RUNTIME_SNAPSHOT_PROVIDER.get()
    if provider is None:
        return None
    return frozenset(str(name) for name in provider())


def _importable(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def available_method_packages(
    snapshot: Optional[Collection[str]] = None,
) -> List[MethodPackage]:
    """Curated advanced packages that are importable in this environment."""

    active_snapshot = (
        frozenset(snapshot) if snapshot is not None else runtime_capability_snapshot()
    )
    if active_snapshot is not None:
        return [
            pkg for pkg in CURATED_METHOD_PACKAGES if pkg.import_name in active_snapshot
        ]
    return [pkg for pkg in CURATED_METHOD_PACKAGES if _importable(pkg.import_name)]


def coder_method_capability_block(
    snapshot: Optional[Collection[str]] = None,
) -> str:
    """Prompt block naming exactly the libraries the coder may import.

    Self-truthing: reflects what is actually installed, so the model is never
    invited to import a package the no-network sandbox cannot provide.
    """
    active_snapshot = (
        frozenset(snapshot) if snapshot is not None else runtime_capability_snapshot()
    )
    available = available_method_packages(active_snapshot)
    baseline = ", ".join(BASELINE_PACKAGES)
    optional_baseline = [
        package
        for package in OPTIONAL_BASELINE_PACKAGES
        if (
            package in active_snapshot
            if active_snapshot is not None
            else _importable(package)
        )
    ]
    lines = [
        "AVAILABLE ANALYTICAL LIBRARIES (import ONLY from this list; the sandbox "
        "has no network and cannot install anything):",
        f"- Always available: the Python standard library, {baseline}.",
    ]
    if optional_baseline:
        lines.append(
            "- Additional baseline libraries installed and verified for THIS run: "
            + ", ".join(optional_baseline)
            + "."
        )
    missing_optional = [
        package
        for package in OPTIONAL_BASELINE_PACKAGES
        if package not in optional_baseline
    ]
    if missing_optional:
        lines.append(
            "- NOT available this run ("
            + ", ".join(missing_optional)
            + "): use matplotlib directly."
        )
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
    missing = [pkg for pkg in CURATED_METHOD_PACKAGES if pkg not in set(available)]
    if missing:
        names = ", ".join(pkg.import_name for pkg in missing)
        lines.append(
            f"- NOT available this run ({names}): do not import them; use the "
            "baseline-library fallback for that method instead."
        )
    lines.append(
        "Importing any package not named above is forbidden, except an exact "
        "documented `easyicu.research_agent.methods.*` module explicitly named "
        "by the code contract for the current method. All other project-local "
        "imports will fail in the sandbox and waste a repair attempt."
    )
    return "\n".join(lines)


__all__ = [
    "BASELINE_PACKAGES",
    "OPTIONAL_BASELINE_PACKAGES",
    "MethodPackage",
    "CURATED_METHOD_PACKAGES",
    "available_method_packages",
    "coder_method_capability_block",
    "runtime_capability_snapshot",
    "set_runtime_capability_snapshot_provider",
]
