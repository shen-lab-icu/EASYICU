"""Static analytical-package declarations shared across agent boundaries.

This module is deliberately data-only.  Execution code may probe which
packages are available, while authority code may bind the same declared
package set into a reproducibility fingerprint without importing execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

BASELINE_PACKAGES: Tuple[str, ...] = (
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "statsmodels",
    "sklearn",
    "pyarrow",
)

OPTIONAL_BASELINE_PACKAGES: Tuple[str, ...] = ("seaborn",)


@dataclass(frozen=True)
class MethodPackage:
    """A curated analytical package and its reliable baseline fallback."""

    import_name: str
    pip_name: str
    capability: str
    families: Tuple[str, ...]
    fallback: str


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

# Distributions that change computed results but that the Coder is NOT invited to
# import directly.  ``patsy`` is the standing case: statsmodels' formula interface
# is built on it, so its version moves numbers, yet a Coder writes ``bs(age, df=4)``
# inside a statsmodels formula rather than importing patsy.  Declaring it as a
# MethodPackage would advertise a direct import nobody needs; leaving it out of the
# fingerprint would let a version bump change results invisibly.  It belongs to
# neither list, so it gets its own — with one owner instead of a literal repeated
# in every builder of the distribution set.
FINGERPRINT_ONLY_DISTRIBUTIONS: Tuple[str, ...] = ("patsy",)


__all__ = [
    "BASELINE_PACKAGES",
    "OPTIONAL_BASELINE_PACKAGES",
    "MethodPackage",
    "CURATED_METHOD_PACKAGES",
    "FINGERPRINT_ONLY_DISTRIBUTIONS",
]
