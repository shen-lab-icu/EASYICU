"""Analysis-pattern auditor (analysis-agnostic ICU footguns).

Why this exists
---------------

The existing :class:`ConceptUsageAuditor` catches aggregation-level
mistakes (mean of an ordinal SOFA component, fillna(0) on labs, etc.).
That works for tabular descriptive / association tasks, but breaks
down the moment the agent does anything else:

* clustering with mixed-type features (KMeans on ordinal scores +
  binary flags + skewed labs, no scaling),
* prediction modelling (no train/test split, target column included
  in the feature matrix, mortality leaked into the predictor set),
* time-series / survival analysis (no censoring handling,
  ``time_to_event`` treated as continuous covariate),
* dimensionality reduction (PCA on raw mixed-scale columns).

This auditor adds a second deterministic pass that is **analysis-
agnostic**: it doesn't care whether the script is a clustering
notebook or a logistic regression. It checks for footguns that
matter regardless of the analysis family, driven by:

1. the variable role table from the :class:`ResearchContext`, and
2. a small set of pattern triggers (calls to scikit-learn / scipy
   APIs) extracted from the script's AST.

Severity model: same ``info / warning / error`` triple as the rest
of the validator stack. ``error`` blocks the step before execution
(via the existing concept-audit gate); ``warning`` is recorded but
allows the step to proceed.

Pure stdlib + ``ast``. No sklearn import.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..ordered_stratified_contract import ordered_stratified_script_findings
from ..schema import (
    AnalysisStep,
    ConceptDescriptor,
    ResearchContext,
    ValidationFinding,
    VariableRole,
)
from ..trajectory.contract import (
    is_continuous_trajectory_representation,
    selected_trajectory_variables,
    trajectory_script_findings,
)


# ---------------------------------------------------------------------------
# Pattern table — names of (sklearn / scipy / numpy) call sites we react to
# ---------------------------------------------------------------------------


_DISTANCE_BASED_ESTIMATORS = (
    "KMeans",
    "MiniBatchKMeans",
    "AgglomerativeClustering",
    "DBSCAN",
    "HDBSCAN",
    "Birch",
    "SpectralClustering",
    "MeanShift",
    "OPTICS",
    "NearestNeighbors",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
)

_LINEAR_PCA_ESTIMATORS = (
    "PCA",
    "TruncatedSVD",
    "FactorAnalysis",
    "FastICA",
    "KernelPCA",
)

_SUPERVISED_ESTIMATORS = (
    "LogisticRegression",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "MLPClassifier",
    "MLPRegressor",
    "SVC",
    "SVR",
    "Logit",  # statsmodels
    "OLS",  # statsmodels
    "GLM",  # statsmodels
)

_SCALERS = (
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "Normalizer",
)

_SPLITTERS = (
    "train_test_split",
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "TimeSeriesSplit",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
)

_ASSOCIATION_ESTIMATORS = {"Logit", "OLS", "GLM", "LogisticRegression", "LinearRegression"}

_SURVIVAL_FAMILIES = (
    "CoxPHFitter",
    "WeibullAFTFitter",
    "KaplanMeierFitter",
    "NelsonAalenFitter",
    "AalenJohansenFitter",
)


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------


def _call_target_name(node: ast.Call) -> Optional[str]:
    """Return the *short* name of the called callable (last attr / id)."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _string_literals_anywhere(tree: ast.AST) -> List[str]:
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
    return out


def _columns_in_call(
    node: ast.Call, alias_map: Dict[str, Set[str]]
) -> Set[str]:
    """Return the set of column names referenced by a call's args / kwargs."""
    cols: Set[str] = set()
    for arg in list(node.args) + [kw.value for kw in node.keywords]:
        cols |= _columns_in_expr(arg, alias_map)
    return cols


def _columns_in_expr(
    node: ast.AST, alias_map: Dict[str, Set[str]]
) -> Set[str]:
    """Conservative column extraction: literal strings + alias lookups."""
    cols: Set[str] = set()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        cols.add(node.value)
    elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        for elt in node.elts:
            cols |= _columns_in_expr(elt, alias_map)
    elif isinstance(node, ast.Subscript):
        # df['x'] / df[['x','y']]
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            cols.add(sl.value)
        elif isinstance(sl, (ast.List, ast.Tuple)):
            for elt in sl.elts:
                cols |= _columns_in_expr(elt, alias_map)
    elif isinstance(node, ast.Name):
        if node.id in alias_map:
            cols |= alias_map[node.id]
    elif isinstance(node, ast.Attribute):
        # df.col -> ".col" attribute reference
        cols.add(node.attr)
    return cols


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


@dataclass
class _ScriptInspection:
    distance_estimators: List[Tuple[str, ast.Call]]
    pca_estimators: List[Tuple[str, ast.Call]]
    supervised_estimators: List[Tuple[str, ast.Call]]
    scalers: List[Tuple[str, ast.Call]]
    splitters: List[Tuple[str, ast.Call]]
    survival_estimators: List[Tuple[str, ast.Call]]
    feature_matrices: List[ast.Call]  # any call that looks like X = df[[...]]
    fit_calls: List[ast.Call]
    string_lits: List[str]


def _inspect_script(tree: ast.Module) -> _ScriptInspection:
    distance: List[Tuple[str, ast.Call]] = []
    pca: List[Tuple[str, ast.Call]] = []
    supervised: List[Tuple[str, ast.Call]] = []
    scalers: List[Tuple[str, ast.Call]] = []
    splitters: List[Tuple[str, ast.Call]] = []
    survival: List[Tuple[str, ast.Call]] = []
    fit_calls: List[ast.Call] = []
    feat: List[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_target_name(node)
        if name is None:
            continue
        if name in _DISTANCE_BASED_ESTIMATORS:
            distance.append((name, node))
        elif name in _LINEAR_PCA_ESTIMATORS:
            pca.append((name, node))
        elif name in _SUPERVISED_ESTIMATORS:
            supervised.append((name, node))
        elif name in _SCALERS:
            scalers.append((name, node))
        elif name in _SPLITTERS:
            splitters.append((name, node))
        elif name in _SURVIVAL_FAMILIES:
            survival.append((name, node))
        if name in {"fit", "fit_predict", "fit_transform"}:
            fit_calls.append(node)
    return _ScriptInspection(
        distance_estimators=distance,
        pca_estimators=pca,
        supervised_estimators=supervised,
        scalers=scalers,
        splitters=splitters,
        survival_estimators=survival,
        feature_matrices=feat,
        fit_calls=fit_calls,
        string_lits=_string_literals_anywhere(tree),
    )


def _build_alias_map(tree: ast.Module) -> Dict[str, Set[str]]:
    """Track variable→{columns} bindings via simple `var = df[[...]]`."""
    alias_map: Dict[str, Set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            cols = _columns_in_expr(node.value, alias_map)
            if not cols:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    alias_map[target.id] = set(cols)
    return alias_map


def _extract_feature_columns_for_estimator(
    estimator_call: ast.Call, alias_map: Dict[str, Set[str]]
) -> Set[str]:
    """Heuristic: find the first ``X``-like arg passed to fit / fit_predict."""
    cols: Set[str] = set()
    # Walk the AST upwards-by-search: look for the nearest ``.fit(X, ...)``
    # whose receiver is the estimator. Simplification: scan all
    # ``.fit(...)`` calls' first arg and union-pick anything that
    # references a column-set alias.
    return cols


def _ordinal_columns_in(
    cols: Iterable[str], var_by_name: Dict[str, ConceptDescriptor]
) -> List[str]:
    out: List[str] = []
    for c in cols:
        v = var_by_name.get(c)
        if v is None:
            continue
        if v.role in {
            VariableRole.ORDINAL_SCORE,
            VariableRole.COMPOSITE_SCORE,
        } and not is_continuous_trajectory_representation(v):
            out.append(c)
    return out


def _outcome_columns_in(
    cols: Iterable[str], var_by_name: Dict[str, ConceptDescriptor]
) -> List[str]:
    out: List[str] = []
    for c in cols:
        v = var_by_name.get(c)
        if v is None:
            continue
        if v.role == VariableRole.OUTCOME:
            out.append(c)
    return out


def _id_or_time_columns_in(
    cols: Iterable[str], var_by_name: Dict[str, ConceptDescriptor]
) -> List[str]:
    out: List[str] = []
    for c in cols:
        v = var_by_name.get(c)
        if v is None:
            continue
        if v.role in {VariableRole.ID, VariableRole.TIME, VariableRole.INDEX}:
            out.append(c)
    return out


def _prediction_or_performance_context(
    *,
    context: ResearchContext,
    step: Optional[AnalysisStep],
) -> bool:
    family = (
        (context.user_preferences.inferred_analysis_family or "").lower()
        if context.user_preferences
        else ""
    )
    combined = " ".join(
        part
        for part in (
            context.research_question or "",
            step.step_id if step is not None else "",
            step.intent if step is not None else "",
            " ".join(step.expected_outputs or []) if step is not None else "",
            step.method or "" if step is not None else "",
        )
        if part
    ).lower()
    return family == "prediction_model" or any(
        term in combined
        for term in (
            "prediction",
            "predictive",
            "classifier",
            "classification",
            "held-out",
            "train/test",
            "cross-validation",
            "cross validation",
            "generalisation",
            "generalization",
            "performance",
            "auroc",
            "brier",
            "calibration",
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AnalysisPatternAuditor:
    """Analysis-agnostic ICU-aware static checks.

    Run alongside :class:`ConceptUsageAuditor`; surfaces footguns that
    show up in clustering / prediction / survival / dimensionality-
    reduction code.
    """

    name = "analysis_pattern_auditor"

    def audit(
        self,
        *,
        context: ResearchContext,
        script_text: str,
        step: Optional[AnalysisStep] = None,
    ) -> List[ValidationFinding]:
        try:
            tree = ast.parse(script_text)
        except SyntaxError:
            return []
        var_by_name = {v.name: v for v in context.variables}
        alias_map = _build_alias_map(tree)
        inspection = _inspect_script(tree)
        findings: List[ValidationFinding] = []
        findings.extend(
            ordered_stratified_script_findings(step=step, script_text=script_text)
            if step is not None
            else []
        )
        findings.extend(
            trajectory_script_findings(
                context=context,
                step=step,
                script_text=script_text,
            )
        )

        # ------------------------------------------------------------
        # 1) Distance-based estimators on ordinal / composite scores
        # ------------------------------------------------------------
        if inspection.distance_estimators:
            ordinal_referenced = _ordinal_columns_in(alias_map.keys(), var_by_name)
            ordinal_referenced.extend(
                variable.name
                for variable in selected_trajectory_variables(
                    context=context,
                    script_text=script_text,
                    step=step,
                )
                if variable.role
                in {VariableRole.ORDINAL_SCORE, VariableRole.COMPOSITE_SCORE}
                and not is_continuous_trajectory_representation(variable)
            )
            # Also scan column literals inside the script.
            for lit in inspection.string_lits:
                if lit in var_by_name:
                    v = var_by_name[lit]
                    if v.role in {
                        VariableRole.ORDINAL_SCORE,
                        VariableRole.COMPOSITE_SCORE,
                    } and not is_continuous_trajectory_representation(v):
                        ordinal_referenced.append(lit)
            ordinal_set = sorted(set(ordinal_referenced))
            for est_name, _node in inspection.distance_estimators:
                if ordinal_set:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="error",
                            message=(
                                f"{est_name} is distance-based but the feature space "
                                f"includes ordinal score(s) {ordinal_set}. Euclidean "
                                "distance over ordinal SOFA-like variables is a "
                                "category error: encode them as an ordinal embedding, "
                                "drop them, or use a Gower-style metric."
                            ),
                            detail={
                                "estimator": est_name,
                                "ordinal_columns": ordinal_set,
                                "step_id": step.step_id if step else None,
                            },
                        )
                    )
                # Scaling check: if any distance estimator is used and no
                # scaler is present, flag a warning. Different feature
                # ranges destroy clustering quality silently.
                if not inspection.scalers:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message=(
                                f"{est_name} is distance-based but no "
                                "StandardScaler / RobustScaler / Normalizer is "
                                "applied to the feature matrix. ICU lab values "
                                "(lactate, creatinine) span orders of magnitude "
                                "and dominate the distance unless scaled."
                            ),
                            detail={"estimator": est_name},
                        )
                    )

        # ------------------------------------------------------------
        # 2) PCA on mixed-scale features without scaling
        # ------------------------------------------------------------
        if inspection.pca_estimators and not inspection.scalers:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "PCA-style decomposition without a preceding scaler. "
                        "Components will be dominated by the variables with "
                        "the largest variance (typically labs). Standardise "
                        "or use a robust scaler before PCA."
                    ),
                    detail={
                        "estimators": [n for n, _ in inspection.pca_estimators],
                    },
                )
            )

        # ------------------------------------------------------------
        # 3) Outcome leakage: outcome variable inside a feature matrix
        #    used by a supervised estimator.
        # ------------------------------------------------------------
        if inspection.supervised_estimators or inspection.distance_estimators:
            # Find any alias whose column set contains an outcome column.
            for alias, cols in alias_map.items():
                outs = _outcome_columns_in(cols, var_by_name)
                if not outs:
                    continue
                # Report only when this alias also looks like a feature
                # matrix: heuristic = capital ``X`` or ``features`` / ``feats``
                if alias not in {"X", "x", "features", "feats", "design", "Xtrain", "X_train"}:
                    continue
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Feature matrix '{alias}' includes outcome "
                            f"variable(s) {outs}. Remove the outcome from the "
                            "predictor matrix before fitting any model."
                        ),
                        detail={
                            "feature_alias": alias,
                            "outcome_columns": outs,
                        },
                    )
                )

        # ------------------------------------------------------------
        # 4) ID / time leakage as a feature
        # ------------------------------------------------------------
        for alias, cols in alias_map.items():
            if alias not in {"X", "x", "features", "feats", "design", "Xtrain", "X_train"}:
                continue
            id_cols = _id_or_time_columns_in(cols, var_by_name)
            if id_cols:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            f"Feature matrix '{alias}' contains ID/time/index "
                            f"columns {id_cols}. These leak entity identity / "
                            "timing into the model and will inflate performance."
                        ),
                        detail={
                            "feature_alias": alias,
                            "id_or_time_columns": id_cols,
                        },
                    )
                )

        # ------------------------------------------------------------
        # 5) Supervised estimator without train/test split
        # ------------------------------------------------------------
        supervised_estimator_names = sorted({n for n, _ in inspection.supervised_estimators})
        non_association_estimators = [
            name for name in supervised_estimator_names if name not in _ASSOCIATION_ESTIMATORS
        ]
        prediction_like = _prediction_or_performance_context(context=context, step=step)
        if (
            inspection.supervised_estimators
            and not inspection.splitters
            and (prediction_like or non_association_estimators)
        ):
            ests = supervised_estimator_names
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"{ests} fit without any splitter (train_test_split / "
                        "KFold / TimeSeriesSplit). Reported performance is "
                        "in-sample and not interpretable as generalisation."
                    ),
                    detail={"estimators": ests},
                )
            )

        # ------------------------------------------------------------
        # 6) Survival analysis sanity: time column should be in role TIME
        # ------------------------------------------------------------
        if inspection.survival_estimators:
            # Look for fit() with duration_col / event_col kwargs.
            duration_col: Optional[str] = None
            event_col: Optional[str] = None
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _call_target_name(node) != "fit":
                    continue
                for kw in node.keywords:
                    if kw.arg in {"duration_col", "duration"}:
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            duration_col = kw.value.value
                    if kw.arg in {"event_col", "event"}:
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            event_col = kw.value.value
            if duration_col and duration_col in var_by_name:
                v = var_by_name[duration_col]
                if v.role not in {VariableRole.TIME, VariableRole.OUTCOME}:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message=(
                                f"Survival fit uses duration_col='{duration_col}' "
                                f"but its declared role is {v.role.value}, not "
                                "time/outcome. Confirm the column really encodes "
                                "follow-up time."
                            ),
                            detail={
                                "duration_col": duration_col,
                                "actual_role": v.role.value,
                            },
                        )
                    )
            if event_col and event_col in var_by_name:
                v = var_by_name[event_col]
                if v.role != VariableRole.OUTCOME:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message=(
                                f"Survival fit uses event_col='{event_col}' but "
                                f"its declared role is {v.role.value}, not "
                                "outcome. Confirm the column encodes the event "
                                "indicator."
                            ),
                            detail={
                                "event_col": event_col,
                                "actual_role": v.role.value,
                            },
                        )
                    )

        # ------------------------------------------------------------
        # 7) Hard-coded random state diversity (best-practice nudge)
        #    Distance / supervised estimators benefit from
        #    deterministic seeds — flag missing random_state= when an
        #    estimator that supports it is constructed.
        # ------------------------------------------------------------
        for est_name, call in inspection.distance_estimators + inspection.supervised_estimators:
            if est_name in {"OLS", "GLM", "Logit"}:  # statsmodels: deterministic
                continue
            if not prediction_like and est_name in _ASSOCIATION_ESTIMATORS:
                continue
            has_random_state = any(
                kw.arg == "random_state" for kw in call.keywords
            )
            if not has_random_state:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="info",
                        message=(
                            f"{est_name} constructed without random_state=. "
                            "For reproducibility, set it explicitly (e.g. "
                            "``random_state=2026``) so the run can be replayed "
                            "deterministically."
                        ),
                        detail={"estimator": est_name},
                    )
                )

        return findings


__all__ = ["AnalysisPatternAuditor"]
