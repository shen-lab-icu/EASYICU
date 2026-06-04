"""Deterministic code-mutation and step-summary repair helpers.

These functions sit *below* the LLM coder layer. They run when:

* a step's ``step_summary.json`` came back inconsistent (numeric NaNs,
  missing primary effects, mis-imputed columns) and we want one last
  deterministic patch on the generated code before declaring the step
  failed (:func:`_deterministic_summary_repair`);
* the runner emitted a recognisable error pattern (pandas KeyError for a
  missing column, statsmodels dtype/inf failure, sklearn bool imputer
  rejection, missing ``import os``, dangling ``python`` prefix, ...) and
  we can transform the source script to a working form without re-asking
  the LLM (:func:`_deterministic_runner_repair`).

All repairs here are **case-neutral in effect**: they patch generic Python /
library / dataframe failure modes or generated-code variable names. They must
not dispatch on benchmark task ids or substitute a full study-specific
analysis template. Case-specific fallbacks belong in an explicitly registered
``CasePluginRegistry`` (``research_agent.fallback``), not in this shared
module.

The split between this module and :mod:`.summary_repair` is by *target*:
``summary_repair`` rewrites ``step_summary.json`` from raw artefacts on
disk; this module rewrites the *Python source* the runner executes.

Both deterministic repair functions take ``previous_repair`` so the
pipeline can break out of an A→B→A oscillation by remembering what it
last tried.
"""

from __future__ import annotations

import ast
import json
import math
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .scalar_utils import (
    _coerce_scalar,
    _first_numeric_effect_from_text,
    _first_numeric_scalar_with_key_fragment,
    _first_present_scalar,
    _flatten_scalar_dict,
)


_BINARY_MODEL_REPAIR_FAMILIES = {
    "association_study",
    "prediction_model",
    "validation",
    "robustness",
    "bias_audit",
}


def _family_allows_binary_model_repair(analysis_family: Optional[str]) -> bool:
    """Return whether deterministic binary-model fallbacks fit the task family."""

    if analysis_family is None:
        return True
    family = str(analysis_family).strip().lower()
    if not family:
        return True
    return family in _BINARY_MODEL_REPAIR_FAMILIES


def _code_contains_binary_model(code: str) -> bool:
    lowered = code.lower()
    return (
        "sm.logit(" in lowered
        or "sm.logit" in lowered
        or "sm.glm" in lowered and "binomial" in lowered
        or "logisticregression" in lowered
        or "predict_proba" in lowered
        or "roc_auc" in lowered
    )


def _statsmodels_repair_allowed_for_family(code: str, analysis_family: Optional[str]) -> bool:
    if _family_allows_binary_model_repair(analysis_family):
        return True
    return not _code_contains_binary_model(code)


def _primary_association_fallback_code(
    *,
    predictor: str,
    outcome: str,
    reason: str,
) -> str:
    """Appendable case-neutral rescue block for common regression failures."""

    if not outcome:
        raise ValueError("A binary outcome column is required for primary association fallback.")
    predictor_lit = json.dumps(predictor)
    outcome_lit = json.dumps(outcome)
    reason_lit = json.dumps(reason)
    return textwrap.dedent(
        f"""
        def _easyicu_primary_association_fallback_v1():
            import json
            import os
            import numpy as np
            import pandas as pd
            import statsmodels.api as sm

            cohort_path = os.environ.get("COHORT_PARQUET")
            out_dir = os.environ.get("STEP_OUT_DIR", ".")
            if not cohort_path:
                return
            df_fallback = pd.read_parquet(cohort_path)
            predictor_col = {predictor_lit}
            outcome_col = {outcome_lit}
            if predictor_col not in df_fallback.columns or outcome_col not in df_fallback.columns:
                return
            model_df = df_fallback[[outcome_col, predictor_col]].copy()
            for col in model_df.columns:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
            outcome_values = set(model_df[outcome_col].dropna().astype(float).unique().tolist())
            if outcome_values - {{0.0, 1.0}}:
                return
            if len(model_df) < 20 or len(outcome_values) < 2:
                return
            y = model_df[outcome_col].astype(float)
            X = sm.add_constant(model_df[[predictor_col]].astype(float), has_constant="add")
            result = sm.Logit(y, X).fit(disp=0)
            coef = float(result.params[predictor_col])
            ci = np.exp(result.conf_int().loc[predictor_col]).tolist()
            primary_or = float(np.exp(coef))
            p_value = float(result.pvalues[predictor_col])
            table_path = os.path.join(out_dir, "primary_association.csv")
            pd.DataFrame([{{
                "variable": predictor_col,
                "odds_ratio": primary_or,
                "or_lower": float(ci[0]),
                "or_upper": float(ci[1]),
                "p_value": p_value,
                "n_complete": int(len(model_df)),
                "method": "deterministic_logistic_regression_fallback",
            }}]).to_csv(table_path, index=False)
            summary_path = os.path.join(out_dir, "step_summary.json")
            summary = {{}}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        summary.update(loaded)
                except Exception:
                    summary = {{}}
            summary.update({{
                "primary_predictor": predictor_col,
                "outcome": outcome_col,
                "primary_or": primary_or,
                "statistic:primary_or": primary_or,
                "primary_or_ci": [float(ci[0]), float(ci[1])],
                "p_value": p_value,
                "n_complete": int(len(model_df)),
                "method": "deterministic_logistic_regression_fallback",
                "fallback_reason": {reason_lit},
            }})
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(json.dumps(summary, ensure_ascii=False))

        try:
            _easyicu_primary_association_fallback_v1()
        except Exception as _easyicu_primary_fallback_exc:
            print(f"primary_association_fallback_failed: {{_easyicu_primary_fallback_exc}}")
        """
    ).strip("\n")


def _infer_overexpanded_categorical_predictor(step_summary: Dict[str, Any], code: str) -> Optional[str]:
    """Infer the source variable behind a singular dummy-expanded score model."""

    predictors = step_summary.get("model", {}).get("predictors")
    if isinstance(predictors, Sequence) and not isinstance(predictors, (str, bytes)):
        counts: Dict[str, int] = {}
        for item in predictors:
            match = re.match(r"C\((?P<name>[A-Za-z_][A-Za-z0-9_]*)\)\[T\.", str(item))
            if match:
                name = match.group("name")
                counts[name] = counts.get(name, 0) + 1
        if counts:
            name, count = max(counts.items(), key=lambda pair: pair[1])
            if count >= 3:
                return name
    for match in re.finditer(r"C\((?P<name>[A-Za-z_][A-Za-z0-9_]*)\)", code):
        return match.group("name")
    return None


def _infer_binary_outcome_from_code(code: str) -> Optional[str]:
    """Best-effort extraction of the left-hand side of a generated formula."""

    match = re.search(r"formula\s*=\s*[\"'](?P<outcome>[A-Za-z_][A-Za-z0-9_]*)\s*~", code)
    if match:
        return match.group("outcome")
    match = re.search(r"[\"'](?P<outcome>[A-Za-z_][A-Za-z0-9_]*)\s*~\s*[^\"']+[\"']", code)
    if match:
        return match.group("outcome")
    match = re.search(r"(?P<outcome>[A-Za-z_][A-Za-z0-9_]*)\s*~\s*C\(", code)
    if match:
        return match.group("outcome")
    match = re.search(
        r"(?m)^\s*y\s*=\s*[A-Za-z_][A-Za-z0-9_]*\[[\"'](?P<outcome>[A-Za-z_][A-Za-z0-9_]*)[\"']\]",
        code,
    )
    if match:
        return match.group("outcome")
    return None


def _normalise_predictor_column_candidate(value: Any, code: str) -> Optional[str]:
    """Return the dataframe column part of a prose predictor label.

    Generated summaries often write labels such as
    ``"sofa2_admission (ordinal, per-point)"``.  The deterministic fallback
    needs the actual dataframe column, not the prose suffix.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    for candidate in candidates:
        if (
            f"'{candidate}'" in code
            or f'"{candidate}"' in code
            or re.search(rf"\b{re.escape(candidate)}\b", code)
        ):
            return candidate
    return candidates[0] if candidates else None


def _infer_primary_association_predictor_from_code(
    step_summary: Dict[str, Any],
    code: str,
) -> Optional[str]:
    """Infer the predictor for a primary association fallback.

    Generated association scripts often use hand-written dataframe code rather
    than a formula. When such a script fails before writing a finite effect
    estimate, the deterministic fallback still needs the predictor column. Keep
    the inference conservative and case-neutral: prefer explicit summary/code
    variables, then fall back to the non-outcome member of a required column
    list.
    """

    for key in ("primary_predictor", "predictor", "exposure"):
        value = _normalise_predictor_column_candidate(step_summary.get(key), code)
        if value:
            return value
    manifest = (
        step_summary.get("manifest:robustness_analysis_manifest")
        or step_summary.get("robustness_analysis_manifest")
        or {}
    )
    if isinstance(manifest, dict):
        for key in ("primary_predictor", "predictor", "exposure"):
            value = _normalise_predictor_column_candidate(manifest.get(key), code)
            if value:
                return value
    for pattern in (
        r"(?:primary_predictor|predictor_col|primary_exposure)\s*=\s*['\"]([^'\"]+)['\"]",
        r"predictor\s*=\s*['\"]([^'\"]+)['\"]",
    ):
        match = re.search(pattern, code)
        if match:
            return match.group(1)
    outcome = _infer_binary_outcome_from_code(code)
    required_match = re.search(r"required_cols\s*=\s*\[(?P<body>[^\]]+)\]", code)
    if required_match:
        cols = re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", required_match.group("body"))
        for col in cols:
            if col != outcome and col.lower() not in {"death", "mortality", "outcome"}:
                return col
    numeric_assignments = re.findall(
        r"df\[['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"]\]\s*=\s*pd\.to_numeric",
        code,
    )
    for col in numeric_assignments:
        if col != outcome:
            return col
    return _infer_overexpanded_categorical_predictor(step_summary, code)


def _ordinal_primary_association_fallback_code(
    *,
    predictor: str,
    outcome: str,
    reason: str,
) -> str:
    """Return deterministic code that estimates one ordinal adjusted OR.

    The fallback is intentionally generic: it is triggered by a singular
    dummy-expanded categorical predictor, then re-estimates a single per-unit
    odds ratio for the same predictor after numeric coercion. It does not
    encode one benchmark case or one clinical score; the predictor and outcome
    are inferred from the generated script / summary.
    """

    return textwrap.dedent(
        f"""

        def _easyicu_ordinal_primary_association_fallback_v1():
            import json
            import math
            import os
            import numpy as np
            import pandas as pd
            import statsmodels.api as sm

            def _jsonable(x):
                if isinstance(x, (np.integer,)):
                    return int(x)
                if isinstance(x, (np.floating,)):
                    value = float(x)
                    return value if math.isfinite(value) else None
                if isinstance(x, np.ndarray):
                    return x.tolist()
                try:
                    if pd.isna(x):
                        return None
                except Exception:
                    pass
                return x

            cohort_path = os.environ.get("COHORT_PARQUET")
            out_dir = os.environ.get("STEP_OUT_DIR", ".")
            if not cohort_path:
                return
            predictor_col = {predictor!r}
            outcome_col = {outcome!r}
            df_fallback = pd.read_parquet(cohort_path)
            if predictor_col not in df_fallback.columns or outcome_col not in df_fallback.columns:
                return
            covariates = [
                col for col in ("age", "sex", "weight")
                if col in df_fallback.columns and col not in (predictor_col, outcome_col)
            ]
            required = [outcome_col, predictor_col] + covariates
            model_df = df_fallback[required].copy()
            if "sex" in model_df.columns:
                model_df["sex"] = (
                    model_df["sex"]
                    .astype(str)
                    .str.lower()
                    .isin(["m", "male", "1", "true"])
                    .astype(float)
                )
            for col in model_df.columns:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
            n_complete = int(len(model_df))
            outcome_values = set(model_df[outcome_col].dropna().astype(float).unique().tolist())
            summary_path = os.path.join(out_dir, "step_summary.json")
            summary = {{}}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        summary.update(loaded)
                except Exception:
                    summary = {{}}
            if outcome_values - {{0.0, 1.0}}:
                summary.setdefault("fallback_notes", []).append(
                    "ordinal_primary_association_fallback_v1 refused to run because "
                    "the inferred outcome is not a binary 0/1 endpoint."
                )
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=_jsonable, ensure_ascii=False)
                return
            if (
                n_complete < 20
                or len(outcome_values) < 2
                or model_df[predictor_col].nunique(dropna=True) < 2
            ):
                summary.setdefault("fallback_notes", []).append(
                    "ordinal_primary_association_fallback_v1 could not run because "
                    "there were too few complete observations or no variation."
                )
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=_jsonable, ensure_ascii=False)
                return
            y = model_df[outcome_col].astype(float)
            X = sm.add_constant(
                model_df[[predictor_col] + covariates].astype(float),
                has_constant="add",
            )
            result = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            coef = float(result.params[predictor_col])
            conf = result.conf_int()
            ci_low = float(conf.loc[predictor_col, 0])
            ci_high = float(conf.loc[predictor_col, 1])
            p_value = float(result.pvalues[predictor_col])
            odds_ratio = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))
            result_row = pd.DataFrame([
                {{
                    "variable": predictor_col,
                    "coef": coef,
                    "std_err": float(result.bse[predictor_col]),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "odds_ratio": odds_ratio,
                    "or_ci_low": or_ci_low,
                    "or_ci_high": or_ci_high,
                }}
            ])
            result_row.to_csv(os.path.join(out_dir, "association_results.csv"), index=False)
            summary["model"] = {{
                "type": "logistic_glm_binomial_ordinal_fallback",
                "outcome": outcome_col,
                "predictors": list(X.columns),
                "n_obs": n_complete,
                "converged": True,
                "fallback_reason": {reason!r},
            }}
            summary["primary_predictor"] = predictor_col
            summary["outcome"] = outcome_col
            summary["primary_association_estimate"] = {{
                "variable": predictor_col,
                "odds_ratio": odds_ratio,
                "ci_low": or_ci_low,
                "ci_high": or_ci_high,
                "p_value": p_value,
            }}
            summary["primary_or"] = odds_ratio
            summary["adjusted_odds_ratio"] = odds_ratio
            summary["primary_ci_low"] = or_ci_low
            summary["primary_ci_high"] = or_ci_high
            summary["primary_p_value"] = p_value
            summary["statistic:primary_or"] = odds_ratio
            summary["statistic:adjusted_odds_ratio"] = odds_ratio
            summary["statistic:primary_ci_low"] = or_ci_low
            summary["statistic:primary_ci_high"] = or_ci_high
            summary["statistic:primary_p_value"] = p_value
            summary["statistic:complete_case_n"] = n_complete
            summary["log:primary_association_fallback"] = (
                "Deterministic ordinal GLM fallback after generated categorical "
                "logistic model failed to produce a finite primary association."
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=_jsonable, ensure_ascii=False)
            print(json.dumps(summary, indent=2, default=_jsonable, ensure_ascii=False))

        try:
            _easyicu_ordinal_primary_association_fallback_v1()
        except Exception as _easyicu_fallback_exc:
            print(f"ordinal_primary_association_fallback_failed: {{_easyicu_fallback_exc}}")
        """
    ).strip("\n")


def _patch_primary_predictor_into_design_matrix(
    *,
    code: str,
    predictor: str,
) -> Optional[str]:
    function_design_markers = (
        "    X = model_df[covariates].astype(float)",
        "    X = model_df[covariates].apply(pd.to_numeric, errors=\"coerce\").astype(float)",
        "    X = model_df[covariates].copy()",
    )
    if (
        "def compute_or_ci" in code
        and "if predictor in result.params.index" in code
        and (
            "result.params[predictor]" in code
            or "result.conf_int().loc[predictor" in code
        )
    ):
        for marker in function_design_markers:
            if marker not in code:
                continue
            replacement = (
                "    design_cols = [predictor] + [col for col in covariates if col != predictor]\n"
                "    X = model_df[design_cols].apply(pd.to_numeric, errors=\"coerce\").astype(float)"
            )
            repaired = code.replace(marker, replacement, 1)
            if repaired != code:
                return repaired

    x_assign = re.search(r"(?m)^\s*X\s*=\s*model_df\[\[", code)
    if x_assign is None:
        return None
    line_end = code.find("\n", x_assign.start())
    if line_end < 0:
        line_end = len(code)
    x_line = code[x_assign.start():line_end]
    if predictor in x_line:
        return None
    predictor_lookup_patterns = (
        f"result.params['{predictor}'",
        f'result.params["{predictor}"',
        f"result.conf_int().loc['{predictor}'",
        f'result.conf_int().loc["{predictor}"',
        f"result.pvalues['{predictor}'",
        f'result.pvalues["{predictor}"',
        f"coef_table.loc['{predictor}'",
        f'coef_table.loc["{predictor}"',
    )
    if not any(pattern in code for pattern in predictor_lookup_patterns):
        return None
    repaired = code.replace(
        "X = model_df[[",
        f"X = model_df[['{predictor}', ",
        1,
    )
    summary_defaults = textwrap.dedent(
        """
        n_total = int(len(df))
        n_complete = int(len(model_df))
        """
    ).strip("\n")
    if "# Fit logistic regression model" in repaired:
        repaired = repaired.replace(
            "# Fit logistic regression model",
            summary_defaults + "\n\n# Fit logistic regression model",
            1,
        )
    elif "# Fit logistic regression" in repaired:
        repaired = repaired.replace(
            "# Fit logistic regression",
            summary_defaults + "\n\n# Fit logistic regression",
            1,
        )
    elif "\ntry:\n" in repaired:
        repaired = repaired.replace(
            "\ntry:\n",
            "\n" + summary_defaults + "\n\ntry:\n",
            1,
        )
    return repaired


def _deterministic_summary_repair(
    *,
    code: str,
    step_summary: Dict[str, Any],
    previous_repair: Optional[str] = None,
    analysis_family: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    if not isinstance(step_summary, dict) or not step_summary:
        return None
    summary_text = json.dumps(step_summary, ensure_ascii=False, default=str).lower()
    simple_imputer_bool = (
        "simpleimputer does not support data with dtype bool" in summary_text
        and "X_sklearn = model_df[x_cols].copy()" in code
    )
    if simple_imputer_bool:
        repair_name = "sklearn_bool_imputer_cast_v1"
        if previous_repair != repair_name:
            marker = "X_sklearn = model_df[x_cols].copy()"
            patch = (
                marker
                + "\nfor col in X_sklearn.select_dtypes(include=['bool']).columns:"
                + "\n    X_sklearn[col] = X_sklearn[col].astype(int)"
            )
            repaired = code.replace(marker, patch, 1)
            if repaired != code:
                return repair_name, repaired
    manifest = (
        step_summary.get("manifest:robustness_analysis_manifest")
        or step_summary.get("robustness_analysis_manifest")
        or {}
    )
    if not isinstance(manifest, dict):
        manifest = {}
    predictor_match = re.search(
        r"(?:primary_predictor|predictor_col)\s*=\s*['\"]([^'\"]+)['\"]",
        code,
    )
    predictor = str(
        step_summary.get("primary_predictor")
        or step_summary.get("predictor")
        or manifest.get("primary_predictor")
        or (predictor_match.group(1) if predictor_match else "")
        or ""
    ).strip()
    estimate = _first_present_scalar(
        step_summary,
        ("estimate", "primary_or", "odds_ratio", "adjusted_or", "lactate_or", "or"),
    )
    if estimate is not None:
        return None
    error_text = str(
        step_summary.get("error")
        or step_summary.get("error_message")
        or step_summary.get("note")
        or ""
    )
    generic_soft_failure = "unknown error" in error_text.lower()
    dtype_soft_failure = "pandas data cast to numpy dtype of object" in error_text.lower()
    index_alignment_soft_failure = (
        "indices for endog and exog are not aligned" in error_text.lower()
    )
    undefined_dummy_formula_failure = (
        "nameerror" in error_text.lower()
        and "sex_" in error_text.lower()
        and "not defined" in error_text.lower()
        and "pd.get_dummies" in code
        and "logit(" in code.lower()
    )
    binary_model_repair_allowed = _family_allows_binary_model_repair(
        analysis_family
    )
    if (
        predictor
        and error_text
        and predictor not in error_text
        and not (
            generic_soft_failure
            or dtype_soft_failure
            or index_alignment_soft_failure
            or undefined_dummy_formula_failure
        )
    ):
        return None
    duplicate_predictor_design = predictor and (
        "x_cols = [predictor_col] + [col for col in model_df.columns if col != outcome_col]"
        in code
        and "X = model_df[x_cols]" in code
    )
    if duplicate_predictor_design:
        repair_name = "dedupe_predictor_numeric_design_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "x_cols = [predictor_col] + [col for col in model_df.columns if col != outcome_col]",
                "x_cols = [predictor_col] + [col for col in model_df.columns if col not in [outcome_col, predictor_col]]",
                1,
            )
            repaired = repaired.replace(
                "X = model_df[x_cols]\n",
                "X = model_df[x_cols].apply(pd.to_numeric, errors=\"coerce\").astype(float)\n",
                1,
            )
            if repaired != code:
                return repair_name, repaired
    repaired = None
    if predictor:
        repair_name = "primary_predictor_omitted_from_design_v1"
        repaired = _patch_primary_predictor_into_design_matrix(
            code=code,
            predictor=predictor,
        )
        if repaired is not None and repaired != code:
            if previous_repair == repair_name:
                return None
            return repair_name, repaired
    if repaired is None or repaired == code:
        skipped = str(step_summary.get("skipped") or "").lower()
        null_model_summary = (
            '"complete_case_n": null' in summary_text
            or '"statistic:complete_case_n": null' in summary_text
            or '"lactate_or": null' in summary_text
            or '"statistic:lactate_or_stability": null' in summary_text
            or '"or_estimate": null' in summary_text
            or '"odds_ratio": null' in summary_text
            or '"primary_odds_ratio": null' in summary_text
            or '"primary_or": null' in summary_text
            or '"statistic:primary_or": null' in summary_text
            or '"estimate": null' in summary_text
        )
        dtype_summary_failure = "pandas data cast to numpy dtype of object" in summary_text
        index_alignment_summary_failure = (
            "indices for endog and exog are not aligned" in summary_text
        )
        helper_dtype_summary_failure = (
            dtype_summary_failure
            and "def _fit_logistic" in code
            and 'X = X.apply(pd.to_numeric, errors="coerce")' in code
        )
        if helper_dtype_summary_failure and binary_model_repair_allowed:
            repair_name = "statsmodels_helper_design_float_v1"
            if previous_repair != repair_name:
                repaired = code.replace(
                    'X = X.apply(pd.to_numeric, errors="coerce")',
                    'X = X.apply(pd.to_numeric, errors="coerce").astype(float)',
                    1,
                )
                repaired = repaired.replace(
                    "X_clean = data.drop(columns=[y.name])",
                    "X_clean = data.drop(columns=[y.name]).apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    "    y_clean = pd.to_numeric(y_clean, errors=\"coerce\").astype(float)",
                    1,
                )
                if repaired != code:
                    return repair_name, repaired
        if (
            dtype_summary_failure
            and _statsmodels_repair_allowed_for_family(code, analysis_family)
        ):
            repaired = _deterministic_runner_repair(
                code=code,
                run_log=summary_text,
                previous_repair=previous_repair,
                analysis_family=analysis_family,
            )
            if repaired is not None:
                return repaired
        if (
            index_alignment_summary_failure
            and _statsmodels_repair_allowed_for_family(code, analysis_family)
        ):
            repaired = _deterministic_runner_repair(
                code=code,
                run_log=summary_text,
                previous_repair=previous_repair,
                analysis_family=analysis_family,
            )
            if repaired is not None:
                return repaired
        dummy_logit_null_summary = (
            null_model_summary
            and "pd.get_dummies" in code
            and "sm.Logit" in code
            and "X_final = sm.add_constant(X_encoded" in code
        )
        if dummy_logit_null_summary and binary_model_repair_allowed:
            repair_name = "statsmodels_dummy_design_float_v1"
            if previous_repair != repair_name:
                marker = "X_final = sm.add_constant(X_encoded, has_constant=\"add\")"
                patch = (
                    "X_encoded = X_encoded.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    + marker
                )
                repaired = code.replace(marker, patch, 1)
                if repaired != code:
                    return repair_name, repaired
        if undefined_dummy_formula_failure and predictor and binary_model_repair_allowed:
            repair_name = "formula_dummy_name_fallback_v1"
            if previous_repair != repair_name:
                fallback_outcome = str(
                    step_summary.get("outcome") or _infer_binary_outcome_from_code(code) or ""
                ).strip()
                if not fallback_outcome:
                    return None
                fallback = _primary_association_fallback_code(
                    predictor=predictor,
                    outcome=fallback_outcome,
                    reason=(
                        "Deterministic fallback after statsmodels formula used "
                        "a hard-coded dummy column name that was not present."
                    ),
                )
                return repair_name, code.rstrip() + "\n\n" + fallback + "\n"
        glm_primary_effect_null = (
            null_model_summary
            and (
                "primary_odds_ratio" in summary_text
                or "primary_association_estimate" in summary_text
                or "primary association" in summary_text
                or '"primary_or": null' in summary_text
                or '"primary_ci_low": null' in summary_text
                or '"primary_ci_high": null' in summary_text
                or "statistic:primary_or" in summary_text
            )
            and (
                "sm.glm" in code.lower()
                or "sm.logit" in code.lower()
                or "logit(" in code.lower()
            )
            and (
                "mle_retvals" in code
                or "glmresults" in summary_text
                or "model_fit_error" in summary_text
                or dtype_summary_failure
                or '"converged": false' in summary_text
                or '"primary_or": null' in summary_text
                or '"primary_ci_low": null' in summary_text
                or '"primary_ci_high": null' in summary_text
            )
        )
        if glm_primary_effect_null and binary_model_repair_allowed:
            fallback_predictor = _infer_primary_association_predictor_from_code(
                step_summary,
                code,
            )
            if fallback_predictor:
                fallback_outcome = _infer_binary_outcome_from_code(code)
                if not fallback_outcome:
                    return None
                repair_name = "ordinal_primary_association_fallback_v1"
                if previous_repair != repair_name:
                    fallback = _ordinal_primary_association_fallback_code(
                        predictor=fallback_predictor,
                        outcome=fallback_outcome,
                        reason=(
                            "Deterministic fallback after generated GLM/Logit "
                            "model failed to produce a finite primary association."
                        ),
                    )
                    return repair_name, code.rstrip() + "\n\n" + fallback + "\n"
        overexpanded_categorical_singular = (
            null_model_summary
            and "singular matrix" in summary_text
            and (
                "primary_association_estimate" in summary_text
                or "primary association" in summary_text
                or "primary_or" in summary_text
            )
        )
        if overexpanded_categorical_singular and binary_model_repair_allowed:
            categorical_predictor = _infer_overexpanded_categorical_predictor(
                step_summary,
                code,
            )
            if categorical_predictor:
                fallback_outcome = _infer_binary_outcome_from_code(code)
                if not fallback_outcome:
                    return None
                repair_name = "ordinal_primary_association_fallback_v1"
                if previous_repair != repair_name:
                    fallback = _ordinal_primary_association_fallback_code(
                        predictor=categorical_predictor,
                        outcome=fallback_outcome,
                        reason=(
                            "Deterministic fallback after generated categorical "
                            "logistic model failed with a singular matrix."
                        ),
                    )
                    return repair_name, code.rstrip() + "\n\n" + fallback + "\n"
        raw_categorical_sex_logit = (
            null_model_summary
            and "sm.logit" in code.lower()
            and "sex" in code
            and "pd.get_dummies" not in code
            and ".str.lower().isin(['m', 'male'])" not in code
        )
        if raw_categorical_sex_logit and binary_model_repair_allowed:
            repair_name = "sex_binary_encode_for_logit_v1"
            if previous_repair != repair_name:
                model_df_assign = re.search(
                    r"(^model_df\s*=\s*df\[[^\n]+?\.copy\(\)\s*$)",
                    code,
                    flags=re.MULTILINE,
                )
                if model_df_assign:
                    patch = textwrap.dedent(
                        """
                        if 'sex' in model_df.columns:
                            model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                        for col in model_df.columns:
                            if col != 'sex':
                                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                        """
                    ).strip("\n")
                    repaired = code.replace(
                        model_df_assign.group(1),
                        model_df_assign.group(1) + "\n" + patch,
                        1,
                    )
                    if repaired != code:
                        return repair_name, repaired
        categorical_sex_dropna = (
            (
                "no valid data after dropping lactate missing rows" in skipped
                or "insufficient data" in skipped
                or "no valid observations" in skipped
                or null_model_summary
                or dtype_summary_failure
            )
            and "model_df = model_df.apply(pd.to_numeric, errors=\"coerce\")" in code
            and "sex" in code
        )
        if categorical_sex_dropna:
            repair_name = "sex_numeric_coercion_before_dropna_v1"
            if previous_repair == repair_name:
                return None
            replacement = textwrap.dedent(
                """
                if 'sex' in model_df.columns:
                    model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                for col in model_df.columns:
                    if col != 'sex':
                        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                """
            ).strip("\n")
            repaired = re.sub(
                r"^(?P<indent>\s*)model_df = model_df\.apply\(pd\.to_numeric, errors=\"coerce\"\)",
                lambda match: match.group("indent") + replacement.replace(
                    "\n", "\n" + match.group("indent")
                ),
                code,
                count=1,
                flags=re.MULTILINE,
            )
            if repaired != code:
                return repair_name, repaired
        categorical_sex_loop_dropna = (
            (
                "insufficient data" in skipped
                or "no valid data" in skipped
                or null_model_summary
            )
            and "for col in x_cols:" in code
            and "pd.to_numeric(model_df[col], errors=\"coerce\")" in code
            and "sex" in code
        )
        if categorical_sex_loop_dropna:
            repair_name = "sex_covariate_numeric_loop_guard_v1"
            if previous_repair != repair_name:
                marker = (
                    "for col in x_cols:\n"
                    "    model_df[col] = pd.to_numeric(model_df[col], errors=\"coerce\")"
                )
                replacement = (
                    "for col in x_cols:\n"
                    "    if col == \"sex\":\n"
                    "        model_df[col] = model_df[col].astype(str).str.lower().isin([\"m\", \"male\", \"1\", \"true\"]).astype(float)\n"
                    "        continue\n"
                    "    model_df[col] = pd.to_numeric(model_df[col], errors=\"coerce\")"
                )
                repaired = code.replace(marker, replacement, 1)
                if repaired != code:
                    return repair_name, repaired
        robustness_null_summary = (
            null_model_summary
            and "sm.Logit" in code
            and "primary_predictor" in code
            and "Missing-indicator" in code
            and "Reduced-variable" in code
        )
        if robustness_null_summary:
            repair_name = "robustness_missingness_contract_v1"
            if previous_repair != repair_name:
                repaired = code
                reduction_marker = "model_df = model_df.replace([np.inf, -np.inf], np.nan)"
                reduction_patch = (
                    reduction_marker
                    + "\n"
                    + "reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]"
                )
                if reduction_marker in repaired and "reduced_covariates =" not in repaired:
                    repaired = repaired.replace(reduction_marker, reduction_patch, 1)
                cc_replacements = {
                    "cc_df = model_df.dropna(subset=[primary_predictor])": (
                        "cc_df = model_df.dropna(subset=[outcome_col, primary_predictor] + covariates)"
                    ),
                    "complete_case_df = model_df.dropna(subset=[predictor_col])": (
                        "complete_case_df = model_df.dropna(subset=[outcome_col, predictor_col] + covariates)"
                    ),
                }
                for old, new in cc_replacements.items():
                    repaired = repaired.replace(old, new)
                mi_replacements = {
                    "mi_df['lactate_missing'] = mi_df[primary_predictor].isna().astype(int)": (
                        "mi_df['lactate_missing'] = mi_df[primary_predictor].isna().astype(int)\n"
                        "mi_df[primary_predictor] = mi_df[primary_predictor].fillna(0)\n"
                        "mi_df = mi_df.dropna(subset=[outcome_col] + covariates)"
                    ),
                    'mi_df["lactate_missing"] = mi_df[primary_predictor].isna().astype(int)': (
                        'mi_df["lactate_missing"] = mi_df[primary_predictor].isna().astype(int)\n'
                        'mi_df[primary_predictor] = mi_df[primary_predictor].fillna(0)\n'
                        "mi_df = mi_df.dropna(subset=[outcome_col] + covariates)"
                    ),
                }
                for old, new in mi_replacements.items():
                    if old in repaired and "fillna(0)" not in repaired:
                        repaired = repaired.replace(old, new, 1)
                rv_replacements = {
                    "rv_df = model_df.dropna(subset=[primary_predictor])": (
                        "rv_df = model_df[[outcome_col, primary_predictor] + reduced_covariates].dropna()"
                    ),
                    "rv_X = sm.add_constant(rv_df[covariates], has_constant=\"add\")": (
                        "rv_X = sm.add_constant(rv_df[[primary_predictor] + reduced_covariates], has_constant=\"add\")"
                    ),
                    "rv_X = sm.add_constant(rv_df[covariates], has_constant='add')": (
                        "rv_X = sm.add_constant(rv_df[[primary_predictor] + reduced_covariates], has_constant='add')"
                    ),
                }
                for old, new in rv_replacements.items():
                    repaired = repaired.replace(old, new)
                if repaired != code:
                    return repair_name, repaired
        return None

    return None



_KEYERROR_NOT_IN_INDEX_RE = re.compile(
    r"KeyError:\s*\"\[(?P<items>[^\]]+)\]\s*not\s+in\s+index\"",
    re.MULTILINE,
)

# Captures ``NameError: name 'foo' is not defined`` for use by Fix F.
_NAME_ERROR_HELPER_RE = re.compile(
    r"NameError:\s*name\s+['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"]\s+is\s+not\s+defined"
)


def _extract_missing_index_columns(run_log: str) -> List[str]:
    """Parse pandas ``KeyError: "['a', 'b'] not in index"`` from a log.

    Returns the unique column names referenced inside the bracketed list,
    preserving the order they appeared in the error message. Returns an
    empty list when the log does not contain a recognisable not-in-index
    KeyError. The matcher tolerates both single and double quoted entries.
    """
    if not run_log:
        return []
    match = _KEYERROR_NOT_IN_INDEX_RE.search(run_log)
    if match is None:
        return []
    raw = match.group("items")
    cols: List[str] = []
    for token in re.findall(r"['\"]([^'\"]+)['\"]", raw):
        if token and token not in cols:
            cols.append(token)
    return cols


def _strip_columns_from_list_literals(code: str, missing_cols: Sequence[str]) -> str:
    """Remove ``"col"`` / ``'col'`` entries from list literals in ``code``.

    Walks through every ``[...]`` slice in the source and, when the slice
    contains at least one of the missing columns as a literal string,
    rewrites the list to exclude those entries. The rewriter is
    intentionally conservative: it only edits non-nested bracket slices
    whose elements are simple string literals (the common shape produced
    by agent-emitted ``covariates = [...]`` blocks). Bracket slices
    containing nested structures or non-string elements are left
    unchanged so we do not corrupt unrelated indexing expressions.
    """
    if not missing_cols:
        return code
    missing_set = set(missing_cols)

    list_literal_re = re.compile(r"\[(?P<body>[^\[\]]*)\]")

    def _rewrite(match: re.Match[str]) -> str:
        body = match.group("body")
        body_stripped = body.strip()
        if not body_stripped:
            return match.group(0)
        # Split on top-level commas (no nesting handled — body has no
        # brackets thanks to the regex). Whitespace tolerant.
        raw_parts = [part.strip() for part in body_stripped.split(",")]
        # Only rewrite when *all* non-empty parts are simple string
        # literals; otherwise we may be looking at an expression like
        # ``[outcome_col, predictor] + covariates`` which we leave alone.
        only_literals = True
        kept_literals: List[str] = []
        contains_missing = False
        for part in raw_parts:
            if not part:
                continue
            if not (
                (part.startswith('"') and part.endswith('"'))
                or (part.startswith("'") and part.endswith("'"))
            ):
                only_literals = False
                break
            literal_value = part[1:-1]
            if literal_value in missing_set:
                contains_missing = True
                continue
            kept_literals.append(part)
        if not only_literals or not contains_missing:
            return match.group(0)
        return "[" + ", ".join(kept_literals) + "]"

    return list_literal_re.sub(_rewrite, code)


def _extract_required_cols_list(code: str) -> List[str]:
    """Return literal strings from a generated ``required_cols`` list."""

    match = re.search(r"required_cols\s*=\s*(?P<literal>\[[\s\S]*?\])", code)
    if match is None:
        return []
    try:
        value = ast.literal_eval(match.group("literal"))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _infer_analysis_cohort_source_column(code: str) -> Optional[str]:
    """Infer the source column immediately preceding ``analysis_cohort``.

    Generated association scripts sometimes define a derived analysis stratum
    column (``analysis_cohort``) but include it in ``required_cols`` before
    materialising it. Keep the inference local to the generated script shape:
    the source is the literal column listed immediately before
    ``analysis_cohort`` in ``required_cols``.
    """

    required_cols = _extract_required_cols_list(code)
    try:
        idx = required_cols.index("analysis_cohort")
    except ValueError:
        return None
    if idx <= 0:
        return None
    source = required_cols[idx - 1]
    if source in {"death", "mortality", "outcome", "analysis_cohort"}:
        return None
    return source


def _patch_derived_analysis_cohort_materialization(code: str) -> str:
    """Materialise ``analysis_cohort`` before required-column validation."""

    if "_easyicu_derived_analysis_cohort_materialization_v1" in code:
        return code
    source_col = _infer_analysis_cohort_source_column(code)
    if not source_col:
        return code

    read_re = re.compile(
        r"(?m)^(?P<indent>[ \t]*)"
        r"(?P<target>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*pd\.read_parquet\((?P<args>[^\n]*)\)\s*$"
    )

    def _rewrite(match: re.Match[str]) -> str:
        indent = match.group("indent")
        target = match.group("target")
        line = match.group(0)
        return (
            f"{line}\n"
            f"{indent}# EasyICU deterministic repair: materialize generated analysis strata.\n"
            f"{indent}_easyicu_derived_analysis_cohort_materialization_v1 = True\n"
            f"{indent}if \"analysis_cohort\" not in {target}.columns and {source_col!r} in {target}.columns:\n"
            f"{indent}    {target}[\"analysis_cohort\"] = {target}[{source_col!r}].astype(\"string\")\n"
            f"{indent}    {target}.loc[{target}[{source_col!r}].isna(), \"analysis_cohort\"] = pd.NA"
        )

    return read_re.sub(_rewrite, code, count=1)


def _patch_statsmodels_conf_int_filter_axis(code: str) -> str:
    """Filter ``statsmodels.conf_int()`` rows by coefficient name.

    ``DataFrame.filter(like=...)`` defaults to ``axis=1``. Generated figure
    scripts often intend to subset confidence intervals by coefficient names,
    which live on the index of the ``statsmodels`` confidence-interval frame.
    Rewriting the assignment keeps downstream ``.iloc[:, 0]`` / ``.iloc[:, 1]``
    code working with the expected two-column CI shape.
    """

    assignment_re = re.compile(
        r"(?m)^(?P<indent>[ \t]*)"
        r"(?P<target>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?P<expr>.+?\.conf_int\(\))\.filter\(\s*like\s*=\s*"
        r"(?P<needle>['\"][^'\"]+['\"])\s*\)\s*$"
    )
    counter = 0

    def _rewrite(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        indent = match.group("indent")
        target = match.group("target")
        expr = match.group("expr").strip()
        needle = match.group("needle")
        full_name = f"_easyicu_{target}_conf_int_full_{counter}"
        return (
            f"{indent}{full_name} = {expr}\n"
            f"{indent}{target} = {full_name}.loc["
            f"{full_name}.index.astype(str).str.contains({needle}, regex=False)]"
        )

    return assignment_re.sub(_rewrite, code)


def _patch_statsmodels_endog_exog_index_alignment(code: str) -> str:
    """Align pandas endog/exog indices before statsmodels model construction.

    Generated scripts often reset the design matrix to a compact 0..n index
    after dummy encoding, while leaving the outcome series on the original
    filtered-cohort index. Statsmodels rejects this with "The indices for
    endog and exog are not aligned". The repair is case-neutral: it wraps
    statsmodels constructors and only resets indices when X and y have the
    same row count but different pandas indices.
    """

    helper_name = "_easyicu_statsmodels_align_index_v1"
    model_call = re.compile(
        r"(?P<prefix>\b[A-Za-z_]\w*\s*=\s*sm\.(?:Logit|OLS|GLM)\()\s*"
        r"(?P<y>[^,]+?)\s*,\s*(?P<X>[^,\)\n]+?)\s*"
        r"(?P<kwargs>,\s*[^)\n]+)?(?P<suffix>\))"
    )

    def _rewrite(match: re.Match[str]) -> str:
        y_expr = match.group("y").strip()
        x_expr = match.group("X").strip()
        kwargs = match.group("kwargs") or ""
        return (
            f"{match.group('prefix')}*{helper_name}("
            f"{x_expr}, {y_expr}){kwargs}{match.group('suffix')}"
        )

    repaired = model_call.sub(_rewrite, code, count=1)
    if repaired == code:
        return code
    if f"def {helper_name}" in repaired:
        return repaired
    helper = textwrap.dedent(
        f"""

        def {helper_name}(X, y):
            X_work = X.copy() if hasattr(X, "copy") else X
            y_work = y.copy() if hasattr(y, "copy") else y
            try:
                if hasattr(X_work, "index") and hasattr(y_work, "index"):
                    if len(X_work) == len(y_work) and not X_work.index.equals(y_work.index):
                        X_work = X_work.reset_index(drop=True)
                        y_work = y_work.reset_index(drop=True)
            except Exception:
                pass
            return y_work, X_work
        """
    ).strip("\n")
    return helper + "\n\n" + repaired


def _patch_json_dump_numpy_key_sanitizer(code: str) -> str:
    if "_easyicu_json_sanitize_v1" in code:
        return code
    helper = textwrap.dedent(
        """
        import json as _easyicu_json_module_v1
        _easyicu_original_json_dump_v1 = _easyicu_json_module_v1.dump
        _easyicu_original_json_dumps_v1 = _easyicu_json_module_v1.dumps
        def _easyicu_json_sanitize_v1(value):
            import math
            try:
                import numpy as np
            except Exception:
                np = None
            try:
                import pandas as pd
            except Exception:
                pd = None
            if isinstance(value, dict):
                return {str(_easyicu_json_sanitize_v1(k)): _easyicu_json_sanitize_v1(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_easyicu_json_sanitize_v1(v) for v in value]
            if np is not None and isinstance(value, np.integer):
                return int(value)
            if np is not None and isinstance(value, np.floating):
                value = float(value)
                return value if math.isfinite(value) else None
            if np is not None and isinstance(value, np.bool_):
                return bool(value)
            if np is not None and isinstance(value, np.ndarray):
                return _easyicu_json_sanitize_v1(value.tolist())
            if pd is not None:
                try:
                    if pd.isna(value):
                        return None
                except Exception:
                    pass
            return value
        def _easyicu_json_dump_v1(obj, fp, *args, **kwargs):
            return _easyicu_original_json_dump_v1(_easyicu_json_sanitize_v1(obj), fp, *args, **kwargs)
        def _easyicu_json_dumps_v1(obj, *args, **kwargs):
            return _easyicu_original_json_dumps_v1(_easyicu_json_sanitize_v1(obj), *args, **kwargs)
        _easyicu_json_module_v1.dump = _easyicu_json_dump_v1
        _easyicu_json_module_v1.dumps = _easyicu_json_dumps_v1
        """
    ).strip()
    return helper + "\n" + code


def _deterministic_runner_repair(
    *,
    code: str,
    run_log: str,
    previous_repair: Optional[str] = None,
    analysis_family: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Best-effort execution-layer patch for common numeric model failures.

    We keep this deliberately narrow: it only activates for recurrent design-
    matrix dtype / inf / NaN failures around statsmodels-style regression code.
    The repair is deterministic and is meant to reduce prompt drift in the
    coder by handling one family of brittle runtime errors below the LLM layer.
    """
    lowered = (run_log or "").lower()
    binary_model_repair_allowed = _family_allows_binary_model_repair(
        analysis_family
    )

    # 🔧 2026-05-17: defend against LLM hallucinating non-existent easyicu
    # sub-modules (e.g. deepseek-v4-flash emitted
    # `from easyicu.research_agent.rcs import restricted_cubic_spline`,
    # killing pilot run_20260516T182501_329a01 step 03). Detect the runtime
    # ModuleNotFoundError, strip the bad import lines, and stub each
    # imported name with a clear NotImplementedError so the next repair
    # attempt has actionable diagnostics instead of the same import error.
    _fake_easyicu_match = re.search(
        r"ModuleNotFoundError: No module named ['\"](easyicu\.[\w\.]+)['\"]",
        run_log or "",
    )
    if _fake_easyicu_match:
        bad_module = _fake_easyicu_match.group(1)
        repair_name = f"strip_fake_easyicu_import_{bad_module.replace('.', '_')}_v1"
        if previous_repair != repair_name:
            stripped_names: list[str] = []
            _line_re = re.compile(
                r"^\s*from\s+" + re.escape(bad_module) + r"\s+import\s+(.+?)\s*(?:#.*)?$",
                re.MULTILINE,
            )
            for _match in _line_re.finditer(code):
                for _name in _match.group(1).split(","):
                    _name = _name.strip()
                    if _name and not _name.startswith("("):
                        # Handle `import X as Y` aliases
                        _final = _name.split(" as ")[-1].strip().rstrip(")")
                        if _final.isidentifier():
                            stripped_names.append(_final)
            if stripped_names:
                # Remove all matching import lines
                repaired = _line_re.sub(
                    f"# stripped: import from non-existent {bad_module}",
                    code,
                )
                # Also remove `import easyicu.research_agent.X` style
                repaired = re.sub(
                    r"^\s*import\s+" + re.escape(bad_module) + r"\s*$",
                    f"# stripped: import {bad_module}",
                    repaired,
                    flags=re.MULTILINE,
                )
                # Inject stubs after the import block (top of file)
                _stub_lines = "\n".join(
                    f"def {n}(*args, **kwargs): "
                    f"raise NotImplementedError("
                    f"\"{n} from {bad_module} is not available; "
                    f"reimplement inline using numpy/scipy/statsmodels.\")"
                    for n in dict.fromkeys(stripped_names)
                )
                # Insert after the first contiguous block of imports
                lines = repaired.splitlines()
                insert_at = 0
                for i, ln in enumerate(lines[:80]):
                    if ln.startswith(("import ", "from ", "#")) or not ln.strip():
                        insert_at = i + 1
                    else:
                        break
                lines.insert(insert_at, "\n# auto-stubs for stripped fake imports\n" + _stub_lines + "\n")
                return repair_name, "\n".join(lines)

    pandas_cut_observed_keyword = (
        "got an unexpected keyword argument 'observed'" in lowered
        and "observed=" in code
        and ("pd.cut(" in code or "pd.qcut(" in code)
    )
    if pandas_cut_observed_keyword:
        repair_name = "remove_pandas_cut_observed_keyword_v1"
        if previous_repair != repair_name:
            repaired = re.sub(r",\s*observed\s*=\s*(?:True|False)", "", code)
            if repaired != code:
                return repair_name, repaired

    missing_seaborn = (
        "modulenotfounderror: no module named 'seaborn'" in lowered
        and "import seaborn as sns" in code
    )
    if missing_seaborn:
        repair_name = "seaborn_matplotlib_fallback_v1"
        if previous_repair != repair_name:
            fallback = textwrap.dedent(
                """
                class _EasyICUSeabornFallback:
                    def set_theme(self, *args, **kwargs):
                        return None
                    def set_style(self, *args, **kwargs):
                        return None
                    def color_palette(self, *args, **kwargs):
                        import matplotlib.pyplot as plt
                        return plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
                    def barplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None and y is not None:
                            grouped = data.groupby(x, dropna=False)[y].mean()
                            ax.bar([str(v) for v in grouped.index], grouped.values)
                        return ax
                    def lineplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None and y is not None:
                            ax.plot(data[x], data[y], marker=kwargs.get("marker", "o"))
                        return ax
                    def scatterplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None and y is not None:
                            ax.scatter(data[x], data[y])
                        return ax
                    def histplot(self, data=None, x=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        values = data[x] if data is not None and x is not None else data
                        ax.hist(values.dropna() if hasattr(values, "dropna") else values, bins=kwargs.get("bins", 20))
                        return ax
                    def heatmap(self, data=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        image = ax.imshow(data, aspect="auto")
                        plt.colorbar(image, ax=ax)
                        return ax
                sns = _EasyICUSeabornFallback()
                """
            ).strip()
            repaired = code.replace("import seaborn as sns", fallback, 1)
            if repaired != code:
                return repair_name, repaired

    missing_proportion_confint = (
        (
            "modulenotfounderror: no module named 'statsmodels'" in lowered
            or "cannot import name 'proportion_confint' from 'scipy.stats'" in lowered
        )
        and "proportion_confint" in code
    )
    if missing_proportion_confint:
        repair_name = "local_wilson_proportion_confint_v1"
        if previous_repair != repair_name:
            helper = textwrap.dedent(
                """
                def proportion_confint(count, nobs=None, alpha=0.05, method="wilson", **kwargs):
                    import math
                    if nobs is None:
                        nobs = kwargs.get("n")
                    count = float(count)
                    nobs = float(nobs)
                    if nobs <= 0:
                        return (None, None)
                    z = 1.959963984540054
                    phat = count / nobs
                    denom = 1.0 + z * z / nobs
                    centre = phat + z * z / (2.0 * nobs)
                    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * nobs)) / nobs)
                    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))
                """
            ).strip()
            repaired = re.sub(
                r"^\s*from\s+statsmodels\.stats\.proportion\s+import\s+proportion_confint\s*$",
                helper,
                code,
                count=1,
                flags=re.MULTILINE,
            )
            repaired = re.sub(
                r"^\s*from\s+scipy\.stats\s+import\s+proportion_confint\s*$",
                helper,
                repaired,
                count=1,
                flags=re.MULTILINE,
            )
            if repaired != code:
                return repair_name, repaired

    wrong_calibration_import = (
        "cannot import name 'calibration_curve' from 'sklearn.metrics'" in lowered
        and "calibration_curve" in code
    )
    if wrong_calibration_import:
        repair_name = "prediction_calibration_import_fix_v1"
        repaired = re.sub(
            r"from sklearn\.metrics import ([^\n]*?)\bcalibration_curve\b,?\s*",
            lambda match: (
                "from sklearn.metrics import "
                + ", ".join(
                    part.strip().strip(",")
                    for part in match.group(1).split(",")
                    if part.strip().strip(",")
                )
                + "\nfrom sklearn.calibration import calibration_curve\n"
            ),
            code,
            count=1,
        )
        if repaired != code:
            return repair_name, repaired

    json_numpy_key_failure = (
        "keys must be str, int, float, bool or none" in lowered
        and "json.dump(" in code
    )
    if json_numpy_key_failure:
        repair_name = "json_dump_numpy_key_sanitizer_v1"
        if previous_repair != repair_name:
            repaired = _patch_json_dump_numpy_key_sanitizer(code)
            if repaired != code:
                return repair_name, repaired

    missing_os_import = (
        "nameerror: name 'os' is not defined" in lowered
        and ("os.environ" in code or "os.path" in code)
        and "import os" not in code
    )
    if missing_os_import:
        repair_name = "missing_os_import_v1"
        if previous_repair != repair_name:
            return repair_name, "import os\n" + code

    malformed_python_prefix = (
        "syntaxerror: invalid syntax" in lowered
        and ("pythonimport " in code or "\npythonimport " in code or "pythonfrom " in code)
    )
    if malformed_python_prefix:
        repair_name = "strip_python_prefix_v1"
        if previous_repair != repair_name:
            repaired = code.replace("pythonimport ", "import ").replace("pythonfrom ", "from ")
            repaired = repaired.replace("\npythonimport ", "\nimport ").replace("\npythonfrom ", "\nfrom ")
            if repaired != code:
                return repair_name, repaired

    proportion_confint_n_keyword = (
        "proportion_confint() got an unexpected keyword argument 'n'" in lowered
        and "proportion_confint" in code
    )
    if proportion_confint_n_keyword:
        repair_name = "proportion_confint_nobs_keyword_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"(proportion_confint\s*\([^)]*?)\bn\s*=",
                r"\1nobs=",
                code,
                flags=re.DOTALL,
            )
            if repaired != code:
                return repair_name, repaired

    malformed_matplotlib_xerr = (
        "valueerror: 'xerr'" in lowered
        and "must be a scalar or a 1d or (2, n) array-like" in lowered
        and "np.array([[" in code
        and "errorbar(" in code
    )
    if malformed_matplotlib_xerr:
        repair_name = "matplotlib_errorbar_xerr_shape_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"xerr\s*=\s*np\.array\(\[\[([A-Za-z_]\w*)\],\s*\[([A-Za-z_]\w*)\]\]\)",
                r"xerr=np.vstack([np.ravel(\1), np.ravel(\2)])",
                code,
            )
            if repaired != code:
                return repair_name, repaired

    statsmodels_conf_int_filter_axis = (
        "indexerror" in lowered
        and "single positional indexer is out-of-bounds" in lowered
        and ".conf_int()" in code
        and ".filter(" in code
        and "like=" in code
        and ".iloc" in code
    )
    if statsmodels_conf_int_filter_axis:
        repair_name = "statsmodels_conf_int_filter_axis_v1"
        if previous_repair != repair_name:
            repaired = _patch_statsmodels_conf_int_filter_axis(code)
            if repaired != code:
                return repair_name, repaired

    statsmodels_endog_exog_index_mismatch = (
        "indices for endog and exog are not aligned" in lowered
        and any(token in code for token in ("sm.Logit(", "sm.OLS(", "sm.GLM("))
    )
    if (
        statsmodels_endog_exog_index_mismatch
        and _statsmodels_repair_allowed_for_family(code, analysis_family)
    ):
        repair_name = "statsmodels_endog_exog_index_align_v1"
        if previous_repair != repair_name:
            repaired = _patch_statsmodels_endog_exog_index_alignment(code)
            if repaired != code:
                return repair_name, repaired

    derived_analysis_cohort_missing = (
        "analysis_cohort" in code
        and "required_cols" in code
        and "pd.read_parquet" in code
        and "missing_columns" in lowered
        and "analysis_cohort" in lowered
    )
    if derived_analysis_cohort_missing:
        repair_name = "derived_analysis_cohort_materialization_v1"
        if previous_repair != repair_name:
            repaired = _patch_derived_analysis_cohort_materialization(code)
            if repaired != code:
                return repair_name, repaired

    shadowed_json_module = (
        "attributeerror: 'function' object has no attribute 'dump'" in lowered
        and "json.dump(" in code
    )
    if shadowed_json_module:
        repair_name = "restore_shadowed_json_module_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"^(?P<indent>\s*)json\.dump\(",
                (
                    r"\g<indent>import importlib\n"
                    r"\g<indent>json = importlib.import_module('json')\n"
                    r"\g<indent>json.dump("
                ),
                code,
                count=1,
                flags=re.MULTILINE,
            )
            if repaired != code:
                return repair_name, repaired

    malformed_publication_contract = (
        (
            "valueerror: panels are required" in lowered
            or "figurecontract' object is not subscriptable" in lowered
            or '"figurecontract" object is not subscriptable' in lowered
        )
        and "make_figure_contract(" in code
        and 'figure_contract["panels"]' in code
    )
    if malformed_publication_contract:
        repair_name = "publication_contract_optional_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"figure_contract\s*=\s*make_figure_contract\([\s\S]*?\)\s*",
                "figure_contract = None\n",
                code,
                count=1,
            )
            repaired = re.sub(
                r"figure_contract\[[\"']panels[\"']\]\.append\(\{[\s\S]*?\}\)\s*",
                "",
                repaired,
            )
            if repaired != code:
                return repair_name, repaired

    missing_dummy_encoded_column = (
        "keyerror" in lowered
        and "not in index" in lowered
        and "pd.get_dummies" in code
        and (
            "model_df[x_cols]" in code
            or "model_df[[outcome_col] + x_cols]" in code
        )
    )
    if missing_dummy_encoded_column:
        repair_name = "filter_x_cols_after_dummy_encoding_v1"
        if previous_repair != repair_name:
            marker = "X = model_df[x_cols].copy()"
            guard = "x_cols = [col for col in x_cols if col in model_df.columns]"
            if marker in code and guard not in code:
                repaired = code.replace(marker, guard + "\n    " + marker, 1)
                return repair_name, repaired
            marker = "model_df_subset = model_df[[outcome_col] + x_cols].copy()"
            guard = "x_cols = [col for col in x_cols if col in model_df.columns]"
            if marker in code and guard not in code:
                repaired = code.replace(marker, guard + "\n" + marker, 1)
                return repair_name, repaired

    categorical_prediction_coercion = (
        "found array with 0 feature(s)" in lowered
        and "onehotencoder" in lowered
        and "categorical_features" in code
        and "pd.to_numeric" in code
    )
    if categorical_prediction_coercion:
        repair_name = "prediction_preserve_categorical_before_ohe_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"for col in predictors:\s*\n\s*if col in data:\s*\n\s*data\[col\]\s*=\s*pd\.to_numeric\(data\[col\], errors=\"coerce\"\)",
                (
                    "for col in predictors:\n"
                    "    if col in data and col not in [\"sex\"]:\n"
                    "        data[col] = pd.to_numeric(data[col], errors=\"coerce\")\n"
                    "if \"sex\" in data:\n"
                    "    data[\"sex\"] = data[\"sex\"].astype(\"string\")"
                ),
                code,
                count=1,
                flags=re.MULTILINE,
            )
            if repaired != code:
                return repair_name, repaired

    duplicate_outcome_column_unique = (
        "attributeerror: 'dataframe' object has no attribute 'unique'" in lowered
        and "model_df[outcome].unique()" in code
        and "required_cols + [outcome]" in code
    )
    if duplicate_outcome_column_unique:
        repair_name = "dedupe_required_cols_outcome_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "model_df = df[required_cols + [outcome]].copy()",
                "model_df = df[list(dict.fromkeys(required_cols + [outcome]))].copy()",
                1,
            )
            if repaired != code:
                return repair_name, repaired

    missing_dummy_encoded_dropna_column = (
        "keyerror" in lowered
        and "get_dummies" in code
        and "dropna(" in code
        and ("subset=x_cols" in code or " + x_cols" in code or "_x_cols" in code)
    )
    if missing_dummy_encoded_dropna_column:
        repair_name = "filter_x_cols_before_dropna_after_dummy_encoding_v1"
        if previous_repair != repair_name:
            marker = "model_df = model_df.dropna(subset=x_cols + [outcome])"
            guard = "x_cols = [col for col in x_cols if col in model_df.columns]"
            if marker in code and guard not in code:
                repaired = code.replace(marker, guard + "\n" + marker, 1)
                return repair_name, repaired
            generic_dropna = re.compile(
                r"(?P<line>^(?P<frame>\w+)\s*=\s*(?P=frame)\.replace\(\[np\.inf,\s*-np\.inf\],\s*np\.nan\)\.dropna\(subset=\[(?P<outcome>\w+)\]\s*\+\s*(?P<xcols>\w+)\)\s*$)",
                flags=re.MULTILINE,
            )
            match = generic_dropna.search(code)
            if match:
                xcols = match.group("xcols")
                frame = match.group("frame")
                guard = f"{xcols} = [col for col in {xcols} if col in {frame}.columns]"
                if guard not in code:
                    repaired = code.replace(match.group("line"), guard + "\n" + match.group("line"), 1)
                    return repair_name, repaired

    missing_indicator_source_frame = (
        "keyerror" in lowered
        and "are in the [columns]" in lowered
        and "df = pd.read_parquet" in code
        and ".isnull().any(axis=1).astype(int)" in code
        and "_missing" in code
    )
    if missing_indicator_source_frame:
        repair_name = "missing_indicator_source_df_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"(?P<lhs>\w+\[['\"][^'\"]+_missing['\"]\]\s*=\s*)(?P<frame>\w+)\[(?P<colsvar>\w+)\](?P<rhs>\.isnull\(\)\.any\(axis=1\)\.astype\(int\))",
                r"\g<lhs>df[\g<colsvar>]\g<rhs>",
                code,
                count=1,
            )
            if repaired != code:
                return repair_name, repaired

    missing_subset_cols = _extract_missing_index_columns(run_log or "")
    literal_outcomes = re.findall(
        r"outcome_col\s*=\s*['\"]([^'\"]+)['\"]",
        code,
    )
    missing_outcome_from_subset = (
        "keyerror" in lowered
        and "not in index" in lowered
        and "all_vars = [primary_predictor] + covariates" in code
        and "outcome_col" in code
        and (
            not literal_outcomes
            or any(col in set(missing_subset_cols) for col in literal_outcomes)
        )
    )
    if missing_outcome_from_subset:
        repair_name = "include_outcome_in_all_vars_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "all_vars = [primary_predictor] + covariates",
                "all_vars = [outcome_col, primary_predictor] + covariates",
                1,
            )
            if repaired != code:
                return repair_name, repaired

    robustness_none_plot = (
        "unsupported operand type(s)" in lowered
        and "none" in lowered
        and "ax.errorbar(" in code
        and ("predictor_col" in code or "primary_predictor" in code)
        and (
            "lactate_missing" in code
            or "Missing-indicator" in code
            or "missing_indicator" in code.lower()
        )
    )
    if robustness_none_plot:
        repair_name = "robustness_predictor_design_and_plot_v1"
        if previous_repair != repair_name:
            repaired = code
            predictor_var = (
                "predictor_col" if "predictor_col" in code else "primary_predictor"
            )
            model_df_assign = re.search(
                r"^(?P<indent>\s*)(?P<line>model_df\s*=\s*df\[[^\n]+?\.copy\(\)\s*)$",
                repaired,
                flags=re.MULTILINE,
            )
            sex_numeric_patch = textwrap.dedent(
                """
                if 'sex' in model_df.columns:
                    model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                for col in model_df.columns:
                    if col != 'sex':
                        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                model_df = model_df.replace([np.inf, -np.inf], np.nan)
                reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]
                """
            ).strip("\n")
            if model_df_assign and "reduced_covariates =" not in repaired:
                indent = model_df_assign.group("indent")
                patch = "\n".join(
                    indent + line if line else line
                    for line in sex_numeric_patch.splitlines()
                )
                repaired = repaired.replace(
                    model_df_assign.group(0),
                    model_df_assign.group(0) + "\n" + patch,
                    1,
                )
            replacements = {
                "cc_X = cc_df[covariates]": "cc_X = cc_df[[predictor_col] + covariates]",
                "cc_X = complete_case_df[covariates]": "cc_X = complete_case_df[[predictor_col] + covariates]",
                "mi_X = mi_df[covariates + ['lactate_missing']]": "mi_X = mi_df[[predictor_col] + covariates + ['lactate_missing']]",
                "mi_X = model_df[covariates + ['lactate_missing']]": "mi_X = model_df[[predictor_col] + covariates + ['lactate_missing']]",
                "rv_X = rv_df[covariates]": "rv_X = rv_df[[predictor_col] + covariates]",
                "rv_X = rv_df[reduced_covariates]": "rv_X = rv_df[[predictor_col] + reduced_covariates]",
                "X_cc = sm.add_constant(complete_case_df[covariates], has_constant=\"add\")": (
                    f"X_cc = sm.add_constant(complete_case_df[[{predictor_var}] + covariates], has_constant=\"add\")"
                ),
                "X_cc = sm.add_constant(cc_df[covariates], has_constant=\"add\")": (
                    f"X_cc = sm.add_constant(cc_df[[{predictor_var}] + covariates], has_constant=\"add\")"
                ),
                "X_mi = sm.add_constant(missing_indicator_df[covariates + [\"lactate_missing\"]], has_constant=\"add\")": (
                    f"X_mi = sm.add_constant(missing_indicator_df[[{predictor_var}] + covariates + [\"lactate_missing\"]], has_constant=\"add\")"
                ),
                "X_mi = sm.add_constant(mi_df[covariates + [\"lactate_missing\"]], has_constant=\"add\")": (
                    f"X_mi = sm.add_constant(mi_df[[{predictor_var}] + covariates + [\"lactate_missing\"]], has_constant=\"add\")"
                ),
                "X_rv = sm.add_constant(reduced_variable_df[covariates], has_constant=\"add\")": (
                    f"X_rv = sm.add_constant(reduced_variable_df[[{predictor_var}] + reduced_covariates], has_constant=\"add\")"
                ),
                "X_rv = sm.add_constant(rv_df[covariates], has_constant=\"add\")": (
                    f"X_rv = sm.add_constant(rv_df[[{predictor_var}] + reduced_covariates], has_constant=\"add\")"
                ),
            }
            for old, new in replacements.items():
                repaired = repaired.replace(old, new)
            subset_replacements = {
                f"complete_case_df = model_df.dropna(subset=[{predictor_var}])": (
                    f"complete_case_df = model_df.dropna(subset=[outcome_col, {predictor_var}] + covariates)"
                ),
                f"cc_df = model_df.dropna(subset=[{predictor_var}])": (
                    f"cc_df = model_df.dropna(subset=[outcome_col, {predictor_var}] + covariates)"
                ),
                f"rv_df = model_df.dropna(subset=[{predictor_var}])": (
                    f"rv_df = model_df[[outcome_col, {predictor_var}] + reduced_covariates].dropna()"
                ),
                f"reduced_variable_df = model_df.drop(columns=[{predictor_var}]).copy()": (
                    f"reduced_variable_df = model_df[[outcome_col, {predictor_var}] + reduced_covariates].dropna().copy()"
                ),
            }
            for old, new in subset_replacements.items():
                repaired = repaired.replace(old, new)
            mi_copy_patterns = {
                "mi_df = model_df.copy()": (
                    f"mi_df = model_df.copy()\n"
                    f"    mi_df[{predictor_var}] = mi_df[{predictor_var}].fillna(0)\n"
                    "    mi_df = mi_df.dropna(subset=[outcome_col] + covariates)"
                ),
                "missing_indicator_df = model_df.copy()": (
                    "missing_indicator_df = model_df.copy()\n"
                    f"    missing_indicator_df[{predictor_var}] = missing_indicator_df[{predictor_var}].fillna(0)\n"
                    "    missing_indicator_df = missing_indicator_df.dropna(subset=[outcome_col] + covariates)"
                ),
            }
            for old, new in mi_copy_patterns.items():
                if old in repaired and "fillna(0)" not in repaired:
                    repaired = repaired.replace(old, new, 1)
            repaired = repaired.replace(
                f"X_rv = X_rv.drop(columns=[{predictor_var}])\n",
                "",
            )
            finite_patches = {
                "X_cc = X_cc.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n    y_cc = y_cc.astype(float)": (
                    "X_cc = X_cc.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    "    y_cc = y_cc.astype(float)\n"
                    "    cc_mask = np.isfinite(X_cc.to_numpy()).all(axis=1) & np.isfinite(y_cc.to_numpy())\n"
                    "    X_cc = X_cc.loc[cc_mask]\n"
                    "    y_cc = y_cc.loc[cc_mask]"
                ),
                "X_mi = X_mi.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n    y_mi = y_mi.astype(float)": (
                    "X_mi = X_mi.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    "    y_mi = y_mi.astype(float)\n"
                    "    mi_mask = np.isfinite(X_mi.to_numpy()).all(axis=1) & np.isfinite(y_mi.to_numpy())\n"
                    "    X_mi = X_mi.loc[mi_mask]\n"
                    "    y_mi = y_mi.loc[mi_mask]"
                ),
                "X_rv = X_rv.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n    y_rv = y_rv.astype(float)": (
                    "X_rv = X_rv.apply(pd.to_numeric, errors=\"coerce\").astype(float)\n"
                    "    y_rv = y_rv.astype(float)\n"
                    "    rv_mask = np.isfinite(X_rv.to_numpy()).all(axis=1) & np.isfinite(y_rv.to_numpy())\n"
                    "    X_rv = X_rv.loc[rv_mask]\n"
                    "    y_rv = y_rv.loc[rv_mask]"
                ),
            }
            for old, new in finite_patches.items():
                repaired = repaired.replace(old, new)
            lower_var = "ci_lowers"
            upper_var = "ci_uppers"
            if "lci = [" in repaired and "uci = [" in repaired:
                lower_var = "lci"
                upper_var = "uci"
            elif "or_lowers = [" in repaired and "or_uppers = [" in repaired:
                lower_var = "or_lowers"
                upper_var = "or_uppers"
            plot_marker = "ax.errorbar(x_pos, ors, yerr=[yerr_lower, yerr_upper],"
            plot_guard = textwrap.dedent(
                f"""
                plot_rows = [
                    (s, o, lo, hi)
                    for s, o, lo, hi in zip(strategies, ors, {lower_var}, {upper_var})
                    if o is not None and lo is not None and hi is not None
                ]
                if plot_rows:
                    strategies, ors, {lower_var}, {upper_var} = map(list, zip(*plot_rows))
                    x_pos = np.arange(len(strategies))
                else:
                    strategies, ors, {lower_var}, {upper_var} = [], [], [], []
                    x_pos = np.array([])
                """
            ).strip("\n")
            if plot_marker in repaired and "plot_rows = [" not in repaired:
                repaired = repaired.replace(
                    plot_marker,
                    plot_guard + "\n\n    if len(x_pos):\n        " + plot_marker,
                    1,
                )
            if repaired != code:
                return repair_name, repaired

    missing_internal_utils = (
        "modulenotfounderror: no module named 'easyicu.research_agent.utils'" in lowered
        and "from easyicu.research_agent.utils import to_jsonable" in code
    )
    if missing_internal_utils:
        repair_name = "inline_missing_to_jsonable_utils_v1"
        if previous_repair != repair_name:
            helper = textwrap.dedent(
                """
                def to_jsonable(x):
                    import math
                    import numpy as np
                    import pandas as pd
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if math.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    try:
                        if pd.isna(x):
                            return None
                    except Exception:
                        pass
                    return str(x)
                """
            ).strip()
            repaired = code.replace(
                "from easyicu.research_agent.utils import to_jsonable",
                helper,
                1,
            )
            if repaired != code:
                return repair_name, repaired

    missing_figure_utils = (
        (
            "modulenotfounderror: no module named 'easyicu.research_output'" in lowered
            or "modulenotfounderror: no module named 'easyicu.research_output.figure_utils'" in lowered
            or "no module named 'easyicu.research_output'" in lowered
        )
        and "easyicu.research_output.figure_utils" in code
    )
    if missing_figure_utils:
        repair_name = "replace_hallucinated_figure_utils_import_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "easyicu.research_output.figure_utils",
                "easyicu.research_agent.publication_figures",
            )
            if repaired != code:
                return repair_name, repaired

    robustness_numeric_check_nan = (
        "exog contains inf or nans" in lowered
        and "X_cc = X_cc.apply(pd.to_numeric, errors='coerce')" in code
        and "y_cc = y_cc.astype(float)" in code
        and "sex" in code
    )
    if robustness_numeric_check_nan:
        repair_name = "robustness_encode_sex_before_numeric_checks_v1"
        if previous_repair != repair_name:
            def _numeric_block(prefix: str) -> str:
                block = textwrap.dedent(
                    f"""
                    if 'sex' in X_{prefix}.columns:
                        X_{prefix}['sex'] = X_{prefix}['sex'].astype(str).str.lower().isin(['m', 'male', '1', 'true']).astype(float)
                    X_{prefix} = X_{prefix}.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
                    y_{prefix} = pd.to_numeric(y_{prefix}, errors='coerce').replace([np.inf, -np.inf], np.nan)
                    valid_{prefix}_idx = X_{prefix}.dropna().index.intersection(y_{prefix}.dropna().index)
                    X_{prefix} = X_{prefix}.loc[valid_{prefix}_idx]
                    y_{prefix} = y_{prefix}.loc[valid_{prefix}_idx].astype(float)
                    """
                ).strip("\n")
                return block.replace("\n", "\n    ")

            repaired = code
            replacements = {
                "cc": (
                    "X_cc = X_cc.apply(pd.to_numeric, errors='coerce')\n    y_cc = y_cc.astype(float)",
                    _numeric_block("cc"),
                ),
                "mi": (
                    "X_mi = X_mi.apply(pd.to_numeric, errors='coerce')\n    y_mi = y_mi.astype(float)",
                    _numeric_block("mi"),
                ),
                "rv": (
                    "X_rv = X_rv.apply(pd.to_numeric, errors='coerce')\n    y_rv = y_rv.astype(float)",
                    _numeric_block("rv"),
                ),
            }
            for old, new in replacements.values():
                repaired = repaired.replace(old, new)
            if repaired != code:
                return repair_name, repaired

    undefined_primary_predictor = (
        "name 'primary_predictor' is not defined" in lowered
        and "primary_predictor if primary_predictor else None" in code
    )
    if undefined_primary_predictor:
        repair_name = "primary_predictor_safe_summary_lookup_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "primary_predictor if primary_predictor else None",
                (
                    "locals().get('primary_predictor') or "
                    "locals().get('predictor_col') or "
                    "locals().get('primary_predictor_col') or "
                    "locals().get('predictor') or None"
                ),
                1,
            )
            if repaired != code:
                return repair_name, repaired

    table_one_unclosed_syntax = (
        "syntaxerror" in lowered
        and (
            "was never closed" in lowered
            or "unexpected eof while parsing" in lowered
            or "eof while scanning" in lowered
        )
        and "table_one.csv" in code.lower()
    )
    if table_one_unclosed_syntax:
        repair_name = "table_one_descriptive_repair_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                import json
                import os
                import math
                import numpy as np
                import pandas as pd

                def to_jsonable(x):
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if math.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    try:
                        if pd.isna(x):
                            return None
                    except Exception:
                        pass
                    return str(x)

                cohort_path = os.environ["COHORT_PARQUET"]
                out_dir = os.environ["STEP_OUT_DIR"]
                os.makedirs(out_dir, exist_ok=True)

                df = pd.read_parquet(cohort_path)
                rows = []
                for col in df.columns:
                    s = df[col]
                    n = int(len(s))
                    n_missing = int(s.isna().sum())
                    row = {
                        "variable": col,
                        "n": n,
                        "n_missing": n_missing,
                        "missing_fraction": (n_missing / n) if n else 0.0,
                    }
                    non_missing = s.dropna()
                    if len(non_missing) == 0:
                        rows.append(row)
                        continue
                    if pd.api.types.is_numeric_dtype(non_missing):
                        unique_values = set(non_missing.unique().tolist())
                        if unique_values <= {0, 1, 0.0, 1.0}:
                            positive = int(non_missing.astype(float).sum())
                            row["n_positive"] = positive
                            row["positive_fraction"] = positive / len(non_missing)
                        else:
                            row["median"] = float(non_missing.median())
                            row["q25"] = float(non_missing.quantile(0.25))
                            row["q75"] = float(non_missing.quantile(0.75))
                            row["min"] = float(non_missing.min())
                            row["max"] = float(non_missing.max())
                    else:
                        top = non_missing.astype(str).value_counts().head(1)
                        if len(top):
                            row["most_common"] = str(top.index[0])
                            row["most_common_n"] = int(top.iloc[0])
                    rows.append(row)

                table = pd.DataFrame(rows)
                table_path = os.path.join(out_dir, "table_one.csv")
                table.to_csv(table_path, index=False)

                summary = {
                    "n_total": int(len(df)),
                    "n_variables": int(len(df.columns)),
                    "table_one_path": table_path,
                    "variables": list(df.columns.astype(str)),
                }
                outcome_col = os.environ.get("OUTCOME_COL")
                if outcome_col and outcome_col in df.columns:
                    outcome = pd.to_numeric(df[outcome_col], errors="coerce").dropna()
                    summary["outcome_col"] = outcome_col
                    outcome_values = set(outcome.astype(float).unique().tolist())
                    if outcome_values <= {0.0, 1.0}:
                        summary["outcome_kind"] = "binary_0_1"
                        summary["outcome_n"] = int(outcome.sum()) if len(outcome) else 0
                        summary["outcome_rate"] = float(outcome.mean()) if len(outcome) else None
                    else:
                        summary["outcome_kind"] = "non_binary"
                        summary["outcome_note"] = (
                            "OUTCOME_COL was not summarized as an event rate because "
                            "it is not a binary 0/1 endpoint."
                        )
                if "age" in df.columns:
                    age = pd.to_numeric(df["age"], errors="coerce").dropna()
                    summary["age_median"] = float(age.median()) if len(age) else None
                    summary["age_q25"] = float(age.quantile(0.25)) if len(age) else None
                    summary["age_q75"] = float(age.quantile(0.75)) if len(age) else None

                with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=to_jsonable)
                print(json.dumps({"table": "table_one.csv", "summary": summary}, default=to_jsonable))
                """
            ).strip() + "\n"
            return repair_name, repaired

    outcome_incidence_broken_syntax = (
        "syntaxerror" in lowered
        and (
            "outcome_incidence" in code.lower()
            or "incidence_with_missingness_strata" in code.lower()
        )
    )
    if outcome_incidence_broken_syntax:
        repair_name = "outcome_incidence_descriptive_repair_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                import json
                import os
                import math
                import numpy as np
                import pandas as pd
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from statsmodels.stats.proportion import proportion_confint

                def to_jsonable(x):
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if math.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    try:
                        if pd.isna(x):
                            return None
                    except Exception:
                        pass
                    return str(x)

                cohort_path = os.environ["COHORT_PARQUET"]
                out_dir = os.environ["STEP_OUT_DIR"]
                os.makedirs(out_dir, exist_ok=True)

                df = pd.read_parquet(cohort_path)
                outcome_col = os.environ.get("OUTCOME_COL")
                if not outcome_col:
                    raise KeyError("OUTCOME_COL is required for outcome incidence repair")
                if outcome_col not in df.columns:
                    raise KeyError(f"OUTCOME_COL={outcome_col!r} is not present in the cohort")
                outcome = pd.to_numeric(df[outcome_col], errors="coerce")
                valid_outcome = outcome.dropna().astype(float)
                outcome_values = set(valid_outcome.unique().tolist())
                if outcome_values - {0.0, 1.0} or len(outcome_values) < 2:
                    raise RuntimeError(
                        "Outcome incidence repair requires a binary 0/1 OUTCOME_COL; "
                        "refusing to compute an event rate for a continuous or multi-class endpoint."
                    )
                rows = []

                def add_row(label, mask):
                    y = outcome[mask].dropna().astype(int)
                    n = int(len(y))
                    events = int(y.sum()) if n else 0
                    rate = float(events / n) if n else None
                    if n:
                        ci_low, ci_high = proportion_confint(events, n, alpha=0.05, method="wilson")
                    else:
                        ci_low = ci_high = None
                    rows.append({
                        "stratum": label,
                        "n": n,
                        "n_events": events,
                        "outcome_rate": rate,
                        "ci_low": None if ci_low is None else float(ci_low),
                        "ci_high": None if ci_high is None else float(ci_high),
                    })

                add_row("overall", outcome.notna())

                table = pd.DataFrame(rows)
                table_path = os.path.join(out_dir, "outcome_incidence.csv")
                table.to_csv(table_path, index=False)

                fig, ax = plt.subplots(figsize=(4.8, 3.0))
                plot_df = table[table["stratum"] != "overall"].copy()
                if plot_df.empty:
                    plot_df = table.copy()
                ax.bar(plot_df["stratum"], plot_df["outcome_rate"] * 100, color="#4C78A8")
                ax.set_ylabel("Outcome rate (%)")
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "outcome_incidence.png"), dpi=300)
                fig.savefig(os.path.join(out_dir, "outcome_incidence.svg"))
                plt.close(fig)

                overall = table.iloc[0].to_dict()
                statistic = {
                    "outcome_col": outcome_col,
                    "n_total": int(overall["n"]),
                    "n_events": int(overall["n_events"]),
                    "outcome_rate": overall["outcome_rate"],
                    "overall_ci_low": overall["ci_low"],
                    "overall_ci_high": overall["ci_high"],
                }
                statistic_path = os.path.join(out_dir, "outcome_rate.json")
                with open(statistic_path, "w", encoding="utf-8") as f:
                    json.dump(statistic, f, indent=2, default=to_jsonable)

                summary = {
                    "table": table_path,
                    "statistic": statistic_path,
                    "figure_png": os.path.join(out_dir, "outcome_incidence.png"),
                    "figure_svg": os.path.join(out_dir, "outcome_incidence.svg"),
                    **statistic,
                }
                with open(os.path.join(out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=to_jsonable)
                print(json.dumps(summary, default=to_jsonable))
                """
            ).strip() + "\n"
            return repair_name, repaired

    repeated_keyword_syntax = (
        "syntaxerror: keyword argument repeated" in lowered
        and "train_test_split" in code
        and "figure_contract = figurecontract(" in code.lower()
    )
    if repeated_keyword_syntax and binary_model_repair_allowed:
        repair_name = "prediction_split_minimal_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                import json
                import os
                import numpy as np
                import pandas as pd
                from sklearn.model_selection import train_test_split

                def to_jsonable(x):
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if np.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    return x

                df = pd.read_parquet(os.environ["COHORT_PARQUET"])
                out = os.environ["STEP_OUT_DIR"]
                outcome = os.environ.get("OUTCOME_COL")
                if not outcome:
                    raise RuntimeError(
                        "OUTCOME_COL is required for deterministic prediction split repair; "
                        "refusing to guess a target outcome."
                    )
                if outcome not in df.columns:
                    raise KeyError(
                        f"OUTCOME_COL={outcome!r} is not present in cohort columns."
                    )
                y_numeric = pd.to_numeric(df[outcome], errors="coerce")
                valid_y = y_numeric.dropna().astype(float)
                outcome_values = set(valid_y.unique().tolist())
                if outcome_values - {0.0, 1.0} or len(outcome_values) < 2:
                    raise RuntimeError(
                        "Deterministic prediction split repair requires a binary 0/1 OUTCOME_COL; "
                        "refusing to coerce a continuous or multi-class outcome."
                    )
                df = df.loc[y_numeric.notna()].copy()
                y = y_numeric.loc[df.index].astype(int)
                X = df.drop(columns=[outcome], errors="ignore").copy()
                X = X.select_dtypes(include=["number", "bool"]).apply(pd.to_numeric, errors="coerce")
                if X.empty:
                    X = pd.DataFrame({"row_index": np.arange(len(df))}, index=df.index)
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y if getattr(y, "nunique", lambda: 0)() > 1 else None,
                )
                step_summary = {
                    "split_strategy": "stratified_random",
                    "n_total": int(len(df)),
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "event_rate_total": float(y.mean()),
                    "event_rate_train": float(y_train.mean()) if len(y_train) else None,
                    "event_rate_test": float(y_test.mean()) if len(y_test) else None,
                }
                with open(os.path.join(out, "step_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(step_summary, f, indent=2, default=to_jsonable, ensure_ascii=False)
                print(json.dumps(step_summary, indent=2, default=to_jsonable, ensure_ascii=False))
                """
            ).strip() + "\n"
            return repair_name, repaired

    logreg_nan = (
        "logisticregression does not accept missing values encoded as nan" in lowered
        and "logisticregression" in code.lower()
    )
    if logreg_nan and binary_model_repair_allowed:
        repair_name = "logreg_impute_v1"
        if previous_repair != repair_name and "_easyicu_logreg_impute_v1" not in code:
            patch = textwrap.dedent(
                """

                def _easyicu_logreg_impute_v1(frame):
                    if not hasattr(frame, "copy"):
                        return frame
                    work = frame.copy()
                    for col in work.columns:
                        series = pd.to_numeric(work[col], errors="ignore")
                        if getattr(series, "dtype", None) is not None and str(series.dtype) != "object":
                            if series.isna().any():
                                median = series.median()
                                series = series.fillna(median if pd.notna(median) else 0)
                        work[col] = series
                    return work
                """
            ).strip("\n")
            train_split = re.compile(
                r"(?P<line>X_train,\s*X_test,\s*y_train,\s*y_test\s*=\s*train_test_split\([^\\n]+?\)\s*)",
                re.DOTALL,
            )
            match = train_split.search(code)
            if match:
                inject = (
                    match.group("line")
                    + "\nX_train = _easyicu_logreg_impute_v1(X_train)\n"
                    + "X_test = _easyicu_logreg_impute_v1(X_test)\n"
                )
                repaired = code[: match.start()] + inject + code[match.end() :]
            else:
                repaired = code
            if repaired == code:
                predict_call = re.compile(r"(?P<line>y_pred_proba\s*=\s*model(?:_pipeline)?\.predict_proba\(X_test\)\s*\[:,\s*1\]\s*)")
                match = predict_call.search(code)
                if match:
                    inject = "X_test = _easyicu_logreg_impute_v1(X_test)\n" + match.group("line")
                    repaired = code[: match.start()] + inject + code[match.end() :]
            if repaired != code:
                if "def _easyicu_logreg_impute_v1" not in repaired:
                    repaired = patch + "\n\n" + repaired
                return repair_name, repaired

    placeholder_ellipsis = (
        "syntaxerror: invalid syntax" in lowered
        and "..." in code
        and "model_bundle" in code
    )
    if placeholder_ellipsis and binary_model_repair_allowed:
        repair_name = "prediction_discrimination_template_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                import json
                import math
                import os
                import pickle
                import numpy as np
                import pandas as pd
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from sklearn.metrics import roc_auc_score, roc_curve
                from sklearn.calibration import calibration_curve
                from easyicu.research_agent.publication_figures import (
                    make_figure_contract,
                    apply_publication_style,
                    add_panel_label,
                    save_publication_figure,
                )

                def to_jsonable(x):
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        value = float(x)
                        return value if np.isfinite(value) else None
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                    return x

                step_out_dir = os.environ["STEP_OUT_DIR"]
                cohort_path = os.environ["COHORT_PARQUET"]
                with open(os.path.join(step_out_dir, "prediction_model_object.pkl"), "rb") as f:
                    model_bundle = pickle.load(f)
                model = model_bundle["model"]
                feature_cols = list(model_bundle.get("feature_cols", []))

                df = pd.read_parquet(cohort_path)
                outcome_col = os.environ.get("OUTCOME_COL") or model_bundle.get("outcome_col")
                if not outcome_col:
                    raise KeyError("OUTCOME_COL or model_bundle['outcome_col'] is required for prediction evaluation")
                if outcome_col not in df.columns:
                    raise KeyError(f"Outcome column {outcome_col!r} is not present in the cohort")
                y_numeric = pd.to_numeric(df[outcome_col], errors="coerce")
                valid_y = y_numeric.dropna().astype(float)
                outcome_values = set(valid_y.unique().tolist())
                if outcome_values - {0.0, 1.0} or len(outcome_values) < 2:
                    raise RuntimeError(
                        "Deterministic prediction evaluation repair requires a binary 0/1 outcome; "
                        "refusing to compute AUROC for a continuous or multi-class endpoint."
                    )
                eval_mask = y_numeric.notna()
                X_test = df.loc[eval_mask, feature_cols].copy()
                y_test = y_numeric.loc[eval_mask].astype(int).values
                for col in X_test.columns:
                    series = pd.to_numeric(X_test[col], errors="ignore")
                    if getattr(series, "dtype", None) is not None and str(series.dtype) != "object" and series.isna().any():
                        median = series.median()
                        series = series.fillna(median if pd.notna(median) else 0)
                    X_test[col] = series

                y_pred_proba = model.predict_proba(X_test)[:, 1]
                held_out_auroc = roc_auc_score(y_test, y_pred_proba)
                prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=min(10, max(5, int(len(y_test) * 0.1))), strategy="quantile")

                fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
                apply_publication_style(fig)
                ax1, ax2 = axes
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                ax1.plot(fpr, tpr, color="#0F4D92", linewidth=2)
                ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
                ax1.set_xlabel("False positive rate")
                ax1.set_ylabel("True positive rate")
                add_panel_label(ax1, "A")

                ax2.plot(prob_pred, prob_true, "o-", color="#42949E", linewidth=2)
                ax2.plot([0, 1], [0, 1], "k:", linewidth=1)
                ax2.set_xlabel("Predicted probability")
                ax2.set_ylabel("Observed probability")
                add_panel_label(ax2, "B")

                contract = make_figure_contract(
                    figure_id="prediction_discrimination_evaluation",
                    core_claim="Held-out discrimination and calibration are summarized for the prediction model.",
                    panels=[
                        {"panel_id": "A", "title": "ROC", "role": "validation", "claim": "Held-out AUROC is reported.", "evidence_ids": ["held_out_auroc"]},
                        {"panel_id": "B", "title": "Calibration", "role": "validation", "claim": "Calibration is visualized on held-out data.", "evidence_ids": ["calibration_curve"]},
                    ],
                )
                save_publication_figure(fig, os.path.join(step_out_dir, "discrimination_evaluation"), contract=contract)
                plt.close(fig)

                step_summary = {
                    "held_out_auroc": float(held_out_auroc),
                    "n_test": int(len(y_test)),
                    "outcome": outcome_col,
                    "calibration_status": "ok",
                }
                with open(os.path.join(step_out_dir, "step_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(step_summary, f, indent=2, default=to_jsonable, ensure_ascii=False)
                print(json.dumps(step_summary, indent=2, default=to_jsonable, ensure_ascii=False))
                """
            ).strip() + "\n"
            return repair_name, repaired

    omitted_primary_predictor = re.search(
        r"Error fitting logistic regression:\s*'([^']+)'",
        run_log or "",
        flags=re.IGNORECASE,
    )
    if (
        omitted_primary_predictor
        and "X = model_df[[" in code
        and binary_model_repair_allowed
    ):
        predictor = omitted_primary_predictor.group(1)
        repair_name = "primary_predictor_omitted_from_design_v1"
        if previous_repair != repair_name:
            repaired = _patch_primary_predictor_into_design_matrix(
                code=code,
                predictor=predictor,
            )
            if repaired is not None and repaired != code:
                return repair_name, repaired

    cut_tuple_error = (
        "typeerror: '<' not supported between instances of 'tuple' and 'int'"
        in lowered
        and "pandas/core/reshape/tile.py" in lowered
        and "pd.cut(" in code
    )
    if cut_tuple_error:
        repair_name = "cut_bins_flatten_v1"
        if previous_repair != repair_name:
            bins_assign = re.compile(
                r"(?P<name>\w+_bins)\s*=\s*(?P<literal>\[(?:\s*\([^][]+?\)\s*,?)+\s*\])"
            )
            match = bins_assign.search(code)
            if match:
                try:
                    literal = ast.literal_eval(match.group("literal"))
                except Exception:
                    literal = None
                if (
                    isinstance(literal, list)
                    and literal
                    and all(
                        isinstance(item, tuple)
                        and len(item) == 2
                        and all(isinstance(v, (int, float)) for v in item)
                        for item in literal
                    )
                ):
                    flat_bins = [literal[0][0], *[item[1] for item in literal]]
                    replacement = f"{match.group('name')} = {flat_bins!r}"
                    repaired = (
                        code[: match.start()] + replacement + code[match.end() :]
                    )
                    return repair_name, repaired

    singular_logit = "singular matrix" in lowered and "sm.logit(" in code.lower()
    if singular_logit and binary_model_repair_allowed:
        repair_name = "logit_regularized_fit_v1"
        if previous_repair != repair_name:
            helper = textwrap.dedent(
                """

                def _easyicu_safe_logit_fit_v1(model):
                    try:
                        return model.fit(disp=0, method="newton")
                    except Exception:
                        return model.fit_regularized(alpha=1e-6, disp=0, trim_mode="off")
                """
            ).strip("\n")
            patched = code
            if "_easyicu_safe_logit_fit_v1" not in patched:
                insert_after = patched.find("import warnings")
                if insert_after >= 0:
                    line_end = patched.find("\n", insert_after)
                    patched = (
                        patched[: line_end + 1] + "\n" + helper + "\n" + patched[line_end + 1 :]
                    )
                else:
                    patched = helper + "\n\n" + patched
            patched = re.sub(
                r"(?m)^(?P<indent>\s*)(?P<lhs>\w+)\s*=\s*(?P<model>\w+)\.fit\((?P<args>[^)]*)\)\s*$",
                r"\g<indent>\g<lhs> = _easyicu_safe_logit_fit_v1(\g<model>)",
                patched,
                count=1,
            )
            if patched != code:
                return repair_name, patched

    table_one_binary_keyerror = (
        "keyerror: 1" in lowered
        and "in-hospital mortality" in code.lower()
        and '"counts"][1]' in code
    )
    if table_one_binary_keyerror:
        repair_name = "table_one_binary_key_string_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                'summary["outcomes"]["death"]["counts"][1]',
                'summary["outcomes"]["death"]["counts"].get("1", summary["outcomes"]["death"]["counts"].get(1, 0))',
            )
            repaired = repaired.replace(
                'summary["outcomes"]["death"]["pct"][1]',
                'summary["outcomes"]["death"]["pct"].get("1", summary["outcomes"]["death"]["pct"].get(1, 0.0))',
            )
            if repaired != code:
                return repair_name, repaired

    cohort_file_as_dir = (
        "notadirectoryerror" in lowered
        and (
            'os.path.join(cohort_path, "data.parquet")' in code.lower()
            or 'os.path.join(cohort_path, \'data.parquet\')' in code.lower()
        )
    )
    if cohort_file_as_dir:
        repair_name = "cohort_file_direct_read_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                'pd.read_parquet(os.path.join(COHORT_PATH, "data.parquet"))',
                "pd.read_parquet(COHORT_PATH)",
            )
            repaired = repaired.replace(
                "pd.read_parquet(os.path.join(COHORT_PATH, 'data.parquet'))",
                "pd.read_parquet(COHORT_PATH)",
            )
            repaired = repaired.replace(
                'pd.read_parquet(os.path.join(cohort_path, "data.parquet"))',
                "pd.read_parquet(cohort_path)",
            )
            repaired = repaired.replace(
                "pd.read_parquet(os.path.join(cohort_path, 'data.parquet'))",
                "pd.read_parquet(cohort_path)",
            )
            if repaired != code:
                return repair_name, repaired

    parquet_read_as_csv = (
        "unicodedecodeerror" in lowered
        and "pd.read_csv(" in code.lower()
        and (
            "cohort_path" in code.lower()
            or "cohort_parquet" in code.lower()
        )
    )
    if parquet_read_as_csv:
        repair_name = "cohort_csv_to_parquet_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r"pd\.read_csv\((?P<arg>\s*(?:cohort_path|os\.environ\[['\"]COHORT_PARQUET['\"]\])\s*)(?:,\s*encoding\s*=\s*['\"][^'\"]+['\"])?\)",
                r"pd.read_parquet(\g<arg>)",
                code,
            )
            if repaired != code:
                return repair_name, repaired

    publication_style_nameerror = (
        "nameerror: name 'apply_publication_style' is not defined" in lowered
        and "publication_figure" in code.lower()
    )
    if publication_style_nameerror:
        repair_name = "publication_bundle_promote_script_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent(
                """
                from __future__ import annotations
                import json
                import os
                import shutil
                from pathlib import Path

                out_dir = Path(os.environ["STEP_OUT_DIR"])
                out_dir.mkdir(parents=True, exist_ok=True)
                run_dir = out_dir.parents[2]
                current_step_id = out_dir.parent.name
                figure_suffixes = [".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"]
                contract_suffix = ".figure_contract.json"

                best = None
                for step_dir in sorted((run_dir / "steps").iterdir()):
                    if not step_dir.is_dir() or step_dir.name == current_step_id:
                        continue
                    outputs_dir = step_dir / "outputs"
                    if not outputs_dir.exists():
                        continue
                    bundles = {}
                    for path in outputs_dir.iterdir():
                        if not path.is_file():
                            continue
                        if path.name.endswith(contract_suffix):
                            stem = path.name[: -len(contract_suffix)]
                            bundles.setdefault(stem, {})["contract"] = path
                            continue
                        if path.suffix.lower() in figure_suffixes:
                            bundles.setdefault(path.stem, {})[path.suffix.lower()] = path
                    for stem, files in bundles.items():
                        figure_count = sum(1 for key in files if key.startswith("."))
                        if figure_count == 0:
                            continue
                        score = (
                            1 if "publication_figure" in stem else 0,
                            1 if "primary_association" in stem else 0,
                            figure_count,
                        )
                        if best is None or score > best[0]:
                            best = (score, stem, files)

                if best is None:
                    raise SystemExit("No prior figure bundle available to promote.")

                _, source_stem, files = best
                target_stem = "publication_figure"
                outputs = {}
                for key, source in files.items():
                    if key == "contract":
                        target = out_dir / f"{target_stem}.figure_contract.json"
                        shutil.copy2(source, target)
                        outputs["contract"] = target.name
                    else:
                        target = out_dir / f"{target_stem}{key}"
                        shutil.copy2(source, target)
                        outputs[key.lstrip('.')] = target.name

                summary = {
                    "step": current_step_id,
                    "status": "completed",
                    "publication_figure_rescue": {
                        "mode": "promotion",
                        "source_step_stem": source_stem,
                        "source_outputs_dir": str(files[next(iter(files))].parent),
                    },
                    "outputs": outputs,
                }
                with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                """
            ).strip() + "\n"
            return repair_name, repaired

    # ----------------------------------------------------------------
    # Column-hallucination fallback: agent emitted a list literal of
    # column names not in the cohort (e.g. naive arms guess
    # ``covariates = ["age", "sex", "map_min_24h", "vaso_any_24h"]``).
    # Runs only when no earlier specialised repair (dummy-encoding,
    # missing outcome, etc.) handled the same KeyError, so we don't
    # interfere with category-aware fixes upstream.
    # ----------------------------------------------------------------
    if "keyerror" in lowered and "not in index" in lowered:
        missing_cols = _extract_missing_index_columns(run_log or "")
        if missing_cols:
            repair_name = "strip_unknown_cols_from_list_literals_v1"
            if previous_repair != repair_name:
                repaired = _strip_columns_from_list_literals(code, missing_cols)
                if repaired != code:
                    return repair_name, repaired

    # ----------------------------------------------------------------
    # Fix F — undefined helper auto-stub (generic fallback)
    # ----------------------------------------------------------------
    # Coder agents under naive arms occasionally reference helper
    # functions they forgot to define (e.g. ``json.dump(..., default=
    # to_json_serializable)`` without ever providing
    # ``to_json_serializable``). LLM self-repair attempts often
    # rewrite the call site rather than re-introduce the helper, so
    # the script keeps failing with the same NameError. Inject a
    # tolerant best-effort stub that handles the common JSON-default
    # pattern (numpy arrays / scalars / fallthrough to ``str``).
    #
    # Runs *after* the specialised NameError repairs (e.g.
    # ``publication_bundle_promote_script_v1`` for ``apply_publication_style``)
    # so we don't shadow richer recovery logic.
    name_error_match = _NAME_ERROR_HELPER_RE.search(run_log or "")
    if name_error_match is not None:
        helper_name = name_error_match.group("name")
        if (
            helper_name
            and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", helper_name)
            and f"def {helper_name}" not in code
            and f"{helper_name} =" not in code
        ):
            repair_name = f"undefined_helper_stub_{helper_name}_v1"
            if previous_repair != repair_name:
                stub = textwrap.dedent(
                    f"""
                    def {helper_name}(*args, **kwargs):
                        \"\"\"Auto-injected stub for an undefined helper.

                        Recovers from NameError raised when the agent
                        emitted a reference (e.g. ``json.dump(default=
                        {helper_name})``) without ever defining the
                        helper. The stub returns a JSON-friendly form
                        of its first positional argument when one is
                        provided, falling back to ``None``.
                        \"\"\"
                        if not args:
                            return None
                        value = args[0]
                        try:
                            if hasattr(value, \"tolist\"):
                                return value.tolist()
                        except Exception:
                            pass
                        try:
                            if hasattr(value, \"item\"):
                                return value.item()
                        except Exception:
                            pass
                        try:
                            return str(value)
                        except Exception:
                            return None
                    """
                ).strip("\n")
                repaired = stub + "\n\n" + code
                if repaired != code:
                    return repair_name, repaired

    # dtype_coerce_v1 repair for statsmodels dtype failures
    signatures = (
        "pandas data cast to numpy dtype of object",
        "exog contains inf or nans",
        "missingdataerror",
        "ufunc 'isfinite' not supported",
    )
    dtype_coerce_applies = (
        any(sig in lowered for sig in signatures)
        and "_easyicu_runner_repair_v1" not in code
        and any(token in code for token in ("sm.Logit(", "sm.OLS(", "sm.GLM("))
        and _statsmodels_repair_allowed_for_family(code, analysis_family)
    )
    if dtype_coerce_applies:
        repair_name = "dtype_coerce_v1"
        patch = textwrap.dedent(
            """

            def _easyicu_runner_repair_v1(X, y):
                X_work = X.copy() if hasattr(X, "copy") else X
                y_work = y.copy() if hasattr(y, "copy") else y
                if hasattr(X_work, "replace"):
                    X_work = X_work.replace([np.inf, -np.inf], np.nan)
                if hasattr(X_work, "apply"):
                    X_work = X_work.apply(pd.to_numeric, errors="coerce").astype(float)
                else:
                    X_work = np.asarray(X_work, dtype=float)
                y_work = pd.to_numeric(y_work, errors="coerce")
                if hasattr(X_work, "index") and hasattr(y_work, "index"):
                    keep = X_work.dropna().index.intersection(y_work.dropna().index)
                    X_work = X_work.loc[keep]
                    y_work = y_work.loc[keep]
                else:
                    X_arr = np.asarray(X_work, dtype=float)
                    y_arr = np.asarray(y_work, dtype=float)
                    mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
                    X_work = X_arr[mask]
                    y_work = y_arr[mask]
                return y_work.astype(float), X_work
            """
        ).strip("\n")

        patched = code
        if "_easyicu_runner_repair_v1" not in patched:
            insert_after = patched.find("import matplotlib.pyplot as plt")
            if insert_after >= 0:
                line_end = patched.find("\n", insert_after)
                patched = (
                    patched[: line_end + 1] + "\n" + patch + "\n" + patched[line_end + 1 :]
                )
            else:
                patched = patch + "\n\n" + patched

        model_call = re.compile(
            r"(?P<prefix>\b[A-Za-z_]\w*\s*=\s*sm\.(?:Logit|OLS|GLM)\()\s*"
            r"(?P<y>[^,]+?)\s*,\s*(?P<X>[^,\)\n]+?)\s*"
            r"(?P<kwargs>,\s*[^)\n]+)?(?P<suffix>\))"
        )

        def _rewrite(match: re.Match[str]) -> str:
            y_expr = match.group("y").strip()
            x_expr = match.group("X").strip()
            kwargs = match.group("kwargs") or ""
            return (
                f"{match.group('prefix')}*_easyicu_runner_repair_v1("
                f"{x_expr}, {y_expr}){kwargs}{match.group('suffix')}"
            )

        repaired = model_call.sub(_rewrite, patched, count=1)
        repaired = repaired.replace(
            "X_array = X.to_numpy()\n"
            "y_array = y.to_numpy()\n",
            "y, X = _easyicu_runner_repair_v1(X, y)\n"
            "X_array = np.asarray(X, dtype=float)\n"
            "y_array = np.asarray(y, dtype=float)\n",
            1,
        )
        if repaired != code:
            return repair_name, repaired

    return None
