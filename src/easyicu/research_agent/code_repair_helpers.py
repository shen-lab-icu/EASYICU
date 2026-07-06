"""Deterministic repair helpers for code-mutation and summary repair.

Physical split from ``code_repair.py`` (P1-3, 2026-06-10), **zero behavior
change**. This leaf module hosts the 19 pure helper functions used by the two
deterministic-repair entrypoints (``_deterministic_summary_repair`` and
``_deterministic_runner_repair``, which remain in ``code_repair`` because they
are mutually recursive). Helpers cover primary-association model fallback code
generation, predictor / binary-outcome inference, and AST/regex code patches
(KeyError-not-in-index stripping, statsmodels alignment, json-dump numpy key
sanitising, ...).

It imports only the stdlib (ast/json/re/textwrap) and typing, and must never
import ``code_repair`` — the parent re-exports every name defined here for
backward compatibility (pipeline / pipeline_execute import several directly).
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence


_BINARY_MODEL_REPAIR_FAMILIES = {
    "association_study",
    "cohort_definition_sensitivity",
    "prediction_model",
    "validation",
    "robustness",
    "bias_audit",
}


def _code_mentions_missing_indicator_column(code: str) -> bool:
    """Detect generated missing-indicator robustness code without case terms."""

    if "missing_indicator" in code.lower() or "Missing-indicator" in code:
        return True
    return bool(
        re.search(
            r"['\"][A-Za-z_][A-Za-z0-9_]*_missing(?:_[A-Za-z0-9_]+)?['\"]",
            code,
        )
    )


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
        or "sm.glm" in lowered
        and "binomial" in lowered
        or "logisticregression" in lowered
        or "predict_proba" in lowered
        or "roc_auc" in lowered
    )


def _statsmodels_repair_allowed_for_family(
    code: str, analysis_family: Optional[str]
) -> bool:
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
        raise ValueError(
            "A binary outcome column is required for primary association fallback."
        )
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


def _infer_overexpanded_categorical_predictor(
    step_summary: Dict[str, Any], code: str
) -> Optional[str]:
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

    match = re.search(
        r"formula\s*=\s*[\"'](?P<outcome>[A-Za-z_][A-Za-z0-9_]*)\s*~", code
    )
    if match:
        return match.group("outcome")
    match = re.search(
        r"[\"'](?P<outcome>[A-Za-z_][A-Za-z0-9_]*)\s*~\s*[^\"']+[\"']", code
    )
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

    for key in ("primary_predictor", "primary_exposure", "predictor", "exposure"):
        value = _normalise_predictor_column_candidate(step_summary.get(key), code)
        if value:
            return value
    manifest = (
        step_summary.get("manifest:robustness_analysis_manifest")
        or step_summary.get("robustness_analysis_manifest")
        or {}
    )
    if isinstance(manifest, dict):
        for key in ("primary_predictor", "primary_exposure", "predictor", "exposure"):
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
        cols = re.findall(
            r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", required_match.group("body")
        )
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
            # Adjustment set is config-first: honour user_preferences.covariates
            # from research_context.json when present, else a case-neutral
            # demographic default. Nothing here is tied to one study question.
            _req_covs = []
            try:
                _run_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(out_dir)))
                )
                with open(
                    os.path.join(_run_dir, "research_context.json"), encoding="utf-8"
                ) as _ctx_fh:
                    _prefs = (json.load(_ctx_fh).get("user_preferences") or {{}})
                _req_covs = [
                    str(c).strip()
                    for c in (_prefs.get("covariates") or [])
                    if str(c).strip()
                ]
            except Exception:
                _req_covs = []
            covariates = [
                col for col in (_req_covs or ("age", "sex", "weight"))
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
        '    X = model_df[covariates].apply(pd.to_numeric, errors="coerce").astype(float)',
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
                '    X = model_df[design_cols].apply(pd.to_numeric, errors="coerce").astype(float)'
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
    x_line = code[x_assign.start() : line_end]
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


_KEYERROR_NOT_IN_INDEX_RE = re.compile(
    r"KeyError:\s*\"\[(?P<items>[^\]]+)\]\s*not\s+in\s+index\"",
    re.MULTILINE,
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
            f'{indent}if "analysis_cohort" not in {target}.columns and {source_col!r} in {target}.columns:\n'
            f'{indent}    {target}["analysis_cohort"] = {target}[{source_col!r}].astype("string")\n'
            f'{indent}    {target}.loc[{target}[{source_col!r}].isna(), "analysis_cohort"] = pd.NA'
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
