"""Mechanical deterministic-repair helpers for generated code.

Physical split from ``repairs.source`` (P1-3, 2026-06-10). This leaf module
hosts narrowly scoped AST/regex source patches such as missing-column cleanup,
statsmodels alignment, and JSON serialization sanitation. Scientific choices
such as the exposure, outcome, adjustment set, estimand, and model remain owned
by the agent-authored plan and code; helpers here must not infer or replace them.

It imports only the stdlib and leaf repair modules, and must never import
``repairs.source``; the parent re-exports shared helpers for compatibility.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import List, Optional, Sequence

from .serialization import serialization_runner_repair
from .typed_schema import patch_host_schema_numeric_alias


def _finding_json_repair(
    code: str, run_log: str, previous_repair: str | None
) -> tuple[str, str] | None:
    return serialization_runner_repair(
        code=code, run_log=run_log, previous_repair=previous_repair
    )


def _schema_alias_repair(
    code: str, findings: Sequence[object]
) -> tuple[str, list[str]]:
    repaired = patch_host_schema_numeric_alias(code, repair_findings=findings)
    names = ["host_schema_numeric_alias_v1"] if repaired != code else []
    return repaired, names


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
    summary_defaults = textwrap.dedent("""
        n_total = int(len(df))
        n_complete = int(len(model_df))
        """).strip("\n")
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
    helper = textwrap.dedent(f"""

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
        """).strip("\n")
    return helper + "\n\n" + repaired


def _patch_json_dump_numpy_key_sanitizer(code: str) -> str:
    """Make numpy scalars/keys serializable in generated code.

    The sanitizer is bound to **this script's own call sites**, not to the
    stdlib module.

    It used to end with ``json.dump = ...; json.dumps = ...``, which rebinds
    the stdlib module for every module in the interpreter -- the generated
    script, and every EasyICU helper the script imports. Measured on
    2026-08-07: after the preamble ran, ``cohort_row_identity_sha256`` (the
    published cohort-identity recipe generated code is told to call) returned
    a digest for ``[1, nan]`` instead of raising, because its
    ``allow_nan=False`` guard was answered ``null`` by the sanitizer. A repair
    for numpy dict keys was silently turning a fail-closed evidence guard into
    a fail-open one.

    ``allow_nan`` cannot separate the two callers -- the generated script that
    needs the repair passes ``allow_nan=False`` too, and sanitizing *its* NaN
    is precisely what the repair exists to do. The discriminator is scope, so
    the call sites in this script are rewritten to the wrappers and the module
    is left alone.

    A script that aliases the module (``import json as j``) keeps its original
    calls and the repair simply does not apply to them: an unrepaired step
    fails, which is the safe direction.
    """

    if "_easyicu_json_sanitize_v1" in code:
        return code
    helper = textwrap.dedent("""
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
        """).strip()
    # Rewrite this script's own call sites instead of rebinding the module.
    # ``dumps`` first: ``json.dump(`` is a prefix of ``json.dumps(``.
    rewritten = re.sub(r"\bjson\.dumps\(", "_easyicu_json_dumps_v1(", code)
    rewritten = re.sub(r"\bjson\.dump\(", "_easyicu_json_dump_v1(", rewritten)
    return helper + "\n" + rewritten
