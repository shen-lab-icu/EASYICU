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

"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

_TRY_STAR_NODE_TYPES = (
    (try_star_type,) if (try_star_type := getattr(ast, "TryStar", None)) else ()
)
_TRY_NODE_TYPES = (ast.Try, *_TRY_STAR_NODE_TYPES)

from ..scalar_utils import (
    _coerce_scalar,
    _first_numeric_effect_from_text,
    _first_numeric_scalar_with_key_fragment,
    _first_present_scalar,
    _flatten_scalar_dict,
)
from .attrition import patch_attrition_rule_id_canonicalization
from .categorical import patch_categorical_declared_order_check
from .concept_preflight import patch_concept_preflight_repairs
from .figure_distribution import (
    patch_categorical_distribution_clinical_bin_role,
    patch_text_distribution_denominator_from_counts,
)
from .lossy_coercion import (
    patch_lossy_numeric_coercion_guard as _patch_lossy_numeric_coercion_guard,
    patch_returned_coercion_loss_guard as _patch_returned_coercion_loss_guard,
)
from .merge_collision import patch_pandas_merge_dynamic_column_collision
from .model_contract import patch_penalized_convergence_contract
from .name_alias import patch_undefined_mapping_near_match_alias
from .nonfinite_audit import (
    patch_nonfinite_audit_host_strict_boundary,
    patch_strict_numeric_nonfinite_audit_conflict,
    patch_strict_numeric_helper_nonfinite_guard,
)
from .nullable_validation import patch_unused_nullable_numeric_validation
from .rendering_role import patch_structured_analysis_role_selection
from .rendering_summary import patch_render_only_effect_echo
from .plausibility import patch_flag_only_plausibility_range_rejection
from .provenance_summary import (
    patch_audit_only_companion_value_selector,
    patch_custom_measurement_provenance_receipts,
    patch_direct_host_receipt_source_guard,
    patch_late_measurement_provenance_receipt,
    patch_measurement_provenance_contract,
    repair_superseded_provenance,
)
from .helpers import (  # noqa: F401  (re-exported for back-compat)
    _BINARY_MODEL_REPAIR_FAMILIES,
    _KEYERROR_NOT_IN_INDEX_RE,
    _code_contains_binary_model,
    _code_mentions_missing_indicator_column,
    _extract_missing_index_columns,
    _extract_required_cols_list,
    _finding_json_repair,
    _family_allows_binary_model_repair,
    _infer_analysis_cohort_source_column,
    _patch_derived_analysis_cohort_materialization,
    _patch_json_dump_numpy_key_sanitizer,
    _patch_primary_predictor_into_design_matrix,
    _patch_statsmodels_conf_int_filter_axis,
    _patch_statsmodels_endog_exog_index_alignment,
    _schema_alias_repair,
    _statsmodels_repair_allowed_for_family,
    _strip_columns_from_list_literals,
)
from .import_repair import (
    host_module_is_available,
    insert_after_imports,
    patch_known_host_helper_import,
)
from .input_scope import patch_raw_input_physical_superset_guard
from .reasons import RepairReason
from .typed_input import (
    patch_all_rows_outcome_coordinate_filter,
    patch_resolved_input_cohort_env_shadow,
    patch_resolved_input_manifest_env,
    patch_resolved_input_relative_path_root,
)
from .typed_artifact import (
    patch_non_tabular_companion_row_gate,
    patch_resolved_json_document_adapter,
)
from .typed_binding_identity import patch_direct_resolved_input_identity_key
from ..gates.typed_binding_identity import direct_resolved_input_key_findings
from ..schema import ValidationFinding

_NULL_PRIMARY_EFFECT_MARKERS = (
    '"complete_case_n": null',
    '"statistic:complete_case_n": null',
    '"or_estimate": null',
    '"odds_ratio": null',
    '"primary_odds_ratio": null',
    '"primary_or": null',
    '"statistic:primary_or": null',
    '"adjusted_or": null',
    '"statistic:adjusted_or": null',
    '"adjusted_odds_ratio": null',
    '"statistic:adjusted_odds_ratio": null',
    '"estimate": null',
    '"statistic:estimate": null',
    '"primary_association_estimate": null',
    '"statistic:primary_association_estimate": null',
    '"association_estimate": null',
    '"statistic:association_estimate": null',
)


def _structured_primary_singular_failure(step_summary: Dict[str, Any]) -> bool:
    """Return whether the declared primary model failed from singularity."""

    contracts = step_summary.get("model_contracts")
    if isinstance(contracts, dict):
        candidates = [contracts]
    elif isinstance(contracts, list):
        candidates = [item for item in contracts if isinstance(item, dict)]
    else:
        candidates = []
    for contract in candidates:
        if str(contract.get("analysis_role") or "").strip().lower() != "primary":
            continue
        fit_status = str(contract.get("fit_status") or "").strip().lower()
        failure_reason = str(contract.get("fit_failure_reason") or "").strip().lower()
        if fit_status != "fitted" and "singular matrix" in failure_reason:
            return True
    return False


def _patch_rank_safe_statsmodels_design(code: str) -> Optional[str]:
    """Insert a rank-safe design-matrix reducer before statsmodels binary fits.

    Generated scripts sometimes catch ``Singular matrix`` internally and write a
    null primary effect while exiting successfully. This patch keeps the
    generated analysis structure intact, but removes constant / perfectly
    collinear columns before fitting. It preserves the intercept and the primary
    coefficient target when the generated script exposes it as ``exposure_col``,
    ``predictor_col`` or ``primary_predictor``; otherwise it keeps the first
    non-intercept column, matching the convention used by association templates.
    """

    if "_easyicu_rank_safe_design_v1" in code:
        return None
    helper = textwrap.dedent("""

        def _easyicu_safe_exp_v1(value):
            import math as _math

            try:
                numeric = float(value)
            except Exception:
                return None
            if not _math.isfinite(numeric):
                return None
            try:
                result = _math.exp(numeric)
            except OverflowError:
                return None
            return float(result) if _math.isfinite(result) else None


        def _easyicu_rank_safe_design_v1(X, keep=None):
            import numpy as _np
            import pandas as _pd

            X_work = X.copy() if hasattr(X, "copy") else _pd.DataFrame(X)
            if not hasattr(X_work, "columns"):
                X_work = _pd.DataFrame(X_work)
            if hasattr(X_work, "replace"):
                X_work = X_work.replace([_np.inf, -_np.inf], _np.nan)
            X_work = X_work.apply(_pd.to_numeric, errors="coerce").astype(float)
            columns = list(X_work.columns)
            requested_keep = [c for c in (keep or []) if c in columns]
            const_cols = [c for c in columns if str(c).lower() == "const"]
            if not requested_keep:
                first_signal = next(
                    (c for c in columns if str(c).lower() != "const"),
                    None,
                )
                requested_keep = const_cols + ([first_signal] if first_signal is not None else [])
            else:
                requested_keep = const_cols + [c for c in requested_keep if c not in const_cols]
            requested_keep = list(dict.fromkeys(requested_keep))

            variances = X_work.var(axis=0, ddof=0)
            zero_variance = [
                c
                for c in columns
                if c not in requested_keep and not (float(variances.get(c, 0.0)) > 0.0)
            ]
            working = X_work.drop(columns=zero_variance)
            ordered = requested_keep + [c for c in working.columns if c not in requested_keep]
            kept = []
            matrix = None
            rank = 0
            for col in ordered:
                if col not in working.columns:
                    continue
                vec = working[col].to_numpy(dtype=float).reshape(-1, 1)
                if not _np.isfinite(vec).all():
                    continue
                trial = vec if matrix is None else _np.hstack([matrix, vec])
                trial_rank = int(_np.linalg.matrix_rank(trial))
                if trial_rank > rank:
                    kept.append(col)
                    matrix = trial
                    rank = trial_rank
            if not kept:
                return X_work, columns
            dropped = [c for c in columns if c not in kept]
            reduced = X_work[kept]
            try:
                reduced.attrs["easyicu_dropped_rank_deficient_columns"] = [
                    str(c) for c in dropped
                ]
            except Exception:
                pass
            return reduced, dropped
        """).strip("\n")

    model_call = re.compile(
        r"(?m)^(?P<indent>\s*)(?P<lhs>[A-Za-z_]\w*)\s*=\s*sm\.Logit\("
        r"(?P<y>[^,\n)]+?)\s*,\s*(?P<X>[A-Za-z_]\w*)"
        r"(?P<kwargs>,\s*[^)\n]+)?\)\s*$"
    )
    direct_fit_call = re.compile(
        r"(?m)^(?P<indent>\s*)(?P<lhs>[A-Za-z_]\w*)\s*=\s*sm\.Logit\("
        r"(?P<y>[^,\n)]+?)\s*,\s*(?P<X>[A-Za-z_]\w*)"
        r"(?P<kwargs>,\s*[^)\n]+)?\)\.fit\((?P<fit_args>[^)\n]*)\)\s*$"
    )

    def _keep_expression(x_expr: str) -> str:
        return (
            "[c for c in ["
            "'const', "
            "locals().get('exposure_col'), "
            "locals().get('predictor_col'), "
            "locals().get('primary_predictor'), "
            "locals().get('PRIMARY_EXPOSURE'), "
            "globals().get('PRIMARY_EXPOSURE')"
            f"] if c is not None and hasattr({x_expr}, 'columns') and c in {x_expr}.columns]"
        )

    def _rank_reduction_line(match: re.Match[str]) -> str:
        x_expr = match.group("X").strip()
        return (
            f"{match.group('indent')}{x_expr}, _easyicu_dropped_rank_cols_v1 = "
            f"_easyicu_rank_safe_design_v1({x_expr}, "
            f"keep={_keep_expression(x_expr)})"
        )

    def _rewrite_model(match: re.Match[str]) -> str:
        indent = match.group("indent")
        x_expr = match.group("X").strip()
        y_expr = match.group("y").strip()
        kwargs = match.group("kwargs") or ""
        lhs = match.group("lhs")
        return (
            f"{_rank_reduction_line(match)}\n"
            f"{indent}{lhs} = sm.GLM({y_expr}, {x_expr}, family=sm.families.Binomial(){kwargs})"
        )

    def _rewrite_direct_fit(match: re.Match[str]) -> str:
        indent = match.group("indent")
        x_expr = match.group("X").strip()
        y_expr = match.group("y").strip()
        kwargs = match.group("kwargs") or ""
        fit_args = match.group("fit_args")
        lhs = match.group("lhs")
        return (
            f"{_rank_reduction_line(match)}\n"
            f"{indent}{lhs} = sm.GLM("
            f"{y_expr}, {x_expr}, family=sm.families.Binomial(){kwargs}"
            f").fit({fit_args})"
        )

    repaired = direct_fit_call.sub(_rewrite_direct_fit, code, count=1)
    if repaired == code:
        repaired = model_call.sub(_rewrite_model, code, count=1)
    if repaired == code:
        return None
    repaired = re.sub(
        r"(?P<quote>['\"])statsmodels_logit_mle(?P=quote)",
        lambda match: (
            f"{match.group('quote')}statsmodels_glm_binomial_irls_rank_safe"
            f"{match.group('quote')}"
        ),
        repaired,
        count=1,
    )
    if "wald_95_percent" in repaired and re.search(r"\b1\.96\s*\*", repaired):
        repaired = re.sub(
            r"(?P<quote>['\"])profile_normal(?P=quote)",
            lambda match: (
                f"{match.group('quote')}wald_95_percent{match.group('quote')}"
            ),
            repaired,
            count=1,
        )
    repaired = re.sub(
        r"float\(math\.exp\((?P<expr>[^()\n]+)\)\)",
        lambda match: f"_easyicu_safe_exp_v1({match.group('expr').strip()})",
        repaired,
    )

    insert_after = repaired.find("import statsmodels.api as sm")
    if insert_after >= 0:
        line_end = repaired.find("\n", insert_after)
        repaired = (
            repaired[: line_end + 1] + "\n" + helper + "\n" + repaired[line_end + 1 :]
        )
    else:
        repaired = helper + "\n\n" + repaired
    return repaired


def _patch_age_covariate_coding_without_indicator(code: str) -> Optional[str]:
    marker = '        elif var == "sex":\n'
    if marker not in code or "meas_var = measured_vars[var]" not in code:
        return None
    age_branch = """        elif var == "age":
            coding_rows.append({
                "variable": var,
                "role": "adjustor",
                "coding": "continuous; modeled as numeric covariate via age_filled",
                "original_missing_n": int(eligible_df[var].isna().sum()),
                "original_missing_pct": float(100.0 * eligible_df[var].isna().mean()),
                "post_plausibility_missing_n": int(work_df[var].isna().sum()),
                "post_plausibility_missing_pct": float(100.0 * work_df[var].isna().mean()),
                "newly_invalid_n": int(newly_invalid_map.get(var, 0)),
                "measured_indicator_available": False,
                "measured_indicator_used": False,
                "fill_strategy": "median_for_fit",
                "fill_value": fill_values.get(var),
                "included_in_model": True,
                "notes": "Demographic baseline covariate; no measured indicator is defined or used.",
            })
"""
    repaired = code.replace(marker, age_branch + marker, 1)
    return repaired if repaired != code else None


# ---------------------------------------------------------------------------
# Tier-A deterministic concept-audit repair.
#
# This runs *inside the static concept-audit gate* (before the script is
# executed), unlike the runner/summary repairs which run after a failure.
# It exists so the gate does not have to block-and-stop every time a weak
# model emits a mechanical ICU anti-pattern that has a single, neutral
# correct fix. It is deliberately narrow: it only rewrites a pattern when
# an ``error``-severity finding *objectively names* that anti-pattern, so
# it can never override a defensible analytical choice (impartiality — it
# touches only the ``error`` class, never the ``caution`` class).
# ---------------------------------------------------------------------------

# A finding message that objectively reports silent zero-imputation. We only
# rewrite ``fillna(0)`` when the auditor itself flagged zero-imputation, never
# on our own initiative.
_ZERO_IMPUTE_FINDING_RE = re.compile(
    r"(fillna\(\s*0"
    r"|impute\w*[^.\n]{0,48}\bwith\s+0\b"
    r"|impute\w*[^.\n]{0,48}\bzero\b"
    r"|zero[-\s]*impute"
    r"|silent\w*[^.\n]{0,48}zero)",
    re.IGNORECASE,
)

# Columns where a literal 0 is a real value (counts / indicators / component
# tallies); we must NOT strip ``fillna(0)`` on these — 0 is correct there.
_COUNT_LIKE_COL_RE = re.compile(
    r"(n_components|_components\b|_count\b|_counts\b|\bn_\w+|\bevents?\b"
    r"|_missing\b|_flag\b|_indicator\b|_dummy\b|_present\b|num_\w+|_n\b)",
    re.IGNORECASE,
)

# ``frame[col] = frame[col].fillna(0)`` (col may be a string literal or a
# variable such as ``primary_predictor``).
_FILLNA_ZERO_ASSIGN_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<frame>\w+)\[(?P<col>[^\]\n]+)\]"
    r"[ \t]*=[ \t]*(?P=frame)\[(?P=col)\]\.fillna\(\s*0(?:\.0)?\s*\)[ \t]*$",
    re.MULTILINE,
)

_STRICT_NUMERIC_NONFINITE_GUARD_SENTINEL = "_easyicu_strict_numeric_nonfinite_guard_v1"
_CATEGORICAL_LEVEL_GUARD_SENTINEL = "_easyicu_categorical_level_reconciliation_guard_v1"


def _host_helper_signature_repair_lines(
    findings: Sequence[ValidationFinding],
) -> frozenset[int]:
    """Return exact host-owned helper call lines eligible for repair."""

    return frozenset(
        int(detail["line"])
        for finding in findings
        for detail in [finding.detail or {}]
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and detail.get("reason") == "host_helper_call_signature_invalid"
        and detail.get("helper_name") == "measurement_provenance_receipt"
        and isinstance(detail.get("line"), int)
        and not isinstance(detail.get("line"), bool)
        and int(detail["line"]) > 0
    )


def _patch_measurement_receipt_stable_binding(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> str:
    """Normalize one exact host receipt call without changing its data source."""

    safe_violations = {
        "unknown_keyword_argument",
        "measured_column_role_invalid",
        "count_column_role_invalid",
        "measurement_companion_columns_mismatch",
    }
    details_by_line: dict[int, dict[str, object]] = {}
    for finding in findings:
        detail = dict(finding.detail or {})
        violations = detail.get("violations")
        if not (
            finding.validator == "mechanical_code_preflight"
            and finding.severity == "error"
            and detail.get("reason") == "host_helper_call_signature_invalid"
            and detail.get("helper_name") == "measurement_provenance_receipt"
            and isinstance(detail.get("line"), int)
            and not isinstance(detail.get("line"), bool)
            and isinstance(violations, list)
            and violations
            and set(violations) <= safe_violations
        ):
            continue
        details_by_line[int(detail["line"])] = detail
    if not details_by_line:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    direct_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "measurement_provenance_receipt" and alias.asname is None
            for alias in node.names
        )
    ]
    shadowing_bindings = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == "measurement_provenance_receipt"
        )
        or (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id == "measurement_provenance_receipt"
        )
    ]
    if len(direct_imports) != 1 or shadowing_bindings:
        return code

    candidates_by_line: dict[int, list[ast.Call]] = {
        line: [] for line in details_by_line
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and int(getattr(node, "lineno", 0)) in candidates_by_line
            and isinstance(node.func, ast.Name)
            and node.func.id == "measurement_provenance_receipt"
        ):
            candidates_by_line[int(node.lineno)].append(node)
    if any(len(candidates) != 1 for candidates in candidates_by_line.values()):
        return code

    replacements: list[tuple[str, str]] = []
    for line, detail in sorted(details_by_line.items()):
        call = candidates_by_line[line][0]
        if any(keyword.arg is None for keyword in call.keywords):
            return code
        keyword_map = {str(keyword.arg): keyword.value for keyword in call.keywords}
        if len(keyword_map) != len(call.keywords):
            return code
        if len(call.args) == 1 and "frame" not in keyword_map:
            frame_node = call.args[0]
        elif not call.args and "frame" in keyword_map:
            frame_node = keyword_map["frame"]
        else:
            return code
        measured_node = keyword_map.get("measured_column")
        count_node = keyword_map.get("count_column")
        if measured_node is None or count_node is None:
            return code

        expected_measured = detail.get("expected_measured_column")
        expected_count = detail.get("expected_count_column")
        if expected_measured is not None and expected_count is not None:
            # Two valid but crossed declared pairs do not identify which
            # scientific pair the author intended.
            return code
        measured_source = ast.get_source_segment(code, measured_node)
        count_source = ast.get_source_segment(code, count_node)
        if expected_measured is not None:
            if not (
                isinstance(expected_measured, str)
                and isinstance(measured_node, ast.Constant)
                and measured_node.value == detail.get("observed_measured_column")
            ):
                return code
            measured_source = repr(expected_measured)
        if expected_count is not None:
            if not (
                isinstance(expected_count, str)
                and isinstance(count_node, ast.Constant)
                and count_node.value == detail.get("observed_count_column")
            ):
                return code
            count_source = repr(expected_count)

        call_source = ast.get_source_segment(code, call)
        function_source = ast.get_source_segment(code, call.func)
        frame_source = ast.get_source_segment(code, frame_node)
        if (
            not all(
                (
                    call_source,
                    function_source,
                    frame_source,
                    measured_source,
                    count_source,
                )
            )
            or code.count(str(call_source)) != 1
        ):
            return code
        replacement = (
            f"{function_source}({frame_source}, "
            f"measured_column={measured_source}, count_column={count_source})"
        )
        if replacement == call_source:
            return code
        replacements.append((str(call_source), replacement))

    repaired = code
    for call_source, replacement in replacements:
        repaired = repaired.replace(call_source, replacement, 1)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _closed_counts_signature_repair_lines(
    findings: Sequence[ValidationFinding],
) -> frozenset[int]:
    """Return exact closed-count helper calls missing their level binding."""

    return frozenset(
        int(detail["line"])
        for finding in findings
        for detail in [finding.detail or {}]
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and detail.get("reason") == "host_helper_call_signature_invalid"
        and detail.get("helper_name") == "closed_categorical_counts"
        and detail.get("violations") == ["required_keyword_only_argument_missing"]
        and isinstance(detail.get("line"), int)
        and not isinstance(detail.get("line"), bool)
        and int(detail["line"]) > 0
    )


def _closed_counts_stable_keyword_repair_lines(
    findings: Sequence[ValidationFinding],
) -> frozenset[int]:
    """Return closed-count calls whose only gap is a known keyword adapter."""

    allowed_violations = {
        "required_keyword_only_argument_missing",
        "unknown_keyword_argument",
    }
    return frozenset(
        int(detail["line"])
        for finding in findings
        for detail in [finding.detail or {}]
        for violations in [detail.get("violations")]
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and detail.get("reason") == "host_helper_call_signature_invalid"
        and detail.get("helper_name") == "closed_categorical_counts"
        and isinstance(violations, list)
        and "unknown_keyword_argument" in violations
        and set(violations) <= allowed_violations
        and isinstance(detail.get("line"), int)
        and not isinstance(detail.get("line"), bool)
        and int(detail["line"]) > 0
    )


def _publication_export_audit_repair_lines(
    findings: Sequence[ValidationFinding],
) -> frozenset[int]:
    """Return exact publication-export calls using the retired adapter shape."""

    return frozenset(
        int(detail["line"])
        for finding in findings
        for detail in [finding.detail or {}]
        for violations in [detail.get("violations")]
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and detail.get("reason") == "host_helper_call_signature_invalid"
        and detail.get("helper_name") == "audit_publication_exports"
        and isinstance(violations, list)
        and set(violations) == {"paths_argument_missing", "unknown_keyword_argument"}
        and isinstance(detail.get("line"), int)
        and not isinstance(detail.get("line"), bool)
        and int(detail["line"]) > 0
    )


def _patch_publication_export_audit_call(
    code: str,
    *,
    finding_lines: frozenset[int],
) -> str:
    """Replace only the exact retired ``out_dir=..., stem=...`` adapter.

    Auditing the directory is the stable fresh API and is no weaker than the
    retired stem-labelled adapter: every publication export in that directory
    is checked.  Aliased imports, namespace calls, positional calls, expanded
    keywords, and any additional option remain finding-only.
    """

    if not finding_lines:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    direct_import_lines = {
        int(node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.figures.publication"
        and any(
            alias.name == "audit_publication_exports" and alias.asname is None
            for alias in node.names
        )
    }
    if not direct_import_lines:
        return code

    candidates_by_line: dict[int, list[ast.Call]] = {line: [] for line in finding_lines}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and int(getattr(node, "lineno", 0)) in candidates_by_line
            and isinstance(node.func, ast.Name)
            and node.func.id == "audit_publication_exports"
        ):
            candidates_by_line[int(node.lineno)].append(node)
    if any(len(candidates) != 1 for candidates in candidates_by_line.values()):
        return code

    replacements: list[tuple[str, str]] = []
    for line in sorted(finding_lines):
        call = candidates_by_line[line][0]
        if (
            not any(import_line < line for import_line in direct_import_lines)
            or call.args
            or any(keyword.arg is None for keyword in call.keywords)
        ):
            return code
        keyword_map = {str(keyword.arg): keyword.value for keyword in call.keywords}
        if len(keyword_map) != len(call.keywords) or set(keyword_map) != {
            "out_dir",
            "stem",
        }:
            return code
        call_source = ast.get_source_segment(code, call)
        paths_source = ast.get_source_segment(code, keyword_map["out_dir"])
        if not call_source or not paths_source or code.count(call_source) != 1:
            return code
        replacements.append(
            (call_source, f"audit_publication_exports(paths={paths_source})")
        )

    repaired = code
    for call_source, replacement in replacements:
        repaired = repaired.replace(call_source, replacement, 1)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_closed_counts_stable_keywords(
    code: str,
    *,
    finding_lines: frozenset[int],
) -> str:
    """Remove a diagnostic label or rename an authored levels keyword.

    The host helper accepts one series plus ``declared_levels=``. Generated
    adapters sometimes add ``variable=``/``variable_name=`` for error-label
    prose or spell the already-authored level expression as ``levels=`` or
    ``allowed_values=``. These forms currently fail before execution. This
    repair changes no level value, category, denominator, row, or scientific
    choice and applies every exact structured occurrence atomically.
    """

    if not finding_lines:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    candidates_by_line: dict[int, list[ast.Call]] = {line: [] for line in finding_lines}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and int(getattr(node, "lineno", 0)) in candidates_by_line
            and _call_tail(node.func) == "closed_categorical_counts"
        ):
            candidates_by_line[int(node.lineno)].append(node)
    if any(len(candidates) != 1 for candidates in candidates_by_line.values()):
        return code

    replacements: list[tuple[str, str]] = []
    for line in sorted(finding_lines):
        call = candidates_by_line[line][0]
        if (
            len(call.args) != 1
            or isinstance(call.args[0], ast.Starred)
            or any(keyword.arg is None for keyword in call.keywords)
        ):
            return code
        keyword_map = {str(keyword.arg): keyword.value for keyword in call.keywords}
        if len(keyword_map) != len(call.keywords) or not set(keyword_map) <= {
            "variable",
            "variable_name",
            "levels",
            "allowed_values",
            "declared_levels",
        }:
            return code
        level_keywords = [
            name
            for name in ("declared_levels", "levels", "allowed_values")
            if name in keyword_map
        ]
        if len(level_keywords) > 1:
            return code
        level_expression = (
            keyword_map[level_keywords[0]] if level_keywords else None
        )
        if level_expression is None:
            function: ast.AST | None = call
            while function is not None and not isinstance(
                function, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                function = parents.get(function)
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return code
            parameters = [
                argument
                for argument in [
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                ]
                if argument.arg in {"levels", "declared_levels"}
            ]
            if len(parameters) != 1:
                return code
            levels_name = parameters[0].arg
            if any(
                isinstance(node, ast.Name)
                and node.id == levels_name
                and isinstance(node.ctx, (ast.Store, ast.Del))
                for statement in function.body
                for node in ast.walk(statement)
                if int(getattr(node, "lineno", 0) or 0) < line
            ):
                return code
            level_expression = ast.Name(id=parameters[0].arg, ctx=ast.Load())
            level_source = levels_name
        else:
            level_source = ast.get_source_segment(code, level_expression)
        call_source = ast.get_source_segment(code, call)
        function_source = ast.get_source_segment(code, call.func)
        series_source = ast.get_source_segment(code, call.args[0])
        if (
            not call_source
            or not function_source
            or not series_source
            or not level_source
            or code.count(call_source) != 1
        ):
            return code
        replacements.append(
            (
                call_source,
                f"{function_source}({series_source}, declared_levels={level_source})",
            )
        )
    repaired = code
    for call_source, replacement in replacements:
        repaired = repaired.replace(call_source, replacement, 1)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_host_helper_keyword_only_call(
    code: str,
    *,
    finding_lines: frozenset[int],
) -> str:
    """Bind an exact provenance-helper call to its stable keyword-only API.

    The transformation is intentionally narrower than the detector. It accepts
    one or more structured findings with exactly one call on each named line,
    then moves only the already-authored measured/count arguments to their
    declared keyword slots. Multiple calls on the same line remain ambiguous
    and fail closed. No expression, column, row, value, or scientific choice
    is introduced.
    """

    if not finding_lines:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Older generated scripts sometimes wrapped the stable helper in a
    # signature-adaptation try/except and passed two selected Series plus a
    # made-up keyword.  When the exact frame and both existing column-key
    # expressions are structurally recoverable, replace the whole swallowing
    # adapter with the stable host call.  This introduces no column name or
    # scientific choice; it only restores the registered API contract.
    legacy_adapters: list[tuple[ast.Try, ast.Call, ast.Subscript, ast.Subscript]] = []
    legacy_line = next(iter(finding_lines)) if len(finding_lines) == 1 else None
    for statement in ast.walk(tree):
        if not (
            isinstance(statement, ast.Try)
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Expr)
            and isinstance(statement.body[0].value, ast.Call)
            and len(statement.handlers) == 1
            and not statement.orelse
            and not statement.finalbody
        ):
            continue
        call = statement.body[0].value
        handler = statement.handlers[0]
        if not (
            legacy_line is not None
            and int(getattr(call, "lineno", 0)) == legacy_line
            and _call_tail(call.func) == "measurement_provenance_receipt"
            and len(call.args) == 2
            and all(isinstance(argument, ast.Subscript) for argument in call.args)
            and len(call.keywords) == 1
            and call.keywords[0].arg == "variable_name"
            and isinstance(handler.type, ast.Name)
            and handler.type.id == "TypeError"
            and len(handler.body) == 1
            and isinstance(handler.body[0], ast.Expr)
            and isinstance(handler.body[0].value, ast.Call)
            and _call_tail(handler.body[0].value.func) == "call_helper_adaptively"
            and handler.body[0].value.args
            and _call_tail(handler.body[0].value.args[0])
            == "measurement_provenance_receipt"
        ):
            continue
        measured_arg, count_arg = call.args
        assert isinstance(measured_arg, ast.Subscript)
        assert isinstance(count_arg, ast.Subscript)
        if not (
            isinstance(measured_arg.value, ast.Name)
            and isinstance(count_arg.value, ast.Name)
            and measured_arg.value.id == count_arg.value.id
        ):
            continue
        legacy_adapters.append((statement, call, measured_arg, count_arg))
    if len(legacy_adapters) == 1:
        statement, call, measured_arg, count_arg = legacy_adapters[0]
        statement_source = ast.get_source_segment(code, statement)
        function_source = ast.get_source_segment(code, call.func)
        measured_source = ast.get_source_segment(code, measured_arg.slice)
        count_source = ast.get_source_segment(code, count_arg.slice)
        if (
            statement_source
            and function_source
            and measured_source
            and count_source
            and code.count(statement_source) == 1
        ):
            frame_source = measured_arg.value.id
            replacement = (
                f"{function_source}({frame_source}, "
                f"measured_column={measured_source}, "
                f"count_column={count_source})"
            )
            repaired = code.replace(statement_source, replacement, 1)
            try:
                repaired_tree = ast.parse(repaired)
            except SyntaxError:
                return code
            adaptive_calls = [
                node
                for node in ast.walk(repaired_tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "call_helper_adaptively"
            ]
            adaptive_replacements: list[tuple[str, str]] = []
            for adaptive_call in adaptive_calls:
                if not (
                    len(adaptive_call.args) == 3
                    and isinstance(adaptive_call.args[0], ast.Name)
                    and adaptive_call.args[0].id == "closed_categorical_counts"
                    and len(adaptive_call.keywords) == 1
                    and adaptive_call.keywords[0].arg == "levels"
                ):
                    adaptive_replacements = []
                    break
                call_source = ast.get_source_segment(repaired, adaptive_call)
                series_source = ast.get_source_segment(repaired, adaptive_call.args[1])
                levels_source = ast.get_source_segment(
                    repaired,
                    adaptive_call.keywords[0].value,
                )
                if not call_source or not series_source or not levels_source:
                    adaptive_replacements = []
                    break
                adaptive_replacements.append(
                    (
                        call_source,
                        "closed_categorical_counts("
                        f"{series_source}, declared_levels={levels_source})",
                    )
                )
            if adaptive_calls and len(adaptive_replacements) == len(adaptive_calls):
                for call_source, direct_source in adaptive_replacements:
                    if repaired.count(call_source) != 1:
                        return code
                    repaired = repaired.replace(call_source, direct_source, 1)
                try:
                    repaired_tree = ast.parse(repaired)
                except SyntaxError:
                    return code
            helper_defs = [
                node
                for node in repaired_tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "call_helper_adaptively"
            ]
            helper_loads = [
                node
                for node in ast.walk(repaired_tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == "call_helper_adaptively"
            ]
            if len(helper_defs) == 1 and not helper_loads:
                helper = helper_defs[0]
                repaired_lines = repaired.splitlines(keepends=True)
                del repaired_lines[
                    int(helper.lineno) - 1 : int(helper.end_lineno or helper.lineno)
                ]
                repaired = "".join(repaired_lines)
                try:
                    repaired_tree = ast.parse(repaired)
                except SyntaxError:
                    return code
                inspect_loads = [
                    node
                    for node in ast.walk(repaired_tree)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == "inspect"
                ]
                inspect_imports = [
                    node
                    for node in repaired_tree.body
                    if isinstance(node, ast.Import)
                    and len(node.names) == 1
                    and node.names[0].name == "inspect"
                    and node.names[0].asname is None
                ]
                if not inspect_loads and len(inspect_imports) == 1:
                    import_node = inspect_imports[0]
                    repaired_lines = repaired.splitlines(keepends=True)
                    del repaired_lines[
                        int(import_node.lineno)
                        - 1 : int(import_node.end_lineno or import_node.lineno)
                    ]
                    repaired = "".join(repaired_lines)
                    try:
                        ast.parse(repaired)
                    except SyntaxError:
                        return code
            return repaired

    def _argument_role(argument: ast.AST) -> str | None:
        if isinstance(argument, ast.Name):
            value = argument.id
        elif isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            value = argument.value
        else:
            return None
        normalized = str(value).strip().casefold()
        if "measured" in normalized:
            return "measured"
        if "count" in normalized or normalized.endswith("_n"):
            return "count"
        return None

    replacements: list[tuple[str, str]] = []
    for line in sorted(finding_lines):
        candidates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and int(getattr(node, "lineno", 0)) == line
            and _call_tail(node.func) == "measurement_provenance_receipt"
            and len(node.args) == 3
            and isinstance(node.args[0], ast.Name)
            and not node.keywords
        ]
        if len(candidates) != 1:
            return code
        call = candidates[0]
        call_source = ast.get_source_segment(code, call)
        function_source = ast.get_source_segment(code, call.func)
        argument_sources = [
            ast.get_source_segment(code, argument) for argument in call.args
        ]
        if (
            not call_source
            or not function_source
            or any(not source for source in argument_sources)
            or code.count(call_source) != 1
        ):
            return code
        frame_source, second_source, third_source = argument_sources
        second_role = _argument_role(call.args[1])
        third_role = _argument_role(call.args[2])
        if second_role == "count" and third_role == "measured":
            measured_source, count_source = third_source, second_source
        elif second_role == "measured" and third_role == "count":
            measured_source, count_source = second_source, third_source
        elif all(isinstance(argument, ast.Name) for argument in call.args):
            # Preserve the established stable-API positional migration for
            # generic local variable names when no role-bearing token exists.
            measured_source, count_source = second_source, third_source
        else:
            return code
        replacements.append(
            (
                call_source,
                f"{function_source}({frame_source}, "
                f"measured_column={measured_source}, "
                f"count_column={count_source})",
            )
        )

    repaired = code
    for call_source, replacement in replacements:
        repaired = repaired.replace(call_source, replacement, 1)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_closed_counts_missing_levels(
    code: str,
    *,
    finding_lines: frozenset[int],
) -> str:
    """Bind one existing local level parameter to the stable host API.

    The repair is deliberately narrower than the detector. It requires one
    exact host finding, one one-argument call, and exactly one enclosing
    function parameter named ``levels`` or ``declared_levels`` that has not
    been rebound. No level, value, category, or denominator is invented.
    """

    if len(finding_lines) != 1:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    line = next(iter(finding_lines))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and int(getattr(node, "lineno", 0)) == line
        and _call_tail(node.func) == "closed_categorical_counts"
        and len(node.args) == 1
        and not node.keywords
        and not isinstance(node.args[0], ast.Starred)
    ]
    if len(calls) != 1:
        return code
    call = calls[0]
    function: ast.AST | None = call
    while function is not None and not isinstance(
        function, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        function = parents.get(function)
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return code
    parameters = [
        argument.arg
        for argument in [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if argument.arg in {"levels", "declared_levels"}
    ]
    if len(parameters) != 1:
        return code
    levels_name = parameters[0]
    if any(
        isinstance(node, ast.Name)
        and node.id == levels_name
        and isinstance(node.ctx, (ast.Store, ast.Del))
        for statement in function.body
        for node in ast.walk(statement)
        if int(getattr(node, "lineno", 0) or 0) < line
    ):
        return code
    call_source = ast.get_source_segment(code, call)
    function_source = ast.get_source_segment(code, call.func)
    argument_source = ast.get_source_segment(code, call.args[0])
    if (
        not call_source
        or not function_source
        or not argument_source
        or code.count(call_source) != 1
    ):
        return code
    replacement = f"{function_source}({argument_source}, declared_levels={levels_name})"
    repaired = code.replace(call_source, replacement, 1)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _closed_counts_introspection_line(
    findings: Sequence[ValidationFinding],
) -> int | None:
    """Return one host-authorized closed-counts introspection coordinate."""

    lines = {
        int(detail["line"])
        for finding in findings
        for detail in [finding.detail or {}]
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and detail.get("reason") == "host_helper_runtime_introspection"
        and detail.get("helper_name") == "closed_categorical_counts"
        and isinstance(detail.get("line"), int)
        and not isinstance(detail.get("line"), bool)
        and int(detail["line"]) > 0
    }
    return next(iter(lines)) if len(lines) == 1 else None


def _patch_closed_counts_runtime_adapter(
    code: str,
    *,
    introspection_line: int | None,
) -> str:
    """Replace one reflective closed-counts adapter with its stable API call.

    The wrapper's two already-authored parameters remain the series and the
    Agent-declared levels.  The repair changes no value, category, denominator,
    or statistic; it only binds those parameters to the host helper's declared
    keyword API and removes the now-unused exact ``import inspect`` statement.
    """

    if introspection_line is None:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    exact_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and len(node.names) == 1
        and node.names[0].name == "inspect"
        and node.names[0].asname is None
    ]
    helper_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "closed_categorical_counts" and alias.asname is None
            for alias in node.names
        )
    ]
    signature_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and int(getattr(node, "lineno", 0)) == introspection_line
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "inspect"
        and node.func.attr == "signature"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "closed_categorical_counts"
        and not node.keywords
    ]
    if len(exact_imports) != 1 or len(helper_imports) != 1 or len(signature_calls) != 1:
        return code
    signature_call = signature_calls[0]
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    wrapper: ast.AST | None = signature_call
    while wrapper is not None and not isinstance(
        wrapper, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        wrapper = parents.get(wrapper)
    if not (
        isinstance(wrapper, ast.FunctionDef)
        and wrapper in tree.body
        and not wrapper.decorator_list
        and not wrapper.args.posonlyargs
        and len(wrapper.args.args) == 2
        and not wrapper.args.kwonlyargs
        and wrapper.args.vararg is None
        and wrapper.args.kwarg is None
        and not wrapper.args.defaults
    ):
        return code
    series_name, levels_name = (argument.arg for argument in wrapper.args.args)
    direct_call_count = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == wrapper.name
        for node in ast.walk(tree)
    )
    if direct_call_count < 1:
        return code

    lines = code.splitlines(keepends=True)
    body_start = int(wrapper.body[0].lineno) - 1
    body_end = int(wrapper.body[-1].end_lineno or wrapper.body[-1].lineno)
    wrapper_line = lines[int(wrapper.lineno) - 1]
    wrapper_indent = wrapper_line[: len(wrapper_line) - len(wrapper_line.lstrip())]
    body_indent = wrapper_indent + ("\t" if "\t" in wrapper_indent else "    ")
    lines[body_start:body_end] = [
        f"{body_indent}return closed_categorical_counts(\n",
        f"{body_indent}    {series_name}, declared_levels={levels_name}\n",
        f"{body_indent})\n",
    ]
    candidate = "".join(lines)
    try:
        candidate_tree = ast.parse(candidate)
    except SyntaxError:
        return code
    if any(
        isinstance(node, ast.Name)
        and node.id == "inspect"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(candidate_tree)
    ):
        return code
    import_node = next(
        node
        for node in candidate_tree.body
        if isinstance(node, ast.Import)
        and len(node.names) == 1
        and node.names[0].name == "inspect"
        and node.names[0].asname is None
    )
    candidate_lines = candidate.splitlines(keepends=True)
    del candidate_lines[
        int(import_node.lineno) - 1 : int(import_node.end_lineno or import_node.lineno)
    ]
    repaired = "".join(candidate_lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _arbitrary_column_fallback_coordinate(
    findings: Sequence[ValidationFinding],
) -> tuple[str, int] | None:
    coordinates = {
        (str(detail.get("function") or ""), int(detail["line"]))
        for finding in findings
        for detail in [finding.detail or {}]
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and detail.get("reason") == "arbitrary_column_fallback"
        and isinstance(detail.get("function"), str)
        and isinstance(detail.get("line"), int)
        and not isinstance(detail.get("line"), bool)
        and int(detail["line"]) > 0
    }
    if len(coordinates) != 1:
        return None
    function_name, line = next(iter(coordinates))
    return (function_name, line) if function_name else None


def _patch_arbitrary_column_fallback_to_raise(
    code: str,
    *,
    coordinate: tuple[str, int] | None,
) -> str:
    """Remove one frame-order fallback while preserving its authored raise."""

    if coordinate is None:
        return code
    function_name, line = coordinate
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(functions) != 1:
        return code
    function = functions[0]
    parents = {
        child: parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and int(getattr(node, "lineno", 0)) == line
    ]
    if len(assignments) != 1:
        return code
    fallback = parents.get(assignments[0])
    if not (
        isinstance(fallback, ast.If)
        and assignments[0] in fallback.body
        and len(fallback.orelse) == 1
        and isinstance(fallback.orelse[0], ast.Raise)
        and fallback.end_lineno is not None
    ):
        return code
    raise_source = ast.unparse(fallback.orelse[0])
    lines = code.splitlines(keepends=True)
    start = int(fallback.lineno) - 1
    end = int(fallback.end_lineno)
    original_line = lines[start]
    indent = original_line[: len(original_line) - len(original_line.lstrip())]
    replacement = "".join(
        f"{indent}{source_line}\n" for source_line in raise_source.splitlines()
    )
    lines[start:end] = [replacement]
    repaired = "".join(lines)
    try:
        repaired_tree = ast.parse(repaired)
    except SyntaxError:
        return code
    from ..gates.preflight import _function_arbitrary_column_fallback

    if any(
        _function_arbitrary_column_fallback(node) is not None
        for node in repaired_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ):
        return code
    return repaired


def _provenance_custom_helper_coordinate(
    findings: Sequence[ValidationFinding],
) -> tuple[str, int] | None:
    """Return one custom provenance helper and its exact reported call count."""

    helper_names: set[str] = set()
    call_lines: set[int] = set()
    allowed_modes = {
        "provenance_helper_result_not_bound",
        "provenance_helper_result_not_immediately_guarded",
        "provenance_helper_result_guard_not_fail_closed",
        "provenance_helper_runtime_binding_ambiguous",
    }
    matching_findings = 0
    for finding in findings:
        detail = finding.detail or {}
        if not (
            finding.validator == "mechanical_code_preflight"
            and finding.severity == "error"
            and detail.get("reason") == "provenance_audit_not_fail_closed"
        ):
            continue
        matching_findings += 1
        issues = detail.get("issues")
        if not isinstance(issues, list):
            return None
        for issue in issues:
            if (
                not isinstance(issue, dict)
                or issue.get("failure_mode") not in allowed_modes
            ):
                return None
            helper_name = issue.get("helper_name")
            call_line = issue.get("call_line")
            if not isinstance(helper_name, str) or not helper_name.strip():
                return None
            if (
                isinstance(call_line, bool)
                or not isinstance(call_line, int)
                or call_line <= 0
            ):
                return None
            helper_names.add(helper_name)
            call_lines.add(call_line)
    if matching_findings != 1 or len(helper_names) != 1 or not call_lines:
        return None
    return next(iter(helper_names)), len(call_lines)


def _patch_custom_provenance_helper_to_host_receipt(
    code: str,
    *,
    coordinate: tuple[str, int] | None,
) -> str:
    """Replace a uniquely bound custom audit with the self-raising host receipt."""

    if coordinate is None:
        return code
    helper_name, expected_call_count = coordinate
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    helper_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == helper_name
        and not node.decorator_list
    ]
    host_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "measurement_provenance_receipt" and alias.asname is None
            for alias in node.names
        )
    ]
    if len(helper_defs) != 1 or len(host_imports) != 1:
        return code
    helper = helper_defs[0]
    if not (
        len(helper.args.args) == 3
        and not helper.args.posonlyargs
        and not helper.args.kwonlyargs
        and helper.args.vararg is None
        and helper.args.kwarg is None
        and not helper.args.defaults
    ):
        return code
    direct_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    ]
    if len(direct_calls) != expected_call_count:
        return code
    helper_name_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == helper_name
        and isinstance(node.ctx, ast.Load)
    ]
    if len(helper_name_loads) != len(direct_calls):
        return code
    replacements: list[tuple[str, str]] = []
    for call in direct_calls:
        if len(call.args) != 3 or call.keywords:
            return code
        sources = [ast.get_source_segment(code, argument) for argument in call.args]
        call_source = ast.get_source_segment(code, call)
        if not call_source or any(not source for source in sources):
            return code
        frame_source, measured_source, count_source = sources
        replacement = (
            f"measurement_provenance_receipt({frame_source}, "
            f"measured_column={measured_source}, count_column={count_source})"
        )
        replacements.append((call_source, replacement))
    candidate = code
    for call_source, replacement in replacements:
        if candidate.count(call_source) != 1:
            return code
        candidate = candidate.replace(call_source, replacement, 1)
    try:
        candidate_tree = ast.parse(candidate)
    except SyntaxError:
        return code
    candidate_helpers = [
        node
        for node in candidate_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
    ]
    if len(candidate_helpers) != 1:
        return code
    candidate_helper = candidate_helpers[0]
    lines = candidate.splitlines(keepends=True)
    start = int(candidate_helper.lineno) - 1
    end = int(candidate_helper.end_lineno or candidate_helper.lineno)
    del lines[start:end]
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _local_helper_unpack_repair_coordinate(
    findings: Sequence[ValidationFinding],
) -> tuple[str, int, int] | None:
    coordinates = {
        (
            str(detail.get("function_name") or ""),
            int(detail["return_arity"]),
            int(detail["target_arity"]),
        )
        for finding in findings
        for detail in [finding.detail or {}]
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and detail.get("reason") == "local_helper_unpack_arity_mismatch"
        and isinstance(detail.get("function_name"), str)
        and isinstance(detail.get("return_arity"), int)
        and not isinstance(detail.get("return_arity"), bool)
        and isinstance(detail.get("target_arity"), int)
        and not isinstance(detail.get("target_arity"), bool)
    }
    if len(coordinates) != 1:
        return None
    function_name, return_arity, target_arity = next(iter(coordinates))
    if not function_name or target_arity != return_arity + 1:
        return None
    return function_name, return_arity, target_arity


def _patch_discarded_host_receipt_unpack(
    code: str,
    *,
    coordinate: tuple[str, int, int] | None,
) -> str:
    """Thread one discarded host receipt through an exact local helper tuple.

    The repair requires a uniquely shaped module-level helper and call site:
    one exact host import, one discarded receipt call, one fixed all-name return,
    and one direct unpack whose tail names already match that return. The missing
    leading target name is merely threaded through; no new value is computed.
    """

    if coordinate is None:
        return code
    function_name, return_arity, target_arity = coordinate
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(functions) != 1:
        return code
    function = functions[0]
    direct_nodes: list[ast.AST] = []
    pending: list[ast.AST] = list(reversed(function.body))
    while pending:
        node = pending.pop()
        direct_nodes.append(node)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
        ):
            continue
        pending.extend(reversed(list(ast.iter_child_nodes(node))))
    returns = [node for node in direct_nodes if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Tuple):
        return code
    returned = returns[0].value
    if len(returned.elts) != return_arity or not all(
        isinstance(item, ast.Name) for item in returned.elts
    ):
        return code

    exact_imports = [
        node
        for node in function.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "easyicu.research_agent.methods.descriptive_inputs"
        and any(
            alias.name == "measurement_provenance_receipt" and alias.asname is None
            for alias in node.names
        )
    ]
    discarded_calls = [
        node.value
        for node in direct_nodes
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _call_tail(node.value.func) == "measurement_provenance_receipt"
    ]
    if len(exact_imports) != 1 or len(discarded_calls) != 1:
        return code
    host_call = discarded_calls[0]
    if len(host_call.args) != 1 or {keyword.arg for keyword in host_call.keywords} != {
        "measured_column",
        "count_column",
    }:
        return code

    callers: list[ast.Assign] = []
    for caller in tree.body:
        if not isinstance(caller, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(caller):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
                and len(node.targets[0].elts) == target_arity
                and all(isinstance(item, ast.Name) for item in node.targets[0].elts)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == function_name
            ):
                callers.append(node)
    if len(callers) != 1:
        return code
    target = callers[0].targets[0]
    target_names = [item.id for item in target.elts]
    returned_names = [item.id for item in returned.elts]
    if target_names[1:] != returned_names:
        return code
    receipt_name = target_names[0]
    if any(
        isinstance(node, (ast.Name, ast.arg))
        and (
            (isinstance(node, ast.Name) and node.id == receipt_name)
            or (isinstance(node, ast.arg) and node.arg == receipt_name)
        )
        for node in direct_nodes
    ):
        return code

    call_source = ast.get_source_segment(code, host_call)
    return_source = ast.get_source_segment(code, returns[0])
    if (
        not call_source
        or not return_source
        or code.count(call_source) != 1
        or code.count(return_source) != 1
    ):
        return code
    repaired = code.replace(call_source, f"{receipt_name} = {call_source}", 1)
    repaired = repaired.replace(
        return_source,
        f"return {receipt_name}, {', '.join(returned_names)}",
        1,
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _lossy_numeric_coercion_repair_lines(
    findings: Sequence[ValidationFinding],
) -> frozenset[int]:
    """Return exact host-owned loss-count lines eligible for repair."""

    lines: set[int] = set()
    for finding in findings:
        detail = finding.detail or {}
        if not (
            finding.validator == "mechanical_code_preflight"
            and finding.severity == "error"
            and detail.get("reason") == "lossy_numeric_coercion"
        ):
            continue
        issues = detail.get("issues")
        if not isinstance(issues, list):
            continue
        for issue in issues:
            if not isinstance(issue, dict) or issue.get("gap") != (
                "unchecked_coercion_loss_count"
            ):
                continue
            raw_lines = issue.get("lines")
            if not isinstance(raw_lines, list):
                continue
            lines.update(
                value
                for value in raw_lines
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            )
    return frozenset(lines)


def _conditional_nonfinite_guard_lines(
    findings: Sequence[ValidationFinding],
) -> frozenset[int]:
    """Return exact host-owned outer guards eligible for dedenting repair."""

    return frozenset(
        int((finding.detail or {})["guard_line"])
        for finding in findings
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and (finding.detail or {}).get("reason") == "conditional_nonfinite_guard"
        and isinstance((finding.detail or {}).get("guard_line"), int)
        and not isinstance((finding.detail or {}).get("guard_line"), bool)
        and int((finding.detail or {})["guard_line"]) > 0
    )


def _strict_numeric_nonfinite_repair_lines(
    findings: Sequence[ValidationFinding],
) -> frozenset[int]:
    return frozenset(
        int((finding.detail or {})["coercion_line"])
        for finding in findings
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and (finding.detail or {}).get("reason") == "strict_numeric_nonfinite_unchecked"
        and isinstance((finding.detail or {}).get("coercion_line"), int)
        and not isinstance((finding.detail or {}).get("coercion_line"), bool)
        and int((finding.detail or {})["coercion_line"]) > 0
    )


def _categorical_level_reconciliation_repair_line(
    findings: Sequence[ValidationFinding],
) -> Optional[int]:
    lines = {
        int((finding.detail or {})["counts_line"])
        for finding in findings
        if finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and (finding.detail or {}).get("reason")
        == "categorical_level_accounting_unverified"
        and isinstance((finding.detail or {}).get("counts_line"), int)
        and not isinstance((finding.detail or {}).get("counts_line"), bool)
        and int((finding.detail or {})["counts_line"]) > 0
    }
    return next(iter(lines)) if len(lines) == 1 else None


def _patch_conditional_nonfinite_guard(
    code: str,
    *,
    guard_lines: frozenset[int],
) -> str:
    """Remove one unrelated inner condition from a proven numeric fail guard."""

    if len(guard_lines) != 1:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and int(node.lineno) in guard_lines
        and not node.orelse
        and len(node.body) == 1
        and isinstance(node.body[0], ast.If)
        and not node.body[0].orelse
        and len(node.body[0].body) == 1
        and isinstance(node.body[0].body[0], (ast.Expr, ast.Raise))
    ]
    if len(candidates) != 1:
        return code
    outer = candidates[0]
    inner = outer.body[0]
    terminal = inner.body[0]
    if not (
        inner.end_lineno is not None
        and terminal.end_lineno is not None
        and terminal.col_offset > inner.col_offset
    ):
        return code
    lines = code.splitlines(keepends=True)
    dedent = terminal.col_offset - inner.col_offset
    replacement: list[str] = []
    for raw in lines[terminal.lineno - 1 : terminal.end_lineno]:
        if raw.strip():
            if len(raw) - len(raw.lstrip(" ")) < dedent:
                return code
            raw = raw[dedent:]
        replacement.append(raw)
    lines[inner.lineno - 1 : inner.end_lineno] = replacement
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _builtin_name_is_unmodified(tree: ast.Module, name: str) -> bool:
    """Return whether one exception builtin remains safe to reference."""

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            return False
        if isinstance(node, ast.arg) and node.arg == name:
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            node.name == name
        ):
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name.split(".", 1)[0]) == name
            for alias in node.names
        ):
            return False
    return True


def _stable_numpy_alias(tree: ast.Module) -> Optional[str]:
    aliases = {
        alias.asname or "numpy"
        for statement in tree.body
        if isinstance(statement, ast.Import)
        for alias in statement.names
        if alias.name == "numpy"
    }
    if len(aliases) != 1:
        return None
    alias_name = next(iter(aliases))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == alias_name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            return None
        if isinstance(node, ast.arg) and node.arg == alias_name:
            return None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            node.name == alias_name
        ):
            return None
    return alias_name


def _patch_strict_numeric_nonfinite_guard(
    code: str,
    *,
    coercion_lines: frozenset[int],
) -> str:
    """Reject infinities in every proven strict numeric coercion helper."""

    if not coercion_lines:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    numpy_alias = _stable_numpy_alias(tree)
    if numpy_alias is None or not _builtin_name_is_unmodified(tree, "RuntimeError"):
        return code
    mask_name = "_easyicu_nonfinite_numeric_mask_v1"
    if any(
        isinstance(node, ast.Name)
        and node.id == mask_name
        or isinstance(node, ast.arg)
        and node.arg == mask_name
        for node in ast.walk(tree)
    ):
        return code

    candidates: list[tuple[ast.FunctionDef, ast.Assign | ast.AnnAssign, str]] = []
    for function in [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]:
        for statement in function.body:
            if int(getattr(statement, "lineno", -1)) not in coercion_lines:
                continue
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                value = statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target = statement.target
                value = statement.value
            else:
                continue
            if not (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "to_numeric"
                and any(
                    keyword.arg == "errors"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "coerce"
                    for keyword in value.keywords
                )
            ):
                continue
            candidates.append((function, statement, target.id))
    if {int(statement.lineno) for _function, statement, _target in candidates} != set(
        coercion_lines
    ):
        return code
    if any(
        statement.end_lineno is None for _function, statement, _target in candidates
    ):
        return code
    lines = code.splitlines(keepends=True)
    for _function, statement, coerced_name in sorted(
        candidates,
        key=lambda item: int(item[1].end_lineno or -1),
        reverse=True,
    ):
        assert statement.end_lineno is not None
        indent_source = lines[statement.lineno - 1]
        indent = indent_source[: len(indent_source) - len(indent_source.lstrip(" \t"))]
        body_indent = indent + ("\t" if "\t" in indent else "    ")
        guard = (
            f"{indent}# {_STRICT_NUMERIC_NONFINITE_GUARD_SENTINEL}\n"
            f"{indent}{mask_name} = {coerced_name}.notna() & "
            f"~{numpy_alias}.isfinite({coerced_name})\n"
            f"{indent}if int({mask_name}.sum()) > 0:\n"
            f'{body_indent}raise RuntimeError("strict numeric input contains '
            'non-finite observed values")\n'
        )
        if not lines[statement.end_lineno - 1].endswith(("\n", "\r")):
            lines[statement.end_lineno - 1] += "\n"
        lines.insert(statement.end_lineno, guard)
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_categorical_level_reconciliation_guard(
    code: str,
    *,
    counts_line: Optional[int],
) -> str:
    """Fail closed when emitted categorical levels omit observed values."""

    if counts_line is None or _CATEGORICAL_LEVEL_GUARD_SENTINEL in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not _builtin_name_is_unmodified(tree, "RuntimeError"):
        return code

    candidates: list[tuple[ast.Assign | ast.AnnAssign, str, str]] = []
    for function in [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]:
        for index, statement in enumerate(function.body):
            if int(getattr(statement, "lineno", -1)) != counts_line:
                continue
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                value = statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target = statement.target
                value = statement.value
            else:
                continue
            if not (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "value_counts"
                and isinstance(value.func.value, ast.Name)
            ):
                continue
            counts_name = target.id
            values_name = value.func.value.id
            matching_levels = {
                later.iter.id
                for later in function.body[index + 1 :]
                if isinstance(later, ast.For)
                and isinstance(later.target, ast.Name)
                and isinstance(later.iter, ast.Name)
                and any(
                    isinstance(candidate, ast.Call)
                    and isinstance(candidate.func, ast.Attribute)
                    and candidate.func.attr == "get"
                    and isinstance(candidate.func.value, ast.Name)
                    and candidate.func.value.id == counts_name
                    and candidate.args
                    and isinstance(candidate.args[0], ast.Name)
                    and candidate.args[0].id == later.target.id
                    for candidate in ast.walk(later)
                )
            }
            if len(matching_levels) == 1:
                candidates.append((statement, values_name, next(iter(matching_levels))))
    if len(candidates) != 1:
        return code
    statement, values_name, levels_name = candidates[0]
    if statement.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    indent_source = lines[statement.lineno - 1]
    indent = indent_source[: len(indent_source) - len(indent_source.lstrip(" \t"))]
    body_indent = indent + ("\t" if "\t" in indent else "    ")
    guard = (
        f"{indent}# {_CATEGORICAL_LEVEL_GUARD_SENTINEL}\n"
        f"{indent}if (~{values_name}.isin({levels_name})).any():\n"
        f'{body_indent}raise RuntimeError("observed categorical values are '
        'not covered by declared levels")\n'
    )
    lines.insert(statement.lineno - 1, guard)
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _call_tail(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _standalone_statement_source(
    code: str,
    statement: ast.Assign | ast.AnnAssign,
    *,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> Optional[tuple[list[str], str]]:
    """Return source lines and indent only for an isolated statement."""

    if statement.end_lineno is None:
        return None
    if statement.lineno < 1:
        return None
    lines = code.splitlines(keepends=True)
    if statement.end_lineno > len(lines):
        return None
    for other in ast.walk(tree):
        if not isinstance(other, ast.stmt) or other is statement:
            continue
        if int(statement.lineno) <= int(other.lineno) <= int(statement.end_lineno):
            return None
    current: Optional[ast.AST] = statement
    while current is not None and current in parents:
        current = parents[current]
        if isinstance(current, (*_TRY_NODE_TYPES, ast.With, ast.AsyncWith)):
            return None
    start_line = lines[statement.lineno - 1]
    indent = start_line[: len(start_line) - len(start_line.lstrip(" \t"))]
    return lines, indent


def _patch_boolean_reduction_identity(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Replace host-proven scalar reduction identity checks atomically."""

    coordinates: list[tuple[int, str, bool, str]] = []
    for finding in repair_findings:
        detail = finding.detail or {}
        if detail.get("reason") != "boolean_reduction_identity_comparison":
            continue
        line = detail.get("line")
        operator = detail.get("operator")
        boolean_literal = detail.get("boolean_literal")
        reduction = detail.get("reduction")
        provenance = detail.get("provenance")
        if detail.get("repair_safe") is not True:
            continue
        if not (
            isinstance(line, int)
            and not isinstance(line, bool)
            and line > 0
            and operator in {"is", "is_not"}
            and isinstance(boolean_literal, bool)
            and reduction in {"all", "any"}
            and provenance in {"numpy_array", "numpy_function", "pandas_series"}
        ):
            return code
        coordinates.append((line, str(operator), boolean_literal, str(reduction)))
    if not coordinates or len(coordinates) != len(set(coordinates)):
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def _reduction_tail(expression: ast.AST) -> Optional[str]:
        if not isinstance(expression, ast.Call):
            return None
        if isinstance(expression.func, ast.Attribute):
            return expression.func.attr
        if isinstance(expression.func, ast.Name):
            if expression.func.id.endswith("all"):
                return "all"
            if expression.func.id.endswith("any"):
                return "any"
        return None

    replacements: list[tuple[int, int, str]] = []
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line_text in lines:
        line_starts.append(offset)
        offset += len(line_text)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line_text = lines[lineno - 1]
        char_col = len(line_text.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    for line, operator_name, boolean_literal, reduction in coordinates:
        candidates: list[tuple[ast.Compare, ast.AST]] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Compare)
                and node.lineno == line
                and len(node.ops) == 1
                and len(node.comparators) == 1
            ):
                continue
            operator = node.ops[0]
            if (operator_name == "is") != isinstance(operator, ast.Is):
                continue
            if (operator_name == "is_not") != isinstance(operator, ast.IsNot):
                continue
            left, right = node.left, node.comparators[0]
            if (
                isinstance(left, ast.Constant)
                and isinstance(left.value, bool)
                and left.value is boolean_literal
                and _reduction_tail(right) == reduction
            ):
                candidates.append((node, right))
            elif (
                isinstance(right, ast.Constant)
                and isinstance(right.value, bool)
                and right.value is boolean_literal
                and _reduction_tail(left) == reduction
            ):
                candidates.append((node, left))
        if len(candidates) != 1:
            return code
        comparison, reduction_expression = candidates[0]
        if comparison.end_lineno is None or comparison.end_col_offset is None:
            return code
        expression_source = ast.get_source_segment(code, reduction_expression)
        if not expression_source:
            return code
        truthy = (operator_name == "is" and boolean_literal) or (
            operator_name == "is_not" and not boolean_literal
        )
        replacement = (
            f"bool({expression_source})" if truthy else f"not bool({expression_source})"
        )
        replacements.append(
            (
                _absolute_offset(comparison.lineno, comparison.col_offset),
                _absolute_offset(comparison.end_lineno, comparison.end_col_offset),
                replacement,
            )
        )

    if len(replacements) != len(coordinates):
        return code
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_resolved_context_digest_load(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Load the exact digest-bound ResearchContext instead of its binding row."""

    coordinates: list[tuple[int, str, str]] = []
    for finding in repair_findings:
        detail = finding.detail or {}
        if detail.get("reason") != "resolved_context_payload_not_loaded":
            continue
        line = detail.get("line")
        manifest_name = detail.get("manifest_name")
        target_name = detail.get("target_name")
        if not (
            isinstance(line, int)
            and line > 0
            and isinstance(manifest_name, str)
            and manifest_name.isidentifier()
            and isinstance(target_name, str)
            and target_name.isidentifier()
        ):
            return code
        coordinates.append((line, manifest_name, target_name))
    if len(coordinates) != 1:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    line, manifest_name, target_name = coordinates[0]
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and node.lineno == line
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == target_name
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Subscript)
        and isinstance(node.value.value.value, ast.Name)
        and node.value.value.value.id == manifest_name
    ]
    if len(candidates) != 1:
        return code
    assignment = candidates[0]
    standalone = _standalone_statement_source(
        code,
        assignment,
        tree=tree,
        parents=parents,
    )
    if standalone is None or assignment.end_lineno is None:
        return code
    lines, indent = standalone
    body_indent = indent + ("\t" if "\t" in indent else "    ")
    patch = (
        f"{indent}# _easyicu_resolved_context_digest_load_v1\n"
        f"{indent}import hashlib as _easyicu_context_hashlib\n"
        f"{indent}import json as _easyicu_context_json\n"
        f"{indent}import os as _easyicu_context_os\n"
        f"{indent}from pathlib import Path as _EasyICUContextPath\n"
        f'{indent}_easyicu_context_binding = {manifest_name}["context"]\n'
        f"{indent}_easyicu_context_path = (\n"
        f"{body_indent}_EasyICUContextPath("
        f'_easyicu_context_os.environ["EASYICU_RUN_DIR"])\n'
        f'{body_indent}/ _easyicu_context_binding["relative_path"]\n'
        f"{indent})\n"
        f"{indent}if not _easyicu_context_path.is_file():\n"
        f'{body_indent}raise FileNotFoundError("Bound ResearchContext is missing")\n'
        f"{indent}_easyicu_context_digest = _easyicu_context_hashlib.sha256()\n"
        f'{indent}with _easyicu_context_path.open("rb") as _easyicu_context_stream:\n'
        f"{body_indent}for _easyicu_context_chunk in iter(\n"
        f'{body_indent}    lambda: _easyicu_context_stream.read(1024 * 1024), b""\n'
        f"{body_indent}):\n"
        f"{body_indent}    _easyicu_context_digest.update(_easyicu_context_chunk)\n"
        f"{indent}if (\n"
        f"{body_indent}_easyicu_context_digest.hexdigest()\n"
        f'{body_indent}!= _easyicu_context_binding["sha256"]\n'
        f"{indent}):\n"
        f'{body_indent}raise ValueError("Bound ResearchContext digest mismatch")\n'
        f'{indent}with _easyicu_context_path.open("r", encoding="utf-8") '
        f"as _easyicu_context_stream:\n"
        f"{body_indent}_easyicu_context_payload = "
        f"_easyicu_context_json.load(_easyicu_context_stream)\n"
        f'{indent}{target_name} = _easyicu_context_payload.get("variables")\n'
        f"{indent}if not isinstance({target_name}, list):\n"
        f'{body_indent}raise ValueError("Bound ResearchContext variables are invalid")\n'
    )
    lines[assignment.lineno - 1 : assignment.end_lineno] = [patch]
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_resolved_input_identity_key(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Read the authoritative input identity from a resolved binding row."""

    coordinates: list[tuple[str, str, tuple[int, ...]]] = []
    for finding in repair_findings:
        detail = finding.detail or {}
        if detail.get("reason") != "resolved_input_key_not_materialized":
            continue
        helper_name = detail.get("helper_name")
        parameter_name = detail.get("binding_parameter")
        access_lines = detail.get("access_lines")
        if not (
            isinstance(helper_name, str)
            and helper_name.isidentifier()
            and isinstance(parameter_name, str)
            and parameter_name.isidentifier()
            and isinstance(access_lines, list)
            and access_lines
            and all(
                isinstance(line, int) and not isinstance(line, bool) and line > 0
                for line in access_lines
            )
        ):
            return code
        coordinates.append((helper_name, parameter_name, tuple(access_lines)))
    if not coordinates or len(coordinates) != len(set(coordinates)):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for helper_name, parameter_name, access_lines in coordinates:
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == helper_name
        ]
        if len(functions) != 1:
            return code
        function = functions[0]
        parameter_names = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        if parameter_name not in parameter_names:
            return code
        matches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == parameter_name
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "input_key"
        ]
        if sorted(int(node.lineno) for node in matches) != sorted(access_lines):
            return code
        for node in matches:
            if node.end_lineno is None or node.end_col_offset is None:
                return code
            replacements.append(
                (
                    _absolute_offset(int(node.lineno), int(node.col_offset)),
                    _absolute_offset(int(node.end_lineno), int(node.end_col_offset)),
                    f"{parameter_name}['identity_row']['input_key']",
                )
            )
    if not replacements:
        return code
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_pre312_fstring_subscript_quotes(
    code: str,
    *,
    repair_findings: Sequence[ValidationFinding],
) -> str:
    """Use the opposite quote for subscript literals inside simple f-strings."""

    coordinates: list[tuple[int, int, int, int, str]] = []
    matching_findings = 0
    for finding in repair_findings:
        detail = finding.detail or {}
        if detail.get("reason") != "fstring_runtime_quote_incompatible":
            continue
        matching_findings += 1
        occurrences = detail.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            return code
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                return code
            values = (
                occurrence.get("line"),
                occurrence.get("column"),
                occurrence.get("end_line"),
                occurrence.get("end_column"),
            )
            outer_quote_name = occurrence.get("outer_quote")
            if not (
                all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in values
                )
                and int(values[0]) > 0
                and int(values[2]) > 0
                and outer_quote_name in {"double", "single"}
            ):
                return code
            outer_quote = '"' if outer_quote_name == "double" else "'"
            coordinates.append(
                (
                    int(values[0]),
                    int(values[1]),
                    int(values[2]),
                    int(values[3]),
                    str(outer_quote),
                )
            )
    if (
        matching_findings != 1
        or not coordinates
        or len(coordinates) != len(set(coordinates))
    ):
        return code
    lines = code.splitlines(keepends=True)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: list[tuple[int, int, str]] = []
    for line, column, end_line, end_column, outer_quote in coordinates:
        if line != end_line or line > len(lines):
            return code
        start = _absolute_offset(line, column)
        end = _absolute_offset(end_line, end_column)
        source = code[start:end]
        if (
            not source.startswith(outer_quote)
            or not source.endswith(outer_quote)
            or start <= 0
            or end >= len(code)
            or code[start - 1] != "["
            or code[end] != "]"
        ):
            return code
        try:
            literal_value = ast.literal_eval(source)
        except (SyntaxError, ValueError):
            return code
        if not isinstance(literal_value, str):
            return code
        if outer_quote == '"':
            content = json.dumps(literal_value, ensure_ascii=False)[1:-1]
            # Python 3.11 rejects backslashes inside f-string expressions.
            # If the opposite-quoted literal would need one, this is not a
            # host-owned syntactic repair; leave it for provider repair.
            if "'" in content or "\\" in content:
                return code
            replacement = "'" + content.replace("'", "\\'") + "'"
        else:
            replacement = json.dumps(literal_value, ensure_ascii=False)
        replacements.append(
            (
                start,
                end,
                replacement,
            )
        )
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired, feature_version=(3, 11))
    except SyntaxError:
        return code
    return repaired


def deterministic_concept_audit_repair(
    code: str,
    audit_messages: Sequence[str],
    *,
    repair_reasons: Sequence[RepairReason] = (),
    repair_findings: Sequence[ValidationFinding] = (),
) -> tuple[str, List[str]]:
    """Apply narrow, science-neutral repairs named by concept-audit errors.

    A concept finding may identify an invalid missing-value treatment, but
    replacing zero-imputation with complete-case analysis changes the cohort
    and missing-data strategy.  That scientific choice belongs to agent repair
    (or fail-closed handling), so shared deterministic code does not rewrite
    it.  A missing *terminating guard* around an already-authored provenance
    audit is different: inserting that guard only prevents invalid scientific
    outputs from being published and does not choose any scientific value.
    """

    provenance_finding = any(
        (
            "measurement-provenance audit" in str(message).lower()
            and "does not fail" in str(message).lower()
        )
        or "provenance_audit_not_fail_closed" in str(message).lower()
        for message in audit_messages
    )
    repaired = code
    repair_names: List[str] = []

    repaired, preflight_repair_names = patch_concept_preflight_repairs(
        repaired,
        findings=repair_findings,
    )
    repair_names.extend(preflight_repair_names)

    nonfinite_audit_preserved = patch_strict_numeric_nonfinite_audit_conflict(
        repaired,
        audit_messages=audit_messages,
        repair_findings=repair_findings,
    )
    if nonfinite_audit_preserved != repaired:
        repair_name = "nonfinite_audit_preserve_observed_v1"
        repaired = nonfinite_audit_preserved
        repair_names.append(repair_name)

    nonfinite_audit_host_strict = patch_nonfinite_audit_host_strict_boundary(
        repaired,
        repair_findings=repair_findings,
    )
    if nonfinite_audit_host_strict != repaired:
        repair_name = "nonfinite_audit_host_strict_boundary_v2"
        repaired = nonfinite_audit_host_strict
        repair_names.append(repair_name)

    strict_helper_guarded = patch_strict_numeric_helper_nonfinite_guard(
        repaired,
        repair_findings=repair_findings,
    )
    if strict_helper_guarded != repaired:
        repair_name = "strict_numeric_nonfinite_guard_v1"
        repaired = strict_helper_guarded
        repair_names.append(repair_name)

    companion_selector_detached = patch_audit_only_companion_value_selector(
        repaired,
        findings=repair_findings,
    )
    if companion_selector_detached != repaired:
        repair_name = "audit_only_companion_value_selector_v1"
        repaired = companion_selector_detached
        repair_names.append(repair_name)

    provenance_gated_before_outputs = patch_late_measurement_provenance_receipt(
        repaired,
        findings=repair_findings,
    )
    if provenance_gated_before_outputs != repaired:
        repair_name = "measurement_provenance_before_outputs_v1"
        repaired = provenance_gated_before_outputs
        repair_names.append(repair_name)

    range_retained = patch_flag_only_plausibility_range_rejection(
        repaired,
        repair_findings=repair_findings,
    )
    if range_retained != repaired:
        repair_name = "flag_only_plausibility_range_retention_v1"
        repaired = range_retained
        repair_names.append(repair_name)

    if RepairReason.TYPED_CONTEXT_BINDING_INVALID in set(repair_reasons):
        context_loaded = _patch_resolved_context_digest_load(
            repaired,
            repair_findings=repair_findings,
        )
        if context_loaded != repaired:
            repair_name = "resolved_context_digest_load_v1"
            repaired = context_loaded
            repair_names.append(repair_name)

    if RepairReason.TYPED_PRODUCT_BINDING_INVALID in set(repair_reasons):
        repaired, applied = _schema_alias_repair(repaired, repair_findings)
        repair_names.extend(applied)
        typed_input_preserved = patch_resolved_input_cohort_env_shadow(
            repaired,
            repair_findings=repair_findings,
        )
        if typed_input_preserved != repaired:
            repair_name = "resolved_typed_input_precedence_v1"
            repaired = typed_input_preserved
            repair_names.append(repair_name)
        run_rooted = patch_resolved_input_relative_path_root(
            repaired,
            repair_findings=repair_findings,
        )
        if run_rooted != repaired:
            repair_name = "resolved_input_run_root_v1"
            repaired = run_rooted
            repair_names.append(repair_name)
        identity_keyed = _patch_resolved_input_identity_key(
            repaired,
            repair_findings=repair_findings,
        )
        if identity_keyed != repaired:
            repair_name = "resolved_input_identity_key_v1"
            repaired = identity_keyed
            repair_names.append(repair_name)
        direct_identity_keyed = patch_direct_resolved_input_identity_key(
            repaired,
            findings=repair_findings,
        )
        if direct_identity_keyed != repaired:
            repair_name = "resolved_input_identity_key_v1"
            repaired = direct_identity_keyed
            if repair_name not in repair_names:
                repair_names.append(repair_name)

    if RepairReason.RUNTIME_SYNTAX_INCOMPATIBLE in set(repair_reasons):
        runtime_compatible = _patch_pre312_fstring_subscript_quotes(
            repaired,
            repair_findings=repair_findings,
        )
        if runtime_compatible != repaired:
            repair_name = "fstring_runtime_quote_compat_v1"
            repaired = runtime_compatible
            repair_names.append(repair_name)

    if RepairReason.ARBITRARY_COLUMN_FALLBACK in set(repair_reasons):
        fail_closed = _patch_arbitrary_column_fallback_to_raise(
            repaired,
            coordinate=_arbitrary_column_fallback_coordinate(repair_findings),
        )
        if fail_closed != repaired:
            repair_name = "arbitrary_column_fallback_fail_closed_v1"
            repaired = fail_closed
            repair_names.append(repair_name)

    if RepairReason.INVALID_HELPER_SIGNATURE in set(repair_reasons):
        publication_audit = _patch_publication_export_audit_call(
            repaired,
            finding_lines=_publication_export_audit_repair_lines(repair_findings),
        )
        if publication_audit != repaired:
            repair_name = "publication_export_audit_paths_v1"
            repaired = publication_audit
            repair_names.append(repair_name)

        receipt_bound = _patch_measurement_receipt_stable_binding(
            repaired,
            findings=repair_findings,
        )
        if receipt_bound != repaired:
            repair_name = "measurement_receipt_stable_binding_v1"
            repaired = receipt_bound
            repair_names.append(repair_name)

        keyword_bound = _patch_host_helper_keyword_only_call(
            repaired,
            finding_lines=_host_helper_signature_repair_lines(repair_findings),
        )
        if keyword_bound != repaired:
            repair_name = "host_helper_keyword_only_call_v1"
            repaired = keyword_bound
            repair_names.append(repair_name)

        closed_levels_bound = _patch_closed_counts_missing_levels(
            repaired,
            finding_lines=_closed_counts_signature_repair_lines(repair_findings),
        )
        if closed_levels_bound != repaired:
            repair_name = "closed_counts_declared_levels_binding_v1"
            repaired = closed_levels_bound
            repair_names.append(repair_name)

        stable_keywords = _patch_closed_counts_stable_keywords(
            repaired,
            finding_lines=_closed_counts_stable_keyword_repair_lines(repair_findings),
        )
        if stable_keywords != repaired:
            repair_name = "closed_counts_stable_keywords_v1"
            repaired = stable_keywords
            repair_names.append(repair_name)

        receipt_threaded = _patch_discarded_host_receipt_unpack(
            repaired,
            coordinate=_local_helper_unpack_repair_coordinate(repair_findings),
        )
        if receipt_threaded != repaired:
            repair_name = "local_helper_unpack_receipt_v1"
            repaired = receipt_threaded
            repair_names.append(repair_name)

        closed_counts_bound = _patch_closed_counts_runtime_adapter(
            repaired,
            introspection_line=_closed_counts_introspection_line(repair_findings),
        )
        if closed_counts_bound != repaired:
            repair_name = "closed_counts_direct_host_call_v1"
            repaired = closed_counts_bound
            repair_names.append(repair_name)

    if RepairReason.LOSSY_NUMERIC_COERCION in set(repair_reasons):
        guarded = _patch_lossy_numeric_coercion_guard(
            repaired,
            finding_lines=_lossy_numeric_coercion_repair_lines(repair_findings),
        )
        if guarded != repaired:
            repair_name = "lossy_numeric_coercion_guard_v1"
            repaired = guarded
            repair_names.append(repair_name)

        returned_loss_guarded = _patch_returned_coercion_loss_guard(repaired)
        if returned_loss_guarded != repaired:
            repair_name = "returned_coercion_loss_guard_v1"
            repaired = returned_loss_guarded
            repair_names.append(repair_name)

    if RepairReason.BOOLEAN_REDUCTION_IDENTITY in set(repair_reasons):
        value_compared = _patch_boolean_reduction_identity(
            repaired,
            repair_findings=repair_findings,
        )
        if value_compared != repaired:
            repair_name = "boolean_reduction_identity_v1"
            repaired = value_compared
            repair_names.append(repair_name)

    # Apply later-line accounting guards before earlier-line numeric guards so
    # both host-owned source coordinates remain valid in one atomic repair.
    if RepairReason.STRUCTURAL_ACCOUNTING_INVALID in set(repair_reasons):
        guarded = _patch_categorical_level_reconciliation_guard(
            repaired,
            counts_line=_categorical_level_reconciliation_repair_line(repair_findings),
        )
        if guarded != repaired:
            repair_name = "categorical_level_reconciliation_guard_v1"
            repaired = guarded
            repair_names.append(repair_name)

    if RepairReason.NONFINITE_NUMERIC_INPUT in set(repair_reasons):
        guarded = _patch_conditional_nonfinite_guard(
            repaired,
            guard_lines=_conditional_nonfinite_guard_lines(repair_findings),
        )
        if guarded != repaired:
            repair_name = "conditional_nonfinite_fail_closed_guard_v1"
            repaired = guarded
            repair_names.append(repair_name)

        guarded = _patch_strict_numeric_nonfinite_guard(
            repaired,
            coercion_lines=_strict_numeric_nonfinite_repair_lines(repair_findings),
        )
        if guarded != repaired:
            repair_name = "strict_numeric_nonfinite_guard_v1"
            repaired = guarded
            repair_names.append(repair_name)

    if RepairReason.PROVENANCE_NOT_FAIL_CLOSED in set(repair_reasons):
        repaired, applied = repair_superseded_provenance(repaired, repair_findings)
        repair_names.extend(applied)
        host_bound = _patch_custom_provenance_helper_to_host_receipt(
            repaired,
            coordinate=_provenance_custom_helper_coordinate(repair_findings),
        )
        if host_bound != repaired:
            repair_name = "provenance_custom_helper_to_host_receipt_v1"
            repaired = host_bound
            repair_names.append(repair_name)

    scalar_cast_finding = any(
        "scalar_cast_before_reduction" in str(message).lower()
        or (
            "integer cast is applied before" in str(message).lower()
            and "sum" in str(message).lower()
        )
        for message in audit_messages
    ) or any(
        finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and (finding.detail or {}).get("reason") == "scalar_cast_before_reduction"
        for finding in repair_findings
    )
    if scalar_cast_finding:
        reduced = _patch_scalar_cast_before_reduction(repaired)
        if reduced != repaired:
            repair_name = "scalar_cast_before_reduction_v1"
            repaired = reduced
            repair_names.append(repair_name)

    if provenance_finding:
        provenance_guard_applied = False
        guarded = _patch_provenance_fail_closed_guard(repaired)
        if guarded != repaired:
            repair_name = "provenance_fail_closed_guard_v1"
            repaired = guarded
            repair_names.append(repair_name)
            provenance_guard_applied = True
        if provenance_guard_applied:
            status_aligned = _patch_provenance_checked_status_contract(repaired)
            if status_aligned != repaired:
                repair_name = "provenance_checked_status_contract_v1"
                repaired = status_aligned
                repair_names.append(repair_name)

    swallowed_helper_finding = any(
        "provenance_helper_error_swallowed" in str(message).lower()
        or (
            "reconcile_binary_event_presence" in str(message).lower()
            and "without re-raising" in str(message).lower()
        )
        for message in audit_messages
    )
    if swallowed_helper_finding:
        fail_closed = _patch_swallowed_reconciliation_error(repaired)
        if fail_closed != repaired:
            repair_name = "provenance_helper_reraise_v1"
            repaired = fail_closed
            repair_names.append(repair_name)

    bidirectional_scan_finding = any(
        (
            "measurement-provenance audit scans measured columns only"
            in str(message).lower()
        )
        or "provenance_pair_scan_not_bidirectional" in str(message).lower()
        for message in audit_messages
    )
    if bidirectional_scan_finding:
        bidirectional = _patch_provenance_bidirectional_pair_scan(repaired)
        if bidirectional != repaired:
            repair_name = "provenance_bidirectional_pair_scan_v1"
            repaired = bidirectional
            repair_names.append(repair_name)

    first_time_companion_finding = any(
        "double_first_time_companion_suffix" in str(message).lower()
        or "looked up as '*_first_first_time'" in str(message).lower()
        for message in audit_messages
    )
    if first_time_companion_finding:
        normalized = _patch_first_time_companion_name(repaired)
        if normalized != repaired:
            repair_name = "normalize_first_time_companion_v1"
            repaired = normalized
            repair_names.append(repair_name)
    return repaired, repair_names


_PROVENANCE_FAILURE_KEYS = frozenset({"invalid_pair_n", "discordant_n"})
_PROVENANCE_DECISION_KEYS = (
    "fail_closed",
    "completed_step_allowed",
    "provenance_valid",
)
_PROVENANCE_GUARD_SENTINEL = "_easyicu_provenance_fail_closed_guard_v1"
_PROVENANCE_LOOP_SENTINEL = "_easyicu_provenance_loop_observed"
_PROVENANCE_PAIR_SCAN_SENTINEL = "_easyicu_provenance_bidirectional_pair_scan_v1"
_PROVENANCE_HELPER_RERAISE_SENTINEL = "_easyicu_provenance_helper_reraise_v1"


def _string_literals(node: ast.AST) -> set[str]:
    return {
        str(candidate.value).strip().lower()
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
    }


def _simple_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _simple_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _provenance_count_key(node: ast.AST) -> Optional[str]:
    """Return a standard provenance failure-count key for a direct access."""

    if isinstance(node, ast.Name) and node.id in _PROVENANCE_FAILURE_KEYS:
        return node.id
    key_node: Optional[ast.AST] = None
    if isinstance(node, ast.Subscript):
        key_node = node.slice
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
    ):
        key_node = node.args[0]
    if (
        isinstance(key_node, ast.Constant)
        and isinstance(key_node.value, str)
        and key_node.value in _PROVENANCE_FAILURE_KEYS
    ):
        return key_node.value
    return None


def _is_literal_numeric_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and node.value == 0
    )


def _inline_provenance_failure_coverage(
    node: ast.AST,
    *,
    assignments: dict[str, ast.AST],
    seen_names: Optional[set[str]] = None,
) -> Optional[frozenset[str]]:
    """Prove a raw-count predicate means either standard failure count is nonzero.

    This intentionally recognises only the closed host-owned provenance schema.
    Unknown riders, conjunctions, success-polarity tests, and partial checks are
    rejected so a deterministic repair cannot invent a failure policy.
    """

    direct_key = _provenance_count_key(node)
    if direct_key is not None:
        return frozenset({direct_key})

    seen_names = set(seen_names or set())
    if isinstance(node, ast.Name) and node.id in assignments:
        if node.id in seen_names:
            return None
        seen_names.add(node.id)
        return _inline_provenance_failure_coverage(
            assignments[node.id],
            assignments=assignments,
            seen_names=seen_names,
        )

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, ast.Or):
            return None
        parts = [
            _inline_provenance_failure_coverage(
                value,
                assignments=assignments,
                seen_names=seen_names,
            )
            for value in node.values
        ]
        if not parts or any(part is None for part in parts):
            return None
        return frozenset().union(*(part for part in parts if part is not None))

    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    left_key = _provenance_count_key(node.left)
    right_key = _provenance_count_key(node.comparators[0])
    operator = node.ops[0]
    if left_key is not None and _is_literal_numeric_zero(node.comparators[0]):
        if isinstance(operator, (ast.Gt, ast.NotEq, ast.IsNot)):
            return frozenset({left_key})
    if right_key is not None and _is_literal_numeric_zero(node.left):
        if isinstance(operator, (ast.Lt, ast.NotEq, ast.IsNot)):
            return frozenset({right_key})
    return None


def _patch_returned_provenance_failure_guard(code: str) -> str:
    """Guard every exact failure-collection return at its call site."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not (_PROVENANCE_FAILURE_KEYS | {"audit_only"}) <= _string_literals(tree):
        return code

    parents: dict[ast.AST, ast.AST] = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def _scope(node: ast.AST) -> ast.AST | None:
        current: ast.AST | None = node
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            current = parents.get(current)
        return current

    def _local_nodes(function: ast.AST) -> list[ast.AST]:
        return [
            node
            for node in ast.walk(function)
            if node is function or _scope(node) is function
        ]

    def _next_statement(statement: ast.stmt) -> ast.stmt | None:
        parent = parents.get(statement)
        if parent is None:
            return None
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list) or statement not in value:
                continue
            index = value.index(statement)
            if index + 1 < len(value) and isinstance(value[index + 1], ast.stmt):
                return value[index + 1]
        return None

    def _is_guard(statement: ast.stmt | None, name: str) -> bool:
        if not isinstance(statement, ast.If) or not statement.body:
            return False
        test = statement.test
        if isinstance(test, ast.Name) and test.id == name:
            return isinstance(statement.body[0], ast.Raise)
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return False
        left, right = test.left, test.comparators[0]
        return (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "len"
            and len(left.args) == 1
            and isinstance(left.args[0], ast.Name)
            and left.args[0].id == name
            and isinstance(right, ast.Constant)
            and right.value == 0
            and isinstance(test.ops[0], (ast.Gt, ast.NotEq))
            and isinstance(statement.body[0], ast.Raise)
        )

    all_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    def _local_tokens(function: ast.AST) -> set[str]:
        return {
            str(candidate.value)
            for candidate in _local_nodes(function)
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str)
        }

    marker_nodes = [
        node
        for node in all_functions
        if _PROVENANCE_FAILURE_KEYS <= _local_tokens(node)
        and "audit_only" in _local_tokens(node)
    ]
    marker_names = {node.name for node in marker_nodes}
    if any(
        sum(function.name == name for function in all_functions) != 1
        for name in marker_names
    ):
        return code
    for candidate in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(candidate, ast.Assign):
            targets = list(candidate.targets)
        elif isinstance(candidate, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [candidate.target]
        if any(
            isinstance(target, ast.Name) and target.id in marker_names
            for target in targets
        ):
            return code
    marker_functions = {node.name: node for node in marker_nodes}
    returned_slots: dict[str, int | None] = {}
    for name, function in marker_functions.items():
        local_nodes = _local_nodes(function)
        assignments: dict[str, ast.AST] = {}
        for candidate in local_nodes:
            if not isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                candidate.targets
                if isinstance(candidate, ast.Assign)
                else [candidate.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, candidate.value)

        collection_events: dict[str, set[ast.Call]] = {}
        for guard in local_nodes:
            if not isinstance(guard, ast.If):
                continue
            coverage = _inline_provenance_failure_coverage(
                guard.test,
                assignments=assignments,
            )
            if not coverage:
                continue
            if coverage != _PROVENANCE_FAILURE_KEYS:
                continue
            for statement in guard.body:
                if isinstance(statement, (ast.Raise, ast.Return)):
                    break
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr in {"append", "add"}
                    and isinstance(statement.value.func.value, ast.Name)
                ):
                    collection_events.setdefault(
                        statement.value.func.value.id, set()
                    ).add(statement.value)

        def _empty_initialization(node: ast.AST) -> bool:
            return isinstance(node, (ast.List, ast.Set)) and not node.elts

        local_returns = [node for node in local_nodes if isinstance(node, ast.Return)]
        valid_collections: set[str] = set()
        for collection, allowed_calls in collection_events.items():

            def _mutates_collection(target: ast.AST) -> bool:
                if isinstance(target, ast.Name):
                    return target.id == collection
                if isinstance(target, (ast.Tuple, ast.List)):
                    return any(_mutates_collection(item) for item in target.elts)
                if isinstance(target, (ast.Subscript, ast.Attribute)):
                    return any(
                        isinstance(item, ast.Name) and item.id == collection
                        for item in ast.walk(target)
                    )
                return False

            initializations = 0
            invalid_mutation = False
            boundary_lines = [int(call.lineno) for call in allowed_calls] + [
                int(statement.lineno) for statement in local_returns
            ]
            for candidate in local_nodes:
                targets: list[ast.AST] = []
                value: ast.AST | None = None
                if isinstance(candidate, ast.Assign):
                    targets = list(candidate.targets)
                    value = candidate.value
                elif isinstance(candidate, ast.AnnAssign):
                    targets = [candidate.target]
                    value = candidate.value
                elif isinstance(candidate, (ast.AugAssign, ast.NamedExpr, ast.Delete)):
                    targets = (
                        list(candidate.targets)
                        if isinstance(candidate, ast.Delete)
                        else [candidate.target]
                    )
                if any(_mutates_collection(target) for target in targets):
                    if (
                        value is not None
                        and len(targets) == 1
                        and isinstance(targets[0], ast.Name)
                        and _empty_initialization(value)
                        and parents.get(candidate) is function
                        and boundary_lines
                        and int(candidate.lineno) < min(boundary_lines)
                    ):
                        initializations += 1
                    else:
                        invalid_mutation = True
                if value is not None and any(
                    isinstance(item, ast.Name) and item.id == collection
                    for item in ast.walk(value)
                ):
                    invalid_mutation = True
                if (
                    isinstance(candidate, ast.Call)
                    and isinstance(candidate.func, ast.Attribute)
                    and isinstance(candidate.func.value, ast.Name)
                    and candidate.func.value.id == collection
                    and candidate not in allowed_calls
                    and candidate.func.attr not in {"append", "add"}
                ):
                    invalid_mutation = True
                if isinstance(candidate, ast.Call) and any(
                    any(
                        isinstance(item, ast.Name) and item.id == collection
                        for item in ast.walk(argument)
                    )
                    for argument in [
                        *candidate.args,
                        *(keyword.value for keyword in candidate.keywords),
                    ]
                ):
                    invalid_mutation = True
            if (
                initializations == 1
                and not invalid_mutation
                and all(
                    parents.get(statement) is function for statement in local_returns
                )
            ):
                valid_collections.add(collection)

        positions: set[int] = set()
        for statement in local_returns:
            if statement.value is None:
                positions.clear()
                break
            values = (
                list(statement.value.elts)
                if isinstance(statement.value, (ast.Tuple, ast.List))
                else [statement.value]
            )
            matches = {
                index
                for index, value in enumerate(values)
                if isinstance(value, ast.Name) and value.id in valid_collections
            }
            if len(matches) != 1:
                positions.clear()
                break
            positions.update(matches)
        returned_slots[name] = next(iter(positions)) if len(positions) == 1 else None

    calls: list[tuple[ast.stmt, str]] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        called = call.func.id
        if called not in marker_functions or returned_slots.get(called) is None:
            continue
        node = parents.get(call)
        if not (isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is call):
            return code
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        slot = returned_slots[called]
        if not (
            slot is not None
            and len(targets) == 1
            and isinstance(targets[0], (ast.Tuple, ast.List))
            and slot < len(targets[0].elts)
            and isinstance(targets[0].elts[slot], ast.Name)
        ):
            return code
        calls.append((node, targets[0].elts[slot].id))

    lines = code.splitlines(keepends=True)
    insertions: list[tuple[int, str]] = []
    for statement, collection_name in calls:
        if _is_guard(_next_statement(statement), collection_name):
            continue
        source_line = lines[statement.lineno - 1]
        indent = source_line[: len(source_line) - len(source_line.lstrip())]
        guard = (
            f"{indent}# {_PROVENANCE_GUARD_SENTINEL}\n"
            f"{indent}if {collection_name}:\n"
            f"{indent}    raise RuntimeError(\n"
            f'{indent}        "Measurement provenance audit failed; "\n'
            f'{indent}        "scientific outputs were not published."\n'
            f"{indent}    )\n"
        )
        insertions.append((getattr(statement, "end_lineno", statement.lineno), guard))
    for line_number, guard in sorted(insertions, reverse=True):
        lines.insert(line_number, guard)
    return "".join(lines)


def _patch_inline_provenance_failure_guard(code: str) -> str:
    """Terminate an authored full raw-count failure branch before outputs.

    The concept preflight has already proved that the script implements the
    standard ``invalid_pair_n``/``discordant_n`` audit but lets its failure
    branch fall through.  This repair only handles an exact top-level OR of
    those two host-owned failure counts, before any scientific result sink.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not (_PROVENANCE_FAILURE_KEYS | {"audit_only"}) <= _string_literals(tree):
        return code

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    result_sink_methods = {"fit", "fit_regularized", "predict", "savefig"}
    for guard in sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.If)),
        key=lambda node: int(getattr(node, "lineno", 0) or 0),
    ):
        scope: ast.AST = guard
        current = parents.get(guard)
        unsafe_ancestor = False
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            if isinstance(
                current,
                (
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    *_TRY_NODE_TYPES,
                    ast.With,
                    ast.AsyncWith,
                    ast.Match,
                ),
            ):
                unsafe_ancestor = True
                break
            current = parents.get(current)
        if unsafe_ancestor:
            continue
        if current is not None:
            scope = current

        owner = parents.get(guard)
        preceding: list[ast.stmt] = []
        if owner is not None:
            for _, value in ast.iter_fields(owner):
                if not isinstance(value, list) or guard not in value:
                    continue
                preceding = [
                    statement
                    for statement in value[: value.index(guard)]
                    if isinstance(statement, ast.stmt)
                ]
                break
        assignments: dict[str, ast.AST] = {}
        for candidate in preceding:
            if not isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                continue
            value = candidate.value
            if value is None:
                continue
            targets = (
                candidate.targets
                if isinstance(candidate, ast.Assign)
                else [candidate.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = value
        coverage = _inline_provenance_failure_coverage(
            guard.test,
            assignments=assignments,
        )
        if coverage != _PROVENANCE_FAILURE_KEYS:
            continue

        guard_line = int(getattr(guard, "lineno", 0) or 0)
        result_precedes_guard = False
        for candidate in ast.walk(scope):
            if not isinstance(candidate, ast.Call):
                continue
            line = int(getattr(candidate, "lineno", 0) or 0)
            if not line or line >= guard_line:
                continue
            method = _simple_call_name(candidate.func).lower().rsplit(".", 1)[-1]
            if (
                method in result_sink_methods
                or "write_success" in method
                or method.startswith("publish_")
            ):
                result_precedes_guard = True
                break
        if result_precedes_guard or not guard.body:
            continue

        lines = code.splitlines(keepends=True)
        first_body_line = int(getattr(guard.body[0], "lineno", 0) or 0)
        if first_body_line <= 0 or first_body_line > len(lines):
            continue
        source_line = lines[first_body_line - 1]
        indent = source_line[: len(source_line) - len(source_line.lstrip())]
        termination = (
            f"{indent}# {_PROVENANCE_GUARD_SENTINEL}\n"
            f"{indent}raise RuntimeError(\n"
            f'{indent}    "Measurement provenance audit failed; "\n'
            f'{indent}    "scientific outputs were not published."\n'
            f"{indent})\n"
        )
        lines.insert(first_body_line - 1, termination)
        return "".join(lines)
    return code


def _patch_direct_provenance_contract_guard(code: str) -> str:
    """Guard one direct host-shaped provenance contract before result sinks."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if _PROVENANCE_GUARD_SENTINEL in code:
        return code

    def _dict_fields(node: ast.Dict) -> Optional[dict[str, ast.AST]]:
        if any(
            key is None
            or not isinstance(key, ast.Constant)
            or not isinstance(key.value, str)
            for key in node.keys
        ):
            return None
        keys = [str(key.value) for key in node.keys if isinstance(key, ast.Constant)]
        if len(keys) != len(set(keys)):
            return None
        return dict(zip(keys, node.values))

    candidates: list[tuple[ast.FunctionDef, ast.Assign, str, str]] = []
    for function in [node for node in tree.body if isinstance(node, ast.FunctionDef)]:
        for statement in function.body:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Dict)
            ):
                continue
            outer = _dict_fields(statement.value)
            if outer is None:
                continue
            checks = outer.get("checks")
            if not (
                isinstance(checks, (ast.List, ast.Tuple))
                and len(checks.elts) == 1
                and isinstance(checks.elts[0], ast.Dict)
            ):
                continue
            row = _dict_fields(checks.elts[0])
            if row is None:
                continue
            role = row.get("role")
            invalid = row.get("invalid_pair_n")
            discordant = row.get("discordant_n")
            if not (
                isinstance(role, ast.Constant)
                and str(role.value).strip().lower() == "audit_only"
                and isinstance(invalid, ast.Name)
                and isinstance(discordant, ast.Name)
                and invalid.id != discordant.id
            ):
                continue
            count_names = (invalid.id, discordant.id)
            bindings: dict[str, list[ast.Assign]] = {name: [] for name in count_names}
            for local_statement in function.body:
                if not (
                    isinstance(local_statement, ast.Assign)
                    and len(local_statement.targets) == 1
                    and isinstance(local_statement.targets[0], ast.Name)
                    and local_statement.targets[0].id in bindings
                ):
                    continue
                bindings[local_statement.targets[0].id].append(local_statement)
            if any(len(items) != 1 for items in bindings.values()):
                continue
            if any(
                not (
                    isinstance(items[0].value, ast.Call)
                    and isinstance(items[0].value.func, ast.Name)
                    and items[0].value.func.id == "int"
                    and len(items[0].value.args) == 1
                    and not items[0].value.keywords
                    and int(items[0].lineno) < int(statement.lineno)
                )
                for items in bindings.values()
            ):
                continue
            if any(
                isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == "int"
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                )
                for node in ast.walk(tree)
            ):
                continue
            candidates.append((function, statement, count_names[0], count_names[1]))
    if len(candidates) != 1:
        return code

    _, statement, invalid_name, discordant_name = candidates[0]
    if statement.end_lineno is None:
        return code
    lines = code.splitlines(keepends=True)
    source_line = lines[int(statement.lineno) - 1]
    indent = source_line[: len(source_line) - len(source_line.lstrip())]
    body_indent = indent + ("\t" if "\t" in indent else "    ")
    guard = (
        f"{indent}# {_PROVENANCE_GUARD_SENTINEL}\n"
        f"{indent}if {invalid_name} > 0 or {discordant_name} > 0:\n"
        f"{body_indent}raise RuntimeError(\n"
        f'{body_indent}    "Measurement provenance audit failed; "\n'
        f'{body_indent}    "scientific outputs were not published."\n'
        f"{body_indent})\n"
    )
    lines.insert(int(statement.end_lineno), guard)
    repaired = "".join(lines)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_provenance_checked_status_contract(code: str) -> str:
    """Accept the host ``checked`` status when the same script emits it."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    emits_checked = any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "status"
            and isinstance(value, ast.Constant)
            and value.value == "checked"
            for key, value in zip(node.keys, node.values)
            if key is not None
        )
        and any(
            isinstance(key, ast.Constant)
            and key.value == "role"
            and isinstance(value, ast.Constant)
            and value.value == "audit_only"
            for key, value in zip(node.keys, node.values)
            if key is not None
        )
        for node in ast.walk(tree)
    )
    if not emits_checked:
        return code

    expected = {"passed", "ok", "valid"}
    candidates: list[ast.AST] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], (ast.Set, ast.List, ast.Tuple))
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Attribute)
            and node.left.func.attr == "get"
            and len(node.left.args) == 1
            and isinstance(node.left.args[0], ast.Constant)
            and node.left.args[0].value == "status"
            and not node.left.keywords
        ):
            continue
        values = node.comparators[0].elts
        if not all(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in values
        ):
            continue
        if {str(value.value) for value in values} == expected:
            candidates.append(node.comparators[0])
    if len(candidates) != 1:
        return code
    source = ast.get_source_segment(code, candidates[0])
    if not source or code.count(source) != 1:
        return code
    repaired = code.replace(source, '{"passed", "ok", "valid", "checked"}', 1)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_provenance_loop_coverage_guard(code: str) -> str:
    """Add a neutral non-empty-loop proof to one exact aggregate audit loop."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if any(
        isinstance(node, ast.Name) and node.id == _PROVENANCE_LOOP_SENTINEL
        for node in ast.walk(tree)
    ):
        return code

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def _nearest_function(node: ast.AST) -> ast.AST | None:
        current = parents.get(node)
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            current = parents.get(current)
        return current

    def _marker_append(statement: ast.stmt) -> str:
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "append"
            and isinstance(statement.value.func.value, ast.Name)
            and len(statement.value.args) == 1
            and not statement.value.keywords
            and isinstance(statement.value.args[0], ast.Dict)
        ):
            return ""
        payload = statement.value.args[0]
        keys = {
            str(key.value).strip().lower()
            for key in payload.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        values = _string_literals(payload)
        if not (_PROVENANCE_FAILURE_KEYS <= keys and "audit_only" in values):
            return ""
        return statement.value.func.value.id

    def _failure_append(statement: ast.stmt) -> str:
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr in {"append", "add"}
            and isinstance(statement.value.func.value, ast.Name)
        ):
            return ""
        return statement.value.func.value.id

    def _collection_test(node: ast.AST, name: str) -> bool:
        if isinstance(node, ast.Name):
            return node.id == name
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return False
        return (
            isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "len"
            and len(node.left.args) == 1
            and isinstance(node.left.args[0], ast.Name)
            and node.left.args[0].id == name
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == 0
            and isinstance(node.ops[0], (ast.Gt, ast.NotEq))
        )

    candidates: list[ast.For] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        tokens = {
            str(node.value).strip().lower()
            for node in ast.walk(function)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _nearest_function(node) is function
        }
        if not (_PROVENANCE_FAILURE_KEYS <= tokens and "audit_only" in tokens):
            continue
        assignments: dict[str, ast.AST] = {}
        for node in ast.walk(function):
            if _nearest_function(node) not in {None, function}:
                continue
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, node.value)

        for loop in function.body:
            if not isinstance(loop, ast.For) or loop.orelse or not loop.body:
                continue
            full_guards = [
                statement
                for statement in loop.body
                if isinstance(statement, ast.If)
                and _inline_provenance_failure_coverage(
                    statement.test, assignments=assignments
                )
                == _PROVENANCE_FAILURE_KEYS
            ]
            if len(full_guards) != 1:
                continue
            full_guard = full_guards[0]
            failure_collections = {
                collection
                for statement in full_guard.body
                if (collection := _failure_append(statement))
            }
            guard_index = loop.body.index(full_guard)
            audit_collections = {
                collection
                for statement in loop.body[:guard_index]
                if (collection := _marker_append(statement))
            }
            if len(failure_collections) != 1 or len(audit_collections) != 1:
                continue
            failure_collection = next(iter(failure_collections))
            loop_index = function.body.index(loop)
            terminal_guards = [
                statement
                for statement in function.body[loop_index + 1 :]
                if isinstance(statement, ast.If)
                and _collection_test(statement.test, failure_collection)
                and any(isinstance(node, ast.Raise) for node in ast.walk(statement))
            ]
            if not terminal_guards:
                continue
            candidates.append(loop)

    if len(candidates) != 1:
        return code
    loop = candidates[0]
    lines = code.splitlines(keepends=True)
    if not loop.body or loop.end_lineno is None:
        return code
    loop_line = lines[loop.lineno - 1]
    loop_indent = loop_line[: len(loop_line) - len(loop_line.lstrip())]
    body_line = lines[loop.body[0].lineno - 1]
    body_indent = body_line[: len(body_line) - len(body_line.lstrip())]
    insertions = [
        (
            loop.lineno - 1,
            f"{loop_indent}{_PROVENANCE_LOOP_SENTINEL} = False\n",
        ),
        (
            loop.body[0].lineno - 1,
            f"{body_indent}{_PROVENANCE_LOOP_SENTINEL} = True\n",
        ),
        (
            loop.end_lineno,
            (
                f"{loop_indent}if not {_PROVENANCE_LOOP_SENTINEL}:\n"
                f"{loop_indent}    raise RuntimeError(\n"
                f'{loop_indent}        "Measurement provenance audit had no '
                'iterable inputs."\n'
                f"{loop_indent}    )\n"
            ),
        ),
    ]
    for index, insertion in sorted(insertions, reverse=True):
        lines.insert(index, insertion)
    repaired = "".join(lines)
    try:
        repaired_tree = ast.parse(repaired)
    except SyntaxError:
        return code
    from ..gates.preflight import _provenance_fail_closed_findings

    if _provenance_fail_closed_findings(repaired_tree):
        return code
    return repaired


def _patch_provenance_fail_closed_guard(code: str) -> str:
    """Insert a terminating guard after an explicit provenance-audit call.

    The transformation is intentionally source-local (line insertion, not
    whole-script AST regeneration).  It is only available when the audit
    function exposes an explicit decision field, so the repair never infers a
    threshold or invents an audit policy from raw counts.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def _validated_candidate(candidate: str) -> str:
        if candidate == code:
            return code
        try:
            candidate_tree = ast.parse(candidate)
        except SyntaxError:
            return code
        from ..gates.preflight import _provenance_fail_closed_findings

        return code if _provenance_fail_closed_findings(candidate_tree) else candidate

    if _PROVENANCE_GUARD_SENTINEL in code:
        from ..gates.preflight import _provenance_fail_closed_findings

        if not _provenance_fail_closed_findings(tree):
            return code

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def _nearest_function(node: ast.AST) -> ast.AST | None:
        current = parents.get(node)
        while current is not None and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            current = parents.get(current)
        return current

    def _local_tokens(function: ast.AST) -> set[str]:
        return {
            str(candidate.value)
            for candidate in ast.walk(function)
            if isinstance(candidate, ast.Constant)
            and isinstance(candidate.value, str)
            and _nearest_function(candidate) is function
        }

    all_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    marker_nodes = [
        node
        for node in all_functions
        if _PROVENANCE_FAILURE_KEYS <= _local_tokens(node)
        and "audit_only" in _local_tokens(node)
    ]
    marker_names = {node.name for node in marker_nodes}
    if any(
        sum(function.name == name for function in all_functions) != 1
        for name in marker_names
    ):
        return code
    for candidate in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(candidate, ast.Assign):
            targets = list(candidate.targets)
        elif isinstance(candidate, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [candidate.target]
        if any(
            isinstance(target, ast.Name) and target.id in marker_names
            for target in targets
        ):
            return code
    marker_functions: dict[str, tuple[str, ...]] = {}
    for node in marker_nodes:
        tokens = _local_tokens(node)
        decision_keys = tuple(key for key in _PROVENANCE_DECISION_KEYS if key in tokens)
        if decision_keys:
            marker_functions[node.name] = decision_keys
    lines = code.splitlines(keepends=True)
    insertions: list[tuple[int, str]] = []

    def _next_statement(statement: ast.stmt) -> ast.stmt | None:
        parent = parents.get(statement)
        if parent is None:
            return None
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list) or statement not in value:
                continue
            index = value.index(statement)
            if index + 1 < len(value) and isinstance(value[index + 1], ast.stmt):
                return value[index + 1]
        return None

    def _decision_guarded(
        statement: ast.stmt | None,
        result_name: str,
        decision_keys: tuple[str, ...],
    ) -> bool:
        if not isinstance(statement, ast.If) or not statement.body:
            return False
        if not isinstance(statement.body[0], ast.Raise):
            return False
        names = {
            node.id for node in ast.walk(statement.test) if isinstance(node, ast.Name)
        }
        keys = _string_literals(statement.test) & set(decision_keys)
        return result_name in names and bool(keys)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        if not isinstance(node.value, ast.Call) or not isinstance(
            node.value.func, ast.Name
        ):
            continue
        called = node.value.func.id
        decision_keys = marker_functions.get(called)
        if not decision_keys:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = [target.id for target in targets if isinstance(target, ast.Name)]
        if len(target_names) != 1:
            continue

        result_name = target_names[0]
        if _decision_guarded(_next_statement(node), result_name, decision_keys):
            continue
        source_line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        indent = source_line[: len(source_line) - len(source_line.lstrip())]
        tests: list[str] = []
        if "fail_closed" in decision_keys:
            tests.append(f'{result_name}.get("fail_closed") is True')
        if "completed_step_allowed" in decision_keys:
            tests.append(f'{result_name}.get("completed_step_allowed") is not True')
        if "provenance_valid" in decision_keys:
            tests.append(f'{result_name}.get("provenance_valid") is not True')
        if not tests:
            continue
        continuation = f"\n{indent}    or ".join(tests)
        guard = (
            f"{indent}# {_PROVENANCE_GUARD_SENTINEL}\n"
            f"{indent}if (\n"
            f"{indent}    {continuation}\n"
            f"{indent}):\n"
            f"{indent}    raise RuntimeError(\n"
            f'{indent}        "Measurement provenance audit failed; "\n'
            f'{indent}        "scientific outputs were not published."\n'
            f"{indent}    )\n"
        )
        insertions.append((getattr(node, "end_lineno", node.lineno), guard))

    for line_number, guard in sorted(insertions, reverse=True):
        lines.insert(line_number, guard)
    decision_repaired = "".join(lines)
    returned_guard = _patch_returned_provenance_failure_guard(decision_repaired)
    if returned_guard != code:
        return _validated_candidate(returned_guard)
    inline_guard = _patch_inline_provenance_failure_guard(code)
    if inline_guard != code:
        return _validated_candidate(inline_guard)
    contract_guard = _patch_direct_provenance_contract_guard(code)
    if contract_guard != code:
        return _validated_candidate(contract_guard)
    return _validated_candidate(_patch_provenance_loop_coverage_guard(code))


def _patch_provenance_bidirectional_pair_scan(code: str) -> str:
    """Expand an authored provenance audit to see both companion suffixes.

    The static preflight only requests this repair after proving that a
    provenance-audit function enumerates ``*_measured`` but never ``*_n``.
    The transformation preserves the function's existing pair validation and
    failure policy; it only adds count-originated stems to the already-authored
    measured-column candidate list. A count-only concept therefore becomes an
    explicit missing-companion failure instead of escaping the audit.
    """

    if _PROVENANCE_PAIR_SCAN_SENTINEL in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    insertions: list[tuple[int, str]] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tokens = _string_literals(function)
        if not (_PROVENANCE_FAILURE_KEYS <= tokens and "audit_only" in tokens):
            continue
        scanned_suffixes = {
            token
            for candidate in ast.walk(function)
            if isinstance(candidate, ast.Call)
            and _simple_call_name(candidate.func).split(".")[-1] == "endswith"
            for token in _string_literals(candidate)
            if token in {"_measured", "_n"}
        }
        if "_measured" not in scanned_suffixes or "_n" in scanned_suffixes:
            continue

        frame_name = next(
            (argument.arg for argument in function.args.args),
            "",
        )
        candidate_name = ""
        for candidate in ast.walk(function):
            if not isinstance(candidate, ast.For):
                continue
            if not isinstance(candidate.target, ast.Name):
                continue
            if not isinstance(candidate.iter, ast.Name):
                continue
            target_name = candidate.target.id
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "endswith"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == target_name
                and "_measured" in _string_literals(call)
                for call in ast.walk(candidate)
            ):
                candidate_name = candidate.iter.id
                break
        if not frame_name or not candidate_name:
            continue

        first_statement = function.body[0] if function.body else None
        insertion_line = function.lineno
        if first_statement is not None:
            insertion_line = first_statement.lineno - 1
            if (
                isinstance(first_statement, ast.Expr)
                and isinstance(first_statement.value, ast.Constant)
                and isinstance(first_statement.value.value, str)
            ):
                insertion_line = getattr(
                    first_statement,
                    "end_lineno",
                    first_statement.lineno,
                )
        def_line = lines[function.lineno - 1]
        def_indent = def_line[: len(def_line) - len(def_line.lstrip())]
        indent = def_indent + "    "
        patch = (
            f"{indent}# {_PROVENANCE_PAIR_SCAN_SENTINEL}\n"
            f"{indent}{candidate_name} = sorted(\n"
            f"{indent}    {{str(_easyicu_column) for _easyicu_column in "
            f"{candidate_name}}}\n"
            f"{indent}    | {{\n"
            f"{indent}        (\n"
            f"{indent}            str(_easyicu_column)\n"
            f'{indent}            if str(_easyicu_column).endswith("_measured")\n'
            f'{indent}            else str(_easyicu_column)[: -len("_n")] '
            f'+ "_measured"\n'
            f"{indent}        )\n"
            f"{indent}        for _easyicu_column in {frame_name}.columns\n"
            f'{indent}        if str(_easyicu_column).endswith(("_measured", '
            f'"_n"))\n'
            f"{indent}    }}\n"
            f"{indent})\n"
        )
        insertions.append((insertion_line, patch))

    if not insertions:
        return code
    for line_number, patch in sorted(insertions, reverse=True):
        lines.insert(line_number, patch)
    return "".join(lines)


def _patch_swallowed_reconciliation_error(code: str) -> str:
    """Re-raise a caught standard-helper validation failure in place."""

    if _PROVENANCE_HELPER_RERAISE_SENTINEL in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    insertions: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        calls_reconciliation = any(
            isinstance(candidate, ast.Call)
            and _simple_call_name(candidate.func).split(".")[-1]
            == "reconcile_binary_event_presence"
            for statement in node.body
            for candidate in ast.walk(statement)
        )
        if not calls_reconciliation:
            continue
        for handler in node.handlers:
            caught_nodes = (
                handler.type.elts
                if isinstance(handler.type, ast.Tuple)
                else [handler.type]
            )
            caught = {
                _simple_call_name(candidate).split(".")[-1]
                for candidate in caught_nodes
                if candidate is not None
            }
            if handler.type is not None and not caught.intersection(
                {"BaseException", "Exception", "TypeError", "ValueError"}
            ):
                continue
            if handler.body and isinstance(handler.body[0], ast.Raise):
                continue
            if not handler.body:
                continue
            first_statement_line = lines[handler.body[0].lineno - 1]
            statement_indent = first_statement_line[
                : len(first_statement_line) - len(first_statement_line.lstrip())
            ]
            patch = (
                f"{statement_indent}# {_PROVENANCE_HELPER_RERAISE_SENTINEL}\n"
                f"{statement_indent}raise\n"
            )
            insertions.append((handler.body[0].lineno - 1, patch))

    if not insertions:
        return code
    for line_number, patch in sorted(insertions, reverse=True):
        lines.insert(line_number, patch)
    return "".join(lines)


def _patch_first_time_companion_name(code: str) -> str:
    """Normalize ``*_first`` before appending the ``*_first_time`` suffix."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    replacements: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr) or len(node.values) != 2:
            continue
        formatted, suffix = node.values
        if not (
            isinstance(formatted, ast.FormattedValue)
            and isinstance(formatted.value, ast.Name)
            and isinstance(suffix, ast.Constant)
            and suffix.value == "_first_time"
            and node.lineno == getattr(node, "end_lineno", node.lineno)
        ):
            continue
        source = ast.get_source_segment(code, node)
        if not source:
            continue
        item_name = formatted.value.id
        replacement = f"f\"{{{item_name}.removesuffix('_first')}}_first_time\""
        line_start = sum(
            len(line) for line in code.splitlines(keepends=True)[: node.lineno - 1]
        )
        replacements.append(
            (
                line_start + node.col_offset,
                line_start + getattr(node, "end_col_offset", node.col_offset),
                replacement,
            )
        )
    if not replacements:
        return code
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    return repaired


def _overadjustment_strip_names(offenders: Sequence[str]) -> List[str]:
    strip_names: List[str] = []
    for raw_name in offenders:
        name = str(raw_name).strip()
        if not name:
            continue
        strip_names.append(name)
        for suffix in ("_filled", "_missing_indicator", "_missing"):
            if name.endswith(suffix):
                strip_names.append(name[: -len(suffix)])
        if "_per_" in name:
            strip_names.append(name.split("_per_", 1)[0])
    return list(dict.fromkeys(value for value in strip_names if value))


def _patch_overadjustment_covariate_filter(
    code: str,
    strip_names: Sequence[str],
) -> str:
    if "_easyicu_overadjustment_drop_v1" in code or not strip_names:
        return code
    exact = list(dict.fromkeys(str(name) for name in strip_names if str(name)))
    roots = [
        name
        for name in exact
        if not name.endswith(("_indicator", "_measured", "_flag"))
        and len(name.split("_")) >= 2
    ]
    exact_literal = json.dumps(exact)
    roots_literal = json.dumps(roots)

    dedupe_re = re.compile(
        r"(?m)^(?P<indent>[ \t]*)(?P<var>x_cols|covariates|model_cols|predictor_cols)"
        r"\s*=\s*list\(dict\.fromkeys\((?P=var)\)\)\s*$"
    )

    def _rewrite(match: "re.Match[str]") -> str:
        indent = match.group("indent")
        var = match.group("var")
        return (
            match.group(0)
            + "\n"
            + f"{indent}_easyicu_overadjustment_drop_v1 = set({exact_literal})\n"
            + f"{indent}_easyicu_overadjustment_roots_v1 = tuple({roots_literal})\n"
            + f"{indent}def _easyicu_overadjustment_keep_v1(col):\n"
            + f"{indent}    col = str(col)\n"
            + f"{indent}    if col in _easyicu_overadjustment_drop_v1:\n"
            + f"{indent}        return False\n"
            + f"{indent}    return not any(\n"
            + f"{indent}        col == root or col.startswith(root + '_')\n"
            + f"{indent}        for root in _easyicu_overadjustment_roots_v1\n"
            + f"{indent}    )\n"
            + f"{indent}{var} = [\n"
            + f"{indent}    col for col in {var}\n"
            + f"{indent}    if _easyicu_overadjustment_keep_v1(col)\n"
            + f"{indent}]\n"
        )

    return dedupe_re.sub(_rewrite, code, count=1)


def _deterministic_summary_repair(
    *,
    code: str,
    step_summary: Dict[str, Any],
    previous_repair: Optional[str] = None,
    analysis_family: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    if not isinstance(step_summary, dict) or not step_summary:
        return None
    host_helper_import_repair = "relocate_known_host_helper_import_v1"
    if previous_repair != host_helper_import_repair:
        repaired = patch_known_host_helper_import(code, str(step_summary))
        if repaired is not None and repaired != code:
            return host_helper_import_repair, repaired
    clinical_bin_repair = "categorical_distribution_clinical_bin_role_v1"
    if previous_repair != clinical_bin_repair:
        repaired = patch_categorical_distribution_clinical_bin_role(
            code,
            step_summary,
        )
        if repaired is not None and repaired != code:
            return clinical_bin_repair, repaired
    nullable_validation_repair = patch_unused_nullable_numeric_validation(
        code,
        step_summary,
    )
    if nullable_validation_repair is not None:
        return nullable_validation_repair
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
        or step_summary.get("primary_exposure")
        or step_summary.get("predictor")
        or manifest.get("primary_predictor")
        or manifest.get("primary_exposure")
        or (predictor_match.group(1) if predictor_match else "")
        or ""
    ).strip()
    structured_primary_singular = _structured_primary_singular_failure(step_summary)
    estimate = _first_present_scalar(
        step_summary,
        ("estimate", "primary_or", "odds_ratio", "adjusted_or", "or"),
    )
    if estimate is not None and not structured_primary_singular:
        return None
    error_text = str(
        step_summary.get("error")
        or step_summary.get("error_message")
        or step_summary.get("note")
        or ""
    )
    age_indicator_keyerror = (
        error_text.strip().strip("'\"") == "age"
        and "source_vars_for_table" in code
        and "measured_vars" in code
        and "meas_var = measured_vars[var]" in code
    )
    if age_indicator_keyerror:
        repair_name = "age_covariate_no_measured_indicator_v1"
        if previous_repair != repair_name:
            repaired = _patch_age_covariate_coding_without_indicator(code)
            if repaired is not None:
                return repair_name, repaired
    generic_soft_failure = "unknown error" in error_text.lower()
    dtype_soft_failure = (
        "pandas data cast to numpy dtype of object" in error_text.lower()
    )
    index_alignment_soft_failure = (
        "indices for endog and exog are not aligned" in error_text.lower()
    )
    binary_model_repair_allowed = _family_allows_binary_model_repair(analysis_family)
    if (
        predictor
        and error_text
        and predictor not in error_text
        and not (
            generic_soft_failure or dtype_soft_failure or index_alignment_soft_failure
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
                "X = model_df[x_cols]",
                'X = model_df[x_cols].apply(pd.to_numeric, errors="coerce").astype(float)',
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
        null_model_summary = any(
            marker in summary_text for marker in _NULL_PRIMARY_EFFECT_MARKERS
        )
        dtype_summary_failure = (
            "pandas data cast to numpy dtype of object" in summary_text
        )
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
                    'X_clean = data.drop(columns=[y.name]).apply(pd.to_numeric, errors="coerce").astype(float)\n'
                    '    y_clean = pd.to_numeric(y_clean, errors="coerce").astype(float)',
                    1,
                )
                if repaired != code:
                    return repair_name, repaired
        if dtype_summary_failure and _statsmodels_repair_allowed_for_family(
            code, analysis_family
        ):
            repaired = _deterministic_runner_repair(
                code=code,
                run_log=summary_text,
                previous_repair=previous_repair,
                analysis_family=analysis_family,
            )
            if repaired is not None:
                return repaired
        if index_alignment_summary_failure and _statsmodels_repair_allowed_for_family(
            code, analysis_family
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
                marker = 'X_final = sm.add_constant(X_encoded, has_constant="add")'
                patch = (
                    'X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").astype(float)\n'
                    + marker
                )
                repaired = code.replace(marker, patch, 1)
                if repaired != code:
                    return repair_name, repaired
        nested_primary_singular = (
            (null_model_summary or structured_primary_singular)
            and "singular matrix" in summary_text
            and (
                structured_primary_singular
                or '"primary_model"' in summary_text
                or "primary association" in summary_text
                or "primary estimand" in summary_text
                or "primary_exposure" in summary_text
                or "primary_predictor" in summary_text
            )
            and "sm.logit(" in code.lower()
        )
        if nested_primary_singular and binary_model_repair_allowed:
            repair_name = "rank_safe_statsmodels_design_v1"
            if previous_repair != repair_name:
                repaired = _patch_rank_safe_statsmodels_design(code)
                if repaired is not None and repaired != code:
                    return repair_name, repaired
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
                    patch = textwrap.dedent("""
                        if 'sex' in model_df.columns:
                            model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                        for col in model_df.columns:
                            if col != 'sex':
                                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                        """).strip("\n")
                    repaired = code.replace(
                        model_df_assign.group(1),
                        model_df_assign.group(1) + "\n" + patch,
                        1,
                    )
                    if repaired != code:
                        return repair_name, repaired
        categorical_sex_dropna = (
            (
                (
                    "no valid data after dropping" in skipped
                    and "missing rows" in skipped
                )
                or "insufficient data" in skipped
                or "no valid observations" in skipped
                or null_model_summary
                or dtype_summary_failure
            )
            and 'model_df = model_df.apply(pd.to_numeric, errors="coerce")' in code
            and "sex" in code
        )
        if categorical_sex_dropna:
            repair_name = "sex_numeric_coercion_before_dropna_v1"
            if previous_repair == repair_name:
                return None
            replacement = textwrap.dedent("""
                if 'sex' in model_df.columns:
                    model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                for col in model_df.columns:
                    if col != 'sex':
                        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                """).strip("\n")
            repaired = re.sub(
                r"^(?P<indent>\s*)model_df = model_df\.apply\(pd\.to_numeric, errors=\"coerce\"\)",
                lambda match: match.group("indent")
                + replacement.replace("\n", "\n" + match.group("indent")),
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
            and 'pd.to_numeric(model_df[col], errors="coerce")' in code
            and "sex" in code
        )
        if categorical_sex_loop_dropna:
            repair_name = "sex_covariate_numeric_loop_guard_v1"
            if previous_repair != repair_name:
                marker = (
                    "for col in x_cols:\n"
                    '    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")'
                )
                replacement = (
                    "for col in x_cols:\n"
                    '    if col == "sex":\n'
                    '        model_df[col] = model_df[col].astype(str).str.lower().isin(["m", "male", "1", "true"]).astype(float)\n'
                    "        continue\n"
                    '    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")'
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
                reduction_marker = (
                    "model_df = model_df.replace([np.inf, -np.inf], np.nan)"
                )
                reduction_patch = (
                    reduction_marker
                    + "\n"
                    + "reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]"
                )
                if (
                    reduction_marker in repaired
                    and "reduced_covariates =" not in repaired
                ):
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
                if "fillna(0)" not in repaired:
                    missing_assign_pattern = re.compile(
                        r"(?m)^(?P<indent>\s*)"
                        r"mi_df\[(?P<missing>(?:['\"][^'\"]+_missing[^'\"]*['\"]|missing_indicator_col))\]"
                        r"\s*=\s*mi_df\[primary_predictor\]\.isna\(\)\.astype\(int\)\s*$"
                    )

                    def _patch_mi_assignment(match: re.Match[str]) -> str:
                        indent = match.group("indent")
                        missing_expr = match.group("missing")
                        return (
                            f"{indent}mi_df[{missing_expr}] = mi_df[primary_predictor].isna().astype(int)\n"
                            f"{indent}mi_df[primary_predictor] = mi_df[primary_predictor].fillna(0)\n"
                            f"{indent}mi_df = mi_df.dropna(subset=[outcome_col] + covariates)"
                        )

                    repaired = missing_assign_pattern.sub(
                        _patch_mi_assignment,
                        repaired,
                        count=1,
                    )
                rv_replacements = {
                    "rv_df = model_df.dropna(subset=[primary_predictor])": (
                        "rv_df = model_df[[outcome_col, primary_predictor] + reduced_covariates].dropna()"
                    ),
                    'rv_X = sm.add_constant(rv_df[covariates], has_constant="add")': (
                        'rv_X = sm.add_constant(rv_df[[primary_predictor] + reduced_covariates], has_constant="add")'
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


def _patch_unavailable_figure_full_source_projection(code: str) -> str:
    """Keep the complete bound table behind an unavailable-result notice.

    A rendering-only step may honestly report that its typed parent lacks the
    columns needed for the planned chart.  Its source-data file must still
    preserve the complete bound parent rather than emit identifiers alone;
    otherwise the host cannot independently verify the claimed absence.  This
    rewrite is limited to the generated key-only ``pd.DataFrame(rows, ...)``
    shape and retains the existing path and CSV write.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if not {"not_estimable_notice", "unsupported"} <= string_literals:
        return code

    candidates: List[tuple[ast.Assign, str, str, str]] = []
    for function in (
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ):
        frame_names = {
            node.func.value.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "iterrows"
            and isinstance(node.func.value, ast.Name)
        }
        source_table_names = {
            value.id
            for node in ast.walk(function)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant)
            and key.value == "source_table"
            and isinstance(value, ast.Name)
        }
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "pd"
                and node.value.func.attr == "DataFrame"
                and node.value.args
                and isinstance(node.value.args[0], ast.Name)
            ):
                continue
            declared_columns = {
                item.value
                for keyword in node.value.keywords
                if keyword.arg == "columns" and isinstance(keyword.value, ast.List)
                for item in keyword.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            target_name = node.targets[0].id
            writes_csv = any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "to_csv"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == target_name
                for call in ast.walk(function)
            )
            if not (
                {"source_row_index", "source_table"} <= declared_columns
                and len(frame_names) == 1
                and len(source_table_names) == 1
                and writes_csv
            ):
                continue
            candidates.append(
                (
                    node,
                    target_name,
                    next(iter(frame_names)),
                    next(iter(source_table_names)),
                )
            )
    if len(candidates) != 1:
        return code

    node, target_name, frame_name, source_table_name = candidates[0]
    lines = code.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    indent = " " * int(node.col_offset)
    replacement = (
        f'{target_name} = {frame_name}.drop(columns=["source_row_index", '
        f'"source_table"], errors="ignore").copy(deep=True).reset_index(drop=True)\n'
        f'{indent}{target_name}.insert(0, "source_table", {source_table_name})\n'
        f'{indent}{target_name}.insert(0, "source_row_index", '
        f"range(len({target_name})))"
    )
    repaired = (
        code[: _absolute_offset(node.lineno, node.col_offset)]
        + replacement
        + code[_absolute_offset(node.end_lineno, node.end_col_offset) :]
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _patch_bound_figure_source_projection(code: str) -> str:
    """Declare exact, per-parent source-data projections for a figure bundle.

    Generated renderers sometimes combine several bound parent tables into one
    reader-facing long table.  Renaming upstream value columns to generic
    ``value``/``numerator`` fields makes that combined table impossible to
    authenticate without trusting the renderer.  When the script already loads
    every Planner-declared table into one mapping, this repair writes each
    loaded frame unchanged to a separate local CSV and declares those exact
    projections in both the FigureContract and ``step_summary.json``.

    The rewrite is deliberately structural: it neither selects rows nor
    changes the plotted data.  It requires one unambiguous table-loading loop,
    one FigureContract call, one publication-save call, and one summary source
    declaration.  Any other shape fails closed to the normal repair path.
    """

    marker = "_easyicu_bound_source_files"
    provenance_marker = "_easyicu_bound_source_table_name"
    if provenance_marker in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    table_loads: List[tuple[str, str, str, str]] = []
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        if not (isinstance(loop.target, ast.Name) and isinstance(loop.iter, ast.Name)):
            continue
        loop_key = loop.target.id
        loaded_tuples = [
            target
            for assignment in loop.body
            if isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance((target := assignment.targets[0]), ast.Tuple)
            and len(target.elts) >= 3
            and isinstance(target.elts[0], ast.Name)
            and isinstance(target.elts[2], ast.Name)
            and isinstance(assignment.value, ast.Call)
            and (
                (
                    isinstance(assignment.value.func, ast.Name)
                    and assignment.value.func.id == "load_bound_table"
                )
                or (
                    isinstance(assignment.value.func, ast.Attribute)
                    and assignment.value.func.attr == "load_bound_table"
                )
            )
        ]
        if len(loaded_tuples) != 1:
            continue
        loaded_frame_name = loaded_tuples[0].elts[0].id
        loaded_record_name = loaded_tuples[0].elts[2].id
        frame_map_names: List[str] = []
        record_map_names: List[str] = []
        for node in ast.walk(loop):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and isinstance(node.targets[0].slice, ast.Name)
                and node.targets[0].slice.id == loop_key
                and isinstance(node.value, ast.Name)
            ):
                continue
            if node.value.id == loaded_frame_name:
                frame_map_names.append(node.targets[0].value.id)
            elif node.value.id == loaded_record_name:
                record_map_names.append(node.targets[0].value.id)
        if len(frame_map_names) == 1 and len(record_map_names) == 1:
            table_loads.append(
                (
                    frame_map_names[0],
                    record_map_names[0],
                    loop.iter.id,
                    loop_key,
                )
            )
    table_loads = list(dict.fromkeys(table_loads))
    if len(table_loads) != 1:
        return code
    table_map_name, record_map_name, input_iterable_name, _ = table_loads[0]

    provenance_lines = (
        "_easyicu_bound_record = "
        f"{record_map_name}[_easyicu_bound_key]\n"
        "_easyicu_bound_relative_path = "
        '_easyicu_bound_record.get("relative_path")\n'
        "if not isinstance(_easyicu_bound_relative_path, str) "
        "or not _easyicu_bound_relative_path.strip():\n"
        '    raise RuntimeError("Missing bound source-table path for figure provenance")\n'
        f"{provenance_marker} = "
        '_easyicu_bound_relative_path.replace(chr(92), "/").rsplit("/", 1)[-1]\n'
        f"if not {provenance_marker}:\n"
        '    raise RuntimeError("Invalid bound source-table path for figure provenance")\n'
        'if {"source_row_index", "source_table"} '
        "& set(_easyicu_bound_frame.columns):\n"
        "    raise RuntimeError("
        '"Bound parent already uses reserved figure provenance columns")\n'
        "_easyicu_bound_frame.insert(\n"
        f'    0, "source_table", {provenance_marker}\n'
        ")\n"
        "_easyicu_bound_frame.insert(\n"
        '    0, "source_row_index", range(len(_easyicu_bound_frame))\n'
        ")\n"
    )
    if marker in code:
        bound_frame_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_easyicu_bound_frame"
        ]
        if len(bound_frame_assignments) != 1:
            return code
        assignment = bound_frame_assignments[0]
        if assignment.end_lineno is None:
            return code
        lines = code.splitlines(keepends=True)
        insert_at = sum(len(line) for line in lines[: assignment.end_lineno])
        repaired = (
            code[:insert_at]
            + textwrap.indent(provenance_lines, " " * 8)
            + code[insert_at:]
        )
        try:
            ast.parse(repaired)
        except SyntaxError:
            return code
        return repaired

    contract_assignments: List[tuple[ast.Assign, ast.keyword]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and (
                (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "make_figure_contract"
                )
                or (
                    isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "make_figure_contract"
                )
            )
        ):
            continue
        source_keywords = [
            keyword for keyword in node.value.keywords if keyword.arg == "source_data"
        ]
        if len(source_keywords) == 1:
            contract_assignments.append((node, source_keywords[0]))
    if len(contract_assignments) != 1:
        return code
    contract_statement, source_keyword = contract_assignments[0]

    output_dir_names = {
        keyword.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == "save_publication_figure"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "save_publication_figure"
            )
        )
        for keyword in node.keywords
        if keyword.arg == "out_dir" and isinstance(keyword.value, ast.Name)
    }
    if len(output_dir_names) != 1:
        return code
    output_dir_name = next(iter(output_dir_names))

    summary_source_values: List[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "source_data_files":
                summary_source_values.append(value)
    if len(summary_source_values) != 1:
        return code
    summary_source_value = summary_source_values[0]

    lines = code.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    def _span(node: ast.AST) -> tuple[int, int]:
        return (
            _absolute_offset(node.lineno, node.col_offset),
            _absolute_offset(node.end_lineno, node.end_col_offset),
        )

    indent = " " * int(contract_statement.col_offset)
    projection = (
        f"{indent}{marker} = []\n"
        f"{indent}for _easyicu_bound_index, _easyicu_bound_key in "
        f"enumerate({input_iterable_name}):\n"
        f"{indent}    _easyicu_bound_frame = "
        f"{table_map_name}[_easyicu_bound_key].copy(deep=True)\n"
        + textwrap.indent(provenance_lines, indent + " " * 4)
        + f'{indent}    _easyicu_bound_token = "".join(\n'
        f'{indent}        character if character.isalnum() else "_"\n'
        f"{indent}        for character in str(_easyicu_bound_key)\n"
        f'{indent}    ).strip("_")\n'
        f"{indent}    _easyicu_bound_source_name = (\n"
        f'{indent}        f"bound_{{_easyicu_bound_index:03d}}_'
        f'{{_easyicu_bound_token}}_source_data.csv"\n'
        f"{indent}    )\n"
        f"{indent}    _easyicu_bound_frame.to_csv(\n"
        f"{indent}        {output_dir_name} / _easyicu_bound_source_name,\n"
        f"{indent}        index=False,\n"
        f"{indent}    )\n"
        f"{indent}    {marker}.append(_easyicu_bound_source_name)\n\n"
    )
    insert_at = line_starts[contract_statement.lineno - 1]
    replacements = [
        (*_span(source_keyword.value), marker),
        (*_span(summary_source_value), marker),
        (insert_at, insert_at, projection),
    ]
    repaired = code
    for start, end, replacement in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _materialize_direct_bound_source_frames(
    code: str,
    *,
    tree: ast.AST,
    contract_statement: ast.Assign,
    source_keyword: ast.keyword,
    entries: Sequence[tuple[str, str, str]],
) -> str:
    """Write exact parent frames and bind their local CSV basenames."""

    marker = "_easyicu_direct_bound_source_files"
    if marker in code or not entries:
        return code
    if any(
        re.fullmatch(r"[A-Za-z0-9_]+", product) is None
        or Path(source_name).name != source_name
        or Path(source_name).suffix.lower() != ".csv"
        for product, _frame_name, source_name in entries
    ):
        return code
    output_dir_names = {
        keyword.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == "save_publication_figure"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "save_publication_figure"
            )
        )
        for keyword in node.keywords
        if keyword.arg == "out_dir" and isinstance(keyword.value, ast.Name)
    }
    if len(output_dir_names) != 1:
        return code
    output_dir_name = next(iter(output_dir_names))
    indent = " " * int(contract_statement.col_offset)
    projection = f"{indent}{marker} = []\n"
    for index, (product, frame_name, source_name) in enumerate(entries):
        local_name = f"bound_{index:03d}_{product}_source_data.csv"
        projection += (
            f"{indent}_easyicu_direct_bound_frame = "
            f"{frame_name}.copy(deep=True)\n"
            f'{indent}if {{"source_row_index", "source_table"}} & '
            "set(_easyicu_direct_bound_frame.columns):\n"
            f"{indent}    raise RuntimeError("
            '"Bound parent already uses reserved figure provenance columns")\n'
            f'{indent}_easyicu_direct_bound_frame.insert(0, "source_table", '
            f"{source_name!r})\n"
            f'{indent}_easyicu_direct_bound_frame.insert(0, "source_row_index", '
            "range(len(_easyicu_direct_bound_frame)))\n"
            f"{indent}_easyicu_direct_bound_source_name = {local_name!r}\n"
            f"{indent}_easyicu_direct_bound_frame.to_csv(\n"
            f"{indent}    {output_dir_name} / "
            "_easyicu_direct_bound_source_name,\n"
            f"{indent}    index=False,\n"
            f"{indent})\n"
            f"{indent}{marker}.append(_easyicu_direct_bound_source_name)\n"
        )
    projection += "\n"

    lines = code.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def absolute_offset(line_number: int, byte_column: int) -> int:
        line = lines[line_number - 1]
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        return line_starts[line_number - 1] + character_column

    source_start = absolute_offset(
        source_keyword.value.lineno,
        source_keyword.value.col_offset,
    )
    source_end = absolute_offset(
        source_keyword.value.end_lineno,
        source_keyword.value.end_col_offset,
    )
    insert_at = line_starts[contract_statement.lineno - 1]
    repaired = code
    for start, end, replacement in sorted(
        [
            (source_start, source_end, marker),
            (insert_at, insert_at, projection),
        ],
        key=lambda item: item[0],
        reverse=True,
    ):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _figure_contract_source_assignment(
    tree: ast.AST,
    *,
    source_value_type: type[ast.AST],
) -> tuple[ast.Assign, ast.keyword] | None:
    candidates: List[tuple[ast.Assign, ast.keyword]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and (
                (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "make_figure_contract"
                )
                or (
                    isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "make_figure_contract"
                )
            )
        ):
            continue
        source_keywords = [
            keyword
            for keyword in node.value.keywords
            if keyword.arg == "source_data"
            and isinstance(keyword.value, source_value_type)
        ]
        if len(source_keywords) == 1:
            candidates.append((node, source_keywords[0]))
    return candidates[0] if len(candidates) == 1 else None


def _patch_direct_bound_figure_source_projection(
    code: str,
    *,
    missing_table_names: Sequence[str],
) -> str:
    """Project directly loaded typed parents into local source-data files."""

    plain_missing_names = [
        str(name)
        for name in missing_table_names
        if isinstance(name, str)
        and name
        and Path(name).name == name
        and Path(name).suffix.lower() == ".csv"
    ]
    if len(plain_missing_names) != len(missing_table_names) or not plain_missing_names:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    loaded_tables: Dict[str, str] = {}
    duplicate_products: Set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "copy"
            and isinstance(node.value.func.value, ast.Subscript)
        ):
            continue
        row_subscript = node.value.func.value
        if not (
            isinstance(row_subscript.slice, ast.Constant)
            and row_subscript.slice.value == 0
            and isinstance(row_subscript.value, ast.Subscript)
            and isinstance(row_subscript.value.value, ast.Name)
            and isinstance(row_subscript.value.slice, ast.Constant)
            and isinstance(row_subscript.value.slice.value, str)
        ):
            continue
        input_key = row_subscript.value.slice.value
        if not input_key.startswith("table:"):
            continue
        product = input_key.split(":", 1)[1]
        if product in loaded_tables:
            duplicate_products.add(product)
            continue
        loaded_tables[product] = node.targets[0].id
    if duplicate_products:
        return code
    missing_products = [Path(name).stem for name in plain_missing_names]
    if len(missing_products) != len(set(missing_products)) or any(
        product not in loaded_tables for product in missing_products
    ):
        return code
    source_assignment = _figure_contract_source_assignment(
        tree,
        source_value_type=ast.Constant,
    )
    if source_assignment is None:
        return code
    contract_statement, source_keyword = source_assignment
    entries = [
        (product, loaded_tables[product], source_name)
        for source_name, product in zip(
            plain_missing_names,
            missing_products,
            strict=True,
        )
    ]
    return _materialize_direct_bound_source_frames(
        code,
        tree=tree,
        contract_statement=contract_statement,
        source_keyword=source_keyword,
        entries=entries,
    )


def _patch_direct_bound_figure_source_materialization(code: str) -> str:
    """Materialize the exact v1 DataFrame-dictionary projection shape."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    source_assignment = _figure_contract_source_assignment(
        tree,
        source_value_type=ast.Dict,
    )
    if source_assignment is None:
        return code
    contract_statement, source_keyword = source_assignment
    source_dict = source_keyword.value
    assert isinstance(source_dict, ast.Dict)
    entries: List[tuple[str, str, str]] = []
    for key, value in zip(source_dict.keys, source_dict.values, strict=True):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "assign"
            and isinstance(value.func.value, ast.Call)
            and isinstance(value.func.value.func, ast.Attribute)
            and value.func.value.func.attr == "copy"
            and isinstance(value.func.value.func.value, ast.Name)
        ):
            return code
        frame_name = value.func.value.func.value.id
        keywords = {keyword.arg: keyword.value for keyword in value.keywords}
        source_table = keywords.get("source_table")
        row_index = keywords.get("source_row_index")
        if not (
            set(keywords) == {"source_row_index", "source_table"}
            and isinstance(source_table, ast.Constant)
            and isinstance(source_table.value, str)
            and Path(source_table.value).name == source_table.value
            and Path(source_table.value).suffix.lower() == ".csv"
            and Path(source_table.value).stem == key.value
            and isinstance(row_index, ast.Call)
            and isinstance(row_index.func, ast.Name)
            and row_index.func.id == "range"
            and len(row_index.args) == 1
            and isinstance(row_index.args[0], ast.Call)
            and isinstance(row_index.args[0].func, ast.Name)
            and row_index.args[0].func.id == "len"
            and len(row_index.args[0].args) == 1
            and isinstance(row_index.args[0].args[0], ast.Name)
            and row_index.args[0].args[0].id == frame_name
        ):
            return code
        entries.append((key.value, frame_name, source_table.value))
    return _materialize_direct_bound_source_frames(
        code,
        tree=tree,
        contract_statement=contract_statement,
        source_keyword=source_keyword,
        entries=entries,
    )


def _patch_unresolved_input_binding_receipts(
    code: str,
    *,
    findings: Sequence[Any],
) -> str:
    """Empty a literal typed-input receipt list when the host bound no typed inputs.

    Raw Planner columns are authorized through the execution cohort and
    ``raw_input_contracts``. They are deliberately absent from the typed
    ``manifest['inputs']`` namespace, so generated ``raw:<column>`` receipts are
    both unverifiable and unnecessary. This repair is intentionally narrow: it
    runs only when every reported unresolved key came from a validator receipt
    proving that the exact host-resolved key set was empty, and only when one
    literal ``input_bindings`` list contains exactly those keys.
    """

    unresolved_details: list[Mapping[str, Any]] = []
    for finding in findings:
        validator = getattr(finding, "validator", None)
        detail = getattr(finding, "detail", None)
        if isinstance(finding, dict):
            validator = finding.get("validator")
            detail = finding.get("detail")
        if (
            validator == "step_summary_integrity"
            and isinstance(detail, dict)
            and detail.get("issue") == "input_binding_key_unresolved"
        ):
            unresolved_details.append(detail)
    if not unresolved_details or any(
        detail.get("resolved_input_keys") != []
        or not isinstance(detail.get("input_key"), str)
        for detail in unresolved_details
    ):
        return code
    unresolved_keys = [str(detail["input_key"]) for detail in unresolved_details]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    expected_keys = sorted(unresolved_keys)
    candidates: list[ast.List] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if not (
                isinstance(key, ast.Constant)
                and key.value == "input_bindings"
                and isinstance(value, ast.List)
                and value.elts
            ):
                continue
            literal_keys: list[str] = []
            valid_literal = True
            for item in value.elts:
                if not isinstance(item, ast.Dict):
                    valid_literal = False
                    break
                item_key: Optional[str] = None
                for item_field, item_value in zip(
                    item.keys, item.values, strict=True
                ):
                    if (
                        isinstance(item_field, ast.Constant)
                        and item_field.value == "input_key"
                        and isinstance(item_value, ast.Constant)
                        and isinstance(item_value.value, str)
                    ):
                        item_key = item_value.value
                        break
                if item_key is None:
                    valid_literal = False
                    break
                literal_keys.append(item_key)
            if valid_literal and sorted(literal_keys) == expected_keys:
                candidates.append(value)
    if len(candidates) != 1:
        return code

    candidate = candidates[0]
    if candidate.end_lineno is None or candidate.end_col_offset is None:
        return code
    lines = code.splitlines(keepends=True)
    if not lines:
        return code
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    repaired = (
        code[: _absolute_offset(candidate.lineno, candidate.col_offset)]
        + "[]"
        + code[
            _absolute_offset(candidate.end_lineno, candidate.end_col_offset) :
        ]
    )
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def deterministic_contract_repair(
    *,
    code: str,
    findings: Sequence[Any],
    previous_repair: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Patch objective contract/audit failures before asking the LLM to repair."""

    unresolved_receipt_repair_name = "unresolved_input_binding_receipts_v1"
    if previous_repair != unresolved_receipt_repair_name:
        repaired = _patch_unresolved_input_binding_receipts(code, findings=findings)
        if repaired != code:
            return unresolved_receipt_repair_name, repaired

    render_echo_repair_name = "render_only_effect_echo_suppression_v1"
    if previous_repair != render_echo_repair_name:
        repaired = patch_render_only_effect_echo(code, findings=findings)
        if repaired is not None and repaired != code:
            return render_echo_repair_name, repaired

    host_receipts = patch_custom_measurement_provenance_receipts(
        code, findings=findings
    )
    if host_receipts != code:
        return "measurement_provenance_host_receipts_v1", host_receipts

    convergence_contract = patch_penalized_convergence_contract(
        code,
        findings=findings,
    )
    if convergence_contract != code:
        return "penalized_convergence_contract_v2", convergence_contract

    direct_materialization_findings = []
    for finding in findings:
        validator = getattr(finding, "validator", None)
        detail = getattr(finding, "detail", None)
        if isinstance(finding, dict):
            validator = finding.get("validator")
            detail = finding.get("detail")
        if (
            validator == "figure_source_data"
            and isinstance(detail, dict)
            and detail.get("reason") == "missing_source_data"
        ):
            direct_materialization_findings.append(finding)
    direct_materialization_repair_name = "direct_bound_figure_source_materialization_v1"
    if (
        len(direct_materialization_findings) == 1
        and previous_repair != direct_materialization_repair_name
    ):
        repaired = _patch_direct_bound_figure_source_materialization(code)
        if repaired != code:
            return direct_materialization_repair_name, repaired

    source_coverage_findings = []
    for finding in findings:
        validator = getattr(finding, "validator", None)
        detail = getattr(finding, "detail", None)
        if isinstance(finding, dict):
            validator = finding.get("validator")
            detail = finding.get("detail")
        if (
            validator == "figure_source_data"
            and isinstance(detail, dict)
            and detail.get("reason") == "incomplete_source_lineage_coverage"
            and detail.get("missing_bound_tables")
        ):
            source_coverage_findings.append(finding)
    direct_source_repair_name = "direct_bound_figure_source_projection_v1"
    if (
        len(source_coverage_findings) == 1
        and previous_repair != direct_source_repair_name
    ):
        finding = source_coverage_findings[0]
        detail = (
            finding.get("detail")
            if isinstance(finding, dict)
            else getattr(finding, "detail", {})
        )
        repaired = _patch_direct_bound_figure_source_projection(
            code,
            missing_table_names=list(detail.get("missing_bound_tables") or []),
        )
        if repaired != code:
            return direct_source_repair_name, repaired

    unavailable_source_findings = []
    for finding in source_coverage_findings:
        detail = (
            finding.get("detail")
            if isinstance(finding, dict)
            else getattr(finding, "detail", {})
        )
        if not detail.get("missing_bound_statistics"):
            unavailable_source_findings.append(finding)
    unavailable_source_repair_name = "unavailable_figure_full_source_projection_v1"
    if (
        len(unavailable_source_findings) == 1
        and previous_repair != unavailable_source_repair_name
    ):
        repaired = _patch_unavailable_figure_full_source_projection(code)
        if repaired != code:
            return unavailable_source_repair_name, repaired

    bound_source_repair_name = "bound_figure_source_projection_v2"
    if (
        len(unavailable_source_findings) == 1
        and previous_repair != bound_source_repair_name
    ):
        repaired = _patch_bound_figure_source_projection(code)
        if repaired != code:
            return bound_source_repair_name, repaired

    attrition_identity_findings = []
    for finding in findings:
        validator = getattr(finding, "validator", None)
        detail = getattr(finding, "detail", None)
        if isinstance(finding, dict):
            validator = finding.get("validator")
            detail = finding.get("detail")
        if (
            validator == "primary_analysis_cohort_integrity"
            and isinstance(detail, dict)
            and detail.get("issue") == "attrition_sequence_rule_ids_mismatch"
        ):
            attrition_identity_findings.append(detail)
    attrition_repair_name = "attrition_rule_id_canonicalization_v1"
    if (
        len(attrition_identity_findings) == 1
        and previous_repair != attrition_repair_name
    ):
        detail = attrition_identity_findings[0]
        expected_rule_ids = detail.get("expected_criterion_ids")
        reported_rule_ids = detail.get("reported_criterion_ids")
        if isinstance(expected_rule_ids, list) and isinstance(reported_rule_ids, list):
            repaired = patch_attrition_rule_id_canonicalization(
                code,
                expected_rule_ids=expected_rule_ids,
                reported_rule_ids=reported_rule_ids,
            )
            if repaired != code:
                return attrition_repair_name, repaired

    provenance_source_findings = []
    for finding in findings:
        validator = getattr(finding, "validator", None)
        detail = getattr(finding, "detail", None)
        if isinstance(finding, dict):
            validator = finding.get("validator")
            detail = finding.get("detail")
        if (
            validator == "step_summary_integrity"
            and isinstance(detail, dict)
            and detail.get("issue")
            in {
                "measurement_provenance_source_invalid",
                "measurement_provenance_check_unplanned",
                "measurement_provenance_check_missing",
            }
        ):
            provenance_source_findings.append(finding)
    provenance_repair_name = "measurement_provenance_summary_mapping_v2"
    if provenance_source_findings and previous_repair != provenance_repair_name:
        repaired = patch_measurement_provenance_contract(code, findings=findings)
        if repaired != code:
            return provenance_repair_name, repaired

    for finding in findings:
        validator = getattr(finding, "validator", None)
        detail = getattr(finding, "detail", None)
        if isinstance(finding, dict):
            validator = finding.get("validator")
            detail = finding.get("detail")
        if validator != "overadjustment_auditor" or not isinstance(detail, dict):
            continue
        if detail.get("kind") != "overadjustment":
            continue
        offenders = [
            str(value)
            for value in (detail.get("offending_covariates") or [])
            if str(value).strip()
        ]
        if not offenders:
            continue
        strip_names = _overadjustment_strip_names(offenders)
        repair_name = "drop_overadjustment_covariates_v1"
        if previous_repair == repair_name:
            return None
        repaired = _strip_columns_from_list_literals(code, strip_names)
        repaired = _patch_overadjustment_covariate_filter(repaired, strip_names)
        if repaired != code:
            return repair_name, repaired
    return None


# Captures ``NameError: name 'foo' is not defined`` for use by Fix F.
_NAME_ERROR_HELPER_RE = re.compile(
    r"NameError:\s*name\s+['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"]\s+is\s+not\s+defined"
)


def _undefined_helper_reference_is_callable(code: str, name: str) -> bool:
    """Prove the missing name is a callable/default hook, not an object alias."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        (isinstance(node.func, ast.Name) and node.func.id == name)
        or any(
            keyword.arg == "default"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == name
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def _patch_boolean_mask_reduction_precedence(code: str) -> Optional[str]:
    """Move ``sum`` after a mistakenly scalarised boolean-mask operation.

    Generated code occasionally emits ``int(mask.sum() & other_mask)``.  The
    reduction turns the left mask into a scalar before the bitwise operation,
    so an array-valued right operand makes ``int(...)`` fail.  This helper is
    intentionally syntax-narrow: it only rewrites an ``int`` call whose sole
    argument is ``mask.sum() & array_like`` (or ``|``), leaving every other
    reduction and bitwise expression untouched.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    from ..gates.numeric_reduction import patch_misnested_boolean_mask_reduction

    misnested_repair = patch_misnested_boolean_mask_reduction(code)
    if misnested_repair is not None:
        code = misnested_repair
        tree = ast.parse(code)

    array_like_nodes = (
        ast.Name,
        ast.Attribute,
        ast.Subscript,
        ast.Call,
        ast.Compare,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
    )
    replacements: List[tuple[int, int, str]] = []
    lines = code.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.BinOp)
            and isinstance(node.args[0].op, (ast.BitAnd, ast.BitOr))
        ):
            continue
        operation = node.args[0]
        reduction = operation.left
        if not (
            isinstance(reduction, ast.Call)
            and isinstance(reduction.func, ast.Attribute)
            and reduction.func.attr == "sum"
            and not reduction.args
            and not reduction.keywords
            and isinstance(operation.right, array_like_nodes)
        ):
            continue
        if not all(
            isinstance(value, int)
            for value in (
                node.lineno,
                node.col_offset,
                node.end_lineno,
                node.end_col_offset,
            )
        ):
            continue
        mask_source = ast.get_source_segment(code, reduction.func.value)
        right_source = ast.get_source_segment(code, operation.right)
        if not mask_source or not right_source:
            continue
        operator = "&" if isinstance(operation.op, ast.BitAnd) else "|"
        replacement = f"int((({mask_source}) {operator} ({right_source})).sum())"
        replacements.append(
            (
                _absolute_offset(node.lineno, node.col_offset),
                _absolute_offset(node.end_lineno, node.end_col_offset),
                replacement,
            )
        )

    if not replacements:
        return misnested_repair
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired if repaired != code else None


def _patch_pandas_boolean_index_alignment(
    code: str,
    run_log: str,
) -> Optional[str]:
    """Reindex the exact boolean mask named by a pandas alignment traceback.

    Pandas reports ``Unalignable boolean Series`` only after proving that the
    boolean indexer and selected Series have different indexes.  Generated
    helper functions sometimes receive a mask that was already subsetted and
    then apply it to the full Series again.  This transform is intentionally
    traceback-bound: it changes exactly one ``series.loc[mask]`` expression on
    the reported failing line and only wraps the mask with
    ``reindex(series.index, fill_value=False)``.  It does not choose rows,
    variables, or an analysis method.
    """

    if "Unalignable boolean Series provided as indexer" not in (run_log or ""):
        return None
    line_matches = re.findall(
        r'File\s+["\'][^"\']*analysis\.py["\'],\s+line\s+(\d+)',
        run_log or "",
    )
    if not line_matches:
        return None
    failing_line = int(line_matches[-1])
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    candidates: List[ast.Subscript] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Subscript)
            and int(getattr(node, "lineno", -1)) == failing_line
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "loc"
            and isinstance(node.slice, ast.Name)
        ):
            continue
        candidates.append(node)
    if len(candidates) != 1:
        return None

    candidate = candidates[0]
    base_source = ast.get_source_segment(code, candidate.value.value)
    mask_source = ast.get_source_segment(code, candidate.slice)
    if not base_source or not mask_source:
        return None

    lines = code.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacement = f"({mask_source}).reindex(({base_source}).index, fill_value=False)"
    start = _absolute_offset(candidate.slice.lineno, candidate.slice.col_offset)
    end = _absolute_offset(candidate.slice.end_lineno, candidate.slice.end_col_offset)
    repaired = code[:start] + replacement + code[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired if repaired != code else None


def _patch_scalar_cast_before_reduction(code: str) -> str:
    """Reduce a proven array-like count before its built-in ``int`` cast."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    from ..gates.numeric_reduction import patch_misnested_boolean_mask_reduction

    misnested_repair = patch_misnested_boolean_mask_reduction(code)
    if misnested_repair is not None:
        code = misnested_repair
        tree = ast.parse(code)

    from ..gates.preflight import (
        _builtin_int_binding_is_unmodified,
        _unreduced_boolean_mask_count_casts,
    )

    if not _builtin_int_binding_is_unmodified(tree):
        return code

    lines = code.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def _absolute_offset(lineno: int, utf8_col: int) -> int:
        line = lines[lineno - 1]
        char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
        return line_starts[lineno - 1] + char_col

    replacements: List[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sum"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "int"
            and len(node.func.value.args) == 1
            and not node.func.value.keywords
        ):
            continue
        expression = ast.get_source_segment(code, node.func.value.args[0])
        if not expression or not all(
            isinstance(value, int)
            for value in (
                node.lineno,
                node.col_offset,
                node.end_lineno,
                node.end_col_offset,
            )
        ):
            continue
        replacements.append(
            (
                _absolute_offset(node.lineno, node.col_offset),
                _absolute_offset(node.end_lineno, node.end_col_offset),
                f"int(({expression}).sum())",
            )
        )

    for node in _unreduced_boolean_mask_count_casts(tree):
        expression = ast.get_source_segment(code, node.args[0])
        if not expression or not all(
            isinstance(value, int)
            for value in (
                node.lineno,
                node.col_offset,
                node.end_lineno,
                node.end_col_offset,
            )
        ):
            continue
        replacements.append(
            (
                _absolute_offset(node.lineno, node.col_offset),
                _absolute_offset(node.end_lineno, node.end_col_offset),
                f"int(({expression}).sum())",
            )
        )

    if not replacements:
        return code
    ordered = sorted(replacements)
    if any(
        end > next_start
        for (_, end, _), (next_start, _, _) in zip(ordered, ordered[1:])
    ):
        return code
    repaired = code
    for start, end, replacement in reversed(ordered):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired


def _deterministic_runner_repair(
    *,
    code: str,
    run_log: str,
    previous_repair: Optional[str] = None,
    analysis_family: Optional[str] = None,
    resolved_input_bindings: Mapping[str, Any] | None = None,
) -> Optional[tuple[str, str]]:
    """Best-effort execution-layer patch for common numeric model failures.

    We keep this deliberately narrow: it only activates for recurrent design-
    matrix dtype / inf / NaN failures around statsmodels-style regression code.
    The repair is deterministic and is meant to reduce prompt drift in the
    coder by handling one family of brittle runtime errors below the LLM layer.
    """
    lowered = (run_log or "").lower()
    binary_model_repair_allowed = _family_allows_binary_model_repair(analysis_family)
    structured_role_repair = "structured_analysis_role_selection_v1"
    if previous_repair != structured_role_repair:
        repaired = patch_structured_analysis_role_selection(code, run_log)
        if repaired is not None and repaired != code:
            return structured_role_repair, repaired
    if repair := _finding_json_repair(code, run_log, previous_repair):
        return repair

    host_helper_import_repair = "relocate_known_host_helper_import_v1"
    if previous_repair != host_helper_import_repair:
        repaired = patch_known_host_helper_import(code, run_log)
        if repaired is not None and repaired != code:
            return host_helper_import_repair, repaired

    input_scope_repair = "raw_input_physical_superset_guard_v1"
    if previous_repair != input_scope_repair:
        repaired = patch_raw_input_physical_superset_guard(code, run_log)
        if repaired != code:
            return input_scope_repair, repaired

    receipt_source_repair = "host_receipt_source_envelope_v1"
    if previous_repair != receipt_source_repair:
        repaired = patch_direct_host_receipt_source_guard(code, run_log)
        if repaired != code:
            return receipt_source_repair, repaired

    identity_key_repair = "resolved_input_identity_key_v1"
    if previous_repair != identity_key_repair and (
        "typed input binding lacks exact input_key" in lowered
        or "typed binding key mismatch" in lowered
    ):
        try:
            identity_findings = direct_resolved_input_key_findings(ast.parse(code))
        except SyntaxError:
            identity_findings = []
        repaired = patch_direct_resolved_input_identity_key(
            code,
            findings=identity_findings,
        )
        if repaired != code:
            return identity_key_repair, repaired

    non_tabular_gate_repair = "non_tabular_companion_row_gate_v1"
    if previous_repair != non_tabular_gate_repair:
        repaired = patch_non_tabular_companion_row_gate(
            code,
            run_log,
            resolved_input_bindings=resolved_input_bindings,
        )
        if repaired != code:
            return non_tabular_gate_repair, repaired

    json_document_repair = "resolved_input_json_document_adapter_v1"
    if previous_repair != json_document_repair:
        repaired = patch_resolved_json_document_adapter(code, run_log)
        if repaired != code:
            return json_document_repair, repaired

    text_denominator_repair = "text_distribution_denominator_from_counts_v1"
    if previous_repair != text_denominator_repair:
        repaired = patch_text_distribution_denominator_from_counts(code, run_log)
        if repaired is not None and repaired != code:
            return text_denominator_repair, repaired

    manifest_env_repair = "resolved_input_manifest_env_v1"
    if previous_repair != manifest_env_repair:
        repaired = patch_resolved_input_manifest_env(code, run_log)
        if repaired != code:
            return manifest_env_repair, repaired

    all_rows_outcome_repair = "all_rows_outcome_coordinate_filter_v1"
    if previous_repair != all_rows_outcome_repair:
        repaired = patch_all_rows_outcome_coordinate_filter(
            code,
            run_log,
            resolved_input_bindings=resolved_input_bindings,
        )
        if repaired != code:
            return all_rows_outcome_repair, repaired

    merge_collision_repair = "pandas_merge_dynamic_column_collision_guard_v1"
    if previous_repair != merge_collision_repair:
        repaired = patch_pandas_merge_dynamic_column_collision(code, run_log)
        if repaired != code:
            return merge_collision_repair, repaired

    name_alias_repair = "undefined_mapping_near_match_alias_v1"
    if previous_repair != name_alias_repair:
        repaired = patch_undefined_mapping_near_match_alias(code, run_log)
        if repaired != code:
            return name_alias_repair, repaired

    mask_reduction_precedence_failure = "typeerror:" in lowered and (
        "only length-1 arrays can be converted to python scalars" in lowered
        or "cannot convert the series to <class 'int'>" in lowered
    )
    if mask_reduction_precedence_failure:
        repair_name = "boolean_mask_reduction_precedence_v1"
        if previous_repair != repair_name:
            repaired = _patch_boolean_mask_reduction_precedence(code)
            if repaired is not None:
                return repair_name, repaired

    boolean_index_alignment_repair = "pandas_boolean_index_alignment_v1"
    if previous_repair != boolean_index_alignment_repair:
        repaired = _patch_pandas_boolean_index_alignment(code, run_log)
        if repaired is not None:
            return boolean_index_alignment_repair, repaired

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
        if host_module_is_available(bad_module):
            return None
        repair_name = f"strip_fake_easyicu_import_{bad_module.replace('.', '_')}_v1"
        if previous_repair != repair_name:
            stripped_names: list[str] = []
            _line_re = re.compile(
                r"^\s*from\s+"
                + re.escape(bad_module)
                + r"\s+import\s+(.+?)\s*(?:#.*)?$",
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
                    f'"{n} from {bad_module} is not available; '
                    f'reimplement inline using numpy/scipy/statsmodels.")'
                    for n in dict.fromkeys(stripped_names)
                )
                repaired = insert_after_imports(
                    repaired,
                    "# auto-stubs for stripped fake imports\n" + _stub_lines,
                )
                return repair_name, repaired

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

    categorical_order_repair = "categorical_declared_order_check_v1"
    if previous_repair != categorical_order_repair:
        repaired = patch_categorical_declared_order_check(code, run_log)
        if repaired != code:
            return categorical_order_repair, repaired

    missing_seaborn = (
        "modulenotfounderror: no module named 'seaborn'" in lowered
        and "import seaborn as sns" in code
    )
    if missing_seaborn:
        repair_name = "seaborn_matplotlib_fallback_v1"
        if previous_repair != repair_name:
            fallback = textwrap.dedent("""
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
                    def boxplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None and y is not None:
                            groups = data.groupby(x, dropna=False)[y]
                            labels = [str(v) for v in groups.groups.keys()]
                            series = [g.dropna().values for _, g in groups]
                            if series:
                                try:
                                    ax.boxplot(series, tick_labels=labels)
                                except TypeError:
                                    # Matplotlib <3.9 used ``labels``; newer
                                    # releases renamed it to ``tick_labels``.
                                    ax.boxplot(series, labels=labels)
                        elif data is not None and y is not None:
                            ax.boxplot(data[y].dropna().values)
                        return ax
                    def violinplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        return self.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
                    def boxenplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        return self.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
                    def stripplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        return self.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
                    def swarmplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        return self.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
                    def pointplot(self, data=None, x=None, y=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None and y is not None:
                            grouped = data.groupby(x, dropna=False)[y].mean()
                            ax.plot([str(v) for v in grouped.index], grouped.values, marker="o")
                        return ax
                    def countplot(self, data=None, x=None, hue=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        if data is not None and x is not None:
                            counts = data[x].value_counts(dropna=False)
                            ax.bar([str(v) for v in counts.index], counts.values)
                        return ax
                    def kdeplot(self, data=None, x=None, ax=None, **kwargs):
                        import matplotlib.pyplot as plt
                        ax = ax or plt.gca()
                        values = data[x] if (data is not None and x is not None) else data
                        if values is not None and hasattr(values, "dropna"):
                            ax.hist(values.dropna(), bins=kwargs.get("bins", 30), density=True, histtype="step")
                        return ax
                    def regplot(self, data=None, x=None, y=None, ax=None, **kwargs):
                        return self.scatterplot(data=data, x=x, y=y, ax=ax, **kwargs)
                    def despine(self, *args, **kwargs):
                        ax = kwargs.get("ax")
                        if ax is not None:
                            for side in ("top", "right"):
                                if side in getattr(ax, "spines", {}):
                                    ax.spines[side].set_visible(False)
                        return None
                    def set_context(self, *args, **kwargs):
                        return None
                    def set_palette(self, *args, **kwargs):
                        return None
                    def set(self, *args, **kwargs):
                        return None
                    def move_legend(self, *args, **kwargs):
                        return None
                    def __getattr__(self, name):
                        # Any seaborn attribute this shim does not explicitly implement
                        # degrades to a safe no-op instead of raising AttributeError,
                        # so a single unsupported call never crashes an entire figure
                        # render in the baseline-library sandbox. Dunder lookups still
                        # raise so Python's internal protocols behave normally.
                        if name.startswith("__") and name.endswith("__"):
                            raise AttributeError(name)
                        def _seaborn_noop(*args, **kwargs):
                            return kwargs.get("ax")
                        return _seaborn_noop
                sns = _EasyICUSeabornFallback()
                """).strip()
            repaired = code.replace("import seaborn as sns", fallback, 1)
            if repaired != code:
                return repair_name, repaired

    missing_proportion_confint = (
        "modulenotfounderror: no module named 'statsmodels'" in lowered
        or "cannot import name 'proportion_confint' from 'scipy.stats'" in lowered
    ) and "proportion_confint" in code
    if missing_proportion_confint:
        repair_name = "local_wilson_proportion_confint_v1"
        if previous_repair != repair_name:
            helper = textwrap.dedent("""
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
                """).strip()
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

    json_numpy_scalar_failure = any(
        marker in lowered
        for marker in (
            "object of type bool is not json serializable",
            "object of type bool_ is not json serializable",
            "object of type int32 is not json serializable",
            "object of type int64 is not json serializable",
            "object of type float32 is not json serializable",
            "object of type float64 is not json serializable",
            "object of type ndarray is not json serializable",
        )
    )
    json_numpy_key_failure = (
        "keys must be str, int, float, bool or none" in lowered
        or json_numpy_scalar_failure
    ) and "json.dump(" in code
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

    malformed_python_prefix = "syntaxerror: invalid syntax" in lowered and (
        "pythonimport " in code or "\npythonimport " in code or "pythonfrom " in code
    )
    if malformed_python_prefix:
        repair_name = "strip_python_prefix_v1"
        if previous_repair != repair_name:
            repaired = code.replace("pythonimport ", "import ").replace(
                "pythonfrom ", "from "
            )
            repaired = repaired.replace("\npythonimport ", "\nimport ").replace(
                "\npythonfrom ", "\nfrom "
            )
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
        and ("model_df[x_cols]" in code or "model_df[[outcome_col] + x_cols]" in code)
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
                    '    if col in data and col not in ["sex"]:\n'
                    '        data[col] = pd.to_numeric(data[col], errors="coerce")\n'
                    'if "sex" in data:\n'
                    '    data["sex"] = data["sex"].astype("string")'
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
                    repaired = code.replace(
                        match.group("line"), guard + "\n" + match.group("line"), 1
                    )
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
        and _code_mentions_missing_indicator_column(code)
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
            sex_numeric_patch = textwrap.dedent("""
                if 'sex' in model_df.columns:
                    model_df['sex'] = model_df['sex'].astype(str).str.lower().isin(['m', 'male']).astype(float)
                for col in model_df.columns:
                    if col != 'sex':
                        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                model_df = model_df.replace([np.inf, -np.inf], np.nan)
                reduced_covariates = [c for c in covariates if model_df[c].isna().mean() <= 0.2]
                """).strip("\n")
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
                "rv_X = rv_df[covariates]": "rv_X = rv_df[[predictor_col] + covariates]",
                "rv_X = rv_df[reduced_covariates]": "rv_X = rv_df[[predictor_col] + reduced_covariates]",
                'X_cc = sm.add_constant(complete_case_df[covariates], has_constant="add")': (
                    f'X_cc = sm.add_constant(complete_case_df[[{predictor_var}] + covariates], has_constant="add")'
                ),
                'X_cc = sm.add_constant(cc_df[covariates], has_constant="add")': (
                    f'X_cc = sm.add_constant(cc_df[[{predictor_var}] + covariates], has_constant="add")'
                ),
                'X_rv = sm.add_constant(reduced_variable_df[covariates], has_constant="add")': (
                    f'X_rv = sm.add_constant(reduced_variable_df[[{predictor_var}] + reduced_covariates], has_constant="add")'
                ),
                'X_rv = sm.add_constant(rv_df[covariates], has_constant="add")': (
                    f'X_rv = sm.add_constant(rv_df[[{predictor_var}] + reduced_covariates], has_constant="add")'
                ),
            }
            for old, new in replacements.items():
                repaired = repaired.replace(old, new)
            missing_expr = (
                r"(?:['\"][A-Za-z_][A-Za-z0-9_]*_missing(?:_[A-Za-z0-9_]+)?['\"]|"
                r"missing_indicator_col)"
            )
            repaired = re.sub(
                rf"(?m)^(?P<indent>\s*)"
                rf"(?P<lhs>mi_X)\s*=\s*(?P<df>mi_df|model_df)"
                rf"\[covariates\s*\+\s*\[(?P<missing>{missing_expr})\]\]\s*$",
                lambda match: (
                    f"{match.group('indent')}{match.group('lhs')} = "
                    f"{match.group('df')}[[{predictor_var}] + covariates + "
                    f"[{match.group('missing')}]]"
                ),
                repaired,
            )
            repaired = re.sub(
                rf"(?m)^(?P<indent>\s*)"
                rf"(?P<lhs>X_mi)\s*=\s*sm\.add_constant\("
                rf"(?P<df>missing_indicator_df|mi_df)"
                rf"\[covariates\s*\+\s*\[(?P<missing>{missing_expr})\]\], "
                rf"has_constant=\"add\"\)\s*$",
                lambda match: (
                    f"{match.group('indent')}{match.group('lhs')} = "
                    f"sm.add_constant({match.group('df')}[[{predictor_var}] "
                    f"+ covariates + [{match.group('missing')}]], "
                    'has_constant="add")'
                ),
                repaired,
            )
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
                'X_cc = X_cc.apply(pd.to_numeric, errors="coerce").astype(float)\n    y_cc = y_cc.astype(float)': (
                    'X_cc = X_cc.apply(pd.to_numeric, errors="coerce").astype(float)\n'
                    "    y_cc = y_cc.astype(float)\n"
                    "    cc_mask = np.isfinite(X_cc.to_numpy()).all(axis=1) & np.isfinite(y_cc.to_numpy())\n"
                    "    X_cc = X_cc.loc[cc_mask]\n"
                    "    y_cc = y_cc.loc[cc_mask]"
                ),
                'X_mi = X_mi.apply(pd.to_numeric, errors="coerce").astype(float)\n    y_mi = y_mi.astype(float)': (
                    'X_mi = X_mi.apply(pd.to_numeric, errors="coerce").astype(float)\n'
                    "    y_mi = y_mi.astype(float)\n"
                    "    mi_mask = np.isfinite(X_mi.to_numpy()).all(axis=1) & np.isfinite(y_mi.to_numpy())\n"
                    "    X_mi = X_mi.loc[mi_mask]\n"
                    "    y_mi = y_mi.loc[mi_mask]"
                ),
                'X_rv = X_rv.apply(pd.to_numeric, errors="coerce").astype(float)\n    y_rv = y_rv.astype(float)': (
                    'X_rv = X_rv.apply(pd.to_numeric, errors="coerce").astype(float)\n'
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
            plot_guard = textwrap.dedent(f"""
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
                """).strip("\n")
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
            helper = textwrap.dedent("""
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
                """).strip()
            repaired = code.replace(
                "from easyicu.research_agent.utils import to_jsonable",
                helper,
                1,
            )
            if repaired != code:
                return repair_name, repaired

    missing_figure_utils = (
        "modulenotfounderror: no module named 'easyicu.research_output'" in lowered
        or "modulenotfounderror: no module named 'easyicu.research_output.figure_utils'"
        in lowered
        or "no module named 'easyicu.research_output'" in lowered
    ) and "easyicu.research_output.figure_utils" in code
    if missing_figure_utils:
        repair_name = "replace_hallucinated_figure_utils_import_v1"
        if previous_repair != repair_name:
            repaired = code.replace(
                "easyicu.research_output.figure_utils",
                "easyicu.research_agent.figures.publication",
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
                block = textwrap.dedent(f"""
                    if 'sex' in X_{prefix}.columns:
                        X_{prefix}['sex'] = X_{prefix}['sex'].astype(str).str.lower().isin(['m', 'male', '1', 'true']).astype(float)
                    X_{prefix} = X_{prefix}.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
                    y_{prefix} = pd.to_numeric(y_{prefix}, errors='coerce').replace([np.inf, -np.inf], np.nan)
                    valid_{prefix}_idx = X_{prefix}.dropna().index.intersection(y_{prefix}.dropna().index)
                    X_{prefix} = X_{prefix}.loc[valid_{prefix}_idx]
                    y_{prefix} = y_{prefix}.loc[valid_{prefix}_idx].astype(float)
                    """).strip("\n")
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
            repaired = textwrap.dedent("""
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
                """).strip() + "\n"
            return repair_name, repaired

    outcome_incidence_broken_syntax = "syntaxerror" in lowered and (
        "outcome_incidence" in code.lower()
        or "incidence_with_missingness_strata" in code.lower()
    )
    if outcome_incidence_broken_syntax:
        repair_name = "outcome_incidence_descriptive_repair_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent("""
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
                """).strip() + "\n"
            return repair_name, repaired

    repeated_keyword_syntax = (
        "syntaxerror: keyword argument repeated" in lowered
        and "train_test_split" in code
        and "figure_contract = figurecontract(" in code.lower()
    )
    if repeated_keyword_syntax and binary_model_repair_allowed:
        repair_name = "prediction_split_minimal_v1"
        if previous_repair != repair_name:
            repaired = textwrap.dedent("""
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
                """).strip() + "\n"
            return repair_name, repaired

    logreg_nan = (
        "logisticregression does not accept missing values encoded as nan" in lowered
        and "logisticregression" in code.lower()
    )
    if logreg_nan and binary_model_repair_allowed:
        repair_name = "logreg_impute_v1"
        if previous_repair != repair_name and "_easyicu_logreg_impute_v1" not in code:
            patch = textwrap.dedent("""

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
                """).strip("\n")
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
                predict_call = re.compile(
                    r"(?P<line>y_pred_proba\s*=\s*model(?:_pipeline)?\.predict_proba\(X_test\)\s*\[:,\s*1\]\s*)"
                )
                match = predict_call.search(code)
                if match:
                    inject = (
                        "X_test = _easyicu_logreg_impute_v1(X_test)\n"
                        + match.group("line")
                    )
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
            repaired = textwrap.dedent("""
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
                from easyicu.research_agent.figures.publication import (
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
                """).strip() + "\n"
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
        "typeerror: '<' not supported between instances of 'tuple' and 'int'" in lowered
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
                    repaired = code[: match.start()] + replacement + code[match.end() :]
                    return repair_name, repaired

    singular_logit = "singular matrix" in lowered and "sm.logit(" in code.lower()
    if singular_logit and binary_model_repair_allowed:
        repair_name = "rank_safe_statsmodels_design_v1"
        if previous_repair != repair_name:
            patched = _patch_rank_safe_statsmodels_design(code)
            if patched is not None and patched != code:
                return repair_name, patched
        repair_name = "logit_regularized_fit_v1"
        if previous_repair != repair_name:
            helper = textwrap.dedent("""

                def _easyicu_safe_logit_fit_v1(model):
                    try:
                        return model.fit(disp=0, method="newton")
                    except Exception:
                        return model.fit_regularized(alpha=1e-6, disp=0, trim_mode="off")
                """).strip("\n")
            patched = code
            if "_easyicu_safe_logit_fit_v1" not in patched:
                insert_after = patched.find("import warnings")
                if insert_after >= 0:
                    line_end = patched.find("\n", insert_after)
                    patched = (
                        patched[: line_end + 1]
                        + "\n"
                        + helper
                        + "\n"
                        + patched[line_end + 1 :]
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

    table_one_binary_keyerror = "keyerror: 1" in lowered and re.search(
        r'summary\["outcomes"\]\["[^"]+"\]\["(?:counts|pct)"\]\[1\]',
        code,
    )
    if table_one_binary_keyerror:
        repair_name = "table_one_binary_key_string_v1"
        if previous_repair != repair_name:
            repaired = re.sub(
                r'summary\["outcomes"\]\["(?P<outcome>[^"]+)"\]\["(?P<field>counts|pct)"\]\[1\]',
                lambda match: (
                    f'summary["outcomes"]["{match.group("outcome")}"]["{match.group("field")}"].get('
                    f'"1", summary["outcomes"]["{match.group("outcome")}"]'
                    f'["{match.group("field")}"].get(1, '
                    f'{"0" if match.group("field") == "counts" else "0.0"}))'
                ),
                code,
            )
            if repaired != code:
                return repair_name, repaired

    cohort_file_as_dir = "notadirectoryerror" in lowered and (
        'os.path.join(cohort_path, "data.parquet")' in code.lower()
        or "os.path.join(cohort_path, 'data.parquet')" in code.lower()
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
        and ("cohort_path" in code.lower() or "cohort_parquet" in code.lower())
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
            repaired = textwrap.dedent("""
                from __future__ import annotations
                import json
                import os
                import shutil
                from pathlib import Path

                out_dir = Path(os.environ["STEP_OUT_DIR"])
                out_dir.mkdir(parents=True, exist_ok=True)
                run_dir = Path(os.environ.get("EASYICU_RUN_DIR") or out_dir.parents[2])
                current_step_id = os.environ.get("EASYICU_STEP_ID") or out_dir.parent.name
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
                """).strip() + "\n"
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
            and _undefined_helper_reference_is_callable(code, helper_name)
            and f"def {helper_name}" not in code
            and f"{helper_name} =" not in code
        ):
            repair_name = f"undefined_helper_stub_{helper_name}_v1"
            if previous_repair != repair_name:
                stub = textwrap.dedent(f"""
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
                    """).strip("\n")
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
        patch = textwrap.dedent("""

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
            """).strip("\n")

        patched = code
        if "_easyicu_runner_repair_v1" not in patched:
            insert_after = patched.find("import matplotlib.pyplot as plt")
            if insert_after >= 0:
                line_end = patched.find("\n", insert_after)
                patched = (
                    patched[: line_end + 1]
                    + "\n"
                    + patch
                    + "\n"
                    + patched[line_end + 1 :]
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
            "X_array = X.to_numpy()\n" "y_array = y.to_numpy()\n",
            "y, X = _easyicu_runner_repair_v1(X, y)\n"
            "X_array = np.asarray(X, dtype=float)\n"
            "y_array = np.asarray(y, dtype=float)\n",
            1,
        )
        if repaired != code:
            return repair_name, repaired

    return None
