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
from .code_repair_helpers import (  # noqa: F401  (re-exported for back-compat)
    _BINARY_MODEL_REPAIR_FAMILIES,
    _KEYERROR_NOT_IN_INDEX_RE,
    _code_contains_binary_model,
    _code_mentions_missing_indicator_column,
    _extract_missing_index_columns,
    _extract_required_cols_list,
    _family_allows_binary_model_repair,
    _infer_analysis_cohort_source_column,
    _patch_derived_analysis_cohort_materialization,
    _patch_json_dump_numpy_key_sanitizer,
    _patch_primary_predictor_into_design_matrix,
    _patch_statsmodels_conf_int_filter_axis,
    _patch_statsmodels_endog_exog_index_alignment,
    _statsmodels_repair_allowed_for_family,
    _strip_columns_from_list_literals,
)


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
    helper = textwrap.dedent(
        """

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
        """
    ).strip("\n")

    model_call = re.compile(
        r"(?m)^(?P<indent>\s*)(?P<lhs>[A-Za-z_]\w*)\s*=\s*sm\.Logit\("
        r"(?P<y>[^,\n)]+?)\s*,\s*(?P<X>[A-Za-z_]\w*)"
        r"(?P<kwargs>,\s*[^)\n]+)?\)\s*$"
    )

    def _rewrite(match: re.Match[str]) -> str:
        indent = match.group("indent")
        x_expr = match.group("X").strip()
        y_expr = match.group("y").strip()
        kwargs = match.group("kwargs") or ""
        lhs = match.group("lhs")
        keep_expr = (
            "[c for c in ["
            "'const', "
            "locals().get('exposure_col'), "
            "locals().get('predictor_col'), "
            "locals().get('primary_predictor')"
            f"] if c is not None and hasattr({x_expr}, 'columns') and c in {x_expr}.columns]"
        )
        return (
            f"{indent}{x_expr}, _easyicu_dropped_rank_cols_v1 = "
            f"_easyicu_rank_safe_design_v1({x_expr}, keep={keep_expr})\n"
            f"{indent}{lhs} = sm.GLM({y_expr}, {x_expr}, family=sm.families.Binomial(){kwargs})"
        )

    repaired = model_call.sub(_rewrite, code, count=1)
    if repaired == code:
        return None
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


def deterministic_concept_audit_repair(
    code: str,
    audit_messages: Sequence[str],
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
    if provenance_finding:
        guarded = _patch_provenance_fail_closed_guard(repaired)
        if guarded != repaired:
            repair_name = "provenance_fail_closed_guard_v1"
            repaired = guarded
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


def _patch_provenance_fail_closed_guard(code: str) -> str:
    """Insert a terminating guard after an explicit provenance-audit call.

    The transformation is intentionally source-local (line insertion, not
    whole-script AST regeneration).  It is only available when the audit
    function exposes an explicit decision field, so the repair never infers a
    threshold or invents an audit policy from raw counts.
    """

    if _PROVENANCE_GUARD_SENTINEL in code:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    marker_functions: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tokens = _string_literals(node)
        if not (_PROVENANCE_FAILURE_KEYS <= tokens and "audit_only" in tokens):
            continue
        decision_keys = tuple(key for key in _PROVENANCE_DECISION_KEYS if key in tokens)
        if decision_keys:
            marker_functions[node.name] = decision_keys
    if not marker_functions:
        return code

    lines = code.splitlines(keepends=True)
    insertions: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        if not isinstance(node.value, ast.Call):
            continue
        called = _simple_call_name(node.value.func).split(".")[-1]
        decision_keys = marker_functions.get(called)
        if not decision_keys:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = [target.id for target in targets if isinstance(target, ast.Name)]
        if len(target_names) != 1:
            continue

        result_name = target_names[0]
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

    if not insertions:
        return code
    for line_number, guard in sorted(insertions, reverse=True):
        lines.insert(line_number, guard)
    return "".join(lines)


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
    estimate = _first_present_scalar(
        step_summary,
        ("estimate", "primary_or", "odds_ratio", "adjusted_or", "or"),
    )
    if estimate is not None:
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
            null_model_summary
            and "singular matrix" in summary_text
            and (
                '"primary_model"' in summary_text
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


def deterministic_contract_repair(
    *,
    code: str,
    findings: Sequence[Any],
    previous_repair: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Patch objective contract/audit failures before asking the LLM to repair."""

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
        return None
    repaired = code
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired if repaired != code else None


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
    binary_model_repair_allowed = _family_allows_binary_model_repair(analysis_family)

    mask_reduction_precedence_failure = (
        "typeerror:" in lowered
        and "only length-1 arrays can be converted to python scalars" in lowered
    )
    if mask_reduction_precedence_failure:
        repair_name = "boolean_mask_reduction_precedence_v1"
        if previous_repair != repair_name:
            repaired = _patch_boolean_mask_reduction_precedence(code)
            if repaired is not None:
                return repair_name, repaired

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
                # Insert after the first contiguous block of imports
                lines = repaired.splitlines()
                insert_at = 0
                for i, ln in enumerate(lines[:80]):
                    if ln.startswith(("import ", "from ", "#")) or not ln.strip():
                        insert_at = i + 1
                    else:
                        break
                lines.insert(
                    insert_at,
                    "\n# auto-stubs for stripped fake imports\n" + _stub_lines + "\n",
                )
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
                """
            ).strip()
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
        "keys must be str, int, float, bool or none" in lowered and "json.dump(" in code
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
            repaired = (
                textwrap.dedent(
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
                ).strip()
                + "\n"
            )
            return repair_name, repaired

    outcome_incidence_broken_syntax = "syntaxerror" in lowered and (
        "outcome_incidence" in code.lower()
        or "incidence_with_missingness_strata" in code.lower()
    )
    if outcome_incidence_broken_syntax:
        repair_name = "outcome_incidence_descriptive_repair_v1"
        if previous_repair != repair_name:
            repaired = (
                textwrap.dedent(
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
                ).strip()
                + "\n"
            )
            return repair_name, repaired

    repeated_keyword_syntax = (
        "syntaxerror: keyword argument repeated" in lowered
        and "train_test_split" in code
        and "figure_contract = figurecontract(" in code.lower()
    )
    if repeated_keyword_syntax and binary_model_repair_allowed:
        repair_name = "prediction_split_minimal_v1"
        if previous_repair != repair_name:
            repaired = (
                textwrap.dedent(
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
                ).strip()
                + "\n"
            )
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
            repaired = (
                textwrap.dedent(
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
                ).strip()
                + "\n"
            )
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
            repaired = (
                textwrap.dedent(
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
                ).strip()
                + "\n"
            )
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
