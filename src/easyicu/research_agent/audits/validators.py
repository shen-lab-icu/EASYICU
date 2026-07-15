"""Deterministic verification gates between LLM output and downstream use.

The validators are the *trust* layer. Their job is to inspect agent
output (plans, scripts, run results) and decide whether it is safe to
let it influence the manuscript scaffold.

There are three of them:

1. :class:`CohortAuditor` — checks that the cohort frame matches its
   declared descriptor (size, id columns present, target outcome
   present, no surprise NaN-only columns).
2. :class:`ConceptUsageAuditor` — static analysis of the generated
   script: flags forbidden patterns like
   ``df['sofa'].mean()`` or  ``df['lact'].mean()`` (skewed lab without
   median fallback).
3. :class:`StatisticalValidator` — runs after the script and inspects
   the artefacts produced. Re-derives outcome incidence from the
   cohort, cross-checks the script's ``step_summary.json`` against
   the cohort, flags out-of-monotonic SOFA strata.

Each validator returns a list of :class:`ValidationFinding` objects
that the pipeline appends to the manifest. Severity ``error`` blocks
manuscript generation; ``warning`` is surfaced but does not block.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from ..analysis_method_suite import figure_product_source_obligations
from ..declared_product_contract import (
    effect_adjustment_family,
    effect_bearing_product,
    effect_estimand_tier,
    effect_measure_family,
    effect_role_family,
    typed_product,
)
from ..replication.paper import compare_metric_values
from ..ordered_stratified_contract import ordered_stratified_numeric_findings
from ..schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    AggregationRule,
    AnalysisStep,
    ConceptDescriptor,
    EvidenceRecord,
    PaperProfile,
    PaperResultLedger,
    ResearchContext,
    ReplicationDeviationReport,
    ValidationFinding,
    VariableRole,
)
from ..llm import LLMClient, LLMMessage
from .outcome_semantics import (
    _finding_claims_mortality_horizon_mismatch,
    _script_copies_named_full_stay_window,
    _script_has_conflicting_mortality_semantics,
    _script_uses_bound_outcome,
)
from ..runtime_artifacts import (
    current_run_evidence_records,
    current_successful_step_records,
    verified_run_evidence_path,
)
from ..trajectory_contract import trajectory_phenotyping_artifact_findings


# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and
# admission-level keys (hadm_id) deliberately do NOT qualify.
_PATIENT_ID_COLUMNS = (
    "subject_id",
    "patient_id",
    "patientid",
    "person_id",
    "uniquepid",
)
# ICU length-of-stay columns, expressed in DAYS in the EasyICU export.
_LOS_DAY_COLUMNS = ("los_icu", "los", "icu_los", "los_icu_days")


def cohort_hygiene_findings(
    df: pd.DataFrame,
    context: ResearchContext,
) -> List[ValidationFinding]:
    """Impartial, advisory cohort-hygiene flags (``warning``, never blocking).

    These surface standard cohort-hygiene questions — patient-level
    non-independence and short-stay exposure — so they are visible and
    recorded. They deliberately do NOT impose an analytical choice (a
    minimum LoS, first-stay deduplication, ...): per the impartiality rule
    the analyst decides, and the auditor only ensures the question was put.
    Severity is always ``warning`` so the gate never fail-closes on them.

    See ``feedback_rules_must_be_impartial`` and
    ``feedback_coverage_gap_vs_missing_policy``.
    """
    findings: List[ValidationFinding] = []
    cols = {str(c).lower() for c in df.columns}

    # (A) Patient-level non-independence assessability. Only relevant when an
    # outcome model is in scope (a stay-level association treats each ICU
    # stay as independent). If no patient identifier is present, that
    # assumption cannot even be CHECKED from this export — structural
    # no-source — so advise re-extraction rather than penalising the analysis
    # or silently assuming independence.
    outcome = getattr(context, "target_outcome", None)
    has_patient_id = any(pid in cols for pid in _PATIENT_ID_COLUMNS)
    if outcome and not has_patient_id:
        findings.append(ValidationFinding(
            validator="cohort_auditor",
            severity="warning",
            message=(
                "Cohort is keyed at the ICU-stay level with no patient "
                "identifier; within-patient non-independence and first-stay "
                "selection cannot be assessed from this export. Re-extract "
                "with a patient identifier (e.g. subject_id) if repeat ICU "
                "stays could affect the outcome model."
            ),
            detail={
                "kind": "cohort_hygiene",
                "subkind": "patient_independence_unassessable",
                "structural_no_source": True,
                "impartial": True,
            },
        ))

    # (B) Short-stay exposure. If an ICU LoS column (days) is present, report
    # the fraction of very short stays. Excluding <24h stays is a defensible
    # convention, NOT a requirement, so this records the distribution and
    # leaves the choice to the analyst.
    los_col = next(
        (c for c in df.columns if str(c).lower() in _LOS_DAY_COLUMNS),
        None,
    )
    if los_col is not None:
        los = pd.to_numeric(df[los_col], errors="coerce").dropna()
        if not los.empty:
            frac_short = float((los < 1.0).mean())
            if frac_short > 0:
                findings.append(ValidationFinding(
                    validator="cohort_auditor",
                    severity="warning",
                    message=(
                        f"{frac_short:.0%} of stays have ICU length-of-stay "
                        f"<1 day (column '{los_col}'); consider whether "
                        "incomplete exposure affects the analysis. No "
                        "minimum-LoS filter is imposed — recorded for the "
                        "analyst to judge."
                    ),
                    detail={
                        "kind": "cohort_hygiene",
                        "subkind": "short_stay_exposure",
                        "fraction_los_under_1_day": frac_short,
                        "los_column": los_col,
                        "impartial": True,
                    },
                ))

    return findings


class CohortAuditor:
    """Confirm the dataframe matches the descriptor it claims to."""

    name = "cohort_auditor"

    def audit(
        self,
        *,
        context: ResearchContext,
        cohort_path: Path,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        try:
            df = pd.read_parquet(cohort_path)
        except Exception as exc:
            return [ValidationFinding(
                validator=self.name, severity="error",
                message=f"Could not read cohort parquet: {exc}",
            )]

        # Row count
        if context.cohort.n_stays != int(len(df)):
            findings.append(ValidationFinding(
                validator=self.name, severity="error",
                message=(
                    f"Row count mismatch: descriptor says n_stays={context.cohort.n_stays:,} "
                    f"but cohort parquet has {len(df):,} rows."
                ),
            ))

        # Required id columns
        for col in context.cohort.id_columns:
            if col not in df.columns:
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=f"Declared id column '{col}' missing from cohort.",
                ))

        # Target outcome present and binary if labelled binary
        outcome = context.target_outcome
        if outcome:
            if outcome not in df.columns:
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=f"Target outcome '{outcome}' missing from cohort.",
                ))
            else:
                v = context.variable(outcome)
                if v and v.role == VariableRole.OUTCOME:
                    s = df[outcome].dropna()
                    if not s.empty and set(s.unique()) - {0, 1, True, False, 0.0, 1.0}:
                        findings.append(ValidationFinding(
                            validator=self.name, severity="warning",
                            message=(
                                f"Target outcome '{outcome}' has non-binary values "
                                f"({sorted(set(s.unique()))[:5]}…); confirm this is intended."
                            ),
                        ))

        # NaN-only columns
        for col in df.columns:
            if df[col].isna().all():
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Column '{col}' is entirely missing in the cohort.",
                ))

        # High-missing flag for any declared variable
        for v in context.variables:
            if v.missingness and v.missingness.fraction_missing > 0.5:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=(
                        f"Variable '{v.name}' has {v.missingness.fraction_missing:.0%} "
                        "missingness; downstream associations are at risk of selection bias."
                    ),
                    detail={"fraction_missing": v.missingness.fraction_missing},
                ))

        # Impartial, advisory cohort-hygiene flags (patient-level
        # non-independence, short-stay exposure). Always severity="warning",
        # so they record the question without enforcing a choice.
        findings.extend(cohort_hygiene_findings(df, context))

        return findings


# ---------------------------------------------------------------------------
# ConceptUsageAuditor
# ---------------------------------------------------------------------------


_FORBIDDEN_AGG_PATTERNS_BY_KIND = {
    # (role, agg method) => human-readable message.
    # Messages are phrased as conservative reporting-practice violations,
    # not as absolute mathematical errors: for bounded ordinal clinical
    # scores, median/IQR or level-distribution summaries are preferred
    # over mean/SD for manuscript-facing reporting. The same column may
    # legitimately enter a regression or Cox model as a linear covariate;
    # this auditor covers reporting/aggregation misuse only, not model
    # specification choices.
    ("ordinal_score", "mean"): "Mean of an ordinal SOFA component may be misleading; for manuscript-facing summaries prefer max-within-window or a level distribution.",
    ("ordinal_score", "std"):  "Standard deviation of an ordinal SOFA component is rarely interpretable; prefer a level distribution.",
    ("composite_score", "mean"): "Mean of a composite ordinal score (total SOFA = sum of 0–4 components) is a reporting-practice violation for bounded integer clinical scores; for manuscript-facing summaries prefer max-within-window, median (IQR) or a level distribution.",
    ("composite_score", "std"):  "Standard deviation of a composite ordinal score may be misleading; prefer median (IQR) or a level distribution.",
    ("ordinal_score_gcs", "mean"): "GCS is ordinal; for manuscript-facing summaries prefer worst (min) or a representative (last / first) value rather than mean.",
}


# Severity policy for forbidden-aggregation patterns (added 2026-05-26,
# corrected 2026-06-13).
#
# Default behaviour: severity="warning" (advisory). Mean/SD of an ordinal
# or composite clinical score is a reporting-practice preference, not an
# objective error, so per the impartiality contract it is surfaced as a
# caution and never hard-blocks a run. Setting EASYICU_AUDIT_ORDINAL_STRICT=1
# restores the historical strict fail-closed benchmark (severity="error" /
# block for primary-analysis & manuscript stages, "warning" for
# probe/descriptive stages) so a supplementary ablation can compare the two
# policies on the same benchmark without re-running unrelated logic.
_PROBE_STAGE_TOKENS = (
    "probe", "descriptive", "exploratory", "qc", "summary",
    "missingness_audit", "score_qc",
)
_BLOCKING_STAGE_TOKENS = (
    "primary_", "manuscript", "final_report", "publication", "evidence_binding",
)


def _forbidden_agg_severity(step: Optional["AnalysisStep"]) -> str:
    """Severity for a forbidden-*aggregation* pattern (mean/std of an
    ordinal / composite clinical score).

    Default: ``"warning"`` (advisory caution, does NOT block the run).

    These patterns are *reporting-practice preferences*, not objective
    mathematical errors — the same column may legitimately enter a model
    as a covariate, and a generic ``describe()``-style helper that returns
    ``{"mean": ..., "median": ...}`` for a selection-bias diagnostic is a
    defensible use of ``.mean()``. The impartiality contract (see
    ``ICU_RULES.general_principles`` kind=="caution": mean-vs-median is a
    *choice* the rule layer must surface but never impose) means these must
    advise, not hard-block. This also matches the sibling lab-mean rule,
    which is already a ``"warning"``. Hard-blocking on a single helper-level
    ``.mean()`` was observed to degrade an otherwise-correct run (whose
    primary analysis treated the score as ordinal categories) all the way
    to ``diagnostic_only`` — a false fail-closed.

    Escape hatch for an ablation that wants the historical strict
    fail-closed benchmark: ``EASYICU_AUDIT_ORDINAL_STRICT=1`` restores
    ``"error"`` (block) for primary-analysis / manuscript stages while
    keeping probe/descriptive stages advisory.
    """
    import os
    if os.environ.get("EASYICU_AUDIT_ORDINAL_STRICT") != "1":
        return "warning"
    sid = (getattr(step, "step_id", "") or "").lower()
    for tok in _PROBE_STAGE_TOKENS:
        if tok in sid:
            return "warning"
    for tok in _BLOCKING_STAGE_TOKENS:
        if tok in sid:
            return "error"
    # Strict ablation, ambiguous stage: block (historical behaviour).
    return "error"


class ConceptUsageAuditor:
    """Static analysis of generated scripts for ICU rule violations."""

    name = "concept_usage_auditor"

    def audit(
        self,
        *,
        context: ResearchContext,
        script_text: str,
        step: Optional[AnalysisStep] = None,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []

        var_by_name = {v.name: v for v in context.variables}
        try:
            tree = ast.parse(script_text)
        except SyntaxError:
            return self._regex_fallback(
                var_by_name=var_by_name,
                script_text=script_text,
                step=step,
            )

        alias_map: Dict[str, Set[str]] = {}
        mean_columns: Set[str] = set()
        median_columns: Set[str] = set()
        fillna_zero_columns: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                cols = _extract_column_names(node.value, alias_map)
                for target in node.targets:
                    if isinstance(target, ast.Name) and cols:
                        alias_map[target.id] = set(cols)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                cols = _extract_column_names(node.value, alias_map) if node.value else set()
                if cols:
                    alias_map[node.target.id] = set(cols)

        def _check(col: str, fn: str) -> None:
            v = var_by_name.get(col)
            if v is None:
                return
            role_key = v.role.value
            key = (role_key, fn)
            if key in _FORBIDDEN_AGG_PATTERNS_BY_KIND:
                findings.append(ValidationFinding(
                    validator=self.name, severity=_forbidden_agg_severity(step),
                    message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[key],
                    detail={"column": col, "function": fn, "step_id": step.step_id if step else None},
                ))
                return
            if v.name.lower() == "gcs" and fn == "mean":
                findings.append(ValidationFinding(
                    validator=self.name, severity=_forbidden_agg_severity(step),
                    message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[("ordinal_score_gcs", "mean")],
                    detail={"column": col, "function": fn},
                ))

        def _call_receiver_key(node: ast.Call) -> Optional[str]:
            func = node.func
            if not isinstance(func, ast.Attribute):
                return None
            try:
                return ast.unparse(func.value)
            except Exception:
                if isinstance(func.value, ast.Name):
                    return func.value.id
                return None

        def _mean_call_is_indicator_fraction(node: ast.Call) -> bool:
            """True for ``.isna().mean()``-style prevalence calculations."""
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "mean":
                return False
            receiver = func.value
            if isinstance(receiver, ast.Compare):
                return True
            try:
                receiver_text = ast.unparse(receiver).lower()
            except Exception:
                receiver_text = ""
            indicator_tokens = (
                ".isna(",
                ".isnull(",
                ".notna(",
                ".notnull(",
            )
            return any(token in receiver_text for token in indicator_tokens)

        mean_receivers: Set[str] = set()
        median_receivers: Set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = _call_function_name(node)
            if func_name is None:
                continue

            referenced_cols = _extract_column_names(node, alias_map)
            if func_name in {"mean", "std"}:
                if func_name == "mean" and _mean_call_is_indicator_fraction(node):
                    continue
                receiver_key = _call_receiver_key(node)
                if func_name == "mean" and receiver_key:
                    mean_receivers.add(receiver_key)
                for col in referenced_cols:
                    _check(col, func_name)
                    if func_name == "mean":
                        mean_columns.add(col)
            elif func_name == "median":
                receiver_key = _call_receiver_key(node)
                if receiver_key:
                    median_receivers.add(receiver_key)
                median_columns.update(referenced_cols)
            elif func_name in {"agg", "aggregate"}:
                agg_names = _aggregation_names_from_call(node)
                for agg_name in agg_names:
                    if agg_name in {"mean", "std"}:
                        for col in referenced_cols:
                            _check(col, agg_name)
                            if agg_name == "mean":
                                mean_columns.add(col)
                    elif agg_name == "median":
                        median_columns.update(referenced_cols)
            elif func_name == "fillna" and _call_has_zero(node):
                fillna_zero_columns.update(
                    col for col in referenced_cols if col in var_by_name
                )
            elif func_name == "eval":
                for expr in _string_literals(node):
                    if ".mean(" in expr or '.agg("mean")' in expr or ".agg('mean')" in expr:
                        findings.append(ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message=(
                                "Detected DataFrame.eval() expression containing mean-style "
                                "aggregation. Review this script manually because string-eval "
                                "can bypass column-level ICU aggregation checks."
                            ),
                            detail={"expression": expr[:200]},
                        ))

        for col in sorted(mean_columns):
            v = var_by_name.get(col)
            if v is None:
                continue
            if (
                v.role == VariableRole.LAB
                and col not in median_columns
                and not (mean_receivers & median_receivers)
            ):
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=(
                        f"Lab variable '{col}' summarised by mean() with no median() in "
                        "the same script. Right-skewed labs are conventionally reported "
                        "as median (IQR)."
                    ),
                    detail={"column": col, "function": "mean"},
                ))

        if fillna_zero_columns:
            findings.append(ValidationFinding(
                validator=self.name, severity="warning",
                message=(
                    "Detected fillna(0) — silent imputation to zero is rarely correct for "
                    "ICU variables. Use a missing-indicator or document the imputation explicitly."
                ),
                detail={"columns": sorted(fillna_zero_columns)},
            ))
        return findings

    def _regex_fallback(
        self,
        *,
        var_by_name: Dict[str, ConceptDescriptor],
        script_text: str,
        step: Optional[AnalysisStep] = None,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        pat_bracket = re.compile(r"""\[(['"])(?P<col>[^'"]+)\1\]\s*\.\s*(?P<fn>mean|std)\s*\(""")
        pat_attr = re.compile(r"""\.(?P<col>[a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*(?P<fn>mean|std)\s*\(""")
        for match in list(pat_bracket.finditer(script_text)) + list(pat_attr.finditer(script_text)):
            col = match.group("col")
            fn = match.group("fn")
            var = var_by_name.get(col)
            if var is None:
                continue
            key = (var.role.value, fn)
            if key in _FORBIDDEN_AGG_PATTERNS_BY_KIND:
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity=_forbidden_agg_severity(step),
                    message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[key],
                    detail={"column": col, "function": fn, "fallback": "regex"},
                ))
        if re.search(r"\.fillna\s*\(\s*0\s*\)", script_text):
            findings.append(ValidationFinding(
                validator=self.name,
                severity="warning",
                message=(
                    "Detected fillna(0) — silent imputation to zero is rarely correct for "
                    "ICU variables. Use a missing-indicator or document the imputation explicitly."
                ),
                detail={"fallback": "regex"},
            ))
        return findings


def _call_function_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _extract_column_names(
    node: Optional[ast.AST],
    alias_map: Dict[str, Set[str]],
) -> Set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return set(alias_map.get(node.id, set()))
    if isinstance(node, ast.Constant):
        return set()
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            base = node.value.id.lower()
            if base in {"df", "data", "cohort", "frame", "table"} or base.endswith("df"):
                return {node.attr}
            return set(alias_map.get(node.value.id, set()))
        return _extract_column_names(node.value, alias_map)
    if isinstance(node, ast.Subscript):
        cols: Set[str] = set()
        key = _subscript_key(node.slice)
        if isinstance(key, str):
            cols.add(key)
        cols.update(_extract_column_names(node.value, alias_map))
        return cols
    if isinstance(node, ast.Call):
        cols: Set[str] = set()
        cols.update(_extract_column_names(node.func, alias_map))
        for arg in node.args:
            cols.update(_extract_column_names(arg, alias_map))
        for kw in node.keywords:
            cols.update(_extract_column_names(kw.value, alias_map))
        return cols
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        cols: Set[str] = set()
        for elt in node.elts:
            cols.update(_extract_column_names(elt, alias_map))
        return cols
    if isinstance(node, ast.Dict):
        cols: Set[str] = set()
        for key in node.keys:
            cols.update(_extract_column_names(key, alias_map))
        for value in node.values:
            cols.update(_extract_column_names(value, alias_map))
        return cols
    if isinstance(node, ast.BinOp):
        return _extract_column_names(node.left, alias_map) | _extract_column_names(node.right, alias_map)
    if isinstance(node, ast.UnaryOp):
        return _extract_column_names(node.operand, alias_map)
    if isinstance(node, ast.Compare):
        cols = _extract_column_names(node.left, alias_map)
        for comparator in node.comparators:
            cols.update(_extract_column_names(comparator, alias_map))
        return cols
    if isinstance(node, ast.BoolOp):
        cols: Set[str] = set()
        for value in node.values:
            cols.update(_extract_column_names(value, alias_map))
        return cols
    return set()


def _subscript_key(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Index):  # pragma: no cover - py<3.9 compatibility
        return _subscript_key(node.value)
    return None


def _aggregation_names_from_call(node: ast.Call) -> Set[str]:
    names: Set[str] = set()
    for arg in node.args[:1]:
        names.update(_strings_from_node(arg))
    for kw in node.keywords:
        if kw.arg in {"func", "aggfunc"}:
            names.update(_strings_from_node(kw.value))
    return {name.lower() for name in names}


def _strings_from_node(node: ast.AST) -> Set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: Set[str] = set()
        for elt in node.elts:
            out.update(_strings_from_node(elt))
        return out
    if isinstance(node, ast.Dict):
        out: Set[str] = set()
        for key in node.keys:
            out.update(_strings_from_node(key))
        for value in node.values:
            out.update(_strings_from_node(value))
        return out
    return set()


def _call_has_zero(node: ast.Call) -> bool:
    args = list(node.args) + [kw.value for kw in node.keywords]
    for arg in args:
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, (int, float))
            and not isinstance(arg.value, bool)
            and arg.value == 0
        ):
            return True
    return False


def _string_literals(node: ast.Call) -> List[str]:
    out: List[str] = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append(arg.value)
    for kw in node.keywords:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            out.append(kw.value.value)
    return out


_LLM_CONCEPT_SCRIPT_CHAR_LIMIT = 64_000


def _concept_audit_script_excerpt(
    script_text: str,
    *,
    char_limit: int = _LLM_CONCEPT_SCRIPT_CHAR_LIMIT,
) -> str:
    """Cover the beginning, modeling body, and output contract of long scripts."""

    text = str(script_text or "")
    if len(text) <= char_limit:
        return text
    marker = "\n# ... concept-audit excerpt omitted ...\n"
    available = char_limit - (2 * len(marker))
    if available < 3:
        return text[:char_limit]
    head_size = available // 3
    middle_size = available // 3
    tail_size = available - head_size - middle_size
    middle_start = max(0, (len(text) - middle_size) // 2)
    return (
        text[:head_size]
        + marker
        + text[middle_start : middle_start + middle_size]
        + marker
        + text[-tail_size:]
    )


class LLMConceptAuditor:
    """Optional LLM-based semantic review after deterministic checks.

    Static rules remain authoritative and run first. This auditor is a
    conservative final sweep for issues that are hard to encode as
    regexes, such as confusing ICU vs hospital mortality or describing
    a missingness-driven stratum as clinically low risk.
    """

    name = "llm_concept_auditor"

    def __init__(self, llm: LLMClient, *, max_tokens: int = 1024) -> None:
        self.llm = llm
        self.max_tokens = int(max_tokens)

    def audit(
        self,
        *,
        context: ResearchContext,
        script_text: str,
        step: Optional[AnalysisStep] = None,
    ) -> List[ValidationFinding]:
        prompt = self._prompt(context=context, script_text=script_text, step=step)
        try:
            raw = self.llm.complete(
                [
                    LLMMessage(
                        role="system",
                        content=(
                            "You are a conservative ICU concept-use auditor. "
                            "Return only JSON. Do not invent findings."
                        ),
                    ),
                    LLMMessage(role="user", content=prompt),
                ],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            return [ValidationFinding(
                validator=self.name,
                severity="warning",
                message=f"LLM concept auditor failed: {exc}",
            )]
        findings = parse_llm_concept_audit_response(
            raw,
            validator=self.name,
            step_id=step.step_id if step else None,
        )
        findings = _downgrade_metadata_supported_outcome_findings(
            findings=findings,
            context=context,
            script_text=script_text,
        )
        findings = _downgrade_audit_only_companion_gating_findings(
            findings=findings,
            script_text=script_text,
        )
        return _downgrade_finalized_exposure_reconciliation_findings(
            findings=findings,
            context=context,
            script_text=script_text,
        )

    def _prompt(
        self,
        *,
        context: ResearchContext,
        script_text: str,
        step: Optional[AnalysisStep],
    ) -> str:
        # A wide ICU context can exceed the prompt budget.  Never take the
        # first columns blindly: preserve plan-declared inputs and their
        # structural companion family first, then variables actually referenced
        # by the script, and only then fill the remaining budget in context
        # order.  This is ordering-neutral and prevents a relevant late column
        # from losing its registered clinical role during concept review.
        companion_suffixes = (
            "_measured",
            "_first_time",
            "_last_time",
            "_first",
            "_max",
            "_min",
            "_mean",
            "_n",
        )

        def _family(name: str) -> str:
            lowered = str(name or "").strip().lower()
            for suffix in companion_suffixes:
                if lowered.endswith(suffix):
                    return lowered[: -len(suffix)]
            return lowered

        declared_inputs = {
            str(value or "").strip().lower()
            for value in ((step.inputs or []) if step is not None else [])
            if ":" not in str(value or "") and str(value or "").strip()
        }
        declared_families = {_family(value) for value in declared_inputs}
        direct_names = {
            value
            for value in (
                context.target_outcome,
                context.primary_exposure,
            )
            if value
        }
        priority_variables = [
            variable
            for variable in context.variables
            if variable.name.lower() in declared_inputs
            or _family(variable.name) in declared_families
            or variable.name in direct_names
        ]
        priority_names = {variable.name for variable in priority_variables}
        referenced_variables = [
            variable
            for variable in context.variables
            if variable.name not in priority_names and variable.name in script_text
        ]
        selected_names = priority_names | {
            variable.name for variable in referenced_variables
        }
        remaining_variables = [
            variable
            for variable in context.variables
            if variable.name not in selected_names
        ]
        selected_variables = (
            priority_variables + referenced_variables + remaining_variables
        )[:80]

        variables = [
            {
                "name": v.name,
                "description": v.description,
                "role": v.role.value,
                "source_concept": v.source_concept,
                "allowed_aggregations": [a.value for a in v.allowed_aggregations],
                "pitfalls": v.pitfalls,
                "clinical_caveats": v.clinical_caveats,
                "cross_database_notes": v.cross_database_notes,
                "analysis_window": v.analysis_window,
                "missingness": (
                    v.missingness.model_dump(mode="json")
                    if v.missingness is not None else None
                ),
            }
            for v in selected_variables
        ]
        return (
            "Review this generated analysis script for ICU concept-use risks "
            "that deterministic regex checks may miss. Focus only on: ordinal "
            "scores treated as continuous, silent missingness assumptions, "
            "alternate per-stay summaries (first/max/min/mean) that bypass "
            "numeric/domain validation or a fail-closed whole-step provenance "
            "audit of the same concept's measured/count/source-status "
            "companion consistency checks, "
            "PaO2/FiO2 or GCS/SOFA/KDIGO misuse, ICU vs hospital mortality "
            "confusion, and causal/clinical treatment claims in analysis code. "
            "Measurement-count columns are normally independent QC fields: do "
            "not require counts to define a physiological value, analytic "
            "exposure, or source-status mask when the script separately compares "
            "(count > 0) with the measured flag, reports discordance or "
            "unavailable count columns, and keeps that comparison audit-only. "
            "When that audit fails closed for the whole completed step, do not "
            "demand that individual physiological first/max/min/mean values be "
            "masked, filtered, or invalidated by measured/count companions. The "
            "value column's own missingness and numeric/domain rules determine its "
            "descriptive or modelling availability; the companions audit source "
            "provenance and must not change its row-level denominator. "
            "When `artifact:primary_exposure_definition` is a row-aligned "
            "finalized table, its exact Planner-selected binary column is the "
            "authoritative exposure. Validate its alignment, completeness, "
            "finiteness, and exact {0,1} domain, then use it directly. Raw "
            "count/measured/representative companions may be checked in a "
            "separate fail-closed provenance audit, but they must neither "
            "redefine nor overwrite the finalized exposure. Do not demand a "
            "second sparse-event reconciliation in this finalized-table branch; "
            "that reconciliation belongs to a raw-definition producer branch. "
            "There is one narrow sparse-event exception: an agent-planned binary "
            "event-presence exposure may use `<concept>_n > 0` when registered "
            "metadata identifies the base concept as an event/indicator rather "
            "than a continuous measurement, the count is complete finite "
            "non-negative integer-like, the complete binary measured flag agrees "
            "exactly with `(count > 0)`, and every non-missing representative "
            "event value is binary positive only on exactly those rows. A "
            "positive-only event export may leave the representative value "
            "structurally missing on reconciled negative rows; do not require "
            "an explicit zero there. Still require every reconciled positive "
            "row to carry the representative positive value, and reject any "
            "non-binary value or positive value on a reconciled negative row. "
            "In that "
            "encoding, count-zero/flag-zero rows are the reconciled negative "
            "class, not unmeasured rows; the script must retain them and report "
            "`indicator_semantics='binary_event_presence'` plus reconciliation "
            "counts. Flag as an error any script that keeps only `measured == 1` "
            "or `count > 0`, accepts an incomplete or discordant triad, applies "
            "this exception to a continuous measurement, or silently accepts "
            "contradictions. The standard method-suite helper "
            "`easyicu.research_agent.methods.source_status."
            "reconcile_binary_event_presence` enforces this triad without "
            "choosing the scientific columns; do not flag a direct use of its "
            "returned values/audit/status_table unless the script later mutates "
            "or bypasses them. "
            "Use severity='error' when a result-bearing script can silently "
            "invent numeric zeros for absent categories, render percentages "
            "without reconciling them to counts and denominators, or select an "
            "alternate per-stay summary in place of the authoritative exposure; "
            "those behaviors can change the displayed scientific result. "
            "If a generic outcome column such as 'death' is explicitly bound in "
            "the variable metadata to ICU mortality, hospital mortality or a "
            "fixed follow-up horizon, do not raise an error unless the script "
            "contradicts that binding or mixes incompatible outcome definitions.\n\n"
            "A named `full_stay` window is an administrative analysis span: it "
            "starts at ICU admission and ends at discharge, with `end_hours` "
            "serving only as an upper safety cap (the default cap is 720 hours). "
            "Copying that planner-locked window into provenance does not turn a "
            "metadata-bound ICU/hospital mortality flag into 30-day mortality. "
            "Call it a fixed-horizon outcome only when the script actually labels "
            "or constructs 28/30-day mortality, uses another mortality column, or "
            "derives the event from event-time/follow-up data.\n\n"
            "Return JSON only: "
            '{"findings":[{"severity":"info|warning|error",'
            '"message":"short finding","detail":{"optional":"context"}}]}. '
            "Use an empty findings list if no issue is visible.\n\n"
            f"Step: {step.step_id if step else '(unknown)'}\n"
            f"Step intent: {step.intent if step else '(unknown)'}\n"
            f"Target outcome: {context.target_outcome}\n"
            "Named time windows:\n"
            + json.dumps(
                [window.model_dump(mode="json") for window in context.time_windows],
                ensure_ascii=False,
                default=str,
            )
            + "\n"
            "Variables:\n"
            + json.dumps(variables, ensure_ascii=False, default=str)
            + "\n\nScript:\n"
            + _concept_audit_script_excerpt(script_text)
        )


def parse_llm_concept_audit_response(
    raw: str,
    *,
    validator: str = "llm_concept_auditor",
    step_id: Optional[str] = None,
) -> List[ValidationFinding]:
    text = _strip_jsonish(raw)
    try:
        payload = json.loads(text)
    except Exception:
        head = (raw or "").strip().replace("\n", " ")[:300]
        return [ValidationFinding(
            validator=validator,
            severity="warning",
            message=f"LLM concept auditor returned unparsable output: {head}",
            detail={"step_id": step_id} if step_id else None,
        )]
    items = payload.get("findings", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    findings: List[ValidationFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        msg = str(item.get("message") or "").strip()
        if not msg:
            continue
        sev = str(item.get("severity") or "warning").lower()
        if sev not in {"info", "warning", "error"}:
            sev = "warning"
        detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
        if step_id:
            detail = dict(detail)
            detail.setdefault("step_id", step_id)
        if _llm_outcome_confusion_is_nonblocking(msg, detail):
            sev = "warning"
        findings.append(ValidationFinding(
            validator=validator,
            severity=sev,  # type: ignore[arg-type]
            message=msg,
            detail=detail or None,
        ))
    return findings


def _llm_outcome_confusion_is_nonblocking(
    message: str,
    detail: Dict[str, Any],
) -> bool:
    lowered_message = (message or "").lower()
    if "icu vs hospital mortality confusion" not in lowered_message:
        return False
    detail_text = json.dumps(detail or {}, ensure_ascii=False).lower()
    soft_signals = (
        "explicitly noted",
        "explicitly treated as",
        "does not verify or enforce consistent usage",
        "possible",
    )
    return any(token in detail_text for token in soft_signals)


def _downgrade_metadata_supported_outcome_findings(
    *,
    findings: Sequence[ValidationFinding],
    context: ResearchContext,
    script_text: str,
) -> List[ValidationFinding]:
    """Prevent optional LLM audit false positives from blocking clear outcomes.

    The deterministic context builder may bind a generic column such as
    ``death`` to ICU, hospital, or fixed-horizon mortality based on the
    research question. Some small auditor models still flag "death is
    ambiguous" even when the script only uses the bound target column.
    In that case the finding remains visible as a warning, but it should
    not block execution. If the script actually mixes outcome definitions,
    the error is preserved.
    """

    outcome = context.target_outcome
    if not outcome:
        return list(findings)
    descriptor = context.variable(outcome)
    if descriptor is None or not descriptor.source_concept:
        return list(findings)
    if str(getattr(descriptor.role, "value", descriptor.role)) != "outcome":
        return list(findings)
    source = descriptor.source_concept.lower()
    if source not in {
        "icu_mortality",
        "hospital_mortality",
        "mortality_28d",
        "mortality_30d",
    }:
        return list(findings)

    if not _script_uses_bound_outcome(script_text=script_text, outcome=outcome):
        return list(findings)
    if _script_has_conflicting_mortality_semantics(
        script_text=script_text,
        outcome=outcome,
        source=source,
    ):
        return list(findings)
    copies_full_stay = source in {"icu_mortality", "hospital_mortality"} and (
        _script_copies_named_full_stay_window(
            context=context,
            script_text=script_text,
            outcome=outcome,
        )
    )

    ambiguity_tokens = (
        "icu vs hospital mortality confusion",
        "mortality confusion",
        "outcome variable",
        "death is ambiguous",
        "without clarifying whether",
        "does not specify whether",
        "lacks explicit clarification",
    )
    downgraded: List[ValidationFinding] = []
    for finding in findings:
        if finding.validator == LLMConceptAuditor.name and finding.severity == "error":
            text = " ".join(
                [
                    finding.message or "",
                    json.dumps(finding.detail or {}, ensure_ascii=False, default=str),
                ]
            ).lower()
            ambiguity = any(token in text for token in ambiguity_tokens)
            horizon_mismatch = (
                copies_full_stay
                and _finding_claims_mortality_horizon_mismatch(text)
            )
            if ambiguity or horizon_mismatch:
                detail = dict(finding.detail or {})
                detail.setdefault(
                    "downgraded_reason",
                    (
                        f"Target outcome '{outcome}' is bound to "
                        f"{descriptor.source_concept} in ResearchContext and "
                        + (
                            "the script only copies the named full_stay "
                            "administrative window without constructing a "
                            "fixed-horizon mortality endpoint."
                            if horizon_mismatch
                            else "the script does not reference a conflicting "
                            "mortality definition."
                        )
                    ),
                )
                downgraded.append(
                    finding.model_copy(
                        update={"severity": "warning", "detail": detail}
                    )
                )
                continue
        downgraded.append(finding)
    return downgraded


def _downgrade_audit_only_companion_gating_findings(
    *,
    findings: Sequence[ValidationFinding],
    script_text: str,
) -> List[ValidationFinding]:
    """Keep the optional semantic auditor from reintroducing value gating.

    A script that records the canonical measured/count comparison and fails the
    whole completed step on invalid or discordant provenance must not then mask
    physiological values row by row with those companion fields.  Some auditor
    models incorrectly demand exactly that.  Preserve genuine errors when the
    whole-step audit contract is absent.
    """

    derived_provenance_flags: set[str] = set()

    def _failure_guard(test: ast.AST) -> bool:
        if isinstance(test, ast.Name):
            return test.id.lower() in derived_provenance_flags & {
                "provenance_failed",
                "provenance_error",
            }
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return (
                isinstance(test.operand, ast.Name)
                and test.operand.id.lower() in derived_provenance_flags
                and test.operand.id.lower() == "provenance_valid"
            )
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            return any(_failure_guard(value) for value in test.values)
        if isinstance(test, ast.Compare) and len(test.ops) == len(test.comparators) == 1:
            left = test.left
            right = test.comparators[0]
            if not isinstance(left, ast.Name) or not isinstance(right, ast.Constant):
                return False
            name = left.id.lower()
            value = right.value
            if name not in derived_provenance_flags:
                return False
            if name == "provenance_valid":
                return isinstance(test.ops[0], (ast.Eq, ast.Is)) and value is False
            if name in {"provenance_failed", "provenance_error"}:
                return isinstance(test.ops[0], (ast.Eq, ast.Is)) and value is True
        return False

    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        tree = None
    ast_tokens = set()
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                ast_tokens.add(node.id.lower())
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                ast_tokens.add(node.value.lower())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            value_tokens = {
                str(candidate.value).lower()
                for candidate in ast.walk(value)
                if isinstance(candidate, ast.Constant)
                and isinstance(candidate.value, str)
            } | {
                candidate.id.lower()
                for candidate in ast.walk(value)
                if isinstance(candidate, ast.Name)
            }
            if not value_tokens & {"invalid_pair_n", "discordant_n"}:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            derived_provenance_flags.update(
                target.id.lower() for target in targets if isinstance(target, ast.Name)
            )
    contract_tokens = {
        "measurement_provenance_audit",
        "invalid_pair_n",
        "discordant_n",
        "audit_only",
    }
    fail_closed_guard = tree is not None and any(
        isinstance(node, ast.If)
        and _failure_guard(node.test)
        and any(isinstance(statement, ast.Raise) for statement in node.body)
        for node in ast.walk(tree)
    )
    audit_contract_present = contract_tokens.issubset(ast_tokens) and fail_closed_guard
    if not audit_contract_present:
        return list(findings)

    false_positive_signals = (
        "do not mask or invalidate modeled",
        "not mask or invalidate modeled",
        "not used to mask",
        "without using the measured flag to mask",
        "based largely on value non-missingness",
        "bypass their measured/source-status consistency checks",
    )
    downgraded: List[ValidationFinding] = []
    for finding in findings:
        if finding.validator == LLMConceptAuditor.name and finding.severity == "error":
            text = " ".join(
                [
                    finding.message or "",
                    json.dumps(finding.detail or {}, ensure_ascii=False, default=str),
                ]
            ).lower()
            companion_context = "measured" in text and any(
                token in text
                for token in ("first-value", "first value", "covariate", "summary")
            )
            if companion_context and any(
                token in text for token in false_positive_signals
            ):
                detail = dict(finding.detail or {})
                detail.setdefault(
                    "downgraded_reason",
                    "The script records the canonical audit-only measured/count "
                    "comparison and fails the whole completed step on invalid or "
                    "discordant provenance. Companion fields must not gate "
                    "row-level physiological values.",
                )
                downgraded.append(
                    finding.model_copy(
                        update={"severity": "warning", "detail": detail}
                    )
                )
                continue
        downgraded.append(finding)
    return downgraded


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _finalized_branch_isolates_reconciliation(script_text: str) -> bool:
    """Prove that raw-event reconciliation is unreachable on a DataFrame path.

    Generated consumers may support two typed forms: a finalized row-aligned
    DataFrame and a raw definition mapping.  Merely defining a raw resolver that
    calls ``reconcile_binary_event_presence`` does not overwrite the finalized
    values.  This helper returns true only when an ``isinstance(...,
    DataFrame)`` branch keeps every reconciliation call (including a one-hop
    local resolver) in the opposite branch and makes no such call afterwards.
    Ambiguous control flow remains fail-closed.
    """

    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        return False

    reconciliation_name = "reconcile_binary_event_presence"
    wrapper_names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(child, ast.Call)
            and _call_name(child) == reconciliation_name
            for child in ast.walk(node)
        ):
            wrapper_names.add(node.name)

    guarded_call_names = wrapper_names | {reconciliation_name}

    class _ExecutableCallVisitor(ast.NodeVisitor):
        def __init__(self, *, skip: ast.AST | None = None) -> None:
            self.skip = skip
            self.calls: List[str] = []

        def visit(self, node: ast.AST) -> Any:
            if node is self.skip:
                return None
            return super().visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

        def visit_Call(self, node: ast.Call) -> None:
            name = _call_name(node)
            if name in guarded_call_names:
                self.calls.append(name)
            self.generic_visit(node)

    def _calls_in(nodes: Sequence[ast.stmt]) -> List[str]:
        visitor = _ExecutableCallVisitor()
        for item in nodes:
            visitor.visit(item)
        return visitor.calls

    parent: Dict[ast.AST, ast.AST] = {}
    for container in ast.walk(tree):
        for child in ast.iter_child_nodes(container):
            parent[child] = container

    for candidate in ast.walk(tree):
        if not isinstance(candidate, ast.If):
            continue
        test = candidate.test
        finalized_body = candidate.body
        raw_body = candidate.orelse
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test = test.operand
            finalized_body, raw_body = raw_body, finalized_body
        if not (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "isinstance"
            and len(test.args) >= 2
            and (
                (isinstance(test.args[1], ast.Name) and test.args[1].id == "DataFrame")
                or (
                    isinstance(test.args[1], ast.Attribute)
                    and test.args[1].attr == "DataFrame"
                )
            )
        ):
            continue
        if _calls_in(finalized_body) or not _calls_in(raw_body):
            continue

        scope: ast.AST = tree
        cursor: ast.AST | None = candidate
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = cursor
                break
            cursor = parent.get(cursor)
        outside = _ExecutableCallVisitor(skip=candidate)
        for statement in getattr(scope, "body", []):
            outside.visit(statement)
        if not outside.calls:
            return True
    return False


def _downgrade_finalized_exposure_reconciliation_findings(
    *,
    findings: Sequence[ValidationFinding],
    context: ResearchContext,
    script_text: str,
) -> List[ValidationFinding]:
    """Do not make consumers rederive a finalized binary exposure.

    A typed row-aligned exposure table has already passed its producer gate. Its
    exact Planner-selected column remains authoritative at downstream steps;
    raw sparse-event companions may audit provenance but cannot redefine it.
    Genuine findings that the script discards or overwrites the finalized values
    remain blocking.
    """

    primary_exposure = str(context.primary_exposure or "").strip()
    script = str(script_text or "")
    normalized_script = re.sub(r"\s+", "", script.lower())
    reconciliation_isolated = _finalized_branch_isolates_reconciliation(script)
    literal_direct_binding = bool(
        primary_exposure
        and "artifact:primary_exposure_definition" in script
        and "dataframe" in script.lower()
        and re.search(
            rf"\[\s*['\"]{re.escape(primary_exposure)}['\"]\s*\]",
            script,
        )
        and ".isin([0,1])" in normalized_script
        and ("isfinite(" in normalized_script or ".notna()" in normalized_script)
    )
    contracted_direct_binding = bool(
        reconciliation_isolated
        and "artifact:primary_exposure_definition" in script
        and "dataframe" in script.lower()
        and "product_contract" in script
        and "executable_column" in script
        and "resolve_finalized_exposure" in script
        and ".isin([0,1])" in normalized_script
        and ("isfinite(" in normalized_script or ".notna()" in normalized_script)
    )
    direct_binding = literal_direct_binding or contracted_direct_binding
    if not direct_binding:
        return list(findings)

    missing_reconciliation_signals = (
        "bypasses the required binary-event triad reconciliation",
        "bypasses binary-event triad reconciliation",
        "bypasses the binary-event triad reconciliation",
        "does not call reconcile_binary_event_presence",
        "does not invoke reconcile_binary_event_presence",
        "without the required companion-consistency audit",
    )
    finalized_override_signals = (
        "ignores its values",
        "ignores the finalized",
        "overwrites",
        "overwritten",
        "discards",
        "replaces the finalized",
        "replaces treatment",
        "replaces exposure",
        "instead of the finalized",
        "raw companions determine",
    )
    downgraded: List[ValidationFinding] = []
    for finding in findings:
        if finding.validator == LLMConceptAuditor.name and finding.severity == "error":
            text = " ".join(
                [
                    finding.message or "",
                    json.dumps(finding.detail or {}, ensure_ascii=False, default=str),
                ]
            ).lower()
            complains_only_about_reconciliation = any(
                signal in text for signal in missing_reconciliation_signals
            ) and not any(signal in text for signal in finalized_override_signals)
            false_override_claim = (
                reconciliation_isolated
                and "reconcile_binary_event_presence" in text
                and any(signal in text for signal in finalized_override_signals)
            )
            if complains_only_about_reconciliation or false_override_claim:
                detail = dict(finding.detail or {})
                detail.setdefault(
                    "downgraded_reason",
                    (
                        "AST control-flow verification shows raw-event "
                        "reconciliation is isolated to the non-DataFrame branch; "
                        "the finalized branch directly binds and validates the "
                        "exact binary column from the row-aligned exposure "
                        "artifact."
                        if false_override_claim
                        else "The script directly binds and validates the exact "
                        "binary column from the finalized row-aligned exposure "
                        "artifact. Downstream raw-event reconciliation may audit "
                        "provenance but must not redefine that authoritative "
                        "exposure."
                    ),
                )
                downgraded.append(
                    finding.model_copy(
                        update={"severity": "warning", "detail": detail}
                    )
                )
                continue
        downgraded.append(finding)
    return downgraded


def _strip_jsonish(text: str) -> str:
    text = (text or "").strip()
    if "```" not in text:
        return text
    start = text.find("```")
    rest = text[start + 3:]
    nl = rest.find("\n")
    if nl >= 0:
        tag = rest[:nl].strip().lower()
        if tag in {"json", "js", "javascript"} or not tag:
            rest = rest[nl + 1:]
    end = rest.find("```")
    if end >= 0:
        rest = rest[:end]
    return rest.strip()


# ---------------------------------------------------------------------------
# Cross-step validators
# ---------------------------------------------------------------------------


class CrossStepCohortLockValidator:
    """Prevent a fixed-cohort step from silently re-filtering prior rows.

    Some reconciliation steps explicitly promise to keep the completed
    analytic cohort fixed.  A generated script must then operate on the
    already-materialised cohort rather than add a new eligibility rule.  This
    gate activates only for that explicit intent and compares a
    machine-readable current cohort count with the most recent successful
    analysis-step count.  ``n_universe`` is intentionally not accepted as the
    current count because it can remain unchanged while ``n_final_cohort`` is
    silently reduced.
    """

    name = "cross_step_cohort_lock"

    _SUCCESSFUL_STATUSES = {
        "ok",
        "complete",
        "completed",
        "repaired",
        "runner_repaired",
    }
    _COUNT_PATHS: tuple[tuple[str, ...], ...] = (
        ("n_final_cohort",),
        ("final_analytic_cohort_n",),
        ("final_cohort_n",),
        ("cohort_count_final",),
        ("cohort", "n_final_rows"),
        ("cohort", "final_analytic_cohort_n"),
        ("locked_cohort", "n_output"),
        ("locked_cohort", "n_final_rows"),
        ("analytic_cohort_n",),
        ("adult_analytic_cohort_n",),
        ("adult_cohort_n",),
        ("cohort_definition", "adult_analytic_cohort_n"),
        ("cohort_definition", "analytic_cohort_n"),
        ("cohort_definition", "adult_cohort_n"),
        ("cohort_counts", "n_adult_analysis_cohort_rows"),
        ("cohort_counts", "n_analytic_cohort_rows"),
        ("cohort_n",),
        ("n_cohort",),
        ("n_total",),
        # The deterministic probe uses ``n_rows`` and is a final fallback when
        # no later analysis summary exposes a cohort count.
        ("n_rows",),
    )

    @staticmethod
    def _as_count(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not pd.notna(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    @classmethod
    def _extract_count(
        cls, summary: Dict[str, Any]
    ) -> Optional[tuple[int, str]]:
        for path in cls._COUNT_PATHS:
            value: Any = summary
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    break
                value = value[key]
            else:
                count = cls._as_count(value)
                if count is not None:
                    return count, ".".join(path)
        return None

    @staticmethod
    def _requires_fixed_cohort(step: AnalysisStep) -> bool:
        text = re.sub(r"\s+", " ", str(step.intent or "").strip().lower())
        if not text:
            return False

        # A true alternative-cohort/sensitivity step is allowed to change N.
        # Do not infer this from a generated summary's analysis_family: the M1
        # failure that motivated the gate mislabeled a fixed reconciliation as
        # cohort-definition sensitivity.
        varying_cohort = any(
            re.search(pattern, text)
            for pattern in (
                r"\balternative cohort(?: definition)?\b",
                r"\bvary(?:ing)? (?:the )?cohort(?: definition)?\b",
                r"\bcohort(?: definition)? sensitivity\b",
                r"\bcompare (?:an |the )?alternative eligibility\b",
            )
        )
        if varying_cohort:
            return False

        return any(
            re.search(pattern, text)
            for pattern in (
                r"\bkeep(?:ing)?\b.{0,160}\bcohort\b.{0,160}\b(?:fixed|unchanged|constant)\b",
                r"\b(?:preserve|preserving|hold|holding)\b.{0,120}\bcohort\b.{0,120}\b(?:fixed|unchanged|constant)\b",
                r"\b(?:fixed|locked|unchanged)\b.{0,80}\b(?:completed|current|existing|analytic|analysis)?\s*cohort\b",
                r"\b(?:completed|current|existing|analytic|analysis)?\s*cohort\b.{0,80}\b(?:fixed|locked|unchanged)\b",
                r"\b(?:do not|don't|must not|without)\b.{0,100}\b(?:redefine|change|refilter|restrict)\w*\b.{0,100}\b(?:the )?cohort\b",
            )
        )

    @classmethod
    def _latest_prior_lock(
        cls, completed_step_records: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for record in reversed(completed_step_records):
            step_id = str(record.get("step_id") or "prior_step")
            normalised_step_id = re.sub(
                r"[^a-z0-9]+", "_", step_id.strip().lower()
            ).strip("_")
            if normalised_step_id.endswith("_figure"):
                continue

            record_status = str(record.get("status") or "").strip().lower()
            if record_status and record_status not in cls._SUCCESSFUL_STATUSES:
                continue
            summary = record.get("step_summary")
            if not isinstance(summary, dict) or summary.get("rendering_only") is True:
                continue
            summary_status = str(summary.get("status") or "").strip().lower()
            if summary_status and summary_status not in cls._SUCCESSFUL_STATUSES:
                continue
            extracted = cls._extract_count(summary)
            if extracted is None:
                continue
            count, path = extracted
            return {"cohort_n": count, "summary_path": path, "step_id": step_id}
        return None

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[ValidationFinding]:
        normalised_step_id = re.sub(
            r"[^a-z0-9]+", "_", str(step.step_id or "").strip().lower()
        ).strip("_")
        if step_summary.get("rendering_only") is True or normalised_step_id.endswith(
            "_figure"
        ):
            # A split figure step reads registered parent outputs and cannot
            # redefine eligibility. Requiring it to restate the analytic N
            # turns the stock "do not redefine the cohort" rendering prompt
            # into a false cohort-drift error and sends valid figures through
            # an irrelevant model-code repair loop.
            return []
        if not self._requires_fixed_cohort(step):
            return []
        prior = self._latest_prior_lock(completed_step_records)
        if prior is None:
            return []

        current = self._extract_count(step_summary)
        if current is None:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Fixed-cohort step {step.step_id} does not report a "
                        "machine-readable final analytic cohort count. Report "
                        f"the unchanged cohort N locked by completed step "
                        f"{prior['step_id']} ({prior['cohort_n']}) and do not "
                        "re-derive eligibility inside this step."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "expected_cohort_n": prior["cohort_n"],
                        "expected_from_step": prior["step_id"],
                        "expected_summary_path": prior["summary_path"],
                        "reported_summary_path": None,
                    },
                )
            ]

        reported_n, reported_path = current
        if reported_n == prior["cohort_n"]:
            return []
        return [
            ValidationFinding(
                validator=self.name,
                severity="error",
                message=(
                    f"Fixed-cohort drift for step {step.step_id}: "
                    f"{reported_path} reports {reported_n}, but completed step "
                    f"{prior['step_id']} locked {prior['cohort_n']}. Treat the "
                    "input cohort as already eligible; remove any new age, "
                    "length-of-stay, identifier, outcome-availability, or other "
                    "row filter and recompute this step on the locked cohort."
                ),
                detail={
                    "step_id": step.step_id,
                    "reported_cohort_n": reported_n,
                    "reported_summary_path": reported_path,
                    "expected_cohort_n": prior["cohort_n"],
                    "expected_from_step": prior["step_id"],
                    "expected_summary_path": prior["summary_path"],
                },
            )
        ]


class CrossStepRegisteredOutputValidator:
    """Reject a false "upstream table unavailable" reconciliation gap.

    Generated reconciliation code sometimes finds the correct upstream step
    but over-filters its evidence records by a guessed semantic filename and
    therefore declares an existing registered table unavailable.  This gate
    compares that explicit availability claim with the upstream step record's
    table evidence and machine-readable output-file declarations.  Genuine
    gaps remain allowed when the completed upstream step registered no table.
    """

    name = "cross_step_registered_output"
    _SUCCESSFUL_STATUSES = CrossStepCohortLockValidator._SUCCESSFUL_STATUSES
    _TABLE_SUFFIXES = (".csv", ".parquet", ".tsv", ".feather", ".xlsx")

    @classmethod
    def _availability_blocks(
        cls, summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                upstream_step = value.get("upstream_step")
                availability_key = next(
                    (
                        key
                        for key in (
                            "source_table_available",
                            "registered_output_readable",
                            "available",
                        )
                        if isinstance(value.get(key), bool)
                    ),
                    None,
                )
                if isinstance(upstream_step, str) and availability_key is not None:
                    blocks.append(
                        {
                            "upstream_step": upstream_step,
                            "available": value[availability_key],
                            "availability_key": availability_key,
                            "path": ".".join(path) or "step_summary",
                            "reported_path": value.get("source_table_path")
                            or value.get("registered_output_path")
                            or value.get("path"),
                        }
                    )
                for key, child in value.items():
                    visit(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return blocks

    @classmethod
    def _table_artifacts(cls, record: Dict[str, Any]) -> List[str]:
        artifacts: List[str] = []
        for evidence_id in record.get("evidence_ids") or []:
            if isinstance(evidence_id, str) and evidence_id.startswith("table_"):
                artifacts.append(evidence_id)

        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            return sorted(set(artifacts))

        def collect(value: Any) -> None:
            if isinstance(value, str):
                if value.strip().lower().endswith(cls._TABLE_SUFFIXES):
                    artifacts.append(value.strip())
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        for key in ("output_files", "outputs"):
            if key in summary:
                collect(summary[key])
        return sorted(set(artifacts))

    @classmethod
    def _upstream_table_lock(
        cls,
        upstream_step: str,
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for record in reversed(completed_step_records):
            if str(record.get("step_id") or "") != upstream_step:
                continue
            record_status = str(record.get("status") or "").strip().lower()
            if record_status and record_status not in cls._SUCCESSFUL_STATUSES:
                continue
            summary = record.get("step_summary")
            if isinstance(summary, dict):
                summary_status = str(summary.get("status") or "").strip().lower()
                if summary_status and summary_status not in cls._SUCCESSFUL_STATUSES:
                    continue
            artifacts = cls._table_artifacts(record)
            if artifacts:
                return {"step_id": upstream_step, "table_artifacts": artifacts}
            return None
        return None

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        for block in self._availability_blocks(step_summary):
            if block["available"]:
                continue
            prior = self._upstream_table_lock(
                block["upstream_step"], completed_step_records
            )
            if prior is None:
                continue
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Registered upstream table was falsely reported "
                        f"unavailable in step {step.step_id}: completed step "
                        f"{prior['step_id']} registered table evidence "
                        f"{prior['table_artifacts']}. Filter manifest records "
                        "by the exact produced_by_step and table kind, resolve "
                        "relative_path from the run directory, and use the sole "
                        "compatible table even when its filename does not repeat "
                        "the current step's semantic label."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "summary_path": block["path"],
                        "availability_key": block["availability_key"],
                        "reported_path": block["reported_path"],
                        "upstream_step": prior["step_id"],
                        "registered_table_artifacts": prior["table_artifacts"],
                    },
                )
            )
        return findings


class StepSummaryFractionValidator:
    """Enforce [0, 1] for probability-like machine-summary fields."""

    name = "step_summary_fraction_scale"

    @classmethod
    def _invalid_fraction_values(
        cls, summary: Dict[str, Any]
    ) -> List[tuple[str, float, str]]:
        invalid: List[tuple[str, float, str]] = []

        def normalise_key(value: Any) -> str:
            return re.sub(
                r"[^a-z0-9]+", "_", str(value).strip().lower()
            ).strip("_")

        effect_scale_names = {
            "hr",
            "or",
            "rd",
            "rr",
            "risk_ratio",
            "relative_risk",
            "odds_ratio",
            "hazard_ratio",
            "risk_difference",
        }

        def is_effect_scale_field(key: Any) -> bool:
            name = normalise_key(key)
            ci_base = re.sub(r"_(?:ci_)?(?:low|high|lower|upper)$", "", name)
            return ci_base in effect_scale_names or any(
                ci_base.endswith(f"_{effect_scale}")
                for effect_scale in effect_scale_names
            )

        def bounded_field_kind(key: Any) -> Optional[str]:
            """Identify fields whose *value* is contractually in [0, 1].

            Do not propagate merely because a structural or methodological key
            contains the substring ``fraction``.  Names such as
            ``fractional_polynomial_power`` and
            ``sampling_fraction_denominator`` are not values on a [0, 1]
            scale.  A mapping directly owned by a true ``*_fraction`` field is
            still allowed to encode category -> fraction values.
            """

            name = normalise_key(key)
            if not name or any(
                token in name for token in ("pct", "percent", "percentage")
            ):
                return None
            if name.startswith("fractional_") or name == "fractional":
                return None
            if name.endswith(("_numerator", "_denominator")):
                return None
            ci_base = re.sub(r"_(?:ci_)?(?:low|high|lower|upper)$", "", name)
            if is_effect_scale_field(key):
                return None
            if ci_base == "at_risk" or ci_base.endswith("_at_risk"):
                # Survival risk-set counts/statuses are not probabilities.
                return None
            if ci_base in {
                "attributable_fraction",
                "population_attributable_fraction",
            }:
                # These effect measures can legitimately be negative.
                return None
            if ci_base == "fraction" or ci_base.endswith("_fraction"):
                return "fraction"
            if ci_base == "probability" or ci_base.endswith("_probability"):
                return "probability"
            if ci_base == "prevalence" or ci_base.endswith("_prevalence"):
                return "prevalence"
            if ci_base.startswith("prevalence_ci"):
                return "prevalence"
            if ci_base == "risk" or ci_base.endswith("_risk"):
                if ci_base in {"excess_risk", "attributable_risk"} or ci_base.endswith(
                    ("_excess_risk", "_attributable_risk")
                ):
                    return None
                return "risk"
            return None

        structural_children = {
            "count",
            "cases",
            "deaths",
            "denominator",
            "event_n",
            "events",
            "n",
            "nobs",
            "non_events",
            "numerator",
            "observations",
            "patients",
            "sample_size",
            "stays",
            "subjects",
            "survivors",
            "total",
            "total_n",
        }
        structural_suffixes = (
            "_count",
            "_denominator",
            "_draws",
            "_folds",
            "_iterations",
            "_n",
            "_numerator",
            "_replicates",
            "_sample_size",
        )
        structural_prefixes = ("n_", "num_", "number_")
        coordinate_children = {
            "category",
            "category_code",
            "code",
            "cutpoint",
            "decimal_places",
            "df",
            "digits",
            "group",
            "group_id",
            "id",
            "index",
            "label",
            "level",
            "level_id",
            "name",
            "order",
            "precision",
            "rank",
            "random_seed",
            "seed",
            "stratum",
            "stratum_id",
            "threshold",
            "timepoint",
            "timepoint_index",
            "version",
        }
        coordinate_suffixes = (
            "_category",
            "_code",
            "_cutpoint",
            "_days",
            "_places",
            "_group",
            "_hours",
            "_id",
            "_index",
            "_label",
            "_level",
            "_minutes",
            "_months",
            "_name",
            "_order",
            "_precision",
            "_rank",
            "_seconds",
            "_seed",
            "_stratum",
            "_threshold",
            "_timepoint",
            "_version",
            "_years",
        )
        scalar_value_children = {
            "estimate",
            "fraction",
            "point_estimate",
            "result",
            "value",
        }
        generic_ci_children = {"ci_low", "ci_high", "ci_lower", "ci_upper"}
        scale_descriptor_names = {
            "effect_measure",
            "estimand",
            "measure",
            "measure_type",
            "metric",
            "metric_name",
            "scale",
            "statistic",
            "type",
            "unit",
            "units",
        }
        bounded_scale_descriptors = {
            "0_1",
            "dimensionless",
            "proportion",
            "unit_interval",
            "unitless",
            "zero_to_one",
        }
        non_bounded_scale_descriptors = {
            "aic",
            "attributable_fraction",
            "attributable_risk",
            "auc",
            "beta",
            "bic",
            "c_statistic",
            "coefficient",
            "count",
            "counts",
            "deviance",
            "excess_risk",
            "frequency",
            "hazard_ratio",
            "hr",
            "iqr",
            "log_likelihood",
            "log_odds",
            "logit",
            "mae",
            "mean",
            "median",
            "mse",
            "n",
            "odds_ratio",
            "or",
            "pct",
            "percent",
            "percentage",
            "population_attributable_fraction",
            "rd",
            "relative_risk",
            "risk_difference",
            "risk_ratio",
            "rmse",
            "rr",
            "sample_size",
            "sd",
            "se",
            "standard_deviation",
            "standard_error",
            "variance",
        }
        domain_changing_tokens = {
            "audit",
            "audits",
            "bootstrap",
            "bootstraps",
            "coefficient",
            "coefficients",
            "count",
            "counts",
            "diagnostic",
            "diagnostics",
            "distribution",
            "distributions",
            "draw",
            "draws",
            "effect",
            "effects",
            "fit",
            "fits",
            "format",
            "formats",
            "formatting",
            "fold",
            "folds",
            "iteration",
            "iterations",
            "metadata",
            "option",
            "options",
            "parameter",
            "parameters",
            "percentile",
            "percentiles",
            "quantile",
            "quantiles",
            "replicate",
            "replicates",
            "rounding",
            "runtime",
            "sample",
            "samples",
            "size",
            "sizes",
            "setting",
            "settings",
            "statistic",
            "statistics",
            "timing",
        }

        def is_structural_child(name: str) -> bool:
            return (
                name in structural_children
                or name.startswith(structural_prefixes)
                or name.endswith(structural_suffixes)
            )

        def blocks_inherited_context(key: Any, name: str) -> bool:
            ci_base = re.sub(r"_(?:ci_)?(?:low|high|lower|upper)$", "", name)
            name_tokens = set(name.split("_"))
            return (
                is_structural_child(name)
                or name in coordinate_children
                or (
                    name.endswith(coordinate_suffixes)
                    and not name.startswith("by_")
                )
                or name in non_bounded_scale_descriptors
                or bool(name_tokens & domain_changing_tokens)
                or any(
                    token in name for token in ("pct", "percent", "percentage")
                )
                or name.startswith("fractional_")
                or is_effect_scale_field(key)
                or ci_base == "at_risk"
                or ci_base.endswith("_at_risk")
                or ci_base
                in {
                    "attributable_fraction",
                    "attributable_risk",
                    "excess_risk",
                    "population_attributable_fraction",
                }
            )

        def is_scale_descriptor_name(name: str) -> bool:
            return name in scale_descriptor_names or name.endswith(
                ("_measure", "_metric", "_scale", "_type", "_unit", "_units")
            )

        def mapping_declares_non_bounded_scale(value: Any) -> bool:
            if not isinstance(value, dict):
                return False
            for key, descriptor in value.items():
                if not is_scale_descriptor_name(normalise_key(key)) or not isinstance(
                    descriptor, str
                ):
                    continue
                if descriptor.strip() == "%":
                    return True
                descriptor_name = normalise_key(descriptor)
                if not descriptor_name:
                    continue
                if (
                    bounded_field_kind(descriptor_name) is not None
                    or descriptor_name in bounded_scale_descriptors
                ):
                    continue
                if (
                    descriptor_name in non_bounded_scale_descriptors
                    or is_effect_scale_field(descriptor_name)
                ):
                    return True
            return False

        def mapping_has_generic_metric_payload(value: Any) -> bool:
            return isinstance(value, dict) and any(
                normalise_key(key) in scalar_value_children | generic_ci_children
                for key in value
            )

        def visit(
            value: Any,
            path: tuple[str, ...] = (),
            bounded_context: Optional[str] = None,
        ) -> None:
            if isinstance(value, dict):
                local_non_bounded_scale = bool(
                    bounded_context
                    and mapping_has_generic_metric_payload(value)
                    and mapping_declares_non_bounded_scale(value)
                )
                sibling_kinds = {
                    kind
                    for key in value
                    if (kind := bounded_field_kind(key)) is not None
                }
                sibling_context = (
                    next(iter(sibling_kinds)) if len(sibling_kinds) == 1 else None
                )
                has_effect_scale_sibling = any(
                    is_effect_scale_field(key) for key in value
                )
                for key, child in value.items():
                    normalised = normalise_key(key)
                    key_context = bounded_field_kind(key)
                    inherited_context = bounded_context
                    if inherited_context:
                        if blocks_inherited_context(key, normalised):
                            inherited_context = None
                        elif local_non_bounded_scale and normalised in (
                            scalar_value_children | generic_ci_children
                        ):
                            inherited_context = None
                    if normalised in generic_ci_children and (
                        sibling_context or bounded_context
                    ) and not has_effect_scale_sibling and not local_non_bounded_scale:
                        key_context = sibling_context or bounded_context
                    visit(
                        child,
                        (*path, str(key)),
                        inherited_context or key_context,
                    )
                return
            if isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)), bounded_context)
                return
            if not bounded_context or isinstance(value, bool) or value is None:
                return
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                invalid.append((".".join(path), number, bounded_context))

        visit(summary)
        return invalid

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        for path, value, metric_kind in self._invalid_fraction_values(step_summary):
            roundoff_sized_overflow = bool(
                math.isfinite(value)
                and (value > 1.0 or value < 0.0)
                and min(abs(value), abs(value - 1.0)) <= 1e-12
            )
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Bounded {metric_kind} mismatch in step {step.step_id}: "
                        f"{path}={value} is outside [0, 1]. Do not retain even "
                        "roundoff-sized overflow in a registered summary; "
                        "normalize deterministically before writing the output."
                    ),
                    detail={
                        "issue": "bounded_metric_out_of_range",
                        "step_id": step.step_id,
                        "summary_path": path,
                        "metric_kind": metric_kind,
                        "reported_value": value,
                        "expected_min": 0.0,
                        "expected_max": 1.0,
                        "roundoff_sized_overflow": roundoff_sized_overflow,
                    },
                )
            )
        findings.extend(self._ambiguous_percent_pair_findings(step, step_summary))
        return findings

    @classmethod
    def _ambiguous_percent_pair_findings(
        cls, step: AnalysisStep, summary: Dict[str, Any]
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []

        def numeric_mapping(value: Any) -> Optional[Dict[str, float]]:
            if not isinstance(value, dict) or not value:
                return None
            parsed: Dict[str, float] = {}
            for key, raw in value.items():
                if isinstance(raw, bool):
                    return None
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    return None
                if not pd.notna(number):
                    return None
                parsed[str(key)] = number
            return parsed

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    key_text = str(key)
                    if key_text.endswith("_percent"):
                        pct_key = f"{key_text}_pct"
                        left = numeric_mapping(child)
                        right = numeric_mapping(value.get(pct_key))
                        if left and right and left.keys() == right.keys() and all(
                            abs(right[item] - 100.0 * left[item]) <= 1e-8
                            for item in left
                        ):
                            summary_path = ".".join((*path, key_text))
                            findings.append(
                                ValidationFinding(
                                    validator=cls.name,
                                    severity="error",
                                    message=(
                                        f"Ambiguous percent/fraction schema in step "
                                        f"{step.step_id}: {summary_path} contains "
                                        "proportions while its sibling *_pct contains "
                                        "the same values multiplied by 100. Rename the "
                                        "first field to *_fraction and keep the second "
                                        "as *_pct so machine consumers cannot interpret "
                                        "a fraction as a percent."
                                    ),
                                    detail={
                                        "step_id": step.step_id,
                                        "summary_path": summary_path,
                                        "pct_summary_path": ".".join(
                                            (*path, pct_key)
                                        ),
                                    },
                                )
                            )
                    visit(child, (*path, key_text))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return findings


class PrimaryModelContractValidator:
    """Fail closed on complex multi-model primary-association contracts.

    This validator deliberately covers binary-logistic and continuous
    linear/quantile adjusted-association steps, not EasyICU's survival,
    prediction, mixed-effects, or clustering families. Supported complex steps
    must expose one fixed ``model_contracts`` record per attempted model and a
    term-level coefficient table for fitted models so primary/secondary roles,
    denominators, adjustment sets, and fit diagnostics are machine-verifiable.
    """

    name = "primary_model_contract"
    _REQUIRED_FIELDS = (
        "model_id",
        "exposure_source",
        "exposure_expression",
        "exposure_role",
        "analysis_role",
        "analysis_set",
        "baseline_missing_policy",
        "n",
        "event_n",
        "fit_status",
        "converged",
        "separation_detected",
        "penalized",
        "fit_method",
    )
    _TERM_ROLES = {"intercept", "exposure", "availability", "adjustment"}
    _EXPOSURE_ROLES = {"primary", "secondary"}
    _ANALYSIS_ROLES = {"primary", "secondary", "sensitivity"}
    _ANALYSIS_SETS = {"source_aware", "complete_case"}
    _BASELINE_MISSING_POLICIES = {
        "drop_missing_baseline",
        "explicit_missing_category",
    }
    _FIT_STATUSES = {"fitted", "not_fitted", "separation_no_estimate"}
    _NONFITTED_RESULT_FIELDS = (
        "estimate",
        "odds_ratio",
        "or",
        "ci_low",
        "ci_high",
        "standard_error",
        "p_value",
    )
    _CLOSED_EFFECT_METHODS = {PLANNED_MODEL_REQUIREMENTS_STEP_METHOD}
    _CLOSED_EFFECT_PRODUCTS = {
        (
            PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
            PLANNED_MODEL_REQUIREMENTS_OUTPUT,
        )
    }
    _OUTCOME_TYPES = {"binary", "continuous"}
    _BINARY_MODEL_FAMILIES = ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES
    _CONTINUOUS_MODEL_FAMILIES = ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES
    _CONTINUOUS_INCOMPATIBLE_SCALES = {
        "log_odds",
        "odds_ratio",
        "or",
        "log_hazard",
        "hazard_ratio",
        "hr",
    }
    _BINARY_INCOMPATIBLE_SCALES = {
        "conditional_quantile_difference",
        "median_difference",
        "median_difference_days",
        "mean_difference",
        "outcome_unit_difference",
    }
    _CONTROLLED_PENALIZED_INTERVAL_METHODS = {
        "bootstrap",
        "firth_profile",
        "profile_likelihood",
        "easyicu_penalized_hessian_v1",
    }
    _CONTROLLED_CONVERGENCE_METHODS = {
        "optimizer_success",
        "kkt_residual",
        "firth_optimizer",
        "bootstrap_refit",
    }
    _OPERATIONAL_NAME_SUFFIXES = {
        "any",
        "count",
        "ever",
        "first",
        "flag",
        "indicator",
        "last",
        "max",
        "mean",
        "measured",
        "median",
        "min",
        "n",
        "observed",
        "raw",
        "sum",
        "value",
    }
    _FIGURE_ONLY_METHODS = {
        "figure_generation",
        "plot_generation",
        "publication_figure_generation",
        "visualization",
    }

    @staticmethod
    def _normalise(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")

    @classmethod
    def _authoritative_completed_records(
        cls,
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Use current checkpoints when a status-bearing ledger is present.

        Status-less records predate the append-only execution ledger and are
        retained only as a legacy compatibility path.  In a modern ledger, a
        later failed checkpoint must revoke an earlier successful summary.
        """

        records = [
            record
            for record in (completed_step_records or [])
            if isinstance(record, Mapping)
        ]
        if not any("status" in record for record in records):
            return records
        return list(current_successful_step_records(records))

    @classmethod
    def _is_closed_planner_owned_step(cls, step: AnalysisStep) -> bool:
        method = cls._normalise(str(step.method or "").split(" with ", 1)[0])
        products = set()
        for output in step.expected_outputs or []:
            kind, separator, name = str(output or "").partition(":")
            if separator:
                products.add((cls._normalise(kind), cls._normalise(name)))
        return method in cls._CLOSED_EFFECT_METHODS and bool(
            products & cls._CLOSED_EFFECT_PRODUCTS
        )

    @classmethod
    def _method_declares_penalty(
        cls,
        contract: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        values = [
            contract.get("fit_method"),
            contract.get("model_family"),
            contract.get("estimator"),
        ]
        if metadata:
            values.extend(
                (
                    metadata.get("fit_method"),
                    metadata.get("model_family"),
                    metadata.get("estimator"),
                )
            )
        blob = "_".join(cls._normalise(value) for value in values if value)
        return any(
            re.search(pattern, blob)
            for pattern in (
                r"(?:^|_)firth(?:_|$)",
                r"(?:^|_)ridge(?:_|$)",
                r"(?:^|_)lasso(?:_|$)",
                r"(?:^|_)elastic_?net(?:_|$)",
                r"(?:^|_)regulari[sz]ed(?:_|$)",
                r"(?:^|_)penali[sz]ed(?:_|$)",
            )
        )

    @classmethod
    def _planned_model_requirement_issues(
        cls,
        *,
        step: AnalysisStep,
        contracts: Sequence[Mapping[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        requirements = {
            item.requirement_id: item.model_dump(mode="python")
            for item in (getattr(step, "model_requirements", []) or [])
        }
        if not requirements:
            return [], {}

        issues: List[Dict[str, Any]] = []
        contracts_by_requirement: Dict[str, List[Mapping[str, Any]]] = {}
        for contract in contracts:
            requirement_id = str(contract.get("requirement_id") or "").strip()
            model_id = str(contract.get("model_id") or "")
            if not requirement_id:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "model_requirement_id_required",
                    }
                )
                continue
            if requirement_id not in requirements:
                issues.append(
                    {
                        "model_id": model_id,
                        "requirement_id": requirement_id,
                        "issue": "unplanned_model_requirement_id",
                    }
                )
                continue
            contracts_by_requirement.setdefault(requirement_id, []).append(contract)

        for requirement_id, requirement in requirements.items():
            matched = contracts_by_requirement.get(requirement_id, [])
            if not matched:
                if cls._planned_requirement_is_required(requirement):
                    issues.append(
                        {
                            "requirement_id": requirement_id,
                            "issue": "required_model_missing",
                            "expected": requirement,
                        }
                    )
                continue
            if len(matched) != 1:
                issues.append(
                    {
                        "requirement_id": requirement_id,
                        "issue": "duplicate_model_requirement_contract",
                        "reported": len(matched),
                    }
                )
                continue

            contract = matched[0]
            reported = {
                "outcome": contract.get("outcome"),
                "outcome_type": contract.get("outcome_type"),
                "method_family": contract.get("method_family")
                or contract.get("model_family"),
                "exposure_source": contract.get("exposure_source"),
                "analysis_role": contract.get("analysis_role"),
                "analysis_set": contract.get("analysis_set"),
            }
            mismatches: Dict[str, Dict[str, Any]] = {}
            for field in (
                "outcome",
                "outcome_type",
                "method_family",
                "analysis_role",
                "analysis_set",
            ):
                if cls._normalise(reported[field]) != cls._normalise(
                    requirement[field]
                ):
                    mismatches[field] = {
                        "expected": requirement[field],
                        "reported": reported[field],
                    }
            if not cls._names_match(
                requirement["exposure_source"], reported["exposure_source"]
            ):
                mismatches["exposure_source"] = {
                    "expected": requirement["exposure_source"],
                    "reported": reported["exposure_source"],
                }
            if mismatches:
                issues.append(
                    {
                        "model_id": contract.get("model_id"),
                        "requirement_id": requirement_id,
                        "issue": "model_requirement_field_mismatch",
                        "mismatches": mismatches,
                    }
                )
        return issues, requirements

    @classmethod
    def _planned_requirement_is_required(
        cls,
        requirement: Mapping[str, Any],
    ) -> bool:
        return bool(requirement.get("required_for_step_success")) or cls._normalise(
            requirement.get("analysis_role")
        ) in {"primary", "secondary"}

    @staticmethod
    def _fit_failure_reason(metadata: Mapping[str, Any]) -> str:
        return str(metadata.get("fit_failure_reason") or "").strip()

    @classmethod
    def _finite_nonfitted_result_fields(
        cls, rows: pd.DataFrame
    ) -> List[str]:
        fields: List[str] = []
        for column in cls._NONFITTED_RESULT_FIELDS:
            if column not in rows.columns:
                continue
            if any(cls._finite_number(value) is not None for value in rows[column]):
                fields.append(column)
        return fields

    @classmethod
    def _finite_nonfitted_summary_result_fields(
        cls,
        step_summary: Mapping[str, Any],
        *,
        model_id: str,
    ) -> List[str]:
        """Find finite inferential results attached to one non-fitted model."""

        fields: Set[str] = set()

        def visit(value: Any, inherited_model_id: str = "") -> None:
            if isinstance(value, Mapping):
                active_model_id = str(
                    value.get("model_id") or inherited_model_id
                ).strip()
                if active_model_id == model_id:
                    for field in cls._NONFITTED_RESULT_FIELDS:
                        if (
                            field in value
                            and cls._finite_number(value.get(field)) is not None
                        ):
                            fields.add(field)
                for child in value.values():
                    visit(child, active_model_id)
            elif isinstance(value, list):
                for child in value:
                    visit(child, inherited_model_id)

        visit(step_summary)
        return sorted(fields)

    @classmethod
    def _names_match(cls, left: Any, right: Any) -> bool:
        left_text = str(left or "").strip().lower()
        right_text = str(right or "").strip().lower()
        a = re.sub(r"[^a-z0-9]", "", left_text)
        b = re.sub(r"[^a-z0-9]", "", right_text)
        if not a or not b:
            return False
        if a == b:
            return True
        left_tokens = [token for token in re.split(r"[^a-z0-9]+", left_text) if token]
        right_tokens = [
            token for token in re.split(r"[^a-z0-9]+", right_text) if token
        ]

        def is_operational_alias(base: List[str], candidate: List[str]) -> bool:
            return bool(
                base
                and len(candidate) > len(base)
                and candidate[: len(base)] == base
                and all(
                    token in cls._OPERATIONAL_NAME_SUFFIXES
                    for token in candidate[len(base) :]
                )
            )

        return is_operational_alias(left_tokens, right_tokens) or is_operational_alias(
            right_tokens, left_tokens
        )

    @classmethod
    def _activates(
        cls,
        step: AnalysisStep,
        context: ResearchContext,
        step_summary: Mapping[str, Any],
    ) -> bool:
        has_planned_requirements = bool(
            getattr(step, "model_requirements", []) or []
        )
        if not (context.primary_exposure or "").strip() and not has_planned_requirements:
            return False
        method = cls._normalise(
            str(step.method or "").lower().split(" with ", 1)[0]
        )
        raw_outputs = [
            str(output or "").strip().lower()
            for output in (step.expected_outputs or [])
        ]
        outputs = set()
        for output in raw_outputs:
            output_kind, separator, output_name = output.partition(":")
            if not separator:
                continue
            outputs.add(
                (cls._normalise(output_kind), cls._normalise(output_name))
            )
        figure_only_outputs = bool(raw_outputs) and all(
            output.startswith("figure:") for output in raw_outputs
        )
        if method in cls._FIGURE_ONLY_METHODS or figure_only_outputs:
            return False
        supported_direct_method = method in (
            cls._BINARY_MODEL_FAMILIES | cls._CONTINUOUS_MODEL_FAMILIES
        )
        if has_planned_requirements:
            # AnalysisStep validation normally guarantees this scope. Keep the
            # runtime predicate defensive because model_copy(update=...) can
            # construct an unvalidated object in internal/test code.
            return (
                method in cls._CLOSED_EFFECT_METHODS
                and bool(outputs & cls._CLOSED_EFFECT_PRODUCTS)
            )
        # Once a step emits the machine contract key, even an empty or malformed
        # value must be audited rather than escaping through a prose router, but
        # only for the adjusted-association families this validator implements.
        # Survival, prediction, mixed-effects, and clustering outputs belong to
        # their own family-specific validators.
        if "model_contracts" in step_summary:
            return method in cls._CLOSED_EFFECT_METHODS or supported_direct_method
        return (
            method in cls._CLOSED_EFFECT_METHODS
            and bool(outputs & cls._CLOSED_EFFECT_PRODUCTS)
        )

    @staticmethod
    def _as_nonnegative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not pd.notna(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    @staticmethod
    def _as_bool(value: Any) -> Optional[bool]:
        return value if isinstance(value, bool) else None

    @classmethod
    def _latest_planned_adjustment(
        cls, completed_step_records: Sequence[Dict[str, Any]]
    ) -> tuple[List[str], List[str]]:
        for record in reversed(
            cls._authoritative_completed_records(completed_step_records)
        ):
            summary = record.get("step_summary")
            if not isinstance(summary, Mapping):
                continue
            planned = summary.get("planned_adjustment_context")
            if not isinstance(planned, Mapping):
                continue
            candidates = planned.get("candidate_covariates")
            excluded = planned.get("not_adjusted_for")
            if isinstance(candidates, list):
                return (
                    [str(value) for value in candidates if str(value).strip()],
                    [
                        str(value)
                        for value in (excluded if isinstance(excluded, list) else [])
                        if str(value).strip()
                    ],
                )
        return [], []

    @classmethod
    def _locked_primary_expression(
        cls,
        *,
        primary_exposure: str,
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> Optional[str]:
        locks: List[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    if (
                        cls._normalise(key) == "representation_locked"
                        and isinstance(child, str)
                    ):
                        locks.append(child)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for record in cls._authoritative_completed_records(
            completed_step_records
        ):
            summary = record.get("step_summary")
            if isinstance(summary, Mapping):
                visit(summary)
        for lock in reversed(locks):
            match = re.search(
                r"(?:np\.)?log1p\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
                lock,
                flags=re.IGNORECASE,
            )
            if match and cls._names_match(primary_exposure, match.group(1)):
                return f"log1p({match.group(1)})"
        return None

    @classmethod
    def _operational_primary_sources(
        cls,
        *,
        declared_primary: str,
        completed_step_records: Sequence[Dict[str, Any]],
        step_summary: Mapping[str, Any],
    ) -> List[str]:
        """Resolve structured context-exposure -> operational-column aliases."""

        sources: List[str] = []

        def primary_matches(value: Any) -> bool:
            if isinstance(value, Mapping):
                value = next(
                    (
                        value.get(key)
                        for key in (
                            "authoritative_context_exposure",
                            "context_exposure",
                            "name",
                        )
                        if value.get(key) is not None
                    ),
                    None,
                )
            return cls._names_match(declared_primary, value)

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                primary = value.get("primary_exposure")
                if primary_matches(primary):
                    for key in (
                        "primary_exposure_source",
                        "operational_column",
                        "exposure_source",
                    ):
                        candidate = value.get(key)
                        if candidate is not None and str(candidate).strip():
                            sources.append(str(candidate).strip())
                    if isinstance(primary, Mapping):
                        candidate = primary.get("operational_column")
                        if candidate is not None and str(candidate).strip():
                            sources.append(str(candidate).strip())
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for record in cls._authoritative_completed_records(
            completed_step_records
        ):
            summary = record.get("step_summary")
            if isinstance(summary, Mapping):
                visit(summary)
        return list(dict.fromkeys(sources))

    @classmethod
    def _expression_key(cls, value: Any) -> str:
        text = str(value or "").lower().replace("np.", "")
        return re.sub(r"\s+", "", text)

    @classmethod
    def _coefficient_rows(cls, out_dir: Path) -> Optional[pd.DataFrame]:
        frames: List[tuple[Path, pd.DataFrame]] = []
        required = {"model_id", "term", "term_role", "source_variable"}
        for path in sorted(Path(out_dir).glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if not required.issubset(frame.columns):
                continue
            if not {"ci_low", "ci_high"}.issubset(frame.columns):
                continue
            if not {"estimate", "odds_ratio", "or"}.intersection(frame.columns):
                continue
            # Figure/source-data bundles can be wide unions containing these
            # columns for only a subset of rows.  Ignore their non-model rows
            # instead of converting missing term roles into the literal role
            # ``nan`` and falsely rejecting an otherwise valid coefficient
            # table.
            frame = frame.loc[
                frame["model_id"].notna()
                & frame["term"].notna()
                & frame["term_role"].notna()
            ].copy()
            if frame.empty:
                continue
            frames.append((path, frame))
        if not frames:
            return None
        # Result tables can share model_id/term/effect columns while carrying
        # marginal risks or contrasts rather than fitted coefficients. Prefer
        # the term-level coefficient schema so missing standard errors created
        # only by a heterogeneous concat do not become false fit failures.
        coefficient_frames = [
            frame
            for path, frame in frames
            if "estimate_type" not in frame.columns
            or "coefficient" in cls._normalise(path.stem)
        ]
        selected = coefficient_frames or [frame for _, frame in frames]
        return pd.concat(selected, ignore_index=True)

    @classmethod
    def _current_adjustment_context(
        cls, step_summary: Mapping[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Read current-step adjustment declarations without case vocabulary."""

        candidates: List[str] = []
        excluded: List[str] = []

        def collect(raw: Any, target: List[str]) -> None:
            if not isinstance(raw, list):
                return
            for item in raw:
                value: Any = item
                if isinstance(item, Mapping):
                    value = next(
                        (
                            item.get(key)
                            for key in ("variable", "name", "source_variable")
                            if item.get(key) is not None
                        ),
                        None,
                    )
                text = str(value or "").strip()
                if text and text.lower() not in {"none", "nan", "null"}:
                    target.append(text)

        collect(step_summary.get("adjustment_covariates"), candidates)
        collect(step_summary.get("excluded_covariates"), excluded)
        planned = step_summary.get("planned_adjustment_context")
        if isinstance(planned, Mapping):
            collect(planned.get("candidate_covariates"), candidates)
            collect(planned.get("not_adjusted_for"), excluded)
        return list(dict.fromkeys(candidates)), list(dict.fromkeys(excluded))

    @classmethod
    def _actual_adjustment_sources_by_model(
        cls, coefficient_rows: pd.DataFrame
    ) -> Dict[str, List[str]]:
        sources: Dict[str, List[str]] = {}
        adjustment_rows = coefficient_rows[
            coefficient_rows["_term_role"].eq("adjustment")
        ]
        for model_id, rows in adjustment_rows.groupby("_model_id", sort=False):
            values = [
                str(value).strip()
                for value in rows["source_variable"]
                if pd.notna(value)
                and str(value).strip()
                and str(value).strip().lower() not in {"none", "nan", "null"}
            ]
            sources[str(model_id)] = list(dict.fromkeys(values))
        return sources

    @classmethod
    def _model_metadata_by_id(
        cls, step_summary: Mapping[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect structured model metadata repeated across summary sections."""

        metadata: Dict[str, Dict[str, Any]] = {}
        fields = {
            "outcome",
            "outcome_type",
            "family",
            "model_family",
            "fit_method",
            "interval_method",
            "intervals_approximate",
            "convergence_method",
            "optimizer_success",
            "max_abs_kkt",
            "convergence_tolerance",
            "fit_failure_reason",
            "categorical_covariates",
            "categorical_predictors",
            "categorical_sources",
            "categorical_variables",
        }

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                model_id = str(value.get("model_id") or "").strip()
                if model_id:
                    bucket = metadata.setdefault(model_id, {})
                    for field in fields:
                        if field in value and value.get(field) is not None:
                            bucket[field] = value.get(field)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(step_summary)
        return metadata

    @classmethod
    def _model_metadata(
        cls,
        contract: Mapping[str, Any],
        metadata_by_id: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        model_id = str(contract.get("model_id") or "")
        combined = dict(metadata_by_id.get(model_id, {}))
        combined.update(contract)
        if "model_family" not in combined and combined.get("family") is not None:
            combined["model_family"] = combined.get("family")
        cls._apply_nested_ridge_convergence_alias(
            contract=contract,
            metadata=combined,
        )
        return combined

    @classmethod
    def _apply_nested_ridge_convergence_alias(
        cls,
        *,
        contract: Mapping[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """Map model-bound sklearn ridge diagnostics to the controlled fields."""

        if (
            "convergence_method" in metadata
            or "optimizer_success" in metadata
            or cls._as_bool(metadata.get("penalized")) is not True
        ):
            return
        fit_method = cls._normalise(metadata.get("fit_method"))
        penalty = cls._normalise(metadata.get("penalty"))
        if not (
            re.search(r"(?:^|_)ridge(?:_|$)", fit_method)
            or penalty == "ridge"
        ):
            return
        diagnostics = contract.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            return
        model_id = str(contract.get("model_id") or "").strip()
        diagnostics_model_id = str(diagnostics.get("model_id") or "").strip()
        if diagnostics_model_id and diagnostics_model_id != model_id:
            return
        iterations = cls._as_nonnegative_int(diagnostics.get("ridge_iterations"))
        if (
            cls._as_bool(diagnostics.get("ridge_converged")) is not True
            or iterations is None
            or iterations < 1
        ):
            return
        metadata["convergence_method"] = "optimizer_success"
        metadata["optimizer_success"] = True

    @classmethod
    def _declared_outcome_type(
        cls,
        metadata: Mapping[str, Any],
        *,
        frame: Optional[pd.DataFrame] = None,
        outcome: str = "",
    ) -> str:
        explicit = cls._normalise(metadata.get("outcome_type"))
        if explicit in cls._OUTCOME_TYPES:
            return explicit
        family = cls._normalise(
            metadata.get("model_family") or metadata.get("fit_method")
        )
        if family in cls._BINARY_MODEL_FAMILIES:
            return "binary"
        if family in cls._CONTINUOUS_MODEL_FAMILIES:
            return "continuous"
        if frame is not None and outcome in frame.columns:
            values = pd.to_numeric(frame[outcome], errors="coerce").dropna()
            if not values.empty and set(values.unique()).issubset({0, 1}):
                return "binary"
            if not values.empty:
                return "continuous"
        # Backward compatibility for older single-binary-outcome contracts.
        return "binary"

    @staticmethod
    def _finite_number(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _fitted_term_interval_issues(
        cls,
        *,
        contract: Mapping[str, Any],
        metadata: Mapping[str, Any],
        rows: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        if cls._normalise(contract.get("fit_status")) != "fitted" or rows.empty:
            return []
        model_id = str(contract.get("model_id") or "")
        penalized = cls._as_bool(contract.get("penalized")) is True
        interval_method = cls._normalise(metadata.get("interval_method"))
        point_only = penalized and interval_method == "unavailable"
        issues: List[Dict[str, Any]] = []
        effect_columns = [
            column
            for column in ("estimate", "odds_ratio", "or")
            if column in rows.columns
        ]
        has_standard_error = "standard_error" in rows.columns
        for _, row in rows.iterrows():
            term = str(row.get("term") or "")
            row_interval_method = cls._normalise(row.get("interval_method"))
            if (
                "reference" in cls._normalise(term)
                or row_interval_method == "not_applicable_reference"
            ):
                continue
            estimate = next(
                (
                    number
                    for column in effect_columns
                    if (number := cls._finite_number(row.get(column))) is not None
                ),
                None,
            )
            low = cls._finite_number(row.get("ci_low"))
            high = cls._finite_number(row.get("ci_high"))
            standard_error = (
                cls._finite_number(row.get("standard_error"))
                if has_standard_error
                else 0.0
            )
            has_any_interval = low is not None or high is not None or (
                has_standard_error and standard_error is not None
            )
            if point_only:
                reasons: List[str] = []
                if estimate is None:
                    reasons.append("nonfinite_estimate")
                if has_any_interval:
                    reasons.append("point_only_contract_contains_interval")
                if reasons:
                    issues.append(
                        {
                            "model_id": model_id,
                            "term": term,
                            "term_role": row.get("term_role"),
                            "issue": "fitted_term_missing_or_invalid_interval",
                            "reasons": reasons,
                        }
                    )
                continue
            reasons: List[str] = []
            if estimate is None:
                reasons.append("nonfinite_estimate")
            if low is None or high is None:
                reasons.append("missing_or_nonfinite_ci")
            elif low > high:
                reasons.append("reversed_ci")
            if has_standard_error and (
                standard_error is None or standard_error < 0
            ):
                reasons.append("missing_nonfinite_or_negative_standard_error")
            if reasons:
                issues.append(
                    {
                        "model_id": model_id,
                        "term": term,
                        "term_role": row.get("term_role"),
                        "issue": "fitted_term_missing_or_invalid_interval",
                        "reasons": reasons,
                    }
                )
        return issues

    @classmethod
    def _effect_scale_issues(
        cls,
        *,
        contract: Mapping[str, Any],
        metadata: Mapping[str, Any],
        rows: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        if rows.empty:
            return []
        outcome_type = cls._declared_outcome_type(metadata)
        issues: List[Dict[str, Any]] = []
        if (
            outcome_type == "continuous"
            and cls._normalise(contract.get("fit_status")) == "fitted"
        ):
            if "effect_scale" not in rows.columns:
                return [
                    {
                        "model_id": contract.get("model_id"),
                        "issue": "continuous_fitted_term_requires_effect_scale",
                        "terms": [str(value) for value in rows["term"].tolist()],
                    }
                ]
            missing_scale_terms = [
                str(row.get("term") or "")
                for _, row in rows.iterrows()
                if not str(row.get("effect_scale") or "").strip()
                or str(row.get("effect_scale") or "").strip().lower()
                in {"nan", "none", "null"}
            ]
            if missing_scale_terms:
                issues.append(
                    {
                        "model_id": contract.get("model_id"),
                        "issue": "continuous_fitted_term_requires_effect_scale",
                        "terms": missing_scale_terms,
                    }
                )
        if "effect_scale" not in rows.columns:
            return issues
        scales = {
            cls._normalise(value)
            for value in rows["effect_scale"]
            if pd.notna(value) and str(value).strip()
        }
        incompatible = (
            scales & cls._CONTINUOUS_INCOMPATIBLE_SCALES
            if outcome_type == "continuous"
            else scales & cls._BINARY_INCOMPATIBLE_SCALES
        )
        if incompatible:
            issues.append(
                {
                    "model_id": contract.get("model_id"),
                    "issue": "effect_scale_model_family_mismatch",
                    "outcome_type": outcome_type,
                    "model_family": metadata.get("model_family")
                    or metadata.get("fit_method"),
                    "reported_effect_scales": sorted(incompatible),
                }
            )
        return issues

    @classmethod
    def _penalized_provenance_issues(
        cls,
        *,
        contract: Mapping[str, Any],
        metadata: Mapping[str, Any],
        rows: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        if (
            cls._as_bool(contract.get("penalized")) is not True
            and not cls._method_declares_penalty(contract, metadata)
        ):
            return []
        model_id = str(contract.get("model_id") or "")
        interval_method = cls._normalise(metadata.get("interval_method"))
        finite_intervals = False
        if not rows.empty:
            finite_intervals = any(
                cls._finite_number(row.get("ci_low")) is not None
                and cls._finite_number(row.get("ci_high")) is not None
                for _, row in rows.iterrows()
            )
        issues: List[Dict[str, Any]] = []
        if finite_intervals and interval_method not in cls._CONTROLLED_PENALIZED_INTERVAL_METHODS:
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "penalized_intervals_require_controlled_provenance",
                    "reported_interval_method": metadata.get("interval_method"),
                    "allowed": sorted(cls._CONTROLLED_PENALIZED_INTERVAL_METHODS),
                }
            )
        if interval_method == "unavailable" and finite_intervals:
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "point_only_contract_contains_interval",
                }
            )
        if cls._as_bool(contract.get("converged")) is True:
            convergence_method = cls._normalise(
                metadata.get("convergence_method")
            )
            optimizer_success = cls._as_bool(metadata.get("optimizer_success"))
            verified = (
                convergence_method in cls._CONTROLLED_CONVERGENCE_METHODS
                and optimizer_success is True
            )
            if verified and convergence_method == "kkt_residual":
                residual = cls._finite_number(metadata.get("max_abs_kkt"))
                tolerance = cls._finite_number(
                    metadata.get("convergence_tolerance")
                )
                if tolerance is None:
                    tolerance = 1e-6
                verified = residual is not None and residual <= tolerance
            if not verified:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "penalized_convergence_not_verified",
                        "reported_convergence_method": metadata.get(
                            "convergence_method"
                        ),
                        "optimizer_success": metadata.get("optimizer_success"),
                    }
                )
        if (
            interval_method == "easyicu_penalized_hessian_v1"
            and cls._as_bool(metadata.get("intervals_approximate")) is not True
        ):
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "penalized_hessian_interval_must_be_approximate",
                }
            )
        return issues

    @classmethod
    def _expected_denominator(
        cls,
        *,
        frame: pd.DataFrame,
        outcome: str,
        outcome_type: str,
        covariates: Sequence[str],
        contract: Mapping[str, Any],
    ) -> Optional[tuple[int, Optional[int]]]:
        if outcome not in frame.columns:
            return None
        outcome_values = pd.to_numeric(frame[outcome], errors="coerce")
        if outcome_type == "binary":
            mask = outcome_values.isin([0, 1])
        elif outcome_type == "continuous":
            mask = outcome_values.notna() & outcome_values.map(math.isfinite)
        else:
            return None
        policy = cls._normalise(contract.get("baseline_missing_policy"))
        if policy in {"drop_missing", "drop_missing_baseline", "complete_case"}:
            for covariate in covariates:
                if covariate not in frame.columns:
                    return None
                mask &= frame[covariate].notna()
        elif policy not in {"explicit_missing_category", "missing_category"}:
            return None

        analysis_set = cls._normalise(contract.get("analysis_set"))
        if analysis_set == "complete_case":
            exposure = str(contract.get("exposure_source") or "")
            if exposure not in frame.columns:
                return None
            values = frame[exposure]
            mask &= values.notna()
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                mask &= numeric.map(lambda value: pd.notna(value) and abs(value) != float("inf"))
        elif analysis_set != "source_aware":
            return None
        event_n = (
            int(outcome_values.loc[mask].sum())
            if outcome_type == "binary"
            else None
        )
        return int(mask.sum()), event_n

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        context: ResearchContext,
        completed_step_records: Sequence[Dict[str, Any]],
        out_dir: Path,
        cohort_path: Path,
    ) -> List[ValidationFinding]:
        if not self._activates(step, context, step_summary):
            return []
        issues: List[Dict[str, Any]] = []
        if self._is_closed_planner_owned_step(step) and not (
            getattr(step, "model_requirements", []) or []
        ):
            issues.append(
                {
                    "issue": "planned_model_requirements_required",
                    "method": step.method,
                    "expected_outputs": list(step.expected_outputs or []),
                }
            )
        raw_contracts = step_summary.get("model_contracts")
        if not isinstance(raw_contracts, list) or not raw_contracts:
            issues.append(
                {
                    "issue": "missing_model_contracts",
                    "required_fields": list(self._REQUIRED_FIELDS),
                }
            )
            contracts: List[Mapping[str, Any]] = []
        else:
            contracts = [
                item for item in raw_contracts if isinstance(item, Mapping)
            ]
            if len(contracts) != len(raw_contracts):
                issues.append({"issue": "model_contract_must_be_object"})

        model_ids: Set[str] = set()
        for index, contract in enumerate(contracts):
            missing = [field for field in self._REQUIRED_FIELDS if field not in contract]
            if missing:
                issues.append(
                    {
                        "model_index": index,
                        "issue": "missing_model_contract_fields",
                        "fields": missing,
                    }
                )
                continue
            model_id = str(contract.get("model_id") or "").strip()
            if not model_id or model_id in model_ids:
                issues.append(
                    {
                        "model_index": index,
                        "model_id": model_id,
                        "issue": "blank_or_duplicate_model_id",
                    }
                )
            model_ids.add(model_id)
            controlled_fields = (
                ("exposure_role", self._EXPOSURE_ROLES),
                ("analysis_role", self._ANALYSIS_ROLES),
                ("analysis_set", self._ANALYSIS_SETS),
                (
                    "baseline_missing_policy",
                    self._BASELINE_MISSING_POLICIES,
                ),
                ("fit_status", self._FIT_STATUSES),
            )
            for field, allowed in controlled_fields:
                reported = self._normalise(contract.get(field))
                if reported not in allowed:
                    issues.append(
                        {
                            "model_id": model_id,
                            "issue": f"noncanonical_{field}",
                            "reported": contract.get(field),
                            "allowed": sorted(allowed),
                        }
                    )

        requirement_issues, requirements_by_id = (
            self._planned_model_requirement_issues(
                step=step,
                contracts=contracts,
            )
        )
        issues.extend(requirement_issues)

        primary_models = [
            contract
            for contract in contracts
            if self._normalise(contract.get("analysis_role")) == "primary"
        ]
        if len(primary_models) != 1:
            issues.append(
                {
                    "issue": "exactly_one_primary_model_required",
                    "reported": len(primary_models),
                }
            )
        declared_primary = str(context.primary_exposure or "")
        declared_primary_sources = (
            [
                declared_primary,
                *self._operational_primary_sources(
                    declared_primary=declared_primary,
                    completed_step_records=completed_step_records,
                    step_summary=step_summary,
                ),
            ]
            if declared_primary.strip()
            else []
        )
        if len(primary_models) == 1 and declared_primary_sources:
            primary = primary_models[0]
            if not any(
                self._names_match(source, primary.get("exposure_source"))
                for source in declared_primary_sources
            ):
                issues.append(
                    {
                        "model_id": primary.get("model_id"),
                        "issue": "primary_exposure_mismatch",
                        "expected": declared_primary_sources,
                        "reported": primary.get("exposure_source"),
                    }
                )
            if self._normalise(primary.get("exposure_role")) != "primary":
                issues.append(
                    {
                        "model_id": primary.get("model_id"),
                        "issue": "primary_model_exposure_role_must_be_primary",
                    }
                )
            locked_expression = self._locked_primary_expression(
                primary_exposure=declared_primary,
                completed_step_records=completed_step_records,
            )
            if locked_expression and self._expression_key(
                primary.get("exposure_expression")
            ) != self._expression_key(locked_expression):
                issues.append(
                    {
                        "model_id": primary.get("model_id"),
                        "issue": "locked_primary_expression_mismatch",
                        "expected": locked_expression,
                        "reported": primary.get("exposure_expression"),
                    }
                )

        if declared_primary_sources:
            for contract in contracts:
                is_declared_exposure = any(
                    self._names_match(source, contract.get("exposure_source"))
                    for source in declared_primary_sources
                )
                exposure_role = self._normalise(contract.get("exposure_role"))
                if is_declared_exposure and exposure_role != "primary":
                    issues.append(
                        {
                            "model_id": contract.get("model_id"),
                            "issue": "declared_primary_exposure_role_mismatch",
                            "reported": contract.get("exposure_role"),
                        }
                    )
                if not is_declared_exposure and exposure_role == "primary":
                    issues.append(
                        {
                            "model_id": contract.get("model_id"),
                            "issue": "alternate_exposure_cannot_be_primary",
                            "reported_source": contract.get("exposure_source"),
                        }
                    )

        candidate_covariates, not_adjusted = self._latest_planned_adjustment(
            completed_step_records
        )
        current_candidates, current_excluded = self._current_adjustment_context(
            step_summary
        )
        if not candidate_covariates:
            candidate_covariates = current_candidates
        if not not_adjusted:
            not_adjusted = current_excluded
        metadata_by_id = self._model_metadata_by_id(step_summary)
        nonfitted_result_fields: Dict[str, Set[str]] = {}
        for contract in contracts:
            model_id = str(contract.get("model_id") or "")
            fit_status = self._normalise(contract.get("fit_status"))
            if fit_status == "fitted":
                continue
            analysis_role = self._normalise(contract.get("analysis_role"))
            requirement_id = str(contract.get("requirement_id") or "").strip()
            requirement = requirements_by_id.get(requirement_id)
            required_for_success = (
                self._planned_requirement_is_required(requirement)
                if requirement is not None
                else analysis_role in {"primary", "secondary"}
            )
            if required_for_success:
                issues.append(
                    {
                        "model_id": model_id,
                        "requirement_id": requirement_id or None,
                        "analysis_role": analysis_role,
                        "fit_status": fit_status,
                        "issue": "required_model_not_fitted",
                    }
                )
            metadata = self._model_metadata(contract, metadata_by_id)
            finite_summary_fields = self._finite_nonfitted_summary_result_fields(
                step_summary,
                model_id=model_id,
            )
            if finite_summary_fields:
                nonfitted_result_fields.setdefault(model_id, set()).update(
                    finite_summary_fields
                )
            if not self._fit_failure_reason(metadata):
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "fit_failure_reason_required",
                        "fit_status": fit_status,
                    }
                )
        actual_covariates_by_model: Dict[str, List[str]] = {}
        coefficient_rows = self._coefficient_rows(Path(out_dir))
        if coefficient_rows is None:
            issues.append(
                {
                    "issue": "missing_term_level_coefficient_table",
                    "required_columns": [
                        "model_id",
                        "term",
                        "term_role",
                        "source_variable",
                        "estimate_or_odds_ratio",
                        "ci_low",
                        "ci_high",
                    ],
                }
            )
        else:
            coefficient_rows = coefficient_rows.copy()
            coefficient_rows["_model_id"] = coefficient_rows["model_id"].astype(str)
            coefficient_rows["_term_role"] = coefficient_rows["term_role"].map(
                self._normalise
            )
            coefficient_rows["_source"] = coefficient_rows[
                "source_variable"
            ].astype(str)
            actual_covariates_by_model = self._actual_adjustment_sources_by_model(
                coefficient_rows
            )
            invalid_roles = sorted(
                set(coefficient_rows["_term_role"]) - self._TERM_ROLES
            )
            if invalid_roles:
                issues.append(
                    {"issue": "invalid_term_roles", "reported": invalid_roles}
                )
            exposure_sources = [
                str(contract.get("exposure_source") or "") for contract in contracts
            ]
            allowed_norm = {self._normalise(value) for value in candidate_covariates}
            excluded_norm = {self._normalise(value) for value in not_adjusted}
            for contract in contracts:
                model_id = str(contract.get("model_id") or "")
                source = str(contract.get("exposure_source") or "")
                metadata = self._model_metadata(contract, metadata_by_id)
                fit_status = self._normalise(contract.get("fit_status"))
                rows = coefficient_rows[
                    coefficient_rows["_model_id"].eq(model_id)
                ]
                if fit_status != "fitted":
                    finite_fields = self._finite_nonfitted_result_fields(rows)
                    if finite_fields:
                        nonfitted_result_fields.setdefault(model_id, set()).update(
                            finite_fields
                        )
                    continue
                if rows.empty:
                    issues.append(
                        {"model_id": model_id, "issue": "missing_coefficient_rows"}
                    )
                    continue
                exposure_rows = rows[rows["_term_role"].eq("exposure")]
                if exposure_rows.empty or any(
                    not self._names_match(source, value)
                    for value in exposure_rows["_source"]
                ):
                    issues.append(
                        {
                            "model_id": model_id,
                            "issue": "exposure_terms_do_not_match_model_source",
                        }
                    )
                for _, row in rows[rows["_term_role"].eq("adjustment")].iterrows():
                    adjustment = str(row["_source"])
                    adjustment_norm = self._normalise(adjustment)
                    other_exposure = next(
                        (
                            other
                            for other in exposure_sources
                            if other != source and self._names_match(other, adjustment)
                        ),
                        None,
                    )
                    if other_exposure is not None:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "mutual_exposure_adjustment",
                                "offending_source": adjustment,
                            }
                        )
                    if excluded_norm and adjustment_norm in excluded_norm:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "forbidden_adjustment_source",
                                "offending_source": adjustment,
                            }
                        )
                    if allowed_norm and adjustment_norm not in allowed_norm:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "adjustment_outside_planned_allowlist",
                                "offending_source": adjustment,
                                "allowed": candidate_covariates,
                            }
                        )
                issues.extend(
                    self._fitted_term_interval_issues(
                        contract=contract,
                        metadata=metadata,
                        rows=rows,
                    )
                )
                issues.extend(
                    self._effect_scale_issues(
                        contract=contract,
                        metadata=metadata,
                        rows=rows,
                    )
                )
                issues.extend(
                    self._penalized_provenance_issues(
                        contract=contract,
                        metadata=metadata,
                        rows=rows,
                    )
                )

            # When the script also exposes model-level results, verify that its
            # term table carries the same exposure estimates and intervals.
            # This catches silent coefficient-index shifts (for example,
            # assigning beta[0], the intercept, to the first predictor) even
            # when all required columns and roles are present.
            model_results = step_summary.get("models")
            if isinstance(model_results, list):
                for model_result in model_results:
                    if not isinstance(model_result, Mapping):
                        continue
                    model_id = str(model_result.get("model_id") or "")
                    exposure_terms = model_result.get("exposure_terms")
                    if not model_id or not isinstance(exposure_terms, list):
                        continue
                    model_rows = coefficient_rows[
                        coefficient_rows["_model_id"].eq(model_id)
                        & coefficient_rows["_term_role"].eq("exposure")
                    ]
                    for expected_term in exposure_terms:
                        if not isinstance(expected_term, Mapping):
                            continue
                        term = str(expected_term.get("term") or "")
                        if not term:
                            continue
                        rows = model_rows[
                            model_rows["term"].astype(str).eq(term)
                        ]
                        if rows.empty:
                            issues.append(
                                {
                                    "model_id": model_id,
                                    "term": term,
                                    "issue": "model_result_term_missing_from_coefficients",
                                }
                            )
                            continue
                        comparisons = (
                            ("estimate", "estimate"),
                            ("odds_ratio", "odds_ratio"),
                            ("ci_low", "ci_low"),
                            ("ci_high", "ci_high"),
                        )
                        for summary_field, table_field in comparisons:
                            expected_value = expected_term.get(summary_field)
                            if expected_value is None or table_field not in rows.columns:
                                continue
                            expected_number = pd.to_numeric(
                                pd.Series([expected_value]), errors="coerce"
                            ).iloc[0]
                            reported_numbers = pd.to_numeric(
                                rows[table_field], errors="coerce"
                            ).dropna()
                            if pd.isna(expected_number) or reported_numbers.empty:
                                continue
                            if not all(
                                abs(float(value) - float(expected_number))
                                <= 1e-9 * max(1.0, abs(float(expected_number)))
                                for value in reported_numbers
                            ):
                                issues.append(
                                    {
                                        "model_id": model_id,
                                        "term": term,
                                        "issue": "coefficient_model_result_mismatch",
                                        "field": summary_field,
                                        "expected": float(expected_number),
                                        "reported": [
                                            float(value)
                                            for value in reported_numbers.unique()[:5]
                                        ],
                                    }
                                )

        for model_id, finite_fields in nonfitted_result_fields.items():
            contract = next(
                (
                    item
                    for item in contracts
                    if str(item.get("model_id") or "") == model_id
                ),
                {},
            )
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "inconsistent_not_fitted_estimate",
                    "fit_status": self._normalise(contract.get("fit_status")),
                    "finite_fields": sorted(finite_fields),
                }
            )

        try:
            cohort = pd.read_parquet(cohort_path)
        except Exception:
            cohort = None
            issues.append({"issue": "cohort_unreadable_for_denominator_audit"})
        for contract in contracts:
            model_id = str(contract.get("model_id") or "")
            metadata = self._model_metadata(contract, metadata_by_id)
            outcome = str(metadata.get("outcome") or context.target_outcome or "")
            outcome_type = self._declared_outcome_type(
                metadata,
                frame=cohort,
                outcome=outcome,
            )
            model_covariates = (
                actual_covariates_by_model.get(model_id, candidate_covariates)
                if coefficient_rows is not None
                else candidate_covariates
            )
            fit_status = self._normalise(contract.get("fit_status"))
            converged = self._as_bool(contract.get("converged"))
            separation = self._as_bool(contract.get("separation_detected"))
            penalized = self._as_bool(contract.get("penalized"))
            method_declares_penalty = self._method_declares_penalty(
                contract, metadata
            )
            effective_penalized = penalized is True or method_declares_penalty
            reported_n = self._as_nonnegative_int(contract.get("n"))
            reported_events = (
                self._as_nonnegative_int(contract.get("event_n"))
                if outcome_type == "binary"
                else None
            )
            if converged is None or separation is None or penalized is None:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "fit_diagnostics_must_be_boolean",
                    }
                )
            if not str(contract.get("fit_method") or "").strip():
                issues.append(
                    {"model_id": model_id, "issue": "fit_method_required"}
                )
            fit_method_text = str(contract.get("fit_method") or "").lower()
            if method_declares_penalty and penalized is False:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "penalized_method_must_report_penalized_true",
                        "fit_method": contract.get("fit_method"),
                    }
                )
            if effective_penalized and reported_n and "firth" not in fit_method_text:
                if "statsmodels" in fit_method_text and any(
                    token in fit_method_text
                    for token in ("regularized", "ridge", "elastic_net")
                ):
                    alpha_match = re.search(
                        r"alpha\s*=\s*([0-9.eE+-]+)", fit_method_text
                    )
                    max_alpha = 1.0 / float(reported_n)
                    if alpha_match is None:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "statsmodels_penalty_strength_not_reported",
                                "required_format": "fit_method includes alpha=<value>",
                                "maximum_weak_ridge_alpha": max_alpha,
                            }
                        )
                    else:
                        try:
                            alpha = float(alpha_match.group(1))
                        except ValueError:
                            alpha = float("nan")
                        if not pd.notna(alpha) or alpha <= 0 or alpha > max_alpha:
                            issues.append(
                                {
                                    "model_id": model_id,
                                    "issue": "statsmodels_penalty_too_strong_for_separation_fallback",
                                    "reported_alpha": alpha,
                                    "maximum_weak_ridge_alpha": max_alpha,
                                    "rationale": (
                                        "The per-observation statsmodels penalty "
                                        "must not dominate an inferential target "
                                        "effect when used only to stabilize separation."
                                    ),
                                }
                            )
                elif "sklearn" in fit_method_text:
                    c_match = re.search(
                        r"(?:^|[^a-z])c\s*=\s*([0-9.eE+-]+)",
                        fit_method_text,
                    )
                    if c_match is None:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "sklearn_penalty_strength_not_reported",
                                "required_format": "fit_method includes C=<value>",
                                "minimum_weak_ridge_c": 1.0,
                            }
                        )
                    else:
                        try:
                            c_value = float(c_match.group(1))
                        except ValueError:
                            c_value = float("nan")
                        if not pd.notna(c_value) or c_value < 1.0:
                            issues.append(
                                {
                                    "model_id": model_id,
                                    "issue": "sklearn_penalty_too_strong_for_separation_fallback",
                                    "reported_c": c_value,
                                    "minimum_weak_ridge_c": 1.0,
                                }
                            )
            if fit_status == "fitted" and converged is not True:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "fitted_model_must_converge",
                    }
                )
            if separation is True and penalized is not True and fit_status not in {
                "separation_no_estimate",
                "not_fitted",
            }:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "separation_requires_penalized_fit_or_no_estimate",
                    }
                )
            if reported_n is None or (
                outcome_type == "binary" and reported_events is None
            ):
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "model_n_and_event_n_must_be_counts",
                    }
                )
            if outcome_type == "continuous" and contract.get("event_n") is not None:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "continuous_outcome_event_n_must_be_null",
                        "reported_event_n": contract.get("event_n"),
                    }
                )
            if cohort is not None:
                expected = self._expected_denominator(
                    frame=cohort,
                    outcome=outcome,
                    outcome_type=outcome_type,
                    covariates=model_covariates,
                    contract=contract,
                )
                if expected is None:
                    issues.append(
                        {
                            "model_id": model_id,
                            "issue": "denominator_contract_unresolvable",
                        }
                    )
                elif (reported_n, reported_events) != expected:
                    issues.append(
                        {
                            "model_id": model_id,
                            "issue": "model_denominator_or_event_mismatch",
                            "expected_n": expected[0],
                            "expected_event_n": expected[1],
                            "reported_n": reported_n,
                            "reported_event_n": reported_events,
                        }
                    )
                if expected is not None and outcome_type == "binary":
                    zero_cells = self._categorical_zero_event_cells(
                        frame=cohort,
                        outcome=outcome,
                        covariates=model_covariates,
                        contract=metadata,
                        coefficient_rows=(
                            coefficient_rows[
                                coefficient_rows["_model_id"].eq(model_id)
                            ]
                            if coefficient_rows is not None
                            else None
                        ),
                    )
                    if zero_cells and separation is not True:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "zero_cell_separation_not_reported",
                                "cells": zero_cells[:10],
                            }
                        )
                    if (
                        zero_cells
                        and fit_status == "fitted"
                        and penalized is not True
                    ):
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "zero_cell_separation_requires_penalized_fit",
                                "cells": zero_cells[:10],
                            }
                        )

        if not issues:
            return []
        return [
            ValidationFinding(
                validator=self.name,
                severity="error",
                message=(
                    f"Complex primary-association step {step.step_id} violates "
                    f"the machine-verifiable multi-model contract ({len(issues)} "
                    "issue(s)). Emit step_summary.model_contracts with the fixed "
                    "fields, keep exactly one context-declared primary exposure, "
                    "label alternate representations secondary/sensitivity, fit "
                    "separate models without mutual adjustment, use only the "
                    "planned baseline covariates, satisfy every planner-owned "
                    "required model requirement, report honest non-fit reasons, "
                    "report exact n/event_n, and "
                    "write a term-level coefficient table with model_id, term, "
                    "term_role, source_variable, effect and CI columns plus "
                    "convergence/separation/penalization diagnostics."
                ),
                detail={
                    "step_id": step.step_id,
                    "issues": issues[:50],
                    "required_model_contract_fields": list(self._REQUIRED_FIELDS),
                },
            )
        ]

    @classmethod
    def _categorical_zero_event_cells(
        cls,
        *,
        frame: pd.DataFrame,
        outcome: str,
        covariates: Sequence[str],
        contract: Mapping[str, Any],
        coefficient_rows: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Return categorical baseline cells with zero events or zero survivors."""

        if outcome not in frame.columns:
            return []
        outcome_values = pd.to_numeric(frame[outcome], errors="coerce")
        mask = outcome_values.isin([0, 1])
        policy = cls._normalise(contract.get("baseline_missing_policy"))
        if policy == "drop_missing_baseline":
            for covariate in covariates:
                if covariate not in frame.columns:
                    return []
                mask &= frame[covariate].notna()
        elif policy != "explicit_missing_category":
            return []
        if cls._normalise(contract.get("analysis_set")) == "complete_case":
            exposure = str(contract.get("exposure_source") or "")
            if exposure not in frame.columns:
                return []
            values = frame[exposure]
            mask &= values.notna()
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                mask &= numeric.map(
                    lambda value: pd.notna(value) and abs(value) != float("inf")
                )
        declared_categorical: Set[str] = set()
        for key in (
            "categorical_covariates",
            "categorical_predictors",
            "categorical_sources",
            "categorical_variables",
        ):
            raw = contract.get(key)
            if isinstance(raw, list):
                declared_categorical.update(
                    cls._normalise(value)
                    for value in raw
                    if str(value or "").strip()
                )

        cells: List[Dict[str, Any]] = []
        for covariate in covariates:
            if covariate not in frame.columns:
                continue
            values = frame.loc[mask, covariate]
            modeled_as_categorical = False
            if coefficient_rows is not None and not coefficient_rows.empty:
                source_rows = coefficient_rows[
                    coefficient_rows["source_variable"].map(cls._normalise).eq(
                        cls._normalise(covariate)
                    )
                ]
                terms = [str(value) for value in source_rows.get("term", [])]
                modeled_as_categorical = len(source_rows) > 1 or any(
                    re.search(r"(?:\bC\s*\(|\[T\.|one[_ -]?hot|dummy)", term, re.I)
                    for term in terms
                )
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            low_cardinality_integer = False
            if len(numeric) >= 20:
                unique_n = int(numeric.nunique(dropna=True))
                low_cardinality_integer = bool(
                    1 < unique_n <= 12
                    and unique_n / len(numeric) <= 0.2
                    and ((numeric - numeric.round()).abs() <= 1e-9).all()
                )
            is_categorical = (
                isinstance(values.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(values)
                or pd.api.types.is_string_dtype(values)
                or pd.api.types.is_bool_dtype(values)
                or cls._normalise(covariate) in declared_categorical
                or modeled_as_categorical
                or low_cardinality_integer
            )
            if not is_categorical:
                continue
            if policy == "explicit_missing_category":
                values = values.astype("object").where(values.notna(), "<missing>")
            observed_outcome = outcome_values.loc[mask]
            grouped = pd.DataFrame(
                {"level": values.astype(str), "outcome": observed_outcome}
            ).groupby("level", dropna=False)["outcome"].agg(["count", "sum"])
            for level, row in grouped.iterrows():
                count = int(row["count"])
                event_n = int(row["sum"])
                if count > 0 and event_n in {0, count}:
                    cells.append(
                        {
                            "variable": covariate,
                            "level": str(level),
                            "n": count,
                            "event_n": event_n,
                        }
                    )
        return cells


class CrossStepReconciliationTraceValidator:
    """Verify that a reconciliation table selects the correct parent rows.

    The supported absolute-risk table schema carries both prevalence and
    outcome-risk rows for each stratum.  Matching only on label can silently
    bind a prevalence row and then report ``n_denominator`` as the stratum N.
    This validator checks the detailed reconciliation CSV against the exact
    registered parent table selected by the step itself.
    """

    name = "cross_step_reconciliation_trace"

    @staticmethod
    def _normalise(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        try:
            number = float(value)
            if pd.notna(number) and number.is_integer():
                return str(int(number))
        except (TypeError, ValueError):
            pass
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    @classmethod
    def _status_alias(cls, value: Any) -> str:
        normalised = cls._normalise(value)
        aliases = {
            "valid_observed": "observed",
            "observed_valid": "observed",
            "no_source": "no_source",
            "no_recorded_source_or_observation": "no_source",
        }
        return aliases.get(normalised, normalised)

    @classmethod
    def _ordinal_level_alias(cls, value: Any) -> str:
        """Normalise semantic labels such as ``level_0`` to parent value ``0``."""

        normalised = cls._normalise(value)
        match = re.fullmatch(r"(?:.*_)?level_?([0-9]+)", normalised)
        return match.group(1) if match else normalised

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if pd.notna(number) else None

    @classmethod
    def _registered_parent_path(cls, summary: Dict[str, Any]) -> Optional[Path]:
        candidates: List[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                upstream_step = value.get("upstream_step") or value.get(
                    "requested_step"
                )
                if isinstance(upstream_step, str):
                    for key, path in value.items():
                        key_text = re.sub(
                            r"[^a-z0-9]+", "_", str(key).strip().lower()
                        ).strip("_")
                        if (
                            "path" in key_text
                            and isinstance(path, str)
                            and path.strip()
                            and Path(path).suffix.lower() in {".csv", ".tsv"}
                        ):
                            candidates.append(path)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(summary)
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}:
                return path
        return None

    @classmethod
    def _reconciliation_table_path(
        cls, summary: Dict[str, Any], out_dir: Path
    ) -> Optional[Path]:
        for path in cls._reconciliation_candidate_paths(summary, out_dir):
            try:
                columns = set(pd.read_csv(path, nrows=1).columns)
            except Exception:
                continue
            variable_present = bool(
                columns.intersection({"source_variable", "variable", "exposure"})
            )
            requested_semantics_present = bool(
                columns.intersection(
                    {
                        "requested_role",
                        "requested_estimate_type",
                        "estimate_type",
                        "stratum_type",
                        "row_role",
                        "row_type",
                    }
                )
            )
            support_present = bool(
                columns.intersection(
                    {
                        "registered_output_status",
                        "row_supported",
                        "registered_supported",
                        "registered_row_supported",
                    }
                )
            )
            if (
                variable_present
                and requested_semantics_present
                and support_present
                and "registered_n" in columns
            ):
                return path
        return None

    @classmethod
    def _reconciliation_candidate_paths(
        cls, summary: Dict[str, Any], out_dir: Path
    ) -> List[Path]:
        names: List[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str) and "reconciliation" in value.lower():
                names.append(value)
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(summary.get("output_files"))
        collect(summary.get("outputs"))
        candidates = [out_dir / name for name in names]
        candidates.extend(sorted(out_dir.glob("*reconciliation*.csv")))
        resolved: List[Path] = []
        seen: Set[Path] = set()
        for path in candidates:
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            canonical = path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            resolved.append(path)
        return resolved

    @classmethod
    def _canonical_current_rows(cls, current: pd.DataFrame) -> pd.DataFrame:
        if {
            "source_variable",
            "requested_stratum",
            "requested_role",
            "registered_output_status",
        }.issubset(current.columns):
            return current
        rows: List[Dict[str, Any]] = []
        for _, source in current.iterrows():
            def first_value(*names: str) -> Any:
                for name in names:
                    if name not in current.columns:
                        continue
                    value = source.get(name)
                    if value is not None and not pd.isna(value):
                        return value
                return None

            variable = first_value("source_variable", "variable", "exposure")
            if variable is None:
                continue
            explicit_role = first_value("requested_role", "row_role", "row_type")
            explicit_role_normalised = cls._normalise(explicit_role)
            stratum_type = cls._normalise(
                first_value("stratum_type", "requested_group_type")
            )
            estimate_type = cls._normalise(
                first_value("requested_estimate_type", "estimate_type")
            )
            requested_level = first_value("requested_level", "level")
            requested_status = first_value(
                "requested_source_status", "source_status"
            )
            requested_stratum_raw = first_value(
                "requested_stratum", "stratum", "requested_group_value"
            )

            if (
                requested_level is not None
                or stratum_type in {"exposure_level", "level"}
                or explicit_role_normalised in {
                    "level",
                    "ordinal_level",
                    "required_valid_ordinal_level",
                }
            ):
                role = "required_valid_ordinal_level"
                requested_stratum = (
                    requested_level
                    if requested_level is not None
                    else requested_stratum_raw
                )
            elif (
                stratum_type == "source_status"
                or explicit_role_normalised
                in {"source_status", "required_source_status"}
                or (
                    explicit_role is None
                    and requested_status is not None
                    and estimate_type == "outcome_risk"
                )
            ):
                role = "required_source_status"
                requested_stratum = (
                    requested_status
                    if requested_status is not None
                    else requested_stratum_raw
                )
            elif (
                stratum_type == "distribution"
                or explicit_role_normalised
                in {"distribution", "required_continuous_representation"}
                or "distribution" in estimate_type
            ):
                role = "required_continuous_representation"
                requested_stratum = (
                    requested_status
                    if requested_status is not None
                    else requested_stratum_raw
                )
            elif explicit_role is not None:
                role = str(explicit_role)
                requested_stratum = requested_stratum_raw
            else:
                continue
            supported = first_value(
                "row_supported",
                "registered_supported",
                "registered_row_supported",
            )
            status_text = first_value("registered_output_status")
            if supported is None and status_text is not None:
                supported = cls._normalise(status_text) == "row_supported"
            if isinstance(supported, str):
                supported = supported.strip().lower() in {"true", "1", "yes"}
            selected_fields = str(
                first_value(
                    "registered_selected_fields",
                    "selected_parent_row_fields",
                    "selected_registered_fields",
                    "selected_parent_row_field_names",
                    "selected_parent_field_names",
                )
                or ""
            )
            selected_field_tokens = {
                cls._normalise(token)
                for token in re.split(r"[;,\s]+", selected_fields)
                if token.strip()
            }
            n_field = first_value("registered_n_field")
            if n_field is None and (
                "n" in selected_field_tokens
                or re.search(r"(?:^|[,\s])n=n(?:[,\s]|$)", selected_fields)
            ):
                n_field = "n"
            event_field = first_value("registered_event_n_field")
            if event_field is None and (
                "event_n" in selected_field_tokens
                or "event_n=event_n" in selected_fields
            ):
                event_field = "event_n"
            risk_field = first_value(
                "registered_risk_field", "registered_outcome_risk_field"
            )
            if risk_field is None and (
                "outcome_risk" in selected_field_tokens
                or "outcome_risk=outcome_risk" in selected_fields
            ):
                risk_field = "outcome_risk"
            distribution_fields: Dict[str, Any] = {}
            for statistic in ("median", "q25", "q75"):
                field = first_value(f"registered_{statistic}_field")
                if field is None and statistic in selected_field_tokens:
                    field = statistic
                distribution_fields[f"registered_{statistic}_field"] = field
            rows.append(
                {
                    "source_variable": variable,
                    "requested_stratum": requested_stratum,
                    "requested_role": role,
                    "registered_output_status": (
                        "row_supported" if supported is True else "row_not_supported"
                    ),
                    "registered_n": source.get("registered_n"),
                    "registered_event_n": first_value("registered_event_n"),
                    "registered_risk": first_value(
                        "registered_risk", "registered_outcome_risk"
                    ),
                    "registered_n_field": n_field,
                    "registered_event_n_field": event_field,
                    "registered_risk_field": risk_field,
                    "registered_median": first_value("registered_median"),
                    "registered_q25": first_value("registered_q25"),
                    "registered_q75": first_value("registered_q75"),
                    **distribution_fields,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def _parent_match(
        cls, parent: pd.DataFrame, row: pd.Series
    ) -> pd.DataFrame:
        required_parent = {
            "exposure",
            "group_type",
            "group_value",
            "estimate_type",
        }
        if not required_parent.issubset(parent.columns):
            return parent.iloc[0:0]
        source = cls._normalise(row.get("source_variable"))
        role = cls._normalise(row.get("requested_role"))
        target = row.get("requested_stratum")
        work = parent[
            parent["exposure"].map(cls._normalise).eq(source)
        ].copy()

        if "ordinal_level" in role:
            return work[
                work["group_type"].map(cls._normalise).eq("exposure_level")
                & work["group_value"]
                .map(cls._ordinal_level_alias)
                .eq(cls._ordinal_level_alias(target))
                & work["estimate_type"].map(cls._normalise).eq("outcome_risk")
            ]
        if "source_status" in role:
            target_status = cls._status_alias(target)
            return work[
                work["group_type"].map(cls._normalise).eq("source_state")
                & work["group_value"].map(cls._status_alias).eq(target_status)
                & work["estimate_type"].map(cls._normalise).eq("outcome_risk")
            ]
        if "continuous_representation" in role:
            return work[
                work["group_type"].map(cls._normalise).eq("continuous_summary")
                & work["estimate_type"]
                .map(cls._normalise)
                .eq("continuous_distribution")
            ]
        return work.iloc[0:0]

    @classmethod
    def _trace_issues(
        cls, current: pd.DataFrame, parent: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        canonical = cls._canonical_current_rows(current)
        for _, row in canonical.iterrows():
            role = cls._normalise(row.get("requested_role"))
            if not any(
                token in role
                for token in (
                    "ordinal_level",
                    "source_status",
                    "continuous_representation",
                )
            ):
                continue
            matched = cls._parent_match(parent, row)
            label = (
                f"{row.get('source_variable')}:{row.get('requested_stratum')}"
            )
            reported_status = cls._normalise(row.get("registered_output_status"))
            if len(matched) == 0:
                if reported_status == "row_supported":
                    issues.append(
                        {"row": label, "issue": "false_parent_support"}
                    )
                continue
            if len(matched) != 1:
                issues.append(
                    {"row": label, "issue": f"ambiguous_parent_rows={len(matched)}"}
                )
                continue
            expected = matched.iloc[0]
            if reported_status != "row_supported":
                issues.append(
                    {"row": label, "issue": "supported_parent_row_reported_missing"}
                )
                continue

            expected_n = cls._as_float(expected.get("n"))
            reported_n = cls._as_float(row.get("registered_n"))
            if expected_n is not None and (
                reported_n is None or abs(reported_n - expected_n) > 1e-8
            ):
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_n_mismatch",
                        "expected": expected_n,
                        "reported": reported_n,
                    }
                )
            n_field = cls._normalise(row.get("registered_n_field"))
            if expected_n is not None and n_field != "n":
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_n_field_must_be_n",
                        "reported": n_field,
                    }
                )

            if "continuous_representation" in role:
                reported_risk = cls._as_float(row.get("registered_risk"))
                if reported_risk is not None:
                    issues.append(
                        {
                            "row": label,
                            "issue": "continuous_distribution_has_false_risk",
                            "reported": reported_risk,
                        }
                    )
                for statistic in ("median", "q25", "q75"):
                    expected_value = cls._as_float(expected.get(statistic))
                    if expected_value is None:
                        continue
                    reported_value = cls._as_float(
                        row.get(f"registered_{statistic}")
                    )
                    if (
                        reported_value is None
                        or abs(reported_value - expected_value) > 1e-10
                    ):
                        issues.append(
                            {
                                "row": label,
                                "issue": f"registered_{statistic}_mismatch",
                                "expected": expected_value,
                                "reported": reported_value,
                            }
                        )
                    reported_field = cls._normalise(
                        row.get(f"registered_{statistic}_field")
                    )
                    if reported_field != statistic:
                        issues.append(
                            {
                                "row": label,
                                "issue": (
                                    f"registered_{statistic}_field_must_be_"
                                    f"{statistic}"
                                ),
                                "reported": reported_field,
                            }
                        )
                continue

            expected_event_n = cls._as_float(expected.get("event_n"))
            reported_event_n = cls._as_float(row.get("registered_event_n"))
            if expected_event_n is not None and (
                reported_event_n is None
                or abs(reported_event_n - expected_event_n) > 1e-8
            ):
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_event_n_mismatch",
                        "expected": expected_event_n,
                        "reported": reported_event_n,
                    }
                )
            event_n_field = cls._normalise(row.get("registered_event_n_field"))
            if expected_event_n is not None and event_n_field != "event_n":
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_event_n_field_must_be_event_n",
                        "reported": event_n_field,
                    }
                )

            expected_risk = cls._as_float(expected.get("outcome_risk"))
            reported_risk = cls._as_float(row.get("registered_risk"))
            if expected_risk is not None and (
                reported_risk is None
                or abs(reported_risk - expected_risk) > 1e-10
            ):
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_risk_mismatch",
                        "expected": expected_risk,
                        "reported": reported_risk,
                    }
                )
            risk_field = cls._normalise(row.get("registered_risk_field"))
            if expected_risk is not None and risk_field != "outcome_risk":
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_risk_field_must_be_outcome_risk",
                        "reported": risk_field,
                    }
                )
        return issues

    @classmethod
    def _declared_range_flag_issues(
        cls,
        *,
        summary: Dict[str, Any],
        current: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Require detailed rows for range flags declared in the summary."""

        declared: Set[str] = set()

        def collect_declared(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if (
                        cls._normalise(key) == "range_flag_counts"
                        and isinstance(child, dict)
                    ):
                        declared.update(cls._normalise(flag) for flag in child)
                    collect_declared(child)
            elif isinstance(value, list):
                for child in value:
                    collect_declared(child)

        collect_declared(summary)
        declared.discard("")
        if not declared:
            return []

        present: Set[str] = set()
        for column in ("local_range_flag", "range_flag", "requested_range_flag"):
            if column in current.columns:
                present.update(
                    cls._normalise(value)
                    for value in current[column].dropna().tolist()
                )
        role_columns = (
            "requested_role",
            "row_role",
            "row_type",
            "stratum_type",
            "requested_group_type",
        )
        value_columns = (
            "requested_stratum",
            "stratum",
            "requested_group_value",
            "requested_row",
        )
        for _, row in current.iterrows():
            roles = {
                cls._normalise(row.get(column))
                for column in role_columns
                if column in current.columns
            }
            if not any("range_flag" in role for role in roles):
                continue
            present.update(
                cls._normalise(row.get(column))
                for column in value_columns
                if column in current.columns and pd.notna(row.get(column))
            )
        present.discard("")

        issues: List[Dict[str, Any]] = []
        for expected in sorted(declared):
            if any(
                expected == actual
                or expected in actual
                or actual in expected
                for actual in present
            ):
                continue
            issues.append(
                {
                    "row": expected,
                    "issue": "missing_declared_range_flag_row",
                }
            )
        return issues

    @classmethod
    def _percentage_issues(cls, out_dir: Path) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for path in sorted(out_dir.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if "row_type" not in frame or "percentage_of_valid_observed" not in frame:
                continue
            source_rows = frame[
                frame["row_type"].map(cls._normalise).eq("source_status")
            ]
            for index, row in source_rows.iterrows():
                fraction = cls._as_float(row.get("percentage_of_valid_observed"))
                pct = cls._as_float(row.get("percentage_of_valid_observed_pct"))
                # A source-status row is outside vs inside the observed subset;
                # its denominator is the locked cohort. Even a numerically
                # bounded value is semantically wrong under a column named
                # ``percentage_of_valid_observed``.
                if fraction is not None or pct is not None:
                    issues.append(
                        {
                            "file": path.name,
                            "row": int(index),
                            "status": row.get("status"),
                            "issue": "source_status_percentage_field_not_applicable",
                            "reported_fraction": fraction,
                            "reported_pct": pct,
                        }
                    )
        return issues

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        out_dir: Path,
    ) -> List[ValidationFinding]:
        parent_path = self._registered_parent_path(step_summary)
        current_path = self._reconciliation_table_path(step_summary, Path(out_dir))
        if parent_path is None:
            return []
        if current_path is None:
            candidates = self._reconciliation_candidate_paths(
                step_summary, Path(out_dir)
            )
            if not candidates:
                return []
            candidate_columns: Dict[str, Any] = {}
            for candidate in candidates:
                try:
                    candidate_columns[str(candidate)] = list(
                        pd.read_csv(candidate, nrows=1).columns
                    )
                except Exception as exc:
                    candidate_columns[str(candidate)] = {
                        "read_error": str(exc)[:300]
                    }
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Step {step.step_id} declared a reconciliation CSV, but "
                        "its schema does not expose the variable, requested-row "
                        "semantics, registered support flag, and registered_n "
                        "needed for parent-table trace verification."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "parent_table": str(parent_path),
                        "issue": "reconciliation_schema_unrecognised",
                        "candidate_columns": candidate_columns,
                    },
                )
            ]
        try:
            parent = pd.read_csv(parent_path)
            current = pd.read_csv(current_path)
        except Exception:
            return []
        issues = self._trace_issues(current, parent)
        issues.extend(
            self._declared_range_flag_issues(
                summary=step_summary,
                current=current,
            )
        )
        issues.extend(self._percentage_issues(Path(out_dir)))
        if not issues:
            return []
        return [
            ValidationFinding(
                validator=self.name,
                severity="error",
                message=(
                    f"Registered-output reconciliation in step {step.step_id} "
                    f"does not trace to the selected parent table ({len(issues)} "
                    "issue(s)). Match outcome-risk rows with "
                    "estimate_type=outcome_risk and use n/event_n/outcome_risk; "
                    "match the parent grouping dimension and value (for example "
                    "group_type/group_value), including level_0 versus 0 aliases; "
                    "map semantic request roles to the parent's actual grouping "
                    "labels (for example an ordinal *_level request may map to "
                    "exposure_level); record selected parent field names; preserve "
                    "detailed rows for every range flag declared in the summary; "
                    "match continuous summaries only with continuous_distribution; "
                    "normalise observed and valid-observed source aliases; keep "
                    "prevalence rows separate. Source-status percentages must use "
                    "the locked cohort denominator (or be NA), never the "
                    "valid-observed denominator."
                ),
                detail={
                    "step_id": step.step_id,
                    "parent_table": str(parent_path),
                    "reconciliation_table": str(current_path),
                    "issues": issues[:30],
                },
            )
        ]


class CrossStepSourceStatusValidator:
    """Keep source-status denominators stable across completed run steps.

    A data-quality step may lock the number of source-consistent observed
    values for a measured concept.  Later descriptive/model steps are free to
    transform the value, but they must not silently redefine which rows were
    observed when they report the same concept on the same cohort.

    The gate is deliberately evidence-driven: it only compares explicit
    ``source_summary`` blocks against an earlier machine-readable
    ``missingness.source_status_counts`` block, and only when the category
    totals match.  Missing or ambiguous evidence is therefore skipped rather
    than guessed.
    """

    name = "cross_step_source_status"

    @staticmethod
    def _normalise(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    @classmethod
    def _is_valid_observed_label(cls, value: Any) -> bool:
        tokens = set(cls._normalise(value).split("_"))
        return (
            "invalid" not in tokens
            and "valid" in tokens
            and bool(tokens.intersection({"observed", "measured", "value", "level"}))
        )

    @classmethod
    def _status_role(cls, value: Any) -> Optional[str]:
        tokens = set(cls._normalise(value).split("_"))
        if cls._is_valid_observed_label(value):
            return "valid_observed"
        if "no" in tokens and tokens.intersection({"source", "recorded", "observation"}):
            return "no_source"
        if (
            tokens.intersection({"measured", "observed"})
            and "missing" in tokens
            and tokens.intersection({"summary", "value"})
        ):
            return "measured_summary_missing"
        if tokens.intersection({"contradictory", "inconsistent", "invalid"}):
            return "contradictory_invalid"
        return None

    @staticmethod
    def _as_count(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not pd.notna(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    @classmethod
    def _prior_locks(
        cls, completed_step_records: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        locks: List[Dict[str, Any]] = []
        successful_statuses = {
            "ok",
            "complete",
            "completed",
            "repaired",
            "runner_repaired",
        }
        for record_index, record in enumerate(completed_step_records):
            status = str(record.get("status") or "").strip().lower()
            if status and status not in successful_statuses:
                continue
            summary = record.get("step_summary")
            if not isinstance(summary, dict):
                continue
            missingness = summary.get("missingness")
            if not isinstance(missingness, dict):
                continue
            source_counts = missingness.get("source_status_counts")
            if not isinstance(source_counts, dict):
                continue
            for scope, by_concept in source_counts.items():
                if not isinstance(by_concept, dict):
                    continue
                for concept, categories in by_concept.items():
                    if not isinstance(categories, dict):
                        continue
                    parsed = {
                        str(category): count
                        for category, raw_count in categories.items()
                        if (count := cls._as_count(raw_count)) is not None
                    }
                    valid_counts = [
                        count
                        for category, count in parsed.items()
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) != 1 or not parsed:
                        continue
                    locks.append(
                        {
                            "concept": cls._normalise(concept),
                            "source_summary": str(concept),
                            "scope": str(scope),
                            "total_n": sum(parsed.values()),
                            "valid_observed_n": valid_counts[0],
                            "step_id": str(record.get("step_id") or "prior_step"),
                            "record_index": record_index,
                        }
                    )
        return locks

    @classmethod
    def _current_status_blocks(cls, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []

        declarations: List[Dict[str, str]] = []

        def collect_declarations(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                summary_variable = value.get("summary_variable")
                if isinstance(summary_variable, str) and summary_variable.strip():
                    alias = cls._normalise(path[-1] if path else "")
                    alias = re.sub(r"_definition$", "", alias)
                    declarations.append(
                        {
                            "alias": alias,
                            "source_summary": summary_variable,
                            "base": re.sub(
                                r"_(?:first|max|min|mean|median)$",
                                "",
                                cls._normalise(summary_variable),
                            ),
                        }
                    )
                for key, child in value.items():
                    collect_declarations(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect_declarations(child, (*path, str(index)))

        collect_declarations(summary)

        def declared_source_for(path: tuple[str, ...]) -> Optional[str]:
            hint = cls._normalise(path[-1] if path else "")
            hint = re.sub(
                r"_(?:measurement_)?status(?:_counts)?$", "", hint
            )
            exact = [
                declaration
                for declaration in declarations
                if declaration["alias"] == hint
            ]
            if len(exact) == 1:
                return exact[0]["source_summary"]
            semantic = [
                declaration
                for declaration in declarations
                if declaration["base"] == hint
                or declaration["base"].startswith(f"{hint}_")
                or hint.startswith(f"{declaration['base']}_")
            ]
            if len(semantic) == 1:
                return semantic[0]["source_summary"]
            return None

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                direct_counts = value.get("source_status_counts")
                source_columns = value.get("source_columns")
                if isinstance(direct_counts, dict) and isinstance(source_columns, list):
                    source_summary = next(
                        (
                            str(column)
                            for column in source_columns
                            if isinstance(column, str) and column.strip()
                        ),
                        None,
                    )
                    parsed_direct = [
                        (str(category), count)
                        for category, raw_count in direct_counts.items()
                        if (count := cls._as_count(raw_count)) is not None
                    ]
                    valid_counts = [
                        count
                        for category, count in parsed_direct
                        if cls._is_valid_observed_label(category)
                    ]
                    if source_summary and len(valid_counts) == 1 and parsed_direct:
                        present_roles = {
                            role
                            for category, _ in parsed_direct
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join(
                                    (*path, "source_status_counts")
                                ),
                                "total_n": sum(count for _, count in parsed_direct),
                                "valid_observed_n": valid_counts[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                # Some reconciliation summaries store one concept per mapping
                # with ``counts`` and ``valid_observed_n`` rather than an
                # explicit source_columns list.  The concept key is still a
                # machine-readable source summary name, so preserve the same
                # four-category completeness and denominator lock.
                concept_counts = value.get("counts")
                concept_valid = cls._as_count(value.get("valid_observed_n"))
                if (
                    isinstance(concept_counts, dict)
                    and concept_valid is not None
                    and path
                ):
                    parsed_concept = [
                        (str(category), count)
                        for category, raw_count in concept_counts.items()
                        if (count := cls._as_count(raw_count)) is not None
                    ]
                    valid_counts = [
                        count
                        for category, count in parsed_concept
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) == 1 and parsed_concept:
                        present_roles = {
                            role
                            for category, _ in parsed_concept
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        source_summary = str(path[-1])
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join((*path, "counts")),
                                "total_n": sum(
                                    count for _, count in parsed_concept
                                ),
                                "valid_observed_n": valid_counts[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                if path and any(
                    "source_status_count" in cls._normalise(segment)
                    for segment in path
                ):
                    parsed_nested = [
                        (str(category), count)
                        for category, raw in value.items()
                        if isinstance(raw, dict)
                        and (
                            count := cls._as_count(
                                raw.get("count", raw.get("n"))
                            )
                        )
                        is not None
                    ]
                    valid_nested = [
                        count
                        for category, count in parsed_nested
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_nested) == 1 and parsed_nested:
                        present_roles = {
                            role
                            for category, _ in parsed_nested
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        source_summary = str(path[-1])
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join(path),
                                "total_n": sum(
                                    count for _, count in parsed_nested
                                ),
                                "valid_observed_n": valid_nested[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                source_summary = value.get("source_summary")
                rows = value.get("measurement_status_counts")
                if not isinstance(rows, list):
                    rows = value.get("counts")
                if isinstance(source_summary, str) and isinstance(rows, list):
                    parsed: List[tuple[str, int]] = []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        count = cls._as_count(row.get("count", row.get("n")))
                        category = row.get("category", row.get("status"))
                        if count is not None and category is not None:
                            parsed.append((str(category), count))
                    valid_counts = [
                        count
                        for category, count in parsed
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) == 1 and parsed:
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join(path) or "step_summary",
                                "total_n": sum(count for _, count in parsed),
                                "valid_observed_n": valid_counts[0],
                            }
                        )
                # Newer descriptive summaries may expose the same contract as
                # scalar counts under ``missingness_and_measurement_status``
                # instead of a list of category rows.  Bind the status block to
                # an explicit nearby ``summary_variable`` declaration; never
                # guess from a human label alone.
                scalar_valid = cls._as_count(value.get("observed_valid_summary_n"))
                scalar_total = cls._as_count(value.get("denominator_n"))
                if scalar_valid is not None and scalar_total is not None:
                    scalar_source = value.get("source_summary") or value.get(
                        "summary_variable"
                    )
                    if not isinstance(scalar_source, str) or not scalar_source.strip():
                        scalar_source = declared_source_for(path)
                    if isinstance(scalar_source, str) and scalar_source.strip():
                        blocks.append(
                            {
                                "concept": cls._normalise(scalar_source),
                                "source_summary": scalar_source,
                                "path": ".".join(path) or "step_summary",
                                "total_n": scalar_total,
                                "valid_observed_n": scalar_valid,
                            }
                        )
                for key, child in value.items():
                    visit(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return blocks

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[ValidationFinding]:
        locks = self._prior_locks(completed_step_records)
        if not locks:
            return []

        findings: List[ValidationFinding] = []
        compared: Set[tuple[str, int, int]] = set()
        for current in self._current_status_blocks(step_summary):
            candidates = [
                lock
                for lock in locks
                if lock["concept"] == current["concept"]
                and lock["total_n"] == current["total_n"]
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda lock: (
                    "analytic" not in self._normalise(lock["scope"]),
                    -int(lock["record_index"]),
                )
            )
            expected = candidates[0]
            comparison_key = (
                current["concept"],
                current["total_n"],
                current["valid_observed_n"],
            )
            if comparison_key in compared:
                continue
            compared.add(comparison_key)
            missing_status_roles = current.get("missing_status_roles") or []
            if missing_status_roles:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Incomplete source-status schema for "
                            f"{current['source_summary']} in step {step.step_id}: "
                            f"missing categories {missing_status_roles}. Report "
                            "all four source-status categories explicitly, using "
                            "zero counts for supported zero-frequency strata "
                            "rather than omitting their machine-summary keys."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "summary_path": current["path"],
                            "source_summary": current["source_summary"],
                            "cohort_n": current["total_n"],
                            "missing_status_roles": missing_status_roles,
                            "expected_from_step": expected["step_id"],
                        },
                    )
                )
            if current["valid_observed_n"] == expected["valid_observed_n"]:
                continue
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Source-status denominator drift for "
                        f"{current['source_summary']}: step {step.step_id} reports "
                        f"{current['valid_observed_n']} valid observed rows of "
                        f"{current['total_n']}, but completed step "
                        f"{expected['step_id']} locked "
                        f"{expected['valid_observed_n']} for the same concept and "
                        "cohort. Preserve the earlier source-status, variable-type, "
                        "and retain/flag range semantics instead of redefining "
                        "validity in this step."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "summary_path": current["path"],
                        "source_summary": current["source_summary"],
                        "cohort_n": current["total_n"],
                        "reported_valid_observed_n": current["valid_observed_n"],
                        "expected_valid_observed_n": expected["valid_observed_n"],
                        "expected_from_step": expected["step_id"],
                        "expected_scope": expected["scope"],
                    },
                )
            )
        return findings


# ---------------------------------------------------------------------------
# StatisticalValidator
# ---------------------------------------------------------------------------


class StatisticalValidator:
    """Cross-check the artefacts a script produced against the cohort."""

    name = "statistical_validator"

    def audit(
        self,
        *,
        context: ResearchContext,
        cohort_path: Path,
        step: AnalysisStep,
        out_dir: Path,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        outcome = context.target_outcome

        primary_exposure = step_summary.get("primary_exposure")
        if isinstance(primary_exposure, Mapping):
            status = str(
                primary_exposure.get("reconciliation_status")
                or primary_exposure.get("status")
                or ""
            ).strip().lower()
            cohort_n = self._finite_nonnegative_count(step_summary.get("cohort_n"))
            missing_n = self._finite_nonnegative_count(
                primary_exposure.get("missing_n")
            )
            counts = primary_exposure.get("counts")
            usable_group_n = 0.0
            counted_total_n = 0.0
            if isinstance(counts, Mapping):
                for label, value in counts.items():
                    normalized_label = str(label).strip().lower()
                    numeric = self._finite_nonnegative_count(value)
                    if numeric is not None:
                        counted_total_n += numeric
                    if any(
                        token in normalized_label
                        for token in ("unavailable", "missing", "unknown")
                    ):
                        continue
                    if numeric is not None:
                        usable_group_n += numeric
            explicit_usable_counts = [
                self._finite_nonnegative_count(primary_exposure.get(key))
                for key in (
                    "available_n",
                    "nonmissing_n",
                    "observed_n",
                    "reconciled_n",
                    "usable_n",
                )
                if key in primary_exposure
            ]
            explicit_no_usable = bool(explicit_usable_counts) and all(
                value is not None and value <= 0 for value in explicit_usable_counts
            )
            all_missing_by_cohort = (
                cohort_n is not None
                and cohort_n > 0
                and missing_n is not None
                and missing_n >= cohort_n
                and usable_group_n <= 0
            )
            all_missing_by_counts = counted_total_n > 0 and usable_group_n <= 0
            if status in {
                "unavailable",
                "failed",
                "error",
                "not_available",
                "fail_closed",
                "failed_closed",
                "fail-closed",
                "failed-closed",
            } or all_missing_by_cohort or all_missing_by_counts or explicit_no_usable:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            "The completed step declares a primary exposure but "
                            "no cohort row has a usable reconciled exposure value. "
                            "Repair the metadata/value binding or fail the step; "
                            "do not publish an all-unavailable primary-exposure result."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "reconciliation_status": status or None,
                            "cohort_n": cohort_n,
                            "missing_n": missing_n,
                            "usable_group_n": usable_group_n,
                        },
                    )
                )

        # Controlled ordered-group summaries remain agent-authored, but every
        # denominator, interval, descriptive statistic, and trend test is
        # independently replayed from the locked cohort before publication.
        findings.extend(
            ordered_stratified_numeric_findings(
                cohort_path=cohort_path,
                step=step,
                out_dir=out_dir,
                step_summary=step_summary,
            )
        )
        findings.extend(
            trajectory_phenotyping_artifact_findings(
                context=context,
                cohort_path=cohort_path,
                step=step,
                out_dir=out_dir,
                step_summary=step_summary,
            )
        )

        # 1. Recompute outcome incidence and compare with reported.
        if outcome:
            try:
                df = pd.read_parquet(cohort_path)
                if outcome in df.columns:
                    truth = float(df[outcome].dropna().astype(int).mean())
                    reported = step_summary.get("outcome_rate")
                    if reported is not None:
                        diff = abs(float(reported) - truth)
                        if diff > 1e-3:
                            findings.append(ValidationFinding(
                                validator=self.name, severity="error",
                                message=(
                                    f"Reported outcome rate {reported:.4f} disagrees with "
                                    f"cohort recompute {truth:.4f} (Δ={diff:.4f})."
                                ),
                                detail={"reported": reported, "truth": truth},
                            ))
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Could not recompute outcome rate: {exc}",
                ))

        # 2. Primary-association OR cross-check (T1.6).
        #    The mock pipeline writes ``primary_association.csv`` with one
        #    row per coefficient (variable, coef, odds_ratio, ...). The
        #    step_summary records ``primary_or`` for the predictor. Re-
        #    derive the OR from the table and flag if the two disagree by
        #    more than 1e-3 — that mirrors the outcome-rate check above.
        pa_csv = out_dir / "primary_association.csv"
        if pa_csv.exists():
            try:
                pa = pd.read_csv(pa_csv)
                reported = step_summary.get("primary_or")
                predictor = step_summary.get("predictor")
                if reported is not None and predictor and "variable" in pa.columns and "odds_ratio" in pa.columns:
                    match = pa.loc[pa["variable"] == predictor, "odds_ratio"]
                    if not match.empty:
                        recomputed = float(match.iloc[0])
                        diff = abs(float(reported) - recomputed)
                        if diff > 1e-3:
                            findings.append(ValidationFinding(
                                validator=self.name, severity="error",
                                message=(
                                    f"Reported primary OR {reported:.4f} disagrees "
                                    f"with recompute from {pa_csv.name} ({recomputed:.4f}, "
                                    f"Δ={diff:.4f})."
                                ),
                                detail={"reported": reported, "recomputed": recomputed,
                                        "predictor": predictor},
                            ))
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Could not parse primary_association.csv: {exc}",
                ))

        # 3. Sanity: the script must have produced some artefact.
        if not any(out_dir.iterdir()):
            findings.append(ValidationFinding(
                validator=self.name, severity="error",
                message=f"Step '{step.step_id}' produced no output artefacts.",
            ))

        # 4. Codex-grade train/test performance metrics (T1.8). Whenever a
        #    step writes ``model_performance_train_test.csv``, re-validate
        #    AUC ∈ [0.5, 1.0], Brier ∈ [0, 0.5] and calibration slope
        #    ∈ [0.5, 2.0]. Out-of-range values are *errors* — they
        #    indicate that the held-out test produced sign-flipped or
        #    grossly mis-calibrated predictions.
        perf_csv = out_dir / "model_performance_train_test.csv"
        if perf_csv.exists():
            try:
                perf = pd.read_csv(perf_csv)
                for _, row in perf.iterrows():
                    model = str(row.get("model", "?"))
                    auc = row.get("auc", float("nan"))
                    brier = row.get("brier", float("nan"))
                    cal_slope = row.get("calibration_slope", float("nan"))
                    if pd.notna(auc) and not (0.5 <= float(auc) <= 1.0):
                        findings.append(ValidationFinding(
                            validator=self.name, severity="error",
                            message=(
                                f"Model '{model}' held-out AUC {auc:.3f} outside "
                                "the plausible discriminative range [0.5, 1.0]."
                            ),
                            detail={"model": model, "auc": float(auc)},
                        ))
                    if pd.notna(brier) and not (0.0 <= float(brier) <= 0.5):
                        findings.append(ValidationFinding(
                            validator=self.name, severity="warning",
                            message=(
                                f"Model '{model}' Brier score {brier:.3f} outside "
                                "the plausible range [0, 0.5]."
                            ),
                            detail={"model": model, "brier": float(brier)},
                        ))
                    if pd.notna(cal_slope) and not (0.5 <= float(cal_slope) <= 2.0):
                        findings.append(ValidationFinding(
                            validator=self.name, severity="warning",
                            message=(
                                f"Model '{model}' calibration slope {cal_slope:.3f} "
                                "outside the well-calibrated range [0.5, 2.0]."
                            ),
                            detail={"model": model, "calibration_slope": float(cal_slope)},
                        ))
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Could not parse {perf_csv.name}: {exc}",
                ))

        # 5. Degenerate-partition disclosure caution (clustering / trajectory).
        #    When a step emits a cluster-size distribution, surface an OBJECTIVE
        #    degeneracy fact — a single group, or one dominant cluster plus a
        #    near-empty (<1% of cohort) pocket — as an advisory caution. This is
        #    the agent-facing mirror of the post-hoc phenotype validity check:
        #    silhouette / ARI computed on such a partition are inflated by
        #    outlier isolation and must NOT be reported as evidence of robust
        #    subphenotypes without disclosing the size imbalance. It is a
        #    WARNING, never a block: a degenerate partition is still a legitimate
        #    (negative) finding to report honestly — the rule layer surfaces the
        #    fact and never imposes k, algorithm, scaling or outlier handling.
        deg = self._degenerate_partition(out_dir, step_summary)
        if deg is not None:
            findings.append(ValidationFinding(
                validator=self.name, severity="warning",
                message=(
                    f"Degenerate cluster partition ({deg['reason']}). Silhouette "
                    "and resampling ARI on such a partition are inflated by "
                    "outlier isolation, not evidence of separated subphenotypes; "
                    "disclose the cluster sizes and do not present this as a "
                    "robust multi-subphenotype solution."
                ),
                detail=deg,
            ))

        return findings

    @staticmethod
    def _finite_nonnegative_count(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or numeric < 0:
            return None
        return numeric

    @staticmethod
    def _degenerate_partition(
        out_dir: Path, step_summary: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a degeneracy descriptor when a cluster-size distribution is
        objectively broken, else ``None``.

        Reuses the post-hoc scorecard threshold (single group, or smallest
        group < 1% of the cohort). Reads ``cluster_sizes.csv`` (columns include
        ``n`` and/or ``pct``) or a ``cluster_sizes`` mapping/list in the step
        summary. Stays silent (``None``) when the step produced no cluster-size
        evidence — absence is not degeneracy.
        """
        ns: List[float] = []
        # Prefer the emitted table; fall back to the step summary.
        try:
            csv_path = out_dir / "cluster_sizes.csv"
            if not csv_path.exists():
                for p in out_dir.glob("*cluster_sizes*.csv"):
                    csv_path = p
                    break
            if csv_path.exists():
                tbl = pd.read_csv(csv_path)
                if "n" in tbl.columns:
                    ns = [float(x) for x in tbl["n"].dropna().tolist()]
                elif "pct" in tbl.columns:
                    ns = [float(x) for x in tbl["pct"].dropna().tolist()]
        except Exception:
            ns = []
        if not ns:
            raw = (step_summary or {}).get("cluster_sizes")
            try:
                if isinstance(raw, dict):
                    ns = [float(v) for v in raw.values()]
                elif isinstance(raw, (list, tuple)):
                    ns = [float(v) for v in raw]
            except Exception:
                ns = []
        ns = [x for x in ns if x is not None and x >= 0]
        if not ns:
            return None
        k = len(ns)
        total = sum(ns)
        if total <= 0:
            return None
        min_frac = min(ns) / total
        if k < 2:
            return {
                "reason": f"single-group solution (k={k})",
                "n_clusters": k,
                "min_cluster_fraction": round(min_frac, 6),
            }
        if min_frac < 0.01:
            return {
                "reason": (
                    f"one dominant cluster plus a near-empty group "
                    f"({min_frac * 100:.2f}% of cohort) across k={k}"
                ),
                "n_clusters": k,
                "min_cluster_fraction": round(min_frac, 6),
            }
        return None


class FigureSourceDataValidator:
    """Verify figure source-data tables are traceable to upstream step tables."""

    name = "figure_source_data"
    _SOURCE_DATA_GLOB = "*source_data*.csv"
    _KEY_COLUMNS = (
        "definition_id",
        "comparison_definition",
        "spec_id",
        "row_id",
        "concept",
        "label",
        # Model-level result tables key rows by the fitted-model label
        # (e.g. adjusted_association.csv from an association model step);
        # the deterministic figure renderer preserves that column verbatim
        # in publication_figure_source_data.csv, so it is a valid trace key.
        "model_label",
        "variable",
        "term",
        "exposure",
        "contrast",
        # Causal effect-estimation steps key each estimated contrast by
        # ``contrast_id`` (e.g. causal_effect.csv); the deterministic forest
        # renderer preserves it verbatim in publication_figure_source_data.csv,
        # so it is a valid per-row trace key. Without it a faithfully-derived
        # causal figure was rejected as "no shared key" (H2 fix3).
        "contrast_id",
        # Ordinal dose-response steps key each graded-exposure level by
        # ``stage`` (e.g. dose_response.csv rows stage=0..K); the figure renderer
        # carries it verbatim into publication_figure_source_data.csv, so it is a
        # valid per-row trace key. Without it a faithfully-derived ordinal forest
        # (odds_ratio per stage identical to the upstream table) was rejected as
        # "no shared key" (E3). The subset + numeric-equality checks below still
        # run, so this only lets a genuinely-traceable figure be verified.
        "stage",
        # A graded-categorical association forest keys each row by the ordinal
        # ``level`` / ``band`` / ``category`` of a single exposure (the exposure
        # NAME is constant across rows; the level is what varies). The association
        # bundle renderer now labels/keys rows by the varying column and carries
        # it into publication_figure_source_data.csv (M1: odds_ratio per
        # sofa2_liver_cat level). Same subset + numeric-equality guards apply.
        "level",
        "band",
        "category",
    )
    _COMPOSITE_KEY_COLUMNS = (
        ("spec_id", "model_id", "term"),
        ("spec_id", "model_id"),
        # Coefficient tables repeat ordinary terms (age/sex/etc.) across
        # multiple models.  ``term`` alone therefore creates a many-to-many
        # join and can falsely compare a primary-model estimate with its
        # complete-case or secondary-model counterpart.
        ("model_id", "term"),
        ("definition_a", "definition_b"),
        ("primary_definition", "comparison_definition"),
    )
    _NUMERIC_COLUMNS = (
        "missing_pct",
        "missing_n",
        "value_missing_pct",
        "value_missing_n",
        "measured_pct",
        "measured_n",
        "measured_one_pct",
        "measured_one_n",
        "n_nonmissing",
        "total_n",
        "n_total",
        "n_included",
        "n_excluded",
        "included_pct_of_rows",
        "overlap_with_primary_n",
        "overlap_with_primary_pct_of_primary",
        "overlap_with_primary_pct_of_definition",
        "moved_in_vs_primary_n",
        "moved_out_vs_primary_n",
        "n_a",
        "n_b",
        "intersection_n",
        "union_n",
        "jaccard",
        "a_in_b_pct",
        "b_in_a_pct",
        "point_estimate",
        "modeled_analytic_n",
        "model_contract_n",
        "event_n",
        "membership_n",
        "estimate",
        "ci_low",
        "ci_high",
        "se",
        "odds_ratio",
        "risk_ratio",
        "risk_difference",
        "p_value",
    )
    _TEXT_COLUMNS = (
        "row_type",
        "group_type",
        "estimate_type",
        "effect_scale",
        "model_id",
        "source_model_id",
        "source_step_id",
        "exposure_source",
        "exposure_expression",
        "exposure_role",
        "analysis_role",
        "analysis_set",
        "baseline_missing_policy",
        "fit_status",
        "fit_method",
        "value_type",
        "replay_mode",
        "coefficient_source_table",
        "coefficient_term",
        "model_contract_source",
        "source_script_sha256",
        "estimability_status",
    )
    _PCT_COUNT_RULES = (
        ("missing_pct", "missing_n", "total_n"),
        ("measured_pct", "measured_n", "total_n"),
        ("measured_pct", "n_nonmissing", "total_n"),
        # Generic long-form figure source-data contract. This catches a common
        # denominator drift where a renderer copies a percent computed against
        # the locked cohort but pairs it with the valid-observed count sum.
        ("percentage", "count", "denominator"),
    )
    _DEFAULT_NUMERIC_ABS_TOL = 1e-9
    # Deterministic summary tables commonly serialize percentages to six
    # decimal places, while figure renderers recompute the same percentages
    # from integer counts at full precision.  Treat only that serialization
    # difference as equivalent; counts, effects, intervals, and p-values keep
    # the stricter default tolerance below.
    _PERCENTAGE_ABS_TOL = 1e-6
    _POSITIONAL_ROW_INDEX_COLUMNS = (
        "source_row_index",
        "_source_row_index",
    )
    _TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".parquet", ".feather"})
    _PURE_RENDER_METHODS = frozenset(
        {
            "chart_generation",
            "figure",
            "figure_generation",
            "plotting",
            "publication_figure",
            "publication_figure_generation",
            "render_figure",
            "visualisation",
            "visualization",
        }
    )
    _PREDICTION_METHODS = frozenset(
        {
            "classification_model",
            "model_validation",
            "prediction",
            "prediction_model",
            "risk_prediction",
        }
    )
    _PREDICTION_SOURCE_ROLES = frozenset(
        {
            "auc",
            "auroc",
            "brier",
            "c_statistic",
            "calibration",
            "calibration_curve",
            "calibration_intercept",
            "calibration_slope",
            "decision_curve",
            "discrimination",
            "false_positive_rate",
            "fpr",
            "horizon_performance",
            "model_performance",
            "observed_risk",
            "predicted_probability",
            "predicted_risk",
            "prediction",
            "prediction_performance",
            "predictions",
            "risk_prediction",
            "risk_predictions",
            "risk_score",
            "roc",
            "roc_curve",
            "true_positive_rate",
            "tpr",
            "validation_performance",
        }
    )
    _PREDICTED_VALUE_ROLES = frozenset(
        {
            "predicted_probability",
            "predicted_risk",
            "prediction",
            "risk_prediction",
            "risk_score",
        }
    )
    _PREDICTED_PROBABILITY_ROLES = frozenset(
        {
            "predicted_probability",
            "predicted_risk",
            "prediction",
            "risk_prediction",
        }
    )
    _PREDICTED_SCORE_ROLES = frozenset({"risk_score"})
    _OBSERVED_OUTCOME_ROLES = frozenset(
        {
            "event",
            "label",
            "observed_outcome",
            "outcome",
            "target",
            "y_true",
        }
    )
    _OBSERVED_CALIBRATION_ROLES = frozenset(
        {
            "observed_probability",
            "observed_rate",
            "observed_risk",
        }
    )
    _PREDICTION_PERFORMANCE_METRICS = frozenset(
        {
            "auc",
            "auroc",
            "brier",
            "brier_score",
            "c_statistic",
            "calibration_intercept",
            "calibration_slope",
            "discrimination",
            "roc_auc",
        }
    )
    _PREDICTION_TIME_ROLES = frozenset(
        {
            "horizon",
            "landmark",
            "prediction_horizon",
            "prediction_time",
            "time_horizon",
        }
    )
    _FALSE_POSITIVE_RATE_ROLES = frozenset({"false_positive_rate", "fpr"})
    _TRUE_POSITIVE_RATE_ROLES = frozenset({"true_positive_rate", "tpr"})
    _UNIT_INTERVAL_PREDICTION_METRICS = frozenset(
        {
            "auc",
            "auroc",
            "brier",
            "brier_score",
            "c_statistic",
            "discrimination",
            "roc_auc",
        }
    )
    _DISCRIMINATION_PREDICTION_METRICS = frozenset(
        {
            "auc",
            "auroc",
            "c_statistic",
            "discrimination",
            "roc_auc",
        }
    )

    @staticmethod
    def _normalise(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    @classmethod
    def _read_tabular(cls, path: Path) -> pd.DataFrame:
        """Read every tabular format accepted by the typed-evidence registry."""

        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".feather":
            return pd.read_feather(path)
        raise ValueError(f"unsupported tabular suffix: {suffix or '<none>'}")

    @classmethod
    def _normalised_method_head(cls, method: Any) -> str:
        normalised = cls._normalise(method)
        return normalised.split("_with_", 1)[0]

    @classmethod
    def _figure_result_family(
        cls,
        *,
        step: AnalysisStep,
        figure_product: str,
    ) -> Optional[str]:
        parsed = typed_product(figure_product)
        if parsed is None or parsed[0] != "figure":
            return None
        obligations = set(figure_product_source_obligations(figure_product))
        if effect_bearing_product(figure_product) or any(
            item.startswith("effect:") for item in obligations
        ):
            return "effect"
        if any(item.startswith("prediction:") for item in obligations):
            return "prediction"
        if cls._normalised_method_head(step.method) in cls._PREDICTION_METHODS:
            return "prediction"
        return None

    @classmethod
    def _figure_source_obligations(
        cls,
        *,
        step: AnalysisStep,
        figure_product: str,
    ) -> Set[str]:
        obligations = set(figure_product_source_obligations(figure_product))
        if obligations:
            return obligations
        family = cls._figure_result_family(
            step=step,
            figure_product=figure_product,
        )
        if family == "effect":
            return {"effect"}
        if family == "prediction":
            return {"prediction:performance"}
        return set()

    @classmethod
    def _planned_result_families(cls, step: AnalysisStep) -> Set[str]:
        families: Set[str] = set()
        for raw in step.expected_outputs or []:
            family = cls._figure_result_family(step=step, figure_product=str(raw))
            if family is not None:
                families.add(family)
        return families

    @staticmethod
    def _role_present(value: Any, role: str) -> bool:
        normalised = re.sub(
            r"[^a-z0-9]+", "_", str(value or "").strip().lower()
        ).strip("_")
        return normalised == role or f"_{role}_" in f"_{normalised}_"

    @classmethod
    def _column_role_present(cls, column: Any, role: str) -> bool:
        """Match a declared semantic column role without substring capture.

        Product identifiers and long-form metric labels may carry namespace
        riders, but tabular columns are the actual replay schema.  Treating a
        token anywhere in a column name as its value role makes metadata such
        as ``auroc_ci_method`` or ``prediction_horizon_hours`` masquerade as a
        numeric value column.
        """

        return cls._normalise(column) == cls._normalise(role)

    @classmethod
    def _time_column_role_present(cls, column: Any, role: str) -> bool:
        normalised = cls._normalise(column)
        expected = cls._normalise(role)
        if normalised == expected:
            return True
        unit_suffixes = {
            "day",
            "days",
            "hour",
            "hours",
            "minute",
            "minutes",
            "month",
            "months",
            "week",
            "weeks",
            "year",
            "years",
        }
        if not normalised.startswith(f"{expected}_"):
            return False
        return normalised.removeprefix(f"{expected}_") in unit_suffixes

    @classmethod
    def _prediction_metric_column_roles(
        cls,
        column: Any,
    ) -> List[Tuple[str, str, str]]:
        """Return structured metric roles as ``(role, group, value_kind)``.

        A closed set of value/bound suffixes keeps numeric metric payloads
        auditable while excluding prose metadata such as ``*_ci_method``.
        Interval bounds are validated but never establish performance without
        a point estimate.
        """

        normalised = cls._normalise(column)
        contexts = {
            "development",
            "external",
            "internal",
            "test",
            "train",
            "validation",
        }
        point_suffixes = {"estimate", "point_estimate", "value"}
        lower_suffixes = {
            "ci_low",
            "ci_lower",
            "confidence_interval_low",
            "confidence_interval_lower",
            "lcl",
            "lower",
        }
        upper_suffixes = {
            "ci_high",
            "ci_upper",
            "confidence_interval_high",
            "confidence_interval_upper",
            "ucl",
            "upper",
        }
        matches: List[Tuple[str, str, str]] = []
        for role in sorted(cls._PREDICTION_PERFORMANCE_METRICS, key=len, reverse=True):
            candidates = [("", normalised)]
            prefix, separator, remainder = normalised.partition("_")
            if separator and prefix in contexts:
                candidates.append((prefix, remainder))
            for context, candidate in candidates:
                group = f"{context}:{role}" if context else role
                if candidate == role:
                    matches.append((role, group, "point"))
                    break
                role_prefix = f"{role}_"
                if not candidate.startswith(role_prefix):
                    continue
                suffix = candidate.removeprefix(role_prefix)
                if suffix in point_suffixes:
                    matches.append((role, group, "point"))
                    break
                if suffix in lower_suffixes:
                    matches.append((role, group, "lower"))
                    break
                if suffix in upper_suffixes:
                    matches.append((role, group, "upper"))
                    break
        return matches

    @classmethod
    def _has_row_paired_prediction_outcome(
        cls,
        frame: pd.DataFrame,
        predictor_columns: Sequence[str],
        outcome_columns: Sequence[str],
        *,
        require_both_classes: bool,
    ) -> bool:
        for predictor in predictor_columns:
            for outcome in outcome_columns:
                paired = frame[[predictor, outcome]].dropna()
                if paired.empty:
                    continue
                if not cls._finite_numeric_values(paired[predictor]):
                    continue
                if not cls._finite_numeric_values(paired[outcome]):
                    continue
                if require_both_classes and not cls._series_is_binary_outcome(
                    paired[outcome]
                ):
                    continue
                return True
        return False

    @classmethod
    def _has_complete_numeric_rows(
        cls,
        frame: pd.DataFrame,
        column_groups: Sequence[Sequence[str]],
        *,
        minimum_rows: int = 1,
        require_distinct_first: bool = False,
    ) -> bool:
        if not column_groups or any(not group for group in column_groups):
            return False
        for columns in itertools.product(*column_groups):
            paired = frame[list(columns)].dropna()
            if len(paired) < minimum_rows:
                continue
            if not all(
                cls._finite_numeric_values(paired[column]) for column in columns
            ):
                continue
            if require_distinct_first:
                first_values = cls._finite_numeric_values(paired[columns[0]])
                if len(set(first_values)) < 2:
                    continue
            return True
        return False

    @staticmethod
    def _series_has_finite_numeric(series: pd.Series) -> bool:
        numeric = pd.to_numeric(series, errors="coerce")
        return any(
            math.isfinite(float(value))
            for value in numeric.dropna().tolist()
        )

    @staticmethod
    def _finite_numeric_values(series: pd.Series) -> List[float]:
        raw = series.dropna()
        if raw.empty:
            return []
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.isna().any():
            return []
        values = [float(value) for value in numeric.tolist()]
        if not values or not all(math.isfinite(value) for value in values):
            return []
        return values

    @classmethod
    def _series_in_unit_interval(cls, series: pd.Series) -> bool:
        values = cls._finite_numeric_values(series)
        return bool(values) and all(0.0 <= value <= 1.0 for value in values)

    @classmethod
    def _series_is_binary_outcome(cls, series: pd.Series) -> bool:
        values = cls._finite_numeric_values(series)
        if not values:
            return False
        has_zero = any(math.isclose(value, 0.0, abs_tol=1e-12) for value in values)
        has_one = any(math.isclose(value, 1.0, abs_tol=1e-12) for value in values)
        return has_zero and has_one and all(
            math.isclose(value, 0.0, abs_tol=1e-12)
            or math.isclose(value, 1.0, abs_tol=1e-12)
            for value in values
        )

    @classmethod
    def _matching_domain_columns(
        cls,
        frame: pd.DataFrame,
        roles: Set[str] | frozenset[str],
        predicate: Callable[[pd.Series], bool],
    ) -> List[str]:
        matching = [
            str(column)
            for column in frame.columns
            if any(cls._column_role_present(column, role) for role in roles)
        ]
        if not matching or not all(predicate(frame[column]) for column in matching):
            return []
        return matching

    @classmethod
    def _prediction_metric_values_valid(
        cls,
        metric: Any,
        series: pd.Series,
    ) -> bool:
        metric_name = cls._normalise(metric)
        values = cls._finite_numeric_values(series)
        if not values:
            return False
        if any(
            cls._role_present(metric_name, role)
            for role in cls._UNIT_INTERVAL_PREDICTION_METRICS
        ):
            return all(0.0 <= value <= 1.0 for value in values)
        return any(
            cls._role_present(metric_name, role)
            for role in cls._PREDICTION_PERFORMANCE_METRICS
        )

    @classmethod
    def _prediction_metric_interval_valid(
        cls,
        *,
        metric: str,
        point: pd.Series,
        lower: pd.Series,
        upper: pd.Series,
    ) -> bool:
        lower_present = lower.notna()
        upper_present = upper.notna()
        if not lower_present.equals(upper_present) or not bool(lower_present.any()):
            return False
        if not bool(point[lower_present].notna().all()):
            return False
        point_slice = point[lower_present]
        lower_slice = lower[lower_present]
        upper_slice = upper[upper_present]
        if not all(
            cls._prediction_metric_values_valid(metric, values)
            for values in (point_slice, lower_slice, upper_slice)
        ):
            return False
        point_values = cls._finite_numeric_values(point_slice)
        lower_values = cls._finite_numeric_values(lower_slice)
        upper_values = cls._finite_numeric_values(upper_slice)
        return bool(point_values) and all(
            low <= estimate <= high
            for estimate, low, high in zip(
                point_values,
                lower_values,
                upper_values,
            )
        )

    @classmethod
    def _matching_finite_columns(
        cls,
        frame: pd.DataFrame,
        roles: Set[str] | frozenset[str],
    ) -> List[str]:
        matching = [
            str(column)
            for column in frame.columns
            if any(cls._column_role_present(column, role) for role in roles)
        ]
        if not matching or not all(
            bool(cls._finite_numeric_values(frame[column])) for column in matching
        ):
            return []
        return matching

    @classmethod
    def _prediction_source_obligations(
        cls,
        *,
        product: str,
        frame: Optional[pd.DataFrame],
        statistic_value: Optional[float] = None,
    ) -> Set[str]:
        """Return replayable prediction display obligations for one source."""

        parsed_product = typed_product(product)
        product_supported = any(
            cls._role_present(product, role)
            for role in cls._PREDICTION_SOURCE_ROLES
        )
        if not product_supported:
            return set()
        if parsed_product is not None and parsed_product[0] == "statistic":
            metric_role = next(
                (
                    role
                    for role in cls._PREDICTION_PERFORMANCE_METRICS
                    if cls._role_present(product, role)
                ),
                None,
            )
            if metric_role is None:
                return set()
            if statistic_value is not None and not (
                cls._prediction_metric_values_valid(
                    metric_role,
                    pd.Series([statistic_value]),
                )
            ):
                return set()
            return {"prediction:performance"}
        if frame is None:
            return set()

        obligations: Set[str] = set()
        probability_columns = cls._matching_domain_columns(
            frame,
            cls._PREDICTED_PROBABILITY_ROLES,
            cls._series_in_unit_interval,
        )
        score_columns = cls._matching_finite_columns(
            frame,
            cls._PREDICTED_SCORE_ROLES,
        )
        observed_outcome_columns = cls._matching_domain_columns(
            frame,
            cls._OBSERVED_OUTCOME_ROLES,
            cls._series_is_binary_outcome,
        )
        probability_outcome_paired = cls._has_row_paired_prediction_outcome(
            frame,
            probability_columns,
            observed_outcome_columns,
            require_both_classes=True,
        )
        score_outcome_paired = cls._has_row_paired_prediction_outcome(
            frame,
            score_columns,
            observed_outcome_columns,
            require_both_classes=True,
        )
        if probability_outcome_paired:
            # Patient-level predictions plus observed outcomes are sufficient to
            # replay discrimination, calibration, aggregate performance, and DCA.
            obligations.update(
                {
                    "prediction:calibration",
                    "prediction:decision",
                    "prediction:performance",
                    "prediction:roc",
                }
            )
        elif score_outcome_paired:
            # An arbitrary continuous score can replay rank discrimination, but
            # it is not a calibrated probability and cannot authorize calibration,
            # Brier, or decision-curve displays.
            obligations.update({"prediction:performance", "prediction:roc"})

        observed_calibration_columns = cls._matching_domain_columns(
            frame,
            cls._OBSERVED_CALIBRATION_ROLES,
            cls._series_in_unit_interval,
        )
        if cls._has_complete_numeric_rows(
            frame,
            (probability_columns, observed_calibration_columns),
            minimum_rows=2,
            require_distinct_first=True,
        ):
            obligations.add("prediction:calibration")

        false_positive_rate_columns = cls._matching_domain_columns(
            frame,
            cls._FALSE_POSITIVE_RATE_ROLES,
            cls._series_in_unit_interval,
        )
        true_positive_rate_columns = cls._matching_domain_columns(
            frame,
            cls._TRUE_POSITIVE_RATE_ROLES,
            cls._series_in_unit_interval,
        )
        threshold_columns = cls._matching_finite_columns(
            frame,
            frozenset({"threshold"}),
        )
        if cls._has_complete_numeric_rows(
            frame,
            (
                threshold_columns,
                false_positive_rate_columns,
                true_positive_rate_columns,
            ),
            minimum_rows=2,
            require_distinct_first=True,
        ):
            obligations.add("prediction:roc")
        net_benefit_columns = cls._matching_finite_columns(
            frame,
            frozenset({"net_benefit"}),
        )
        probability_threshold_columns = cls._matching_domain_columns(
            frame,
            frozenset({"threshold"}),
            cls._series_in_unit_interval,
        )
        if cls._has_complete_numeric_rows(
            frame,
            (probability_threshold_columns, net_benefit_columns),
            minimum_rows=2,
            require_distinct_first=True,
        ):
            obligations.add("prediction:decision")

        performance_rows: Set[Any] = set()
        discrimination_rows: Set[Any] = set()
        performance_payload_valid = True
        performance_payload_has_valid_value = False
        metric_payloads: Dict[str, Dict[str, Any]] = {}
        generic_metric_intervals: Dict[str, List[str]] = {
            "lower": [],
            "upper": [],
        }
        for column in frame.columns:
            parsed_interval = cls._confidence_interval_bound(column)
            if parsed_interval is not None and not parsed_interval[0]:
                generic_metric_intervals[parsed_interval[1]].append(str(column))
        for column in frame.columns:
            matching_metrics = cls._prediction_metric_column_roles(column)
            if not matching_metrics:
                continue
            column_valid = all(
                cls._prediction_metric_values_valid(metric, frame[column])
                for metric, _group, _kind in matching_metrics
            )
            if not column_valid:
                performance_payload_valid = False
                continue
            for metric, group, kind in matching_metrics:
                payload = metric_payloads.setdefault(
                    group,
                    {"role": metric, "point": [], "lower": [], "upper": []},
                )
                payload[kind].append(str(column))
                if kind != "point":
                    continue
                performance_payload_has_valid_value = True
                point_rows = set(frame[column].dropna().index.tolist())
                performance_rows.update(point_rows)
                if metric in cls._DISCRIMINATION_PREDICTION_METRICS:
                    discrimination_rows.update(point_rows)
        if generic_metric_intervals["lower"] or generic_metric_intervals["upper"]:
            point_payloads = [
                payload for payload in metric_payloads.values() if payload["point"]
            ]
            if len(point_payloads) == 1 and not (
                point_payloads[0]["lower"] or point_payloads[0]["upper"]
            ):
                point_payloads[0]["lower"].extend(
                    generic_metric_intervals["lower"]
                )
                point_payloads[0]["upper"].extend(
                    generic_metric_intervals["upper"]
                )
            elif point_payloads:
                performance_payload_valid = False
        for payload in metric_payloads.values():
            has_interval = bool(payload["lower"] or payload["upper"])
            if not has_interval:
                continue
            if not (
                len(payload["point"]) == 1
                and len(payload["lower"]) == 1
                and len(payload["upper"]) == 1
            ):
                performance_payload_valid = False
                continue
            if not cls._prediction_metric_interval_valid(
                metric=str(payload["role"]),
                point=frame[payload["point"][0]],
                lower=frame[payload["lower"][0]],
                upper=frame[payload["upper"][0]],
            ):
                performance_payload_valid = False
        label_columns = [
            column
            for column in frame.columns
            if cls._normalise(column) in {"metric", "name", "statistic"}
        ]
        value_columns = [
            column
            for column in frame.columns
            if cls._normalise(column) in {"estimate", "result", "value"}
        ]
        long_interval_columns = generic_metric_intervals
        for label_column in label_columns:
            for row_index, metric_label in frame[label_column].items():
                metric_role = next(
                    (
                        role
                        for role in cls._PREDICTION_PERFORMANCE_METRICS
                        if cls._normalise(metric_label) == role
                    ),
                    None,
                )
                if metric_role is None:
                    continue
                present_values = [
                    value_column
                    for value_column in value_columns
                    if pd.notna(frame.at[row_index, value_column])
                ]
                row_valid = bool(present_values) and all(
                    cls._prediction_metric_values_valid(
                        metric_role,
                        frame.loc[[row_index], value_column],
                    )
                    for value_column in present_values
                )
                present_lower = [
                    column
                    for column in long_interval_columns["lower"]
                    if pd.notna(frame.at[row_index, column])
                ]
                present_upper = [
                    column
                    for column in long_interval_columns["upper"]
                    if pd.notna(frame.at[row_index, column])
                ]
                if present_lower or present_upper:
                    row_valid = row_valid and (
                        len(present_values) == 1
                        and len(present_lower) == 1
                        and len(present_upper) == 1
                        and cls._prediction_metric_interval_valid(
                            metric=metric_role,
                            point=frame.loc[[row_index], present_values[0]],
                            lower=frame.loc[[row_index], present_lower[0]],
                            upper=frame.loc[[row_index], present_upper[0]],
                        )
                    )
                if row_valid:
                    performance_payload_has_valid_value = True
                    performance_rows.add(row_index)
                    if metric_role in cls._DISCRIMINATION_PREDICTION_METRICS:
                        discrimination_rows.add(row_index)
                else:
                    performance_payload_valid = False

        if performance_payload_has_valid_value and performance_payload_valid:
            obligations.add("prediction:performance")
        elif not performance_payload_valid:
            # A valid sibling must not launder an out-of-domain value carrying
            # the same semantic metric role in the same source-data payload.
            obligations.discard("prediction:performance")

        candidate_time_columns = [
            column
            for column in frame.columns
            if any(
                cls._time_column_role_present(column, role)
                for role in cls._PREDICTION_TIME_ROLES
            )
        ]
        valid_time_varying_discrimination = False
        for time_column in candidate_time_columns:
            time_values = cls._finite_numeric_values(frame[time_column])
            if len(set(time_values)) < 2:
                continue

            paired_metric_values = cls._finite_numeric_values(
                frame.loc[list(discrimination_rows), time_column]
                if discrimination_rows
                else pd.Series(dtype=float)
            )
            if len(set(paired_metric_values)) >= 2:
                valid_time_varying_discrimination = True
                break

            replayable_raw_horizons: Set[float] = set()
            for horizon_value, group in frame.dropna(
                subset=[time_column]
            ).groupby(time_column):
                probability_replay = cls._has_row_paired_prediction_outcome(
                    group,
                    probability_columns,
                    observed_outcome_columns,
                    require_both_classes=True,
                )
                score_replay = cls._has_row_paired_prediction_outcome(
                    group,
                    score_columns,
                    observed_outcome_columns,
                    require_both_classes=True,
                )
                if probability_replay or score_replay:
                    numeric_horizon = cls._as_float(horizon_value)
                    if numeric_horizon is not None:
                        replayable_raw_horizons.add(numeric_horizon)
            if len(replayable_raw_horizons) >= 2:
                valid_time_varying_discrimination = True
                break

        if (
            "prediction:performance" in obligations
            and valid_time_varying_discrimination
        ):
            obligations.add("prediction:time_varying_discrimination")
        return obligations

    @staticmethod
    def _effect_semantics_support_figure(
        *,
        semantic_signals: Sequence[str],
        figure_product: str,
    ) -> bool:
        """Require one source to preserve the figure's scientific semantics."""

        output_measure = effect_measure_family(figure_product)
        output_role = effect_role_family(figure_product)
        registered_roles = {
            obligation.split(":", 1)[1]
            for obligation in figure_product_source_obligations(figure_product)
            if obligation.startswith("effect:")
        }
        output_tier = effect_estimand_tier(figure_product)
        output_adjustment = effect_adjustment_family(figure_product)
        input_measures = {
            family
            for signal in semantic_signals
            if (family := effect_measure_family(signal)) is not None
        }
        input_roles = {
            family
            for signal in semantic_signals
            if (family := effect_role_family(signal)) is not None
        }
        input_tiers = {
            family
            for signal in semantic_signals
            if (family := effect_estimand_tier(signal)) is not None
        }
        input_adjustments = {
            family
            for signal in semantic_signals
            if (family := effect_adjustment_family(signal)) is not None
        }
        if output_measure is not None and output_measure not in input_measures:
            return False
        required_roles = ({output_role} if output_role is not None else set()) | (
            registered_roles
        )
        if registered_roles and not input_measures:
            # A specialised effect display (for example subgroup or interaction)
            # must preserve both its role and an explicit effect scale. A generic
            # ``estimate`` column is not enough to establish forest-plot meaning.
            return False
        if required_roles:
            if not required_roles.issubset(input_roles):
                return False
        elif input_roles:
            return False
        if output_tier is not None:
            if output_tier not in input_tiers:
                return False
        elif input_tiers & {"secondary", "sensitivity", "corroborative"}:
            return False
        if (
            output_adjustment is not None
            and output_adjustment not in input_adjustments
        ):
            return False
        return True

    @classmethod
    def _contract_scoped_effect_product(
        cls,
        *,
        product: str,
        source_frame: pd.DataFrame,
        upstream_step_id: str,
        completed_step_records: Optional[Sequence[Dict[str, Any]]],
    ) -> str:
        """Add an estimand tier only when rows match validated model contracts.

        A generic coefficient table name cannot by itself prove that selected
        rows are primary, secondary, or sensitivity estimates.  Once the
        figure source has value-matched that table, its exact ``model_id`` and
        exposure rows may inherit the tier from the successful parent step's
        machine-readable model contracts.  Free text and variable names are
        never routing authority.
        """

        parsed = typed_product(product)
        if (
            parsed is None
            or not effect_bearing_product(product)
            or "model_id" not in source_frame.columns
            or not completed_step_records
        ):
            return product
        model_ids = {
            str(value).strip()
            for value in source_frame["model_id"].dropna().tolist()
            if str(value).strip()
        }
        if not model_ids:
            return product
        parent_records = [
            record
            for record in current_successful_step_records(completed_step_records)
            if str(record.get("step_id") or "").strip() == upstream_step_id
        ]
        if len(parent_records) != 1:
            return product
        summary = parent_records[0].get("step_summary")
        contracts = (
            summary.get("model_contracts")
            if isinstance(summary, Mapping)
            else None
        )
        contract_by_model: Dict[str, Mapping[str, Any]] = {}
        for contract in contracts or []:
            if not isinstance(contract, Mapping):
                continue
            model_id = str(contract.get("model_id") or "").strip()
            if not model_id or model_id in contract_by_model:
                return product
            contract_by_model[model_id] = contract
        if not model_ids <= set(contract_by_model):
            return product
        selected_contracts = [contract_by_model[model_id] for model_id in model_ids]
        tiers = {
            cls._normalise(contract.get("analysis_role"))
            for contract in selected_contracts
        }
        allowed_tiers = {"primary", "secondary", "sensitivity", "corroborative"}
        if len(tiers) != 1 or not tiers <= allowed_tiers or any(
            cls._normalise(contract.get("fit_status")) != "fitted"
            for contract in selected_contracts
        ):
            return product
        if not {"term_role", "source_variable"} <= set(source_frame.columns):
            return product
        exposure_rows = source_frame.loc[
            source_frame["term_role"].map(cls._normalise).eq("exposure")
        ]
        if exposure_rows.empty or len(exposure_rows) != len(source_frame):
            return product
        for _, row in exposure_rows.iterrows():
            contract = contract_by_model.get(
                str(row.get("model_id") or "").strip()
            )
            if contract is None or str(
                row.get("source_variable") or ""
            ).strip() != str(contract.get("exposure_source") or "").strip():
                return product
        tier = next(iter(tiers))
        kind, name = parsed
        if effect_estimand_tier(product) is not None:
            return product
        return f"{kind}:{tier}_{name}"

    @classmethod
    def _confidence_interval_bound(
        cls,
        column: Any,
    ) -> Optional[Tuple[str, str]]:
        normalised = cls._normalise(column)
        patterns = (
            r"^(?P<prefix>.*?)(?:_)?(?:ci|confidence_interval)_"
            r"(?P<bound>low|lower|lcl|high|upper|ucl)$",
            r"^(?P<prefix>.*?)(?:_)?(?P<bound>low|lower|lcl|high|upper|ucl)_"
            r"(?:ci|confidence_interval)$",
            r"^(?P<prefix>.*?)(?:_)?(?P<bound>lcl|ucl)$",
            r"^(?P<prefix>.*?)(?:_)?(?P<bound>lower|upper)$",
        )
        for pattern in patterns:
            matched = re.fullmatch(pattern, normalised)
            if matched is None:
                continue
            bound = matched.group("bound")
            side = "lower" if bound in {"low", "lower", "lcl"} else "upper"
            return matched.group("prefix").strip("_"), side
        return None

    @classmethod
    def _ratio_intervals_valid(
        cls,
        frame: pd.DataFrame,
        ratio_point_columns: Sequence[str],
    ) -> bool:
        interval_columns: Dict[str, Dict[str, List[str]]] = {}
        for column in frame.columns:
            parsed = cls._confidence_interval_bound(column)
            if parsed is None:
                continue
            prefix, side = parsed
            interval_columns.setdefault(
                prefix,
                {"lower": [], "upper": []},
            )[side].append(str(column))
        normalised_points = {
            str(column): cls._normalise(column) for column in ratio_point_columns
        }

        def matched_ratio_points(prefix: str) -> List[str]:
            prefix_family = effect_measure_family(f"table:{prefix}")
            return [
                column
                for column, normalised in normalised_points.items()
                if normalised == prefix
                or (
                    prefix_family is not None
                    and effect_measure_family(f"table:{normalised}")
                    == prefix_family
                )
            ]

        explicitly_covered_points = {
            column
            for prefix in interval_columns
            if prefix
            for column in matched_ratio_points(prefix)
        }
        for prefix, sides in interval_columns.items():
            if prefix:
                matched_points = matched_ratio_points(prefix)
            else:
                matched_points = [
                    column
                    for column in normalised_points
                    if column not in explicitly_covered_points
                ]
            if not matched_points:
                # A signed interval for another estimand in the same table is
                # not a ratio-scale interval and must not poison the ratio.
                continue
            if len(matched_points) != 1:
                return False
            if len(sides["lower"]) != 1 or len(sides["upper"]) != 1:
                return False
            point_column = matched_points[0]
            lower_column = sides["lower"][0]
            upper_column = sides["upper"][0]
            point_raw = frame[point_column]
            lower_raw = frame[lower_column]
            upper_raw = frame[upper_column]
            lower_present = lower_raw.notna()
            upper_present = upper_raw.notna()
            if not lower_present.equals(upper_present) or not bool(lower_present.any()):
                return False
            if not bool(point_raw[lower_present].notna().all()):
                return False
            point = pd.to_numeric(point_raw[lower_present], errors="coerce")
            lower = pd.to_numeric(lower_raw[lower_present], errors="coerce")
            upper = pd.to_numeric(upper_raw[upper_present], errors="coerce")
            if point.isna().any() or lower.isna().any() or upper.isna().any():
                return False
            point_values = [float(value) for value in point.tolist()]
            lower_values = [float(value) for value in lower.tolist()]
            upper_values = [float(value) for value in upper.tolist()]
            if not all(
                math.isfinite(estimate)
                and math.isfinite(low)
                and math.isfinite(high)
                and 0.0 < low <= estimate <= high
                for estimate, low, high in zip(
                    point_values,
                    lower_values,
                    upper_values,
                )
            ):
                return False
        return True

    @classmethod
    def _source_supports_result_family(
        cls,
        *,
        product: str,
        frame: Optional[pd.DataFrame] = None,
        family: Optional[str],
        figure_products: Sequence[str] = (),
    ) -> bool:
        """Return whether a typed value source can authenticate the figure family.

        The source product and its immutable table schema are host-owned.  Figure
        contract prose and panel roles are intentionally not consulted.
        """

        if family is None:
            return True
        parsed_product = typed_product(product)
        columns = list(frame.columns) if frame is not None else []
        if family == "prediction":
            source_obligations = cls._prediction_source_obligations(
                product=product,
                frame=frame,
            )
            if not source_obligations:
                return False
            if not figure_products:
                return True
            return all(
                {
                    obligation
                    for obligation in (
                        figure_product_source_obligations(figure)
                        or ("prediction:performance",)
                    )
                    if obligation.startswith("prediction:")
                }.issubset(source_obligations)
                for figure in figure_products
            )
        if family != "effect":
            return True

        if not effect_bearing_product(product):
            return False
        if parsed_product is not None and parsed_product[0] == "statistic":
            semantic_signals = [product]
            return all(
                cls._effect_semantics_support_figure(
                    semantic_signals=semantic_signals,
                    figure_product=figure,
                )
                for figure in figure_products
                if effect_bearing_product(figure)
                or any(
                    obligation.startswith("effect:")
                    for obligation in figure_product_source_obligations(figure)
                )
            )
        if frame is None:
            return False
        typed_columns = [f"table:{column}" for column in columns]
        generic_value_columns = {
            "coef",
            "coefficient",
            "effect",
            "effect_estimate",
            "estimate",
            "point_estimate",
            "value",
        }
        finite_effect_columns = [
            signal
            for signal, column in zip(typed_columns, columns)
            if (
                effect_bearing_product(signal)
                or effect_measure_family(signal) is not None
                or cls._normalise(column) in generic_value_columns
            )
            and cls._series_has_finite_numeric(frame[column])
        ]
        if not finite_effect_columns:
            return False
        source_measure = effect_measure_family(product)
        ratio_families = {"hazard_ratio", "odds_ratio", "risk_ratio"}
        ratio_point_columns: List[str] = []
        for signal, column in zip(typed_columns, columns):
            column_measure = effect_measure_family(signal)
            if column_measure not in ratio_families and not (
                source_measure in ratio_families
                and cls._normalise(column) in generic_value_columns
            ):
                continue
            ratio_point_columns.append(str(column))
            values = cls._finite_numeric_values(frame[column])
            if not values or any(value <= 0.0 for value in values):
                return False
        if ratio_point_columns and not cls._ratio_intervals_valid(
            frame,
            ratio_point_columns,
        ):
            return False
        semantic_signals = [
            product,
            *(
                signal
                for signal in finite_effect_columns
                if effect_bearing_product(signal)
                or effect_measure_family(signal) is not None
            ),
        ]
        effect_figures = [
            figure
            for figure in figure_products
            if effect_bearing_product(figure)
            or any(
                obligation.startswith("effect:")
                for obligation in figure_product_source_obligations(figure)
            )
        ]
        if not effect_figures:
            return True
        return all(
            cls._effect_semantics_support_figure(
                semantic_signals=semantic_signals,
                figure_product=figure,
            )
            for figure in effect_figures
        )

    @classmethod
    def _source_supports_figures(
        cls,
        *,
        step: AnalysisStep,
        product: str,
        frame: Optional[pd.DataFrame],
        figure_products: Sequence[str],
        require_all: bool,
    ) -> bool:
        checks = [
            cls._source_supports_result_family(
                product=product,
                frame=frame,
                family=cls._figure_result_family(
                    step=step,
                    figure_product=figure,
                ),
                figure_products=[figure],
            )
            for figure in figure_products
        ]
        if not checks:
            return True
        return all(checks) if require_all else any(checks)

    @classmethod
    def _extract_statistic_value(
        cls,
        step_summary: Any,
        product_name: str,
    ) -> Optional[float]:
        """Extract one unambiguous finite scalar for an exact statistic product."""

        target = cls._normalise(product_name)
        candidates: List[float] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                declared_name = value.get("name") or value.get("statistic")
                if declared_name is not None and cls._normalise(declared_name) == target:
                    for field in ("value", "estimate", "result"):
                        numeric = cls._as_float(value.get(field))
                        if numeric is not None:
                            candidates.append(numeric)
                for key, child in value.items():
                    if cls._normalise(key) == target:
                        numeric = cls._as_float(child)
                        if numeric is not None:
                            candidates.append(numeric)
                    if isinstance(child, (Mapping, list, tuple)):
                        visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(step_summary)
        if not candidates:
            return None
        first = candidates[0]
        if any(
            not math.isclose(item, first, rel_tol=1e-9, abs_tol=1e-9)
            for item in candidates[1:]
        ):
            return None
        return first

    @classmethod
    def _source_contains_statistic(
        cls,
        source_df: pd.DataFrame,
        *,
        product_name: str,
        expected: float,
    ) -> bool:
        target = cls._normalise(product_name)

        def values_match(series: pd.Series) -> bool:
            raw = series.dropna()
            if raw.empty:
                return False
            values = pd.to_numeric(raw, errors="coerce")
            if values.isna().any():
                return False
            return all(
                math.isfinite(float(value))
                and math.isclose(
                    float(value), expected, rel_tol=1e-9, abs_tol=1e-9
                )
                for value in values
            )

        target_family = cls._statistic_family(target)
        for column in source_df.columns:
            column_name = cls._normalise(column)
            if column_name != target and (
                target_family is None
                or cls._statistic_family(column_name) != target_family
            ):
                continue
            if values_match(source_df[column]):
                return True

        label_columns = [
            column
            for column in source_df.columns
            if cls._normalise(column) in {"metric", "name", "product", "statistic"}
        ]
        value_columns = [
            column
            for column in source_df.columns
            if cls._normalise(column) in {"estimate", "result", "value"}
        ]
        for label_column in label_columns:
            normalised_labels = source_df[label_column].map(cls._normalise)
            matching_rows = normalised_labels.eq(target)
            if target_family is not None:
                matching_rows |= normalised_labels.map(
                    cls._statistic_family
                ).eq(target_family)
            if not matching_rows.any():
                continue
            for value_column in value_columns:
                if values_match(source_df.loc[matching_rows, value_column]):
                    return True
        return False

    @classmethod
    def _statistic_family(cls, value: Any) -> Optional[str]:
        normalised = cls._normalise(value)
        effect_family = effect_measure_family(f"statistic:{normalised}")
        if effect_family is not None:
            return f"effect:{effect_family}"
        for family, aliases in {
            "auroc": {"auc", "auroc", "c_statistic", "roc_auc"},
            "brier": {"brier", "brier_score"},
            "calibration_intercept": {"calibration_intercept"},
            "calibration_slope": {"calibration_slope"},
        }.items():
            if normalised in aliases:
                return f"prediction:{family}"
        return None

    @classmethod
    def _statistic_payload_issue(
        cls,
        source_df: pd.DataFrame,
        *,
        required_statistics: Mapping[str, tuple[str, float]],
    ) -> Optional[Dict[str, Any]]:
        """Return the first unbound numeric cell in a statistic-only source.

        Finding one truthful scalar must not authenticate unrelated plotted
        numbers in the same source-data file.  Table-backed sources are checked
        by the table comparator instead; this helper governs the scalar-only
        fallback.
        """

        required = [
            (
                cls._normalise(product_name),
                cls._statistic_family(product_name),
                expected,
            )
            for product_name, expected in required_statistics.values()
        ]

        def matching_expected(label: Any) -> List[float]:
            normalised = cls._normalise(label)
            family = cls._statistic_family(normalised)
            return [
                expected
                for target, target_family, expected in required
                if normalised == target
                or (
                    family is not None
                    and target_family is not None
                    and family == target_family
                )
            ]

        def agrees(value: Any, expected_values: Sequence[float]) -> bool:
            numeric = cls._as_float(value)
            return numeric is not None and any(
                math.isclose(
                    numeric, expected, rel_tol=1e-9, abs_tol=1e-9
                )
                for expected in expected_values
            )

        label_columns = [
            column
            for column in source_df.columns
            if cls._normalise(column) in {"metric", "name", "product", "statistic"}
        ]
        value_columns = [
            column
            for column in source_df.columns
            if cls._normalise(column) in {"estimate", "result", "value"}
        ]
        verified_cells: Set[tuple[Any, Any]] = set()
        if label_columns and value_columns:
            for row_index, row in source_df.iterrows():
                expected_values = [
                    expected
                    for label_column in label_columns
                    for expected in matching_expected(row[label_column])
                ]
                for value_column in value_columns:
                    if pd.isna(row[value_column]):
                        continue
                    if not agrees(row[value_column], expected_values):
                        return {
                            "reason": "unbound_statistic_value",
                            "column": str(value_column),
                            "row": str(row_index),
                            "value": row[value_column],
                        }
                    verified_cells.add((row_index, value_column))

        exempt_columns = {
            *cls._KEY_COLUMNS,
            *cls._POSITIONAL_ROW_INDEX_COLUMNS,
            "source_step_id",
            "source_table",
        }
        for column in source_df.columns:
            normalised_column = cls._normalise(column)
            if column in label_columns or normalised_column in exempt_columns:
                continue
            for row_index, value in source_df[column].items():
                if (row_index, column) in verified_cells or pd.isna(value):
                    continue
                numeric = cls._as_float(value)
                if numeric is None:
                    continue
                expected_values = matching_expected(column)
                if not agrees(numeric, expected_values):
                    return {
                        "reason": "unbound_statistic_value",
                        "column": str(column),
                        "row": str(row_index),
                        "value": numeric,
                    }
        return None

    @staticmethod
    def _iter_string_values(value: Any) -> List[str]:
        values: List[str] = []
        if isinstance(value, str):
            if value.strip():
                values.append(value.strip())
        elif isinstance(value, Mapping):
            for child in value.values():
                values.extend(FigureSourceDataValidator._iter_string_values(child))
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                values.extend(FigureSourceDataValidator._iter_string_values(child))
        return values

    @classmethod
    def _registered_figure_paths(
        cls,
        *,
        step: AnalysisStep,
        step_summary: Mapping[str, Any],
        out_dir: Path,
    ) -> Dict[tuple[str, str], List[Path]]:
        """Resolve exact planned figure roles to their registered files.

        A directory-wide contract/source-data scan is insufficient: an honest
        decoy bundle must never authenticate a different file registered under
        the Planner's figure role.  Exact typed registry keys are authoritative;
        a same-name file fallback is retained for legacy summaries.
        """

        declared = {
            parsed
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
        }
        resolved: Dict[tuple[str, str], List[Path]] = {
            product: [] for product in declared
        }

        def candidate_paths(value: Any) -> List[Path]:
            paths: List[Path] = []
            for raw_path in cls._iter_string_values(value):
                suffix = Path(raw_path).suffix.lower()
                if suffix not in {".png", ".svg", ".pdf", ".tif", ".tiff"}:
                    continue
                relative = Path(raw_path)
                candidate = relative if relative.is_absolute() else out_dir / relative
                paths.append(candidate)
            return paths

        for container_key in ("output_files", "outputs"):
            container = step_summary.get(container_key)
            if not isinstance(container, Mapping):
                continue
            for raw_role, value in container.items():
                role = typed_product(raw_role)
                if role in declared:
                    resolved[role].extend(candidate_paths(value))

        legacy_paths: List[Path] = []
        for key in ("figure_files", "figure_file", "figure_path"):
            legacy_paths.extend(candidate_paths(step_summary.get(key)))
        for product in declared:
            if resolved[product]:
                continue
            resolved[product].extend(
                path for path in legacy_paths if path.stem == product[1]
            )

        return {
            product: list(dict.fromkeys(paths))
            for product, paths in resolved.items()
        }

    @classmethod
    def _registered_same_step_tables(
        cls,
        *,
        step: AnalysisStep,
        step_summary: Mapping[str, Any],
        out_dir: Path,
        run_dir: Path,
        excluded_paths: Sequence[Path] = (),
    ) -> Dict[Path, str]:
        """Return distinct planned tabular outputs available to a mixed step.

        A figure's own contract-declared source CSV is never eligible as the
        upstream value source.  Otherwise the writable output could register the
        same file as both ``table:*`` and ``*source_data.csv`` and authenticate
        arbitrary values by comparing the file with itself.
        """

        declared = {
            parsed: f"{parsed[0]}:{parsed[1]}"
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None
            and parsed[0] in {"artifact", "dataset", "table"}
        }
        excluded = {
            path.resolve()
            for path in excluded_paths
            if path.exists()
        }
        result_families = cls._planned_result_families(step)
        tables: Dict[Path, str] = {}
        for container_key in ("output_files", "outputs"):
            container = step_summary.get(container_key)
            if not isinstance(container, Mapping):
                continue
            for raw_role, value in container.items():
                role = typed_product(raw_role)
                if role not in declared:
                    continue
                for raw_path in cls._iter_string_values(value):
                    if Path(raw_path).suffix.lower() not in cls._TABULAR_SUFFIXES:
                        continue
                    relative = Path(raw_path)
                    candidate = relative if relative.is_absolute() else out_dir / relative
                    if (
                        cls._safe_regular_run_file(candidate, run_dir=run_dir)
                        and candidate.parent.resolve() == out_dir.resolve()
                        and candidate.resolve() not in excluded
                    ):
                        try:
                            frame = cls._read_tabular(candidate)
                        except Exception:
                            continue
                        product = declared[role]
                        if not any(
                            cls._source_supports_result_family(
                                product=product,
                                frame=frame,
                                family=family,
                            )
                            for family in result_families
                        ) and result_families:
                            continue
                        tables[candidate.resolve()] = product
        return tables

    @classmethod
    def _declared_bundle_source_tables(
        cls,
        *,
        step: AnalysisStep,
        step_summary: Mapping[str, Any],
        out_dir: Path,
        run_dir: Path,
        resolved_input_bindings: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> tuple[Optional[Dict[Path, Set[str]]], List[ValidationFinding]]:
        """Bind each planned numeric figure to its exact local source bundle.

        ``None`` means the step has no typed planned figure and the legacy
        source-data scan may be used.  A returned mapping is authoritative: it
        binds each local source table to the exact planned figure product(s)
        whose contract declared it, so one honest family cannot launder another.
        """

        planned = {
            parsed: str(raw)
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
        }
        if not planned:
            return None, []

        declared_input_kinds = {
            parsed[0]
            for raw in (step.inputs or [])
            if (parsed := typed_product(raw)) is not None
        }
        declared_input_kinds.update(
            str(binding.get("declared_kind") or "").strip().lower()
            for binding in (resolved_input_bindings or {}).values()
            if isinstance(binding, Mapping)
        )
        declared_result_kinds = {
            parsed[0]
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None
            and parsed[0] != "figure"
        }
        has_data_input = bool(
            (declared_input_kinds | declared_result_kinds)
            & {"artifact", "dataset", "model", "statistic", "table"}
        )
        has_untyped_input = any(
            typed_product(raw) is None for raw in (step.inputs or [])
        )
        planned_result_families = cls._planned_result_families(step)
        method_head = cls._normalised_method_head(step.method)
        compute_and_render = bool(planned) and method_head not in cls._PURE_RENDER_METHODS
        host_requires_source = bool(
            has_data_input
            or has_untyped_input
            or planned_result_families
            or compute_and_render
        )
        registered = cls._registered_figure_paths(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
        )
        source_tables: Dict[Path, Set[str]] = {}
        findings: List[ValidationFinding] = []

        for product, raw_product in planned.items():
            paths = registered.get(product, [])
            if not paths:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' did not bind planned "
                            f"figure {raw_product!r} to an exact output file."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "reason": "missing_declared_figure_registration",
                        },
                    )
                )
                continue

            stems = {path.stem for path in paths}
            if len(stems) != 1:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Planned figure {raw_product!r} is registered to "
                            "multiple unrelated file stems; one figure bundle "
                            "must share a single stem across export formats."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "figure_stems": sorted(stems),
                            "reason": "ambiguous_declared_figure_bundle",
                        },
                    )
                )
                continue
            stem = next(iter(stems))
            contract_path = out_dir / f"{stem}.figure_contract.json"
            contract: Any = None
            contract_is_safe = (
                cls._safe_regular_run_file(contract_path, run_dir=run_dir)
                and contract_path.parent.resolve() == out_dir.resolve()
            )
            if contract_is_safe:
                try:
                    contract = json.loads(contract_path.read_text(encoding="utf-8"))
                except Exception:
                    contract = None
            panels = contract.get("panels") if isinstance(contract, Mapping) else []
            result_like = bool(
                isinstance(contract, dict)
                and FigureContractQualityValidator._is_result_like_contract(
                    contract,
                    panels if isinstance(panels, list) else [],
                )
            )
            unsafe_exports = [
                path.name
                for path in paths
                if not cls._safe_regular_run_file(path, run_dir=run_dir)
                or path.parent.resolve() != out_dir.resolve()
            ]
            if unsafe_exports:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Planned figure bundle '{stem}' contains an unsafe "
                            "or missing registered export."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "unsafe_figure_exports": sorted(unsafe_exports),
                            "reason": "unsafe_declared_figure_path",
                        },
                    )
                )
                continue
            if not isinstance(contract, Mapping) or not contract_is_safe:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Planned figure bundle '{stem}' has no readable, "
                            "same-stem .figure_contract.json file."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "figure_stem": stem,
                            "reason": "missing_figure_contract",
                        },
                    )
                )
                continue
            raw_figure_id = str(contract.get("figure_id") or "").strip()
            safe_figure_id = re.fullmatch(
                r"(?:figure:)?([A-Za-z0-9][A-Za-z0-9_.-]*)",
                raw_figure_id,
                flags=re.IGNORECASE,
            )
            figure_id = (
                cls._normalise(safe_figure_id.group(1))
                if safe_figure_id is not None
                else ""
            )
            if not figure_id or figure_id != cls._normalise(stem):
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Figure contract '{contract_path.name}' identifies "
                            "a different figure than its registered export."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "figure_stem": stem,
                            "contract_figure_id": contract.get("figure_id"),
                            "reason": "figure_contract_export_mismatch",
                        },
                    )
                )
                continue

            requires_source = host_requires_source or result_like
            if not requires_source:
                continue

            declared_sources = contract.get("source_data")
            raw_source_names = (
                [declared_sources]
                if isinstance(declared_sources, str)
                else list(declared_sources)
                if isinstance(declared_sources, (list, tuple, set))
                else ([] if declared_sources is None else [declared_sources])
            )
            invalid_source_descriptors = [
                {
                    "index": index,
                    "value_type": type(value).__name__,
                }
                for index, value in enumerate(raw_source_names)
                if not isinstance(value, str)
            ]
            if invalid_source_descriptors:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Figure contract '{contract_path.name}' must declare "
                            "source_data as local CSV basename strings."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "invalid_source_data_descriptors": (
                                invalid_source_descriptors
                            ),
                            "reason": "invalid_contract_source_data",
                        },
                    )
                )
                continue
            source_names = [str(value) for value in raw_source_names]
            local_sources: List[Path] = []
            unsafe_sources: List[str] = []
            for value in source_names:
                name = str(value or "").strip()
                if not name or Path(name).suffix.lower() != ".csv":
                    continue
                if Path(name).name != name or "/" in name or "\\" in name:
                    unsafe_sources.append(name)
                    continue
                source_path = out_dir / name
                if (
                    not cls._safe_regular_run_file(source_path, run_dir=run_dir)
                    or source_path.parent.resolve() != out_dir.resolve()
                ):
                    unsafe_sources.append(name)
                    continue
                local_sources.append(source_path)
            if unsafe_sources:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Figure contract '{contract_path.name}' declares "
                            "unsafe or missing local source-data files."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "unsafe_source_data": sorted(set(unsafe_sources)),
                            "reason": "invalid_contract_source_data",
                        },
                    )
                )
                continue
            if not local_sources:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Result figure bundle '{stem}' has no local CSV "
                            "declared in contract.source_data."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "figure_stem": stem,
                            "reason": "missing_source_data",
                        },
                    )
                )
                continue
            canonical_figure = f"{product[0]}:{product[1]}"
            for source_path in local_sources:
                source_tables.setdefault(source_path.resolve(), set()).add(
                    canonical_figure
                )

        return source_tables, findings

    def audit(
        self,
        *,
        step: AnalysisStep,
        out_dir: Path,
        run_dir: Path,
        step_summary: Dict[str, Any],
        completed_step_records: Optional[Sequence[Dict[str, Any]]] = None,
        resolved_input_bindings: Optional[
            Mapping[str, Mapping[str, Any]]
        ] = None,
    ) -> List[ValidationFinding]:
        if not self._is_rendering_step(step=step, step_summary=step_summary):
            return []
        figure_products = [
            f"{parsed[0]}:{parsed[1]}"
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
        ]
        declared_sources, bundle_findings = self._declared_bundle_source_tables(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
            run_dir=run_dir,
            resolved_input_bindings=resolved_input_bindings,
        )
        if bundle_findings:
            return bundle_findings
        if declared_sources is None:
            source_tables = sorted(out_dir.glob(self._SOURCE_DATA_GLOB))
            source_figure_products = {
                path.resolve(): set(figure_products) for path in source_tables
            }
        else:
            source_tables = sorted(declared_sources)
            source_figure_products = {
                path.resolve(): set(products)
                for path, products in declared_sources.items()
            }
        if not source_tables:
            return []

        result_families = self._planned_result_families(step)
        same_step_tables = self._registered_same_step_tables(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
            run_dir=run_dir,
            excluded_paths=source_tables,
        )
        same_step_statistics: Dict[str, tuple[str, float]] = {}
        for raw_output in step.expected_outputs or []:
            product = typed_product(raw_output)
            if product is None or product[0] != "statistic":
                continue
            canonical = f"{product[0]}:{product[1]}"
            if result_families and not any(
                self._source_supports_result_family(
                    product=canonical,
                    family=family,
                )
                for family in result_families
            ):
                continue
            value = self._extract_statistic_value(step_summary, product[1])
            if value is not None:
                same_step_statistics[f"same_step:{canonical}"] = (
                    product[1],
                    value,
                )

        bound_input_bindings: Dict[str, Mapping[str, Any]] = {}
        if resolved_input_bindings is None:
            upstream_step_ids = self._upstream_step_ids(
                step=step,
                step_summary=step_summary,
            )
            if same_step_tables or same_step_statistics:
                upstream_step_ids.add(str(step.step_id))
        else:
            upstream_step_ids: Set[str] = set()
            invalid_bindings: List[str] = []
            for raw_input, binding in resolved_input_bindings.items():
                if not isinstance(binding, Mapping):
                    invalid_bindings.append(str(raw_input))
                    continue
                declared_kind = str(
                    binding.get("declared_kind") or ""
                ).strip().lower()
                producer_id = str(binding.get("produced_by_step") or "").strip()
                evidence_id = str(binding.get("evidence_id") or "").strip()
                digest = str(binding.get("sha256") or "").strip()
                product = str(binding.get("product") or "").strip()
                parsed_input = typed_product(raw_input)
                if (
                    declared_kind
                    not in {"artifact", "dataset", "model", "statistic", "table"}
                    or parsed_input != (declared_kind, self._normalise(product))
                    or
                    not self._safe_step_id(producer_id)
                    or not evidence_id
                    or not product
                    or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
                ):
                    invalid_bindings.append(str(raw_input))
                    continue
                bound_input_bindings[str(raw_input)] = binding
                upstream_step_ids.add(producer_id)
            if same_step_tables or same_step_statistics:
                upstream_step_ids.add(str(step.step_id))
            if invalid_bindings:
                return [
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' has invalid "
                            "host-resolved typed input bindings; source-data "
                            "provenance cannot be authenticated."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "invalid_resolved_inputs": sorted(invalid_bindings),
                            "reason": "invalid_resolved_input_binding",
                        },
                    )
                ]
            declared_upstream_ids = self._explicit_upstream_step_ids(step_summary)
            contradictory_ids = declared_upstream_ids - upstream_step_ids
            if contradictory_ids:
                return [
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' reports upstream "
                            "steps that disagree with its host-resolved typed "
                            "bindings."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "declared_upstream_step_ids": sorted(
                                declared_upstream_ids
                            ),
                            "resolved_upstream_step_ids": sorted(upstream_step_ids),
                            "reason": "resolved_upstream_binding_mismatch",
                        },
                    )
                ]
        unsafe_step_ids = sorted(
            step_id
            for step_id in upstream_step_ids
            if not self._safe_step_id(step_id)
        )
        if unsafe_step_ids:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' declared unsafe upstream "
                        "step identifiers. Upstream lineage must use plain "
                        "run-local step ids, never paths."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "unsafe_upstream_step_ids": unsafe_step_ids,
                    },
                )
            ]
        if not upstream_step_ids:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' wrote source data without "
                        "declaring any upstream step. Source-data provenance "
                        "cannot be verified without an exact run-local parent."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "source_tables": [path.name for path in source_tables],
                        "reason": "missing_upstream_step_binding",
                    },
                )
            ]
        upstream_step_ids = {
            step_id for step_id in upstream_step_ids if self._safe_step_id(step_id)
        }

        authoritative_evidence = self._authoritative_table_evidence(
            run_dir=run_dir,
            completed_step_records=completed_step_records,
        )
        authoritative_tables = (
            None
            if authoritative_evidence is None
            else {
                item["path"]
                for item in authoritative_evidence.values()
            }
        )
        if same_step_tables and authoritative_tables is not None:
            authoritative_tables.update(same_step_tables)

        required_table_paths: Set[Path] = set()
        required_statistics: Dict[str, tuple[str, float]] = dict(
            same_step_statistics
        )
        table_products: Dict[Path, str] = dict(same_step_tables)
        table_frames: Dict[Path, pd.DataFrame] = {}
        declared_table_aliases: Dict[str, Set[Path]] = {}
        bound_tabular_paths: Set[Path] = set()
        unsupported_value_inputs: List[str] = []
        if resolved_input_bindings is not None:
            invalid_bound_evidence: List[str] = []
            current_records = {
                str(record.get("step_id") or "").strip(): record
                for record in current_successful_step_records(
                    completed_step_records or []
                )
            }
            for raw_input, binding in bound_input_bindings.items():
                declared_kind = str(
                    binding.get("declared_kind") or ""
                ).strip().lower()
                evidence_id = str(binding.get("evidence_id") or "").strip()
                producer_id = str(binding.get("produced_by_step") or "").strip()
                product_name = self._normalise(binding.get("product"))
                canonical_product = f"{declared_kind}:{product_name}"
                bound_path = Path(str(binding.get("absolute_path") or ""))
                expected_sha = str(binding.get("sha256") or "").strip().lower()
                if declared_kind == "table":
                    item = (
                        authoritative_evidence.get(evidence_id)
                        if authoritative_evidence is not None
                        else None
                    )
                    if (
                        item is None
                        or item["sha256"] != expected_sha
                        or item["produced_by_step"] != producer_id
                    ):
                        invalid_bound_evidence.append(raw_input)
                        continue
                    bound_path = item["path"]
                    evidence_path = item.get("evidence_path")
                    if isinstance(evidence_path, Path):
                        declared_table_aliases.setdefault(
                            evidence_path.name, set()
                        ).add(bound_path.resolve())
                elif (
                    not self._safe_regular_run_file(bound_path, run_dir=run_dir)
                    or self._sha256_file(bound_path) != expected_sha
                ):
                    invalid_bound_evidence.append(raw_input)
                    continue
                if declared_kind == "statistic":
                    record = current_records.get(producer_id)
                    if (
                        record is None
                        or evidence_id
                        != str(record.get("step_summary_evidence_id") or "").strip()
                        or evidence_id
                        not in {
                            str(item)
                            for item in (record.get("evidence_ids") or [])
                        }
                    ):
                        invalid_bound_evidence.append(raw_input)
                        continue
                    value = self._extract_statistic_value(
                        record.get("step_summary"), product_name
                    )
                    if value is None:
                        invalid_bound_evidence.append(raw_input)
                        continue
                    if not result_families or any(
                        self._source_supports_result_family(
                            product=canonical_product,
                            family=family,
                        )
                        for family in result_families
                    ):
                        required_statistics[raw_input] = (product_name, value)
                    else:
                        unsupported_value_inputs.append(raw_input)
                    continue
                if declared_kind == "model":
                    unsupported_value_inputs.append(raw_input)
                    continue

                if bound_path.suffix.lower() not in self._TABULAR_SUFFIXES:
                    unsupported_value_inputs.append(raw_input)
                    continue
                try:
                    frame = self._read_tabular(bound_path)
                except Exception:
                    invalid_bound_evidence.append(raw_input)
                    continue
                if result_families and not any(
                    self._source_supports_result_family(
                        product=canonical_product,
                        frame=frame,
                        family=family,
                    )
                    for family in result_families
                ):
                    unsupported_value_inputs.append(raw_input)
                    continue
                resolved_path = bound_path.resolve()
                bound_tabular_paths.add(resolved_path)
                required_table_paths.add(resolved_path)
                table_products[resolved_path] = canonical_product
                table_frames[resolved_path] = frame
            if invalid_bound_evidence:
                return [
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' has typed bindings "
                            "that do not resolve to current hash-verified evidence."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "invalid_resolved_inputs": sorted(
                                invalid_bound_evidence
                            ),
                            "reason": "resolved_input_evidence_mismatch",
                        },
                    )
                ]
            bound_tabular_paths.update(same_step_tables)
            required_table_paths.update(same_step_tables)
            authoritative_tables = set(bound_tabular_paths)
        if completed_step_records is not None:
            current_parent_ids = {
                str(record.get("step_id") or "").strip()
                for record in current_successful_step_records(
                    completed_step_records
                )
            }
            same_step_ids = (
                {str(step.step_id)}
                if same_step_tables or same_step_statistics
                else set()
            )
            stale_parent_ids = sorted(
                upstream_step_ids - current_parent_ids - same_step_ids
            )
            if stale_parent_ids:
                return [
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' cites upstream step(s) "
                            "whose latest checkpoint is not successful. Historical "
                            "outputs cannot authenticate a current figure."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "noncurrent_upstream_step_ids": stale_parent_ids,
                        },
                    )
                ]
        upstream_tables = self._upstream_tables(
            run_dir=run_dir,
            current_out_dir=out_dir,
            upstream_step_ids=upstream_step_ids,
            authoritative_tables=authoritative_tables,
        )
        for bound_path in bound_tabular_paths:
            if bound_path not in upstream_tables:
                upstream_tables.append(bound_path)
        for same_step_table in same_step_tables:
            if same_step_table not in upstream_tables:
                upstream_tables.append(same_step_table)
        if not upstream_tables and not required_statistics:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity=(
                        "error" if authoritative_tables is not None else "warning"
                    ),
                    message=(
                        f"Figure step '{step.step_id}' has no replayable, "
                        "hash-verified upstream table or statistic source for its result "
                        "figure. Model files and non-tabular artifacts cannot "
                        "authenticate plotted values by themselves."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "upstream_step_ids": sorted(upstream_step_ids),
                        "source_tables": [p.name for p in source_tables],
                        "unsupported_value_inputs": sorted(
                            set(unsupported_value_inputs)
                        ),
                        "reason": "non_replayable_figure_input",
                    },
                )
            ]

        findings: List[ValidationFinding] = []
        matched_table_paths: Set[Path] = set()
        matched_statistics: Set[str] = set()
        required_figure_obligations: Dict[str, Set[str]] = {
            figure: self._figure_source_obligations(
                step=step,
                figure_product=figure,
            )
            for figure in figure_products
            if self._figure_result_family(
                step=step,
                figure_product=figure,
            )
            is not None
        }
        matched_figure_obligations: Dict[str, Set[str]] = {
            figure: set() for figure in required_figure_obligations
        }

        def credit_table_source(
            source_path: Path,
            source_frame: pd.DataFrame,
            table_paths: Set[Path],
        ) -> None:
            for table_path in table_paths:
                resolved = table_path.resolve()
                product = table_products.get(resolved, f"table:{table_path.stem}")
                frame = table_frames.get(resolved)
                if frame is None:
                    try:
                        frame = self._read_tabular(resolved)
                    except Exception:
                        continue
                    table_frames[resolved] = frame
                semantic_product = self._contract_scoped_effect_product(
                    product=product,
                    source_frame=source_frame,
                    upstream_step_id=self._table_step_id(table_path, run_dir=run_dir),
                    completed_step_records=completed_step_records,
                )
                for figure in source_figure_products.get(
                    source_path.resolve(), set()
                ):
                    if figure not in required_figure_obligations:
                        continue
                    family = self._figure_result_family(
                        step=step,
                        figure_product=figure,
                    )
                    if family == "prediction":
                        matched_figure_obligations[figure].update(
                            required_figure_obligations[figure]
                            & self._prediction_source_obligations(
                                product=product,
                                frame=frame,
                            )
                        )
                    elif self._source_supports_figures(
                        step=step,
                        product=semantic_product,
                        frame=frame,
                        figure_products=[figure],
                        require_all=True,
                    ):
                        matched_figure_obligations[figure].update(
                            required_figure_obligations[figure]
                        )

        def credit_statistic_source(
            source_path: Path,
            statistic_ids: Set[str],
        ) -> None:
            for statistic_id in statistic_ids:
                product_name, expected = required_statistics[statistic_id]
                product = f"statistic:{product_name}"
                for figure in source_figure_products.get(
                    source_path.resolve(), set()
                ):
                    if figure not in required_figure_obligations:
                        continue
                    family = self._figure_result_family(
                        step=step,
                        figure_product=figure,
                    )
                    if family == "prediction":
                        matched_figure_obligations[figure].update(
                            required_figure_obligations[figure]
                            & self._prediction_source_obligations(
                                product=product,
                                frame=None,
                                statistic_value=expected,
                            )
                        )
                    elif self._source_supports_figures(
                        step=step,
                        product=product,
                        frame=None,
                        figure_products=[figure],
                        require_all=True,
                    ):
                        matched_figure_obligations[figure].update(
                            required_figure_obligations[figure]
                        )

        for source_path in source_tables:
            if not self._safe_regular_run_file(source_path, run_dir=run_dir):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table {source_path.name} is not "
                            "a regular, non-symlink file contained by this run."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "reason": "unsafe_source_data_path",
                        },
                    )
                )
                continue
            try:
                source_df = pd.read_csv(source_path)
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=f"Could not read figure source-data table {source_path.name}: {exc}",
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "reason": "source_data_read_failed",
                        },
                    )
                )
                continue
            if source_df.empty:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table {source_path.name} is "
                            "empty and cannot authenticate a rendered result."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "reason": "source_data_empty",
                        },
                    )
                )
                continue
            unsafe_declared_tables = sorted(
                {
                    str(item).strip()
                    for item in source_df.get(
                        "source_table", pd.Series(dtype=object)
                    ).dropna()
                    if str(item).strip()
                    and not self._safe_declared_table_name(str(item).strip())
                }
            )
            if unsafe_declared_tables:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table '{source_path.name}' "
                            "declares an unsafe source_table path. The claim must "
                            "be one plain upstream filename."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "unsafe_declared_source_tables": unsafe_declared_tables,
                        },
                    )
                )
                continue
            unsafe_declared_steps = sorted(
                {
                    str(item).strip()
                    for item in source_df.get(
                        "source_step_id", pd.Series(dtype=object)
                    ).dropna()
                    if str(item).strip() and not self._safe_step_id(item)
                }
            )
            if unsafe_declared_steps:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table '{source_path.name}' "
                            "declares an unsafe source_step_id."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "unsafe_declared_source_step_ids": unsafe_declared_steps,
                        },
                    )
                )
                continue
            findings.extend(
                self._percentage_count_consistency_findings(
                    source_df=source_df,
                    source_path=source_path,
                    step_id=step.step_id,
                )
            )
            findings.extend(
                self._structured_sensitivity_trace_findings(
                    source_df=source_df,
                    source_path=source_path,
                    step_id=step.step_id,
                    run_dir=run_dir,
                    upstream_step_ids=upstream_step_ids,
                )
            )
            source_statistic_matches: Set[str] = set()
            for statistic_id, (product_name, expected_value) in (
                required_statistics.items()
            ):
                if self._source_contains_statistic(
                    source_df,
                    product_name=product_name,
                    expected=expected_value,
                ):
                    source_statistic_matches.add(statistic_id)
            source_matched_table_paths: Set[Path] = set()

            def finalize_source_match() -> bool:
                if source_matched_table_paths:
                    matched_table_paths.update(source_matched_table_paths)
                    credit_table_source(
                        source_path,
                        source_df,
                        source_matched_table_paths,
                    )
                    if source_statistic_matches:
                        matched_statistics.update(source_statistic_matches)
                        credit_statistic_source(
                            source_path, source_statistic_matches
                        )
                    return True
                if not source_statistic_matches:
                    return False
                statistic_issue = self._statistic_payload_issue(
                    source_df,
                    required_statistics=required_statistics,
                )
                if statistic_issue is not None:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="error",
                            message=(
                                f"Figure source-data table '{source_path.name}' "
                                "contains numeric result payload that is not "
                                "bound to a verified statistic."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "source_table": source_path.name,
                                **statistic_issue,
                            },
                        )
                    )
                    return False
                matched_statistics.update(source_statistic_matches)
                credit_statistic_source(source_path, source_statistic_matches)
                return True

            # A faithful figure may bind a table produced by any exact typed
            # parent, not merely a step-id naming convention.  ``source_table``
            # remains a binding claim: when present, only that basename (and,
            # when supplied, that exact source_step_id) may authenticate rows.
            declared_tables = self._resolve_declared_source_tables_across_run(
                run_dir=run_dir,
                source_df=source_df,
                current_out_dir=out_dir,
                authoritative_tables=authoritative_tables,
                allowed_step_ids=(
                    upstream_step_ids
                    if resolved_input_bindings is not None
                    else None
                ),
            )
            candidate_tables = list(upstream_tables)
            for path in declared_tables:
                if path not in candidate_tables:
                    candidate_tables.append(path)
            declared_row_names = pd.Series("", index=source_df.index, dtype=str)
            if "source_table" in source_df.columns:
                declared_row_names = source_df["source_table"].map(
                    lambda item: (
                        Path(str(item)).name
                        if pd.notna(item) and str(item).strip()
                        else ""
                    )
                )
            declared_names = {
                name for name in declared_row_names.astype(str) if name.strip()
            }
            comparisons: List[Dict[str, Any]] = []
            ordered_upstream_tables: List[Path] = []
            if declared_names:
                # ``source_table`` is a binding provenance claim, not merely a
                # routing hint.  Validate each row group only against tables with
                # the declared basename; an unrelated sibling table must never
                # launder a forged declaration by happening to share keys/values.
                blank_rows = declared_row_names.eq("")
                if blank_rows.any():
                    comparisons.append(
                        {
                            "ok": False,
                            "reason": "missing_declared_source_table",
                            "n_rows": int(blank_rows.sum()),
                            "message": (
                                "source_table is declared for this figure, but "
                                f"{int(blank_rows.sum())} source-data row(s) do "
                                "not name their upstream table"
                            ),
                        }
                    )
                for declared_name in sorted(declared_names):
                    group_df = source_df.loc[
                        declared_row_names.eq(declared_name)
                    ].copy()
                    group_tables = sorted(
                        {
                            path
                            for path in candidate_tables
                            if path.name == declared_name
                            or path.resolve()
                            in declared_table_aliases.get(declared_name, set())
                        },
                        key=str,
                    )
                    declared_parent_step: Optional[str] = None
                    if "source_step_id" in group_df.columns:
                        declared_step_values = group_df["source_step_id"].map(
                            lambda item: (
                                str(item).strip() if pd.notna(item) else ""
                            )
                        )
                        declared_parent_steps = {
                            item for item in declared_step_values if item
                        }
                        if (
                            len(declared_parent_steps) != 1
                            or declared_step_values.eq("").any()
                        ):
                            comparisons.append(
                                {
                                    "ok": False,
                                    "reason": "ambiguous_declared_source_step",
                                    "declared_source_table": declared_name,
                                    "declared_source_step_ids": sorted(
                                        declared_parent_steps
                                    ),
                                    "message": (
                                        f"declared source table {declared_name} "
                                        "must identify exactly one source_step_id"
                                    ),
                                }
                            )
                            continue
                        declared_parent_step = next(iter(declared_parent_steps))
                        group_tables = [
                            path
                            for path in group_tables
                            if self._table_step_id(path, run_dir=run_dir)
                            == declared_parent_step
                        ]
                    elif len(group_tables) > 1:
                        comparisons.append(
                            {
                                "ok": False,
                                "reason": "ambiguous_declared_source_table_lineage",
                                "declared_source_table": declared_name,
                                "candidate_source_steps": sorted(
                                    {
                                        self._table_step_id(path, run_dir=run_dir)
                                        for path in group_tables
                                    }
                                ),
                                "message": (
                                    f"declared source table {declared_name} exists "
                                    "in multiple upstream steps; source_step_id is "
                                    "required to bind exact lineage"
                                ),
                            }
                        )
                        continue
                    ordered_upstream_tables.extend(group_tables)
                    if not group_tables:
                        comparisons.append(
                            {
                                "ok": False,
                                "reason": "declared_source_table_not_found",
                                "declared_source_table": declared_name,
                                "message": (
                                    f"declared source table {declared_name} was "
                                    "not found in an upstream step"
                                ),
                            }
                        )
                        continue
                    group_comparison_pairs = [
                        (
                            upstream_path,
                            self._compare_source_to_upstream(
                                source_df=group_df,
                                source_path=source_path,
                                upstream_path=upstream_path,
                            ),
                        )
                        for upstream_path in group_tables
                    ]
                    group_comparisons = [
                        item for _, item in group_comparison_pairs
                    ]
                    comparisons.extend(group_comparisons)
                    if any(item.get("ok") for item in group_comparisons):
                        source_matched_table_paths.update(
                            path.resolve()
                            for path, item in group_comparison_pairs
                            if item.get("ok")
                        )
                        # Keep only failures from groups that have no matching
                        # declared parent.  A duplicate basename in another step
                        # may legitimately be the referenced parent.
                        comparisons = [
                            item
                            for item in comparisons
                            if item not in group_comparisons
                            or item.get("ok")
                        ]
                failed_comparisons = [
                    item for item in comparisons if not item.get("ok")
                ]
                if not failed_comparisons:
                    if finalize_source_match():
                        continue
                comparisons = failed_comparisons
            else:
                ordered_upstream_tables = self._prioritize_declared_source_tables(
                    source_df=source_df,
                    upstream_tables=candidate_tables,
                )
                comparison_pairs = [
                    (
                        upstream_path,
                        self._compare_source_to_upstream(
                            source_df=source_df,
                            source_path=source_path,
                            upstream_path=upstream_path,
                        ),
                    )
                    for upstream_path in ordered_upstream_tables
                ]
                comparisons = [item for _, item in comparison_pairs]
                successful_paths = {
                    path.resolve()
                    for path, item in comparison_pairs
                    if item.get("ok")
                }
                if successful_paths:
                    source_matched_table_paths.update(successful_paths)
                if finalize_source_match():
                    continue
            if not comparisons:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table '{source_path.name}' does "
                            "not reproduce any bound table or statistic value."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "required_statistics": sorted(required_statistics),
                            "reason": "no_verifiable_figure_values",
                        },
                    )
                )
                continue
            actionable = [
                item for item in comparisons if item.get("reason") != "no_shared_key"
            ]
            best = actionable[0] if actionable else comparisons[0]
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure source-data table '{source_path.name}' is not a "
                        "traceable subset of the declared upstream table(s); "
                        f"{best.get('message', 'no matching upstream rows found')}."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "source_table": source_path.name,
                        "upstream_step_ids": sorted(upstream_step_ids),
                        "candidate_upstream_tables": [
                            str(p.relative_to(run_dir)) if p.is_relative_to(run_dir) else str(p)
                            for p in ordered_upstream_tables
                        ],
                        "best_mismatch": best,
                    },
                )
            )
        missing_table_paths = required_table_paths - matched_table_paths
        missing_statistics = set(required_statistics) - matched_statistics
        if missing_table_paths or missing_statistics:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' source-data bundle does "
                        "not cover every bound result source. Each typed parent "
                        "must be independently value-verified."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "missing_bound_tables": sorted(
                            path.name for path in missing_table_paths
                        ),
                        "missing_bound_statistics": sorted(missing_statistics),
                        "reason": "incomplete_source_lineage_coverage",
                    },
                )
            )
        missing_figure_sources = {
            figure: {
                "declared_sources": sorted(
                    path.name
                    for path, products in source_figure_products.items()
                    if figure in products
                ),
                "missing_obligations": sorted(
                    required_obligations
                    - matched_figure_obligations.get(figure, set())
                ),
            }
            for figure, required_obligations in required_figure_obligations.items()
            if not required_obligations.issubset(
                matched_figure_obligations.get(figure, set())
            )
        }
        if missing_figure_sources:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' has a planned result "
                        "figure whose own source bundle is not backed by a "
                        "semantically compatible, value-verified product."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "missing_figure_sources": missing_figure_sources,
                        "reason": "missing_figure_family_source",
                    },
                )
            )
        return findings

    @classmethod
    def _percentage_count_consistency_findings(
        cls,
        *,
        source_df: pd.DataFrame,
        source_path: Path,
        step_id: str,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        for pct_col, count_col, total_col in cls._PCT_COUNT_RULES:
            if not {pct_col, count_col, total_col} <= set(source_df.columns):
                continue
            pct = pd.to_numeric(source_df[pct_col], errors="coerce")
            count = pd.to_numeric(source_df[count_col], errors="coerce")
            total = pd.to_numeric(source_df[total_col], errors="coerce")
            valid = total > 0
            if not valid.any():
                continue
            expected = 100.0 * count[valid] / total[valid]
            observed = pct[valid]
            diff = (observed - expected).abs()
            bad = diff[(diff > 0.05) & ~(observed.isna() & expected.isna())]
            if bad.empty:
                continue
            idx = int(bad.index[0])
            findings.append(
                ValidationFinding(
                    validator=cls.name,
                    severity="error",
                    message=(
                        f"Figure source-data table '{source_path.name}' has "
                        f"inconsistent percentage/count columns: {pct_col} "
                        f"does not match 100*{count_col}/{total_col}."
                    ),
                    detail={
                        "step_id": step_id,
                        "source_table": source_path.name,
                        "pct_column": pct_col,
                        "count_column": count_col,
                        "total_column": total_col,
                        "row_index": idx,
                        "observed_pct": None if pd.isna(pct.loc[idx]) else float(pct.loc[idx]),
                        "expected_pct": None
                        if pd.isna(expected.loc[idx])
                        else float(expected.loc[idx]),
                        "abs_diff": float(bad.loc[idx]),
                    },
                )
            )
        return findings

    @classmethod
    def _structured_sensitivity_trace_findings(
        cls,
        *,
        source_df: pd.DataFrame,
        source_path: Path,
        step_id: str,
        run_dir: Path,
        upstream_step_ids: Set[str],
    ) -> List[ValidationFinding]:
        """Require fitted sensitivity rows to identify their exact model.

        Simple legacy sensitivity tables remain valid.  The stronger contract
        activates only when the parent step declares a full
        ``robustness_model_contracts`` grid; in that case a scalar plot row must
        say which ``spec_id x model_id`` contract and coefficient term supplied
        the estimate.
        """

        required_shape = {
            "spec_id",
            "effect_scale",
            "point_estimate",
            "ci_low",
            "ci_high",
        }
        if not required_shape <= set(source_df.columns):
            return []

        parent_payloads: List[tuple[str, Path, Dict[str, Any]]] = []
        for parent_step_id in sorted(upstream_step_ids):
            outputs_dir = run_dir / "steps" / parent_step_id / "outputs"
            summary_path = outputs_dir / "step_summary.json"
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            contracts = payload.get("robustness_model_contracts")
            if isinstance(contracts, list) and contracts:
                parent_payloads.append((parent_step_id, outputs_dir, payload))
        if not parent_payloads:
            return []

        point = pd.to_numeric(source_df["point_estimate"], errors="coerce")
        low = pd.to_numeric(source_df["ci_low"], errors="coerce")
        high = pd.to_numeric(source_df["ci_high"], errors="coerce")
        estimated = point.notna() & low.notna() & high.notna()
        if "converged" in source_df.columns:
            estimated &= source_df["converged"].map(
                lambda value: cls._normalise(value) in {"true", "1", "yes"}
            )
        if "independent_variant" in source_df.columns:
            estimated &= ~source_df["independent_variant"].map(
                lambda value: cls._normalise(value) in {"false", "0", "no"}
            )
        rows = source_df.loc[estimated].copy()
        if rows.empty:
            return []

        required_trace = {
            "model_id",
            "event_n",
            "exposure_expression",
            "analysis_set",
            "fit_method",
            "coefficient_source_table",
            "coefficient_term",
            "model_contract_source",
        }
        missing_columns = sorted(required_trace - set(rows.columns))
        issues: List[Dict[str, Any]] = []
        if missing_columns:
            issues.append(
                {
                    "issue": "missing_structured_sensitivity_trace_columns",
                    "columns": missing_columns,
                }
            )
        else:
            parent_step_id, outputs_dir, parent = parent_payloads[0]
            primary_model_id = str(parent.get("primary_model_id") or "")
            all_contracts: List[Dict[str, Any]] = []
            for item in parent.get("model_contracts") or []:
                if not isinstance(item, dict):
                    continue
                contract = dict(item)
                if str(contract.get("model_id") or "") == primary_model_id:
                    contract["spec_id"] = "primary"
                    all_contracts.append(contract)
            all_contracts.extend(
                dict(item)
                for item in parent.get("robustness_model_contracts") or []
                if isinstance(item, dict)
            )
            coefficient_cache: Dict[str, Optional[pd.DataFrame]] = {}
            for row_index, row in rows.iterrows():
                spec_id = str(row.get("spec_id") or "")
                model_id = str(row.get("model_id") or "")
                label = f"{spec_id}:{model_id or '<missing>'}"
                blank_fields = [
                    field
                    for field in required_trace
                    if pd.isna(row.get(field)) or not str(row.get(field)).strip()
                ]
                if blank_fields:
                    issues.append(
                        {
                            "row": label,
                            "row_index": int(row_index),
                            "issue": "blank_structured_sensitivity_trace",
                            "fields": sorted(blank_fields),
                        }
                    )
                    continue
                matched = [
                    item
                    for item in all_contracts
                    if str(item.get("spec_id") or "") == spec_id
                    and str(item.get("model_id") or "") == model_id
                ]
                if len(matched) != 1:
                    issues.append(
                        {
                            "row": label,
                            "issue": "ambiguous_model_contract_trace",
                            "matches": len(matched),
                        }
                    )
                    continue
                contract = matched[0]
                for source_field, contract_field in (
                    ("modeled_analytic_n", "n"),
                    ("event_n", "event_n"),
                ):
                    expected = cls._as_float(contract.get(contract_field))
                    reported = cls._as_float(row.get(source_field))
                    if expected is not None and reported != expected:
                        issues.append(
                            {
                                "row": label,
                                "issue": f"{source_field}_contract_mismatch",
                                "expected": expected,
                                "reported": reported,
                            }
                        )
                for field in (
                    "exposure_expression",
                    "analysis_set",
                    "fit_method",
                ):
                    if cls._normalise(row.get(field)) != cls._normalise(
                        contract.get(field)
                    ):
                        issues.append(
                            {
                                "row": label,
                                "issue": f"{field}_contract_mismatch",
                                "expected": contract.get(field),
                                "reported": row.get(field),
                            }
                        )

                coefficient_name = Path(
                    str(row.get("coefficient_source_table") or "")
                ).name
                if coefficient_name not in coefficient_cache:
                    coefficient_path = outputs_dir / coefficient_name
                    try:
                        coefficient_cache[coefficient_name] = pd.read_csv(
                            coefficient_path, float_precision="round_trip"
                        )
                    except Exception:
                        coefficient_cache[coefficient_name] = None
                coefficients = coefficient_cache[coefficient_name]
                if coefficients is None:
                    issues.append(
                        {
                            "row": label,
                            "issue": "coefficient_source_unreadable",
                            "source": coefficient_name,
                        }
                    )
                    continue
                coefficient_match = coefficients[
                    coefficients.get("model_id", pd.Series(dtype=str))
                    .astype(str)
                    .eq(model_id)
                ]
                if "spec_id" in coefficients.columns:
                    coefficient_match = coefficient_match[
                        coefficient_match["spec_id"].astype(str).eq(spec_id)
                    ]
                if "term" in coefficients.columns:
                    coefficient_match = coefficient_match[
                        coefficient_match["term"]
                        .astype(str)
                        .eq(str(row.get("coefficient_term") or ""))
                    ]
                if len(coefficient_match) != 1:
                    issues.append(
                        {
                            "row": label,
                            "issue": "ambiguous_coefficient_trace",
                            "matches": int(len(coefficient_match)),
                            "source": coefficient_name,
                        }
                    )

            if len(parent_payloads) > 1:
                issues.append(
                    {
                        "issue": "multiple_structured_sensitivity_parents",
                        "parents": [item[0] for item in parent_payloads],
                    }
                )

        if not issues:
            return []
        return [
            ValidationFinding(
                validator=cls.name,
                severity="error",
                message=(
                    f"Figure source-data table '{source_path.name}' does not "
                    "preserve the parent step's structured sensitivity-model trace."
                ),
                detail={
                    "step_id": step_id,
                    "source_table": source_path.name,
                    "issues": issues[:50],
                },
            )
        ]

    @classmethod
    def _is_rendering_step(
        cls, *, step: AnalysisStep, step_summary: Dict[str, Any]
    ) -> bool:
        if bool(
            (step_summary or {}).get("rendering_only")
            or (step_summary or {}).get("render_only")
        ):
            return True
        if any(
            (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
            for raw in (step.expected_outputs or [])
        ):
            return True
        method = cls._normalise(step.method)
        if method in {
            "chart_generation",
            "figure",
            "figure_generation",
            "plotting",
            "publication_figure",
            "publication_figure_generation",
            "render_figure",
            "visualisation",
            "visualization",
        }:
            return True
        return any(
            Path(value).suffix.lower()
            in {".png", ".svg", ".pdf", ".tif", ".tiff"}
            for value in cls._iter_string_values(step_summary or {})
        )

    @classmethod
    def _upstream_step_ids(
        cls, *, step: AnalysisStep, step_summary: Dict[str, Any]
    ) -> Set[str]:
        found = cls._explicit_upstream_step_ids(step_summary)

        text = f"{step.intent}\n{step.method}\n{json.dumps(step_summary or {}, default=str)}"
        for match in re.finditer(r"\bstep\s*['\"]([A-Za-z0-9_.:-]+)['\"]", text):
            candidate = match.group(1).strip()
            if candidate and candidate != step.step_id:
                found.add(candidate)

        step_id = str(step.step_id)
        for suffix in (
            "_figure",
            "_publication_figure",
            "_figure_generation",
            "_render_figure",
        ):
            if step_id.endswith(suffix) and len(step_id) > len(suffix):
                found.add(step_id[: -len(suffix)])
        return found

    @classmethod
    def _explicit_upstream_step_ids(
        cls, step_summary: Mapping[str, Any]
    ) -> Set[str]:
        """Return structured producer claims without prose/name inference."""

        found: Set[str] = set()
        for key in (
            "upstream_step_id",
            "source_step_id",
            "producer_step_id",
        ):
            value = (step_summary or {}).get(key)
            if isinstance(value, str) and value.strip():
                found.add(value.strip())
        for key in (
            "upstream_step_ids",
            "source_step_ids",
            "producer_step_ids",
        ):
            value = (step_summary or {}).get(key)
            if isinstance(value, (list, tuple, set)):
                found.update(str(item).strip() for item in value if str(item).strip())
        return found

    @staticmethod
    def _safe_step_id(step_id: Any) -> bool:
        return bool(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*",
                str(step_id or "").strip(),
            )
        )

    @staticmethod
    def _safe_declared_table_name(value: Any) -> bool:
        text = str(value or "").strip()
        return bool(
            text
            and text not in {".", ".."}
            and Path(text).name == text
            and "/" not in text
            and "\\" not in text
        )

    @classmethod
    def _table_step_id(cls, path: Path, *, run_dir: Path) -> str:
        try:
            relative = Path(path).resolve().relative_to(Path(run_dir).resolve())
        except ValueError:
            return ""
        parts = relative.parts
        if len(parts) < 4 or parts[0] != "steps" or parts[2] != "outputs":
            return ""
        return parts[1] if cls._safe_step_id(parts[1]) else ""

    @staticmethod
    def _safe_regular_run_file(path: Path, *, run_dir: Path) -> bool:
        """Require a regular, non-symlink file contained by the run root."""

        root = Path(run_dir).resolve()
        candidate = Path(path)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            lexical_relative = candidate.absolute().relative_to(root)
        except (OSError, ValueError):
            return False
        if not candidate.is_file() or candidate.is_symlink():
            return False
        cursor = root
        for part in lexical_relative.parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink():
                return False
        return True

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _authoritative_table_evidence(
        cls,
        *,
        run_dir: Path,
        completed_step_records: Optional[Sequence[Dict[str, Any]]],
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Resolve active table evidence ids back to immutable step outputs.

        ``None`` is the explicit legacy signal: no modern per-step authority is
        available, so old run fixtures may use the contained filesystem scan.
        A modern run returns a mapping (possibly empty); only current
        successful, hash-matching table artifacts are eligible as parents.
        """

        evidence_records = current_run_evidence_records(
            run_dir,
            per_step_records=completed_step_records,
        )
        if evidence_records is None:
            return None
        current_ids = (
            {
                str(record.get("step_id") or "").strip()
                for record in current_successful_step_records(
                    completed_step_records
                )
            }
            if completed_step_records is not None
            else None
        )
        root = Path(run_dir).resolve()
        authorised: Dict[str, Dict[str, Any]] = {}
        for record in evidence_records:
            if str(record.get("kind") or "").strip().lower() != "table":
                continue
            step_id = str(record.get("produced_by_step") or "").strip()
            if (
                not cls._safe_step_id(step_id)
                or (current_ids is not None and step_id not in current_ids)
            ):
                continue
            expected_sha = str(record.get("sha256") or "").strip().lower()
            evidence_id = str(record.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            evidence_path = verified_run_evidence_path(root, record)
            if evidence_path is None:
                continue
            evidence_name = evidence_path.name
            logical_name = (
                evidence_name.split("__", 1)[1]
                if "__" in evidence_name
                else evidence_name
            )
            if not cls._safe_declared_table_name(logical_name):
                continue
            output_path = root / "steps" / step_id / "outputs" / logical_name
            if (
                cls._safe_regular_run_file(output_path, run_dir=root)
                and cls._sha256_file(output_path) == expected_sha
            ):
                authorised[evidence_id] = {
                    "path": output_path.resolve(),
                    "evidence_path": evidence_path.resolve(),
                    "sha256": expected_sha,
                    "produced_by_step": step_id,
                }
        return authorised

    @classmethod
    def _upstream_tables(
        cls,
        *,
        run_dir: Path,
        current_out_dir: Path,
        upstream_step_ids: Set[str],
        authoritative_tables: Optional[Set[Path]] = None,
    ) -> List[Path]:
        tables: List[Path] = []
        root = Path(run_dir).resolve()
        for step_id in sorted(upstream_step_ids):
            if not cls._safe_step_id(step_id):
                continue
            outputs = run_dir / "steps" / step_id / "outputs"
            if not outputs.exists() or outputs.is_symlink():
                continue
            for path in sorted(outputs.iterdir()):
                if (
                    path.suffix.lower() not in cls._TABULAR_SUFFIXES
                    or not cls._safe_regular_run_file(path, run_dir=root)
                ):
                    continue
                if path.parent.resolve() == current_out_dir.resolve():
                    continue
                if (
                    authoritative_tables is not None
                    and path.resolve() not in authoritative_tables
                ):
                    continue
                tables.append(path)
        return tables

    @classmethod
    def _prioritize_declared_source_tables(
        cls,
        *,
        source_df: pd.DataFrame,
        upstream_tables: Sequence[Path],
    ) -> List[Path]:
        """Put explicitly declared parent tables first.

        Figure source-data tables are often clean, manuscript-facing
        summaries derived from a registered audit table rather than byte-for-
        byte row subsets. A ``source_table`` column is the deterministic
        breadcrumb that says which parent table should be used for provenance
        checks. Keep all tables as fallbacks, but score the declared parent
        first so a coincidental key in an unrelated audit table does not drive
        the mismatch explanation.
        """

        if "source_table" not in source_df.columns:
            return list(upstream_tables)
        declared = {
            Path(str(item)).name
            for item in source_df["source_table"].dropna().astype(str)
            if str(item).strip()
        }
        if not declared:
            return list(upstream_tables)
        return sorted(
            upstream_tables,
            key=lambda path: (path.name not in declared, str(path)),
        )

    @classmethod
    def _resolve_declared_source_tables_across_run(
        cls,
        *,
        run_dir: Path,
        source_df: pd.DataFrame,
        current_out_dir: Path,
        authoritative_tables: Optional[Set[Path]] = None,
        allowed_step_ids: Optional[Set[str]] = None,
    ) -> List[Path]:
        """Locate the figure's self-declared ``source_table`` parents anywhere.

        The ``source_table`` column names the upstream table each figure row was
        derived from. That table may live in ANY prior step's ``outputs/`` (a
        probe/audit table, not just the ``_figure``-suffix sibling), so resolve
        the declared filenames across ``run_dir/steps/*/outputs`` rather than
        only the steps ``_upstream_step_ids`` found. The figure's own output dir
        is excluded so a figure can never be declared traceable to itself.

        Returns the matched parent paths (first-seen order); ``[]`` when the
        column is absent or nothing matches. This only ADDS comparison
        candidates; the caller still runs the subset + value-equality checks, so
        a figure whose values do not match the table it names still fails.
        """

        if "source_table" not in source_df.columns:
            return []
        declared_names = {
            str(item).strip()
            for item in source_df["source_table"].dropna().astype(str)
            if cls._safe_declared_table_name(item)
        }
        if not declared_names:
            return []
        steps_dir = Path(run_dir) / "steps"
        if not steps_dir.exists():
            return []
        current_resolved = current_out_dir.resolve()
        resolved: List[Path] = []
        seen: Set[Path] = set()
        for path in sorted(steps_dir.glob("*/outputs/*")):
            if (
                path.suffix.lower() not in cls._TABULAR_SUFFIXES
                or
                path.name not in declared_names
                or not cls._safe_regular_run_file(path, run_dir=run_dir)
            ):
                continue
            if (
                allowed_step_ids is not None
                and cls._table_step_id(path, run_dir=run_dir)
                not in allowed_step_ids
            ):
                continue
            if path.parent.resolve() == current_resolved:
                continue
            rp = path.resolve()
            if authoritative_tables is not None and rp not in authoritative_tables:
                continue
            if rp not in seen:
                seen.add(rp)
                resolved.append(path)
        return resolved

    @classmethod
    def _compare_source_to_upstream(
        cls,
        *,
        source_df: pd.DataFrame,
        source_path: Path,
        upstream_path: Path,
    ) -> Dict[str, Any]:
        try:
            upstream_df = cls._read_tabular(upstream_path)
        except Exception as exc:
            return {
                "ok": False,
                "reason": "upstream_read_failed",
                "upstream_table": upstream_path.name,
                "message": f"could not read upstream table {upstream_path.name}: {exc}",
            }
        if upstream_df.empty:
            return {
                "ok": False,
                "reason": "upstream_empty",
                "upstream_table": upstream_path.name,
                "message": f"upstream table {upstream_path.name} is empty",
            }

        used_structural_fallback = False
        key_cols = next(
            (
                tuple(cols)
                for cols in cls._COMPOSITE_KEY_COLUMNS
                if all(col in source_df.columns and col in upstream_df.columns for col in cols)
            ),
            None,
        )
        if key_cols is None:
            key = next(
                (
                    col
                    for col in cls._KEY_COLUMNS
                    if col in source_df.columns and col in upstream_df.columns
                ),
                None,
            )
            key_cols = (key,) if key is not None else None
        source = source_df.copy()
        upstream = upstream_df.copy()
        positional_key_label: Optional[str] = None
        selected_position_col: Optional[str] = None
        positional_columns = [
            col for col in cls._POSITIONAL_ROW_INDEX_COLUMNS if col in source.columns
        ]
        parsed_positions: Dict[str, pd.Series] = {}
        for position_col in positional_columns:
            row_index = pd.to_numeric(source[position_col], errors="coerce")
            invalid = (
                row_index.isna()
                | (row_index < 0)
                | (row_index >= len(upstream))
                | (row_index % 1 != 0)
            )
            if invalid.any():
                first_bad = int(invalid[invalid].index[0])
                return {
                    "ok": False,
                    "reason": "source_row_index_out_of_bounds",
                    "key_column": position_col,
                    "upstream_table": upstream_path.name,
                    "message": (
                        f"{position_col} values must be unique integer row "
                        f"positions within {upstream_path.name}; first invalid "
                        f"source-data row is {first_bad}"
                    ),
                }
            parsed_positions[position_col] = row_index.astype(int)

        if len(positional_columns) == 2:
            canonical = parsed_positions["source_row_index"]
            legacy = parsed_positions["_source_row_index"]
            conflict = canonical.ne(legacy)
            if conflict.any():
                first_bad = int(conflict[conflict].index[0])
                return {
                    "ok": False,
                    "reason": "conflicting_source_row_index_aliases",
                    "upstream_table": upstream_path.name,
                    "message": (
                        "source_row_index and _source_row_index must identify "
                        "the same upstream row; first conflict is at source-data "
                        f"row {first_bad}"
                    ),
                }

        if positional_columns:
            selected_position_col = (
                "source_row_index"
                if "source_row_index" in parsed_positions
                else "_source_row_index"
            )

        # A single figure source CSV may use long form for multiple panels: the
        # same parent row then appears once per panel, while a generic column
        # such as ``estimate`` maps to a different upstream measure in each
        # panel.  Validate those panels separately only when every non-empty
        # panel covers the exact same unique parent-position set.  This keeps
        # the grouping structural (not a free-text scientific guess) and still
        # requires every value column in every panel to match its parent rows.
        if selected_position_col is not None and "panel_id" in source.columns:
            panel_ids = source["panel_id"].fillna("").astype(str).str.strip()
            unique_panels = [value for value in panel_ids.unique() if value]
            if len(unique_panels) > 1 and panel_ids.ne("").all():
                panel_position_sets: List[Set[int]] = []
                panel_groups: List[tuple[str, pd.DataFrame]] = []
                panels_are_complete = True
                for panel_id in unique_panels:
                    panel_mask = panel_ids.eq(panel_id)
                    panel_positions = parsed_positions[selected_position_col].loc[
                        panel_mask
                    ]
                    if panel_positions.duplicated().any():
                        panels_are_complete = False
                        break
                    panel_position_sets.append(set(panel_positions.astype(int)))
                    panel_groups.append((panel_id, source.loc[panel_mask].copy()))
                if (
                    panels_are_complete
                    and panel_position_sets
                    and all(
                        positions == panel_position_sets[0]
                        for positions in panel_position_sets[1:]
                    )
                ):
                    panel_results = {
                        panel_id: cls._compare_source_to_upstream(
                            source_df=panel_df,
                            source_path=source_path,
                            upstream_path=upstream_path,
                        )
                        for panel_id, panel_df in panel_groups
                    }
                    failed_panel = next(
                        (
                            (panel_id, result)
                            for panel_id, result in panel_results.items()
                            if not result.get("ok")
                        ),
                        None,
                    )
                    if failed_panel is not None:
                        panel_id, result = failed_panel
                        return {
                            **result,
                            "panel_id": panel_id,
                            "message": (
                                f"panel {panel_id} failed source verification: "
                                f"{result.get('message', 'unknown mismatch')}"
                            ),
                        }
                    return {
                        "ok": True,
                        "reason": "source_subset_matches",
                        "source_table": source_path.name,
                        "upstream_table": upstream_path.name,
                        "key_column": selected_position_col,
                        "n_source_rows": int(len(source_df)),
                        "join_mode": "panel_stratified_positional",
                        "verified_panels": panel_results,
                    }

        if key_cols is None and selected_position_col is not None:
            positional_key_label = selected_position_col
            join_col = "__easyicu_parent_row_position"
            while join_col in source.columns or join_col in upstream.columns:
                join_col = f"_{join_col}"
            source[join_col] = parsed_positions[selected_position_col].astype(str)
            upstream[join_col] = pd.Series(
                range(len(upstream)), index=upstream.index, dtype=int
            ).astype(str)
            key_cols = (join_col,)
        if key_cols is None:
            # Structural fallback: no composite / named / positional key matched,
            # but a faithfully-derived figure often preserves the parent's OWN key
            # column under a name not in _KEY_COLUMNS (e.g. category_code,
            # lactate_group, group). Accept ANY column present in BOTH frames that
            # is (a) not a numeric value/measure and (b) identifier-like in the
            # source (mostly-distinct), choosing the one whose source values best
            # join into the upstream. The value-equality checks below still run on
            # every shared numeric column, so this only enables the JOIN and never
            # masks a fabricated value. This moves traceability OFF the hard-coded
            # key-name allowlist that needed a new entry per case
            # (contrast_id/stage/level/... -> group/category_code) onto structural
            # evidence. Because structurally selected identifiers have no
            # semantic contract, the join is allowed to PASS only when every
            # numeric source-data value column has a same-name upstream value
            # column and is actually checked below. A truthful count must never
            # launder an unrelated renamed/forged estimate. Only reached when the
            # existing resolution already returned no_shared_key, so it cannot
            # change any currently-passing figure's key.
            n_src = max(len(source), 1)
            best: Optional[tuple[tuple[float, float], str]] = None
            for col in source.columns:
                if col not in upstream.columns:
                    continue
                if (
                    col in {*cls._POSITIONAL_ROW_INDEX_COLUMNS, "source_table"}
                    or col in cls._NUMERIC_COLUMNS
                ):
                    continue
                left_num = pd.to_numeric(source[col], errors="coerce")
                right_num = pd.to_numeric(upstream[col], errors="coerce")
                # a column fully numeric in both frames is a value/measure, not a key
                if left_num.notna().all() and right_num.notna().all():
                    continue
                s_vals = source[col].dropna().astype(str)
                if s_vals.empty:
                    continue
                distinct_ratio = s_vals.nunique() / n_src
                if distinct_ratio < 0.5:  # a real per-row key is mostly-distinct
                    continue
                u_vals = set(upstream[col].dropna().astype(str))
                overlap = float(s_vals.isin(u_vals).mean())  # joinable fraction
                score = (overlap, distinct_ratio)
                if overlap > 0 and (best is None or score > best[0]):
                    best = (score, col)
            if best is not None:
                key_cols = (best[1],)
                used_structural_fallback = True
        if key_cols is None:
            return {
                "ok": False,
                "reason": "no_shared_key",
                "upstream_table": upstream_path.name,
                "message": f"no shared key column with {upstream_path.name}",
            }
        for key in key_cols:
            source[key] = source[key].astype(str)
            upstream[key] = upstream[key].astype(str)

        def _key_set(frame: pd.DataFrame) -> Set[tuple[str, ...]]:
            return set(
                frame[list(key_cols)]
                .dropna()
                .astype(str)
                .itertuples(index=False, name=None)
            )

        upstream_keys = _key_set(upstream)
        missing_keys = sorted(_key_set(source) - upstream_keys)
        key_label = positional_key_label or "+".join(key_cols)

        def _format_key(row: pd.Series) -> str:
            return "|".join(str(row[col]) for col in key_cols)

        if missing_keys:
            return {
                "ok": False,
                "reason": "source_rows_not_in_upstream",
                "key_column": key_label,
                "upstream_table": upstream_path.name,
                "missing_keys": ["|".join(item) for item in missing_keys[:20]],
                "n_missing_keys": len(missing_keys),
                "message": (
                    f"{len(missing_keys)} {key_label} value(s) are absent from "
                    f"{upstream_path.name}"
                ),
            }

        merged = source.merge(
            upstream,
            on=list(key_cols),
            how="left",
            suffixes=("_source", "_upstream"),
        )
        mismatches: List[Dict[str, Any]] = []
        ignored_for_dynamic_numeric = {
            *key_cols,
            *cls._POSITIONAL_ROW_INDEX_COLUMNS,
            "source_table",
        }
        shared_columns = (set(source.columns) & set(upstream.columns)) - set(key_cols)
        text_name = re.compile(
            r"(?:^|_)(?:label|name|category|group|stratum|term|id|level|stage|band|role|status|"
            r"method|table|source|description|note)(?:_|$)"
        )
        value_name = re.compile(
            r"(?:^|_)(?:estimate|effect|rate|risk|odds|hazard|ratio|percent|pct|"
            r"count|ci|lower|upper|mean|median|se|p|statistic|value|n)(?:_|$)"
        )

        def _clean_numeric(raw: pd.Series) -> pd.Series:
            # Figure source tables sometimes serialize display values as
            # ``91%`` or ``1,234``.  Parse those forms for verification while
            # retaining the original missing/non-finite semantics below.
            text = raw.astype(str).str.strip()
            text = text.str.replace(",", "", regex=False)
            text = text.str.replace("%", "", regex=False)
            text = text.str.replace("−", "-", regex=False)
            text = text.str.replace(
                r"^\(([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\)$",
                r"-\1",
                regex=True,
            )
            return pd.to_numeric(text, errors="coerce").astype(float)

        def _is_value_column(frame: pd.DataFrame, col: str) -> bool:
            if col in ignored_for_dynamic_numeric or col in cls._TEXT_COLUMNS:
                return False
            raw = frame[col]
            if pd.api.types.is_bool_dtype(raw) or str(col).lower() in {
                "is_continuous",
                "treated",
            }:
                return False
            present = raw.notna() & raw.astype(str).str.strip().ne("")
            if not present.any():
                return False
            parsed = _clean_numeric(raw[present])
            numeric_evidence = bool(
                pd.api.types.is_numeric_dtype(raw) or parsed.notna().all()
            )
            # Text-like suffixes normally identify labels/roles rather than
            # values.  A name that also declares a value role (for example a
            # numeric ``estimate_label``) must not escape verification merely
            # because it contains ``label``.
            if text_name.search(str(col).lower()) and not (
                value_name.search(str(col).lower()) and numeric_evidence
            ):
                return False
            return bool(
                col in cls._NUMERIC_COLUMNS
                or numeric_evidence
                or value_name.search(str(col).lower())
            )

        source_value_columns = {
            col for col in source.columns if _is_value_column(source, col)
        }
        upstream_value_columns = {
            col for col in upstream.columns if _is_value_column(upstream, col)
        }

        def _merged_source(col: str) -> pd.Series:
            suffixed = f"{col}_source"
            return merged[suffixed] if suffixed in merged.columns else merged[col]

        def _merged_upstream(col: str) -> pd.Series:
            suffixed = f"{col}_upstream"
            return merged[suffixed] if suffixed in merged.columns else merged[col]

        def _numeric_comparison(
            source_name: str, upstream_name: str
        ) -> tuple[bool, bool, pd.Series, pd.Series, pd.Series, pd.Series]:
            left_raw = _merged_source(source_name)
            right_raw = _merged_upstream(upstream_name)
            left_present = left_raw.notna() & left_raw.astype(str).str.strip().ne("")
            right_present = right_raw.notna() & right_raw.astype(str).str.strip().ne("")
            left = _clean_numeric(left_raw)
            right = _clean_numeric(right_raw)
            left_finite = left.notna() & left.map(math.isfinite)
            right_finite = right.notna() & right.map(math.isfinite)
            comparable = left_present & right_present & left_finite & right_finite
            abs_tolerance = (
                cls._PERCENTAGE_ABS_TOL
                if any(
                    token in name.lower()
                    for name in (source_name, upstream_name)
                    for token in ("_pct", "percent")
                )
                else cls._DEFAULT_NUMERIC_ABS_TOL
            )
            diff = (left - right).abs()
            same_nonfinite = (
                left_present
                & right_present
                & left.eq(right)
                & ~left_finite
                & ~right_finite
            )
            parse_failure = (
                (left_present & left.isna()) | (right_present & right.isna())
            )
            bad = (
                (left_present ^ right_present)
                | parse_failure
                | (comparable & (diff > abs_tolerance))
                | (
                    left_present
                    & right_present
                    & ~comparable
                    & ~same_nonfinite
                    & ~parse_failure
                )
            )
            return (
                bool(comparable.any() and not bad.any()),
                bool(bad.any()),
                bad,
                left,
                right,
                diff,
            )

        def _value_family(col: str) -> str:
            """Classify value-column names for safe cross-name matching.

            Row-aligned equality proves that a numeric vector came from the
            parent table, but it does not by itself prove semantic identity. A
            count vector must not authenticate an ``estimate`` merely because
            the numbers happen to coincide.  Keep the families deliberately
            small and case-neutral; same-name comparisons remain authoritative.
            """

            name = re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip(
                "_"
            )
            tokens = set(name.split("_")) if name else set()
            if tokens & {"percent", "percentage", "pct"}:
                return "percent"
            if (
                name in {"n", "count", "denominator", "sample_size"}
                or tokens & {"count", "events", "deaths"}
                or "denominator" in tokens
                or "sample" in tokens and "size" in tokens
                or name.startswith("n_")
                or name.endswith("_n")
            ):
                return "count"
            if tokens & {
                "risk",
                "rate",
                "proportion",
                "prevalence",
                "incidence",
                "probability",
            }:
                return "rate"
            if (
                ("ci" in tokens and tokens & {"low", "lower", "lcl"})
                or tokens & {"lcl"}
            ):
                return "ci_low"
            if (
                ("ci" in tokens and tokens & {"high", "upper", "ucl"})
                or tokens & {"ucl"}
            ):
                return "ci_high"
            if name in {
                "se",
                "stderr",
                "std_err",
                "std_error",
                "standard_err",
                "standard_error",
            } or (
                bool(tokens & {"std", "standard"})
                and bool(tokens & {"err", "error"})
            ):
                return "standard_error"
            if name in {"p", "pval", "p_val", "pvalue", "p_value"} or (
                "p" in tokens and bool(tokens & {"val", "value"})
            ):
                return "p_value"
            if tokens & {"mean", "median", "quantile"}:
                return "location_summary"
            if tokens & {"order", "position", "rank"}:
                return "ordering"
            if tokens & {"ratio", "odds", "hazard"} or name in {
                "or",
                "hr",
                "rr",
            }:
                return "ratio"
            if tokens & {"estimate", "effect", "statistic"}:
                return "generic_estimate"
            if "value" in tokens:
                return "generic_value"
            return "other_numeric"

        def _structured_source_family(source_name: str) -> str:
            family = _value_family(source_name)
            if family != "generic_estimate":
                return family
            semantic_values: Set[str] = set()
            for semantic_col in ("value_type", "estimate_type", "effect_scale"):
                if semantic_col not in source.columns:
                    continue
                semantic_values.update(
                    cls._normalise(item)
                    for item in source[semantic_col].dropna().astype(str)
                    if str(item).strip()
                )
            semantic_families: Set[str] = set()
            if semantic_values & {
                "distribution",
                "continuous_distribution",
                "distribution_mean",
                "distribution_median",
                "location_summary",
                "mean",
                "median",
                "quantile",
            }:
                semantic_families.add("location_summary")
            if semantic_values & {
                "risk",
                "rate",
                "probability",
                "prevalence",
                "incidence",
                "absolute_risk",
                "event_rate",
                "mortality_rate",
            }:
                semantic_families.add("rate")
            if semantic_values & {
                "odds",
                "hazard",
                "ratio",
                "association",
                "effect",
                "odds_ratio",
                "hazard_ratio",
                "risk_ratio",
                "association_estimate",
                "effect_estimate",
                "or",
                "hr",
                "rr",
            }:
                semantic_families.add("ratio")
            if len(semantic_families) == 1:
                return next(iter(semantic_families))
            return family

        def _cross_name_families_compatible(
            source_name: str, upstream_name: str
        ) -> bool:
            source_family = _structured_source_family(source_name)
            upstream_family = _value_family(upstream_name)
            # Unknown/generic numeric names have no semantic contract.  Exact
            # same-name columns were already handled above; across names, an
            # equal vector such as ``display_metric`` == ``age`` must not be
            # treated as proof that the displayed quantity came from the
            # claimed upstream measure.
            if source_family in {
                "generic_value",
                "other_numeric",
            } or upstream_family in {"generic_value", "other_numeric"}:
                return False
            # Ordering is presentation metadata, not a scientific value
            # family. Only the explicit ``plot_*`` derivation below may bind
            # it to a complete row-aligned upstream ordering vector.
            if "ordering" in {source_family, upstream_family}:
                return False
            if "count" in {source_family, upstream_family}:
                return source_family == upstream_family
            # Percent-labelled columns require either a same-family raw vector
            # or the explicit derived percentage logic below; they cannot
            # silently inherit the scale of a 0-1 risk/rate column.
            if "percent" in {source_family, upstream_family}:
                return source_family == upstream_family
            inferential_specific = {
                "ci_low",
                "ci_high",
                "standard_error",
                "p_value",
            }
            if source_family in inferential_specific or upstream_family in inferential_specific:
                return source_family == upstream_family
            # A presentation-neutral estimate may project a rate/risk or ratio
            # when its complete vector matches.  Location summaries require a
            # structured source semantic (value_type/estimate_type/effect_scale)
            # so an unrelated mean-age vector cannot authenticate an outcome
            # estimate merely because the numbers happen to coincide.
            # e.g. a renderer's ``estimate`` may faithfully project an
            # upstream ``mortality_rate``.
            if source_family == "generic_estimate":
                return upstream_family in {"rate", "ratio"}
            return source_family == upstream_family

        def _explicit_semantic_target_columns(source_name: str) -> List[str]:
            """Resolve a concrete source declaration to its named parent value.

            A declaration such as ``value_type=mortality_rate`` is stronger
            than the broad ``rate`` family inferred from it. When that exact
            normalised value column exists upstream, bind to it so a sibling
            rate/effect column with coincident values cannot authenticate the
            claim. Generic declarations that do not name an upstream column
            retain the family-level compatibility path below.
            """

            declared = {
                cls._normalise(item)
                for semantic_col in ("value_type", "estimate_type", "effect_scale")
                if semantic_col in source.columns
                for item in source[semantic_col].dropna().astype(str)
                if str(item).strip()
            }
            if not declared:
                return []
            return sorted(
                upstream_col
                for upstream_col in upstream_value_columns
                if cls._normalise(upstream_col) in declared
                and _cross_name_families_compatible(source_name, upstream_col)
            )

        verified_value_mappings: Dict[str, str] = {}
        used_upstream_value_columns: Set[str] = set()
        ambiguous_value_mappings: Dict[str, List[str]] = {}
        for source_col in sorted(source_value_columns):
            # A same-name value is authoritative: if it disagrees, never search
            # another column for a coincidental numeric match that could launder
            # the mismatch.
            if source_col in upstream_value_columns:
                verified, disagrees, bad, left, right, diff = _numeric_comparison(
                    source_col, source_col
                )
                if verified:
                    verified_value_mappings[source_col] = source_col
                    used_upstream_value_columns.add(source_col)
                elif disagrees:
                    idx = int(bad[bad].index[0])
                    abs_tolerance = (
                        cls._PERCENTAGE_ABS_TOL
                        if any(
                            token in source_col.lower()
                            for token in ("_pct", "percent")
                        )
                        else cls._DEFAULT_NUMERIC_ABS_TOL
                    )
                    mismatches.append(
                        {
                            "column": source_col,
                            "upstream_column": source_col,
                            "key": _format_key(merged.loc[idx]),
                            "source": (
                                None if pd.isna(left.loc[idx]) else float(left.loc[idx])
                            ),
                            "upstream": (
                                None if pd.isna(right.loc[idx]) else float(right.loc[idx])
                            ),
                            "abs_diff": (
                                None if pd.isna(diff.loc[idx]) else float(diff.loc[idx])
                            ),
                            "abs_tolerance": abs_tolerance,
                        }
                    )
                continue

            explicit_targets = _explicit_semantic_target_columns(source_col)
            if explicit_targets:
                if len(explicit_targets) > 1:
                    ambiguous_value_mappings[source_col] = explicit_targets
                    continue
                target = explicit_targets[0]
                verified, disagrees, bad, left, right, diff = _numeric_comparison(
                    source_col, target
                )
                if verified:
                    verified_value_mappings[source_col] = target
                    used_upstream_value_columns.add(target)
                elif disagrees:
                    idx = int(bad[bad].index[0])
                    mismatches.append(
                        {
                            "column": source_col,
                            "upstream_column": target,
                            "semantic_binding": True,
                            "key": _format_key(merged.loc[idx]),
                            "source": (
                                None if pd.isna(left.loc[idx]) else float(left.loc[idx])
                            ),
                            "upstream": (
                                None if pd.isna(right.loc[idx]) else float(right.loc[idx])
                            ),
                            "abs_diff": (
                                None if pd.isna(diff.loc[idx]) else float(diff.loc[idx])
                            ),
                        }
                    )
                # A concrete declaration is binding: never search a sibling
                # same-family column after its named target disagrees.
                continue

            # Renderers may use a presentation-neutral alias (for example
            # ``ci_low`` for upstream ``or_ci_low``).  Verify renamed values by
            # their complete row-aligned numeric vector and record the mapping;
            # zero comparisons or a partial/mixed match never count as proof.
            if source_col.startswith("plot_"):
                continue
            matching_upstream_columns: List[str] = []
            for upstream_col in sorted(upstream_value_columns):
                if upstream_col in used_upstream_value_columns:
                    continue
                if not _cross_name_families_compatible(source_col, upstream_col):
                    continue
                verified, _disagrees, _bad, _left, _right, _diff = (
                    _numeric_comparison(source_col, upstream_col)
                )
                if verified:
                    matching_upstream_columns.append(upstream_col)
            if len(matching_upstream_columns) == 1:
                matched = matching_upstream_columns[0]
                verified_value_mappings[source_col] = matched
                used_upstream_value_columns.add(matched)
            elif len(matching_upstream_columns) > 1:
                ambiguous_value_mappings[source_col] = matching_upstream_columns

        def _derived_matches(
            source_col: str,
            expected_vectors: Sequence[pd.Series],
            *,
            tolerance: Optional[float] = None,
        ) -> bool:
            tolerance = (
                cls._DEFAULT_NUMERIC_ABS_TOL
                if tolerance is None
                else tolerance
            )
            left_raw = _merged_source(source_col)
            left_present = left_raw.notna() & left_raw.astype(str).str.strip().ne("")
            left = _clean_numeric(left_raw)
            if not left_present.any() or (left_present & left.isna()).any():
                return False
            for expected in expected_vectors:
                expected = pd.to_numeric(expected, errors="coerce").astype(float)
                comparable = (
                    left_present
                    & left.notna()
                    & expected.notna()
                    & left.map(math.isfinite)
                    & expected.map(math.isfinite)
                )
                matched = comparable & ((left - expected).abs() <= tolerance)
                if comparable.any() and (matched | ~left_present).all():
                    return True
            return False

        # Derived display columns remain fail-closed, but can be authenticated
        # from already verified source values.  This preserves honest renderer
        # aliases without allowing an unrelated truthful count to launder a
        # forged estimate.
        for source_col in sorted(
            source_value_columns - set(verified_value_mappings)
        ):
            source_family = _structured_source_family(source_col)
            for verified_source_col in sorted(verified_value_mappings):
                verified_family = _structured_source_family(verified_source_col)
                compatible_alias = source_family == verified_family
                if source_family == "generic_estimate":
                    compatible_alias = verified_family in {"rate", "ratio"}
                if not compatible_alias or source_family in {
                    "generic_value",
                    "other_numeric",
                    "ordering",
                }:
                    continue
                if _derived_matches(
                    source_col,
                    [_clean_numeric(_merged_source(verified_source_col))],
                ):
                    verified_value_mappings[source_col] = (
                        f"derived:alias({verified_source_col})"
                    )
                    break

        for width_col in ("ci_width", "errorbar_width"):
            if (
                width_col in source_value_columns
                and "ci_low" in verified_value_mappings
                and "ci_high" in verified_value_mappings
                and _derived_matches(
                    width_col,
                    [
                        _clean_numeric(_merged_source("ci_high"))
                        - _clean_numeric(_merged_source("ci_low"))
                    ],
                )
            ):
                verified_value_mappings[width_col] = "derived:ci_high-ci_low"

        # A renderer may make an upstream long table presentation-ready by
        # adding a denominator and percentage. Authenticate that denominator
        # only when it equals the complete upstream count total within an
        # explicit structural stratum (row/group/estimate type), then derive
        # the percentage from two already-authenticated values. This cannot
        # bless an arbitrary display column or a subset-dependent denominator.
        if (
            "denominator" in source_value_columns
            and "denominator" not in verified_value_mappings
        ):
            grouped_total_candidates: List[tuple[str, pd.Series]] = []
            seen_grouped_totals: Set[tuple[str, str]] = set()
            for source_count_col in ("count", "n", "membership_n", "n_included"):
                upstream_count_col = verified_value_mappings.get(source_count_col)
                if upstream_count_col not in upstream.columns:
                    continue
                upstream_counts = _clean_numeric(upstream[upstream_count_col])
                grouped_total_candidates.append(
                    (
                        f"derived:sum({upstream_count_col})",
                        pd.Series(
                            upstream_counts.sum(min_count=1),
                            index=merged.index,
                            dtype=float,
                        ),
                    )
                )
                for group_col in ("row_type", "group_type", "estimate_type"):
                    pair = (str(upstream_count_col), group_col)
                    if pair in seen_grouped_totals or group_col not in upstream.columns:
                        continue
                    seen_grouped_totals.add(pair)
                    group_key = upstream[group_col].fillna("<missing>").astype(str)
                    totals = upstream_counts.groupby(group_key).sum(min_count=1)
                    merged_group_key = (
                        _merged_upstream(group_col).fillna("<missing>").astype(str)
                    )
                    grouped_total_candidates.append(
                        (
                            f"derived:sum({upstream_count_col})_by_{group_col}",
                            merged_group_key.map(totals),
                        )
                    )
            for derivation, expected in grouped_total_candidates:
                if _derived_matches("denominator", [expected]):
                    verified_value_mappings["denominator"] = derivation
                    break

        if (
            "percentage" in source_value_columns
            and "percentage" not in verified_value_mappings
            and "denominator" in verified_value_mappings
        ):
            denominator = _clean_numeric(_merged_source("denominator")).replace(
                0.0, float("nan")
            )
            percentage_vectors = [
                100.0 * _clean_numeric(_merged_source(count_col)) / denominator
                for count_col in ("count", "n", "membership_n", "n_included")
                if count_col in verified_value_mappings and count_col in source.columns
            ]
            if percentage_vectors and _derived_matches(
                "percentage",
                percentage_vectors,
                tolerance=cls._PERCENTAGE_ABS_TOL,
            ):
                verified_value_mappings["percentage"] = (
                    "derived:100*verified_count/verified_denominator"
                )

        total_candidates = [
            col
            for col in ("total_n", "n_total", "denominator", "denominator_n")
            if col in verified_value_mappings and col in source.columns
        ]
        missing_candidates = [
            col
            for col in (
                "missing_n",
                "value_missing_n",
                "raw_missing_n",
                "analysis_unavailable_n",
            )
            if col in verified_value_mappings and col in source.columns
        ]
        complement_vectors = [
            _clean_numeric(_merged_source(total_col))
            - _clean_numeric(_merged_source(missing_col))
            for total_col in total_candidates
            for missing_col in missing_candidates
        ]
        for measured_col in ("measured_n", "n_nonmissing"):
            if (
                measured_col in source_value_columns
                and complement_vectors
                and _derived_matches(measured_col, complement_vectors)
            ):
                verified_value_mappings[measured_col] = (
                    "derived:denominator-minus-unavailable"
                )
        if (
            "measured_pct" in source_value_columns
            and "measured_n" in verified_value_mappings
            and total_candidates
        ):
            measured = _clean_numeric(_merged_source("measured_n"))
            pct_vectors = [
                100.0
                * measured
                / _clean_numeric(_merged_source(total_col)).replace(0.0, float("nan"))
                for total_col in total_candidates
            ]
            if _derived_matches(
                "measured_pct",
                pct_vectors,
                tolerance=cls._PERCENTAGE_ABS_TOL,
            ):
                verified_value_mappings["measured_pct"] = (
                    "derived:100*measured_n/denominator"
                )

        for plot_col in sorted(
            col
            for col in source_value_columns
            if col.startswith("plot_") and col not in verified_value_mappings
        ):
            if _value_family(plot_col) == "ordering":
                matching_order_columns = []
                for upstream_col in sorted(upstream.columns):
                    if _value_family(upstream_col) != "ordering":
                        continue
                    verified, _disagrees, _bad, _left, _right, _diff = (
                        _numeric_comparison(plot_col, upstream_col)
                    )
                    if verified:
                        matching_order_columns.append(upstream_col)
                if len(matching_order_columns) == 1:
                    verified_value_mappings[plot_col] = (
                        f"derived:ordering({matching_order_columns[0]})"
                    )
                elif len(matching_order_columns) > 1:
                    ambiguous_value_mappings[plot_col] = matching_order_columns
                continue
            target = plot_col.removeprefix("plot_").removesuffix("_pct")
            if "ci_low" in target:
                source_candidates = [
                    col for col in verified_value_mappings if "ci_low" in col
                ]
            elif "ci_high" in target:
                source_candidates = [
                    col for col in verified_value_mappings if "ci_high" in col
                ]
            elif "estimate" in target:
                source_candidates = [
                    col
                    for col in verified_value_mappings
                    if any(token in col for token in ("estimate", "risk", "rate"))
                    and "ci_" not in col
                ]
            else:
                source_candidates = []
            plot_vectors: List[pd.Series] = []
            for candidate in source_candidates:
                base = _clean_numeric(_merged_source(candidate))
                plot_vectors.extend([base, 100.0 * base])
            if plot_vectors and _derived_matches(
                plot_col,
                plot_vectors,
                tolerance=cls._PERCENTAGE_ABS_TOL,
            ):
                verified_value_mappings[plot_col] = (
                    f"derived:display-scale({','.join(source_candidates)})"
                )
        for col in cls._TEXT_COLUMNS:
            source_col = f"{col}_source"
            upstream_col = f"{col}_upstream"
            if source_col not in merged.columns or upstream_col not in merged.columns:
                continue
            left = merged[source_col].fillna("").astype(str).str.strip().str.lower()
            right = merged[upstream_col].fillna("").astype(str).str.strip().str.lower()
            bad = left != right
            if bad.any():
                idx = int(bad[bad].index[0])
                mismatches.append(
                    {
                        "column": col,
                        "key": _format_key(merged.loc[idx]),
                        "source": merged.loc[idx, source_col],
                        "upstream": merged.loc[idx, upstream_col],
                    }
                )
        if mismatches:
            return {
                "ok": False,
                "reason": "source_values_disagree",
                "key_column": key_label,
                "upstream_table": upstream_path.name,
                "mismatches": mismatches[:20],
                "n_mismatches": len(mismatches),
                "message": f"source-data values disagree with {upstream_path.name}",
            }
        unverified_source_columns = sorted(
            source_value_columns - set(verified_value_mappings)
        )
        if not verified_value_mappings or unverified_source_columns:
            if unverified_source_columns:
                verification_detail = (
                    "these source-data value columns were not verified against "
                    "any row-aligned upstream value vector: "
                    f"{unverified_source_columns}; one verified column cannot "
                    "authenticate another renamed, formatted, or transformed value"
                )
            else:
                verification_detail = (
                    "no source-data value column was available for a real "
                    "row-aligned comparison"
                )
            return {
                "ok": False,
                "reason": "no_verifiable_values",
                "key_column": key_label,
                "upstream_table": upstream_path.name,
                "unverified_source_value_columns": unverified_source_columns,
                "verified_source_value_columns": sorted(verified_value_mappings),
                "verified_value_mappings": verified_value_mappings,
                "ambiguous_value_mappings": ambiguous_value_mappings,
                "message": (
                    f"source rows joined to {upstream_path.name} on {key_label}, "
                    f"but {verification_detail}"
                ),
            }
        return {
            "ok": True,
            "reason": "source_subset_matches",
            "source_table": source_path.name,
            "upstream_table": upstream_path.name,
            "key_column": key_label,
            "n_source_rows": int(len(source_df)),
            "verified_value_mappings": verified_value_mappings,
            "join_mode": "structural_fallback" if used_structural_fallback else "declared_key",
        }


class FigureContractQualityValidator:
    """Audit manuscript-facing figure contracts beyond file/source existence."""

    name = "figure_contract_quality"
    _CONTRACT_GLOB = "*.figure_contract.json"
    _FALLBACK_TERMS = (
        "rescue",
        "fallback",
        "placeholder",
        "did not emit exports",
        "no generated figure",
    )
    _RESULT_ROLES = {
        "relationship",
        "robustness",
        "forest_odds_ratio",
        "forest_risk_difference",
        "forest_risk_ratio",
        "association",
        "effect",
        "descriptive_result",
        "primary_estimand",
        "model_performance",
        "calibration",
        "temporal_absolute_risk",
        "survival_effect",
        "phenotype_structure",
        "phenotype_profile",
        "stability",
        "causal_contrast",
        "distribution",
    }
    # Supporting/context panel roles. A figure whose EVERY panel carries one of
    # these roles is an audit/diagnostic/overview figure — legitimately allowed to
    # be single-panel — and must NOT be gated by the manuscript-facing result-figure
    # ">= 2 panels" rule. Decided on the structured panel ``role`` rather than free
    # text, because a supporting figure's id or core_claim can contain a result-role
    # word (e.g. "distribution", "effect") without being a primary result figure.
    _SUPPORTING_ROLES = {
        "audit",
        "diagnostic",
        "qa",
        "qa_only",
        "exploratory",
        "overview",
        "context",
        "data_quality",
        "missingness",
    }
    _RAW_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+){1,}\b")

    def audit(
        self,
        *,
        step: AnalysisStep,
        out_dir: Path,
        run_dir: Path,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        if not FigureSourceDataValidator._is_rendering_step(
            step=step,
            step_summary=step_summary,
        ):
            return []
        findings: List[ValidationFinding] = []
        contract_paths = sorted(out_dir.glob(self._CONTRACT_GLOB))
        if not contract_paths and self._has_figure_exports(out_dir):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' wrote figure exports "
                        "without a .figure_contract.json file; manuscript-facing "
                        "figures must declare panel claims and source evidence."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "out_dir": str(out_dir),
                    },
                )
            )
            return findings
        for contract_path in contract_paths:
            findings.extend(
                self.audit_contract_file(
                    contract_path,
                    step=step,
                    step_summary=step_summary,
                    manuscript_facing=True,
                )
        )
        return findings

    @staticmethod
    def _has_figure_exports(out_dir: Path) -> bool:
        figure_suffixes = {".svg", ".pdf", ".png", ".tiff", ".tif", ".pptx"}
        return any(
            path.is_file() and path.suffix.lower() in figure_suffixes
            for path in out_dir.iterdir()
        )

    def audit_contract_file(
        self,
        contract_path: Path,
        *,
        step: Optional[AnalysisStep] = None,
        step_summary: Optional[Dict[str, Any]] = None,
        manuscript_facing: Optional[bool] = None,
    ) -> List[ValidationFinding]:
        try:
            raw = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=f"Could not read figure contract {contract_path.name}: {exc}",
                    detail={"path": str(contract_path)},
                )
            ]
        if not isinstance(raw, dict):
            return []

        is_manuscript = (
            bool(manuscript_facing)
            if manuscript_facing is not None
            else self._looks_manuscript_facing(raw, contract_path, step, step_summary)
        )
        if not is_manuscript:
            return []

        figure_id = str(raw.get("figure_id") or contract_path.stem)
        panels = raw.get("panels")
        panels_list = panels if isinstance(panels, list) else []
        text_blob = self._contract_text(raw)
        findings: List[ValidationFinding] = []

        fallback_terms = [
            term for term in self._FALLBACK_TERMS if term in text_blob.lower()
        ]
        if fallback_terms:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"{figure_id} is marked as a fallback/rescue figure; "
                        "manuscript-facing figures must be regenerated from "
                        "registered source data instead of accepted as rescue output."
                    ),
                    detail={
                        "path": str(contract_path),
                        "terms": sorted(set(fallback_terms)),
                        "step_id": getattr(step, "step_id", None),
                    },
                )
            )

        result_like = self._is_result_like_contract(raw, panels_list)
        if (
            result_like
            and len(panels_list) < 2
            and not self._is_supporting_figure_step(step)
            and not self._contract_looks_supporting_figure(raw, figure_id)
        ):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"{figure_id} has only {len(panels_list)} panel(s); "
                        "manuscript-facing result figures need at least two "
                        "data-backed panels so the primary estimate, robustness, "
                        "and audit context are not collapsed into one forest plot."
                    ),
                    detail={
                        "path": str(contract_path),
                        "panel_count": len(panels_list),
                        "step_id": getattr(step, "step_id", None),
                    },
                )
            )

        blank_titles = [
            str(panel.get("panel_id") or idx + 1)
            for idx, panel in enumerate(panels_list)
            if isinstance(panel, dict) and not str(panel.get("title") or "").strip()
        ]
        if blank_titles:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"{figure_id} has panel(s) without titles: "
                        + ", ".join(blank_titles)
                    ),
                    detail={"path": str(contract_path), "panel_ids": blank_titles},
                )
            )

        weak_claims = [
            str(panel.get("panel_id") or idx + 1)
            for idx, panel in enumerate(panels_list)
            if isinstance(panel, dict)
            and len(str(panel.get("claim") or "").strip()) < 24
        ]
        if weak_claims:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"{figure_id} has panel(s) with weak or missing claims: "
                        + ", ".join(weak_claims)
                    ),
                    detail={"path": str(contract_path), "panel_ids": weak_claims},
                )
            )

        machine_labels = sorted(
            {
                token
                for token in self._RAW_IDENTIFIER_RE.findall(
                    self._reader_facing_text(raw)
                )
                if token not in {"figure_id", "source_data", "evidence_ids"}
            }
        )
        if machine_labels:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"{figure_id} includes machine-style labels in the "
                        "figure contract; manuscript figures should expose "
                        "reader-facing labels."
                    ),
                    detail={
                        "path": str(contract_path),
                        "examples": machine_labels[:10],
                    },
                )
            )
        return findings

    @classmethod
    def _looks_manuscript_facing(
        cls,
        raw: Dict[str, Any],
        contract_path: Path,
        step: Optional[AnalysisStep],
        step_summary: Optional[Dict[str, Any]],
    ) -> bool:
        haystack = cls._contract_text(raw) + f"\n{contract_path.name}"
        if step is not None:
            haystack += f"\n{step.step_id}\n{step.intent}\n{step.method}"
            haystack += "\n" + json.dumps(
                getattr(step, "expected_outputs", []) or [],
                default=str,
            )
        if step_summary:
            haystack += "\n" + json.dumps(step_summary, default=str)
        lowered = haystack.lower()
        if any(token in lowered for token in ("exploratory", "diagnostic", "qa only")):
            return False
        return any(
            token in lowered
            for token in ("figure", "publication", "manuscript", "render")
        )

    @classmethod
    def _is_result_like_contract(
        cls,
        raw: Dict[str, Any],
        panels: Sequence[Any],
    ) -> bool:
        panel_roles = [
            cls._normalise_supporting_identifier(panel.get("role"))
            for panel in panels
            if isinstance(panel, dict)
        ]
        # An all-supporting-role figure (every panel is audit/diagnostic/overview/…)
        # is not a manuscript-facing PRIMARY result figure. Exclude it here so the
        # ">= 2 panels" rule does not fire on a legitimately single-panel audit or
        # overview figure (e.g. probe_overview, reporting_followup_distribution) whose
        # id/core_claim happens to contain a result-role substring.
        labelled_roles = [role for role in panel_roles if role]
        if (
            panels
            and len(panel_roles) == len(panels)
            and all(role in cls._SUPPORTING_ROLES for role in panel_roles)
        ):
            return False
        if any(role in cls._RESULT_ROLES for role in labelled_roles):
            return True
        text_blob = cls._contract_text(raw).lower()
        return any(role in text_blob for role in cls._RESULT_ROLES)

    # Exact artifact roles retained for legacy contracts whose panel was
    # mistakenly labelled with a result role (for example, role="robustness"
    # on the separate audit_panel figure). Free-text substrings are deliberately
    # excluded: ``audited_primary_effect`` is a primary result, not an audit
    # artifact merely because its identifier contains "audit".
    _SUPPORTING_ARTIFACT_IDS = {
        "audit",
        "audit_panel",
        "data_completeness_panel",
        "data_quality",
        "data_quality_panel",
        "diagnostic",
        "diagnostic_panel",
        "measurement_process_audit",
        "missingness",
        "missingness_measurement_panel",
        "overview",
        "probe_overview",
        "qa",
        "qa_panel",
        "quality_control",
        "quality_control_panel",
    }

    @staticmethod
    def _normalise_supporting_identifier(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"^figure\s*:\s*", "", text)
        return re.sub(r"[^a-z0-9]+", "_", text).strip("_")

    @classmethod
    def _is_supporting_figure_step(cls, step: Optional[AnalysisStep]) -> bool:
        """True when the step is a SUPPORTING audit/QC figure, not the primary
        result figure.

        Such a supplementary figure must not be held to the primary-result
        ">= 2 data-backed panels" rule: its very existence as a SEPARATE figure
        means the audit context is not collapsed into the primary result figure
        (which the rule exists to prevent). Without this, an LLM coder that tags
        a lone audit panel with a result role ("robustness"/"stability") makes a
        supplementary figure hard-fail the whole run — the M3 subphenotype block.
        The deterministic audit renderer additionally emits >= 2 supporting-role
        panels, so this is the belt to that renderer's suspenders.
        """
        if step is None:
            return False
        step_id = cls._normalise_supporting_identifier(
            getattr(step, "step_id", "")
        )
        step_id = re.sub(r"^\d+_", "", step_id)
        if step_id.endswith("_figure"):
            step_id = step_id[: -len("_figure")]
        if step_id in cls._SUPPORTING_ARTIFACT_IDS:
            return True
        expected_outputs = getattr(step, "expected_outputs", None) or []
        return any(
            cls._normalise_supporting_identifier(output)
            in cls._SUPPORTING_ARTIFACT_IDS
            for output in expected_outputs
            if str(output or "").strip().lower().startswith("figure:")
        )

    @classmethod
    def _contract_looks_supporting_figure(
        cls, raw: Dict[str, Any], figure_id: str
    ) -> bool:
        """True when the CONTRACT itself identifies a supporting audit/QC figure.

        The real figure_skill call sites do not thread the step, so an exact
        normalized figure_id remains as a compatibility signal for separately
        registered supporting artifacts. Panel roles are handled structurally
        by :meth:`_is_result_like_contract`; titles, claims, and identifier
        substrings never grant this exemption.
        """
        fid = cls._normalise_supporting_identifier(figure_id)
        return fid in cls._SUPPORTING_ARTIFACT_IDS

    @staticmethod
    def _contract_text(raw: Dict[str, Any]) -> str:
        parts: List[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(raw)
        return "\n".join(parts)

    @staticmethod
    def _reader_facing_text(raw: Dict[str, Any]) -> str:
        parts = [
            str(raw.get("title") or ""),
            str(raw.get("core_claim") or ""),
            str(raw.get("statistics_note") or ""),
        ]
        panels = raw.get("panels")
        if isinstance(panels, list):
            for panel in panels:
                if not isinstance(panel, dict):
                    continue
                parts.extend([
                    str(panel.get("title") or ""),
                    str(panel.get("claim") or ""),
                    str(panel.get("review_risk") or ""),
                ])
        return "\n".join(part for part in parts if part)


class ClinicalConstraintValidator:
    """ICU-specific semantic warnings over planned and executed analyses."""

    name = "clinical_constraint_validator"
    _CAUSAL_STEP_METHODS = {
        "causal_inference",
        "causal_emulation",
        "g_computation",
        "ipw",
        "iptw",
        "propensity_score",
        "psm",
        "target_trial",
        "target_trial_emulation",
        "treatment_response",
        "effect_modification",
        "interaction_model",
    }
    _CAUSAL_STEP_FAMILIES = {
        "causal_inference",
        "treatment_response",
        "reinforcement_learning",
    }

    @staticmethod
    def _normalise(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")

    def audit(
        self,
        *,
        context: ResearchContext,
        step: AnalysisStep,
        out_dir: Path,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        family = (
            (context.user_preferences.inferred_analysis_family or "").lower()
            if context.user_preferences else ""
        )
        question = (context.research_question or "").lower()
        timing = (
            (context.user_preferences.timing_and_design or "").lower()
            if context.user_preferences and context.user_preferences.timing_and_design
            else ""
        )
        combined = " ".join(
            filter(
                None,
                [question, timing, (step.intent or "").lower(), json.dumps(step_summary, ensure_ascii=False).lower()],
            )
        )
        method_head = self._normalise(
            str(step.method or "").lower().split(" with ", 1)[0]
        )
        step_family = self._normalise(step_summary.get("analysis_family"))
        causal_step_owner = (
            method_head in self._CAUSAL_STEP_METHODS
            or step_family in self._CAUSAL_STEP_FAMILIES
        )

        if causal_step_owner:
            if not any(term in combined for term in ("time zero", "time-zero", "eligibility", "anchor", "alignment")):
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "Treatment-effect style analysis without an explicit time-zero or alignment description "
                        "risks immortal time bias. Document eligibility, anchor time, and treatment assignment timing."
                    ),
                    detail={
                        "analysis_family": step_family or family or "unspecified",
                        "method": method_head,
                    },
                ))
            if "post-treatment" in combined:
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "A post-treatment variable appears in the analysis description. "
                        "Confirm this is not conditioning on a mediator or downstream treatment effect."
                    ),
                ))

        if family == "survival" or any(term in combined for term in ("survival", "cox", "kaplan", "hazard")):
            if any(term in combined for term in ("length of stay", "los", "discharge")) and "competing" not in combined:
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "Length-of-stay or discharge-oriented survival analyses often require a competing-risks framing. "
                        "Consider discharge/death competition explicitly rather than a single-event survival model."
                    ),
                ))
            if "time-varying" in combined and "landmark" not in combined and "time updated" not in combined:
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "Time-varying covariates are mentioned without an explicit handling strategy. "
                        "Specify landmarking, time-updated modeling, or another deterministic design."
                    ),
                ))

        return findings


class StatisticalGuard:
    """Broader statistical QA checks beyond per-step numerical consistency."""

    name = "statistical_guard"

    def audit(
        self,
        *,
        context: ResearchContext,
        cohort_path: Path,
        step: AnalysisStep,
        out_dir: Path,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        family = (
            (context.user_preferences.inferred_analysis_family or "").lower()
            if context.user_preferences else ""
        )
        df = pd.read_parquet(cohort_path)

        if family == "prediction_model" or (out_dir / "model_performance_train_test.csv").exists():
            summary_text = json.dumps(step_summary or {}, ensure_ascii=False).lower()

            def _summary_has_any(tokens: Sequence[str]) -> bool:
                return any(token in summary_text for token in tokens)

            perf_csv = out_dir / "model_performance_train_test.csv"
            perf_candidates = [perf_csv] if perf_csv.exists() else []
            if not perf_candidates:
                for candidate in out_dir.glob("*.csv"):
                    name = candidate.name.lower()
                    if any(token in name for token in ("performance", "prediction", "model")):
                        perf_candidates.append(candidate)
            has_performance_metric = _summary_has_any(
                (
                    "auroc",
                    "auc",
                    "brier",
                    "calibration",
                    "cv_auroc",
                    "held_out_auroc",
                    "cv_brier",
                )
            )
            perf_columns: Set[str] = set()
            for candidate in perf_candidates:
                try:
                    perf_columns.update(str(c).lower() for c in pd.read_csv(candidate, nrows=5).columns)
                except Exception:
                    continue
            has_performance_metric = has_performance_metric or any(
                token in perf_columns
                for token in (
                    "auroc",
                    "auc",
                    "brier",
                    "brier_score",
                    "cv_auroc_mean",
                    "held_out_auroc",
                )
            )
            if not has_performance_metric:
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "Prediction-model style analysis did not emit held-out performance artefacts. "
                        "Report train/test (or equivalent validation) performance before publication."
                    ),
                ))
            else:
                has_calibration = (
                    "calibration_slope" in perf_columns
                    or "calibration_intercept" in perf_columns
                    or "brier" in summary_text
                    or "brier_score" in summary_text
                    or "calibration" in summary_text
                )
                if not has_calibration:
                    findings.append(ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message="Prediction model performance is missing calibration_slope or Brier/calibration metadata.",
                    ))
            has_split_metadata = any(
                k in (step_summary or {})
                for k in (
                    "n_train",
                    "n_test",
                    "split_strategy",
                    "cv_folds",
                    "validation_scheme",
                    "cross_validation",
                )
            ) or _summary_has_any(("5-fold", "cross-validation", "cv_folds", "split_strategy"))
            if not has_split_metadata:
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "Prediction analysis did not document a train/test split or equivalent validation scheme. "
                        "Guard against leakage by recording split_strategy, n_train, and n_test."
                    ),
                ))
            if context.target_outcome and context.target_outcome in df.columns:
                try:
                    events = int(pd.to_numeric(df[context.target_outcome], errors="coerce").fillna(0).astype(int).sum())
                except Exception:
                    events = 0
                requested_covariates = len(getattr(context.user_preferences, "covariates", []) or [])
                if requested_covariates > 0 and events < max(10, 10 * requested_covariates):
                    findings.append(ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            f"Only {events} events were available for {requested_covariates} requested adjustment covariates. "
                            "This may be an events-per-variable problem for a stable prediction model."
                        ),
                        detail={"events": events, "requested_covariates": requested_covariates},
                    ))

        if family == "survival" or any(term in (step.intent or "").lower() for term in ("survival", "cox", "kaplan", "hazard")):
            step_text = json.dumps(step_summary, ensure_ascii=False).lower()
            if "cox" in step_text or "cox" in (step.method or "").lower():
                documented = any(
                    token in step_text
                    for token in ("ph_assumption", "proportional hazards", "schoenfeld")
                )
                documented = documented or any("ph" in p.name.lower() and p.suffix.lower() in {".csv", ".json", ".txt"} for p in out_dir.iterdir())
                if not documented:
                    findings.append(ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message="Cox-style survival analysis did not document a proportional-hazards assumption check.",
                    ))

        for csv_path in [p for p in out_dir.iterdir() if p.suffix.lower() == ".csv"]:
            try:
                tab = pd.read_csv(csv_path)
            except Exception:
                continue
            pval_cols = [c for c in tab.columns if c.lower() in {"p", "p_value", "pvalue", "pval"}]
            adjust_cols = [c for c in tab.columns if c.lower() in {"q_value", "adjusted_p", "p_adj", "padj", "fdr"}]
            family_col = next(
                (
                    column
                    for column in ("hypothesis_family_id", "family_id")
                    if column in tab.columns
                ),
                None,
            )
            # A coefficient dump is not itself a hypothesis family. Only a
            # typed, prespecified family can create an actionable multiplicity
            # warning; nuisance and sensitivity rows are never counted merely
            # because they carry p-values.
            if not pval_cols or family_col is None:
                continue
            scoped = tab.copy()
            if "term_role" in scoped.columns:
                roles = scoped["term_role"].map(
                    lambda value: re.sub(
                        r"[^a-z0-9]+", "_", str(value or "").lower()
                    ).strip("_")
                )
                scoped = scoped.loc[
                    ~roles.isin(
                        {"intercept", "adjustment", "availability", "nuisance"}
                    )
                ]
            if "analysis_role" in scoped.columns:
                analysis_roles = scoped["analysis_role"].map(
                    lambda value: re.sub(
                        r"[^a-z0-9]+", "_", str(value or "").lower()
                    ).strip("_")
                )
                scoped = scoped.loc[~analysis_roles.eq("sensitivity")]
            scoped = scoped.loc[scoped[family_col].notna()].copy()
            scoped[family_col] = scoped[family_col].astype(str).str.strip()
            scoped = scoped.loc[scoped[family_col].ne("")]
            warned = False
            for family_id, family_rows in scoped.groupby(family_col, sort=False):
                finite_p_value_count = sum(
                    int(
                        pd.to_numeric(family_rows[column], errors="coerce")
                        .dropna()
                        .between(0.0, 1.0)
                        .sum()
                    )
                    for column in pval_cols
                )
                finite_adjusted_count = sum(
                    int(
                        pd.to_numeric(family_rows[column], errors="coerce")
                        .dropna()
                        .between(0.0, 1.0)
                        .sum()
                    )
                    for column in adjust_cols
                )
                if finite_p_value_count <= 1 or finite_adjusted_count >= finite_p_value_count:
                    continue
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"Table '{csv_path.name}' contains multiple p-values without an adjusted-p / q-value column. "
                        "If this is a family of simultaneous tests, control multiplicity explicitly."
                    ),
                    detail={
                        "table": csv_path.name,
                        "hypothesis_family_id": str(family_id),
                        "p_value_columns": pval_cols,
                        "finite_p_value_count": finite_p_value_count,
                        "finite_adjusted_p_count": finite_adjusted_count,
                    },
                ))
                warned = True
                break
            if warned:
                break

        return findings


class ReplicationDesignAuditor:
    """Validate whether a parsed paper is reproducible in EasyICU."""

    name = "replication_design_auditor"

    def audit(
        self,
        *,
        paper_profile: PaperProfile,
        deviation_report: ReplicationDeviationReport,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        if paper_profile.paper_type == "unsupported_or_underspecified":
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        "Paper is unsupported or underspecified for strict replication: "
                        + "; ".join(paper_profile.unsupported_reasons or ["no reason recorded"])
                    ),
                )
            )
        for item in deviation_report.items:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity=item.severity,
                    message=f"{item.item}: {item.reason}",
                    detail={
                        "original": item.original,
                        "easyicu_proxy": item.easyicu_proxy,
                    },
                )
            )
        return findings


class ReplicationResultComparator:
    """Compare original-paper claims to EasyICU structured metrics."""

    name = "replication_result_comparator"

    _metric_map = {
        "or": "primary_or",
        "hr": "primary_or",
        "rr": "primary_or",
        "auroc": "auroc",
        "auc": "auroc",
        "brier_score": "brier_score",
        "p_value": "primary_pvalue",
        "p": "primary_pvalue",
        "n": "n_stays",
    }

    def compare(
        self,
        *,
        paper_profile: PaperProfile,
        ledger: PaperResultLedger,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for claim in paper_profile.key_claims:
            easyicu_value = ledger.easyicu_metrics.get(
                self._metric_map.get((claim.metric or "").lower(), "")
            )
            alignment, reason = compare_metric_values(
                metric=claim.metric,
                paper_value=claim.numeric_value,
                paper_direction=claim.direction,
                easyicu_value=easyicu_value,
            )
            rows.append(
                {
                    "claim_id": claim.claim_id,
                    "paper_claim": claim.sentence,
                    "paper_value": claim.paper_value or "",
                    "easyicu_value": "" if easyicu_value is None else str(easyicu_value),
                    "alignment_status": alignment,
                    "reason_if_mismatch": reason,
                    "metric": claim.metric or "",
                }
            )
        return rows

    def findings_from_rows(self, rows: Sequence[Dict[str, Any]]) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        if not rows:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message="No result-alignment rows were produced for the parsed paper claims.",
                )
            ]
        for row in rows:
            if row.get("alignment_status") != "not_aligned":
                continue
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"Claim {row.get('claim_id')} was not aligned with EasyICU results: "
                        f"{row.get('reason_if_mismatch')}"
                    ),
                    detail=dict(row),
                )
            )
        return findings


class PublicationClaimAuditor:
    """Block showcase manuscripts that misrepresent the replication relationship."""

    name = "publication_claim_auditor"

    def audit(
        self,
        *,
        manuscript_text: str,
        deviation_report: ReplicationDeviationReport,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        text = manuscript_text or ""
        lower = text.lower()
        prohibited = (
            "exactly reproduced",
            "identical to the original paper",
            "fully reproduced the original study",
            "same dataset as the original paper",
        )
        for phrase in prohibited:
            if phrase in lower:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=f"Showcase manuscript over-claims replication fidelity via phrase: {phrase!r}.",
                    )
                )
        if "replication" not in lower:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message="Showcase manuscript does not state that it is a replication study.",
                )
            )
        if "easyicu" not in lower:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message="Showcase manuscript does not identify EasyICU as the cohort source.",
                )
            )
        if deviation_report.items and not re.search(r"\bdeviation|differ|limitation|harmoni[sz]ation\b", lower):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message="Showcase manuscript does not explain replication deviations/limitations.",
                )
            )
        if re.search(r"\boriginal paper\b", lower) and "original paper reported" not in lower:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message="References to the original paper should be explicitly framed as reported original results.",
                )
            )
        return findings


__all__ = [
    "CohortAuditor",
    "ConceptUsageAuditor",
    "CrossStepCohortLockValidator",
    "CrossStepRegisteredOutputValidator",
    "CrossStepReconciliationTraceValidator",
    "CrossStepSourceStatusValidator",
    "StepSummaryFractionValidator",
    "FigureContractQualityValidator",
    "LLMConceptAuditor",
    "parse_llm_concept_audit_response",
    "StatisticalValidator",
    "ReplicationDesignAuditor",
    "ReplicationResultComparator",
    "PublicationClaimAuditor",
    "ClinicalConstraintValidator",
    "StatisticalGuard",
    "dedupe_findings",
]


def dedupe_findings(
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Collapse byte-identical findings within the same authority scope.

    The pilot run on 2026-05-15 surfaced the same
    ``concept_usage_auditor`` message recorded 5 times in a single run
    because step-level audits fire across every step that touches the
    flagged column. The output reads like 5 separate problems when it
    is one. This helper keeps the first occurrence (preserves order),
    records the rolled-up count under ``detail['duplicate_count']``,
    and merges ``evidence_ids`` across the collapsed group so no
    reference is lost.

    Findings that already declare a non-empty ``detail`` are still merged when
    their owner scope matches: ``detail.step_id`` participates in the dedupe
    key.  This prevents the same prose emitted by two independent steps from
    being collapsed under the first step's authority and then incorrectly
    retired when only that first step succeeds.  Other detail remains
    shallow-copied and the duplicate count overwrites only the dedicated key.
    """
    seen: Dict[tuple, int] = {}
    out: List[ValidationFinding] = []
    for f in findings:
        owner_step_id = str((f.detail or {}).get("step_id") or "").strip() or None
        key = (f.validator, f.severity, f.message, owner_step_id)
        if key not in seen:
            seen[key] = len(out)
            out.append(f)
            continue
        idx = seen[key]
        existing = out[idx]
        new_detail: Dict[str, Any] = dict(existing.detail or {})
        new_detail["duplicate_count"] = new_detail.get("duplicate_count", 1) + 1
        merged_evidence = list(existing.evidence_ids)
        for eid in f.evidence_ids:
            if eid not in merged_evidence:
                merged_evidence.append(eid)
        out[idx] = existing.model_copy(
            update={"detail": new_detail, "evidence_ids": merged_evidence},
        )
    return out
