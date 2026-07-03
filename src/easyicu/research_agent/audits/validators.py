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
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from ..replication.paper import compare_metric_values
from ..schema import (
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
        return _downgrade_metadata_supported_outcome_findings(
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
                "missingness": (
                    v.missingness.model_dump(mode="json")
                    if v.missingness is not None else None
                ),
            }
            for v in context.variables[:80]
        ]
        return (
            "Review this generated analysis script for ICU concept-use risks "
            "that deterministic regex checks may miss. Focus only on: ordinal "
            "scores treated as continuous, silent missingness assumptions, "
            "PaO2/FiO2 or GCS/SOFA/KDIGO misuse, ICU vs hospital mortality "
            "confusion, and causal/clinical treatment claims in analysis code. "
            "If a generic outcome column such as 'death' is explicitly bound in "
            "the variable metadata to ICU mortality, hospital mortality or a "
            "fixed follow-up horizon, do not raise an error unless the script "
            "contradicts that binding or mixes incompatible outcome definitions.\n\n"
            "Return JSON only: "
            '{"findings":[{"severity":"info|warning|error",'
            '"message":"short finding","detail":{"optional":"context"}}]}. '
            "Use an empty findings list if no issue is visible.\n\n"
            f"Step: {step.step_id if step else '(unknown)'}\n"
            f"Step intent: {step.intent if step else '(unknown)'}\n"
            f"Target outcome: {context.target_outcome}\n"
            "Variables:\n"
            + json.dumps(variables, ensure_ascii=False, default=str)
            + "\n\nScript:\n"
            + script_text[:12000]
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
    source = descriptor.source_concept.lower()
    if source not in {
        "icu_mortality",
        "hospital_mortality",
        "mortality_28d",
        "mortality_30d",
    }:
        return list(findings)

    code = (script_text or "").lower()
    contradictory_tokens_by_source = {
        "icu_mortality": (
            "hospital_death",
            "death_hosp",
            "hospital_mortality",
            "hospital mortality",
            "in-hospital mortality",
            "28-day mortality",
            "30-day mortality",
        ),
        "hospital_mortality": (
            "death_icu",
            "icu_mortality",
            "icu mortality",
            "28-day mortality",
            "30-day mortality",
        ),
        "mortality_28d": (
            "death_icu",
            "icu_mortality",
            "icu mortality",
            "hospital_death",
            "hospital_mortality",
            "hospital mortality",
            "30-day mortality",
        ),
        "mortality_30d": (
            "death_icu",
            "icu_mortality",
            "icu mortality",
            "hospital_death",
            "hospital_mortality",
            "hospital mortality",
            "28-day mortality",
        ),
    }
    if any(token in code for token in contradictory_tokens_by_source[source]):
        return list(findings)

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
            if any(token in text for token in ambiguity_tokens):
                detail = dict(finding.detail or {})
                detail.setdefault(
                    "downgraded_reason",
                    (
                        f"Target outcome '{outcome}' is bound to "
                        f"{descriptor.source_concept} in ResearchContext and "
                        "the script does not reference a conflicting mortality definition."
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
        "variable",
        "term",
        "exposure",
        "contrast",
    )
    _COMPOSITE_KEY_COLUMNS = (
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
        "estimate",
        "ci_low",
        "ci_high",
        "se",
        "odds_ratio",
        "risk_ratio",
        "risk_difference",
        "p_value",
    )
    _TEXT_COLUMNS = ("effect_scale",)
    _PCT_COUNT_RULES = (
        ("missing_pct", "missing_n", "total_n"),
        ("measured_pct", "measured_n", "total_n"),
        ("measured_pct", "n_nonmissing", "total_n"),
    )

    def audit(
        self,
        *,
        step: AnalysisStep,
        out_dir: Path,
        run_dir: Path,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        if not self._is_rendering_step(step=step, step_summary=step_summary):
            return []
        source_tables = sorted(out_dir.glob(self._SOURCE_DATA_GLOB))
        if not source_tables:
            return []

        upstream_step_ids = self._upstream_step_ids(step=step, step_summary=step_summary)
        if not upstream_step_ids:
            return []
        upstream_tables = self._upstream_tables(
            run_dir=run_dir,
            current_out_dir=out_dir,
            upstream_step_ids=upstream_step_ids,
        )
        if not upstream_tables:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"Figure step '{step.step_id}' wrote source data, but no "
                        "candidate upstream source table was found for "
                        f"{sorted(upstream_step_ids)}."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "upstream_step_ids": sorted(upstream_step_ids),
                        "source_tables": [p.name for p in source_tables],
                    },
                )
            ]

        findings: List[ValidationFinding] = []
        for source_path in source_tables:
            try:
                source_df = pd.read_csv(source_path)
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=f"Could not read figure source-data table {source_path.name}: {exc}",
                        detail={"source_table": source_path.name},
                    )
                )
                continue
            if source_df.empty:
                continue
            findings.extend(
                self._percentage_count_consistency_findings(
                    source_df=source_df,
                    source_path=source_path,
                    step_id=step.step_id,
                )
            )
            ordered_upstream_tables = self._prioritize_declared_source_tables(
                source_df=source_df,
                upstream_tables=upstream_tables,
            )
            comparisons = [
                self._compare_source_to_upstream(
                    source_df=source_df,
                    source_path=source_path,
                    upstream_path=upstream_path,
                )
                for upstream_path in ordered_upstream_tables
            ]
            if any(item.get("ok") for item in comparisons):
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
    def _is_rendering_step(
        cls, *, step: AnalysisStep, step_summary: Dict[str, Any]
    ) -> bool:
        if bool((step_summary or {}).get("rendering_only")):
            return True
        haystack = f"{step.step_id} {step.method} {step.intent}".lower()
        return "figure" in haystack or "render" in haystack

    @classmethod
    def _upstream_step_ids(
        cls, *, step: AnalysisStep, step_summary: Dict[str, Any]
    ) -> Set[str]:
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
    def _upstream_tables(
        cls,
        *,
        run_dir: Path,
        current_out_dir: Path,
        upstream_step_ids: Set[str],
    ) -> List[Path]:
        tables: List[Path] = []
        for step_id in sorted(upstream_step_ids):
            outputs = run_dir / "steps" / step_id / "outputs"
            if not outputs.exists():
                continue
            for path in sorted(outputs.iterdir()):
                if not path.is_file():
                    continue
                if path.parent.resolve() == current_out_dir.resolve():
                    continue
                if path.suffix.lower() == ".csv":
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
    def _compare_source_to_upstream(
        cls,
        *,
        source_df: pd.DataFrame,
        source_path: Path,
        upstream_path: Path,
    ) -> Dict[str, Any]:
        try:
            upstream_df = pd.read_csv(upstream_path)
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
        if key_cols is None and "source_row_index" in source.columns:
            row_index = pd.to_numeric(source["source_row_index"], errors="coerce")
            invalid = row_index.isna() | (row_index < 0) | (row_index >= len(upstream))
            invalid = invalid | (row_index % 1 != 0)
            if invalid.any():
                first_bad = int(invalid[invalid].index[0])
                return {
                    "ok": False,
                    "reason": "source_row_index_out_of_bounds",
                    "upstream_table": upstream_path.name,
                    "message": (
                        "source_row_index values must be integer row positions "
                        f"within {upstream_path.name}; first invalid row is {first_bad}"
                    ),
                }
            source["_source_row_index"] = row_index.astype(int).astype(str)
            upstream = upstream.reset_index().rename(columns={"index": "_source_row_index"})
            upstream["_source_row_index"] = upstream["_source_row_index"].astype(str)
            key_cols = ("_source_row_index",)
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
        key_label = "+".join(key_cols)

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
        numeric_columns = list(cls._NUMERIC_COLUMNS)
        ignored_for_dynamic_numeric = {
            *key_cols,
            "source_row_index",
            "source_table",
        }
        for col in sorted(set(source.columns) & set(upstream.columns)):
            if col in ignored_for_dynamic_numeric or col in numeric_columns:
                continue
            left_raw = source[col]
            right_raw = upstream[col]
            left_present = left_raw.notna() & left_raw.astype(str).str.strip().ne("")
            right_present = right_raw.notna() & right_raw.astype(str).str.strip().ne("")
            if not left_present.any() or not right_present.any():
                continue
            left_num = pd.to_numeric(left_raw[left_present], errors="coerce")
            right_num = pd.to_numeric(right_raw[right_present], errors="coerce")
            if left_num.notna().all() and right_num.notna().all():
                numeric_columns.append(col)

        for col in numeric_columns:
            source_col = f"{col}_source"
            upstream_col = f"{col}_upstream"
            if source_col not in merged.columns or upstream_col not in merged.columns:
                continue
            left = pd.to_numeric(merged[source_col], errors="coerce").astype(float)
            right = pd.to_numeric(merged[upstream_col], errors="coerce").astype(float)
            diff = (left - right).abs()
            bad = diff[(diff > 1e-9) & ~(left.isna() & right.isna())]
            if not bad.empty:
                idx = int(bad.index[0])
                mismatches.append(
                    {
                        "column": col,
                        "key": _format_key(merged.loc[idx]),
                        "source": None if pd.isna(left.loc[idx]) else float(left.loc[idx]),
                        "upstream": None if pd.isna(right.loc[idx]) else float(right.loc[idx]),
                        "abs_diff": float(bad.iloc[0]),
                    }
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
                "key_column": key,
                "upstream_table": upstream_path.name,
                "mismatches": mismatches[:20],
                "n_mismatches": len(mismatches),
                "message": f"source-data values disagree with {upstream_path.name}",
            }
        return {
            "ok": True,
            "reason": "source_subset_matches",
            "source_table": source_path.name,
            "upstream_table": upstream_path.name,
            "key_column": key_label,
            "n_source_rows": int(len(source_df)),
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
        if result_like and len(panels_list) < 2:
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
        role_text = " ".join(
            str(panel.get("role") or "")
            for panel in panels
            if isinstance(panel, dict)
        ).lower()
        text_blob = cls._contract_text(raw).lower()
        return any(role in role_text or role in text_blob for role in cls._RESULT_ROLES)

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
        prediction_like = family == "prediction_model" or any(
            term in combined
            for term in (
                "prediction workflow",
                "prediction model",
                "mortality prediction",
                "auroc",
                "brier",
                "calibration",
            )
        )
        treatment_like = any(
            term in combined
            for term in (
                "target trial",
                "treatment",
                "treated",
                "untreated",
                "intervention",
                "therapy",
                "drug",
                "dose",
                "assignment",
                "vasopressor",
                "ventilation",
            )
        )
        causal_exposure_language = "exposure" in combined and any(
            term in combined
            for term in (
                "causal",
                "effect of",
                "treatment effect",
                "intervention effect",
                "target trial",
            )
        )

        if (
            not prediction_like
            and (
                family in {"causal_inference", "treatment_response", "reinforcement_learning"}
                or treatment_like
                or causal_exposure_language
            )
        ):
            if not any(term in combined for term in ("time zero", "time-zero", "eligibility", "anchor", "alignment")):
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "Treatment-effect style analysis without an explicit time-zero or alignment description "
                        "risks immortal time bias. Document eligibility, anchor time, and treatment assignment timing."
                    ),
                    detail={"analysis_family": family or "unspecified"},
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
            if pval_cols and len(tab) > 1 and not adjust_cols:
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"Table '{csv_path.name}' contains multiple p-values without an adjusted-p / q-value column. "
                        "If this is a family of simultaneous tests, control multiplicity explicitly."
                    ),
                    detail={"table": csv_path.name, "p_value_columns": pval_cols},
                ))
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
    """Collapse byte-identical ``(validator, severity, message)`` findings.

    The pilot run on 2026-05-15 surfaced the same
    ``concept_usage_auditor`` message recorded 5 times in a single run
    because step-level audits fire across every step that touches the
    flagged column. The output reads like 5 separate problems when it
    is one. This helper keeps the first occurrence (preserves order),
    records the rolled-up count under ``detail['duplicate_count']``,
    and merges ``evidence_ids`` across the collapsed group so no
    reference is lost.

    Findings that already declare a non-empty ``detail`` are still
    merged: their detail is shallow-copied and the duplicate count
    overwrites only the dedicated key.
    """
    seen: Dict[tuple, int] = {}
    out: List[ValidationFinding] = []
    for f in findings:
        key = (f.validator, f.severity, f.message)
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
