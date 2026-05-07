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

from .schema import (
    AggregationRule,
    AnalysisStep,
    ConceptDescriptor,
    EvidenceRecord,
    ResearchContext,
    ValidationFinding,
    VariableRole,
)
from .llm import LLMClient, LLMMessage


# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


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

        return findings


# ---------------------------------------------------------------------------
# ConceptUsageAuditor
# ---------------------------------------------------------------------------


_FORBIDDEN_AGG_PATTERNS_BY_KIND = {
    # role + agg method => human-readable message
    ("ordinal_score", "mean"): "Taking mean() of an ordinal SOFA component is a category error; aggregate by max within window.",
    ("ordinal_score", "std"):  "std() of an ordinal SOFA component is meaningless; report a level distribution instead.",
    ("composite_score", "mean"): "Total SOFA is a sum of 0–4 components; treat as ordinal/integer-count, not continuous (use max-within-window or report distribution).",
    ("composite_score", "std"):  "std() of a composite ordinal score is misleading; prefer median (IQR) or distribution table.",
    ("ordinal_score_gcs", "mean"): "GCS is ordinal; report worst (min) or representative (last/first), not mean.",
}


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
                    validator=self.name, severity="error",
                    message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[key],
                    detail={"column": col, "function": fn, "step_id": step.step_id if step else None},
                ))
                return
            if v.name.lower() == "gcs" and fn == "mean":
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[("ordinal_score_gcs", "mean")],
                    detail={"column": col, "function": fn},
                ))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = _call_function_name(node)
            if func_name is None:
                continue

            referenced_cols = _extract_column_names(node, alias_map)
            if func_name in {"mean", "std"}:
                for col in referenced_cols:
                    _check(col, func_name)
                    if func_name == "mean":
                        mean_columns.add(col)
            elif func_name == "median":
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
            if v.role == VariableRole.LAB and col not in median_columns:
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
                    severity="error",
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
        if isinstance(arg, ast.Constant) and arg.value in {0, 0.0}:
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
        return parse_llm_concept_audit_response(
            raw,
            validator=self.name,
            step_id=step.step_id if step else None,
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
                "role": v.role.value,
                "allowed_aggregations": [a.value for a in v.allowed_aggregations],
                "pitfalls": v.pitfalls,
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
            "confusion, and causal/clinical treatment claims in analysis code.\n\n"
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
        findings.append(ValidationFinding(
            validator=validator,
            severity=sev,  # type: ignore[arg-type]
            message=msg,
            detail=detail or None,
        ))
    return findings


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

        # 2. SOFA-zero anomaly cross-check. Mock code writes
        #    ``sofa_strata.csv``; real LLMs often choose names such as
        #    ``stratum_audit.csv`` with ``mortality_rate`` columns. Accept
        #    both shapes so the deterministic guard stays active.
        sofa_anomaly_seen = False
        for sofa_csv in (
            out_dir / "sofa_strata.csv",
            out_dir / "stratum_audit.csv",
            out_dir / "sofa2_mortality.csv",
        ):
            if not sofa_csv.exists():
                continue
            try:
                strata = pd.read_csv(sofa_csv)
                rate_col = next(
                    (
                        c for c in (
                            "outcome_rate",
                            "mortality_rate",
                            "death_rate",
                            "icu_mortality_rate",
                        )
                        if c in strata.columns
                    ),
                    None,
                )
                if rate_col is None:
                    continue
                excluded = {
                    rate_col,
                    "n", "count", "total", "n_total", "n_death",
                    "death", "deaths", "sum",
                    "outcome_ci_low", "outcome_ci_high",
                    "mortality_ci_low", "mortality_ci_high",
                    "mortality_rate_ci_low", "mortality_rate_ci_high",
                }
                score_cols = [
                    c for c in strata.columns
                    if c not in excluded
                    and pd.api.types.is_numeric_dtype(strata[c])
                ]
                score_cols.sort(key=lambda c: (0 if "sofa" in c.lower() else 1, c))
                if not score_cols:
                    continue
                sc = score_cols[0]
                scores = pd.to_numeric(strata[sc], errors="coerce")
                rates = pd.to_numeric(strata[rate_col], errors="coerce")
                r0_values = rates.loc[scores == 0].dropna()
                r1_values = rates.loc[scores == 1].dropna()
                if r0_values.empty or r1_values.empty:
                    continue
                r0 = float(r0_values.iloc[0])
                r1 = float(r1_values.iloc[0])
                if r0 > r1:
                    findings.append(ValidationFinding(
                        validator=self.name, severity="warning",
                        message=(
                            f"{sc}==0 outcome rate ({r0:.3f}) exceeds {sc}==1 "
                            f"({r1:.3f}). This is non-monotonic and is a known "
                            "signature of component-level missingness rather than "
                            "absent organ dysfunction. Verify component "
                            "availability before interpreting clinically."
                        ),
                        detail={
                            "score": sc,
                            "rate_at_zero": r0,
                            "rate_at_one": r1,
                            "source_file": sofa_csv.name,
                        },
                    ))
                    sofa_anomaly_seen = True
                    break
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Could not parse {sofa_csv.name}: {exc}",
                ))

        if not sofa_anomaly_seen:
            try:
                zero_one_pairs = (
                    ("sofa2_zero_rate", "sofa2_one_rate"),
                    ("mortality_sofa0", "mortality_sofa1"),
                    ("sofa0_mortality_rate", "sofa1_mortality_rate"),
                )
                for zero_key, one_key in zero_one_pairs:
                    if zero_key not in step_summary or one_key not in step_summary:
                        continue
                    r0 = float(step_summary[zero_key])
                    r1 = float(step_summary[one_key])
                    if r0 > r1:
                        findings.append(ValidationFinding(
                            validator=self.name, severity="warning",
                            message=(
                                f"SOFA score==0 outcome rate ({r0:.3f}) exceeds "
                                f"score==1 ({r1:.3f}). This is non-monotonic and "
                                "is a known signature of component-level missingness."
                            ),
                            detail={
                                "score": "sofa",
                                "rate_at_zero": r0,
                                "rate_at_one": r1,
                                "source": "step_summary",
                            },
                        ))
                        sofa_anomaly_seen = True
                        break
            except (TypeError, ValueError):
                pass

        if not sofa_anomaly_seen and outcome:
            step_text = f"{step.step_id} {step.intent}".lower()
            if (
                "stratum" in step_text
                or "score==0" in step_text
                or "sofa_zero" in step_text
            ):
                try:
                    df = pd.read_parquet(cohort_path)
                    score_candidates = [
                        v.name for v in context.variables
                        if "sofa" in v.name.lower()
                    ]
                    for sc in score_candidates:
                        if sc not in df.columns or outcome not in df.columns:
                            continue
                        sub = df[[sc, outcome]].dropna().copy()
                        if sub.empty:
                            continue
                        sub[sc] = pd.to_numeric(sub[sc], errors="coerce")
                        sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
                        sub = sub.dropna(subset=[sc, outcome])
                        grouped = sub.groupby(sc)[outcome].mean()
                        if 0 not in grouped.index or 1 not in grouped.index:
                            continue
                        r0 = float(grouped.loc[0])
                        r1 = float(grouped.loc[1])
                        if r0 > r1:
                            findings.append(ValidationFinding(
                                validator=self.name, severity="warning",
                                message=(
                                    f"{sc}==0 outcome rate ({r0:.3f}) exceeds "
                                    f"{sc}==1 ({r1:.3f}). This is non-monotonic "
                                    "and was recomputed directly from the cohort "
                                    "because the stratum audit artefact omitted "
                                    "mortality rates."
                                ),
                                detail={
                                    "score": sc,
                                    "rate_at_zero": r0,
                                    "rate_at_one": r1,
                                    "source": "cohort_recompute",
                                },
                            ))
                            sofa_anomaly_seen = True
                            break
                except Exception as exc:
                    findings.append(ValidationFinding(
                        validator=self.name, severity="warning",
                        message=f"Could not recompute SOFA-zero anomaly: {exc}",
                    ))

        # 3. Primary-association OR cross-check (T1.6).
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

        # 4. Sanity: the script must have produced some artefact.
        if not any(out_dir.iterdir()):
            findings.append(ValidationFinding(
                validator=self.name, severity="error",
                message=f"Step '{step.step_id}' produced no output artefacts.",
            ))

        # 5. Codex-grade train/test performance metrics (T1.8). Whenever a
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

        # 6. T1.8 score-0 elevated-mortality data-quality signal. When
        #    a step's summary reports a sofa2_zero_rate that exceeds the
        #    overall mortality_rate, surface this as a *data_quality_signal*
        #    finding rather than an error — codex's central scientific
        #    claim was exactly this kind of finding.
        try:
            zero_rate = step_summary.get("sofa2_zero_rate")
            overall = step_summary.get("mortality_rate")
            if zero_rate is not None and overall is not None:
                z, o = float(zero_rate), float(overall)
                if z == z and o == o and z > o + 1e-6:
                    findings.append(ValidationFinding(
                        validator=self.name, severity="warning",
                        message=(
                            "Data-quality signal: SOFA-2 score==0 stratum "
                            f"mortality {z:.3f} exceeds overall mortality "
                            f"{o:.3f}. Consistent with a component-availability "
                            "artefact in upstream concept construction."
                        ),
                        detail={
                            "sofa2_zero_rate": z, "overall_rate": o,
                            "category": "data_quality_signal",
                        },
                    ))
        except (TypeError, ValueError):
            pass

        return findings


__all__ = [
    "CohortAuditor",
    "ConceptUsageAuditor",
    "LLMConceptAuditor",
    "parse_llm_concept_audit_response",
    "StatisticalValidator",
]
