"""Audit validators — concept-audit owner plus the historical import surface.

ConceptUsageAuditor, LLMConceptAuditor, and their helpers remain here so
the ``validators.authorized_complete`` module-global seam tests patch keeps
working. Every other validator now lives in a sibling owner module
(``_v_support``/``cohort``/``cross_step``/``statistical``/
``figure_source``/``figure_contract``/``clinical``/``publication``) and is
re-exported below. New code should import from the owner modules.
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

from ..planning.analysis_method_suite import figure_product_source_obligations
from ..contracts.declared_product import (
    effect_adjustment_family,
    effect_bearing_product,
    effect_estimand_tier,
    effect_measure_family,
    effect_role_family,
    typed_product,
)
from ..contracts.fraction_scale import (
    is_scale_descriptor_field,
    normalize_metric_key,
)
from ..contracts.model_tokens import canonical_association_method
from ..contracts.model_contract_match import reported_model_requirement_fields
from ..replication.metrics import compare_metric_values
from ..contracts.ordered_stratified import ordered_stratified_numeric_findings
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
from ..providers.protocol import LLMClient, LLMMessage
from ..providers.factory import authorized_complete
from ..research_context.builder import _descriptor_endpoint_semantic_key
from ..research_context.outbound import (
    format_outbound_safe_context,
    outbound_safe_script,
)
from ..authority.provider_budget import (
    ProviderCallBudgetError,
    StepProviderCallBudget,
    complete_with_provider_budget,
)
from .outcome_semantics import (
    _finding_claims_mortality_horizon_mismatch,
    _script_copies_named_full_stay_window,
    _script_has_conflicting_mortality_semantics,
    _script_uses_bound_outcome,
)
from ..authority.runtime_artifacts import (
    current_run_evidence_records,
    current_successful_step_records,
    verified_run_evidence_path,
)
from ..trajectory.contract import trajectory_phenotyping_artifact_findings

# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and

_FORBIDDEN_AGG_PATTERNS_BY_KIND = {
    # (role, agg method) => human-readable message.
    # Messages are phrased as conservative reporting-practice violations,
    # not as absolute mathematical errors: for bounded ordinal clinical
    # scores, median/IQR or level-distribution summaries are preferred
    # over mean/SD for manuscript-facing reporting. The same column may
    # legitimately enter a regression or Cox model as a linear covariate;
    # this auditor covers reporting/aggregation misuse only, not model
    # specification choices.
    (
        "ordinal_score",
        "mean",
    ): "Mean of an ordinal SOFA component may be misleading; for manuscript-facing summaries prefer max-within-window or a level distribution.",
    (
        "ordinal_score",
        "std",
    ): "Standard deviation of an ordinal SOFA component is rarely interpretable; prefer a level distribution.",
    (
        "composite_score",
        "mean",
    ): "Mean of a composite ordinal score (total SOFA = sum of 0–4 components) is a reporting-practice violation for bounded integer clinical scores; for manuscript-facing summaries prefer max-within-window, median (IQR) or a level distribution.",
    (
        "composite_score",
        "std",
    ): "Standard deviation of a composite ordinal score may be misleading; prefer median (IQR) or a level distribution.",
    (
        "ordinal_score_gcs",
        "mean",
    ): "GCS is ordinal; for manuscript-facing summaries prefer worst (min) or a representative (last / first) value rather than mean.",
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
    "probe",
    "descriptive",
    "exploratory",
    "qc",
    "summary",
    "missingness_audit",
    "score_qc",
)
_BLOCKING_STAGE_TOKENS = (
    "primary_",
    "manuscript",
    "final_report",
    "publication",
    "evidence_binding",
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
                cols = (
                    _extract_column_names(node.value, alias_map)
                    if node.value
                    else set()
                )
                if cols:
                    alias_map[node.target.id] = set(cols)

        def _check(col: str, fn: str) -> None:
            v = var_by_name.get(col)
            if v is None:
                return
            role_key = v.role.value
            key = (role_key, fn)
            if key in _FORBIDDEN_AGG_PATTERNS_BY_KIND:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity=_forbidden_agg_severity(step),
                        message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[key],
                        detail={
                            "column": col,
                            "function": fn,
                            "step_id": step.step_id if step else None,
                        },
                    )
                )
                return
            if v.name.lower() == "gcs" and fn == "mean":
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity=_forbidden_agg_severity(step),
                        message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[
                            ("ordinal_score_gcs", "mean")
                        ],
                        detail={"column": col, "function": fn},
                    )
                )

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
                    if (
                        ".mean(" in expr
                        or '.agg("mean")' in expr
                        or ".agg('mean')" in expr
                    ):
                        findings.append(
                            ValidationFinding(
                                validator=self.name,
                                severity="warning",
                                message=(
                                    "Detected DataFrame.eval() expression containing mean-style "
                                    "aggregation. Review this script manually because string-eval "
                                    "can bypass column-level ICU aggregation checks."
                                ),
                                detail={"expression": expr[:200]},
                            )
                        )

        for col in sorted(mean_columns):
            v = var_by_name.get(col)
            if v is None:
                continue
            if (
                v.role == VariableRole.LAB
                and col not in median_columns
                and not (mean_receivers & median_receivers)
            ):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="warning",
                        message=(
                            f"Lab variable '{col}' summarised by mean() with no median() in "
                            "the same script. Right-skewed labs are conventionally reported "
                            "as median (IQR)."
                        ),
                        detail={"column": col, "function": "mean"},
                    )
                )

        if fillna_zero_columns:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "Detected fillna(0) — silent imputation to zero is rarely correct for "
                        "ICU variables. Use a missing-indicator or document the imputation explicitly."
                    ),
                    detail={"columns": sorted(fillna_zero_columns)},
                )
            )
        return findings

    def _regex_fallback(
        self,
        *,
        var_by_name: Dict[str, ConceptDescriptor],
        script_text: str,
        step: Optional[AnalysisStep] = None,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        pat_bracket = re.compile(
            r"""\[(['"])(?P<col>[^'"]+)\1\]\s*\.\s*(?P<fn>mean|std)\s*\("""
        )
        pat_attr = re.compile(
            r"""\.(?P<col>[a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*(?P<fn>mean|std)\s*\("""
        )
        for match in list(pat_bracket.finditer(script_text)) + list(
            pat_attr.finditer(script_text)
        ):
            col = match.group("col")
            fn = match.group("fn")
            var = var_by_name.get(col)
            if var is None:
                continue
            key = (var.role.value, fn)
            if key in _FORBIDDEN_AGG_PATTERNS_BY_KIND:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity=_forbidden_agg_severity(step),
                        message=_FORBIDDEN_AGG_PATTERNS_BY_KIND[key],
                        detail={"column": col, "function": fn, "fallback": "regex"},
                    )
                )
        if re.search(r"\.fillna\s*\(\s*0\s*\)", script_text):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        "Detected fillna(0) — silent imputation to zero is rarely correct for "
                        "ICU variables. Use a missing-indicator or document the imputation explicitly."
                    ),
                    detail={"fallback": "regex"},
                )
            )
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
            if base in {"df", "data", "cohort", "frame", "table"} or base.endswith(
                "df"
            ):
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
        return _extract_column_names(node.left, alias_map) | _extract_column_names(
            node.right, alias_map
        )
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


def _concept_audit_step_roster_block(
    plan_step_roster: Optional[Sequence[Mapping[str, Any]]],
) -> str:
    """The plan's other steps, so "nobody did this" can be distinguished.

    Deliberately id/role/method only. The other steps' rule prose would be the
    largest block in this prompt and would invite auditing them; what the
    auditor needs is not their content but the fact that they exist and own
    work, so a requirement discharged elsewhere stops looking undischarged.
    """

    if not plan_step_roster:
        return ""
    lines = []
    for entry in plan_step_roster:
        step_id = str(entry.get("step_id") or "").strip()
        if not step_id:
            continue
        role = str(entry.get("planned_analysis_role") or "").strip()
        method = str(entry.get("method") or "").strip()
        lines.append(f"- {step_id} [{role or 'unspecified'}] {method}".rstrip())
    if not lines:
        return ""
    return (
        "Other steps in this locked plan (id, planned role, method):\n"
        + "\n".join(lines)
        + "\nYou are auditing ONLY the step named above. A requirement the plan "
        "assigns to a different step in this roster is that step's obligation, "
        "not a defect in this script: report it only if THIS step's own declared "
        "contract carries it. In particular a cohort, eligibility, exclusion or "
        "attrition rule is discharged by the step that owns the cohort contract, "
        "and by the time this step runs that step has already run and its output "
        "is an input here. Do not require this script to re-apply it.\n"
    )


def _concept_audit_endpoint_block(
    study_endpoint: Optional[Mapping[str, Any]],
) -> str:
    """The declared endpoint, or an explicit statement that none was declared.

    Says which case it is either way. Rendering nothing when nothing is declared
    would let the auditor keep supplying the missing rule from the research
    question -- silently, and differently each run, which is how one task got
    blocked twice for opposite censoring choices. An absent declaration is a
    fact about the plan and is reported as one.
    """

    if not study_endpoint:
        return (
            "Declared study endpoint: NONE. The plan declared no typed endpoint, "
            "so no follow-up time, time origin, censoring rule or level set is "
            "authoritative for this run. Do not supply one from the research "
            "question and do not report the script for contradicting a rule that "
            "is not declared anywhere; a missing declaration is the plan's "
            "defect, not this script's.\n"
        )
    payload = {
        key: value
        for key, value in study_endpoint.items()
        if key not in {"authorization", "schema_version"}
    }
    return (
        "Declared study endpoint (from the locked plan -- AUTHORITATIVE):\n"
        + json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
        + "\nThis declaration, not the research question and not your own reading "
        "of it, defines the endpoint, the follow-up clock, its origin, what "
        "censors follow-up, and the closed level set. A script implementing "
        "exactly these fields is correct even if you would have designed the "
        "study differently; report a mismatch only against a field printed "
        "above, and name that field.\n"
    )


class LLMConceptAuditor:
    """Optional LLM-based semantic review after deterministic checks.

    Static rules remain authoritative and run first. This auditor is a
    conservative final sweep for issues that are hard to encode as
    regexes, such as confusing ICU vs hospital mortality or describing
    a missingness-driven stratum as clinically low risk.
    """

    name = "llm_concept_auditor"

    def __init__(self, llm: LLMClient, *, max_tokens: int = 2_048) -> None:
        self.llm = llm
        self.max_tokens = int(max_tokens)

    def audit(
        self,
        *,
        context: ResearchContext,
        script_text: str,
        step: Optional[AnalysisStep] = None,
        provider_budget: Optional[StepProviderCallBudget] = None,
        study_endpoint: Optional[Mapping[str, Any]] = None,
        plan_step_roster: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[ValidationFinding]:
        prompt = self._prompt(
            context=context,
            script_text=script_text,
            step=step,
            study_endpoint=study_endpoint,
            plan_step_roster=plan_step_roster,
        )
        try:
            raw = complete_with_provider_budget(
                budget=provider_budget,
                category="concept_audit",
                call=lambda: authorized_complete(
                    self.llm,
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
                ),
            )
        except ProviderCallBudgetError:
            raise
        except Exception as exc:
            detail: Dict[str, Any] = {
                "issue_code": "llm_concept_audit_provider_failure",
                "error_type": type(exc).__name__,
            }
            if step is not None:
                detail["step_id"] = step.step_id
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        "LLM concept auditor did not return a review because "
                        f"its provider call failed: {str(exc)[:300]}"
                    ),
                    detail=detail,
                )
            ]
        findings = parse_llm_concept_audit_response(
            raw,
            validator=self.name,
            step_id=step.step_id if step else None,
        )
        if any(
            str((finding.detail or {}).get("issue_code") or "")
            == "llm_concept_audit_response_invalid"
            for finding in findings
        ):
            return findings
        return _reclassify_llm_concept_findings(
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
        study_endpoint: Optional[Mapping[str, Any]] = None,
        plan_step_roster: Optional[Sequence[Mapping[str, Any]]] = None,
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

        safe_context = format_outbound_safe_context(
            context,
            variable_names={variable.name for variable in selected_variables},
        )
        safe_script = outbound_safe_script(step, script_text)
        step_contract = (
            {
                key: value
                for key, value in step.model_dump(mode="json").items()
                if key
                in {
                    "step_id",
                    "planned_analysis_role",
                    "method",
                    "inputs",
                    "expected_outputs",
                    "icu_rule_refs",
                    "model_requirements",
                    "input_consumption_contracts",
                    "table_one_spec",
                    "trajectory_stability_spec",
                }
            }
            if step is not None
            else {}
        )
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
            "A direct, uncaught call imported exactly as "
            "`measurement_provenance_receipt` from "
            "`easyicu.research_agent.methods.descriptive_inputs` is already a "
            "host-owned fail-closed boundary: it raises on missing, invalid, or "
            "discordant measured/count pairs. Do not demand a second status "
            "guard or inspection of its successful receipt. "
            # The host's own system prompt (rule 1) tells the agent that a total
            # score MAY be modelled as a numeric covariate provided the choice is
            # stated explicitly. Without the sentence below this auditor refused
            # that exact compliance: a real run declared the coding on its own
            # balance-table rows and was still blocked for "ordinal scores
            # treated as continuous", so the two host layers published opposite
            # contracts and the step died on the one the agent obeyed.
            "The system rule permits an ordinal score to be modelled as a "
            "numeric covariate when that choice is stated explicitly. So a "
            "rank-preserving numeric representation used as an ADJUSTMENT "
            "covariate, and labelled as such by the script's own output (for "
            "example a scale or coding field carried on the covariate's row), "
            "has met that requirement: do not report it as an "
            "ordinal-treated-as-continuous defect on that basis alone. Report "
            "instead an ordinal entered numerically with no such declaration "
            "anywhere in the script, an ordinal serving as the primary exposure "
            "or estimand without a declared coding, or an ordinal averaged "
            "rather than summarised rank-preservingly. "
            "A value returned by a direct call imported exactly as "
            "`strict_numeric_input` from that same host module has already "
            "failed closed on every non-missing value that is unconvertible, "
            "semantically nonnumeric, or non-finite. When a result-bearing "
            "summary consumes only that returned `.values` Series (including "
            "through a local wrapper), do not demand a second `isfinite` guard "
            "or infer that later JSON null handling can hide non-finite input. "
            "If you believe a named result variable bypasses this exact host "
            "boundary, use issue_code "
            "`strict_numeric_nonfinite_guard_required` and include every "
            "affected name in `detail.variables`. "
            # The sibling of the two boundaries above, for the event/time pair.
            # Added with the boundary itself: a host helper the auditor does not
            # recognise gets a "add a second guard" finding on the step that
            # correctly called it, which is how a compliant script is blocked.
            "A direct, uncaught call imported exactly as "
            "`event_time_reconciliation_receipt` from "
            "`easyicu.research_agent.methods.survival_inputs` is a host-owned "
            "fail-closed boundary: it raises when an event row's time cannot "
            "place it on the follow-up axis and when an event code falls outside "
            "the declared closed set. Do not demand a second reconciliation of "
            "the same pair, and do not report a censored row carrying no event "
            "time as a defect -- that is the expected shape, and excluding on it "
            "removes non-events from the denominator. "
            "A Step input whose exact ConceptDescriptor names a source_concept, "
            "analysis_window, and aggregation-compatible materialized column is "
            "a host-owned binding. Direct use of that exact input is therefore "
            "the authoritative planned summary; do not require generated code "
            "to re-prove the host metadata or flag it merely because its column "
            "name contains first/max/min/mean. Flag a substitution only when the "
            "script selects a different column or contradicts that binding. "
            "A ConceptDescriptor `valid_range` is host-owned physiological "
            "plausibility metadata, not a locked eligibility or exclusion rule. "
            "Its typed `range_policy` is therefore `flag_only` with "
            "`retain_and_flag`: retaining a finite continuous value outside that "
            "range is not an ERROR unless a separate typed protocol contract "
            "explicitly locks exclusion. Do still report non-finite values, "
            "violations of a binary domain, and invalid ordinal levels as ERRORs. "
            "If you request exclusion, invalidation, or fail-close solely because "
            "a finite continuous value is outside this plausibility range, use "
            "issue_code `plausibility_range_exclusion_required` and include "
            "`detail.variable`, `detail.requested_action`, and "
            "`detail.value_class='finite_outside_plausibility_range'`. "
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
            "Do not assume that a generically named `n` field is the total "
            "denominator when the typed upstream product does not declare that "
            "meaning. If non-negative integer `n_nonmissing` and `missing_n` "
            "components are present, `n_nonmissing / (n_nonmissing + missing_n)` "
            "reconciles availability by construction; require a positive finite "
            "component sum, but do not demand that it equal an otherwise "
            "undefined `n`. "
            "Read an instruction to display standardized feature patterns "
            "together with an instruction not to recompute scaling as follows: "
            "the script must not refit or replace upstream analytical "
            "preprocessing, clustering, profiles, or model inputs, but it may "
            "apply and label a local rendering-only normalization to values read "
            "from the completed parent product. Do not call that display "
            "transform a recomputation of upstream scientific scaling when raw "
            "parent values remain the bound source-data authority and the "
            "transformed values are not reused as a scientific artifact. Still "
            "flag an undeclared transform, mutation of the authoritative parent "
            "table, refitting of upstream analysis, or downstream reuse of the "
            "rendering transform. "
            "For clustering stability steps, distinguish an estimator's fit seed "
            "(for example KMeans.random_state used to compare initializations) "
            "from the silhouette sampling seed. A deterministic helper that "
            "first materializes a bounded sample with "
            "np.random.default_rng(seed) and then calls silhouette_score on the "
            "sampled arrays is compliant even when estimator fit seeds vary; do "
            "not report a silhouette-seed mismatch merely because a KMeans "
            "initialization seed appears nearby. When the bound feature-matrix "
            "product declares imputation/scaling upstream, direct use of that "
            "artifact is the authoritative analytical representation: do not "
            "demand a second scaling pass. If a profile step's contract binds "
            "raw per-stay summaries as its reporting source, using that raw "
            "source for descriptive profiles while using the transformed matrix "
            "only for row/identity reconciliation is correct; flag only the "
            "opposite direction (reporting transformed coordinates as raw clinical "
            "summaries) or an undeclared transformation. A clustering stability "
            "script that records an explicit threshold-based decision rule has "
            "discharged that requirement; do not claim that the rule is absent "
            "when its thresholds and Boolean decision are present in diagnostics "
            "or the output table. "
            "If a generic outcome column such as 'death' is explicitly bound in "
            "the variable metadata to ICU mortality, hospital mortality or a "
            "fixed follow-up horizon, do not raise an error unless the script "
            "contradicts that binding or mixes incompatible outcome definitions.\n\n"
            "The Planner-declared step contract below is binding scientific "
            "authority. Audit whether the code implements that contract; do not "
            "redesign its cohort, exposure, operator, threshold, input set, model "
            "roster, or estimand merely because another defensible design exists. "
            "Do not add a positive-only filter when the locked analysis needs both "
            "exposure levels, and interpret comparison operators literally: for a "
            "non-negative binary field, `>= 0` retains both zero and one. A "
            "host-bound derived ConceptDescriptor may already incorporate its "
            "`derived_from_concepts`; do not require the generated script to load "
            "or filter on each raw component unless the Planner contract declares "
            "that component separately. Conversely, report an ERROR when code "
            "contradicts an explicit ICU rule or required model contract. "
            "Measurement/count/source-status companions are audit-only metadata, "
            "not automatic adjustment covariates. Flag them when they enter a model "
            "design without Planner authority, including a measured flag together "
            "with its deterministic missing-indicator complement, which makes an "
            "intercept-bearing design rank-deficient. Their use in a separate "
            "provenance audit is valid and should not be flagged.\n\n"
            "A named `full_stay` window is an administrative analysis span: it "
            "starts at ICU admission and ends at discharge, with `end_hours` "
            "serving only as an upper safety cap (the default cap is 720 hours). "
            "Copying that planner-locked window into provenance does not turn a "
            "metadata-bound ICU/hospital mortality flag into 30-day mortality. "
            "Call it a fixed-horizon outcome only when the script actually labels "
            "or constructs 28/30-day mortality, uses another mortality column, or "
            "derives the event from event-time/follow-up data.\n\n"
            "Every finding must include `detail.issue_code`. For these narrow "
            "cases use exactly `audit_only_companion_row_gating_required`, "
            "`strict_numeric_nonfinite_guard_required`, "
            "`finalized_exposure_missing_reconciliation`, "
            "`finalized_exposure_overridden`, or "
            "`finalized_exposure_forced_raw_reconciliation`, or "
            "`plausibility_range_exclusion_required`; use `other` for "
            "anything else. Message text is explanatory only, never routing.\n\n"
            "Return JSON only: "
            '{"findings":[{"severity":"info|warning|error",'
            '"message":"short finding","detail":{"issue_code":"other",'
            '"optional":"context"}}]}. '
            "Use an empty findings list if no issue is visible; return at most four "
            "findings, with messages under 60 words and at most 20 variables each.\n\n"
            f"Step: {step.step_id if step else '(unknown)'}\n"
            f"Step intent: {step.intent if step else '(unknown)'}\n"
            # MEASURED (h1, sweep47_A): step 04_primary_landmark_survival_analysis
            # was blocked `repair_failed` for "Prevalent exposure is only audited,
            # not excluded before the 24-hour follow-up landmark" -- and the plan
            # assigns that exclusion to 01_define_analysis_cohort, whose own
            # icu_rule_refs read "Exclude prevalent events before the 24-hour
            # landmark as a cohort-definition step". Step 01 had already run.
            #
            # This auditor sees ONE step's contract and no roster, so a
            # study-level obligation discharged elsewhere is indistinguishable
            # from one nobody discharged. It was making a whole-plan judgement
            # from a step-local view, and the step it faulted was not the owner.
            + _concept_audit_step_roster_block(plan_step_roster)
            + "Planner-declared step contract:\n"
            + json.dumps(
                step_contract,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
            # MEASURED: without this block, this auditor blocked survival steps
            # for using "los_hosp instead of the contract-required ICU discharge
            # censoring represented by los_icu" -- a requirement that appears in
            # no artifact of that run. `los_icu` is absent from every one of that
            # task's 13 analysis plans, and the plans that stated a follow-up
            # rule at all stated hospital discharge. Two runs of the same task
            # were blocked for opposite choices. With nothing declared, the
            # auditor was comparing the script against its own reading of the
            # research question -- which is exactly the guess the endpoint
            # declaration exists to remove, on both sides of the comparison.
            + _concept_audit_endpoint_block(study_endpoint)
            + f"Target outcome: {context.target_outcome}\n"
            "Named time windows:\n"
            + json.dumps(
                [window.model_dump(mode="json") for window in context.time_windows],
                ensure_ascii=False,
                default=str,
            )
            + "\n"
            "Outbound-safe context:\n"
            + safe_context
            + "\n\nScript:\n"
            + _concept_audit_script_excerpt(safe_script)
        )


_LLM_CONCEPT_ISSUE_CODES = frozenset(
    {
        "audit_only_companion_row_gating_required",
        "strict_numeric_nonfinite_guard_required",
        "finalized_exposure_missing_reconciliation",
        "finalized_exposure_overridden",
        "finalized_exposure_forced_raw_reconciliation",
        "plausibility_range_exclusion_required",
        "other",
    }
)


def _invalid_llm_concept_audit_response(
    *,
    validator: str,
    step_id: Optional[str],
    reason: str,
    raw: str,
) -> List[ValidationFinding]:
    detail: Dict[str, Any] = {
        "issue_code": "llm_concept_audit_response_invalid",
        "response_issue": reason,
    }
    if step_id:
        detail["step_id"] = step_id
    head = (raw or "").strip().replace("\n", " ")[:300]
    return [
        ValidationFinding(
            validator=validator,
            severity="error",
            message=(
                "LLM concept auditor returned a response outside its required "
                f"JSON schema ({reason}): {head}"
            ),
            detail=detail,
        )
    ]


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
        return _invalid_llm_concept_audit_response(
            validator=validator,
            step_id=step_id,
            reason="invalid_json",
            raw=raw,
        )
    if not isinstance(payload, dict) or set(payload) != {"findings"}:
        return _invalid_llm_concept_audit_response(
            validator=validator,
            step_id=step_id,
            reason="top_level_object_must_contain_only_findings",
            raw=raw,
        )
    items = payload["findings"]
    if not isinstance(items, list):
        return _invalid_llm_concept_audit_response(
            validator=validator,
            step_id=step_id,
            reason="findings_must_be_a_list",
            raw=raw,
        )
    findings: List[ValidationFinding] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return _invalid_llm_concept_audit_response(
                validator=validator,
                step_id=step_id,
                reason=f"finding_{index}_must_be_an_object",
                raw=raw,
            )
        msg = item.get("message")
        sev = item.get("severity")
        detail = item.get("detail")
        if not isinstance(msg, str) or not msg.strip():
            return _invalid_llm_concept_audit_response(
                validator=validator,
                step_id=step_id,
                reason=f"finding_{index}_message_must_be_nonempty_string",
                raw=raw,
            )
        if sev not in {"info", "warning", "error"}:
            return _invalid_llm_concept_audit_response(
                validator=validator,
                step_id=step_id,
                reason=f"finding_{index}_severity_is_invalid",
                raw=raw,
            )
        if not isinstance(detail, dict):
            return _invalid_llm_concept_audit_response(
                validator=validator,
                step_id=step_id,
                reason=f"finding_{index}_detail_must_be_an_object",
                raw=raw,
            )
        issue_code = detail.get("issue_code")
        if issue_code not in _LLM_CONCEPT_ISSUE_CODES:
            return _invalid_llm_concept_audit_response(
                validator=validator,
                step_id=step_id,
                reason=f"finding_{index}_issue_code_is_invalid",
                raw=raw,
            )
        msg = msg.strip()
        if step_id:
            detail = dict(detail)
            detail.setdefault("step_id", step_id)
        if _llm_outcome_confusion_is_nonblocking(msg, detail):
            sev = "warning"
        findings.append(
            ValidationFinding(
                validator=validator,
                severity=sev,  # type: ignore[arg-type]
                message=msg,
                detail=detail or None,
            )
        )
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
    # Owner-issued concept metadata keeps ``source_concept`` as the physical
    # concept id (e.g. ``death``) and carries the endpoint semantics in the
    # description.  Project to the same endpoint semantic key the context
    # builder compares, so a metadata-bound mortality flag is downgraded on
    # exactly the same terms as an inferred one.
    source = (
        _descriptor_endpoint_semantic_key(descriptor)
        or descriptor.source_concept.lower()
    )
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
                copies_full_stay and _finding_claims_mortality_horizon_mismatch(text)
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
                    finding.model_copy(update={"severity": "warning", "detail": detail})
                )
                continue
        downgraded.append(finding)
    return downgraded


_MEASUREMENT_RECEIPT_MODULE = "easyicu.research_agent.methods.descriptive_inputs"
_MEASUREMENT_RECEIPT_HELPER = "measurement_provenance_receipt"
_MEASUREMENT_VALUE_SUFFIXES = ("_first", "_last", "_max", "_mean", "_median", "_min")


def _measurement_concept_root(value: str) -> str:
    normalized = str(value or "").strip().lower()
    for suffix in ("_measured", "_n", *_MEASUREMENT_VALUE_SUFFIXES):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _is_standard_main_guard(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
        and len(node.ops) == len(node.comparators) == 1
        and isinstance(node.ops[0], ast.Eq)
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == "__main__"
    )


def _direct_host_measurement_receipt_roots(tree: ast.Module) -> set[str]:
    """Prove direct execution of exact self-raising host receipt calls."""

    exact_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == _MEASUREMENT_RECEIPT_MODULE
        and any(
            alias.name == _MEASUREMENT_RECEIPT_HELPER and alias.asname is None
            for alias in node.names
        )
    ]
    if len(exact_imports) != 1:
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id == _MEASUREMENT_RECEIPT_HELPER:
                return set()
        if isinstance(node, ast.arg) and node.arg == _MEASUREMENT_RECEIPT_HELPER:
            return set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == _MEASUREMENT_RECEIPT_HELPER:
                return set()

    entrypoints: set[str] = set()
    for statement in tree.body:
        if not isinstance(statement, ast.If) or not _is_standard_main_guard(
            statement.test
        ):
            continue
        entrypoints.update(
            item.value.func.id
            for item in statement.body
            if isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Call)
            and isinstance(item.value.func, ast.Name)
            and not item.value.args
            and not item.value.keywords
        )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.decorator_list
    }
    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    roots: set[str] = set()
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == _MEASUREMENT_RECEIPT_HELPER
        ):
            continue
        if len(call.args) != 1 or {kw.arg for kw in call.keywords} != {
            "measured_column",
            "count_column",
        }:
            return set()
        values = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        if not all(
            isinstance(values[name], ast.Constant)
            and isinstance(values[name].value, str)
            for name in ("measured_column", "count_column")
        ):
            return set()
        measured_root = _measurement_concept_root(values["measured_column"].value)
        count_root = _measurement_concept_root(values["count_column"].value)
        if not measured_root or measured_root != count_root:
            return set()

        current: ast.AST = call
        while not isinstance(current, ast.stmt):
            parent = parents.get(id(current))
            if parent is None or isinstance(
                parent, (ast.comprehension, ast.IfExp, ast.Lambda)
            ):
                return set()
            current = parent
        scope = parents.get(id(current))
        if not (
            isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            and scope.name in entrypoints
            and functions.get(scope.name) is scope
        ):
            return set()
        roots.add(measured_root)
    return roots


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

    issue_code = "audit_only_companion_row_gating_required"
    derived_provenance_flags: Dict[str, bool] = {}

    def _named_value_selectors_are_value_owned(
        detail: Mapping[str, Any], tree: ast.AST
    ) -> Optional[bool]:
        raw_variables = detail.get("variables")
        if not isinstance(raw_variables, list):
            return None
        names = {
            value
            for value in raw_variables
            if isinstance(value, str) and value.isidentifier()
        }
        assignments: Dict[str, List[ast.Assign]] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                assignments.setdefault(node.targets[0].id, []).append(node)
        observed = False
        for name in names:
            definitions = assignments.get(name, [])
            if len(definitions) != 1:
                continue
            selectors = [
                node
                for node in ast.walk(definitions[0].value)
                if isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "loc"
                and isinstance(node.value.value, ast.Name)
                and isinstance(node.slice, ast.Name)
            ]
            for selector in selectors:
                observed = True
                mask_definitions = assignments.get(selector.slice.id, [])
                if len(mask_definitions) != 1:
                    return False
                receiver = selector.value.value.id
                if not any(
                    isinstance(candidate, ast.Name)
                    and candidate.id == receiver
                    and isinstance(candidate.ctx, ast.Load)
                    for candidate in ast.walk(mask_definitions[0].value)
                ):
                    return False
        return True if observed else None

    def _failure_guard(test: ast.AST) -> bool:
        if isinstance(test, ast.Name):
            name = test.id.lower()
            return name in {"invalid_pair_n", "discordant_n"} or (
                derived_provenance_flags.get(name) is True
            )
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return (
                isinstance(test.operand, ast.Name)
                and derived_provenance_flags.get(test.operand.id.lower()) is False
            )
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            return any(_failure_guard(value) for value in test.values)
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == len(test.comparators) == 1
        ):
            left = test.left
            right = test.comparators[0]
            if not isinstance(left, ast.Name) or not isinstance(right, ast.Constant):
                return False
            name = left.id.lower()
            value = right.value
            if name not in derived_provenance_flags:
                return False
            return (
                isinstance(test.ops[0], (ast.Eq, ast.Is))
                and value is derived_provenance_flags[name]
            )
        return False

    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        tree = None
    ast_tokens = set()
    host_receipt_roots: set[str] = set()
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
            if not {"invalid_pair_n", "discordant_n"}.issubset(value_tokens):
                continue
            if not any(
                isinstance(candidate, ast.Name)
                and candidate.id.lower() == "measurement_provenance_audit"
                for candidate in ast.walk(value)
            ):
                continue
            failure_value: Optional[bool] = None
            if isinstance(value, ast.Call) and _call_name(value).lower() == "any":
                failure_value = True
            elif (
                isinstance(value, ast.UnaryOp)
                and isinstance(value.op, ast.Not)
                and isinstance(value.operand, ast.Call)
                and _call_name(value.operand).lower() == "any"
            ):
                failure_value = False
            if failure_value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            derived_provenance_flags.update(
                {
                    target.id.lower(): failure_value
                    for target in targets
                    if isinstance(target, ast.Name)
                }
            )
        host_receipt_roots = _direct_host_measurement_receipt_roots(tree)
    contract_tokens = {
        "measurement_provenance_audit",
        "invalid_pair_n",
        "discordant_n",
        "audit_only",
    }
    fail_closed_guard = tree is not None and any(
        isinstance(node, ast.If)
        and _failure_guard(node.test)
        and bool(node.body)
        and isinstance(node.body[0], ast.Raise)
        for node in ast.walk(tree)
    )
    audit_contract_present = bool(host_receipt_roots) or (
        contract_tokens.issubset(ast_tokens) and fail_closed_guard
    )
    if not audit_contract_present:
        return list(findings)

    downgraded: List[ValidationFinding] = []
    for finding in findings:
        if finding.validator == LLMConceptAuditor.name and finding.severity == "error":
            if str((finding.detail or {}).get("issue_code") or "") == issue_code:
                detail = dict(finding.detail or {})
                value_selector_proof = (
                    _named_value_selectors_are_value_owned(detail, tree)
                    if tree is not None
                    else None
                )
                if value_selector_proof is False:
                    downgraded.append(finding)
                    continue
                variables = {
                    _measurement_concept_root(str(value))
                    for value in detail.get("variables", [])
                    if str(value).strip()
                }
                if (
                    host_receipt_roots
                    and variables
                    and not variables.issubset(host_receipt_roots)
                ):
                    downgraded.append(finding)
                    continue
                detail.setdefault(
                    "downgraded_reason",
                    (
                        "The script directly invokes the exact host-owned, "
                        "self-raising measurement provenance receipt for every "
                        "reported concept; no second status guard is required."
                        if host_receipt_roots
                        else "The script records the canonical audit-only measured/count "
                        "comparison and fails the whole completed step on invalid or "
                        "discordant provenance. Companion fields must not gate "
                        "row-level physiological values."
                    ),
                )
                downgraded.append(
                    finding.model_copy(update={"severity": "warning", "detail": detail})
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


_PRIMARY_EXPOSURE_ARTIFACT = "artifact:primary_exposure_definition"
_RESOLVED_INPUTS_ENV = "EASYICU_RESOLVED_INPUTS_JSON"
_MODEL_EXPOSURE_SINKS = {
    "CoxPHFitter",
    "GEE",
    "GLM",
    "KaplanMeierFitter",
    "Logit",
    "MixedLM",
    "MNLogit",
    "OLS",
    "PHReg",
    "gee",
    "glm",
    "logit",
    "mixedlm",
    "mnlogit",
    "ols",
    "phreg",
}
# Exact estimator-training entry points are structural API evidence, unlike a
# loose substring search over source text.  Keep this library-neutral: an
# unlisted estimator that trains through ``train(...)`` must not disappear
# behind a decoy recognized model, while preprocessing helpers such as
# ``train_test_split`` are not claimed merely because their name contains the
# token ``train``.
_MODEL_TRAINING_SINKS = frozenset(
    {
        "fit",
        "fit_regularized",
        "partial_fit",
        "train",
    }
)
_VISUAL_EXPOSURE_SINKS = {
    "bar",
    "barh",
    "boxplot",
    "errorbar",
    "fill_between",
    "hist",
    "plot",
    "scatter",
    "step",
    "violinplot",
}


def _qualified_call_name(node: ast.Call) -> str:
    """Return a dotted call name without treating string contents as code."""

    parts: list[str] = []
    cursor: ast.AST = node.func
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
    return ".".join(reversed(parts))


def _subscript_string_key(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Subscript):
        return None
    candidate = node.slice
    if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
        return candidate.value
    return None


def _is_resolved_inputs_environment_lookup(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Subscript)
        and _subscript_string_key(node) == _RESOLVED_INPUTS_ENV
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and node.value.attr == "environ"
    )


def _host_manifest_names(tree: ast.Module) -> set[str]:
    """Names proven to deserialize the host resolved-inputs manifest."""

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
    ]
    manifest_path_names: set[str] = set()
    manifest_text_names: set[str] = set()

    def _targets(node: ast.Assign | ast.AnnAssign) -> List[ast.expr]:
        return node.targets if isinstance(node, ast.Assign) else [node.target]

    def _path_expression(node: ast.AST) -> bool:
        if _is_resolved_inputs_environment_lookup(node):
            return True
        if isinstance(node, ast.Name):
            return node.id in manifest_path_names
        return bool(
            isinstance(node, ast.Call)
            and _call_name(node) == "Path"
            and len(node.args) == 1
            and _path_expression(node.args[0])
        )

    changed = True
    while changed:
        changed = False
        for node in assignments:
            if not _path_expression(node.value):
                continue
            for target in _targets(node):
                if (
                    isinstance(target, ast.Name)
                    and target.id not in manifest_path_names
                ):
                    manifest_path_names.add(target.id)
                    changed = True

    def _manifest_text_expression(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in manifest_text_names
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return False
        return node.func.attr in {"read_text", "read_bytes"} and _path_expression(
            node.func.value
        )

    changed = True
    while changed:
        changed = False
        for node in assignments:
            if not _manifest_text_expression(node.value):
                continue
            for target in _targets(node):
                if (
                    isinstance(target, ast.Name)
                    and target.id not in manifest_text_names
                ):
                    manifest_text_names.add(target.id)
                    changed = True

    manifest_names: set[str] = set()
    for node in assignments:
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and _call_name(value) == "loads"
            and len(value.args) == 1
            and _manifest_text_expression(value.args[0])
        ):
            continue
        manifest_names.update(
            target.id for target in _targets(node) if isinstance(target, ast.Name)
        )
    return manifest_names


def _is_exact_primary_exposure_lookup(
    node: ast.AST,
    *,
    manifest_names: set[str],
    allow_resolved_inputs: bool,
) -> bool:
    if not (
        isinstance(node, ast.Subscript)
        and _subscript_string_key(node) == _PRIMARY_EXPOSURE_ARTIFACT
    ):
        return False
    container = node.value
    if (
        allow_resolved_inputs
        and isinstance(container, ast.Name)
        and container.id == "resolved_inputs"
    ):
        return True
    return bool(
        isinstance(container, ast.Subscript)
        and _subscript_string_key(container) == "inputs"
        and isinstance(container.value, ast.Name)
        and container.value.id in manifest_names
    )


def _reachable_local_function_names(tree: ast.Module) -> set[str]:
    """Return local functions reachable from executable module-level calls."""

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    class _CallCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()

        def visit_Call(self, node: ast.Call) -> None:
            name = _call_name(node)
            if name in functions:
                self.names.add(name)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

    collector = _CallCollector()
    for statement in tree.body:
        collector.visit(statement)
    reachable = set(collector.names)
    pending = list(reachable)
    while pending:
        name = pending.pop()
        function = functions.get(name)
        if function is None:
            continue
        nested = _CallCollector()
        for statement in function.body:
            nested.visit(statement)
        new_names = nested.names - reachable
        reachable.update(new_names)
        pending.extend(new_names)
    return reachable


def _authoritative_exposure_names(tree: ast.Module) -> set[str]:
    """Names proven to receive the host-bound primary-exposure artifact."""

    manifest_names = _host_manifest_names(tree)
    reachable_functions = _reachable_local_function_names(tree)
    parent: Dict[ast.AST, ast.AST] = {}
    for container in ast.walk(tree):
        for child in ast.iter_child_nodes(container):
            parent[child] = container

    def _in_executable_scope(node: ast.AST) -> bool:
        cursor: Optional[ast.AST] = parent.get(node)
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cursor.name in reachable_functions
            cursor = parent.get(cursor)
        return True

    resolved_inputs_unshadowed = not any(
        (
            isinstance(node, ast.Name)
            and node.id == "resolved_inputs"
            and not isinstance(node.ctx, ast.Load)
        )
        or isinstance(node, ast.arg)
        and node.arg == "resolved_inputs"
        for node in ast.walk(tree)
    )
    names: set[str] = set()
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(tree):
        if (
            not isinstance(node, (ast.Assign, ast.AnnAssign))
            or node.value is None
            or not _in_executable_scope(node)
        ):
            continue
        if not _is_exact_primary_exposure_lookup(
            node.value,
            manifest_names=manifest_names,
            allow_resolved_inputs=resolved_inputs_unshadowed,
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and node.value is not None
                and _in_executable_scope(node)
            ):
                referenced = {
                    item.id
                    for item in ast.walk(node.value)
                    if isinstance(item, ast.Name)
                }
                if not referenced & names:
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.add(target.id)
                        changed = True
            if (
                not isinstance(node, ast.Call)
                or _call_name(node) not in functions
                or not _in_executable_scope(node)
            ):
                continue
            function = functions[_call_name(node)]
            parameters = [*function.args.posonlyargs, *function.args.args]
            for parameter, argument in zip(parameters, node.args):
                if (
                    any(
                        isinstance(item, ast.Name) and item.id in names
                        for item in ast.walk(argument)
                    )
                    and parameter.arg not in names
                ):
                    names.add(parameter.arg)
                    changed = True
            keyword_parameters = {parameter.arg for parameter in parameters}
            for keyword in node.keywords:
                if (
                    keyword.arg in keyword_parameters
                    and any(
                        isinstance(item, ast.Name) and item.id in names
                        for item in ast.walk(keyword.value)
                    )
                    and keyword.arg not in names
                ):
                    names.add(str(keyword.arg))
                    changed = True
    return names


def _verified_authoritative_exposure_flow(
    script_text: str,
    *,
    primary_exposure: str,
) -> bool:
    """Require one selected exposure value to reach validation and consumption."""

    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        return False
    authority_names = _authoritative_exposure_names(tree)
    if not authority_names:
        return False
    parent: Dict[ast.AST, ast.AST] = {}
    for container in ast.walk(tree):
        for child in ast.iter_child_nodes(container):
            parent[child] = container
    reachable_functions = _reachable_local_function_names(tree)

    def _in_executable_scope(node: ast.AST) -> bool:
        cursor: Optional[ast.AST] = parent.get(node)
        while cursor is not None:
            if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cursor.name in reachable_functions
            cursor = parent.get(cursor)
        return True

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in reachable_functions
    }

    def _own_returns(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[ast.Return]:
        returns: list[ast.Return] = []

        class _ReturnCollector(ast.NodeVisitor):
            def visit_Return(self, node: ast.Return) -> None:
                returns.append(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return None

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return None

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return None

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return None

        collector = _ReturnCollector()
        for statement in function.body:
            collector.visit(statement)
        return returns

    def _return_is_authority_only(node: ast.AST) -> bool:
        """Reject helper returns that can choose or blend an unrelated value."""

        def _contains_selected(candidate: ast.AST) -> bool:
            return any(
                name.id in selected
                for name in ast.walk(candidate)
                if isinstance(name, ast.Name)
            )

        if isinstance(node, ast.IfExp):
            return _return_is_authority_only(node.body) and _return_is_authority_only(
                node.orelse
            )
        if not _contains_selected(node):
            return False
        return not any(
            isinstance(candidate, ast.Subscript)
            and not any(
                isinstance(name, ast.Name) and name.id in selected
                for name in ast.walk(candidate.value)
            )
            for candidate in ast.walk(node)
        )

    contract_columns = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and _in_executable_scope(node)
        and "executable_column"
        in {
            str(item.value)
            for item in ast.walk(node.value)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        and any(
            isinstance(item, ast.Name) and item.id in authority_names
            for item in ast.walk(node.value)
        )
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }

    def selects_authority(node: ast.AST) -> bool:
        for item in ast.walk(node):
            if not isinstance(item, ast.Subscript):
                continue
            if not any(
                isinstance(name, ast.Name) and name.id in authority_names
                for name in ast.walk(item.value)
            ):
                continue
            slice_names = {
                name.id for name in ast.walk(item.slice) if isinstance(name, ast.Name)
            }
            slice_literals = {
                str(value.value)
                for value in ast.walk(item.slice)
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            }
            if primary_exposure in slice_literals or slice_names & contract_columns:
                return True
        return False

    selected: set[str] = set()
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and _in_executable_scope(node)
    ]
    changed = True
    while changed:
        changed = False
        selected_return_functions: set[str] = set()
        for function_name, function in functions.items():
            returns = _own_returns(function)
            if not returns or not isinstance(function.body[-1], ast.Return):
                continue
            if all(
                returned.value is not None and _return_is_authority_only(returned.value)
                for returned in returns
            ):
                selected_return_functions.add(function_name)
        for node in assignments:
            referenced = {
                item.id for item in ast.walk(node.value) if isinstance(item, ast.Name)
            }
            returns_selected = bool(
                isinstance(node.value, ast.Call)
                and _call_name(node.value) in selected_return_functions
            )
            if (
                not selects_authority(node.value)
                and not referenced & selected
                and not returns_selected
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            new = {
                target.id for target in targets if isinstance(target, ast.Name)
            } - selected
            if new:
                selected.update(new)
                changed = True
    if not selected:
        return False

    def _expression_consumes_selected(node: ast.AST) -> bool:
        return selects_authority(node) or any(
            name.id in selected for name in ast.walk(node) if isinstance(name, ast.Name)
        )

    # A name-based taint set is only safe while no executable scope rebinds a
    # selected name from unrelated data.  Reject such shadowing rather than
    # treating a same-spelled local variable as the host-bound exposure.
    def _assignment_preserves_selected(
        node: ast.Assign | ast.AnnAssign,
    ) -> bool:
        returns_selected = bool(
            isinstance(node.value, ast.Call)
            and _call_name(node.value) in selected_return_functions
        )
        return _expression_consumes_selected(node.value) or returns_selected

    def _branch_side(node: ast.AST, branch: ast.If) -> Optional[str]:
        cursor: ast.AST = node
        while parent.get(cursor) is not branch:
            next_cursor = parent.get(cursor)
            if next_cursor is None:
                return None
            cursor = next_cursor
        if cursor in branch.body:
            return "body"
        if cursor in branch.orelse:
            return "orelse"
        return None

    def _mutually_exclusive(left: ast.AST, right: ast.AST) -> bool:
        for candidate in ast.walk(tree):
            if not isinstance(candidate, ast.If):
                continue
            left_side = _branch_side(left, candidate)
            right_side = _branch_side(right, candidate)
            if {left_side, right_side} == {"body", "orelse"}:
                return True
        return False

    assignments_by_target: Dict[str, list[ast.Assign | ast.AnnAssign]] = {}
    for node in assignments:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in selected:
                assignments_by_target.setdefault(target.id, []).append(node)
    for target_assignments in assignments_by_target.values():
        preserving = [
            node for node in target_assignments if _assignment_preserves_selected(node)
        ]
        for node in target_assignments:
            if node in preserving:
                continue
            if not any(_mutually_exclusive(node, other) for other in preserving):
                return False

    # Function parameters live in a different lexical scope.  A parameter
    # whose spelling collides with a selected outer name is authoritative only
    # when every executable call binds that parameter from selected data.
    executable_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _in_executable_scope(node)
    ]
    for function_name, function in functions.items():
        parameters = [*function.args.posonlyargs, *function.args.args]
        calls_to_function = [
            call for call in executable_calls if _call_name(call) == function_name
        ]
        for index, parameter in enumerate(parameters):
            if parameter.arg not in selected:
                continue
            values: list[ast.AST] = []
            for call in calls_to_function:
                if index < len(call.args):
                    values.append(call.args[index])
                    continue
                values.extend(
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == parameter.arg
                )
            if not values or not all(_expression_consumes_selected(v) for v in values):
                return False

    def _fail_closed_guard(node: ast.If) -> bool:
        if not node.body or not isinstance(node.body[0], ast.Raise):
            return False
        cursor: Optional[ast.AST] = parent.get(node)
        while cursor is not None:
            if isinstance(cursor, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
                return True
            if isinstance(cursor, ast.If) and not any(
                isinstance(name, ast.Name)
                and (name.id in authority_names or name.id in selected)
                for name in ast.walk(cursor.test)
            ):
                return False
            if isinstance(
                cursor,
                (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith),
            ):
                return False
            cursor = parent.get(cursor)
        return False

    guard_calls = [
        (call, node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and _in_executable_scope(node)
        and _fail_closed_guard(node)
        for call in ast.walk(node.test)
        if isinstance(call, ast.Call) and _expression_consumes_selected(call)
    ]

    def _is_negated_in_guard(call: ast.Call, test: ast.AST) -> bool:
        cursor: Optional[ast.AST] = parent.get(call)
        while cursor is not None and cursor is not test:
            if isinstance(cursor, ast.UnaryOp) and isinstance(
                cursor.op, (ast.Not, ast.Invert)
            ):
                return True
            if isinstance(cursor, ast.Compare) and any(
                isinstance(value, ast.Constant) and value.value is False
                for value in [cursor.left, *cursor.comparators]
            ):
                return True
            cursor = parent.get(cursor)
        return isinstance(test, ast.UnaryOp) and isinstance(
            test.op, (ast.Not, ast.Invert)
        )

    domain_checked = any(
        _call_name(call) == "isin"
        and _is_negated_in_guard(call, test)
        and {0, 1}
        <= {
            item.value
            for item in ast.walk(call)
            if isinstance(item, ast.Constant)
            and isinstance(item.value, (int, float))
            and not isinstance(item.value, bool)
        }
        for call, test in guard_calls
    )
    missing_checked = any(
        _call_name(call) == "isna" and not _is_negated_in_guard(call, test)
        for call, test in guard_calls
    )
    finite_checked = any(
        _call_name(call) == "isfinite" and _is_negated_in_guard(call, test)
        for call, test in guard_calls
    )

    def _call_consumes_selected(call: ast.Call) -> bool:
        arguments: list[ast.AST] = [
            *call.args,
            *(item.value for item in call.keywords),
        ]
        if isinstance(call.func, ast.Attribute):
            arguments.append(call.func.value)
        return any(
            name.id in selected
            for argument in arguments
            for name in ast.walk(argument)
            if isinstance(name, ast.Name)
        )

    def _is_model_sink(call: ast.Call) -> bool:
        name = _call_name(call)
        if name in _MODEL_EXPOSURE_SINKS or name in _MODEL_TRAINING_SINKS:
            return True
        return _qualified_call_name(call).endswith(".MixedLM.from_formula")

    model_roots = [call for call in executable_calls if _is_model_sink(call)]
    visual_roots = [
        call for call in executable_calls if _call_name(call) in _VISUAL_EXPOSURE_SINKS
    ]
    consumed = (
        bool(model_roots) and all(_call_consumes_selected(call) for call in model_roots)
    ) or (
        not model_roots and any(_call_consumes_selected(call) for call in visual_roots)
    )
    return domain_checked and missing_checked and finite_checked and consumed


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
    authority_names = _authoritative_exposure_names(tree)
    if not authority_names:
        return False

    reconciliation_name = "reconcile_binary_event_presence"
    wrapper_names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(child, ast.Call) and _call_name(child) == reconciliation_name
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
            and any(
                isinstance(name, ast.Name) and name.id in authority_names
                for name in ast.walk(test.args[0])
            )
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


def _has_executable_reconciliation_call(script_text: str) -> bool:
    try:
        tree = ast.parse(str(script_text or ""))
    except SyntaxError:
        return True
    return any(
        isinstance(node, ast.Call)
        and _call_name(node) == "reconcile_binary_event_presence"
        for node in ast.walk(tree)
    )


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
    reconciliation_isolated = _finalized_branch_isolates_reconciliation(script)
    reconciliation_absent = not _has_executable_reconciliation_call(script)
    direct_binding = _verified_authoritative_exposure_flow(
        script,
        primary_exposure=primary_exposure,
    )
    contracted_direct_binding = bool(
        direct_binding
        and "product_contract" in script
        and "executable_column" in script
    )
    if not direct_binding:
        return list(findings)

    downgraded: List[ValidationFinding] = []
    for finding in findings:
        if finding.validator == LLMConceptAuditor.name and finding.severity == "error":
            issue_code = str((finding.detail or {}).get("issue_code") or "")
            complains_only_about_reconciliation = (
                issue_code == "finalized_exposure_missing_reconciliation"
            )
            false_override_claim = (
                issue_code == "finalized_exposure_overridden"
                and reconciliation_isolated
            )
            false_forced_reconciliation_claim = (
                reconciliation_absent
                and contracted_direct_binding
                and issue_code == "finalized_exposure_forced_raw_reconciliation"
            )
            if (
                complains_only_about_reconciliation
                or false_override_claim
                or false_forced_reconciliation_claim
            ):
                detail = dict(finding.detail or {})
                detail.setdefault(
                    "downgraded_reason",
                    (
                        "AST control-flow verification shows raw-event "
                        "reconciliation is isolated to the non-DataFrame branch; "
                        "the finalized branch directly binds and validates the "
                        "exact binary column from the row-aligned exposure "
                        "artifact."
                        if false_override_claim or false_forced_reconciliation_claim
                        else "The script directly binds and validates the exact "
                        "binary column from the finalized row-aligned exposure "
                        "artifact. Downstream raw-event reconciliation may audit "
                        "provenance but must not redefine that authoritative "
                        "exposure."
                    ),
                )
                downgraded.append(
                    finding.model_copy(update={"severity": "warning", "detail": detail})
                )
                continue
        downgraded.append(finding)
    return downgraded


def _reclassify_flag_only_plausibility_range_findings(
    *,
    findings: Sequence[ValidationFinding],
    context: ResearchContext,
) -> List[ValidationFinding]:
    """Keep plausible-range metadata from silently changing the analysis set."""

    issue_code = "plausibility_range_exclusion_required"
    exclusion_actions = {"drop", "exclude", "fail_close", "fail_closed", "invalidate"}

    def _requests_exclusion(value: object) -> bool:
        """Recognize the typed action when an auditor appends its target."""

        normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
        return any(
            normalized == action or normalized.startswith(f"{action}_")
            for action in exclusion_actions
        )

    reclassified: List[ValidationFinding] = []
    for finding in findings:
        detail = dict(finding.detail or {})
        if not (
            finding.validator == LLMConceptAuditor.name
            and finding.severity == "error"
            and str(detail.get("issue_code") or "") == issue_code
            and _requests_exclusion(detail.get("requested_action"))
            and str(detail.get("value_class") or "").strip().lower()
            == "finite_outside_plausibility_range"
        ):
            reclassified.append(finding)
            continue

        variable = str(detail.get("variable") or "").strip()
        descriptor = context.variable(variable) if variable else None
        observed_domain = (
            dict(getattr(descriptor, "observed_domain", None) or {})
            if descriptor is not None
            else {}
        )
        strict_discrete_domain = bool(
            descriptor is not None
            and (
                descriptor.is_ordinal
                or descriptor.ordinal_levels
                or observed_domain.get("is_binary") is True
            )
        )
        if (
            descriptor is None
            or descriptor.valid_range is None
            or strict_discrete_domain
        ):
            reclassified.append(finding)
            continue

        detail.setdefault(
            "downgraded_reason",
            "ConceptDescriptor.valid_range is a flag-only physiological "
            "plausibility range. Finite continuous values outside it remain in "
            "the Planner-owned analysis set unless a typed protocol contract "
            "locks another action.",
        )
        detail.setdefault("range_policy_authority", "concept_descriptor_flag_only")

        # This adapter owns retention only. The deterministic plausibility gate
        # separately owns comparison, preservation, and receipt delivery.
        detail.setdefault("retain_and_flag_half_satisfied", "retain")
        detail.setdefault(
            "flag_obligation",
            "The script still owes a structured out-of-range count or "
            "indicator in its canonical step receipt; this downgrade is not "
            "evidence it exists. The deterministic gate enforces it.",
        )
        reclassified.append(
            finding.model_copy(update={"severity": "warning", "detail": detail})
        )
    return reclassified


def _reclassify_llm_concept_findings(
    *,
    findings: Sequence[ValidationFinding],
    context: ResearchContext,
    script_text: str,
) -> List[ValidationFinding]:
    """Apply the same host-owned policy chain to fresh and cached LLM audits."""

    reclassified = _downgrade_metadata_supported_outcome_findings(
        findings=findings,
        context=context,
        script_text=script_text,
    )
    reclassified = _downgrade_audit_only_companion_gating_findings(
        findings=reclassified,
        script_text=script_text,
    )
    reclassified = _reclassify_flag_only_plausibility_range_findings(
        findings=reclassified,
        context=context,
    )
    return _downgrade_finalized_exposure_reconciliation_findings(
        findings=reclassified,
        context=context,
        script_text=script_text,
    )


def _strip_jsonish(text: str) -> str:
    text = (text or "").strip()
    if "```" not in text:
        return text
    start = text.find("```")
    rest = text[start + 3 :]
    nl = rest.find("\n")
    if nl >= 0:
        tag = rest[:nl].strip().lower()
        if tag in {"json", "js", "javascript"} or not tag:
            rest = rest[nl + 1 :]
    end = rest.find("```")
    if end >= 0:
        rest = rest[:end]
    return rest.strip()


from ._v_support import (  # noqa: F401 — re-export facade
    _LOS_DAY_COLUMNS,
    _PATIENT_ID_COLUMNS,
    cohort_hygiene_findings,
    dedupe_findings,
)
from .cohort import (  # noqa: F401 — re-export facade
    CohortAuditor,
)
from .cross_step import (  # noqa: F401 — re-export facade
    CrossStepCohortLockValidator,
    CrossStepReconciliationTraceValidator,
    CrossStepRegisteredOutputValidator,
    CrossStepSourceStatusValidator,
    PrimaryModelContractValidator,
    StepSummaryFractionValidator,
)
from .statistical import (  # noqa: F401 — re-export facade
    StatisticalGuard,
    StatisticalValidator,
)
from .figures import (  # noqa: F401 — re-export facade
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from .clinical import (  # noqa: F401 — re-export facade
    ClinicalConstraintValidator,
)
from .publication import (  # noqa: F401 — re-export facade
    PublicationClaimAuditor,
    ReplicationDesignAuditor,
    ReplicationResultComparator,
)


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
