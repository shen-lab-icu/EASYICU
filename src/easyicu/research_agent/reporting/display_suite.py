"""Article-level display-suite coverage for research-agent runs.

This module answers one question: are the registered tables and figure
contracts rich enough for an article package? It is deliberately separate from
``reporting.readiness`` so reporting code only orchestrates artifact writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..authority.evidence_store import EvidenceStore
from ..figures.contracts import (
    figure_contract_paths,
    figure_contract_text,
    figure_contract_tier,
    panel_chart_type,
    panel_text,
    relative_contract_paths,
)
from ..authority.runtime_artifacts import current_evidence_records
from ..schema import AnalysisPlan, ResearchContext
from ..planning.study_design import infer_study_design_family
from .figure_claim_boundaries import build_figure_claim_boundary_audit

# A "result-bearing" figure contract is one whose text carries the study's
# actual findings (as opposed to audit/provenance displays). The base tokens
# cover effect-style analyses; each study-design family adds its own result
# vocabulary so descriptive/phenotyping/time-to-event runs are not fail-closed
# by association/prediction-only wording (their playbooks explicitly forbid
# forcing effect language into the figures).
_RESULT_LIKE_BASE_TOKENS = (
    "association",
    "effect",
    "risk ratio",
    "risk difference",
    "odds ratio",
    "sensitivity",
    "robustness",
    "prediction",
)
_RESULT_LIKE_FAMILY_TOKENS: Dict[str, tuple] = {
    "descriptive": ("prevalence", "incidence", "distribution", "event rate", "median"),
    "phenotyping": ("phenotype", "cluster", "embedding", "stability", "consensus"),
    "time_to_event": ("hazard", "survival", "cumulative incidence", "kaplan"),
    "prediction": ("calibration", "discrimination", "auroc", "roc"),
    "causal_emulation": (
        "causal contrast",
        "target trial",
        "iptw",
        "g-computation",
        "average treatment effect",
    ),
}

_TABLE_ONE_DIRECT_TERMS = (
    "table_one",
    "table one",
    "table 1",
    "baseline characteristic",
)
_TABLE_ONE_SUBJECT_TERMS = (
    "cohort",
    "covariate",
    "patient",
    "baseline",
    "demographic",
)
_TABLE_ONE_DESCRIPTOR_TERMS = ("summary", "characteristic", "overview")
_AUDIT_DISPLAY_CATEGORIES = {
    "audit",
    "data_quality",
    "missingness",
    "provenance",
    "sensitivity",
    "robustness",
}

DISPLAY_SUITE_AUDIT_SCHEMA_VERSION = "easyicu.display_suite_audit/3"
DISPLAY_SUITE_AUDIT_REGISTRATION = (
    "display_suite_audit",
    "statistic",
    "Article display-suite coverage audit for publication gating.",
)


def display_suite_audit_payload(gates: Mapping[str, Any]) -> Dict[str, Any]:
    """Project readiness gates into the canonical display-suite artifact.

    The display-suite owner defines this persisted schema.  Reporting hosts may
    write and register the returned value, but must not duplicate its fields.
    Required readiness keys are indexed deliberately so an incomplete host
    projection fails closed instead of producing a partial audit.
    """

    return {
        "schema_version": DISPLAY_SUITE_AUDIT_SCHEMA_VERSION,
        "display_suite_complete": gates["display_suite_complete"],
        "table_count": gates["display_table_count"],
        "figure_contract_count": gates["display_figure_contract_count"],
        "result_figure_contract_count": gates["display_result_figure_contract_count"],
        "primary_publication_figure_contract_count": gates[
            "display_primary_publication_figure_contract_count"
        ],
        "supporting_figure_contract_count": gates[
            "display_supporting_figure_contract_count"
        ],
        "other_figure_contract_count": gates["display_other_figure_contract_count"],
        "primary_publication_contract_paths": gates[
            "display_primary_publication_contract_paths"
        ],
        "supporting_figure_contract_paths": gates[
            "display_supporting_figure_contract_paths"
        ],
        "other_figure_contract_paths": gates["display_other_figure_contract_paths"],
        "contract_panel_count": gates["display_contract_panel_count"],
        "primary_publication_panel_count": gates[
            "display_primary_publication_panel_count"
        ],
        "supporting_panel_count": gates["display_supporting_panel_count"],
        "contract_role_count": gates["display_contract_role_count"],
        "primary_publication_role_count": gates[
            "display_primary_publication_role_count"
        ],
        "supporting_role_count": gates["display_supporting_role_count"],
        "chart_types": gates["display_chart_types"],
        "primary_publication_chart_types": gates[
            "display_primary_publication_chart_types"
        ],
        "supporting_chart_types": gates["display_supporting_chart_types"],
        "absolute_risk_visual_present": gates["display_absolute_risk_visual_present"],
        "primary_publication_absolute_risk_visual_present": gates[
            "display_primary_publication_absolute_risk_visual_present"
        ],
        "supporting_absolute_risk_visual_present": gates[
            "display_supporting_absolute_risk_visual_present"
        ],
        "primary_publication_result_figure_contract_count": gates[
            "display_primary_publication_result_figure_contract_count"
        ],
        "supporting_result_figure_contract_count": gates[
            "display_supporting_result_figure_contract_count"
        ],
        "categories": gates["display_categories"],
        "table_one_expected": gates["display_table_one_expected"],
        "table_one_present": gates["display_table_one_present"],
        "audit_context_present": gates["display_audit_context_present"],
        "figure_claim_boundary_status": gates[
            "display_figure_claim_boundary_status"
        ],
        "figure_claim_boundary_ready": gates[
            "display_figure_claim_boundary_ready"
        ],
        "figure_claim_boundaries": gates["display_figure_claim_boundaries"],
        "figure_claim_boundary_errors": gates[
            "display_figure_claim_boundary_errors"
        ],
        "errors": gates["display_suite_errors"],
        "design_advice": gates.get("display_design_advice", []),
    }


def _display_table_key(relative_path: str) -> str:
    return Path(str(relative_path or "")).name.split("__", 1)[-1]


def _declares_table_one_text(text: str) -> bool:
    lowered = str(text or "").lower()
    if any(term in lowered for term in _TABLE_ONE_DIRECT_TERMS):
        return True
    return any(term in lowered for term in _TABLE_ONE_SUBJECT_TERMS) and any(
        term in lowered for term in _TABLE_ONE_DESCRIPTOR_TERMS
    )


def _plan_expects_table_one(plan: Optional[AnalysisPlan]) -> bool:
    if plan is None:
        return False
    for step in plan.steps:
        items = [step.intent, step.method, *(step.expected_outputs or [])]
        for item in items:
            text = str(item or "")
            if "table" in text.lower() and _declares_table_one_text(text):
                return True
    return False


def _display_categories_for_text(text: str) -> set[str]:
    lowered = str(text or "").lower()
    categories: set[str] = set()
    if _declares_table_one_text(lowered):
        categories.add("table_one")
    if any(term in lowered for term in ("attrition", "denominator", "flow")):
        categories.add("cohort_flow")
    if any(
        term in lowered
        for term in (
            "association",
            "effect",
            "odds ratio",
            "risk ratio",
            "risk difference",
            "primary",
            "regression",
            "estimate",
        )
    ):
        categories.add("primary_effect")
    if any(term in lowered for term in ("sensitivity", "robustness", "variant")):
        categories.add("sensitivity")
    if any(
        term in lowered
        for term in (
            "audit",
            "quality",
            "missingness",
            "measurement",
            "provenance",
            "source definition",
            "zero fill",
            "complete case",
            "coercion",
            "convergence",
        )
    ):
        categories.add("data_quality")
    if any(
        term in lowered
        for term in ("auroc", "calibration", "discrimination", "prediction")
    ):
        categories.add("prediction")
    return categories


def panel_has_absolute_risk_context(panel: Mapping[str, Any]) -> bool:
    """Return whether a panel visibly provides absolute-risk context.

    This semantic belongs to the display-suite owner and is shared by the
    article-maturity projection. A panel can simultaneously audit a
    measurement process and show observed outcome risk, so a single role enum
    is not sufficient authority on its own.
    """

    role = str(panel.get("role") or "").strip().lower()
    if role in {"descriptive_result", "absolute_risk", "prevalence", "event_rate"}:
        return True
    text = panel_text(panel)
    absolute_terms = (
        "absolute risk",
        "absolute outcome risk",
        "outcome risk",
        "event rate",
        "event-rate",
        "exposure prevalence",
        "prevalence",
    )
    return any(token in text for token in absolute_terms)


# Compatibility for downstream callers that imported the historical private
# helper before the semantic became a shared owner contract.
_panel_has_absolute_risk_context = panel_has_absolute_risk_context


def summarize_display_suite_status(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    evidence: EvidenceStore,
    run_dir: Path,
    publication: Dict[str, Any],
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Summarise article-level display coverage.

    This gate sits above individual figure validation. A result figure can be
    technically valid while the manuscript package is still too thin or
    repetitive for article use.
    """

    table_keys: set[str] = set()
    categories: set[str] = set()
    for record in current_evidence_records(evidence.records(), per_step_records):
        if (
            per_step_records is not None
            and not str(record.produced_by_step or "").strip()
        ):
            # Run-level logs/statistics are audit or packaging products, not
            # proof that a current step produced a manuscript display. Current
            # publication figures are evaluated separately via their contract.
            continue
        text = " ".join(
            [
                record.evidence_id,
                record.kind,
                record.description,
                _display_table_key(record.relative_path),
                json.dumps(record.metadata or {}, ensure_ascii=False, default=str),
            ]
        )
        categories.update(_display_categories_for_text(text))
        if record.kind == "table":
            table_keys.add(_display_table_key(record.relative_path))

    family = infer_study_design_family(context)
    result_like_tokens = _RESULT_LIKE_BASE_TOKENS + _RESULT_LIKE_FAMILY_TOKENS.get(
        family, ()
    )
    contract_paths = figure_contract_paths(
        run_dir,
        per_step_records=per_step_records,
    )
    primary_contract_paths = [
        path
        for path in contract_paths
        if figure_contract_tier(path, run_dir) == "primary_publication"
    ]
    supporting_contract_paths = [
        path
        for path in contract_paths
        if figure_contract_tier(path, run_dir) == "supporting_step"
    ]
    other_contract_paths = [
        path
        for path in contract_paths
        if figure_contract_tier(path, run_dir) == "other"
    ]
    panel_count = 0
    primary_panel_count = 0
    supporting_panel_count = 0
    role_names: set[str] = set()
    primary_role_names: set[str] = set()
    supporting_role_names: set[str] = set()
    chart_types: set[str] = set()
    primary_chart_types: set[str] = set()
    supporting_chart_types: set[str] = set()
    result_like_contracts = 0
    primary_result_like_contracts = 0
    supporting_result_like_contracts = 0
    has_absolute_risk_visual = False
    primary_has_absolute_risk_visual = False
    supporting_has_absolute_risk_visual = False
    for contract_path in contract_paths:
        tier = figure_contract_tier(contract_path, run_dir)
        try:
            raw = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        text = figure_contract_text(raw)
        categories.update(_display_categories_for_text(text))
        panels = raw.get("panels")
        if not isinstance(panels, list):
            continue
        panel_count += len(panels)
        if tier == "primary_publication":
            primary_panel_count += len(panels)
        elif tier == "supporting_step":
            supporting_panel_count += len(panels)
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            role = str(panel.get("role") or "").strip().lower()
            if role:
                role_names.add(role)
                if tier == "primary_publication":
                    primary_role_names.add(role)
                elif tier == "supporting_step":
                    supporting_role_names.add(role)
            chart_type = panel_chart_type(panel)
            if chart_type:
                chart_types.add(chart_type)
                if tier == "primary_publication":
                    primary_chart_types.add(chart_type)
                elif tier == "supporting_step":
                    supporting_chart_types.add(chart_type)
            if panel_has_absolute_risk_context(panel):
                has_absolute_risk_visual = True
                if tier == "primary_publication":
                    primary_has_absolute_risk_visual = True
                elif tier == "supporting_step":
                    supporting_has_absolute_risk_visual = True
        if any(token in text.lower() for token in result_like_tokens):
            result_like_contracts += 1
            if tier == "primary_publication":
                primary_result_like_contracts += 1
            elif tier == "supporting_step":
                supporting_result_like_contracts += 1

    table_one_expected = _plan_expects_table_one(plan)
    has_table_one = "table_one" in categories
    has_audit_context = bool(categories & _AUDIT_DISPLAY_CATEGORIES)
    figure_contract_count = len(contract_paths)
    errors: List[str] = []
    design_advice: List[str] = []
    if table_one_expected and not has_table_one:
        errors.append(
            "Table 1/baseline cohort display was declared but not registered."
        )
    if not publication.get("publication_figure_bundle_ready"):
        errors.append("No complete publication figure bundle is registered.")
    if primary_result_like_contracts == 0:
        errors.append(
            "No primary publication result-bearing figure contract is registered."
        )
    if primary_panel_count < 2:
        design_advice.append("Primary publication figure exposes fewer than two panels.")
    if len(primary_role_names) < 2:
        design_advice.append("Primary publication figure lacks panel-role diversity.")
    if not has_audit_context:
        errors.append(
            "No audit, sensitivity, robustness, missingness, or provenance display is registered."
        )
    if len(categories) < 3:
        design_advice.append(
            "Display suite covers fewer than three article content categories."
        )
    if (
        family == "association"
        and primary_result_like_contracts > 0
        and not primary_has_absolute_risk_visual
    ):
        errors.append(
            "Primary association figure lacks a visual prevalence, event-rate, or absolute-risk "
            "panel; adjusted relative estimates and risk-difference sensitivity panels alone "
            "are not an article-level display."
        )
    generic_chart_types = {"bar", "forest", "heatmap", "unspecified"}
    if (
        family == "association"
        and primary_result_like_contracts >= 1
        and primary_panel_count >= 3
        and primary_chart_types
        and primary_chart_types <= generic_chart_types
    ):
        design_advice.append(
            "Primary association figure is limited to generic bar/forest/heatmap panels; "
            "consider a complementary visual form if it clarifies the question, "
            "such as flow, dot-interval absolute-risk, distribution, curve, or specification display."
        )
    claim_boundaries = build_figure_claim_boundary_audit(
        plan=plan,
        run_dir=run_dir,
        per_step_records=per_step_records,
    )
    if (
        plan is not None
        and plan.design_selection is not None
        and primary_result_like_contracts > 0
        and not claim_boundaries.boundary_ready
    ):
        errors.append(
            "Primary result figures do not have complete selected-design "
            "supports/cannot-prove boundaries."
        )

    return {
        "display_suite_complete": not errors,
        "display_table_count": len(table_keys),
        "display_figure_contract_count": figure_contract_count,
        "display_result_figure_contract_count": result_like_contracts,
        "display_primary_publication_figure_contract_count": len(
            primary_contract_paths
        ),
        "display_supporting_figure_contract_count": len(supporting_contract_paths),
        "display_other_figure_contract_count": len(other_contract_paths),
        "display_primary_publication_contract_paths": relative_contract_paths(
            primary_contract_paths, run_dir
        ),
        "display_supporting_figure_contract_paths": relative_contract_paths(
            supporting_contract_paths, run_dir
        ),
        "display_other_figure_contract_paths": relative_contract_paths(
            other_contract_paths, run_dir
        ),
        "display_contract_panel_count": panel_count,
        "display_primary_publication_panel_count": primary_panel_count,
        "display_supporting_panel_count": supporting_panel_count,
        "display_contract_role_count": len(role_names),
        "display_primary_publication_role_count": len(primary_role_names),
        "display_supporting_role_count": len(supporting_role_names),
        "display_chart_types": sorted(chart_types),
        "display_primary_publication_chart_types": sorted(primary_chart_types),
        "display_supporting_chart_types": sorted(supporting_chart_types),
        "display_absolute_risk_visual_present": has_absolute_risk_visual,
        "display_primary_publication_absolute_risk_visual_present": (
            primary_has_absolute_risk_visual
        ),
        "display_supporting_absolute_risk_visual_present": (
            supporting_has_absolute_risk_visual
        ),
        "display_primary_publication_result_figure_contract_count": (
            primary_result_like_contracts
        ),
        "display_supporting_result_figure_contract_count": (
            supporting_result_like_contracts
        ),
        "display_categories": sorted(categories),
        "display_table_one_expected": table_one_expected,
        "display_table_one_present": has_table_one,
        "display_audit_context_present": has_audit_context,
        "display_figure_claim_boundary_status": claim_boundaries.status,
        "display_figure_claim_boundary_ready": claim_boundaries.boundary_ready,
        "display_figure_claim_boundaries": [
            item.model_dump(mode="json") for item in claim_boundaries.figures
        ],
        "display_figure_claim_boundary_errors": list(claim_boundaries.errors),
        "display_suite_errors": errors,
        "display_design_advice": design_advice,
    }
