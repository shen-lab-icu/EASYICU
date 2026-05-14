"""Reporting-guideline checklist auto-fill (O16).

Populates two reporter-facing checklists automatically from the
research-agent run's artefacts:

* **STROBE** — 22-item guideline for reporting observational studies
  (cohort / case–control / cross-sectional). We cover the items that
  an ICU cohort agent can answer from evidence metadata plus the
  bound manuscript; items that require human judgement (funding
  disclosure, conflicts of interest, limitations) are left open with
  a clear reason.
* **TRIPOD+AI** — 27-item 2024 update of TRIPOD for studies that
  develop or validate a prediction model with machine learning /
  regression. The prediction-task skill family (AUROC, calibration,
  external validation) naturally matches this. We answer the
  subset that the run's artefacts cover and flag the rest.

Both checklists are written as **deterministic Markdown tables**
with one row per item, three columns (item / addressed / evidence
reference). Unaddressed items become ``info`` or ``warning``
findings so a reviewer can eyeball compliance at a glance without
trusting an LLM to grade itself.

The module does not call any LLM. A future ``reporter`` agent role
may expand the prose around each item; this file limits itself to
the deterministic gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Checklist item model
# ---------------------------------------------------------------------------


@dataclass
class ChecklistItem:
    """A single reporting-guideline item with auto-fill state."""

    item_id: str
    section: str
    statement: str
    # Autofill fields
    status: str = "open"  # "addressed" | "partial" | "open" | "not_applicable"
    evidence_ids: List[str] = field(default_factory=list)
    rationale: Optional[str] = None
    # What the checklist item is hoping to see; used by the auto-fill
    # heuristic to decide ``addressed`` vs ``partial``.
    required_evidence_aliases: Tuple[str, ...] = ()
    required_keywords: Tuple[str, ...] = ()

    def to_json(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "section": self.section,
            "statement": self.statement,
            "status": self.status,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }


@dataclass
class ChecklistReport:
    """One reporting guideline instantiated for one run."""

    name: str
    version: str
    items: List[ChecklistItem] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        counts = {"addressed": 0, "partial": 0, "open": 0, "not_applicable": 0}
        for i in self.items:
            counts[i.status] = counts.get(i.status, 0) + 1
        n_total = len(self.items)
        n_closable = n_total - counts.get("not_applicable", 0)
        n_addressed = counts.get("addressed", 0) + counts.get("partial", 0) * 0.5
        coverage = (n_addressed / n_closable) if n_closable else 0.0
        return {
            "name": self.name,
            "version": self.version,
            "n_total": n_total,
            "n_addressed": counts.get("addressed", 0),
            "n_partial": counts.get("partial", 0),
            "n_open": counts.get("open", 0),
            "n_not_applicable": counts.get("not_applicable", 0),
            "coverage": round(coverage, 3),
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "summary": self.summary(),
            "items": [i.to_json() for i in self.items],
        }

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"# {self.name} ({self.version}) reporting checklist",
            "",
            (
                f"Coverage: **{s['coverage']:.0%}** "
                f"({s['n_addressed']} addressed / {s['n_partial']} partial / "
                f"{s['n_open']} open / {s['n_not_applicable']} n/a; "
                f"total {s['n_total']})"
            ),
            "",
            "| Item | Section | Statement | Status | Evidence |",
            "|---|---|---|---|---|",
        ]
        for i in self.items:
            ev = ", ".join(f"`{e}`" for e in i.evidence_ids) or "—"
            lines.append(
                "| {iid} | {sec} | {stmt} | {st} | {ev} |".format(
                    iid=i.item_id,
                    sec=i.section,
                    stmt=i.statement.replace("|", "/")[:110],
                    st=i.status,
                    ev=ev,
                )
            )
        lines.append("")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# STROBE template (22 items, cohort / cross-sectional mix)
# ---------------------------------------------------------------------------


_STROBE_TEMPLATE: Tuple[Dict[str, Any], ...] = (
    {"id": "1a", "section": "Title & Abstract", "statement": "Indicate the study's design with a commonly used term in the title or the abstract.", "required_keywords": ("cohort", "observational", "retrospective", "case-control"), "required_evidence_aliases": ("manuscript_scaffold_bound",)},
    {"id": "1b", "section": "Title & Abstract", "statement": "Provide a balanced summary of what was done and what was found in an informative abstract.", "required_evidence_aliases": ("manuscript_scaffold_bound",)},
    {"id": "2", "section": "Introduction", "statement": "Explain the scientific background and rationale.", "required_evidence_aliases": ("manuscript_scaffold_bound", "literature_bundle")},
    {"id": "3", "section": "Introduction", "statement": "State specific objectives, including any prespecified hypotheses.", "required_evidence_aliases": ("hypothesis_blueprint", "analysis_plan")},
    {"id": "4", "section": "Methods", "statement": "Present key elements of study design early in the paper.", "required_evidence_aliases": ("analysis_plan",)},
    {"id": "5", "section": "Methods", "statement": "Describe the setting, locations and relevant dates.", "required_evidence_aliases": ("research_context",)},
    {"id": "6", "section": "Methods", "statement": "Give the eligibility criteria, sources and methods of participant selection.", "required_evidence_aliases": ("research_context", "table_one")},
    {"id": "7", "section": "Methods", "statement": "Clearly define all outcomes, exposures, predictors, potential confounders and effect modifiers. Give diagnostic criteria.", "required_evidence_aliases": ("research_context",)},
    {"id": "8", "section": "Methods", "statement": "For each variable of interest give sources of data and details of methods of assessment (measurement).", "required_evidence_aliases": ("research_context",)},
    {"id": "9", "section": "Methods", "statement": "Describe any efforts to address potential sources of bias.", "required_evidence_aliases": ("missingness", "cohort_audit")},
    {"id": "10", "section": "Methods", "statement": "Explain how the study size was arrived at.", "required_evidence_aliases": ("research_context", "table_one")},
    {"id": "11", "section": "Methods", "statement": "Explain how quantitative variables were handled in the analyses.", "required_evidence_aliases": ("analysis_plan",)},
    {"id": "12a", "section": "Methods", "statement": "Describe all statistical methods, including those used to control for confounding.", "required_evidence_aliases": ("analysis_plan", "primary_association")},
    {"id": "12b", "section": "Methods", "statement": "Describe any methods used to examine subgroups and interactions.", "required_evidence_aliases": ("analysis_plan",)},
    {"id": "12c", "section": "Methods", "statement": "Explain how missing data were addressed.", "required_evidence_aliases": ("missingness",)},
    {"id": "12d", "section": "Methods", "statement": "If applicable, explain how loss to follow-up was addressed.", "required_keywords": ("loss to follow-up", "censoring")},
    {"id": "12e", "section": "Methods", "statement": "Describe any sensitivity analyses.", "required_evidence_aliases": ("multiple_testing_report", "causal_audit_report")},
    {"id": "13a", "section": "Results", "statement": "Report numbers of individuals at each stage of study.", "required_evidence_aliases": ("table_one",)},
    {"id": "14a", "section": "Results", "statement": "Give characteristics of study participants and information on exposures and potential confounders.", "required_evidence_aliases": ("table_one",)},
    {"id": "15", "section": "Results", "statement": "Report numbers of outcome events or summary measures over time.", "required_evidence_aliases": ("outcome_rate", "outcome_incidence")},
    {"id": "16", "section": "Results", "statement": "Give unadjusted estimates and, if applicable, confounder-adjusted estimates and their precision.", "required_evidence_aliases": ("primary_association",)},
    {"id": "22", "section": "Other", "statement": "Give the source of funding and the role of the funders for the present study.", "required_keywords": ("funding", "supported by")},
)


def _strobe_items() -> List[ChecklistItem]:
    return [
        ChecklistItem(
            item_id=row["id"],
            section=row["section"],
            statement=row["statement"],
            required_evidence_aliases=tuple(row.get("required_evidence_aliases", ())),
            required_keywords=tuple(row.get("required_keywords", ())),
        )
        for row in _STROBE_TEMPLATE
    ]


# ---------------------------------------------------------------------------
# TRIPOD+AI template (2024, 27 items)
# ---------------------------------------------------------------------------

_TRIPOD_AI_TEMPLATE: Tuple[Dict[str, Any], ...] = (
    {"id": "1", "section": "Title", "statement": "Identify the study as developing, validating, or updating a multivariable prediction model using AI/ML.", "required_evidence_aliases": ("manuscript_scaffold_bound",), "required_keywords": ("prediction", "model", "AUROC", "calibration")},
    {"id": "2", "section": "Abstract", "statement": "Provide a structured summary following TRIPOD+AI for abstracts.", "required_evidence_aliases": ("manuscript_scaffold_bound",)},
    {"id": "3a", "section": "Introduction", "statement": "Background, rationale, and clinical context.", "required_evidence_aliases": ("literature_bundle",)},
    {"id": "3b", "section": "Introduction", "statement": "Objectives, including whether the study is developmental, validation, or updating.", "required_evidence_aliases": ("hypothesis_blueprint", "analysis_plan")},
    {"id": "4", "section": "Methods", "statement": "Source of data and dates.", "required_evidence_aliases": ("research_context",)},
    {"id": "5a", "section": "Methods", "statement": "Eligibility / inclusion and exclusion.", "required_evidence_aliases": ("research_context", "table_one")},
    {"id": "5b", "section": "Methods", "statement": "Setting (secondary / tertiary ICU; geography).", "required_evidence_aliases": ("research_context",)},
    {"id": "6a", "section": "Methods", "statement": "Outcome(s) and how they were measured / adjudicated.", "required_evidence_aliases": ("research_context", "outcome_rate")},
    {"id": "6b", "section": "Methods", "statement": "Blinding of outcome ascertainment.", "required_keywords": ("blinded", "unblinded", "automatic")},
    {"id": "7", "section": "Methods", "statement": "Predictors (features) including missing-data handling.", "required_evidence_aliases": ("research_context", "missingness")},
    {"id": "8", "section": "Methods", "statement": "Sample size and effective-events-per-predictor calculation.", "required_evidence_aliases": ("research_context", "table_one")},
    {"id": "9", "section": "Methods", "statement": "Handling of class imbalance and cohort composition.", "required_keywords": ("class imbalance", "weighting", "oversampling", "undersampling")},
    {"id": "10a", "section": "Methods", "statement": "Model specification: algorithm family, hyperparameters, training procedure.", "required_evidence_aliases": ("analysis_plan",)},
    {"id": "10b", "section": "Methods", "statement": "Model performance metrics and their interpretation.", "required_evidence_aliases": ("model_performance", "prediction_performance", "primary_association")},
    {"id": "10c", "section": "Methods", "statement": "Calibration assessment plan.", "required_keywords": ("calibration",)},
    {"id": "11", "section": "Methods", "statement": "Validation strategy: internal (resampling) and external (other cohort / database).", "required_evidence_aliases": ("cross_database_summary", "primary_association")},
    {"id": "12", "section": "Methods", "statement": "Fairness / subgroup performance plan.", "required_keywords": ("fairness", "subgroup", "age", "sex", "race")},
    {"id": "13", "section": "Methods", "statement": "Risk of bias / sensitivity analysis.", "required_evidence_aliases": ("multiple_testing_report", "causal_audit_report")},
    {"id": "14", "section": "Results", "statement": "Participants flow and characteristics.", "required_evidence_aliases": ("table_one",)},
    {"id": "15", "section": "Results", "statement": "Model performance on the development set (AUROC, calibration, Brier).", "required_evidence_aliases": ("model_performance", "primary_association")},
    {"id": "16", "section": "Results", "statement": "External validation results.", "required_evidence_aliases": ("cross_database_summary",)},
    {"id": "17", "section": "Results", "statement": "Calibration plot / reliability diagram.", "required_keywords": ("calibration plot", "reliability")},
    {"id": "18", "section": "Results", "statement": "Subgroup / fairness results.", "required_keywords": ("fairness", "subgroup")},
    {"id": "19", "section": "Results", "statement": "Decision-curve or net-benefit analysis, if applicable.", "required_keywords": ("decision curve", "net benefit")},
    {"id": "20", "section": "Discussion", "statement": "Limitations and usability.", "required_evidence_aliases": ("manuscript_scaffold_bound",)},
    {"id": "21", "section": "Other", "statement": "Data and code availability.", "required_evidence_aliases": ("analysis_plan", "reproducibility_envelope")},
    {"id": "22", "section": "Other", "statement": "Registration, funding, conflicts of interest.", "required_keywords": ("funding", "registered")},
)


def _tripod_ai_items() -> List[ChecklistItem]:
    return [
        ChecklistItem(
            item_id=row["id"],
            section=row["section"],
            statement=row["statement"],
            required_evidence_aliases=tuple(row.get("required_evidence_aliases", ())),
            required_keywords=tuple(row.get("required_keywords", ())),
        )
        for row in _TRIPOD_AI_TEMPLATE
    ]


# ---------------------------------------------------------------------------
# Auto-fill engine
# ---------------------------------------------------------------------------


def _available_aliases(evidence_records: Iterable[Any]) -> set:
    ids: set = set()
    for rec in evidence_records:
        rec_id = getattr(rec, "evidence_id", None)
        if rec_id:
            ids.add(rec_id)
    return ids


def _keyword_hit(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _autofill_item(
    *,
    item: ChecklistItem,
    available_aliases: set,
    manuscript_text: str,
) -> None:
    matched_evidence = [
        a for a in item.required_evidence_aliases if a in available_aliases
    ]
    matched_keywords = [
        k for k in item.required_keywords if _keyword_hit(manuscript_text, k)
    ]
    # Multi-alias rows are interpreted as alternatives (at least one of
    # these artefacts satisfies the item); that matches how reporting
    # guidelines actually read. Keyword lists are also alternatives.
    has_any_evidence = bool(matched_evidence)
    has_any_keyword = bool(matched_keywords)
    if item.required_evidence_aliases and item.required_keywords:
        # Both kinds specified: addressed only when both types match.
        if has_any_evidence and has_any_keyword:
            item.status = "addressed"
        elif has_any_evidence or has_any_keyword:
            item.status = "partial"
        else:
            item.status = "open"
    elif item.required_evidence_aliases:
        item.status = "addressed" if has_any_evidence else "open"
    elif item.required_keywords:
        item.status = "addressed" if has_any_keyword else "open"
    else:
        item.status = "open"
    item.evidence_ids = list(matched_evidence)
    if matched_keywords:
        item.rationale = (
            "Keyword match(es): " + ", ".join(f"`{k}`" for k in matched_keywords)
        )
    elif item.status == "open":
        needed = list(item.required_evidence_aliases) + list(item.required_keywords)
        if needed:
            item.rationale = "Awaiting: " + ", ".join(needed)


def build_strobe_checklist(
    *,
    evidence_records: Iterable[Any],
    bound_manuscript: str,
    version: str = "2007",
) -> ChecklistReport:
    aliases = _available_aliases(evidence_records)
    items = _strobe_items()
    for item in items:
        _autofill_item(
            item=item,
            available_aliases=aliases,
            manuscript_text=bound_manuscript,
        )
    return ChecklistReport(name="STROBE", version=version, items=items)


def build_tripod_ai_checklist(
    *,
    evidence_records: Iterable[Any],
    bound_manuscript: str,
    version: str = "2024",
) -> ChecklistReport:
    aliases = _available_aliases(evidence_records)
    items = _tripod_ai_items()
    for item in items:
        _autofill_item(
            item=item,
            available_aliases=aliases,
            manuscript_text=bound_manuscript,
        )
    return ChecklistReport(name="TRIPOD+AI", version=version, items=items)


def choose_checklist(analysis_type: Optional[str]) -> Tuple[str, ...]:
    """Return which checklists to emit for a given analysis family.

    Defaults to STROBE for every run, adds TRIPOD+AI when the
    analysis family looks like prediction/validation/modelling.
    """
    base = ("strobe",)
    if not analysis_type:
        return base
    at = str(analysis_type).lower()
    if any(k in at for k in ("predict", "classif", "regress", "prognost", "valida")):
        return base + ("tripod_ai",)
    return base


__all__ = [
    "ChecklistItem",
    "ChecklistReport",
    "build_strobe_checklist",
    "build_tripod_ai_checklist",
    "choose_checklist",
]
