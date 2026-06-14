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
    {
        "id": "1a",
        "section": "Title & Abstract",
        "statement": "Indicate the study's design with a commonly used term in the title or the abstract.",
        "required_keywords": (
            "cohort",
            "observational",
            "retrospective",
            "case-control",
        ),
        "required_evidence_aliases": ("manuscript_scaffold_bound",),
    },
    {
        "id": "1b",
        "section": "Title & Abstract",
        "statement": "Provide a balanced summary of what was done and what was found in an informative abstract.",
        "required_evidence_aliases": ("manuscript_scaffold_bound",),
    },
    {
        "id": "2",
        "section": "Introduction",
        "statement": "Explain the scientific background and rationale.",
        "required_evidence_aliases": ("manuscript_scaffold_bound", "literature_bundle"),
    },
    {
        "id": "3",
        "section": "Introduction",
        "statement": "State specific objectives, including any prespecified hypotheses.",
        "required_evidence_aliases": ("hypothesis_blueprint", "analysis_plan"),
    },
    {
        "id": "4",
        "section": "Methods",
        "statement": "Present key elements of study design early in the paper.",
        "required_evidence_aliases": ("analysis_plan",),
    },
    {
        "id": "5",
        "section": "Methods",
        "statement": "Describe the setting, locations and relevant dates.",
        "required_evidence_aliases": ("research_context",),
    },
    {
        "id": "6",
        "section": "Methods",
        "statement": "Give the eligibility criteria, sources and methods of participant selection.",
        "required_evidence_aliases": ("research_context", "table_one"),
    },
    {
        "id": "7",
        "section": "Methods",
        "statement": "Clearly define all outcomes, exposures, predictors, potential confounders and effect modifiers. Give diagnostic criteria.",
        "required_evidence_aliases": ("research_context",),
    },
    {
        "id": "8",
        "section": "Methods",
        "statement": "For each variable of interest give sources of data and details of methods of assessment (measurement).",
        "required_evidence_aliases": ("research_context",),
    },
    {
        "id": "9",
        "section": "Methods",
        "statement": "Describe any efforts to address potential sources of bias.",
        "required_evidence_aliases": ("missingness", "cohort_audit"),
    },
    {
        "id": "10",
        "section": "Methods",
        "statement": "Explain how the study size was arrived at.",
        "required_evidence_aliases": ("research_context", "table_one"),
    },
    {
        "id": "11",
        "section": "Methods",
        "statement": "Explain how quantitative variables were handled in the analyses.",
        "required_evidence_aliases": ("analysis_plan",),
    },
    {
        "id": "12a",
        "section": "Methods",
        "statement": "Describe all statistical methods, including those used to control for confounding.",
        "required_evidence_aliases": ("analysis_plan", "primary_association"),
    },
    {
        "id": "12b",
        "section": "Methods",
        "statement": "Describe any methods used to examine subgroups and interactions.",
        "required_keywords": (
            "subgroup",
            "interaction",
            "stratified",
            "effect modification",
            "effect modifier",
        ),
    },
    {
        "id": "12c",
        "section": "Methods",
        "statement": "Explain how missing data were addressed.",
        "required_evidence_aliases": ("missingness",),
    },
    {
        "id": "12d",
        "section": "Methods",
        "statement": "If applicable, explain how loss to follow-up was addressed.",
        "required_keywords": ("loss to follow-up", "censoring"),
    },
    {
        "id": "12e",
        "section": "Methods",
        "statement": "Describe any sensitivity analyses.",
        "required_evidence_aliases": ("multiple_testing_report", "causal_audit_report"),
    },
    {
        # "Numbers at each stage" is the participant-flow / attrition item, NOT
        # the baseline-characteristics table — the prior ``table_one`` alias was
        # mis-specified. EasyICU agents emit the flow as a cohort_attrition /
        # cohort-flow artefact; credit it when that real artefact exists.
        "id": "13a",
        "section": "Results",
        "statement": "Report numbers of individuals at each stage of study.",
        "required_evidence_aliases": (
            "cohort_attrition", "attrition", "cohort_flow", "participant_flow",
            "study_flow",
        ),
    },
    {
        # Baseline characteristics table. Kept strict on purpose: it stays open
        # unless the agent actually surfaced a baseline-characteristics artefact
        # (a computed-but-unbound Table 1 is an honest reporting gap, not a
        # detector miss).
        "id": "14a",
        "section": "Results",
        "statement": "Give characteristics of study participants and information on exposures and potential confounders.",
        "required_evidence_aliases": (
            "table_one", "baseline_characteristics", "cohort_characteristics",
        ),
    },
    {
        "id": "15",
        "section": "Results",
        "statement": "Report numbers of outcome events or summary measures over time.",
        "required_evidence_aliases": (
            "outcome_rate", "outcome_incidence", "outcome_events",
            "mortality_by_exposure", "event_counts",
        ),
    },
    {
        # Effect estimate(s) with precision. The prior single ``primary_association``
        # alias missed the equivalent results artefacts EasyICU agents actually
        # emit (final_results_summary / evidence_bound_answer / robustness panel);
        # credit any of those real deliverables, consistent with the
        # artefact-bound == reported contract used across this checklist.
        "id": "16",
        "section": "Results",
        "statement": "Give unadjusted estimates and, if applicable, confounder-adjusted estimates and their precision.",
        "required_evidence_aliases": (
            "primary_association", "final_results_summary", "adjusted_association",
            "association_estimates", "evidence_bound_answer_to_research",
            "robustness_panel", "robustness_summary",
        ),
    },
    {
        "id": "22",
        "section": "Other",
        "statement": "Give the source of funding and the role of the funders for the present study.",
        "required_keywords": ("funding", "supported by"),
    },
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
    {
        "id": "1",
        "section": "Title",
        "statement": "Identify the study as developing, validating, or updating a multivariable prediction model using AI/ML.",
        "required_evidence_aliases": ("manuscript_scaffold_bound",),
        "required_keywords": ("prediction", "model", "AUROC", "calibration"),
    },
    {
        "id": "2",
        "section": "Abstract",
        "statement": "Provide a structured summary following TRIPOD+AI for abstracts.",
        "required_evidence_aliases": ("manuscript_scaffold_bound",),
    },
    {
        "id": "3a",
        "section": "Introduction",
        "statement": "Background, rationale, and clinical context.",
        "required_evidence_aliases": ("literature_bundle",),
    },
    {
        "id": "3b",
        "section": "Introduction",
        "statement": "Objectives, including whether the study is developmental, validation, or updating.",
        "required_evidence_aliases": ("hypothesis_blueprint", "analysis_plan"),
    },
    {
        "id": "4",
        "section": "Methods",
        "statement": "Source of data and dates.",
        "required_evidence_aliases": ("research_context",),
    },
    {
        "id": "5a",
        "section": "Methods",
        "statement": "Eligibility / inclusion and exclusion.",
        "required_evidence_aliases": ("research_context", "table_one"),
    },
    {
        "id": "5b",
        "section": "Methods",
        "statement": "Setting (secondary / tertiary ICU; geography).",
        "required_evidence_aliases": ("research_context",),
    },
    {
        "id": "6a",
        "section": "Methods",
        "statement": "Outcome(s) and how they were measured / adjudicated.",
        "required_evidence_aliases": ("research_context", "outcome_rate"),
    },
    {
        "id": "6b",
        "section": "Methods",
        "statement": "Blinding of outcome ascertainment.",
        "required_keywords": ("blinded", "unblinded", "automatic"),
    },
    {
        "id": "7",
        "section": "Methods",
        "statement": "Predictors (features) including missing-data handling.",
        "required_evidence_aliases": ("research_context", "missingness"),
    },
    {
        "id": "8",
        "section": "Methods",
        "statement": "Sample size and effective-events-per-predictor calculation.",
        "required_evidence_aliases": ("research_context", "table_one"),
    },
    {
        "id": "9",
        "section": "Methods",
        "statement": "Handling of class imbalance and cohort composition.",
        "required_keywords": (
            "class imbalance",
            # bare stem (matches "imbalance"/"imbalanced"/"class-imbalance") so a
            # run that handled imbalance is credited even when it does not use the
            # exact two-word phrase.
            "imbalance",
            "weighting",
            "oversampling",
            "undersampling",
            "minority class",
        ),
    },
    {
        "id": "10a",
        "section": "Methods",
        "statement": "Model specification: algorithm family, hyperparameters, training procedure.",
        "required_evidence_aliases": ("analysis_plan",),
    },
    {
        "id": "10b",
        "section": "Methods",
        "statement": "Model performance metrics and their interpretation.",
        "required_evidence_aliases": (
            "model_performance",
            "prediction_performance",
            "primary_association",
        ),
    },
    {
        "id": "10c",
        "section": "Methods",
        "statement": "Calibration assessment plan.",
        "required_keywords": ("calibration",),
    },
    {
        "id": "11",
        "section": "Methods",
        "statement": "Validation strategy: internal (resampling) and external (other cohort / database).",
        # The item asks for the validation STRATEGY, described in Methods prose.
        # An internal scheme (held-out / patient-level split / cross-validation /
        # bootstrap) is the realistic bar for a single-database development study;
        # the previous cross-database-only evidence alias never matched it.
        # External RESULTS are a separate item (16), which stays open when no
        # external cohort exists. Keyword-only so a described internal strategy
        # is credited on its own.
        "required_keywords": (
            "held-out",
            "hold-out",
            "cross-validation",
            "resampling",
            "bootstrap",
            "internal validation",
            "patient-level split",
            "train-test split",
        ),
    },
    {
        "id": "12",
        "section": "Methods",
        "statement": "Fairness / subgroup performance plan.",
        "required_keywords": ("fairness", "subgroup", "age", "sex", "race"),
    },
    {
        "id": "13",
        "section": "Methods",
        "statement": "Risk of bias / sensitivity analysis.",
        "required_evidence_aliases": ("multiple_testing_report", "causal_audit_report"),
    },
    {
        "id": "14",
        "section": "Results",
        "statement": "Participants flow and characteristics.",
        "required_evidence_aliases": ("table_one",),
    },
    {
        "id": "15",
        "section": "Results",
        "statement": "Model performance on the development set (AUROC, calibration, Brier).",
        "required_evidence_aliases": ("model_performance", "primary_association"),
    },
    {
        "id": "16",
        "section": "Results",
        "statement": "External validation results.",
        "required_evidence_aliases": ("cross_database_summary",),
    },
    {
        "id": "17",
        "section": "Results",
        "statement": "Calibration plot / reliability diagram.",
        # This Results item is a FIGURE (a calibration plot / reliability
        # diagram). A produced calibration/reliability figure satisfies it even
        # when the prose does not use the exact words "calibration plot" /
        # "reliability diagram" (the prior keyword-only criteria never matched a
        # figure the run actually emitted). These are generic, case-neutral
        # figure-artefact names; `_alias_satisfied` prefix-matching credits e.g.
        # `discrimination_calibration` / `discrimination_calibration_panel`.
        "required_evidence_aliases": (
            "calibration_plot",
            "calibration_curve",
            "reliability_diagram",
            "discrimination_calibration",
        ),
    },
    {
        "id": "18",
        "section": "Results",
        "statement": "Subgroup / fairness results.",
        "required_keywords": ("fairness", "subgroup"),
    },
    {
        "id": "19",
        "section": "Results",
        "statement": "Decision-curve or net-benefit analysis, if applicable.",
        "required_keywords": ("decision curve", "net benefit"),
    },
    {
        "id": "20",
        "section": "Discussion",
        "statement": "Limitations and usability.",
        "required_evidence_aliases": ("manuscript_scaffold_bound",),
    },
    {
        "id": "21",
        "section": "Other",
        "statement": "Data and code availability.",
        "required_evidence_aliases": ("analysis_plan", "reproducibility_envelope"),
    },
    {
        "id": "22",
        "section": "Other",
        "statement": "Registration, funding, conflicts of interest.",
        "required_keywords": ("funding", "registered"),
    },
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
# Internal phenotype-discovery reporting core (clustering + trajectory)
# ---------------------------------------------------------------------------
#
# Subphenotype clustering and longitudinal trajectory analysis have no EQUATOR
# reporting guideline, so they would otherwise leave the reporting-completeness
# dimension permanently unscored. This curated internal core lists the
# methodological reporting elements a reviewer of an ICU phenotype-discovery
# study expects. It is deliberately *process* completeness (did you report the
# selection criterion / stability / sizes), NOT a verdict on the cluster values
# — impartial: it prompts disclosure, it does not impose a "good enough"
# threshold. Items tagged ``longitudinal_only`` apply only when the run is a
# trajectory analysis; for a cross-sectional clustering run they are marked
# ``not_applicable`` (excluded from the denominator) rather than penalised.
_INTERNAL_PHENOTYPE_TEMPLATE: Tuple[Dict[str, Any], ...] = (
    {
        "id": "P1",
        "section": "Design",
        "statement": "State the phenotype-discovery objective and that the clusters/trajectories are hypothesis-generating, not validated biology.",
        "required_evidence_aliases": ("analysis_plan", "manuscript_scaffold_bound"),
    },
    {
        "id": "P2",
        "section": "Features",
        "statement": "List the clustering/trajectory input features, justify their selection, and confirm the outcome is NOT among the inputs (leakage).",
        "required_evidence_aliases": ("research_context", "analysis_plan"),
    },
    {
        "id": "P3",
        "section": "Features",
        "statement": "Describe feature scaling/standardisation and how mixed units/measurement scales were handled before distance/likelihood computation.",
        "required_keywords": (
            "scaling",
            "standardi",
            "z-score",
            "normalis",
            "normaliz",
        ),
    },
    {
        "id": "P4",
        "section": "Methods",
        "statement": "State the algorithm and the criterion used to choose the number of clusters/classes (silhouette, gap, BIC/AIC, elbow).",
        "required_keywords": (
            "silhouette",
            "bic",
            "aic",
            "gap statistic",
            "elbow",
            "number of clusters",
            "number of classes",
        ),
    },
    {
        "id": "P5",
        "section": "Methods",
        "statement": "Report a stability / reproducibility assessment (bootstrap, split-half, adjusted Rand index across seeds).",
        "required_keywords": (
            "stability",
            "bootstrap",
            "reproducib",
            "split-half",
            "adjusted rand",
            "consensus",
        ),
    },
    {
        "id": "P6",
        "section": "Results",
        "statement": "Report the size of each cluster/class and flag degenerate or near-empty groups.",
        "required_evidence_aliases": ("cluster_sizes", "cluster_summary", "table_one"),
    },
    {
        "id": "P7",
        "section": "Results",
        "statement": "Report an internal validity index for the solution (silhouette width, Calinski-Harabasz, posterior class-membership/entropy).",
        "required_keywords": (
            "silhouette",
            "posterior",
            "entropy",
            "calinski",
            "davies",
        ),
    },
    {
        "id": "P8",
        "section": "Results",
        "statement": "Characterise clusters/classes on clinically interpretable variables and compare outcomes across groups.",
        "required_evidence_aliases": (
            "primary_association",
            "table_one",
            "outcome_rate",
            "outcome_incidence",
            # Phenotype-native artefact names the clustering/trajectory agents
            # actually emit for this item: a per-cluster/-class variable profile
            # and a per-group outcome comparison. These are generic, case-neutral
            # names (not tied to any one benchmark cohort) and ``_alias_satisfied``
            # still credits only when the agent produced one of them.
            "cluster_characteristics",
            "cluster_profiles",
            "class_characteristics",
            "class_profiles",
            "cluster_mortality",
            "cluster_outcomes",
            "outcome_by_cluster",
            "outcome_by_class",
        ),
    },
    {
        "id": "P9",
        "section": "Trajectory",
        "statement": "Justify the longitudinal model class (GBTM/LCGA/k-means on trajectories), the time alignment/anchoring, and compare at least two candidate specifications.",
        "required_keywords": (
            "gbtm",
            "lcga",
            "group-based",
            "latent class",
            "trajectory model",
            "model comparison",
        ),
        "longitudinal_only": True,
    },
    {
        "id": "P10",
        "section": "Interpretation",
        "statement": "Discuss robustness limits and that cluster/trajectory labels require external validation before any clinical use.",
        "required_evidence_aliases": ("manuscript_scaffold_bound",),
    },
)

# Manuscript cues that a phenotype run is longitudinal (so the trajectory-only
# items apply rather than being marked not-applicable). Used only as a FALLBACK
# when the task kind is unknown — the bare word "trajectory" is deliberately
# excluded because it appears in generic clinical prose ("a stable clinical
# trajectory") and in step/evidence names ("01_phenotype_trajectory_clustering")
# that do not imply longitudinal trajectory MODELLING. These cues name an actual
# longitudinal-modelling method or repeated-measures design.
_LONGITUDINAL_CUES: Tuple[str, ...] = (
    "longitudinal",
    "time-updated",
    "repeated measures",
    "gbtm",
    "lcga",
    "latent class growth",
    "group-based trajector",
    "trajectory model",
    "growth mixture",
)

# Task kinds whose longitudinal status is known authoritatively (so an agent that
# mislabels a cross-sectional k-means run as "trajectory clustering" — M3 — does
# not flip the trajectory-only items on via incidental wording).
_CROSS_SECTIONAL_PHENOTYPE_KINDS = frozenset({"subphenotype_clustering"})
_LONGITUDINAL_PHENOTYPE_KINDS = frozenset({"longitudinal_trajectory_analysis"})


def _phenotype_run_is_longitudinal(
    task_kind: Optional[str], bound_manuscript: str
) -> bool:
    """Decide whether trajectory-only checklist items apply.

    The task kind is authoritative when known: ``subphenotype_clustering`` is
    cross-sectional, ``longitudinal_trajectory_analysis`` is longitudinal. Only
    when the kind is unknown do we fall back to scanning the manuscript prose
    (with markdown citations/links stripped, so step names like
    ``01_phenotype_trajectory_clustering`` do not count) for an explicit
    longitudinal-modelling cue.
    """
    k = (task_kind or "").strip().lower()
    if k in _LONGITUDINAL_PHENOTYPE_KINDS:
        return True
    if k in _CROSS_SECTIONAL_PHENOTYPE_KINDS:
        return False
    text = re.sub(r"\[[^\]]+\]\([^)]*\)", "", bound_manuscript or "")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).lower()
    return any(cue in text for cue in _LONGITUDINAL_CUES)


_LONGITUDINAL_ONLY_ITEMS = frozenset(
    row["id"] for row in _INTERNAL_PHENOTYPE_TEMPLATE if row.get("longitudinal_only")
)


def _internal_phenotype_items() -> List[ChecklistItem]:
    return [
        ChecklistItem(
            item_id=row["id"],
            section=row["section"],
            statement=row["statement"],
            required_evidence_aliases=tuple(row.get("required_evidence_aliases", ())),
            required_keywords=tuple(row.get("required_keywords", ())),
        )
        for row in _INTERNAL_PHENOTYPE_TEMPLATE
    ]


# ---------------------------------------------------------------------------
# Auto-fill engine
# ---------------------------------------------------------------------------


_ARTIFACT_ID_PREFIXES = ("table_", "figure_", "statistic_", "log_", "code_")
_HASH_SUFFIX = re.compile(r"_[0-9a-f]{6,}$")


def _semantic_aliases_from_record(rec: Any) -> set:
    """Recover the agent's *semantic* artefact name(s) from one evidence record.

    Evidence ids are emitted as ``<type>_<name>_<hash>`` and the relative path as
    ``evidence/<type>_<name>_<hash>__<name>.<ext>`` — so the clean, run-stable
    name the agent chose (e.g. ``cohort_attrition``, ``final_results_summary``)
    is recoverable. ``_available_aliases`` previously kept only the hashed id, so
    every checklist item keyed to a step-output artefact was systematically
    false-open. This reads the real names; it does not loosen matching — an item
    is still credited only when the agent actually produced the named artefact.
    """
    out: set = set()
    rec_id = getattr(rec, "evidence_id", None)
    if rec_id:
        out.add(rec_id)
        # Strip the leading type prefix and the trailing content hash.
        stem = rec_id
        for pfx in _ARTIFACT_ID_PREFIXES:
            if stem.startswith(pfx):
                stem = stem[len(pfx):]
                break
        stem = _HASH_SUFFIX.sub("", stem)
        if stem:
            out.add(stem)
    rel = getattr(rec, "relative_path", None) or ""
    if rel:
        base = rel.rsplit("/", 1)[-1]
        # The substantive name follows the ``__`` separator: ``..__<name>.<ext>``.
        if "__" in base:
            tail = base.split("__", 1)[1]
            tail = tail.rsplit(".", 1)[0]  # drop extension
            if tail:
                out.add(tail)
    return {a.lower() for a in out if a}


def _available_aliases(evidence_records: Iterable[Any]) -> set:
    ids: set = set()
    for rec in evidence_records:
        ids |= _semantic_aliases_from_record(rec)
    return ids


# Single-token, lowercase, length>=6 keywords are treated as inflectable stems:
# several checklist keywords are deliberately truncated before the inflection
# point (``standardi``, ``normaliz``, ``normalis``, ``reproducib``) so they can
# match ``standardized`` / ``normalization`` etc. A trailing ``\b`` made that
# impossible — ``\bstandardi\b`` never matches ``standardized`` (no boundary
# between ``i`` and ``z``), so a writer who DID describe standardisation was
# scored as not having reported it. We match those as a word *prefix* instead.
# Short tokens (``age``, ``bic``, ``sex``, ``aic``) and any keyword with a hyphen
# or space (``z-score``, ``loss to follow-up``) keep exact whole-word matching so
# the stem rule cannot, e.g., credit ``bic`` against ``bicarbonate``.
_STEM_KEYWORD_RE = re.compile(r"[a-z]{6,}$")


def _keyword_hit(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    kw = keyword.strip()
    if _STEM_KEYWORD_RE.fullmatch(kw):
        pattern = r"\b" + re.escape(kw) + r"[a-z]*"
    else:
        pattern = r"\b" + re.escape(kw) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _alias_satisfied(required: str, available_aliases: set) -> bool:
    """A required alias is satisfied by an exact match OR by an available alias
    that is the same artefact under a more specific name.

    The agent names artefacts with descriptive suffixes (``missingness`` →
    ``missingness_summary`` / ``missingness_profile``; ``table_one`` →
    ``table_one_locked_cohort``; ``outcome_rate`` → ``outcome_rate_by_stage``).
    Exact set membership false-opens those, so match on a ``_``-delimited token
    prefix in either direction. The ``_`` boundary keeps it precise — it credits
    ``missingness_summary`` for ``missingness`` but not ``completeness`` (no
    prefix relation).
    """
    if required in available_aliases:
        return True
    pref = required + "_"
    for avail in available_aliases:
        if avail.startswith(pref) or required.startswith(avail + "_"):
            return True
    return False


def _autofill_item(
    *,
    item: ChecklistItem,
    available_aliases: set,
    manuscript_text: str,
) -> None:
    matched_evidence = [
        a for a in item.required_evidence_aliases
        if _alias_satisfied(a, available_aliases)
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
        item.rationale = "Keyword match(es): " + ", ".join(
            f"`{k}`" for k in matched_keywords
        )
    elif item.status == "open":
        needed = list(item.required_evidence_aliases) + list(item.required_keywords)
        if needed:
            item.rationale = "Awaiting: " + ", ".join(needed)


# STROBE item 12d ("If applicable, explain how loss to follow-up was addressed")
# lives under item 12 (statistical methods) and is, by STROBE's own design, a
# COHORT-with-follow-up item. For a cross-sectional / point-treatment analysis of
# an ICU admission there is no follow-up and hence no loss to follow-up, so the
# honest score is *not applicable*, not *open* — penalising it would mark a
# design feature as a reporting omission. We treat it as applicable only for kinds
# that inherently carry a time-to-event / follow-up dimension; everything else
# routes 12d to N/A. This is deliberately narrow: a survival kind that genuinely
# should discuss censoring/loss to follow-up stays applicable (and open if the
# writer did not address it) — we never auto-N/A a follow-up design.
_STROBE_FOLLOWUP_KINDS = frozenset({"survival_analysis"})


def build_strobe_checklist(
    *,
    evidence_records: Iterable[Any],
    bound_manuscript: str,
    version: str = "2007",
    task_kind: Optional[str] = None,
) -> ChecklistReport:
    aliases = _available_aliases(evidence_records)
    items = _strobe_items()
    kind = str(task_kind) if task_kind else None
    followup_applicable = kind is None or kind in _STROBE_FOLLOWUP_KINDS
    for item in items:
        if item.item_id == "12d" and not followup_applicable:
            # No follow-up dimension in this design: loss to follow-up cannot
            # apply. Honest N/A (removed from the applicable denominator), not a
            # penalised open. When the kind is unknown we leave it applicable.
            item.status = "not_applicable"
            item.rationale = (
                "Not applicable: cross-sectional / point-treatment design with no "
                "longitudinal follow-up, so there is no loss to follow-up to address."
            )
            continue
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


def build_internal_phenotype_checklist(
    *,
    evidence_records: Iterable[Any],
    bound_manuscript: str,
    version: str = "internal-1",
    task_kind: Optional[str] = None,
) -> ChecklistReport:
    """Internal reporting core for subphenotype clustering / trajectory runs.

    Whether the trajectory-only items apply is decided by ``task_kind`` when
    known (``subphenotype_clustering`` → cross-sectional, trajectory items
    ``not_applicable``; ``longitudinal_trajectory_analysis`` → longitudinal), and
    otherwise inferred from the manuscript prose. This avoids inflating the
    coverage denominator with a longitudinal-modelling item on a cross-sectional
    run that an agent merely labelled "trajectory clustering".
    """
    aliases = _available_aliases(evidence_records)
    is_longitudinal = _phenotype_run_is_longitudinal(task_kind, bound_manuscript)
    items = _internal_phenotype_items()
    for item in items:
        if item.item_id in _LONGITUDINAL_ONLY_ITEMS and not is_longitudinal:
            item.status = "not_applicable"
            item.rationale = "not applicable: cross-sectional clustering run"
            continue
        _autofill_item(
            item=item,
            available_aliases=aliases,
            manuscript_text=bound_manuscript,
        )
    return ChecklistReport(
        name="Internal phenotype-discovery core", version=version, items=items
    )


# Authoritative task-kind -> reporting-checklist-name(s) map. The scorecard
# routes by ``task.kind`` and the pipeline emits by inferred analysis family;
# this keeps the two in agreement on which checklist file a kind expects.
_KIND_TO_CHECKLISTS: Dict[str, Tuple[str, ...]] = {
    "mortality_prediction": ("strobe", "tripod_ai"),
    "subphenotype_clustering": ("internal_phenotype",),
    "longitudinal_trajectory_analysis": ("internal_phenotype",),
}


def checklist_names_for_kind(kind: Optional[str]) -> Tuple[str, ...]:
    """Reporting checklist name(s) a benchmark task ``kind`` expects.

    Defaults to STROBE for the observational/association kinds. Used by both the
    scorecard (to locate the emitted file) and run launchers (to force emission
    of the kind-matched checklist via ``reporting_checklist_names``).
    """
    return _KIND_TO_CHECKLISTS.get(str(kind or "").lower(), ("strobe",))


def choose_checklist(analysis_type: Optional[str]) -> Tuple[str, ...]:
    """Return which checklists to emit for a given analysis family.

    Defaults to STROBE for every run, adds TRIPOD+AI when the
    analysis family looks like prediction/validation/modelling.
    """
    base = ("strobe",)
    if not analysis_type:
        return base
    at = str(analysis_type).lower()
    # Clustering / trajectory phenotype discovery has no EQUATOR guideline, so
    # it routes to the internal core instead of STROBE (checked first because
    # the family key "trajectory_clustering" would otherwise miss).
    if any(k in at for k in ("cluster", "trajector", "phenotyp", "subphenotype")):
        return ("internal_phenotype",)
    if any(k in at for k in ("predict", "classif", "regress", "prognost", "valida")):
        return base + ("tripod_ai",)
    return base


__all__ = [
    "ChecklistItem",
    "ChecklistReport",
    "build_strobe_checklist",
    "build_tripod_ai_checklist",
    "build_internal_phenotype_checklist",
    "choose_checklist",
    "checklist_names_for_kind",
]
