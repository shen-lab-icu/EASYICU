"""Five-dimension Tier-1 deterministic scorecard for the EasyICU evaluation protocol.

This module is the **bridge** between a finished research-agent run (its
on-disk readiness artifacts) and the §M1 five-dimension Tier-1 rubric used
by the manuscript's evaluation-protocol results (Fig.3 scorecard matrix).

It is deliberately a **pure-function / deterministic** layer (no LLM, no
pipeline state): every dimension subscore is mechanically recomputable from
the readiness artifacts a run already writes, which is the whole selling
point of Tier 1 (deterministic checks, *not* an LLM judging another LLM —
see the writing framework §L2.1 / §M).

Inputs (per run, written by ``pipeline_report.write_readiness_artifacts``):

* ``run_status.json``     — ``gates`` dict (execution / evidence / numeric).
* ``evidence_audit.json`` — evidence kinds + completeness.
* ``numeric_audit.json``  — numeric verification.
* ``claim_ledger.csv``    — per-claim binding status.
* ``analysis_plan.json``  — planned steps + expected outputs (plan dim).

The five dimensions (§M1):

1. ``plan``                  — coverage of the methodological steps + the
   minimum display set (≥Table 1 + ≥1 result figure + applicable audit panel).
2. ``code``                  — end-to-end execution (completed / failed steps).
3. ``result_validity``       — main estimate vs the locked reference (§M2).
   Left *unscored* until a locked reference is frozen and observed metrics
   are supplied — this is honest about §5 step 3 still being pending.
4. ``evidence_binding``      — every result sentence bound to hashed evidence,
   numbers verified (an EasyICU-specific dimension).
5. ``audit_conclusion_safety`` — the task's preset audit hazard is handled and
   no out-of-gate conclusion leaks into the manuscript (EasyICU-specific).

Plus the §M3 tristate total (``gate_reportable`` / ``analysis_only`` /
``diagnostic_only``), mechanically recomputed from ``gates`` with execution as
the hard ceiling.

This module does NOT modify or retire the existing ``grade_bench_task`` pure
grader — it *reuses* it for the numeric-bound and required-warning /
forbidden-output checks that feed ``result_validity`` and
``audit_conclusion_safety``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .icu_agent_bench import ICUAgentBenchTask, grade_bench_task
from .icu_rules import (
    detect_outcome_as_predictor,
    detect_overadjustment,
    outcome_leakage_caution,
    overadjustment_caution,
    treatment_mediator_caution,
)
from .plan_utils import read_model_covariate_names

DimensionLevel = Literal["Full", "Partial", "Marginal", "Fail"]
"""Four-level colour bin for the Fig.3 scorecard heatmap (§M4)."""

Tristate = Literal["gate_reportable", "analysis_only", "diagnostic_only"]
"""§M3 per-task overall verdict; execution_complete is the hard ceiling."""

# Default subscore -> level thresholds (§M1: "先连续子分，再分箱成 4 级").
# Per-dimension hard rules below can override the threshold result (e.g. a
# failed core step caps ``code`` regardless of the completed-step ratio).
_FULL_THRESHOLD = 0.85
_PARTIAL_THRESHOLD = 0.55
_MARGINAL_THRESHOLD = 0.25


def bin_level(
    subscore: float,
    *,
    full: float = _FULL_THRESHOLD,
    partial: float = _PARTIAL_THRESHOLD,
    marginal: float = _MARGINAL_THRESHOLD,
) -> DimensionLevel:
    """Bin a ``[0, 1]`` continuous subscore into a 4-level colour bin."""
    if subscore >= full:
        return "Full"
    if subscore >= partial:
        return "Partial"
    if subscore >= marginal:
        return "Marginal"
    return "Fail"


class DimensionScore(BaseModel):
    """One of the five Tier-1 dimensions for a single task run."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # Continuous ``[0, 1]`` subscore (goes into Fig.3 source data). ``None``
    # means the dimension is not yet scorable for this run (e.g. result
    # validity before a locked reference is frozen).
    subscore: Optional[float] = None
    # 4-level bin. ``None`` mirrors an unscored ``subscore`` (rendered NA).
    level: Optional[DimensionLevel] = None
    # The deterministic signals that produced the score — kept verbatim so a
    # reviewer can re-derive the cell from the run artifacts.
    signals: Dict[str, object] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class FiveDimensionScorecard(BaseModel):
    """The §M1 five-dimension Tier-1 scorecard + §M3 tristate for one run."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    run_id: Optional[str] = None

    plan: DimensionScore
    code: DimensionScore
    result_validity: DimensionScore
    evidence_binding: DimensionScore
    audit_conclusion_safety: DimensionScore

    # Additive sixth dimension (kind-routed reporting-guideline coverage).
    # Kept OUT of the canonical ``dimensions()`` column order so the locked
    # Fig.3 5-column schema is unchanged until promoting it is an explicit
    # design decision; exposed via ``all_dimensions()`` for the six-dim view.
    reporting_completeness: Optional[DimensionScore] = None
    # Additive fairness / subgroup dimension (TRIPOD+AI emphasis); same
    # out-of-canonical-order treatment as ``reporting_completeness``.
    fairness_subgroup: Optional[DimensionScore] = None

    tristate: Tristate

    def dimensions(self) -> List[DimensionScore]:
        """The five dimensions in the canonical Fig.3 column order."""
        return [
            self.plan,
            self.code,
            self.result_validity,
            self.evidence_binding,
            self.audit_conclusion_safety,
        ]

    def all_dimensions(self) -> List[DimensionScore]:
        """Canonical five + the additive ``reporting_completeness`` /
        ``fairness_subgroup`` dimensions when attached (extended view)."""
        dims = list(self.dimensions())
        if self.reporting_completeness is not None:
            dims.append(self.reporting_completeness)
        if self.fairness_subgroup is not None:
            dims.append(self.fairness_subgroup)
        return dims

    def source_data_row(self) -> Dict[str, object]:
        """One Fig.3 ``source_data`` row: per-dim subscore+level + tristate."""
        row: Dict[str, object] = {"task_id": self.task_id, "run_id": self.run_id}
        for dim in self.dimensions():
            row[f"{dim.name}__subscore"] = dim.subscore
            row[f"{dim.name}__level"] = dim.level
        row["tristate"] = self.tristate
        return row


# ---------------------------------------------------------------------------
# Per-dimension deterministic scoring
# ---------------------------------------------------------------------------

# Minimum display set (§M1 plan rule / §M6 default): ≥Table 1 + ≥1 result
# figure + applicable audit panel. Hard tasks (H1/H2/H3) require ≥2 result
# figures — encoded via ``min_result_figures`` derived from difficulty.
_AUDIT_OUTPUT_HINTS = ("audit", "completeness", "sensitivity", "leakage", "calibration")
_TABLE_ONE_HINTS = ("table_one", "table one", "table 1", "baseline")
_FIGURE_HINTS = ("figure", "plot", "forest", "curve")


def _output_items(task: ICUAgentBenchTask, plan_steps: Sequence[dict]) -> List[str]:
    """Lowercased list of declared output *items* (one per display artifact).

    Each task ``expected_output`` and each plan-step intent / expected output
    is one item. Counting figures over items (rather than over keyword hits)
    avoids double-counting a single figure described with two hint words
    (e.g. "forest plot" should count as one figure, not two).
    """
    items: List[str] = [str(x) for x in task.expected_outputs]
    for step in plan_steps:
        intent = str(step.get("intent", "")).strip()
        if intent:
            items.append(intent)
        items.extend(str(x) for x in (step.get("expected_outputs") or []))
    return [i.lower() for i in items if i]


def _min_result_figures(task: ICUAgentBenchTask) -> int:
    return 2 if task.difficulty == "advanced" else 1


def score_plan(
    task: ICUAgentBenchTask,
    *,
    plan_steps: Sequence[dict],
    gates: Dict[str, object],
    plan_illegal: bool = False,
) -> DimensionScore:
    """Plan dimension: declared steps + minimum display-set coverage.

    ``plan_illegal`` lets a caller pass a CTAS / illegal-concept verdict
    (a plan validator finding). When set, the plan is capped at ``Fail``.
    """
    required = int(gates.get("required_step_count") or 0)
    items = _output_items(task, plan_steps)

    has_table_one = any(any(h in it for h in _TABLE_ONE_HINTS) for it in items)
    figure_hits = sum(1 for it in items if any(h in it for h in _FIGURE_HINTS))
    has_audit_panel = any(any(h in it for h in _AUDIT_OUTPUT_HINTS) for it in items)
    n_required_figs = _min_result_figures(task)
    figure_ok = figure_hits >= n_required_figs

    structural_ok = bool(plan_steps) and required > 0
    display = (
        (1.0 if has_table_one else 0.0)
        + (1.0 if figure_ok else 0.0)
        + (1.0 if has_audit_panel else 0.0)
    ) / 3.0
    subscore = 0.4 * (1.0 if structural_ok else 0.0) + 0.6 * display

    notes: List[str] = []
    if not has_table_one:
        notes.append("no table-one display item declared")
    if not figure_ok:
        notes.append(f"declares {figure_hits} figure(s); needs >= {n_required_figs}")
    if not has_audit_panel:
        notes.append("no audit-panel display item declared")

    if plan_illegal or not structural_ok:
        level: DimensionLevel = "Fail"
        if plan_illegal:
            notes.append("plan flagged illegal (CTAS / unauthorised concept)")
        else:
            notes.append("empty or step-less plan")
    else:
        level = bin_level(subscore)

    return DimensionScore(
        name="plan",
        subscore=round(subscore, 4),
        level=level,
        signals={
            "required_step_count": required,
            "has_table_one": has_table_one,
            "result_figure_count": figure_hits,
            "min_result_figures": n_required_figs,
            "has_audit_panel": has_audit_panel,
            "plan_illegal": plan_illegal,
        },
        notes=notes,
    )


def score_code(*, gates: Dict[str, object]) -> DimensionScore:
    """Code dimension: end-to-end execution from ``gates``."""
    required = int(gates.get("required_step_count") or 0)
    completed = int(gates.get("completed_step_count") or 0)
    failed = list(gates.get("failed_steps") or [])
    execution_complete = bool(gates.get("execution_complete"))

    if required > 0:
        subscore = completed / required
    else:
        subscore = 1.0 if execution_complete else 0.0

    notes: List[str] = []
    if execution_complete and not failed:
        level: DimensionLevel = "Full"
    elif completed > 0:
        # Some core results produced despite failure(s) -> at most Partial.
        level = bin_level(subscore, full=1.01)  # full unreachable: cap below Full
        notes.append(f"{len(failed)} failed step(s); {completed}/{required} completed")
    else:
        level = "Fail"
        notes.append("no core analysis step completed")

    return DimensionScore(
        name="code",
        subscore=round(subscore, 4),
        level=level,
        signals={
            "execution_complete": execution_complete,
            "required_step_count": required,
            "completed_step_count": completed,
            "failed_step_count": len(failed),
        },
        notes=notes,
    )


def score_result_validity(
    task: ICUAgentBenchTask,
    *,
    numeric_audit: Dict[str, object],
    observed_metrics: Optional[Dict[str, float]] = None,
    locked_reference_frozen: bool = False,
    validity_errors: Optional[Sequence[str]] = None,
    validity_cautions: Optional[Sequence[str]] = None,
) -> DimensionScore:
    """Result-validity dimension (§M2 locked reference + numeric_audit).

    Returns an *unscored* dimension (``subscore``/``level`` = ``None``) when
    the locked reference is not yet frozen or no observed metrics / gold
    numeric targets are available — honest about §5 step 3 being pending.
    The numeric-bound comparison reuses ``grade_bench_task``.

    ``validity_errors`` carries deterministic, gold-free validity flaws (e.g.
    overadjustment — conditioning on a constituent of the exposure). A detected
    flaw caps the dimension at ``Fail`` even with no locked reference: an
    objective error is distinct from the honest *unscored* state where nothing
    is wrong, so the dimension is not faked, only failed when warranted.
    """
    gold = task.gold_answer
    have_targets = bool(gold and gold.numeric_targets)
    numeric_verified = bool(numeric_audit.get("numeric_verified"))
    validity_errors = list(validity_errors or [])
    validity_cautions = list(validity_cautions or [])

    if not (locked_reference_frozen and have_targets and observed_metrics):
        if validity_errors:
            return DimensionScore(
                name="result_validity",
                subscore=0.0,
                level="Fail",
                signals={
                    "locked_reference_frozen": locked_reference_frozen,
                    "validity_errors": validity_errors,
                    "validity_cautions": validity_cautions,
                    "numeric_verified": numeric_verified,
                },
                notes=(
                    [
                        "validity error(s) detected without a locked reference: "
                        + "; ".join(validity_errors)
                    ]
                    + [f"caution: {c}" for c in validity_cautions]
                ),
            )
        # A caution (e.g. overadjustment could not be auto-checked) surfaces as a
        # note but does NOT score or fail the dimension — it stays honestly
        # unscored. Errors gate; cautions only prompt verification.
        notes = ["unscored: locked reference not frozen / no observed metrics yet"]
        notes += [f"caution: {c}" for c in validity_cautions]
        return DimensionScore(
            name="result_validity",
            subscore=None,
            level=None,
            signals={
                "locked_reference_frozen": locked_reference_frozen,
                "has_numeric_targets": have_targets,
                "has_observed_metrics": bool(observed_metrics),
                "numeric_verified": numeric_verified,
                "validity_cautions": validity_cautions,
            },
            notes=notes,
        )

    graded = grade_bench_task(task, observed_metrics=observed_metrics)
    correctness = graded.correctness if graded.correctness is not None else 0.0
    subscore = correctness if numeric_verified else min(correctness, 0.5)

    notes = list(graded.notes)
    if not numeric_verified:
        notes.append("numeric drift: numeric_audit not verified (capped)")

    if correctness >= 0.999 and numeric_verified:
        level: DimensionLevel = "Full"
    elif correctness <= 0.0 or not numeric_verified:
        level = "Fail" if correctness <= 0.0 else "Partial"
    else:
        level = bin_level(subscore)

    return DimensionScore(
        name="result_validity",
        subscore=round(subscore, 4),
        level=level,
        signals={
            "correctness_fraction_in_bound": round(correctness, 4),
            "numeric_verified": numeric_verified,
            "numeric_error_count": int(numeric_audit.get("numeric_error_count") or 0),
        },
        notes=notes,
    )


def score_evidence_binding(
    *,
    evidence_audit: Dict[str, object],
    numeric_audit: Dict[str, object],
    claim_rows: Sequence[Dict[str, str]],
    min_evidence_kinds: int = 5,
) -> DimensionScore:
    """Evidence-binding dimension: bound claims + verified numbers (§M1)."""
    evidence_complete = bool(evidence_audit.get("evidence_complete"))
    missing = int(evidence_audit.get("missing_evidence_count") or 0)
    kinds = evidence_audit.get("kinds") or {}
    n_kinds = len(kinds) if isinstance(kinds, dict) else 0
    numeric_verified = bool(numeric_audit.get("numeric_verified"))

    total_claims = len(claim_rows)
    bound = sum(
        1
        for r in claim_rows
        if str(r.get("status", "")).lower() in {"bound", "ok", "verified"}
    )
    demoted = sum(
        1
        for r in claim_rows
        if str(r.get("status", "")).lower() in {"demoted", "flagged", "downgraded"}
    )
    bound_frac = bound / total_claims if total_claims else 1.0

    kinds_ok = n_kinds >= min_evidence_kinds

    notes: List[str] = []
    if evidence_complete and numeric_verified and missing == 0:
        subscore = 1.0 if kinds_ok else 0.9
        level: DimensionLevel = "Full" if kinds_ok else "Partial"
        if not kinds_ok:
            notes.append(f"only {n_kinds}/{min_evidence_kinds} evidence kinds present")
    elif missing > 0 and demoted >= missing:
        # Unbound sentences exist but were all gate-flagged / demoted (safe).
        subscore = max(0.55, bound_frac)
        level = "Partial"
        notes.append(f"{missing} unbound result sentence(s) demoted by the gate")
    else:
        subscore = max(0.0, bound_frac - 0.1 * missing)
        level = bin_level(subscore)
        if missing > 0:
            notes.append(f"{missing} unbound result sentence(s) not demoted")
        if not numeric_verified:
            notes.append("numbers not verified")

    return DimensionScore(
        name="evidence_binding",
        subscore=round(subscore, 4),
        level=level,
        signals={
            "evidence_complete": evidence_complete,
            "missing_evidence_count": missing,
            "evidence_kinds": n_kinds,
            "numeric_verified": numeric_verified,
            "claims_total": total_claims,
            "claims_bound": bound,
            "claims_demoted": demoted,
        },
        notes=notes,
    )


def score_audit_conclusion_safety(
    task: ICUAgentBenchTask,
    *,
    observed_warnings: Optional[Sequence[str]] = None,
    observed_outputs: Optional[Sequence[str]] = None,
    cohort_hygiene_cautions: Optional[Sequence[str]] = None,
    tristate: Tristate,
) -> DimensionScore:
    """Audit-conclusion-safety dimension (§M1, EasyICU-specific).

    Rewards (a) handling the task's preset audit hazard — encoded as the
    task's ``gold_answer.required_warnings`` (the per-task hazard answer key,
    kept out of shared prompts per prompt hygiene) — and (b) *not* leaking an
    out-of-gate conclusion (``gold_answer.forbidden_outputs``). Fail-closed
    demotion (``analysis_only`` / ``diagnostic_only``) is treated as SAFE:
    the selling point is that an error is safely withheld, not published.

    ``cohort_hygiene_cautions`` (the advisory flags from
    ``CohortAuditor`` / ``cohort_hygiene_findings``) are recorded as a
    transparent signal but deliberately carry **no subscore penalty here**.
    Two reasons, both from the impartiality rule: a "no patient identifier"
    caution is a *structural no-source* property of the export — penalising
    it would punish the analysis for a data limitation it cannot fix — and a
    short-stay-exposure caution is a defensible analytical choice whose
    *engagement* can only be judged against the manuscript's limitations /
    robustness section. That manuscript-level engagement check lands with the
    §M2 locked-reference wiring; until then the cautions are surfaced for the
    reader, not scored.
    """
    graded = grade_bench_task(
        task,
        observed_warnings=observed_warnings,
        observed_outputs=observed_outputs,
    )
    gold = task.gold_answer
    has_hazard_key = bool(gold and gold.required_warnings)
    has_forbidden_key = bool(gold and gold.forbidden_outputs)

    hazard_hit = graded.provenance_completeness if has_hazard_key else None
    forbidden_tripped = (graded.hallucination_rate or 0.0) >= 1.0
    no_forbidden = 0.0 if forbidden_tripped else 1.0

    notes: List[str] = list(graded.notes)

    hygiene_cautions = list(cohort_hygiene_cautions or [])
    if hygiene_cautions:
        notes.append(
            f"{len(hygiene_cautions)} cohort-hygiene caution(s) surfaced "
            "(advisory; no subscore impact — see docstring)"
        )

    if has_hazard_key and has_forbidden_key:
        subscore = 0.5 * (hazard_hit or 0.0) + 0.5 * no_forbidden
    elif has_hazard_key:
        subscore = hazard_hit or 0.0
    elif has_forbidden_key:
        subscore = no_forbidden
    else:
        # No per-task hazard key yet: fall back to fail-closed safety. Both
        # gate_reportable and a fail-closed demotion are safe; the only
        # unsafe state is a leaked forbidden conclusion.
        subscore = no_forbidden
        notes.append("no per-task audit-hazard key; scored on fail-closed safety only")

    # Hard rule: an out-of-gate conclusion leaked into the manuscript is the
    # defining unsafe failure regardless of hazard handling.
    if forbidden_tripped:
        level: DimensionLevel = "Fail"
        notes.append("out-of-gate / forbidden conclusion leaked into outputs")
    else:
        level = bin_level(subscore)

    return DimensionScore(
        name="audit_conclusion_safety",
        subscore=round(subscore, 4),
        level=level,
        signals={
            "has_hazard_key": has_hazard_key,
            "hazard_warnings_hit": hazard_hit,
            "forbidden_conclusion_leaked": forbidden_tripped,
            "cohort_hygiene_cautions": hygiene_cautions,
            "tristate": tristate,
        },
        notes=notes,
    )


# Reporting-guideline routing by task kind (EQUATOR family). Case-neutral:
# keyed on the analytical *kind*, never on a specific benchmark item,
# variable, database or score. Prediction models -> TRIPOD+AI; observational
# association / survival / causal / descriptive -> STROBE; unsupervised
# clustering and longitudinal trajectory have no EQUATOR guideline, so they
# route to an internal reporting core (unscored until that core is emitted).
_REPORTING_GUIDELINE_BY_KIND: Dict[str, str] = {
    "mortality_prediction": "tripod",
    "subphenotype_clustering": "internal",
    "longitudinal_trajectory_analysis": "internal",
}


def reporting_guideline_for_kind(kind: str) -> str:
    """Map a benchmark task kind to its reporting guideline (default STROBE)."""
    return _REPORTING_GUIDELINE_BY_KIND.get(kind, "strobe")


def score_reporting_completeness(
    task: ICUAgentBenchTask,
    *,
    checklist: Optional[Dict[str, object]] = None,
) -> DimensionScore:
    """Reporting-completeness dimension: coverage of the kind-matched reporting
    guideline checklist the run already emits (e.g. ``reporting_checklist_strobe``).

    Routes by ``task.kind`` (see ``_REPORTING_GUIDELINE_BY_KIND``) and scores the
    fraction of *applicable* checklist items addressed. Returns an *unscored*
    dimension when no applicable checklist is available — honest for the
    clustering / trajectory kinds whose internal reporting core is not yet
    defined, rather than penalising them against an inapplicable checklist.
    """
    guideline = reporting_guideline_for_kind(task.kind)
    summary: Dict[str, object] = {}
    if isinstance(checklist, dict):
        s = checklist.get("summary")
        summary = s if isinstance(s, dict) else {}

    n_total = int(summary.get("n_total") or 0)
    if n_total <= 0:
        return DimensionScore(
            name="reporting_completeness",
            subscore=None,
            level=None,
            signals={"guideline": guideline, "checklist_present": bool(summary)},
            notes=[
                "unscored: no applicable reporting checklist emitted for kind "
                f"'{task.kind}' (guideline={guideline})"
            ],
        )

    n_addressed = int(summary.get("n_addressed") or 0)
    n_partial = int(summary.get("n_partial") or 0)
    coverage = summary.get("coverage")
    subscore = (
        float(coverage)
        if isinstance(coverage, (int, float))
        else (n_addressed + 0.5 * n_partial) / n_total
    )
    subscore = max(0.0, min(1.0, subscore))

    n_open = int(summary.get("n_open") or max(0, n_total - n_addressed - n_partial))
    notes: List[str] = []
    if n_open:
        notes.append(f"{n_open}/{n_total} {guideline.upper()} item(s) open")

    return DimensionScore(
        name="reporting_completeness",
        subscore=round(subscore, 4),
        level=bin_level(subscore),
        signals={
            "guideline": guideline,
            "n_total": n_total,
            "n_addressed": n_addressed,
            "n_partial": n_partial,
            "n_open": n_open,
        },
        notes=notes,
    )


_FAIRNESS_ITEM_HINTS = (
    "subgroup",
    "fairness",
    "interaction",
    "sociodemographic",
    "disparit",
    "equity",
)


def score_fairness_subgroup(
    task: ICUAgentBenchTask,
    *,
    checklist: Optional[Dict[str, object]] = None,
) -> DimensionScore:
    """Fairness / subgroup dimension (TRIPOD+AI emphasis): did the run report
    subgroup / fairness analysis?

    Scores the addressed fraction of the fairness/subgroup-specific items in the
    kind-matched reporting checklist the run emits (STROBE item 12b; TRIPOD+AI
    items 12/18). Returns *unscored* when the checklist has no fairness item, so
    a guideline without an explicit subgroup item is not penalised. Records
    disclosure, not a mandated analysis — a task that legitimately has no
    subgroup question simply scores this dimension as not-addressed, which the
    reader interprets in context (impartiality rule).
    """
    items: Sequence[object] = []
    if isinstance(checklist, dict):
        raw = checklist.get("items")
        items = raw if isinstance(raw, list) else []
    fairness_items = [
        it
        for it in items
        if isinstance(it, dict)
        and any(h in str(it.get("statement", "")).lower() for h in _FAIRNESS_ITEM_HINTS)
    ]
    if not fairness_items:
        return DimensionScore(
            name="fairness_subgroup",
            subscore=None,
            level=None,
            signals={"fairness_items": 0},
            notes=["unscored: no fairness/subgroup item in the emitted checklist"],
        )

    addressed = sum(1 for it in fairness_items if str(it.get("status")) == "addressed")
    partial = sum(1 for it in fairness_items if str(it.get("status")) == "partial")
    n = len(fairness_items)
    subscore = (addressed + 0.5 * partial) / n

    open_ids = [
        str(it.get("item_id"))
        for it in fairness_items
        if str(it.get("status")) not in {"addressed", "partial"}
    ]
    notes: List[str] = []
    if open_ids:
        notes.append("fairness/subgroup not addressed: item(s) " + ", ".join(open_ids))

    return DimensionScore(
        name="fairness_subgroup",
        subscore=round(subscore, 4),
        level=bin_level(subscore),
        signals={"fairness_items": n, "addressed": addressed, "partial": partial},
        notes=notes,
    )


def compute_tristate(gates: Dict[str, object]) -> Tristate:
    """§M3 tristate from ``gates`` (execution is the hard ceiling)."""
    if bool(gates.get("manuscript_ready")):
        return "gate_reportable"
    if bool(gates.get("execution_complete")):
        return "analysis_only"
    return "diagnostic_only"


# ---------------------------------------------------------------------------
# Top-level bridge
# ---------------------------------------------------------------------------


def score_run(
    task: ICUAgentBenchTask,
    *,
    gates: Dict[str, object],
    plan_steps: Sequence[dict],
    evidence_audit: Dict[str, object],
    numeric_audit: Dict[str, object],
    claim_rows: Sequence[Dict[str, str]],
    observed_metrics: Optional[Dict[str, float]] = None,
    observed_warnings: Optional[Sequence[str]] = None,
    observed_outputs: Optional[Sequence[str]] = None,
    cohort_hygiene_cautions: Optional[Sequence[str]] = None,
    reporting_checklist: Optional[Dict[str, object]] = None,
    validity_errors: Optional[Sequence[str]] = None,
    validity_cautions: Optional[Sequence[str]] = None,
    locked_reference_frozen: bool = False,
    plan_illegal: bool = False,
    run_id: Optional[str] = None,
) -> FiveDimensionScorecard:
    """Compute the full five-dimension scorecard + tristate from loaded artifacts.

    Pure function over already-parsed artifact payloads — no I/O. Use
    :func:`score_run_from_dir` for the file-loading convenience wrapper. The
    additive ``reporting_completeness`` dimension is populated when a
    kind-matched reporting checklist payload is supplied.
    """
    tristate = compute_tristate(gates)
    return FiveDimensionScorecard(
        reporting_completeness=score_reporting_completeness(
            task, checklist=reporting_checklist
        ),
        fairness_subgroup=score_fairness_subgroup(task, checklist=reporting_checklist),
        task_id=task.task_id,
        run_id=run_id,
        plan=score_plan(
            task, plan_steps=plan_steps, gates=gates, plan_illegal=plan_illegal
        ),
        code=score_code(gates=gates),
        result_validity=score_result_validity(
            task,
            numeric_audit=numeric_audit,
            observed_metrics=observed_metrics,
            locked_reference_frozen=locked_reference_frozen,
            validity_errors=validity_errors,
            validity_cautions=validity_cautions,
        ),
        evidence_binding=score_evidence_binding(
            evidence_audit=evidence_audit,
            numeric_audit=numeric_audit,
            claim_rows=claim_rows,
        ),
        audit_conclusion_safety=score_audit_conclusion_safety(
            task,
            observed_warnings=observed_warnings,
            observed_outputs=observed_outputs,
            cohort_hygiene_cautions=cohort_hygiene_cautions,
            tristate=tristate,
        ),
        tristate=tristate,
    )


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_claim_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_cohort_hygiene_cautions(run_dir: Path) -> List[str]:
    """Best-effort scan of ``audit_log.jsonl`` for cohort-hygiene cautions.

    Returns the messages of any logged finding whose ``detail.kind`` is
    ``cohort_hygiene`` (emitted by ``cohort_hygiene_findings``). Missing or
    malformed logs degrade to an empty list — these cautions are advisory.
    """
    path = run_dir / "audit_log.jsonl"
    if not path.exists():
        return []
    cautions: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or '"cohort_hygiene"' not in line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        detail = row.get("detail") if isinstance(row, dict) else None
        if isinstance(detail, dict) and detail.get("kind") == "cohort_hygiene":
            msg = row.get("message")
            if isinstance(msg, str):
                cautions.append(msg)
    return cautions


def _load_regression_covariates(run_dir: Path) -> List[str]:
    """Best-effort: variable names from any model coefficient table the run
    wrote (used for the gold-free overadjustment check).

    Delegates to the shared content-based reader so the post-hoc backstop and
    the mid-flight auditor agree on what counts as a coefficient table — runs
    emit ``primary_association.csv`` / ``model_coefficients.csv`` /
    ``regression_results.csv`` interchangeably, so a filename match alone would
    miss real outputs. Missing/malformed files degrade to an empty list.
    """
    return read_model_covariate_names(run_dir)


def _load_reporting_checklist(
    run_dir: Path, task: ICUAgentBenchTask
) -> Dict[str, object]:
    """Load the kind-matched reporting-guideline checklist a run emitted.

    Returns ``{}`` when the routed guideline has no emitted checklist (e.g. the
    ``internal`` core for clustering / trajectory), so the dimension degrades to
    *unscored* rather than scoring against an inapplicable guideline.
    """
    guideline = reporting_guideline_for_kind(task.kind)
    candidates = {
        "strobe": ["reporting_checklist_strobe.json"],
        "tripod": [
            "reporting_checklist_tripod.json",
            "reporting_checklist_tripod_ai.json",
        ],
        "internal": ["reporting_checklist_internal_phenotype.json"],
    }.get(guideline, [])
    for name in candidates:
        doc = _load_json(run_dir / name)
        if doc:
            return doc
    return {}


_PHENOTYPE_KINDS = frozenset(
    {"subphenotype_clustering", "longitudinal_trajectory_analysis"}
)


def _load_cluster_validity(run_dir: Path) -> Dict[str, object]:
    """Best-effort read of a phenotype run's emitted validity metrics.

    Looks for a small documented metrics file (silhouette / n_clusters /
    min_cluster_fraction / ...). Returns ``{}`` when none is present, so the
    dimension stays honestly *unscored* rather than inventing a verdict.
    """
    for name in (
        "cluster_validity.json",
        "clustering_validity.json",
        "validity_metrics.json",
    ):
        doc = _load_json(run_dir / name)
        if doc:
            return doc
    for path in run_dir.glob("steps/*/outputs/cluster_validity.json"):
        doc = _load_json(path)
        if doc:
            return doc
    return {}


def _phenotype_validity_errors(run_dir: Path, kind: str) -> List[str]:
    """Impartial, deterministic objective-error check for clustering/trajectory.

    Flags only the *degenerate* end — a partition that is objectively broken
    regardless of analytical taste (silhouette no better than chance, a single
    group, a near-empty group). It never imposes a "good enough" threshold on a
    valid solution; softer process gaps (which selection criterion, how many
    models compared) belong to the reporting checklist, not a validity Fail.
    Returns ``[]`` (→ honest NA) when the run emitted no validity metrics.
    """
    if str(kind) not in _PHENOTYPE_KINDS:
        return []
    doc = _load_cluster_validity(run_dir)
    if not doc:
        return []
    errs: List[str] = []
    sil = doc.get("silhouette")
    if isinstance(sil, (int, float)) and not isinstance(sil, bool) and sil <= 0:
        errs.append(
            f"degenerate clustering: silhouette={float(sil):.3f} ≤ 0 "
            "(groups no better than chance)"
        )
    k = doc.get("n_clusters")
    if k is None:
        k = doc.get("n_classes")
    if isinstance(k, (int, float)) and not isinstance(k, bool) and k < 2:
        errs.append(f"single-group solution (k={int(k)}): no subphenotypes separated")
    frac = doc.get("min_cluster_fraction")
    if (
        isinstance(frac, (int, float))
        and not isinstance(frac, bool)
        and 0 <= frac < 0.01
    ):
        errs.append(
            f"near-empty group ({float(frac) * 100:.2f}% of cohort): "
            "unstable/degenerate partition"
        )
    return errs


def score_run_from_dir(
    task: ICUAgentBenchTask,
    run_dir: Path | str,
    *,
    observed_metrics: Optional[Dict[str, float]] = None,
    observed_warnings: Optional[Sequence[str]] = None,
    observed_outputs: Optional[Sequence[str]] = None,
    exposure_concept: Optional[str] = None,
    outcome_concept: Optional[str] = None,
    locked_reference_frozen: bool = False,
    plan_illegal: bool = False,
    run_id: Optional[str] = None,
) -> FiveDimensionScorecard:
    """Load a run's readiness artifacts from ``run_dir`` and score it.

    Pass ``exposure_concept`` (the primary predictor of interest, e.g.
    ``"sepsis3"``) to enable the gold-free overadjustment + treatment-mediator
    checks against the run's regression covariates, and ``outcome_concept`` (the
    study endpoint, e.g. ``"death_icu"``) to enable the outcome-leakage check.
    Both must be supplied explicitly — the bench task does not declare them, so
    they are never inferred heuristically.

    Reads ``run_status.json`` (gates), ``evidence_audit.json``,
    ``numeric_audit.json``, ``claim_ledger.csv`` and ``analysis_plan.json``.
    Missing files degrade gracefully to empty payloads (the dimensions then
    score low / unscored rather than raising).
    """
    run_dir = Path(run_dir)
    run_status = _load_json(run_dir / "run_status.json")
    gates = run_status.get("gates") if isinstance(run_status, dict) else {}
    gates = gates if isinstance(gates, dict) else {}

    plan_doc = _load_json(run_dir / "analysis_plan.json")
    plan_steps = plan_doc.get("steps") if isinstance(plan_doc, dict) else []
    plan_steps = plan_steps if isinstance(plan_steps, list) else []

    validity_errors: List[str] = []
    validity_cautions: List[str] = []
    if exposure_concept or outcome_concept:
        covariates = _load_regression_covariates(run_dir)
        if exposure_concept:
            offenders = detect_overadjustment(exposure_concept, covariates)
            if offenders:
                validity_errors.append(
                    "overadjustment: adjusted for "
                    + ", ".join(offenders)
                    + f" which constitute(s)/derive(s) the exposure '{exposure_concept}'"
                )
            else:
                caution = overadjustment_caution(exposure_concept, covariates)
                if caution:
                    validity_cautions.append(caution)
            mediator_caution = treatment_mediator_caution(exposure_concept, covariates)
            if mediator_caution:
                validity_cautions.append(mediator_caution)
        # Outcome leakage: the declared endpoint among the predictors is
        # self-leakage (objective error); a different endpoint concept used as a
        # covariate is a timing-dependent caution. The caution-tier scan does not
        # require ``outcome_concept`` to be declared.
        leak = detect_outcome_as_predictor(covariates, study_outcome=outcome_concept)
        if leak:
            validity_errors.append(
                f"outcome leakage: the study outcome '{outcome_concept}' appears "
                "among the model predictors " + ", ".join(leak)
            )
        endpoint_caution = outcome_leakage_caution(
            covariates, study_outcome=outcome_concept
        )
        if endpoint_caution:
            validity_cautions.append(endpoint_caution)
    validity_errors.extend(_phenotype_validity_errors(run_dir, task.kind))

    return score_run(
        task,
        gates=gates,
        plan_steps=plan_steps,
        evidence_audit=_load_json(run_dir / "evidence_audit.json"),
        numeric_audit=_load_json(run_dir / "numeric_audit.json"),
        claim_rows=_load_claim_rows(run_dir / "claim_ledger.csv"),
        observed_metrics=observed_metrics,
        observed_warnings=observed_warnings,
        observed_outputs=observed_outputs,
        cohort_hygiene_cautions=_load_cohort_hygiene_cautions(run_dir),
        reporting_checklist=_load_reporting_checklist(run_dir, task),
        validity_errors=validity_errors,
        validity_cautions=validity_cautions,
        locked_reference_frozen=locked_reference_frozen,
        plan_illegal=plan_illegal,
        run_id=(
            run_id or run_status.get("run_id")
            if isinstance(run_status, dict)
            else run_id
        ),
    )


__all__ = [
    "DimensionLevel",
    "Tristate",
    "DimensionScore",
    "FiveDimensionScorecard",
    "bin_level",
    "compute_tristate",
    "score_plan",
    "score_code",
    "score_result_validity",
    "score_evidence_binding",
    "score_audit_conclusion_safety",
    "score_reporting_completeness",
    "score_fairness_subgroup",
    "reporting_guideline_for_kind",
    "score_run",
    "score_run_from_dir",
]
