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
from .plan_utils import read_adjustment_covariates
from .runtime_artifacts import (
    current_run_evidence_paths,
    current_successful_step_records,
    load_run_artifact_authority,
)
from .validity_signals import (
    ValiditySignal,
    assess_validity_signals,
    validity_positive_subscore,
)
from .viability import assess_cohort_viability, step_summary_block_signal

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
# A "Table 1" is the cohort baseline-characteristics table (STROBE item 14).
# Agents name it very diversely (table_one, baseline characteristics, cohort
# summary, "covariate summary by exposure", "distribution summary of core
# covariates", "cohort profile", "cohort overview", ...), so a fixed substring
# list either misses real Table-1s or, if broadened with a bare "summary",
# false-matches results/robustness summaries. Detect by NAMING PATTERN instead:
# a direct literal, OR a cohort/sample noun co-occurring with a descriptive-stats
# word. This is general (no per-task names), and excludes flow/audit tables
# (e.g. cohort_attrition, covariate_missingness_audit) that lack a descriptor.
_TABLE_ONE_DIRECT = ("table_one", "table one", "table 1", "baseline characteristic")
_TABLE_ONE_SUBJECT_TOKENS = (
    "cohort",
    "covariate",
    "patient",
    "baseline",
    "demographic",
)
# Tight descriptors: a baseline *characteristics summary*. Deliberately EXCLUDES
# "distribution"/"profile"/"description" — those match single-variable
# distributions and missingness profiles that are not a Table 1.
_TABLE_ONE_DESCRIPTOR_TOKENS = ("summary", "characteristic", "overview")


def _declares_table_one(items: Sequence[str]) -> bool:
    """True if the plan declares a Table-1 (baseline characteristics) ARTIFACT.

    Checks declared table artifacts only (``table:`` items) — never prose step
    intents, whose long sentences incidentally co-mention a cohort noun and a
    stats word. Matches a direct literal, or a cohort/sample noun co-occurring
    with a characteristics-summary descriptor.
    """
    for it in items:
        s = it.lower()
        if "table:" not in s and "table_" not in s and "table " not in s:
            continue
        if any(d in s for d in _TABLE_ONE_DIRECT):
            return True
        if any(t in s for t in _TABLE_ONE_SUBJECT_TOKENS) and any(
            t in s for t in _TABLE_ONE_DESCRIPTOR_TOKENS
        ):
            return True
    return False


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

    # The expected display set is kind-aware, consistent with the planner's
    # kind-aware Table-1 guidance and the reporting-checklist routing: a baseline
    # characteristics "Table 1" is a STROBE/TRIPOD artifact and is NOT expected
    # for clustering / trajectory (phenotype-discovery) kinds, whose guideline is
    # the internal phenotype core. Counting Table-1 against those kinds penalised
    # a plan for correctly omitting an inapplicable artifact (H3 false 0.8).
    table_one_expected = reporting_guideline_for_kind(task.kind) != "internal"

    has_table_one = _declares_table_one(items)
    figure_hits = sum(1 for it in items if any(h in it for h in _FIGURE_HINTS))
    has_audit_panel = any(any(h in it for h in _AUDIT_OUTPUT_HINTS) for it in items)
    n_required_figs = _min_result_figures(task)
    # Credit a result figure that was either declared in the plan OR actually
    # produced as a publication-figure bundle. The publication-figure skill emits
    # the result figure outside the declared plan steps, and the replanner can
    # drop an initially-declared figure step across revisions; in both cases a
    # figure IS delivered, so reading only declared steps undercounts it.
    produced_publication_figure = bool(
        gates.get("publication_figure_bundle_ready")
    ) or bool(gates.get("publication_figure_stems"))
    figure_ok = figure_hits >= n_required_figs or produced_publication_figure

    structural_ok = bool(plan_steps) and required > 0
    display_components: List[float] = []
    if table_one_expected:
        display_components.append(1.0 if has_table_one else 0.0)
    display_components.append(1.0 if figure_ok else 0.0)
    display_components.append(1.0 if has_audit_panel else 0.0)
    display = sum(display_components) / len(display_components)
    subscore = 0.4 * (1.0 if structural_ok else 0.0) + 0.6 * display

    notes: List[str] = []
    if table_one_expected and not has_table_one:
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
            "table_one_expected": table_one_expected,
            "has_table_one": has_table_one,
            "result_figure_count": figure_hits,
            "min_result_figures": n_required_figs,
            "produced_publication_figure": produced_publication_figure,
            "has_audit_panel": has_audit_panel,
            "plan_illegal": plan_illegal,
        },
        notes=notes,
    )


def score_code(
    *, gates: Dict[str, object], self_inflicted_block: Optional[str] = None
) -> DimensionScore:
    """Code dimension: end-to-end execution from ``gates``.

    ``self_inflicted_block`` (when supplied by ``score_run_from_dir``) is a
    factual note that the run blocked its primary deliverable on a task-viable
    cohort — surfaced so a ``diagnostic_only`` verdict is not mistaken for an
    infeasible task. It does not change the subscore (execution genuinely did
    not complete); it only labels *why*.
    """
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

    signals: Dict[str, object] = {
        "execution_complete": execution_complete,
        "required_step_count": required,
        "completed_step_count": completed,
        "failed_step_count": len(failed),
    }
    if self_inflicted_block:
        signals["self_inflicted_block"] = True
        notes.append(self_inflicted_block)

    return DimensionScore(
        name="code",
        subscore=round(subscore, 4),
        level=level,
        signals=signals,
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
    positive_subscore: Optional[float] = None,
    positive_signals: Optional[Sequence[ValiditySignal]] = None,
    execution_complete: bool = True,
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
        # No objective error: score the gold-free, kind-routed VALUE-based validity
        # signals (the manuscript's Fig.3 baseline — no frozen reference needed).
        # These read actual values (split overlap, adjusted-set SMD, positivity
        # verdict) and judge correctness against standard thresholds; a kind with
        # no value-readable central check contributes None → stays NA below.
        #
        # Gate behind execution_complete: result_validity judges *a produced
        # result*. A run that never completed its analysis (e.g. a self-blocked /
        # diagnostic_only run) has no result to validate, so an early-step check
        # passing must not award positive validity — that would contradict the
        # tristate. Such a run stays honestly unscored here (teeth above still cap
        # to Fail if an objective error was nonetheless detectable).
        if positive_subscore is not None and execution_complete:
            sig_pairs = [(s.name, s.status, s.detail) for s in (positive_signals or [])]
            n_assessed = sum(1 for _, st, _ in sig_pairs if st in ("pass", "fail"))
            level = bin_level(positive_subscore)
            extra_notes: List[str] = []
            # A single assessable check cannot establish FULL result validity — it
            # only tells us the one thing we could verify. Cap the label below Full
            # so a 1/1 pass is not overstated; the numeric subscore is unchanged and
            # n_assessed is surfaced for transparency.
            if n_assessed < 2 and level == "Full":
                level = "Partial"
                extra_notes.append(
                    f"label capped to Partial: only {n_assessed} central validity "
                    "check assessable for this kind (subscore unchanged)"
                )
            return DimensionScore(
                name="result_validity",
                subscore=round(positive_subscore, 4),
                level=level,
                signals={
                    "scoring_mode": "gold_free_kind_value_signals",
                    "validity_positive_subscore": round(positive_subscore, 4),
                    "validity_signals": [(n, st) for n, st, _ in sig_pairs],
                    "n_assessed": n_assessed,
                    "locked_reference_frozen": locked_reference_frozen,
                    "validity_cautions": validity_cautions,
                },
                notes=(
                    [f"{n}={st}: {d}" for n, st, d in sig_pairs]
                    + extra_notes
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

    floor_only = False
    if has_hazard_key and has_forbidden_key:
        subscore = 0.5 * (hazard_hit or 0.0) + 0.5 * no_forbidden
    elif has_hazard_key:
        subscore = hazard_hit or 0.0
    elif has_forbidden_key:
        subscore = no_forbidden
    else:
        # No per-task hazard key AND no forbidden-output key: we can only verify
        # the fail-closed FLOOR (nothing forbidden leaked). That is not evidence
        # of full conclusion safety — whether the run actually surfaced and
        # handled this question's known methodological hazard is UNASSESSED. So
        # do not award Full here; the floor caps the level at Partial with an
        # explicit note (overclaiming Full was the prior behaviour).
        subscore = no_forbidden
        floor_only = True
        notes.append(
            "no per-task audit-hazard key: only the fail-closed floor (no "
            "forbidden conclusion leaked) was verified; hazard handling unassessed"
        )

    # Hard rule: an out-of-gate conclusion leaked into the manuscript is the
    # defining unsafe failure regardless of hazard handling.
    if forbidden_tripped:
        level: DimensionLevel = "Fail"
        notes.append("out-of-gate / forbidden conclusion leaked into outputs")
    elif floor_only:
        # Floor passed but full safety unverified -> honest Partial, never Full.
        level = "Partial"
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
            "floor_only_no_hazard_key": floor_only,
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


def compute_tristate(
    gates: Dict[str, object],
    *,
    result_validity_level: Optional[str] = None,
) -> Tristate:
    """§M3 tristate from ``gates`` (execution is the hard ceiling).

    A hard validity failure (``result_validity_level == "Fail"``) caps the
    verdict at ``analysis_only``: the analysis executed and may have produced a
    bound manuscript, but its primary result is not defensible enough to license
    a reportable conclusion (e.g. overadjustment for a constituent of the
    exposure, or a finite estimate from an invalid model). The validity ceiling
    can only DEMOTE a ``gate_reportable`` verdict, never promote — a run that did
    not execute stays ``diagnostic_only`` regardless. ``None`` (validity not
    scored for this task kind) leaves the gate-based verdict unchanged.
    """
    # Fail-closed floor: a run that exhausted its replan budget without
    # converging is a runaway loop, not a reportable result — it floors to
    # ``diagnostic_only`` regardless of whatever limped through the gates.
    if bool(gates.get("replan_budget_exhausted")):
        return "diagnostic_only"
    if bool(gates.get("manuscript_ready")):
        base: Tristate = "gate_reportable"
    elif bool(gates.get("execution_complete")):
        base = "analysis_only"
    else:
        base = "diagnostic_only"
    if base == "gate_reportable" and str(result_validity_level) == "Fail":
        return "analysis_only"
    return base


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
    validity_positive_subscore: Optional[float] = None,
    validity_signals: Optional[Sequence[ValiditySignal]] = None,
    locked_reference_frozen: bool = False,
    plan_illegal: bool = False,
    self_inflicted_block: Optional[str] = None,
    run_id: Optional[str] = None,
) -> FiveDimensionScorecard:
    """Compute the full five-dimension scorecard + tristate from loaded artifacts.

    Pure function over already-parsed artifact payloads — no I/O. Use
    :func:`score_run_from_dir` for the file-loading convenience wrapper. The
    additive ``reporting_completeness`` dimension is populated when a
    kind-matched reporting checklist payload is supplied.
    """
    # Score result_validity first so a hard validity Fail can cap the tristate
    # (it must never license a reportable conclusion from an invalid result).
    result_validity = score_result_validity(
        task,
        numeric_audit=numeric_audit,
        observed_metrics=observed_metrics,
        locked_reference_frozen=locked_reference_frozen,
        validity_errors=validity_errors,
        validity_cautions=validity_cautions,
        positive_subscore=validity_positive_subscore,
        positive_signals=validity_signals,
        execution_complete=bool(gates.get("execution_complete")),
    )
    tristate = compute_tristate(gates, result_validity_level=result_validity.level)
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
        code=score_code(gates=gates, self_inflicted_block=self_inflicted_block),
        result_validity=result_validity,
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


def _current_primary_model_evidence_ids(run_dir: Path) -> Optional[set[str]]:
    manifest = load_run_artifact_authority(run_dir)
    if manifest is None:
        return None
    records = manifest.get("per_step_records")
    records = records if isinstance(records, list) else []
    evidence_ids: set[str] = set()
    for record in current_successful_step_records(records):
        step_id = str(record.get("step_id") or "").strip()
        lowered = step_id.lower()
        if not step_id or any(
            token in lowered for token in ("figure", "repair", "audit")
        ):
            continue
        summary = record.get("step_summary")
        summary = summary if isinstance(summary, dict) else {}
        has_primary_model = any(
            summary.get(key)
            for key in (
                "primary_model",
                "primary_adjusted_association",
                "primary_association",
            )
        )
        textual_primary_model = "primary" in lowered and any(
            token in lowered for token in ("association", "model", "effect")
        )
        if has_primary_model or textual_primary_model:
            evidence_ids.update(
                str(evidence_id)
                for evidence_id in record.get("evidence_ids") or []
                if str(evidence_id).strip()
            )
    return evidence_ids


def _load_regression_covariates(run_dir: Path) -> List[str]:
    """Best-effort: the model's adjustment set (used for the gold-free
    overadjustment / leakage checks).

    Delegates to the shared reader so the post-hoc backstop and the mid-flight
    auditor agree on the adjustment set: a per-covariate coefficient table when
    the run wrote one (``primary_association.csv`` / ``model_coefficients.csv`` /
    ``regression_results.csv``, matched by content not filename), else the
    covariate set recovered from the analysis code — so a run that reports only a
    model-level OR summary is not invisible to the check. Missing/malformed
    sources degrade to an empty list.
    """
    primary_evidence_ids = _current_primary_model_evidence_ids(run_dir)
    if primary_evidence_ids is not None:
        current_paths = current_run_evidence_paths(
            run_dir,
            evidence_ids=primary_evidence_ids,
        )
        return read_adjustment_covariates(run_dir, files=current_paths or [])
    return read_adjustment_covariates(run_dir)


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


def _find_run_artifacts(run_dir: Path, *substrings: str) -> List[Path]:
    """All emitted files whose name contains any of ``substrings`` (case-folded).

    Searches the run root, the ``evidence/`` store (where files carry a
    ``<kind>_<hash>__<name>`` prefix), and ``steps/*/outputs/``. Newest first so
    a re-run's artifact wins. Used to locate validity metrics the agent emitted
    under its own filename rather than a fixed contract name.
    """
    subs = tuple(s.lower() for s in substrings)
    hits: List[Path] = []
    current_paths = current_run_evidence_paths(run_dir)
    candidates = (
        current_paths
        if current_paths is not None
        else (
            list(run_dir.glob("*"))
            + list(run_dir.glob("evidence/*"))
            + list(run_dir.glob("steps/*/outputs/*"))
        )
    )
    for path in candidates:
        if path.is_file() and any(s in path.name.lower() for s in subs):
            hits.append(path)
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits


def _num(value: object) -> Optional[float]:
    """Coerce to float, rejecting bools/NaN/non-numeric — else ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN check


def _load_cluster_validity(run_dir: Path) -> Dict[str, object]:
    """Best-effort read of a phenotype run's emitted validity metrics.

    Normalises to ``{silhouette, n_clusters, min_cluster_fraction, algorithm}``
    from whatever the run actually emitted. An explicit validity file wins; if
    absent we recover the same fields from the agent's own clustering artifacts
    (``clustering_algorithm_details.json`` selection record, ``cluster_sizes``
    table, ``*selection_metrics*`` / ``*quality_and_stability*`` tables) — so a
    run is not invisible to the impartial degeneracy check merely because it
    named its outputs differently. Returns ``{}`` when nothing is found, so the
    dimension stays honestly *unscored* rather than inventing a verdict.
    """
    # 1. An explicit, contract-named validity file is authoritative when present.
    for name in (
        "cluster_validity.json",
        "clustering_validity.json",
        "validity_metrics.json",
    ):
        for path in _find_run_artifacts(run_dir, name):
            if path.name.split("__", 1)[-1] != name:
                continue
            doc = _load_json(path)
            if doc:
                return doc

    # 2. Recover the same fields from the agent's own emitted artifacts.
    out: Dict[str, object] = {}

    for p in _find_run_artifacts(run_dir, "clustering_algorithm_details"):
        details = _load_json(p)
        if not isinstance(details, dict):
            continue
        if details.get("algorithm"):
            out.setdefault("algorithm", str(details["algorithm"]))
        if _num(details.get("selected_k")) is not None:
            out.setdefault("n_clusters", int(_num(details["selected_k"])))
        if _num(details.get("selected_silhouette_score")) is not None:
            out.setdefault("silhouette", _num(details["selected_silhouette_score"]))
        # H3-style: a per-k selection_metrics list with the selected row flagged.
        sel = details.get("selection_metrics")
        if isinstance(sel, list):
            chosen = next(
                (r for r in sel if isinstance(r, dict) and r.get("selected")), None
            )
            if isinstance(chosen, dict):
                if _num(chosen.get("silhouette_score")) is not None:
                    out.setdefault("silhouette", _num(chosen["silhouette_score"]))
                if _num(chosen.get("min_cluster_pct")) is not None:
                    out.setdefault(
                        "min_cluster_fraction",
                        _num(chosen["min_cluster_pct"]) / 100.0,
                    )
        if out.get("silhouette") is not None and "n_clusters" in out:
            break

    # min_cluster_fraction from the emitted cluster-sizes table if still unknown.
    if "min_cluster_fraction" not in out:
        for p in _find_run_artifacts(run_dir, "cluster_sizes"):
            try:
                with p.open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
            except (OSError, ValueError):
                continue
            if not rows:
                continue
            pcts = [_num(r.get("percentage")) for r in rows]
            pcts = [x for x in pcts if x is not None]
            ns = [_num(r.get("n")) or _num(r.get("count")) for r in rows]
            ns = [x for x in ns if x is not None]
            if pcts:
                fracs = sorted(x / 100.0 for x in pcts)
                out["cluster_fractions"] = fracs
                out["min_cluster_fraction"] = fracs[0]
                out["max_cluster_fraction"] = fracs[-1]
                out.setdefault("n_clusters", len(rows))
                break
            if ns and sum(ns) > 0:
                total = sum(ns)
                fracs = sorted(x / total for x in ns)
                out["cluster_fractions"] = fracs
                out["min_cluster_fraction"] = fracs[0]
                out["max_cluster_fraction"] = fracs[-1]
                out.setdefault("n_clusters", len(rows))
                break

    return out


# Internal indices whose geometry is only valid for distance/centroid-based
# partitions. For model-based clustering (mixtures) the native fit criterion is
# the likelihood/BIC, so a poor silhouette is informative but not a definitional
# failure — it surfaces as a caution, never a Fail.
_DISTANCE_BASED_CLUSTERING = (
    "kmeans",
    "k-means",
    "agglomerative",
    "hierarchical",
    "ward",
    "spectral",
    "dbscan",
    "kmedoids",
    "k-medoids",
    "birch",
)
_MODEL_BASED_CLUSTERING = (
    "gaussianmixture",
    "gmm",
    "mixture",
    "bayesian",
    "lcga",
    "gbtm",
)


def _is_distance_based(algorithm: Optional[str]) -> Optional[bool]:
    """True=distance/centroid, False=model-based, None=unknown.

    Unknown defaults to treating silhouette conservatively as a *caution* (we do
    not know the partition geometry), never a Fail.
    """
    a = (algorithm or "").lower()
    if any(k in a for k in _MODEL_BASED_CLUSTERING):
        return False
    if any(k in a for k in _DISTANCE_BASED_CLUSTERING):
        return True
    return None


# A single cluster holding at least this share of the cohort is the "one
# dominant cluster" the degeneracy guard is named for. Below it, several
# substantial groups coexist and a sub-1% cluster is a rare phenotype, not a
# collapsed partition.
_DOMINANT_CLUSTER_SHARE = 0.80
# A cluster clearing this floor counts as a substantial, characterisable group.
_SUBSTANTIAL_CLUSTER_FLOOR = 0.01


def _partition_is_degenerate(doc: Dict[str, object]) -> bool:
    """True only when a sub-1% cluster sits inside a genuinely collapsed partition.

    A small cluster is an OBJECTIVE failure only when it is the symptom of "one
    dominant cluster plus an outlier pocket" — i.e. effectively a single group.
    When several substantial clusters coexist (the agent chose k by a principled
    criterion and got a balanced solution with one rare group), the small cluster
    is a defensible analytical outcome, not a broken partition, so it is a caution
    rather than a Fail. We decide this from the full size distribution when
    available (count of clusters clearing the 1% floor, or the largest share);
    when only ``min_cluster_fraction`` was recoverable and the distribution cannot
    be reconstructed, we stay fail-closed on the sub-1% signal.
    """
    fracs = doc.get("cluster_fractions")
    if isinstance(fracs, list):
        substantial = sum(
            1
            for f in fracs
            if _num(f) is not None and _num(f) >= _SUBSTANTIAL_CLUSTER_FLOOR
        )
        return substantial <= 1
    max_frac = _num(doc.get("max_cluster_fraction"))
    if max_frac is not None:
        return max_frac >= _DOMINANT_CLUSTER_SHARE
    # No distribution to disambiguate: a contract validity file declared a
    # near-empty group with nothing else to weigh, so keep the firm signal.
    return True


def _phenotype_validity_errors(run_dir: Path, kind: str) -> List[str]:
    """Impartial, deterministic objective-error check for clustering/trajectory.

    Flags only the *degenerate* end — a partition that is objectively broken
    regardless of analytical taste or clustering family: a single group, a
    near-empty group, or (for a distance/centroid method only) a non-positive
    silhouette meaning the geometry is no better than chance. It never imposes a
    "good enough" threshold on a valid solution; softer process gaps (which
    selection criterion, how many models compared) belong to the reporting
    checklist, not a validity Fail. A non-positive silhouette under a *model-
    based* method is a caution, not an error (see ``_phenotype_validity_cautions``)
    because silhouette is not that method's fit criterion. Returns ``[]`` (→
    honest NA) when the run emitted no validity metrics.
    """
    if str(kind) not in _PHENOTYPE_KINDS:
        return []
    doc = _load_cluster_validity(run_dir)
    if not doc:
        return []
    errs: List[str] = []
    k = _num(doc.get("n_clusters"))
    if k is None:
        k = _num(doc.get("n_classes"))
    if k is not None and k < 2:
        errs.append(f"single-group solution (k={int(k)}): no subphenotypes separated")
    frac = _num(doc.get("min_cluster_fraction"))
    if frac is not None and 0 <= frac < 0.01 and _partition_is_degenerate(doc):
        errs.append(
            f"near-empty group ({frac * 100:.2f}% of cohort): "
            "degenerate partition — one dominant cluster plus an outlier pocket, "
            "not separated subphenotypes"
        )
    # A non-positive silhouette means points are, on average, closer to a
    # neighbouring cluster than their own — objectively poor separation. We fail
    # it for distance/centroid methods and for an UNKNOWN family (flag firmly);
    # only an explicitly *model-based* family is carved out to a caution, because
    # there silhouette is not the fit criterion (see the cautions helper).
    sil = _num(doc.get("silhouette"))
    if (
        sil is not None
        and sil <= 0
        and _is_distance_based(doc.get("algorithm")) is not False
    ):
        algo = doc.get("algorithm") or "unspecified method"
        errs.append(
            f"degenerate clustering: silhouette={sil:.3f} ≤ 0 under {algo} — "
            "groups no better than chance separation"
        )
    return errs


def _phenotype_validity_cautions(run_dir: Path, kind: str) -> List[str]:
    """Impartial caution-tier signals for clustering/trajectory (never gate).

    A non-positive silhouette under a *model-based* method (e.g. a Gaussian
    mixture, whose fit criterion is the likelihood/BIC, not silhouette) or under
    an *unknown* clustering family is surfaced for human review but does NOT fail
    the dimension — silhouette geometry may simply be the wrong lens for that
    partition. Returns ``[]`` when nothing applies.
    """
    if str(kind) not in _PHENOTYPE_KINDS:
        return []
    doc = _load_cluster_validity(run_dir)
    if not doc:
        return []
    cautions: List[str] = []
    sil = _num(doc.get("silhouette"))
    if (
        sil is not None
        and sil <= 0
        and _is_distance_based(doc.get("algorithm")) is False
    ):
        algo = doc.get("algorithm") or "unspecified method"
        cautions.append(
            f"weak cluster separation: silhouette={sil:.3f} ≤ 0 under {algo}; "
            "silhouette is not this family's fit criterion, so verify separation "
            "against the criterion actually used (e.g. BIC/likelihood) before "
            "interpreting the classes"
        )
    # A sub-1% cluster inside an otherwise multi-cluster partition is a rare
    # group, not a degeneracy (that case is handled as an error). Surface it for
    # human review without gating: report its size and verify stability before
    # interpreting it as a distinct subphenotype.
    frac = _num(doc.get("min_cluster_fraction"))
    if (
        frac is not None
        and 0 <= frac < _SUBSTANTIAL_CLUSTER_FLOOR
        and not _partition_is_degenerate(doc)
    ):
        fracs = doc.get("cluster_fractions")
        n_substantial = (
            sum(
                1
                for f in fracs
                if _num(f) is not None and _num(f) >= _SUBSTANTIAL_CLUSTER_FLOOR
            )
            if isinstance(fracs, list)
            else None
        )
        detail = (
            f" alongside {n_substantial} clusters ≥1%"
            if n_substantial is not None
            else ""
        )
        cautions.append(
            f"small cluster ({frac * 100:.2f}% of cohort){detail}: a rare group "
            "within a multi-cluster partition, not a degenerate solution — report "
            "its size and verify stability before interpreting it as a distinct "
            "subphenotype"
        )
    return cautions


def _locked_cohort_path(run_dir: Path) -> Optional[Path]:
    """The materialised analysis cohort a run actually modelled on."""
    for name in ("cohort_analysis.parquet", "cohort.parquet"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _deliberate_block_reason(run_dir: Path) -> Optional[str]:
    """A short reason if a step *deliberately* recorded a non-execution /
    blocked-modeling status (the agent chose not to run), vs a hard crash.

    Distinguishing a chosen block from a crash matters: a crash is a code
    failure (already reflected in ``code``); a deliberate block on viable data is
    a self-paralysis failure mode that would otherwise be invisible.
    """
    authority = load_run_artifact_authority(run_dir)
    if authority is not None:
        records = authority.get("per_step_records")
        records = records if isinstance(records, list) else []
        summaries = [
            summary
            for record in current_successful_step_records(records)
            if isinstance((summary := record.get("step_summary")), dict)
        ]
    else:
        paths = list(run_dir.glob("steps/*/outputs/step_summary.json")) + list(
            run_dir.glob("steps/*/step_summary.json")
        )
        summaries = [_load_json(path) for path in paths]
    for doc in summaries:
        if not isinstance(doc, dict):
            continue
        signal = step_summary_block_signal(doc)
        if signal:
            return signal
    return None


def detect_self_inflicted_block(
    run_dir: Path,
    gates: Dict[str, object],
    *,
    outcome: Optional[str] = None,
) -> Optional[str]:
    """Impartial, deterministic check: did the run block its primary deliverable
    on a task-VIABLE cohort?

    Reports facts only — that the locked analysis cohort had enough rows, both
    outcome classes with a non-trivial minority (when ``outcome`` is supplied),
    and well-populated predictor columns — yet the run chose a
    non-execution/blocked status with no data-driven cause. It never asserts
    which model should have been fit; it only separates an *agent self-paralysis*
    failure (a solvable task the run fumbled) from a *genuinely infeasible* task,
    so a ``diagnostic_only`` verdict is not silently read as the latter. Fires
    conservatively: stays silent (→ no claim) on a hard crash, a small/eventless
    cohort, or when the cohort cannot be read.
    """
    if bool(gates.get("execution_complete")):
        return None
    reason = _deliberate_block_reason(run_dir)
    if not reason:
        return None  # a hard crash / genuine failure, not a deliberate self-block
    path = _locked_cohort_path(run_dir)
    if path is None:
        return None
    try:
        import pandas as pd  # lazy: keep the scorecard module import light

        df = pd.read_parquet(path)
    except Exception:
        return None

    verdict = assess_cohort_viability(df, outcome=outcome)
    if not verdict.viable:
        return None  # too small / no outcome variation / too few usable predictors

    return (
        "execution block appears self-inflicted: the locked analysis cohort is "
        "task-viable (" + verdict.note + ") yet the run recorded a "
        f'non-execution/blocked status ("{reason}") with no data-driven cause — '
        "an agent self-paralysis failure mode, distinct from an infeasible task"
    )


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

    # Score the plan the agent ACTUALLY EXECUTED, i.e. the latest revision, not
    # the initial analysis_plan.json. The replanner improves the plan in place
    # (e.g. adds a Table-1 / cohort-characteristics step in a later revision);
    # reading only the base plan undercounts those improvements.
    plan_path = run_dir / "analysis_plan.json"
    _best_rev = -1
    for _rev in run_dir.glob("analysis_plan_revision_*.json"):
        try:
            _n = int(_rev.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if _n > _best_rev:
            _best_rev, plan_path = _n, _rev
    plan_doc = _load_json(plan_path)
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
    validity_cautions.extend(_phenotype_validity_cautions(run_dir, task.kind))

    self_inflicted_block = detect_self_inflicted_block(
        run_dir, gates, outcome=outcome_concept
    )

    # Gold-free, kind-routed VALUE-based validity signals (the Fig.3 baseline).
    # Used only when no objective validity error fired — an error still caps Fail.
    _validity_signals = assess_validity_signals(task.kind, run_dir)

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
        validity_signals=_validity_signals,
        validity_positive_subscore=validity_positive_subscore(_validity_signals),
        locked_reference_frozen=locked_reference_frozen,
        plan_illegal=plan_illegal,
        self_inflicted_block=self_inflicted_block,
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
    "detect_self_inflicted_block",
    "score_run",
    "score_run_from_dir",
]
