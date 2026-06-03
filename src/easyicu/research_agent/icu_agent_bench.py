"""Typed scaffolding for the EasyICU agent **evaluation protocol**.

================================================================================
INTERNAL EVALUATION PROTOCOL — NOT A PUBLISHED BENCHMARK
================================================================================
This module supports the EasyICU research agent's **internal evaluation
protocol** (see ``02_npj_Digital_Medicine/tier_evaluation_protocol_20260527.md``).
It is **not** a community benchmark and the manuscript does not describe it
that way. The historical class name ``ICUAgentBench`` is retained only for
backwards compatibility with on-disk task JSON, existing run directories, and
downstream test fixtures; references in code or paper-facing text should call
this the **EasyICU evaluation protocol** or **evaluation scaffold**.

Specifically:

* The default suite contains exactly **one** partially frozen task
  (``synthetic_cohort_completeness_qc``) with a checkable gold answer; all
  other tasks are descriptive specifications, not adjudicated benchmark
  items.
* No external adjudication, peer-reviewed task curation, or frozen public
  task-suite manifest exists yet.
* The protocol is layered (Tier 1 = deterministic checks, Tier 2 = LLM-jury
  process audit, Tier 3 = clinician spot-check). Only Tier 1 has been
  executed for the present submission; Tier 2 and Tier 3 are outlined in
  the Supplementary Methods but not run.

DO NOT cite this module's outputs as "benchmark scores" in manuscripts,
posters, or grant applications. WriterAgent / ManuscriptAgent must surface
results as ``Tier 1 deterministic-gate outcomes`` of the EasyICU evaluation
protocol, not as benchmark scores.

When (and only when) an external, peer-reviewed, frozen task suite is added
with adjudicated gold answers, this banner will be revised and the
``ICUAgentBench`` class names may be deprecated in favour of explicitly
``EvaluationProtocol*`` naming.
================================================================================

The grading layer (``grade_bench_task`` + ``aggregate_bench_report``) is
deliberately a **pure-function** layer that does *not* depend on the
pipeline, agents, or LLM. This keeps the protocol machinery exercisable
from unit tests against deterministic fixtures, and lets the same grader
be reused by an external runner if one is added in the future.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


TaskCategory = Literal["self_check", "evaluation"]
"""Coarse classification used to keep machine-checkable self-test tasks
out of the suite-level *evaluation* headline.

* ``"self_check"`` — tasks whose gold answers are derived from in-repo
  deterministic fixtures (e.g. the ``synthetic_cohort``). Their job is
  to exercise the grading machinery in CI; they MUST NOT be averaged
  into the headline correctness number a paper reports for the agent.
* ``"evaluation"`` — tasks intended to grade an actual agent run on
  ICU data (real or synthetic-but-agent-driven). Aggregate reports
  default to this category.
"""


BenchTaskKind = Literal[
    "cohort_extraction",
    "aki_staging",
    "sofa_extraction",
    "ventilation_duration",
    "sepsis_onset",
    "mortality_prediction",
    "competing_risk_analysis",
    "survival_analysis",
    "longitudinal_trajectory_analysis",
    "cross_database_replication",
]


class ICUAgentBenchMetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: bool = True
    reproducibility: bool = True
    provenance_completeness: bool = True
    icu_semantic_validity: bool = True
    hallucination_rate: bool = True
    statistical_validity: bool = True
    execution_success_rate: bool = True


class ICUAgentBenchNumericBound(BaseModel):
    """Inclusive ``[lower, upper]`` bound for an observed numeric metric.

    Either side may be ``None`` to express an open bound. ``grade_bench_task``
    treats a metric as *in-bounds* when ``lower <= value <= upper``.
    """

    model_config = ConfigDict(extra="forbid")

    lower: Optional[float] = None
    upper: Optional[float] = None

    def contains(self, value: float) -> bool:
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


class ICUAgentBenchGoldAnswer(BaseModel):
    """Per-task expected behavior used by :func:`grade_bench_task`.

    A task with a gold answer is *checkable*: the grader scores an observed
    run against numeric bounds, required guardrail warnings, and forbidden
    output substrings — no human adjudication required. Tasks without a
    gold answer remain descriptive specifications only.
    """

    model_config = ConfigDict(extra="forbid")

    # metric_name -> inclusive numeric bound on the observed value.
    numeric_targets: Dict[str, ICUAgentBenchNumericBound] = Field(default_factory=dict)

    # Guardrail tags that MUST appear in the observed warnings.
    required_warnings: List[str] = Field(default_factory=list)

    # Substrings that MUST NOT appear in any observed output.
    forbidden_outputs: List[str] = Field(default_factory=list)

    # Free-form description of how this gold answer was derived (which
    # data, what statistics, what tolerance, etc.). Goes into reports.
    derivation: str = ""

    # Optional fixture name (e.g. ``"synthetic_cohort"``) for tasks that
    # can run against a deterministic in-repo fixture. Real-data tasks
    # leave this empty.
    data_fixture: Optional[str] = None


class ICUAgentBenchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    kind: BenchTaskKind
    title: str
    objective: str
    expected_outputs: List[str] = Field(default_factory=list)
    semantic_guardrails: List[str] = Field(default_factory=list)
    evaluation_notes: List[str] = Field(default_factory=list)
    target_databases: List[str] = Field(default_factory=list)
    gold_answer_status: Literal["planned", "frozen"] = "planned"
    gold_answer: Optional[ICUAgentBenchGoldAnswer] = None
    difficulty: Literal["basic", "intermediate", "advanced"] = "intermediate"
    # See ``TaskCategory`` docstring. Defaults to ``"evaluation"`` so
    # any existing task added without this field keeps appearing in
    # the headline aggregate — the burden of declaring a task as a
    # self-test sits with whoever adds the self-test, not with
    # callers of older tasks.
    category: TaskCategory = "evaluation"


class ICUAgentBenchSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.icu_agent_bench/1"
    name: str = "ICUAgentBench"
    maturity: Literal["prototype", "frozen_partial"] = "prototype"
    tasks: List[ICUAgentBenchTask] = Field(default_factory=list)
    metrics: ICUAgentBenchMetricSpec = Field(default_factory=ICUAgentBenchMetricSpec)

    def frozen_task_ids(self) -> List[str]:
        """Task ids that ship with a checkable gold answer."""
        return [t.task_id for t in self.tasks if t.gold_answer is not None]

    def evaluation_task_ids(self) -> List[str]:
        """Task ids in the ``evaluation`` category (the headline-relevant set)."""
        return [t.task_id for t in self.tasks if t.category == "evaluation"]

    def self_check_task_ids(self) -> List[str]:
        """Task ids in the ``self_check`` category (CI fixtures, not headline)."""
        return [t.task_id for t in self.tasks if t.category == "self_check"]

    def category_by_task(self) -> Dict[str, TaskCategory]:
        """Map task_id -> category for grader/aggregator lookups."""
        return {t.task_id: t.category for t in self.tasks}


class ICUAgentBenchTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    run_id: Optional[str] = None
    correctness: Optional[float] = None
    reproducibility: Optional[float] = None
    provenance_completeness: Optional[float] = None
    icu_semantic_validity: Optional[float] = None
    hallucination_rate: Optional[float] = None
    statistical_validity: Optional[float] = None
    execution_success_rate: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class ICUAgentBenchReport(BaseModel):
    """Suite-level result bundle with category-aware aggregates.

    Important: ``aggregate`` is the **evaluation** headline only. Self-check
    tasks (machine-test fixtures that should always pass) live in
    ``self_check_aggregate`` so they cannot inflate a paper's reported
    correctness number. Per-task results in ``task_results`` keep every
    row regardless of category — call sites can still slice/inspect.
    """

    model_config = ConfigDict(extra="forbid")

    suite_name: str = "ICUAgentBench"
    task_results: List[ICUAgentBenchTaskResult] = Field(default_factory=list)
    aggregate: Dict[str, float] = Field(default_factory=dict)
    self_check_aggregate: Dict[str, float] = Field(default_factory=dict)


def default_icu_agent_bench_suite() -> ICUAgentBenchSuite:
    return ICUAgentBenchSuite(
        tasks=[
            ICUAgentBenchTask(
                task_id="cross_db_sofa_mortality",
                kind="cross_database_replication",
                title="SOFA-mortality replication across EasyICU databases",
                objective=(
                    "Reproduce the same SOFA-mortality hypothesis across "
                    "EasyICU-supported public ICU databases using standardized "
                    "concept extraction."
                ),
                expected_outputs=[
                    "concept availability matrix",
                    "per-database effect estimates",
                    "cross-database forest plot",
                    "degraded-database sensitivity analysis",
                ],
                semantic_guardrails=[
                    "Detect and warn on blocked or degraded databases before modeling.",
                    "Use the same EasyICU concept definitions across databases.",
                    "Do not silently omit a requested database from the replication grid.",
                ],
                evaluation_notes=[
                    "Gold effect intervals are planned but not frozen in the prototype suite.",
                ],
                target_databases=["mimic", "miiv", "eicu", "aumc", "hirid", "sic"],
                difficulty="advanced",
            ),
            ICUAgentBenchTask(
                task_id="cohort_extraction",
                kind="cohort_extraction",
                title="Deterministic ICU cohort extraction",
                objective="Reconstruct a cohort with explicit inclusion/exclusion criteria and preserved provenance.",
                expected_outputs=[
                    "cohort definition summary",
                    "provenance bundle",
                    "table one",
                ],
                semantic_guardrails=[
                    "No raw SQL exposure to the agent.",
                    "All concepts must resolve through the EasyICU registry.",
                ],
                difficulty="basic",
            ),
            ICUAgentBenchTask(
                task_id="aki_staging",
                kind="aki_staging",
                title="AKI staging within a deterministic window",
                objective="Derive AKI staging using explicit temporal semantics and auditable staging logic.",
                expected_outputs=["AKI staging table", "time-window provenance"],
                semantic_guardrails=[
                    "Respect ICU admission anchors and post-admission windows.",
                    "Do not use post-outcome leakage features.",
                ],
            ),
            ICUAgentBenchTask(
                task_id="sofa_extraction",
                kind="sofa_extraction",
                title="SOFA extraction and component audit",
                objective="Extract total and component SOFA values while preserving ordinal-score semantics.",
                expected_outputs=[
                    "SOFA summary table",
                    "component missingness audit",
                    "SOFA figure",
                ],
                semantic_guardrails=[
                    "Do not average ordinal SOFA components.",
                    "Surface sofa==0/component-missingness anomalies when present.",
                ],
            ),
            ICUAgentBenchTask(
                task_id="ventilation_duration",
                kind="ventilation_duration",
                title="Ventilation duration derivation",
                objective="Compute ventilation duration with explicit episode resolution and censoring notes.",
                expected_outputs=["duration table", "episode provenance"],
            ),
            ICUAgentBenchTask(
                task_id="sepsis_onset",
                kind="sepsis_onset",
                title="Sepsis onset timing",
                objective="Resolve sepsis onset using deterministic event definitions and temporal alignment.",
                expected_outputs=["onset cohort summary", "timing audit"],
            ),
            ICUAgentBenchTask(
                task_id="mortality_prediction",
                kind="mortality_prediction",
                title="Mortality prediction",
                objective="Train and evaluate an ICU mortality prediction workflow with calibration-aware reporting.",
                expected_outputs=[
                    "performance table",
                    "calibration evidence",
                    "publication figure",
                ],
                semantic_guardrails=[
                    "Check train/test leakage and calibration.",
                    "Warn on insufficient event count or missing validation.",
                ],
                difficulty="advanced",
            ),
            ICUAgentBenchTask(
                task_id="competing_risk_analysis",
                kind="competing_risk_analysis",
                title="Competing-risk analysis",
                objective="Evaluate outcomes with explicit competing-risk warnings and event semantics.",
                expected_outputs=[
                    "event table",
                    "competing-risk note",
                    "effect summary",
                ],
                semantic_guardrails=[
                    "Warn when a simple binary model ignores competing events."
                ],
                difficulty="advanced",
            ),
            ICUAgentBenchTask(
                task_id="survival_analysis",
                kind="survival_analysis",
                title="Time-to-event survival analysis",
                objective="Produce a survival workflow with PH checks and survival-curve provenance.",
                expected_outputs=["KM curve", "Cox summary", "PH diagnostics"],
                semantic_guardrails=["Warn when Cox PH assumptions are untested."],
                difficulty="advanced",
            ),
            ICUAgentBenchTask(
                task_id="longitudinal_trajectory_analysis",
                kind="longitudinal_trajectory_analysis",
                title="Longitudinal trajectory analysis",
                objective="Analyse repeated ICU measurements with explicit temporal resolution and trajectory outputs.",
                expected_outputs=[
                    "trajectory summary",
                    "trajectory figure",
                    "cluster/stability notes",
                ],
                semantic_guardrails=[
                    "Preserve longitudinal semantics rather than flattening prematurely.",
                    "Document temporal resolution and alignment decisions.",
                ],
                difficulty="advanced",
            ),
            # ---- Frozen self-checkable tasks --------------------------------
            # These tasks use the in-repo ``synthetic_cohort`` fixture so the
            # grading machinery can be exercised end-to-end without any
            # external ICU database. The bounds were measured from
            # ``np.random.default_rng(7), n=800`` and widened by ~5% to
            # tolerate floating-point drift across platforms.
            ICUAgentBenchTask(
                task_id="synthetic_cohort_completeness_qc",
                kind="sofa_extraction",
                category="self_check",
                title="SOFA-2 component-completeness QC on synthetic cohort",
                objective=(
                    "Surface the outcome-blind component-completeness signal in "
                    "the synthetic cohort fixture and confirm descriptive "
                    "statistics fall within frozen golden bounds. Acts as a "
                    "self-checkable smoke task for ICUAgentBench grading machinery."
                ),
                expected_outputs=[
                    "descriptive statistics",
                    "component completeness note",
                ],
                semantic_guardrails=[
                    "Surface SOFA-2 component completeness before outcome modeling.",
                    "Do not silently impute missing SOFA-2.",
                ],
                target_databases=[],  # synthetic; no real DB
                difficulty="basic",
                gold_answer_status="frozen",
                gold_answer=ICUAgentBenchGoldAnswer(
                    data_fixture="synthetic_cohort",
                    numeric_targets={
                        "n_rows": ICUAgentBenchNumericBound(lower=800, upper=800),
                        "death_rate": ICUAgentBenchNumericBound(lower=0.10, upper=0.15),
                        "sofa2_low_component_frac": ICUAgentBenchNumericBound(lower=0.08, upper=0.11),
                        "sofa2_mean": ICUAgentBenchNumericBound(lower=6.0, upper=6.5),
                        "vaso_rate": ICUAgentBenchNumericBound(lower=0.40, upper=0.48),
                    },
                    required_warnings=["component_completeness_qc"],
                    forbidden_outputs=["silently imputed"],
                    derivation=(
                        "Bounds measured from synthetic_cohort fixture "
                        "(numpy.random.default_rng(7), n=800) and widened "
                        "by approximately 5% to tolerate floating-point "
                        "drift across platforms."
                    ),
                ),
            ),
            ICUAgentBenchTask(
                task_id="synthetic_cohort_table_one",
                kind="cohort_extraction",
                category="self_check",
                title="Table-One descriptive statistics on synthetic cohort",
                objective=(
                    "Emit a table-one style descriptive summary of the "
                    "synthetic cohort with frozen percentile and mean "
                    "bounds. Exercises the grader's ``cohort_extraction`` "
                    "kind without depending on any real database."
                ),
                expected_outputs=[
                    "table one",
                    "cohort definition summary",
                ],
                semantic_guardrails=[
                    "Compute medians and IQRs, not just means.",
                    "Never include post-outcome features in baseline summaries.",
                ],
                target_databases=[],
                difficulty="basic",
                gold_answer_status="frozen",
                gold_answer=ICUAgentBenchGoldAnswer(
                    data_fixture="synthetic_cohort",
                    numeric_targets={
                        "age_median": ICUAgentBenchNumericBound(lower=62.0, upper=65.0),
                        "age_p25": ICUAgentBenchNumericBound(lower=52.0, upper=56.0),
                        "age_p75": ICUAgentBenchNumericBound(lower=71.0, upper=75.0),
                        "los_median": ICUAgentBenchNumericBound(lower=3.5, upper=4.3),
                        "lact_mean": ICUAgentBenchNumericBound(lower=2.9, upper=3.6),
                        "creat_mean": ICUAgentBenchNumericBound(lower=1.3, upper=1.6),
                    },
                    required_warnings=[],
                    forbidden_outputs=["post-outcome", "missing_data_imputed"],
                    derivation=(
                        "Bounds measured from synthetic_cohort fixture "
                        "with ~5% padding. The forbidden substrings catch "
                        "two common table-one mistakes: (a) leaking the "
                        "outcome into baseline columns; (b) silently "
                        "imputing missing baseline values."
                    ),
                ),
            ),
            ICUAgentBenchTask(
                task_id="synthetic_cohort_stratified_sofa_mortality",
                kind="survival_analysis",
                category="self_check",
                title="Stratified SOFA-2 mortality (excluding zero-fraction)",
                objective=(
                    "After excluding sofa2==0 patients (which conflate "
                    "missing values with truly low scores), demonstrate a "
                    "monotonically increasing mortality signal across SOFA-2 "
                    "tertiles. This task locks the *conditional* SOFA-mortality "
                    "association — the unconditional one is reversed by the "
                    "missing-data-as-zero anomaly captured in the audit task."
                ),
                expected_outputs=[
                    "stratified mortality table",
                    "exclusion rationale",
                ],
                semantic_guardrails=[
                    "Exclude or flag sofa2==0 patients before stratifying.",
                    "Report tertile boundaries explicitly, not magic numbers.",
                ],
                target_databases=[],
                difficulty="intermediate",
                gold_answer_status="frozen",
                gold_answer=ICUAgentBenchGoldAnswer(
                    data_fixture="synthetic_cohort",
                    numeric_targets={
                        "death_low_tertile_excl_zero": ICUAgentBenchNumericBound(
                            lower=0.03, upper=0.09
                        ),
                        "death_high_tertile_excl_zero": ICUAgentBenchNumericBound(
                            lower=0.17, upper=0.27
                        ),
                        "death_delta_hi_minus_lo": ICUAgentBenchNumericBound(
                            lower=0.10, upper=0.22
                        ),
                    },
                    required_warnings=["excluded sofa2==0 patients"],
                    forbidden_outputs=[
                        "pooled sofa2 directly without missingness audit",
                    ],
                    derivation=(
                        "Bounds from synthetic_cohort after dropping "
                        "sofa2==0 rows and tertile-splitting the remainder "
                        "by SOFA-2 quantiles. The delta-hi-minus-lo target "
                        "guards against a model that erases stratification "
                        "by averaging across the whole cohort."
                    ),
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Grading layer
# ---------------------------------------------------------------------------


def grade_bench_task(
    task: ICUAgentBenchTask,
    *,
    observed_metrics: Optional[Dict[str, float]] = None,
    observed_warnings: Optional[Sequence[str]] = None,
    observed_outputs: Optional[Sequence[str]] = None,
    run_id: Optional[str] = None,
) -> ICUAgentBenchTaskResult:
    """Score an observed run against the task's gold answer.

    Pure function — no I/O, no LLM, no pipeline state. Returns a
    :class:`ICUAgentBenchTaskResult` with the metrics that this gold
    answer can adjudicate; metrics requiring external judgment
    (``reproducibility``, ``icu_semantic_validity``,
    ``statistical_validity``) are left ``None`` for higher layers to
    fill in.

    Score conventions:

    * ``correctness`` — fraction of ``numeric_targets`` whose observed
      value fell inside the declared bound. ``None`` when the task has
      no gold answer or declares no numeric targets.
    * ``provenance_completeness`` — fraction of ``required_warnings``
      that appeared in ``observed_warnings`` as substrings.
    * ``hallucination_rate`` — ``1.0`` if any ``forbidden_outputs``
      substring appears in any observed output, else ``0.0``.
    * ``execution_success_rate`` — ``1.0`` when ``observed_metrics`` is
      provided (even if empty), ``0.0`` otherwise. Treat this as
      "the run produced *something* to score".
    """

    observed_metrics = observed_metrics or {}
    observed_warnings_list = list(observed_warnings or [])
    observed_outputs_list = list(observed_outputs or [])

    correctness: Optional[float] = None
    provenance: Optional[float] = None
    hallucination: Optional[float] = None
    execution_success: Optional[float] = 1.0 if observed_metrics else 0.0
    notes: List[str] = []

    gold = task.gold_answer
    if gold is not None:
        # ---- correctness over numeric targets ------------------------
        if gold.numeric_targets:
            hits = 0
            total = 0
            for name, bound in gold.numeric_targets.items():
                total += 1
                if name not in observed_metrics:
                    notes.append(f"missing metric: {name}")
                    continue
                if bound.contains(observed_metrics[name]):
                    hits += 1
                else:
                    notes.append(
                        f"out-of-bound metric: {name}={observed_metrics[name]} "
                        f"not in [{bound.lower}, {bound.upper}]"
                    )
            correctness = hits / total if total else None

        # ---- required guardrail warnings -----------------------------
        if gold.required_warnings:
            joined = "\n".join(observed_warnings_list)
            hits = sum(1 for w in gold.required_warnings if w in joined)
            provenance = hits / len(gold.required_warnings)
            for w in gold.required_warnings:
                if w not in joined:
                    notes.append(f"missing required warning: {w}")

        # ---- forbidden output substrings -----------------------------
        if gold.forbidden_outputs:
            joined = "\n".join(observed_outputs_list)
            tripped = [s for s in gold.forbidden_outputs if s in joined]
            hallucination = 1.0 if tripped else 0.0
            for s in tripped:
                notes.append(f"forbidden output substring observed: {s}")

    return ICUAgentBenchTaskResult(
        task_id=task.task_id,
        run_id=run_id,
        correctness=correctness,
        provenance_completeness=provenance,
        hallucination_rate=hallucination,
        execution_success_rate=execution_success,
        notes=notes,
    )


def aggregate_bench_report(
    suite: ICUAgentBenchSuite,
    results: Sequence[ICUAgentBenchTaskResult],
) -> ICUAgentBenchReport:
    """Combine per-task results into a suite-level report.

    Results are bucketed by ``ICUAgentBenchTask.category``:

    * ``"evaluation"`` results feed the ``aggregate`` field — the
      number a paper should cite as the agent's bench correctness.
    * ``"self_check"`` results feed ``self_check_aggregate`` — a
      separate, non-headline number used to monitor CI fixtures.

    Results whose task is *not* in the suite (e.g. fabricated test ids)
    default to ``"evaluation"`` so call sites that supply only
    ad-hoc results still get a headline number.

    Within each bucket the aggregate is the arithmetic mean of each
    metric across results where that metric is not ``None`` — missing
    metrics do not contribute zero.
    """

    metric_names = (
        "correctness",
        "reproducibility",
        "provenance_completeness",
        "icu_semantic_validity",
        "hallucination_rate",
        "statistical_validity",
        "execution_success_rate",
    )

    cat_by_task = suite.category_by_task()
    eval_results, self_check_results = [], []
    for r in results:
        if cat_by_task.get(r.task_id, "evaluation") == "self_check":
            self_check_results.append(r)
        else:
            eval_results.append(r)

    def _mean_over(bucket: List[ICUAgentBenchTaskResult]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in metric_names:
            values = [
                getattr(r, name) for r in bucket if getattr(r, name) is not None
            ]
            if values:
                out[name] = sum(values) / len(values)
        return out

    return ICUAgentBenchReport(
        suite_name=suite.name,
        task_results=list(results),
        aggregate=_mean_over(eval_results),
        self_check_aggregate=_mean_over(self_check_results),
    )


# ---------------------------------------------------------------------------
# A/B comparison harness
# ---------------------------------------------------------------------------
# This layer exists so experimental ablations (with/without memory,
# with/without retrieval, with/without reviewer round) can be cashed
# out as a single quantitative claim suitable for a methods-paper
# results section. It is a pure-function sink: a downstream component
# produces two ``ICUAgentBenchReport`` objects, this layer turns them
# into a per-task delta table plus a human-readable verdict.


class TaskMetricDelta(BaseModel):
    """One row of an A/B comparison: treatment - baseline on a single metric."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    metric: str
    baseline: Optional[float] = None
    treatment: Optional[float] = None
    delta: Optional[float] = None  # treatment - baseline; None if either side missing
    # Bench category of the underlying task. Populated when ``compare_bench_reports``
    # is called with a ``suite`` argument so consumers can filter the
    # per-task rows the same way ``aggregate_deltas`` is filtered.
    category: TaskCategory = "evaluation"


class BenchABComparison(BaseModel):
    """Pure data record of an A/B comparison between two bench reports."""

    model_config = ConfigDict(extra="forbid")

    baseline_name: str
    treatment_name: str

    # Per-(task, metric) deltas. Sorted by task_id, then metric, so
    # rendered comparisons stay diff-stable across runs.
    per_task_deltas: List[TaskMetricDelta] = Field(default_factory=list)

    # Suite-level aggregate deltas keyed by metric name. Computed as the
    # arithmetic mean of non-None per-task deltas for that metric over
    # the **evaluation** category only — i.e. the headline number a
    # paper would cite. ``self_check_aggregate_deltas`` is the parallel
    # value over self-test fixtures (CI signal, not headline).
    aggregate_deltas: Dict[str, float] = Field(default_factory=dict)
    self_check_aggregate_deltas: Dict[str, float] = Field(default_factory=dict)

    # Per-task correctness winners — useful for "treatment helps on
    # which task families?" qualitative analysis. These lists span ALL
    # categories: filter by ``TaskMetricDelta.category`` in ``per_task_deltas``
    # if you need an evaluation-only or self-check-only slice.
    tasks_with_correctness_gain: List[str] = Field(default_factory=list)
    tasks_with_correctness_loss: List[str] = Field(default_factory=list)
    tasks_unchanged: List[str] = Field(default_factory=list)

    # One-paragraph human-readable summary. Cites the **evaluation**
    # correctness delta only, so it's safe to drop into a paper paragraph.
    verdict: str = ""


_BENCH_METRIC_NAMES = (
    "correctness",
    "reproducibility",
    "provenance_completeness",
    "icu_semantic_validity",
    "hallucination_rate",
    "statistical_validity",
    "execution_success_rate",
)


def compare_bench_reports(
    baseline: ICUAgentBenchReport,
    treatment: ICUAgentBenchReport,
    *,
    baseline_name: str = "baseline",
    treatment_name: str = "treatment",
    suite: Optional[ICUAgentBenchSuite] = None,
) -> BenchABComparison:
    """Diff two bench reports task-by-task and metric-by-metric.

    Pure function. No mutation of the inputs.

    * Tasks that appear on only one side contribute only the side they
      appear on; the missing side is recorded as ``None`` and that
      row's ``delta`` is ``None``.
    * When ``suite`` is provided, each delta row is tagged with the
      task's ``TaskCategory`` and the aggregate deltas are bucketed:
      ``aggregate_deltas`` covers the **evaluation** tasks (the
      paper headline), ``self_check_aggregate_deltas`` covers the
      self-test fixtures.
    * Without a ``suite`` argument every task is treated as
      ``"evaluation"`` and ``self_check_aggregate_deltas`` is empty —
      back-compat path.

    The ``verdict`` string summarises the **evaluation** correctness
    delta in natural language — safe to drop into a paper paragraph.
    """

    cat_by_task: Dict[str, TaskCategory] = (
        suite.category_by_task() if suite is not None else {}
    )

    base_by_task = {r.task_id: r for r in baseline.task_results}
    trt_by_task = {r.task_id: r for r in treatment.task_results}
    all_task_ids = sorted(set(base_by_task) | set(trt_by_task))

    deltas: List[TaskMetricDelta] = []
    gains: List[str] = []
    losses: List[str] = []
    unchanged: List[str] = []

    for task_id in all_task_ids:
        b_result = base_by_task.get(task_id)
        t_result = trt_by_task.get(task_id)
        category: TaskCategory = cat_by_task.get(task_id, "evaluation")
        for metric in _BENCH_METRIC_NAMES:
            b_val = getattr(b_result, metric, None) if b_result is not None else None
            t_val = getattr(t_result, metric, None) if t_result is not None else None
            delta = None
            if b_val is not None and t_val is not None:
                delta = t_val - b_val
            deltas.append(
                TaskMetricDelta(
                    task_id=task_id,
                    metric=metric,
                    baseline=b_val,
                    treatment=t_val,
                    delta=delta,
                    category=category,
                )
            )
            if metric == "correctness" and delta is not None:
                if delta > 0:
                    gains.append(task_id)
                elif delta < 0:
                    losses.append(task_id)
                else:
                    unchanged.append(task_id)

    def _mean_by(filter_category: TaskCategory) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for metric in _BENCH_METRIC_NAMES:
            values = [
                d.delta
                for d in deltas
                if d.metric == metric
                and d.delta is not None
                and d.category == filter_category
            ]
            if values:
                out[metric] = sum(values) / len(values)
        return out

    aggregate = _mean_by("evaluation")
    self_check_aggregate = _mean_by("self_check")

    corr = aggregate.get("correctness")
    if corr is None:
        verdict = (
            f"No comparable evaluation-category correctness signal between "
            f"{baseline_name} and {treatment_name}."
        )
    else:
        direction = "improves" if corr > 0 else ("regresses" if corr < 0 else "matches")
        eval_gains = sum(
            1
            for d in deltas
            if d.metric == "correctness"
            and d.category == "evaluation"
            and d.delta is not None
            and d.delta > 0
        )
        eval_losses = sum(
            1
            for d in deltas
            if d.metric == "correctness"
            and d.category == "evaluation"
            and d.delta is not None
            and d.delta < 0
        )
        eval_unchanged = sum(
            1
            for d in deltas
            if d.metric == "correctness"
            and d.category == "evaluation"
            and d.delta == 0
        )
        verdict = (
            f"{treatment_name} {direction} mean evaluation correctness by "
            f"{corr:+.3f} vs {baseline_name} "
            f"(eval gains: {eval_gains}, losses: {eval_losses}, "
            f"unchanged: {eval_unchanged})."
        )

    return BenchABComparison(
        baseline_name=baseline_name,
        treatment_name=treatment_name,
        per_task_deltas=deltas,
        aggregate_deltas=aggregate,
        self_check_aggregate_deltas=self_check_aggregate,
        tasks_with_correctness_gain=gains,
        tasks_with_correctness_loss=losses,
        tasks_unchanged=unchanged,
        verdict=verdict,
    )


def _render_task_line(task: ICUAgentBenchTask) -> List[str]:
    out = [f"- **{task.task_id}** ({task.kind}) — {task.objective}"]
    if task.target_databases:
        out.append(f"  - Target databases: {', '.join(task.target_databases)}")
    out.append(f"  - Gold answers: {task.gold_answer_status}")
    if task.expected_outputs:
        out.append(f"  - Expected outputs: {', '.join(task.expected_outputs)}")
    if task.semantic_guardrails:
        out.append(f"  - Guardrails: {', '.join(task.semantic_guardrails)}")
    if task.gold_answer is not None:
        ga = task.gold_answer
        if ga.numeric_targets:
            target_pairs = ", ".join(
                f"{name} in [{b.lower}, {b.upper}]"
                for name, b in ga.numeric_targets.items()
            )
            out.append(f"  - Numeric targets: {target_pairs}")
        if ga.required_warnings:
            out.append(f"  - Required warnings: {', '.join(ga.required_warnings)}")
        if ga.data_fixture:
            out.append(f"  - Data fixture: {ga.data_fixture}")
    return out


def icu_agent_bench_markdown(suite: Optional[ICUAgentBenchSuite] = None) -> str:
    suite = suite or default_icu_agent_bench_suite()
    frozen = suite.frozen_task_ids()
    eval_tasks = [t for t in suite.tasks if t.category == "evaluation"]
    self_check_tasks = [t for t in suite.tasks if t.category == "self_check"]

    lines = [
        f"# {suite.name}",
        "",
        f"_Status: {suite.maturity} evaluation framework / planned benchmark suite_",
        "",
        f"_Frozen tasks: {len(frozen)} of {len(suite.tasks)} "
        f"(evaluation: {len(eval_tasks)}, self-check: {len(self_check_tasks)})_",
        "",
        "This document defines task families and target metrics for ICUAgentBench.",
        "It should be described as a prototype evaluation framework until task",
        "implementations, gold answers, adjudication rules, and frozen runners",
        "are all finalized.",
        "",
        "**Category convention.** Tasks marked *evaluation* are the headline-",
        "relevant set — they are the ones whose mean correctness an external",
        "paper should cite. Tasks marked *self-check* are deterministic CI",
        "fixtures used to regression-test the grading machinery itself; they",
        "MUST NOT be averaged into the evaluation headline.",
        "",
        "## Evaluation tasks",
        "",
    ]
    for task in eval_tasks:
        lines.extend(_render_task_line(task))

    if self_check_tasks:
        lines.extend(["", "## Self-check tasks", ""])
        for task in self_check_tasks:
            lines.extend(_render_task_line(task))

    lines.extend(["", "## Metrics", ""])
    for name, enabled in suite.metrics.model_dump().items():
        lines.append(f"- {name}: {'enabled' if enabled else 'disabled'}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "TaskCategory",
    "ICUAgentBenchMetricSpec",
    "ICUAgentBenchNumericBound",
    "ICUAgentBenchGoldAnswer",
    "ICUAgentBenchTask",
    "ICUAgentBenchSuite",
    "ICUAgentBenchTaskResult",
    "ICUAgentBenchReport",
    "default_icu_agent_bench_suite",
    "grade_bench_task",
    "aggregate_bench_report",
    "TaskMetricDelta",
    "BenchABComparison",
    "compare_bench_reports",
    "icu_agent_bench_markdown",
]
