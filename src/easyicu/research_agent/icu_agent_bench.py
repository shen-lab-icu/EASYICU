"""Typed ICUAgentBench scaffolding for benchmark/evaluation runs.

The current EASYICU benchmark runner grew organically from internal rule and
analysis tasks. This module adds a formal, reusable schema layer so future
paper-facing evaluation can talk about a stable ICUAgentBench task suite
instead of ad hoc JSON blobs.

Important: this is currently a prototype evaluation framework and planned
benchmark suite schema. It does not, by itself, constitute a frozen release
with gold answers, adjudicated metrics, or an external benchmark runner.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    difficulty: Literal["basic", "intermediate", "advanced"] = "intermediate"


class ICUAgentBenchSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "easyicu.icu_agent_bench/1"
    name: str = "ICUAgentBench"
    maturity: Literal["prototype"] = "prototype"
    tasks: List[ICUAgentBenchTask] = Field(default_factory=list)
    metrics: ICUAgentBenchMetricSpec = Field(default_factory=ICUAgentBenchMetricSpec)


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
    model_config = ConfigDict(extra="forbid")

    suite_name: str = "ICUAgentBench"
    task_results: List[ICUAgentBenchTaskResult] = Field(default_factory=list)
    aggregate: Dict[str, float] = Field(default_factory=dict)


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
        ]
    )


def icu_agent_bench_markdown(suite: Optional[ICUAgentBenchSuite] = None) -> str:
    suite = suite or default_icu_agent_bench_suite()
    lines = [
        f"# {suite.name}",
        "",
        f"_Status: {suite.maturity} evaluation framework / planned benchmark suite_",
        "",
        "This document defines task families and target metrics for ICUAgentBench.",
        "It should be described as a prototype evaluation framework until task",
        "implementations, gold answers, adjudication rules, and frozen runners",
        "are all finalized.",
        "",
        "## Tasks",
        "",
    ]
    for task in suite.tasks:
        lines.append(f"- **{task.task_id}** ({task.kind}) — {task.objective}")
        if task.target_databases:
            lines.append(f"  - Target databases: {', '.join(task.target_databases)}")
        lines.append(f"  - Gold answers: {task.gold_answer_status}")
        if task.expected_outputs:
            lines.append(f"  - Expected outputs: {', '.join(task.expected_outputs)}")
        if task.semantic_guardrails:
            lines.append(f"  - Guardrails: {', '.join(task.semantic_guardrails)}")
    lines.extend(["", "## Metrics", ""])
    for name, enabled in suite.metrics.model_dump().items():
        lines.append(f"- {name}: {'enabled' if enabled else 'disabled'}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ICUAgentBenchMetricSpec",
    "ICUAgentBenchTask",
    "ICUAgentBenchSuite",
    "ICUAgentBenchTaskResult",
    "ICUAgentBenchReport",
    "default_icu_agent_bench_suite",
    "icu_agent_bench_markdown",
]
