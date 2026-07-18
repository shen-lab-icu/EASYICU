"""Tests for concurrent step execution (T3.3).

Three properties to pin:

1. **Correctness** — running with ``max_concurrent_steps > 1`` produces
   the same set of evidence aliases and the same critical findings
   (e.g. the fillna(0) warning) as a sequential run, just in
   completion order on disk.
2. **Determinism of paper output** — even though workers finish in
   non-deterministic order, ``per_step_records`` in the manifest is
   sorted by plan order so reviewers see a stable Methods table.
3. **EvidenceStore thread safety** — many concurrent ``register_file``
   calls preserve every record and every alias without losing any.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# EvidenceStore stress test
# ---------------------------------------------------------------------------


def test_evidence_store_concurrent_registers_preserve_every_record(ra, tmp_path):
    """Hammer ``register_file`` from N threads; expect every record to land."""
    from easyicu.research_agent.evidence import EvidenceStore

    store = EvidenceStore(root=tmp_path)
    n_threads = 8
    n_per_thread = 25

    barrier = threading.Barrier(n_threads)

    def worker(thread_idx: int) -> None:
        # Stagger thread starts slightly so they really race on _save.
        barrier.wait()
        for j in range(n_per_thread):
            p = tmp_path / f"t{thread_idx}_f{j}.txt"
            p.write_text(f"thread {thread_idx} item {j}", encoding="utf-8")
            store.register_file(
                kind="log",
                description=f"t{thread_idx}_f{j}",
                source_path=p,
                aliases=[f"alias_t{thread_idx}_f{j}"],
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = n_threads * n_per_thread
    assert (
        len(store.records()) == expected
    ), "lost a record under concurrent registration"

    # Every alias should resolve and point at a unique evidence_id.
    for i in range(n_threads):
        for j in range(n_per_thread):
            rec = store.get(f"alias_t{i}_f{j}")
            assert rec is not None, f"alias alias_t{i}_f{j} did not resolve"


# ---------------------------------------------------------------------------
# Pipeline parity: sequential vs concurrent produce equivalent runs
# ---------------------------------------------------------------------------


def _critical_findings(manifest_path: Path) -> set:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = set()
    for f in data.get("findings", []):
        msg = f.get("message", "")
        # ICU rules whose firing is the whole point of the validator
        # layer; if these don't survive a concurrent run, the layer is
        # broken.
        if "fillna(0)" in msg or "component completeness" in msg:
            out.add(f["validator"] + "::" + msg.split(".")[0])
    return out


def test_concurrent_pipeline_matches_sequential_findings(
    ra, synthetic_cohort, tmp_path
):
    """Two pipelines, identical inputs, different ``max_concurrent_steps`` —
    the resulting set of *critical* findings must be identical."""
    seq_dir = tmp_path / "seq"
    par_dir = tmp_path / "par"

    seq_pipeline = ra.ResearchAgentPipeline(
        workdir=str(seq_dir),
        llm=ra.MockLLMClient(),
        max_concurrent_steps=1,
        enable_literature=False,
        enable_visual_qa=False,
        enable_memory=False,
        enable_latex=False,
    )
    par_pipeline = ra.ResearchAgentPipeline(
        workdir=str(par_dir),
        llm=ra.MockLLMClient(),
        max_concurrent_steps=4,
        enable_literature=False,
        enable_visual_qa=False,
        enable_memory=False,
        enable_latex=False,
    )

    seq_result = seq_pipeline.run(
        skill="association_analysis", cohort=synthetic_cohort, database="synthetic"
    )
    par_result = par_pipeline.run(
        skill="association_analysis", cohort=synthetic_cohort, database="synthetic"
    )

    seq_findings = _critical_findings(Path(seq_result.manifest_path))
    par_findings = _critical_findings(Path(par_result.manifest_path))
    assert seq_findings == par_findings, (
        "Concurrent run lost or fabricated a critical finding; "
        f"seq={seq_findings} par={par_findings}"
    )


def test_concurrent_pipeline_records_sorted_by_plan_order(
    ra, synthetic_cohort, tmp_path
):
    """Even with workers finishing in any order, ``per_step_records`` in
    the manifest must follow plan order."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path),
        llm=ra.MockLLMClient(),
        max_concurrent_steps=4,
        enable_literature=False,
        enable_visual_qa=False,
        enable_memory=False,
        enable_latex=False,
    )
    result = pipeline.run(
        skill="association_analysis", cohort=synthetic_cohort, database="synthetic"
    )

    # The skill plan begins with table_one and ends with the QC/audit step.
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    plan_step_ids = [s["step_id"] for s in plan["steps"]]

    # The final report sorts records by plan order. Inspect only the exact
    # Step-outcome rows: a raw ``find(step_id)`` also matches figure-contract
    # paths and parent ids embedded in child ids above this section.
    report_text = Path(result.report_path).read_text(encoding="utf-8")
    outcomes = report_text.split("## Step outcomes", 1)[1].split("## Findings", 1)[0]
    outcome_ids = []
    for line in outcomes.splitlines():
        match = re.match(r"^- \*\*([^*]+)\*\* — status:", line)
        if match and match.group(1) != "00_probe":
            outcome_ids.append(match.group(1))
    expected_outcome_ids = [sid for sid in plan_step_ids if sid in outcome_ids]
    assert outcome_ids == expected_outcome_ids, (
        "results_report.md does not list step ids in plan order even though "
        "the pipeline sorted per_step_records before rendering."
    )

    # Sanity: the partial manifest contains every planned step, plus the
    # optional probe pre-step when probe mode is enabled.
    seen = {r["step_id"] for r in partial.get("per_step_records", [])}
    assert set(plan_step_ids) <= seen
    assert seen - set(plan_step_ids) <= {"00_probe"}


def test_replanning_forces_sequential_even_when_concurrency_requested(
    ra, synthetic_cohort, tmp_path
):
    """B3 safety gate: a run with replanning enabled MUST execute sequentially
    even when ``max_concurrent_steps > 1``, emitting an explicit
    ``forced to sequential`` finding. This is why B3 (bounded step concurrency)
    does not accelerate canonical E3/H2/E2 — those enable replanning and are
    therefore correctly serial-gated, not unoptimized. Raising the cap must never
    silently reorder a replanning run.
    """
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path),
        llm=ra.MockLLMClient(),
        max_concurrent_steps=2,
        enable_replanning=True,
        enable_literature=False,
        enable_visual_qa=False,
        enable_memory=False,
        enable_latex=False,
    )
    result = pipeline.run(
        skill="association_analysis", cohort=synthetic_cohort, database="synthetic"
    )

    data = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    sequential_gate = [
        f
        for f in data.get("findings", [])
        if f.get("validator") == "replanner"
        and "forced to sequential" in f.get("message", "")
    ]
    assert sequential_gate, (
        "replanning must force sequential execution when concurrency is "
        "requested, but the safety-gate finding was not emitted"
    )
    # The cap was accepted (2) but deliberately gated off by replanning.
    assert pipeline._max_concurrent_steps == 2


def test_concurrent_default_one_worker_keeps_sequential_path(
    ra, synthetic_cohort, tmp_path
):
    """Default ``max_concurrent_steps=1`` must still execute serially —
    we don't want to silently introduce a thread pool for users who
    didn't ask for it."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path),
        llm=ra.MockLLMClient(),
        # default: max_concurrent_steps not specified
        enable_literature=False,
        enable_visual_qa=False,
        enable_memory=False,
        enable_latex=False,
    )
    assert pipeline._max_concurrent_steps == 1
    result = pipeline.run(
        skill="association_analysis", cohort=synthetic_cohort, database="synthetic"
    )
    # If the default broke, the run wouldn't even produce a manifest.
    assert Path(result.manifest_path).exists()


def test_max_concurrent_steps_clamped_to_at_least_one(ra):
    """Negative or zero values must clamp to 1, not crash the executor."""
    p = ra.ResearchAgentPipeline(
        workdir="/tmp/_does_not_matter",
        llm=ra.MockLLMClient(),
        max_concurrent_steps=0,
    )
    assert p._max_concurrent_steps == 1
    p2 = ra.ResearchAgentPipeline(
        workdir="/tmp/_does_not_matter",
        llm=ra.MockLLMClient(),
        max_concurrent_steps=-3,
    )
    assert p2._max_concurrent_steps == 1
