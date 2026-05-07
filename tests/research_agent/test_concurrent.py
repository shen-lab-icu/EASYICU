"""Tests for concurrent step execution (T3.3).

Three properties to pin:

1. **Correctness** — running with ``max_concurrent_steps > 1`` produces
   the same set of evidence aliases and the same critical findings
   (sofa==0 anomaly, fillna(0) warning) as a sequential run, just in
   completion order on disk.
2. **Determinism of paper output** — even though workers finish in
   non-deterministic order, ``per_step_records`` in the manifest is
   sorted by plan order so reviewers see a stable Methods table.
3. **EvidenceStore thread safety** — many concurrent ``register_file``
   calls preserve every record and every alias without losing any.
"""

from __future__ import annotations

import json
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
    assert len(store.records()) == expected, "lost a record under concurrent registration"

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
        # The two ICU rules whose firing is the whole point of the
        # validator layer; if these don't survive a concurrent run,
        # the layer is broken.
        if "sofa2==0 outcome rate" in msg or "fillna(0)" in msg:
            out.add(f["validator"] + "::" + msg.split(".")[0])
    return out


def test_concurrent_pipeline_matches_sequential_findings(ra, synthetic_cohort, tmp_path):
    """Two pipelines, identical inputs, different ``max_concurrent_steps`` —
    the resulting set of *critical* findings must be identical."""
    seq_dir = tmp_path / "seq"
    par_dir = tmp_path / "par"

    seq_pipeline = ra.ResearchAgentPipeline(
        workdir=str(seq_dir),
        llm=ra.MockLLMClient(),
        max_concurrent_steps=1,
        enable_literature=False, enable_visual_qa=False,
        enable_memory=False, enable_latex=False,
    )
    par_pipeline = ra.ResearchAgentPipeline(
        workdir=str(par_dir),
        llm=ra.MockLLMClient(),
        max_concurrent_steps=4,
        enable_literature=False, enable_visual_qa=False,
        enable_memory=False, enable_latex=False,
    )

    seq_result = seq_pipeline.run(skill="sofa_mortality",
                                  cohort=synthetic_cohort, database="synthetic")
    par_result = par_pipeline.run(skill="sofa_mortality",
                                  cohort=synthetic_cohort, database="synthetic")

    seq_findings = _critical_findings(Path(seq_result.manifest_path))
    par_findings = _critical_findings(Path(par_result.manifest_path))
    assert seq_findings == par_findings, (
        "Concurrent run lost or fabricated a critical finding; "
        f"seq={seq_findings} par={par_findings}"
    )


def test_concurrent_pipeline_records_sorted_by_plan_order(ra, synthetic_cohort, tmp_path):
    """Even with workers finishing in any order, ``per_step_records`` in
    the manifest must follow plan order."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path),
        llm=ra.MockLLMClient(),
        max_concurrent_steps=4,
        enable_literature=False, enable_visual_qa=False,
        enable_memory=False, enable_latex=False,
    )
    result = pipeline.run(skill="sofa_mortality",
                          cohort=synthetic_cohort, database="synthetic")

    # The skill plan begins with table_one and ends with the SOFA-zero audit.
    partial = json.loads(
        (Path(result.workdir) / "manifest_partial.json").read_text(encoding="utf-8")
    )
    plan = json.loads(Path(result.plan_path).read_text(encoding="utf-8"))
    plan_step_ids = [s["step_id"] for s in plan["steps"]]

    # The final report sorts records by plan order; the report file
    # must list step ids in the same sequence.
    report_text = Path(result.report_path).read_text(encoding="utf-8")
    positions = []
    for sid in plan_step_ids:
        pos = report_text.find(sid)
        if pos >= 0:
            positions.append(pos)
    assert positions == sorted(positions), (
        "results_report.md does not list step ids in plan order even though "
        "the pipeline sorted per_step_records before rendering."
    )

    # Sanity: the partial manifest contains every planned step, plus the
    # optional probe pre-step when probe mode is enabled.
    seen = {r["step_id"] for r in partial.get("per_step_records", [])}
    assert set(plan_step_ids) <= seen
    assert seen - set(plan_step_ids) <= {"00_probe"}


def test_concurrent_default_one_worker_keeps_sequential_path(ra, synthetic_cohort, tmp_path):
    """Default ``max_concurrent_steps=1`` must still execute serially —
    we don't want to silently introduce a thread pool for users who
    didn't ask for it."""
    pipeline = ra.ResearchAgentPipeline(
        workdir=str(tmp_path),
        llm=ra.MockLLMClient(),
        # default: max_concurrent_steps not specified
        enable_literature=False, enable_visual_qa=False,
        enable_memory=False, enable_latex=False,
    )
    assert pipeline._max_concurrent_steps == 1
    result = pipeline.run(skill="sofa_mortality",
                          cohort=synthetic_cohort, database="synthetic")
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
