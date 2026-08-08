"""Cohort cache (T3.5) — short-circuit identical re-runs.

Pin five contracts:

1. Identical inputs reuse a prior run only after the host verifies a
   manuscript-ready terminal authority; analysis-only runs remain fresh.
2. Cache off (default) means every run is fresh, even with identical
   inputs.
3. Mutating the cohort invalidates the cache.
4. Mutating the question / skill / target outcome invalidates the cache.
5. ``resume_run_id`` bypasses the cache so a partial-run resume
   doesn't accidentally land on a cache hit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _run(ra, *, cohort, workdir: Path, enable_cache: bool, **kwargs):
    pipeline = ra.ResearchAgentPipeline(
        workdir=workdir,
        llm=ra.MockLLMClient(),
        enable_cache=enable_cache,
    )
    return pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=cohort,
        cohort_name="cache_test",
        database="synthetic",
        target_outcome="death",
        **kwargs,
    )


def test_cache_off_runs_each_time(ra, synthetic_cohort, tmp_path: Path):
    a = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=False)
    b = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=False)
    assert a.run_id != b.run_id, "cache off must produce a fresh run_id every time"


def test_analysis_only_runs_are_not_cached_as_completed(
    ra, synthetic_cohort, tmp_path: Path
):
    a = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)
    b = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)
    assert b.run_id != a.run_id
    for result in (a, b):
        status = json.loads(
            (Path(result.workdir) / "run_status.json").read_text(encoding="utf-8")
        )
        assert status["status"] == "analysis_only"
        assert status["gates"]["manuscript_ready"] is False
    cache_index = tmp_path / ".cache" / "cache_index.json"
    assert (
        not cache_index.exists()
        or json.loads(cache_index.read_text(encoding="utf-8")) == {}
    )


def test_cache_lookup_reuses_a_host_verified_completed_candidate(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.authority import pipeline_cache

    result = _write_cache_candidate(ra, tmp_path)
    run_dir = Path(result.workdir)
    paths = {
        "context": Path(result.context_path),
        "plan": Path(result.plan_path),
        "report": Path(result.report_path),
        "manuscript": Path(result.manuscript_path),
    }
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    monkeypatch.setattr(
        pipeline_cache,
        "_completed_run_payload",
        lambda **_kwargs: (paths, manifest),
    )
    cache = pipeline_cache.PipelineCache(tmp_path / ".cache")
    cache.save_index(
        {
            "verified": {
                "run_id": result.run_id,
                "workdir": result.workdir,
                "evidence_count": "1",
                "findings_count": "0",
            }
        }
    )

    reused = cache.lookup("verified", scientific_identity={"question": "q"})

    assert reused is not None
    assert reused.run_id == result.run_id
    assert reused.workdir == result.workdir


def test_cache_invalidates_on_cohort_change(ra, synthetic_cohort, tmp_path: Path):
    a = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)
    # Mutate the cohort: add a column → bytes change → cache miss.
    mutated = synthetic_cohort.copy()
    mutated["new_col"] = 1
    b = _run(ra, cohort=mutated, workdir=tmp_path, enable_cache=True)
    assert b.run_id != a.run_id, "different cohort must miss the cache"


def test_cache_invalidates_on_question_change(ra, synthetic_cohort, tmp_path: Path):
    a = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        enable_cache=True,
    )
    b = pipeline.run(
        question="A different research question entirely.",  # ← changed
        cohort=synthetic_cohort,
        cohort_name="cache_test",
        database="synthetic",
        target_outcome="death",
    )
    assert b.run_id != a.run_id


def test_cache_invalidates_on_skill_change(ra, synthetic_cohort, tmp_path: Path):
    from easyicu.research_agent.authority.pipeline_cache import PipelineCache

    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"cohort")
    cache = PipelineCache(tmp_path / ".cache")
    a = _compute_unit_key(cache, cohort_path, skill_key="association_analysis")
    b = _compute_unit_key(cache, cohort_path, skill_key="association_analysis")
    c = _compute_unit_key(cache, cohort_path, skill_key="prediction_model")
    assert b == a
    assert c != a


def test_resume_bypasses_cache(ra, synthetic_cohort, tmp_path: Path):
    """When resume_run_id is supplied, the cache lookup must be
    skipped — otherwise we'd silently land on a different run_id and
    the resume would be a no-op against the wrong directory."""
    a = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)

    # Try to resume a non-existent run_id while the cache contains a
    # hit for the same inputs. Without the bypass the run would
    # return the cached PipelineResult (with the original run_id).
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        enable_cache=True,
    )
    b = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="cache_test",
        database="synthetic",
        target_outcome="death",
        resume_run_id="run_does_not_exist_yet",
    )
    assert b.run_id == "run_does_not_exist_yet"
    assert b.run_id != a.run_id


def test_cache_recovers_from_deleted_run_dir(ra, synthetic_cohort, tmp_path: Path):
    """If the prior run_dir is deleted between runs, the next call
    must detect the stale entry, evict it, and run fresh — not
    return a PipelineResult pointing at a non-existent directory."""
    a = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)
    import shutil

    shutil.rmtree(Path(a.workdir))
    b = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)
    assert b.run_id != a.run_id
    assert Path(b.manifest_path).exists()


def _compute_unit_key(cache, cohort_path: Path, **overrides) -> str:
    kwargs = {
        "cohort_path": cohort_path,
        "question": "Does exposure affect outcome?",
        "target_outcome": "outcome",
        "skill_key": "association_analysis",
        "database": "synthetic",
        "llm": None,
        "stop_after_analysis": False,
        "manuscript_language": "en",
        "flags": {"enable_llm_concept_audit": True},
        "science_inputs": {
            "primary_exposure": "exposure",
            "inclusion_criteria": ["adult"],
            "time_windows": [{"name": "baseline", "start_hours": 0, "end_hours": 6}],
            "context": {"disable_icu_context": False},
        },
        "identity_hashes": {
            "engine_code_sha256": "engine-a",
            "validator_code_sha256": "validator-a",
            "prompt_pack_sha256": "prompt-a",
            "concept_dictionary_sha256": "concept-a",
        },
    }
    identity_overrides = overrides.pop("identity_hashes", None)
    if identity_overrides:
        kwargs["identity_hashes"].update(identity_overrides)
    kwargs.update(overrides)
    return cache.compute_key(**kwargs)


def _write_cache_candidate(
    ra,
    root: Path,
    *,
    run_id: str = "run_cache_candidate",
    status: str = "manuscript_ready",
    notes: str = "complete",
    step_status: str = "ok",
    final_sequence: int = 2,
    partial_sequence: int = 1,
):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.run_input import seal_run_input_capsule
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    question = "Cache candidate question."
    cohort_path = run_dir / "cohort.parquet"
    cohort_path.write_bytes(b"cache-candidate-cohort")
    context_path = run_dir / "research_context.json"
    context_path.write_text(
        ResearchContext(
            research_question=question,
            cohort=CohortDescriptor(
                cohort_name="cache_candidate",
                database="synthetic",
                n_stays=1,
                n_patients=1,
            ),
            variables=[],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    plan_path = run_dir / "analysis_plan.json"
    plan_path.write_text(
        json.dumps({"steps": [{"step_id": "01_model"}]}),
        encoding="utf-8",
    )
    report_path = run_dir / "results_report.md"
    report_path.write_text("report", encoding="utf-8")
    manuscript_path = run_dir / "manuscript_scaffold_bound.md"
    manuscript_path.write_text("manuscript", encoding="utf-8")
    script_path = run_dir / "analysis.py"
    script_path.write_text("print('complete')\n", encoding="utf-8")
    summary_path = run_dir / "step_summary.json"
    summary_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    gates = {
        "execution_complete": True,
        "evidence_complete": True,
        "numeric_verified": True,
        "analysis_validated": True,
        "manuscript_ready": True,
        "publication_ready": status == "publication_ready",
    }
    status_path = run_dir / "run_status.json"
    status_path.write_text(
        json.dumps({"status": status, "gates": gates, "research_question": question}),
        encoding="utf-8",
    )

    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Frozen research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    scientific_identity = {"question": question}
    seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=scientific_identity,
        initial_environment={"llm_signature": "test"},
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    for evidence_id, kind, description, source_path in (
        ("analysis_plan", "log", "Frozen analysis plan.", plan_path),
        (
            "manuscript_scaffold_bound",
            "log",
            "Bound manuscript.",
            manuscript_path,
        ),
        ("run_status", "log", "Terminal run status.", status_path),
    ):
        evidence.register_file(
            kind=kind,
            description=description,
            source_path=source_path,
            evidence_id=evidence_id,
            producer="pipeline",
            generation_mode="system",
        )
    evidence.register_file(
        kind="code",
        description="Executed analysis script.",
        source_path=script_path,
        produced_by_step="01_model",
        evidence_id="model_script",
        producer="coder",
        generation_mode="agent_generated",
    )
    evidence.register_file(
        kind="statistic",
        description="Machine-readable step summary.",
        source_path=summary_path,
        produced_by_step="01_model",
        script_evidence_id="model_script",
        evidence_id="model_summary",
        producer="pipeline",
        generation_mode="system",
    )
    evidence_payload = [record.model_dump(mode="json") for record in evidence.records()]
    manifest = {
        "checkpoint_sequence": final_sequence,
        "run_id": run_id,
        "research_question": question,
        "started_at": "2026-07-15T11:59:00+00:00",
        "finished_at": "2026-07-15T12:00:00+00:00",
        "context_path": "research_context.json",
        "plan_path": "analysis_plan.json",
        "report_path": "results_report.md",
        "manuscript_path": "manuscript_scaffold_bound.md",
        "artifact_paths": {"run_status": "run_status.json"},
        "readiness": gates,
        "writer_probe_mode": False,
        "per_step_records": [
            {
                "step_id": "01_model",
                "status": step_status,
                "evidence_ids": ["model_script", "model_summary"],
                "script_evidence_id": "model_script",
                "step_summary_evidence_id": "model_summary",
            }
        ],
        "evidence": evidence_payload,
        "findings": [],
        "notes": notes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "checkpoint_sequence": partial_sequence,
                "run_id": run_id,
                "per_step_records": manifest["per_step_records"],
            }
        ),
        encoding="utf-8",
    )
    return ra.PipelineResult(
        run_id=run_id,
        workdir=str(run_dir),
        context_path=str(run_dir / "research_context.json"),
        plan_path=str(run_dir / "analysis_plan.json"),
        manifest_path=str(run_dir / "manifest.json"),
        report_path=str(run_dir / "results_report.md"),
        manuscript_path=str(run_dir / "manuscript_scaffold_bound.md"),
        evidence_count=len(evidence_payload),
        findings_count=0,
    )


def test_cache_key_binds_science_inputs_and_runtime_hashes(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.authority.pipeline_cache import PipelineCache

    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"cohort")
    cache = PipelineCache(tmp_path / ".cache")
    baseline = _compute_unit_key(cache, cohort_path)

    changed_science = _compute_unit_key(
        cache,
        cohort_path,
        science_inputs={
            "primary_exposure": "different_exposure",
            "inclusion_criteria": ["adult"],
            "time_windows": [{"name": "baseline", "start_hours": 0, "end_hours": 6}],
            "context": {"disable_icu_context": False},
        },
    )
    changed_validator = _compute_unit_key(
        cache,
        cohort_path,
        identity_hashes={"validator_code_sha256": "validator-b"},
    )
    changed_prompt = _compute_unit_key(
        cache,
        cohort_path,
        identity_hashes={"prompt_pack_sha256": "prompt-b"},
    )
    changed_concept = _compute_unit_key(
        cache,
        cohort_path,
        identity_hashes={"concept_dictionary_sha256": "concept-b"},
    )
    changed_engine = _compute_unit_key(
        cache,
        cohort_path,
        identity_hashes={"engine_code_sha256": "engine-b"},
    )
    changed_metadata = _compute_unit_key(
        cache,
        cohort_path,
        identity_hashes={
            "metadata_projection_sha256": "projector-b",
            "metadata_sidecar_sha256": "sidecar-b",
            "icu_rules_sha256": "icu-rules-b",
            "metadata_implementation_bundle_sha256": "metadata-bundle-b",
        },
    )
    assert (
        len(
            {
                baseline,
                changed_science,
                changed_validator,
                changed_prompt,
                changed_concept,
                changed_engine,
                changed_metadata,
            }
        )
        == 7
    )


def test_cache_key_binds_materialised_experiment_spec(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.authority.pipeline_cache import PipelineCache

    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"cohort")
    spec_path = tmp_path / "experiment_spec.yaml"
    spec_path.write_text("primary_exposure: exposure_a\n", encoding="utf-8")
    cache = PipelineCache(tmp_path / ".cache")
    first = _compute_unit_key(cache, cohort_path)
    spec_path.write_text("primary_exposure: exposure_b\n", encoding="utf-8")
    second = _compute_unit_key(cache, cohort_path)
    assert second != first


def test_cache_rejects_manifest_only_completed_run_without_authority(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.authority.pipeline_cache import PipelineCache

    cache = PipelineCache(tmp_path / ".cache")
    result = _write_cache_candidate(ra, tmp_path)
    fake_identity = {"research_question": "fabricated"}
    cache.record_hit(
        "complete",
        result,
        scientific_identity=fake_identity,
    )
    assert cache.lookup("complete", scientific_identity=fake_identity) is None
    assert "complete" not in cache.load_index()


def test_cache_rejects_valid_run_under_wrong_scientific_identity(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.pipeline_cache import PipelineCache

    result = _write_cache_candidate(ra, tmp_path)
    cache = PipelineCache(tmp_path / ".cache")
    cache.record_hit(
        "complete",
        result,
        scientific_identity={"question": "Cache candidate question."},
    )
    index = cache.load_index()
    assert len(index) == 1
    cache_key = next(iter(index))
    assert (
        cache.lookup(
            cache_key,
            scientific_identity={"research_question": "wrong study"},
        )
        is None
    )
    assert cache_key not in cache.load_index()


def test_cache_rejects_mutated_root_status_even_when_manifest_is_ready(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.pipeline_cache import PipelineCache

    result = _write_cache_candidate(ra, tmp_path)
    run_dir = Path(result.workdir)
    capsule = json.loads(
        (run_dir / "run_input_capsule.json").read_text(encoding="utf-8")
    )
    cache = PipelineCache(tmp_path / ".cache")
    cache.record_hit(
        "complete",
        result,
        scientific_identity=capsule["scientific_identity"],
    )
    cache_key = next(iter(cache.load_index()))
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "manuscript_ready", "gates": {}}),
        encoding="utf-8",
    )

    assert (
        cache.lookup(
            cache_key,
            scientific_identity=capsule["scientific_identity"],
        )
        is None
    )


def test_cache_rejects_mutated_selected_evidence(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.pipeline_cache import PipelineCache

    result = _write_cache_candidate(ra, tmp_path)
    run_dir = Path(result.workdir)
    capsule = json.loads(
        (run_dir / "run_input_capsule.json").read_text(encoding="utf-8")
    )
    cache = PipelineCache(tmp_path / ".cache")
    cache.record_hit(
        "complete",
        result,
        scientific_identity=capsule["scientific_identity"],
    )
    cache_key = next(iter(cache.load_index()))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    context_record = next(
        record
        for record in manifest["evidence"]
        if record["evidence_id"] == "research_context"
    )
    evidence_path = run_dir / context_record["relative_path"]
    evidence_path.write_text("{}", encoding="utf-8")

    assert (
        cache.lookup(
            cache_key,
            scientific_identity=capsule["scientific_identity"],
        )
        is None
    )


@pytest.mark.parametrize(
    ("status", "notes", "step_status", "final_sequence", "partial_sequence"),
    [
        ("analysis_only", "complete", "ok", 2, 1),
        ("human_review_rejected", "complete", "ok", 2, 1),
        ("manuscript_ready", "paused_after_analysis", "ok", 2, 1),
        ("manuscript_ready", "complete", "contract_failed", 2, 1),
        ("manuscript_ready", "complete", "ok", 1, 2),
    ],
)
def test_cache_rejects_partial_paused_blocked_or_superseded_runs(
    ra,
    tmp_path: Path,
    status: str,
    notes: str,
    step_status: str,
    final_sequence: int,
    partial_sequence: int,
) -> None:
    from easyicu.research_agent.authority.pipeline_cache import PipelineCache

    cache = PipelineCache(tmp_path / ".cache")
    result = _write_cache_candidate(
        ra,
        tmp_path,
        status=status,
        notes=notes,
        step_status=step_status,
        final_sequence=final_sequence,
        partial_sequence=partial_sequence,
    )
    # Simulate both a new write and a stale legacy index entry: neither path
    # may surface an incomplete run as a complete PipelineResult.
    scientific_identity = {"research_question": "candidate"}
    cache.record_hit(
        "candidate",
        result,
        scientific_identity=scientific_identity,
    )
    assert cache.lookup("candidate", scientific_identity=scientific_identity) is None
    cache.save_index(
        {
            "candidate": {
                "run_id": result.run_id,
                "workdir": result.workdir,
                "evidence_count": "1",
                "findings_count": "0",
            }
        }
    )
    assert cache.lookup("candidate", scientific_identity=scientific_identity) is None
    assert "candidate" not in cache.load_index()
