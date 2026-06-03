"""Cohort cache (T3.5) — short-circuit identical re-runs.

Pin five contracts:

1. Two runs with identical inputs and ``enable_cache=True`` return
   the *same* run_id and the same evidence count — the second one
   doesn't re-execute the pipeline.
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


def _run(ra, *, cohort, workdir: Path, enable_cache: bool, **kwargs):
    pipeline = ra.ResearchAgentPipeline(
        workdir=workdir, llm=ra.MockLLMClient(), enable_cache=enable_cache,
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


def test_cache_hit_reuses_run_id_and_workdir(ra, synthetic_cohort, tmp_path: Path):
    a = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)
    b = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)
    assert b.run_id == a.run_id, "cache hit should return the prior run_id"
    assert b.workdir == a.workdir
    assert b.evidence_count == a.evidence_count
    # The cache index file is on disk under the workdir's .cache dir.
    cache_index = tmp_path / ".cache" / "cache_index.json"
    assert cache_index.exists(), "cache index file must be created"
    data = json.loads(cache_index.read_text(encoding="utf-8"))
    assert any(v.get("run_id") == a.run_id for v in data.values())


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
        workdir=tmp_path, llm=ra.MockLLMClient(), enable_cache=True,
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
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path, llm=ra.MockLLMClient(), enable_cache=True,
    )
    a = pipeline.run(
        cohort=synthetic_cohort,
        cohort_name="cache_test", database="synthetic",
        target_outcome="death", skill="association_analysis",
    )
    b = pipeline.run(
        cohort=synthetic_cohort,
        cohort_name="cache_test", database="synthetic",
        target_outcome="death", skill="association_analysis",
    )
    # Same skill → hit
    assert b.run_id == a.run_id

    # Different skill → miss
    c = pipeline.run(
        cohort=synthetic_cohort,
        cohort_name="cache_test", database="synthetic",
        target_outcome="death", skill="prediction_model",
    )
    assert c.run_id != a.run_id


def test_resume_bypasses_cache(ra, synthetic_cohort, tmp_path: Path):
    """When resume_run_id is supplied, the cache lookup must be
    skipped — otherwise we'd silently land on a different run_id and
    the resume would be a no-op against the wrong directory."""
    a = _run(ra, cohort=synthetic_cohort, workdir=tmp_path, enable_cache=True)

    # Try to resume a non-existent run_id while the cache contains a
    # hit for the same inputs. Without the bypass the run would
    # return the cached PipelineResult (with the original run_id).
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path, llm=ra.MockLLMClient(), enable_cache=True,
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
