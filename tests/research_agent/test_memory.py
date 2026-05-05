"""RunMemory: a record persists and shows up in the prompt digest."""

from __future__ import annotations

from pathlib import Path


def test_record_and_digest_round_trip(ra, tmp_path: Path):
    schema = ra.schema
    mem = ra.RunMemory(root=tmp_path)

    f1 = schema.ValidationFinding(
        validator="statistical_validator", severity="warning",
        message="sofa2==0 outcome rate exceeds sofa2==1 — likely missing components.",
    )
    f2 = schema.ValidationFinding(
        validator="cohort_auditor", severity="error",
        message="Row count mismatch.",
    )
    rec = mem.record(
        run_id="run_test_001",
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        database="miiv", target_outcome="death",
        findings=[f1, f2],
        workdir=tmp_path,
    )
    assert rec.findings_count == 2
    assert rec.error_count == 1
    assert rec.warning_count == 1

    digest = mem.digest_for_prompt(
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        database="miiv", target_outcome="death",
    )
    assert "run_test_001" in digest
    assert "sofa2" in digest.lower() or "sofa-2" in digest.lower()


def test_no_runs_returns_friendly_digest(ra, tmp_path: Path):
    mem = ra.RunMemory(root=tmp_path)
    digest = mem.digest_for_prompt(
        research_question="any", database="miiv", target_outcome="death",
    )
    assert "no prior runs" in digest.lower()


def test_relevance_ranking_prefers_same_database(ra, tmp_path: Path):
    mem = ra.RunMemory(root=tmp_path)
    mem.record(
        run_id="run_eicu_a", research_question="Does sofa predict mortality?",
        database="eicu", target_outcome="death", findings=[], workdir=tmp_path,
    )
    mem.record(
        run_id="run_miiv_a", research_question="Does sofa predict mortality?",
        database="miiv", target_outcome="death", findings=[], workdir=tmp_path,
    )
    ranked = mem.relevant_to(
        research_question="Does sofa predict mortality in MIMIC-IV?",
        database="miiv", target_outcome="death",
    )
    assert ranked[0].run_id == "run_miiv_a"
