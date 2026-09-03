"""Tests for the cross-run experience bank (Commit 3, Phase-1 widening).

Pins:

1. ``jaccard_similarity`` is symmetric, identity-1, disjoint-0, and
   stopword-insensitive.
2. ``ExperienceBank.add`` deduplicates on (kind, summary), refreshes
   timestamps, and merges detail.
3. JSONL load/save round-trips records byte-stably.
4. ``retrieve`` returns ranked records above the threshold; same-
   database boost works; top-k is honoured.
5. ``mine_experience_from_run`` is deterministic on its inputs:
   * concept_usage_hint fires only when all gates pass, no error
     findings, plan has a concept-related step, and cohort_name is
     non-empty.
   * failure_counter_example fires for each unique superseded step,
     in sorted order.
6. End-to-end: ``ResearchAgentPipeline.reflect_and_persist_experience``
   reads a ``run_status.json`` and writes back records via the bank
   when enabled; is a no-op when disabled.
7. Planner isolation: retrieved legacy records never enter Planner context;
   completed-run lessons are mirrored only into permissioned quarantine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.learning.experience import (
    ExperienceBank,
    ExperienceRecord,
    jaccard_similarity,
    mine_experience_from_run,
)

# ---------------------------------------------------------------------
# 1. Jaccard similarity
# ---------------------------------------------------------------------


def test_jaccard_identity_is_one() -> None:
    s = "Does early lactate predict ICU mortality among sepsis patients?"
    assert jaccard_similarity(s, s) == 1.0


def test_jaccard_is_symmetric() -> None:
    a = "Lactate trajectories in sepsis"
    b = "Sepsis-related lactate dynamics"
    assert jaccard_similarity(a, b) == jaccard_similarity(b, a)


def test_jaccard_disjoint_is_zero() -> None:
    assert jaccard_similarity("alpha beta", "gamma delta") == 0.0


def test_jaccard_strips_stopwords() -> None:
    # Stopword-only differences should not lower similarity to <1.
    a = "lactate predicts mortality"
    b = "the lactate predicts the mortality"
    assert jaccard_similarity(a, b) == 1.0


# ---------------------------------------------------------------------
# 2-3. ExperienceBank persistence
# ---------------------------------------------------------------------


def _rec(
    kind: str = "concept_usage_hint",
    research_question: str = "Q",
    database: str = "miiv",
    cohort_name: str = "sepsis3_aware",
    summary: str = "lesson A",
) -> ExperienceRecord:
    return ExperienceRecord(
        kind=kind,
        research_question=research_question,
        database=database,
        cohort_name=cohort_name,
        summary=summary,
    )


def test_bank_add_deduplicates_on_kind_summary(tmp_path: Path) -> None:
    bank = ExperienceBank(path=tmp_path / "bank.jsonl")
    bank.add(_rec(summary="lesson A"))
    bank.add(_rec(summary="lesson A"))
    bank.add(_rec(summary="lesson B"))
    assert len(bank.records()) == 2


def test_bank_add_refreshes_timestamp_and_merges_detail(tmp_path: Path) -> None:
    bank = ExperienceBank(path=tmp_path / "bank.jsonl")
    bank.add(
        ExperienceRecord(
            kind="concept_usage_hint",
            research_question="Q",
            database="miiv",
            cohort_name="cohort",
            summary="lesson A",
            detail={"k1": "v1"},
            produced_at="2026-01-01T00:00:00+00:00",
        )
    )
    bank.add(
        ExperienceRecord(
            kind="concept_usage_hint",
            research_question="Q",
            database="miiv",
            cohort_name="cohort",
            summary="lesson A",
            detail={"k2": "v2"},
            produced_at="2026-02-01T00:00:00+00:00",
        )
    )
    [r] = bank.records()
    assert r.produced_at == "2026-02-01T00:00:00+00:00"
    assert r.detail == {"k1": "v1", "k2": "v2"}


def test_bank_persists_to_jsonl_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "bank.jsonl"
    bank = ExperienceBank(path=path)
    bank.add(_rec(summary="A"))
    bank.add(_rec(summary="B"))
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
    # Each line is valid JSON with the expected key order.
    payload = json.loads(raw[0])
    assert list(payload)[:4] == ["kind", "research_question", "database", "cohort_name"]
    # Reload returns the same records (order preserved).
    bank2 = ExperienceBank(path=path)
    assert [r.summary for r in bank2.records()] == ["A", "B"]


def test_bank_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "bank.jsonl"
    path.write_text(
        '{"kind": "concept_usage_hint", "research_question": "Q", '
        '"database": "miiv", "cohort_name": "c", "summary": "A"}\n'
        "this is not JSON\n"
        '{"kind": "concept_usage_hint", "research_question": "Q", '
        '"database": "miiv", "cohort_name": "c", "summary": "B"}\n',
        encoding="utf-8",
    )
    bank = ExperienceBank(path=path)
    summaries = sorted(r.summary for r in bank.records())
    assert summaries == ["A", "B"]


# ---------------------------------------------------------------------
# 4. Retrieve
# ---------------------------------------------------------------------


def test_retrieve_ranks_by_lexical_similarity(tmp_path: Path) -> None:
    bank = ExperienceBank(path=tmp_path / "bank.jsonl")
    bank.add(_rec(research_question="lactate sepsis mortality", summary="A"))
    bank.add(_rec(research_question="vasopressor hemodynamics", summary="B"))
    bank.add(_rec(research_question="lactate trajectory shock", summary="C"))
    hits = bank.retrieve(research_question="lactate sepsis outcome")
    summaries = [r.summary for r, _ in hits]
    # A (lactate + sepsis) should outrank C (only lactate); B is unrelated
    assert summaries[0] == "A"
    assert "B" not in summaries


def test_retrieve_database_match_adds_boost(tmp_path: Path) -> None:
    bank = ExperienceBank(path=tmp_path / "bank.jsonl")
    bank.add(_rec(research_question="lactate sepsis", database="eicu", summary="eicu"))
    bank.add(_rec(research_question="lactate sepsis", database="miiv", summary="miiv"))
    hits = bank.retrieve(research_question="lactate sepsis", database="miiv")
    assert hits[0][0].database == "miiv"


def test_retrieve_top_k_and_min_similarity(tmp_path: Path) -> None:
    bank = ExperienceBank(path=tmp_path / "bank.jsonl")
    for i in range(5):
        bank.add(_rec(research_question=f"lactate sepsis q{i}", summary=f"S{i}"))
    # Unrelated record below threshold
    bank.add(_rec(research_question="completely unrelated topic", summary="N"))
    hits = bank.retrieve(
        research_question="lactate sepsis target",
        top_k=2,
        min_similarity=0.3,
    )
    assert len(hits) == 2
    for rec, score in hits:
        assert score >= 0.3
        assert rec.summary != "N"


# ---------------------------------------------------------------------
# 5. Deterministic reflector
# ---------------------------------------------------------------------


def test_mine_concept_usage_hint_fires_when_all_clean() -> None:
    records = mine_experience_from_run(
        research_question="Does SOFA-2 stratify mortality?",
        database="miiv",
        cohort_name="sepsis3_aware",
        gates={
            "execution_complete": True,
            "evidence_complete": True,
            "numeric_verified": True,
            "analysis_validated": True,
        },
        findings=[],
        plan_step_ids=["01_cohort_summary", "02_sofa2_audit", "03_primary"],
        producer_run_id="run_test",
    )
    kinds = [r.kind for r in records]
    assert "concept_usage_hint" in kinds


def test_mine_no_concept_hint_when_a_gate_fails() -> None:
    records = mine_experience_from_run(
        research_question="Q",
        database="miiv",
        cohort_name="c",
        gates={
            "execution_complete": True,
            "evidence_complete": False,  # broken
            "numeric_verified": True,
            "analysis_validated": True,
        },
        findings=[],
        plan_step_ids=["01_sofa2_audit"],
    )
    assert all(r.kind != "concept_usage_hint" for r in records)


def test_mine_no_concept_hint_when_error_finding_present() -> None:
    records = mine_experience_from_run(
        research_question="Q",
        database="miiv",
        cohort_name="c",
        gates={
            "execution_complete": True,
            "evidence_complete": True,
            "numeric_verified": True,
            "analysis_validated": True,
        },
        findings=[{"severity": "error", "message": "boom"}],
        plan_step_ids=["01_sofa2_audit"],
    )
    assert all(r.kind != "concept_usage_hint" for r in records)


def test_mine_failure_counter_example_per_superseded_step() -> None:
    records = mine_experience_from_run(
        research_question="Q",
        database="eicu",
        cohort_name="c",
        gates={
            "execution_complete": True,
            "evidence_complete": True,
            "numeric_verified": True,
            "analysis_validated": True,
        },
        findings=[],
        superseded_errors=[
            {"step_id": "03_complete_case", "message": "Error code: 502"},
            {"step_id": "03_complete_case", "message": "Request timed out."},
            {"step_id": "04_primary", "message": "boom"},
        ],
        plan_step_ids=["03_complete_case", "04_primary"],
    )
    counter = [r for r in records if r.kind == "failure_counter_example"]
    # One per distinct step, sorted
    assert [r.detail["step_id"] for r in counter] == [
        "03_complete_case",
        "04_primary",
    ]


def test_mine_is_deterministic_on_same_inputs() -> None:
    args = dict(
        research_question="Q",
        database="miiv",
        cohort_name="c",
        gates={
            "execution_complete": True,
            "evidence_complete": True,
            "numeric_verified": True,
            "analysis_validated": True,
        },
        findings=[],
        plan_step_ids=["01_cohort_summary", "02_sofa2_audit"],
    )
    r1 = mine_experience_from_run(**args)
    r2 = mine_experience_from_run(**args)
    # Ignore produced_at (timestamp); summaries + details identical
    assert [r.summary for r in r1] == [r.summary for r in r2]
    assert [r.detail for r in r1] == [r.detail for r in r2]


# ---------------------------------------------------------------------
# 6. Pipeline integration
# ---------------------------------------------------------------------


def _make_pipeline_with_bank(tmp_path: Path, enable: bool):
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from easyicu.research_agent.providers.mocks import MockLLMClient

    return ResearchAgentPipeline(
        workdir=tmp_path,
        llm=MockLLMClient(),
        enable_experience_bank=enable,
        experience_bank_path=str(tmp_path / "bank.jsonl"),
    )


def _fake_run_status() -> dict:
    return {
        "gates": {
            "execution_complete": True,
            "evidence_complete": True,
            "numeric_verified": True,
            "analysis_validated": True,
        },
        "findings": [],
        "superseded_errors": [
            {"step_id": "03_complete_case", "message": "Error code: 502"},
        ],
        "plan_steps": [
            {"step_id": "01_cohort_summary"},
            {"step_id": "02_sofa2_audit"},
            {"step_id": "03_complete_case"},
        ],
    }


def _fake_context(question: str = "Does SOFA-2 predict mortality?"):
    from types import SimpleNamespace

    return SimpleNamespace(
        research_question=question,
        cohort=SimpleNamespace(cohort_name="sepsis3_aware"),
        target_outcome="death",
    )


def test_pipeline_reflect_and_persist_writes_bank(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_abc"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps(_fake_run_status()), encoding="utf-8"
    )
    pipeline = _make_pipeline_with_bank(tmp_path, enable=True)
    records = pipeline.reflect_and_persist_experience(
        run_dir=run_dir,
        context=_fake_context(),
        database="miiv",
        cohort_name="sepsis3_aware",
    )
    # At least one of each kind given the fake run_status above
    kinds = {r.kind for r in records}
    assert "concept_usage_hint" in kinds
    assert "failure_counter_example" in kinds
    # Bank file written
    bank_path = tmp_path / "bank.jsonl"
    assert bank_path.exists() and bank_path.read_text(encoding="utf-8").strip()


def test_pipeline_reflect_noop_when_flag_disabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_abc"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps(_fake_run_status()), encoding="utf-8"
    )
    pipeline = _make_pipeline_with_bank(tmp_path, enable=False)
    records = pipeline.reflect_and_persist_experience(
        run_dir=run_dir,
        context=_fake_context(),
        database="miiv",
        cohort_name="sepsis3_aware",
    )
    assert records == []
    assert not (tmp_path / "bank.jsonl").exists()


def test_pipeline_retrieve_returns_empty_when_disabled(tmp_path: Path) -> None:
    pipeline = _make_pipeline_with_bank(tmp_path, enable=False)
    assert pipeline.retrieve_experience_hints(research_question="Q") == []


def test_pipeline_retrieve_returns_hits_when_enabled(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.jsonl"
    # Seed the bank manually with a relevant record.
    seed = ExperienceBank(path=bank_path)
    seed.add(
        ExperienceRecord(
            kind="concept_usage_hint",
            research_question="Does SOFA-2 stratify mortality on MIMIC-IV?",
            database="miiv",
            cohort_name="sepsis3_aware",
            summary="Prefer load_sepsis3 over manual ICD regex.",
        )
    )
    pipeline = _make_pipeline_with_bank(tmp_path, enable=True)
    hits = pipeline.retrieve_experience_hints(
        research_question="Does SOFA-2 predict mortality on MIMIC-IV?",
        database="miiv",
    )
    assert hits, "expected at least one lexical match"
    assert hits[0][0].summary.startswith("Prefer load_sepsis3")


def test_pipeline_does_not_surface_unreviewed_experience_to_planner_prompt(
    ra,
    synthetic_cohort,
    tmp_path: Path,
) -> None:
    bank_path = tmp_path / "bank.jsonl"
    seed = ExperienceBank(path=bank_path)
    seed.add(
        ExperienceRecord(
            kind="concept_usage_hint",
            research_question="Does SOFA-2 predict mortality on MIMIC-IV?",
            database="miiv",
            cohort_name="sepsis3_aware",
            summary="Prefer load_sepsis3 over manual ICD regex.",
        )
    )

    from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient

    planner = PatternScriptedMockLLMClient([], contextual_default=True)
    router = ra.LLMRouter(default=ra.MockLLMClient(), planner=planner)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "runs",
        llm=router,
        enable_memory=True,
        enable_experience_bank=True,
        experience_bank_path=bank_path,
        runner_kind="subprocess",
    )

    result = pipeline.run(
        question="Does SOFA-2 predict mortality on MIMIC-IV?",
        cohort=synthetic_cohort,
        cohort_name="sepsis3_aware",
        database="miiv",
        target_outcome="death",
        stop_after_analysis=True,
    )

    planner_prompts = [
        message.content
        for messages, _kwargs in planner.calls
        for message in messages
        if message.role == "user"
        and "ICU-AWARE RESEARCH PLAN" in message.content
    ]
    assert planner_prompts, "planner prompt was not captured"
    prompt = "\n".join(planner_prompts)
    assert "Experience hints for planner:" not in prompt
    assert "Prefer load_sepsis3 over manual ICD regex." not in prompt

    audit_path = Path(result.workdir) / "experience_hints.md"
    assert not audit_path.exists()
    quarantined = list((tmp_path / "runs" / ".memory_v2").rglob("*.json"))
    assert len(quarantined) == 1
    payload = json.loads(quarantined[0].read_text(encoding="utf-8"))
    assert payload["review_status"] == "quarantined"
    assert payload["namespace"].startswith("run_lessons/quarantine/")


def test_permissioned_quarantine_mirror_failure_is_nonfatal_to_completed_run(
    ra,
    synthetic_cohort,
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fail_quarantine(*_args, **_kwargs):
        raise OSError("simulated quarantine storage failure")

    monkeypatch.setattr(
        "easyicu.research_agent.orchestration.finalize.quarantine_run_lesson",
        _fail_quarantine,
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "runs",
        llm=ra.MockLLMClient(),
        enable_memory=True,
        runner_kind="subprocess",
    )

    result = pipeline.run(
        question="Does SOFA-2 predict mortality on MIMIC-IV?",
        cohort=synthetic_cohort,
        cohort_name="sepsis3_aware",
        database="miiv",
        target_outcome="death",
        stop_after_analysis=True,
    )

    assert Path(result.manifest_path).exists()


def test_pipeline_reflect_handles_missing_run_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_missing"
    run_dir.mkdir()
    pipeline = _make_pipeline_with_bank(tmp_path, enable=True)
    records = pipeline.reflect_and_persist_experience(
        run_dir=run_dir,
        context=_fake_context(),
        database="miiv",
        cohort_name="sepsis3_aware",
    )
    assert records == []


def test_pipeline_reflect_is_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_abc"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps(_fake_run_status()), encoding="utf-8"
    )
    pipeline = _make_pipeline_with_bank(tmp_path, enable=True)
    pipeline.reflect_and_persist_experience(
        run_dir=run_dir,
        context=_fake_context(),
        database="miiv",
        cohort_name="sepsis3_aware",
    )
    bank = ExperienceBank(path=tmp_path / "bank.jsonl")
    n_after_first = len(bank.records())
    pipeline.reflect_and_persist_experience(
        run_dir=run_dir,
        context=_fake_context(),
        database="miiv",
        cohort_name="sepsis3_aware",
    )
    bank2 = ExperienceBank(path=tmp_path / "bank.jsonl")
    assert len(bank2.records()) == n_after_first
