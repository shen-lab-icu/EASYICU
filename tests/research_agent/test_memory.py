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


def test_meta_planner_digest_ranks_skill_keys(ra, tmp_path: Path):
    mem = ra.RunMemory(root=tmp_path)
    mem.record(
        run_id="run_sofa",
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        database="miiv",
        target_outcome="death",
        findings=[
            ra.schema.ValidationFinding(
                validator="statistical_validator",
                severity="warning",
                message="sofa2 zero missingness anomaly",
            )
        ],
        workdir=tmp_path / "run_sofa",
    )
    ranking = mem.rank_skill_keys(
        skill_keys=["association_analysis", "prediction_model"],
        research_question="SOFA mortality",
        database="miiv",
        target_outcome="death",
    )
    assert ranking[0][0] == "association_analysis"
    digest = mem.meta_planner_digest(
        skill_keys=["association_analysis", "prediction_model"],
        research_question="SOFA mortality",
        database="miiv",
        target_outcome="death",
    )
    assert "Meta-planner skill ranking" in digest


def test_run_memory_distills_strategy_cards(ra, tmp_path: Path):
    mem = ra.RunMemory(root=tmp_path)
    mem.record(
        run_id="run_sofa_strategy",
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        database="miiv",
        target_outcome="death",
        findings=[
            ra.schema.ValidationFinding(
                validator="statistical_validator",
                severity="warning",
                message="sofa2==0 outcome rate suggests component missingness in score strata",
            )
        ],
        workdir=tmp_path / "run_sofa_strategy",
    )

    cards = mem.relevant_strategy_cards(
        research_question="SOFA mortality",
        database="miiv",
        target_outcome="death",
    )
    assert cards
    assert cards[0].strategy_id == "ordinal_score_missingness_audit"
    assert "miiv" in cards[0].applicable_databases
    assert "sofa2" in cards[0].concept_dependencies

    digest = mem.strategy_digest_for_prompt(
        research_question="SOFA mortality",
        database="miiv",
        target_outcome="death",
    )
    assert "StrategyCards" in digest
    assert "component availability" in digest
    assert "databases: miiv" in digest


def test_strategy_cards_hide_blocked_concept_dependencies(ra, tmp_path: Path, monkeypatch):
    from easyicu.research_agent import memory as memory_module

    mem = ra.RunMemory(root=tmp_path)
    now = "2026-01-01T00:00:00+00:00"
    mem._upsert_strategy_card(
        ra.StrategyCard(
            strategy_id="blocked_urine_strategy",
            task_family="aki_outcome_association",
            trigger_tokens=["aki", "mortality"],
            recommended_plan=["Use urine-output staging."],
            guardrails=["Requires urine output."],
            supporting_run_ids=["run_blocked"],
            updated_at=now,
            applicable_databases=["miiv"],
            concept_dependencies=["urine"],
        )
    )
    mem._upsert_strategy_card(
        ra.StrategyCard(
            strategy_id="plain_aki_strategy",
            task_family="aki_outcome_association",
            trigger_tokens=["aki", "mortality"],
            recommended_plan=["Use creatinine-only feasibility framing when needed."],
            guardrails=["Declare reduced concept set."],
            supporting_run_ids=["run_plain"],
            updated_at=now,
            applicable_databases=["miiv"],
        )
    )

    def fake_feasibility(*, concepts, databases):
        return {
            "concept_dependencies": list(concepts),
            "cross_database_feasibility": {str(databases[0]): "blocked"},
            "degraded_reason": {str(databases[0]): "urine unavailable"},
            "availability": {},
        }

    monkeypatch.setattr(
        memory_module,
        "hypothesis_cross_database_feasibility",
        fake_feasibility,
    )

    cards = mem.relevant_strategy_cards(
        research_question="AKI mortality",
        database="miiv",
        target_outcome="death",
    )

    assert "blocked_urine_strategy" not in {card.strategy_id for card in cards}
    assert "plain_aki_strategy" in {card.strategy_id for card in cards}
