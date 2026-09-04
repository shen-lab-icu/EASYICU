from easyicu.research_agent.discovery.idea_mining_construct_answerability import (
    assess_idea_constructs,
    assess_research_construct,
)


def test_direct_and_catalog_derived_concepts_stay_distinct() -> None:
    direct = assess_research_construct(
        "rass",
        database="miiv",
        available_concepts={"rass"},
    )
    derived = assess_research_construct(
        "fluid_balance_cumulative",
        database="miiv",
        available_concepts={"fluid_balance_cumulative"},
    )

    assert direct.resolution_kind == "direct_observed"
    assert direct.source_state == "present_in_export"
    assert direct.verdict == "ready"
    assert derived.resolution_kind == "validated_derived"
    assert derived.source_state == "present_in_export"
    assert derived.verdict == "ready"


def test_sedation_stop_is_not_equated_with_last_drug_record() -> None:
    result = assess_research_construct(
        "停镇静药",
        database="miiv",
        available_concepts={"propofol_rate", "rass"},
    )

    assert result.construct_id == "sedation_discontinuation"
    assert result.resolution_kind == "event_reconstructable"
    assert result.verdict == "needs_review"
    assert result.materialized is False
    assert "最后一条给药记录" in result.semantic_warning
    assert "medication_interval_end_authority" in result.unresolved_requirements


def test_sedation_presence_without_rate_is_proxy_only() -> None:
    result = assess_research_construct(
        "sedation discontinuation",
        database="miiv",
        available_concepts={"propofol", "rass"},
    )

    assert result.resolution_kind == "proxy_only"
    assert result.verdict == "needs_review"
    assert result.source_state == "proxy_only"


def test_extubation_failure_requires_episode_materialization() -> None:
    result = assess_research_construct(
        "拔管失败",
        database="miiv",
        available_concepts={"mech_vent", "vent_end", "death"},
    )

    assert result.construct_id == "extubation_failure"
    assert result.resolution_kind == "event_reconstructable"
    assert result.verdict == "needs_review"
    assert result.materialized is False
    assert "呼吸机记录中断" in result.semantic_warning


def test_ventilator_free_days_is_not_an_extubation_event() -> None:
    result = assess_research_construct(
        "extubation",
        database="miiv",
        available_concepts={"vent_free_days_28"},
    )

    assert result.resolution_kind == "proxy_only"
    assert result.verdict == "needs_review"


def test_cumulative_balance_can_be_derived_from_registered_primitives() -> None:
    result = assess_research_construct(
        "累计液体平衡",
        database="miiv",
        available_concepts={"total_input_ml", "urine"},
    )

    assert result.construct_id == "cumulative_fluid_balance"
    assert result.resolution_kind == "validated_derived"
    assert result.source_state == "constructible_from_export"
    assert result.verdict == "needs_review"
    assert result.required_primitives == ("total_input_ml", "urine")


def test_metadata_only_idea_preflight_is_informative_but_never_ready() -> None:
    rows = assess_idea_constructs(
        "ICU患者镇静药减量后持续昏迷，想研究停药后的清醒恢复时间",
        mapped_concepts=("propofol_rate", "rass"),
    )

    by_id = {row["construct_id"]: row for row in rows}
    assert by_id["sedation_discontinuation"]["verdict"] == "needs_review"
    assert by_id["awakening_after_sedation"]["verdict"] == "needs_review"
    assert by_id["sedation_discontinuation"]["source_state"] == "source_not_selected"
    assert all(row["verdict"] != "ready" for row in rows)


def test_copilot_paraphrase_still_resolves_sedation_and_awakening() -> None:
    rows = assess_idea_constructs(
        "ICU患者停用镇静药后持续意识不清，想研究延迟苏醒",
        mapped_concepts=(),
    )

    assert {row["construct_id"] for row in rows} >= {
        "sedation_discontinuation",
        "awakening_after_sedation",
    }


def test_unknown_construct_fails_closed() -> None:
    result = assess_research_construct(
        "unregistered bedside intuition score",
        database="miiv",
        available_concepts={"map", "hr"},
    )

    assert result.resolution_kind == "unavailable"
    assert result.source_state == "not_in_database"
    assert result.verdict == "blocked"
