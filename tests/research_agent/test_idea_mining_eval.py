from __future__ import annotations

import easyicu.research_agent as ra
from easyicu.research_agent.idea_mining_eval import (
    default_idea_quality_eval_path,
    load_idea_quality_eval_set,
    score_idea_quality_predictions,
    summarize_idea_quality_eval_set,
)


def test_load_default_idea_quality_eval_set_and_summarize() -> None:
    eval_set = load_idea_quality_eval_set()
    summary = summarize_idea_quality_eval_set(eval_set)

    assert default_idea_quality_eval_path().exists()
    assert eval_set.schema_version == "easyicu.idea_quality_eval/1"
    assert summary["n_items"] >= 10
    assert summary["go_no_go_counts"]["hold"] >= 5
    assert summary["feasibility_route_counts"]["current_export_executable"] == 1
    assert summary["trap_tag_counts"]["false_gap_risk"] >= 2


def test_score_idea_quality_predictions_counts_false_gap_and_false_recommend() -> None:
    eval_set = load_idea_quality_eval_set()
    predictions = [
        {
            "item_id": "already_done_direct_same_topic",
            "novelty_label": "apparently_gap",
            "go_no_go": "recommend",
            "feasibility_route": "current_export_executable",
        },
        {
            "item_id": "sparse_executable_screened",
            "novelty_label": "sparse",
            "go_no_go": "recommend",
            "feasibility_route": "current_export_executable",
        },
    ]

    score = score_idea_quality_predictions(eval_set, predictions)

    assert score.n_items == len(eval_set.items)
    assert score.n_predictions == 2
    assert score.false_gap_count == 1
    assert score.false_recommend_count == 1
    assert score.all_available_fields_correct == 1
    assert "not_measured_in_database" in score.missing_prediction_ids


def test_package_lazy_exports_idea_quality_helpers() -> None:
    assert callable(ra.load_idea_quality_eval_set)
    assert callable(ra.score_idea_quality_predictions)
    assert ra.IDEA_QUALITY_EVAL_SCHEMA_VERSION == "easyicu.idea_quality_eval/1"
