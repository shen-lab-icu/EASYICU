"""The Dev9 paper anchors are a gold-free post-run review contract."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "figure2_canonical9"
    / "dev9_comparator_shadow_review_v1.json"
)

EXPECTED_DIMENSIONS = {
    "study_population",
    "time_zero_and_windows",
    "variable_operationalization",
    "missingness_and_censoring",
    "primary_model_and_sensitivities",
    "table_and_figure_completeness",
    "conclusion_boundaries",
}

FORBIDDEN_ANSWER_KEYS = {
    "gold_answer",
    "numeric_target",
    "expected_value",
    "expected_effect",
    "expected_effect_direction",
    "expected_prevalence",
    "expected_odds_ratio",
    "expected_hazard_ratio",
    "result_tolerance",
}


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_shadow_review_covers_canonical9_and_all_seven_dimensions() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert payload["audience"] == "evaluator_only"
    assert payload["agent_visibility"] == "forbidden_before_execution"
    assert set(payload["dimensions"]) == EXPECTED_DIMENSIONS
    assert tuple(task["task_id"] for task in payload["tasks"]) == FIGURE2_TASK_IDS
    assert all(1 <= len(task["anchors"]) <= 2 for task in payload["tasks"])


def test_shadow_review_cannot_become_a_numeric_answer_key() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert "numeric_gold_answer" in payload["use_policy"]["forbidden"]
    assert "expected_effect_direction" in payload["use_policy"]["forbidden"]
    assert "result_similarity_pass_fail" in payload["use_policy"]["forbidden"]
    assert "magnitude or direction is never" in payload["acceptance_rule"]
    for mapping in _walk(payload):
        assert FORBIDDEN_ANSWER_KEYS.isdisjoint(mapping)


def test_anchor_records_contain_locator_only_not_copied_results() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    forbidden_anchor_fields = {
        "abstract",
        "results",
        "effect_estimate",
        "effect_direction",
        "figure_template",
    }

    for task in payload["tasks"]:
        assert task["focus"]
        for anchor in task["anchors"]:
            assert set(anchor) == {"citation_id", "url"}
            assert forbidden_anchor_fields.isdisjoint(anchor)
            assert anchor["url"].startswith(
                ("https://pubmed.ncbi.nlm.nih.gov/", "https://pmc.ncbi.nlm.nih.gov/")
            )
