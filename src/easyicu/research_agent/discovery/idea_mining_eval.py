"""Small regression harness for idea-mining quality checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import easyicu
from pydantic import BaseModel, ConfigDict, Field

from .idea_mining_schema import GoNoGoDecision, NoveltyLabel


IDEA_QUALITY_EVAL_SCHEMA_VERSION = "easyicu.idea_quality_eval/1"


class IdeaQualityExpectedLabel(BaseModel):
    """Expected triage labels for one manually curated eval item."""

    model_config = ConfigDict(extra="forbid")

    novelty_label: Optional[NoveltyLabel] = None
    go_no_go: Optional[GoNoGoDecision] = None
    feasibility_route: Optional[str] = None


class IdeaQualityEvalItem(BaseModel):
    """One benchmark-like idea-quality item."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    title: str
    source_route: str
    literature_idea: Dict[str, object] = Field(default_factory=dict)
    expected: IdeaQualityExpectedLabel
    trap_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class IdeaQualityEvalSet(BaseModel):
    """A bounded fixture set for S4-S6 idea-quality regression."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = IDEA_QUALITY_EVAL_SCHEMA_VERSION
    created_for_task: str
    items: List[IdeaQualityEvalItem]


class IdeaQualityPrediction(BaseModel):
    """Predicted labels for one eval item."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    novelty_label: Optional[NoveltyLabel] = None
    go_no_go: Optional[GoNoGoDecision] = None
    feasibility_route: Optional[str] = None


class IdeaQualityScorecard(BaseModel):
    """Aggregate exact-match diagnostics for an eval run."""

    model_config = ConfigDict(extra="forbid")

    n_items: int
    n_predictions: int
    novelty_label_correct: int = 0
    go_no_go_correct: int = 0
    feasibility_route_correct: int = 0
    all_available_fields_correct: int = 0
    false_gap_count: int = 0
    false_recommend_count: int = 0
    missing_prediction_ids: List[str] = Field(default_factory=list)
    unexpected_prediction_ids: List[str] = Field(default_factory=list)


def default_idea_quality_eval_path() -> Path:
    """Return the repo-local default fixture path."""

    package_file = Path(easyicu.__file__).resolve()
    return (
        package_file.parents[2]
        / "benchmarks"
        / "idea_mining"
        / "quality_eval_set.json"
    )


def load_idea_quality_eval_set(
    path: Optional[str | Path] = None,
) -> IdeaQualityEvalSet:
    """Load the curated idea-quality eval set."""

    eval_path = Path(path) if path is not None else default_idea_quality_eval_path()
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    return IdeaQualityEvalSet.model_validate(payload)


def summarize_idea_quality_eval_set(eval_set: IdeaQualityEvalSet) -> Dict[str, object]:
    """Return bounded counts useful for smoke-test reporting."""

    novelty_counts: Dict[str, int] = {}
    go_no_go_counts: Dict[str, int] = {}
    route_counts: Dict[str, int] = {}
    trap_counts: Dict[str, int] = {}
    for item in eval_set.items:
        if item.expected.novelty_label:
            key = item.expected.novelty_label
            novelty_counts[key] = novelty_counts.get(key, 0) + 1
        if item.expected.go_no_go:
            key = item.expected.go_no_go
            go_no_go_counts[key] = go_no_go_counts.get(key, 0) + 1
        if item.expected.feasibility_route:
            key = item.expected.feasibility_route
            route_counts[key] = route_counts.get(key, 0) + 1
        for tag in item.trap_tags:
            trap_counts[tag] = trap_counts.get(tag, 0) + 1
    return {
        "schema_version": eval_set.schema_version,
        "n_items": len(eval_set.items),
        "novelty_label_counts": novelty_counts,
        "go_no_go_counts": go_no_go_counts,
        "feasibility_route_counts": route_counts,
        "trap_tag_counts": trap_counts,
    }


def score_idea_quality_predictions(
    eval_set: IdeaQualityEvalSet,
    predictions: Sequence[IdeaQualityPrediction | Mapping[str, object]],
) -> IdeaQualityScorecard:
    """Score predicted labels against the curated eval set."""

    parsed = [
        (
            pred
            if isinstance(pred, IdeaQualityPrediction)
            else IdeaQualityPrediction.model_validate(pred)
        )
        for pred in predictions
    ]
    pred_by_id = {pred.item_id: pred for pred in parsed}
    item_ids = {item.item_id for item in eval_set.items}
    score = IdeaQualityScorecard(
        n_items=len(eval_set.items),
        n_predictions=len(parsed),
        missing_prediction_ids=sorted(item_ids - set(pred_by_id)),
        unexpected_prediction_ids=sorted(set(pred_by_id) - item_ids),
    )
    for item in eval_set.items:
        pred = pred_by_id.get(item.item_id)
        if pred is None:
            continue
        expected = item.expected
        field_results: List[bool] = []
        if expected.novelty_label is not None:
            ok = pred.novelty_label == expected.novelty_label
            score.novelty_label_correct += int(ok)
            field_results.append(ok)
            if expected.novelty_label == "already_done" and pred.novelty_label in {
                "sparse",
                "apparently_gap",
            }:
                score.false_gap_count += 1
        if expected.go_no_go is not None:
            ok = pred.go_no_go == expected.go_no_go
            score.go_no_go_correct += int(ok)
            field_results.append(ok)
            if expected.go_no_go != "recommend" and pred.go_no_go == "recommend":
                score.false_recommend_count += 1
        if expected.feasibility_route is not None:
            ok = pred.feasibility_route == expected.feasibility_route
            score.feasibility_route_correct += int(ok)
            field_results.append(ok)
        if field_results and all(field_results):
            score.all_available_fields_correct += 1
    return score


__all__ = [
    "IDEA_QUALITY_EVAL_SCHEMA_VERSION",
    "IdeaQualityEvalItem",
    "IdeaQualityEvalSet",
    "IdeaQualityExpectedLabel",
    "IdeaQualityPrediction",
    "IdeaQualityScorecard",
    "default_idea_quality_eval_path",
    "load_idea_quality_eval_set",
    "score_idea_quality_predictions",
    "summarize_idea_quality_eval_set",
]
