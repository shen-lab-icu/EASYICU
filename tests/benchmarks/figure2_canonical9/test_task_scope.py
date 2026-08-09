from __future__ import annotations

import json

import pytest

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from benchmarks.figure2_canonical9.prompt_preflight import (
    PromptPreflightError,
    _strict_rows,
)
from benchmarks.figure2_canonical9.task_scope import (
    Canonical9TaskScopeError,
    canonical_task_scope,
)


def test_task_scope_defaults_to_full_suite_and_orders_subsets() -> None:
    assert canonical_task_scope(None) == FIGURE2_TASK_IDS
    assert canonical_task_scope([]) == FIGURE2_TASK_IDS
    assert canonical_task_scope(
        ["h3_trajectory_clustering", "e1_sepsis3_prevalence_mortality"]
    ) == (
        "e1_sepsis3_prevalence_mortality",
        "h3_trajectory_clustering",
    )


@pytest.mark.parametrize(
    ("task_ids", "message"),
    [
        ([""], "non-empty"),
        (["e1_sepsis3_prevalence_mortality"] * 2, "unique"),
        (["not_a_task"], "unknown Canonical9 task"),
    ],
)
def test_task_scope_rejects_invalid_selection(task_ids, message) -> None:
    with pytest.raises(Canonical9TaskScopeError, match=message):
        canonical_task_scope(task_ids)


def test_prompt_rows_require_full_suite_unless_subset_is_explicit(tmp_path) -> None:
    path = tmp_path / "e1.jsonl"
    path.write_text(
        json.dumps({"key": "e1_sepsis3_prevalence_mortality"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PromptPreflightError, match="task order mismatch"):
        _strict_rows(path)
    rows = _strict_rows(path, task_ids=["e1_sepsis3_prevalence_mortality"])
    assert [row["key"] for row in rows] == [
        "e1_sepsis3_prevalence_mortality"
    ]
