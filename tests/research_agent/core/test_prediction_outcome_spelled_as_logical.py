"""A binary outcome spelled True/False is the same outcome as 0/1.

Data preparation may store an event flag as a logical column.  The static
prediction owner used to refuse that dtype outright, so a complete, closed
binary outcome stopped the primary step before any fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.prediction_model_executor import (
    run_prediction_model,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context(row_count: int) -> ResearchContext:
    return ResearchContext(
        research_question="Predict a binary outcome from prespecified variables.",
        cohort=CohortDescriptor(
            cohort_name="logical_outcome_fixture",
            database="synthetic",
            n_stays=row_count,
            id_columns=["patient_stay_id"],
            outcome_columns=["event"],
            provenance={
                "replacement_row_identity": {
                    "output_identity_column": "patient_stay_id",
                    "mapping_file_sha256": "b" * 64,
                    "patient_group_derivation": {
                        "algorithm": "prefix_before_:s",
                        "delimiter": ":s",
                    },
                }
            },
        ),
        variables=[
            ConceptDescriptor(name="age", dtype="float64"),
            ConceptDescriptor(name="marker", dtype="float64"),
            ConceptDescriptor(
                name="event",
                role=VariableRole.OUTCOME,
                dtype="bool",
                observed_domain={"n_unique": 2, "is_binary": True},
            ),
        ],
        target_outcome="event",
    )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    subjects = np.repeat(np.arange(120), np.where(np.arange(120) % 5 == 0, 2, 1))
    stays = np.concatenate(
        [np.arange(count) + 1 for count in np.bincount(subjects)]
    )
    age = rng.normal(62, 11, len(subjects))
    marker = rng.normal(0, 1, len(subjects))
    logit = -1.5 + 0.03 * (age - 60) + 0.8 * marker
    event = rng.random(len(subjects)) < 1.0 / (1.0 + np.exp(-logit))
    event[:10] = np.arange(10) % 2 == 0
    frame = pd.DataFrame(
        {
            "patient_stay_id": [f"p{s}:s{k}" for s, k in zip(subjects, stays, strict=True)],
            "age": age,
            "marker": marker,
            "event": event,
        }
    )
    frame.loc[frame.index[::11], "marker"] = np.nan
    return frame


def _run(tmp_path: Path, frame: pd.DataFrame, name: str) -> dict:
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "research_context.json").write_text(
        _context(len(frame)).model_dump_json(), encoding="utf-8"
    )
    cohort = run_dir / "cohort.csv"
    frame.to_csv(cohort, index=False)
    return run_prediction_model(
        frame=frame,
        declared_columns=("age", "marker", "event"),
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=cohort,
        out_dir=run_dir / "out",
        run_dir=run_dir,
        step_id="primary_model",
    )


def test_a_logical_outcome_fits_exactly_as_its_zero_one_spelling(tmp_path: Path) -> None:
    logical = _frame()
    numeric = logical.assign(event=logical["event"].astype(int))
    assert pd.api.types.is_bool_dtype(logical["event"].dtype)

    _run(tmp_path, logical, "logical")
    _run(tmp_path, numeric, "numeric")

    scores = [
        pd.read_csv(tmp_path / name / "out" / "prediction_scores.csv")
        for name in ("logical", "numeric")
    ]
    pd.testing.assert_frame_equal(scores[0], scores[1])
    # The scores table carries the numeric outcome every validator requires.
    assert set(scores[0]["outcome"]) == {0, 1}


@pytest.mark.parametrize(
    "event",
    [
        pd.array([True, False, None] * 44, dtype="boolean"),
        np.array(["yes", "no"] * 66, dtype=object),
    ],
    ids=["logical_with_a_gap", "words"],
)
def test_an_incomplete_or_unspelled_outcome_is_still_refused(tmp_path: Path, event) -> None:
    frame = _frame().iloc[:132].copy()
    frame["event"] = event

    with pytest.raises(RuntimeError, match="prediction outcome 'event'"):
        _run(tmp_path, frame, "refused")
