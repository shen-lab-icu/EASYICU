"""Three contrasts must not arrive on the figure under one label.

The adjusted-association producer already writes what each row IS:
``exposure_level``, ``reference_level``, ``contrast`` and
``is_primary_contrast`` are four of the twenty columns in its own product
contract.  The renderer reads twelve of those twenty and none of these four, and
its row label falls back to ``analysis_set`` then ``estimator_kind`` -- values
that are IDENTICAL across the rows of one model.

So a four-level ordinal exposure renders three rows reading

    complete_case
    complete_case
    complete_case

and no reader can tell which one is stage 3 versus stage 0.

MEASURED over the recorded corpus: 99 emitted estimate tables carry these
columns and 33 of them have more than one row -- 32 with three, one with four.
A third of every adjusted-association figure the host can draw is affected, and
staged exposures (KDIGO stage, SOFA stage) are exactly the tasks that produce
them.

The rule the labels follow, in order: the producer's own ``contrast`` sentence;
failing that ``<level> vs <reference>`` built from the two columns beside it;
only then the analysis set or estimator that describes the whole model rather
than the row.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
)
from easyicu.research_agent.execution.runners.adjusted_association_figure_executor import (  # noqa: E501
    ADJUSTED_ASSOCIATION_FIGURE_INPUT,
    run_adjusted_association_figure,
)

from .test_adjusted_association_figure_executor import (
    _REAL_ROW,
    _write_bound_table,
)


def _staged_rows():
    """One model, three contrasts against the same reference.

    Shaped like the producer's own output for an ordinal exposure: the estimate
    rises with the stage, every row shares the model-level fields, and exactly
    one row is the primary contrast.
    """

    base = {
        **_REAL_ROW,
        "exposure": "aki_stage_max",
        "requirement_id": "primary_logistic_aki_stage_max_death",
        "reference_level": "0",
    }
    return [
        {
            **base,
            "estimate": "1.84",
            "ci_low": "1.61",
            "ci_high": "2.10",
            "exposure_level": "1",
            "contrast": "stage 1 vs stage 0",
            "is_primary_contrast": "False",
        },
        {
            **base,
            "estimate": "3.02",
            "ci_low": "2.55",
            "ci_high": "3.58",
            "exposure_level": "2",
            "contrast": "stage 2 vs stage 0",
            "is_primary_contrast": "False",
        },
        {
            **base,
            "estimate": "6.48",
            "ci_low": "6.02",
            "ci_high": "6.97",
            "exposure_level": "3",
            "contrast": "stage 3 vs stage 0",
            "is_primary_contrast": "True",
        },
    ]


def _render(tmp_path: Path, rows):
    run_dir, manifest = _write_bound_table(tmp_path, rows)
    summary = run_adjusted_association_figure(
        out_dir=tmp_path / "out",
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id="08_adjusted_effect_figure",
        figure_product="adjusted_effect",
    )
    source = (tmp_path / "out" / "adjusted_effect_source_data.csv").read_text(
        encoding="utf-8"
    )
    return summary, source


def test_the_three_contrasts_do_not_share_one_label(tmp_path):
    summary, source = _render(tmp_path, _staged_rows())

    assert summary["estimates_drawn"] == 3
    labels = summary.get("row_labels")
    assert labels is not None, "the figure does not report what it labelled its rows"
    assert len(set(labels)) == 3, labels


def test_each_label_names_the_contrast_the_producer_wrote(tmp_path):
    """The producer's own sentence, verbatim -- not one rebuilt from the levels.

    A first version only asserted the labels were distinct and contained the
    level, so a mutation that ignored ``contrast`` entirely survived: the
    ``<level> vs <reference>`` fallback produced distinct labels too. The
    producer writes the sentence a reader should see, and that sentence is what
    must reach the axis.
    """

    summary, _source = _render(tmp_path, _staged_rows())

    labels = [str(label) for label in summary["row_labels"]]
    assert labels == [
        "stage 1 vs stage 0",
        "stage 2 vs stage 0",
        "stage 3 vs stage 0",
    ], labels


def test_the_primary_contrast_is_identifiable_on_the_figure(tmp_path):
    """A staged figure whose headline row is anonymous cannot carry a claim."""

    summary, _source = _render(tmp_path, _staged_rows())

    primary = summary.get("primary_contrast_label")
    assert primary, "the figure does not say which row is the primary contrast"
    assert "3" in str(primary), primary
    assert list(summary["row_labels"]).count(primary) == 1


def test_a_single_row_model_is_labelled_exactly_as_before(tmp_path):
    """The 66 one-row tables in the corpus must not change.

    Reading four more columns is only allowed to add information to rows that
    carry it; a model with one contrast has none, and its label is still the
    analysis set.
    """

    summary, _source = _render(tmp_path, [_REAL_ROW])

    assert summary["estimates_drawn"] == 1
    assert list(summary["row_labels"]) == ["complete_case"]


def test_a_row_that_declares_no_contrast_falls_back_rather_than_blanking(tmp_path):
    """A producer that fills only the level columns still gets a real label."""

    rows = _staged_rows()
    for row in rows:
        row["contrast"] = ""
    summary, _source = _render(tmp_path, rows)

    labels = [str(label) for label in summary["row_labels"]]
    assert len(set(labels)) == 3, labels
    for label in labels:
        assert label.strip(), labels
        assert "vs" in label, labels


def test_two_rows_claiming_to_be_primary_name_neither(tmp_path):
    """Naming the wrong row as the headline is worse than naming none.

    A mutation that took the first marked row survived until this existed --
    every other case has exactly one primary, so 'pick the first' and 'require
    exactly one' were indistinguishable.
    """

    rows = _staged_rows()
    rows[0]["is_primary_contrast"] = "True"
    summary, _source = _render(tmp_path, rows)

    assert summary["primary_contrast_label"] is None
    # The rows are still labelled; only the headline claim is withheld.
    assert len(set(summary["row_labels"])) == 3


def test_no_row_claiming_to_be_primary_names_none(tmp_path):
    rows = _staged_rows()
    for row in rows:
        row["is_primary_contrast"] = "False"
    summary, _source = _render(tmp_path, rows)

    assert summary["primary_contrast_label"] is None


def test_the_contrast_columns_are_carried_into_the_source_data(tmp_path):
    """A reader checking the figure must find the same four columns beside it."""

    _summary, source = _render(tmp_path, _staged_rows())

    header = next(csv.reader(source.splitlines()))
    for column in (
        "exposure_level",
        "reference_level",
        "contrast",
        "is_primary_contrast",
    ):
        assert column in header, header


def test_the_producer_really_declares_these_columns():
    """Anchors the whole file: this is the producer's own contract, not an invention."""

    for column in (
        "exposure_level",
        "reference_level",
        "contrast",
        "is_primary_contrast",
    ):
        assert column in ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS, column


def test_the_recorded_corpus_still_contains_multi_row_tables():
    """Stops being meaningful only if staged exposures stop being produced."""

    import pandas as pd

    corpus = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")
    if not corpus.exists():
        pytest.skip("recorded run corpus is not mounted")

    multi = single = 0
    for path in corpus.glob("batch_*/*/aware/run_*/steps/*/outputs/*.csv"):
        name = path.name
        if "association" not in name and "adjusted" not in name:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - a malformed CSV is not this test's subject
            continue
        if "contrast" not in frame.columns and "exposure_level" not in frame.columns:
            continue
        if len(frame) > 1:
            multi += 1
        else:
            single += 1

    if multi + single == 0:
        pytest.skip("no recorded estimates table carries contrast columns")
    assert multi > 0, "the corpus no longer contains a multi-contrast table"
    assert single > 0, "the one-row case must stay covered too"
