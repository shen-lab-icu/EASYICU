"""A confounder stated by its concept is read as its summary before time zero.

The study setup names a trial's confounders by the source's concept ids
(``crea``, ``adv_resp``).  The trial's extraction summarizes each over
``[0, T0)`` into the columns the cohort builder names (``crea_mean``,
``adv_resp_max``); no column carries the bare concept id.  The compile reads
such a confounder as one summary the host chooses -- an event status by
whether it was recorded, any other value by its mean -- states the choice on
the card, and records the column the weights read.  Only a confounder the
input holds no such column of waits for an extraction.  Whether a summary is
observed by time zero stays the timing owner's to prove: each synthetic column
carries the role the ICU rules give it, and a column of a role whose timing
that owner does not prove is not adjusted for, and says so.  Synthetic
contexts only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.icu_rules import classify_variable
from easyicu.research_agent.planning.target_trial_compile import CONFOUNDER_SUMMARIES
from easyicu.research_agent.schema import ConceptDescriptor
from tests.support.target_trial import (
    TIME_ZERO,
    compiled_target_trial,
    target_trial_context,
    target_trial_spec,
)

_BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
_BEFORE_T0 = f"icu_admission[0,{TIME_ZERO}]h"


def _value(name: str, concept: str, window: str = _BEFORE_T0) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name, role=classify_variable(name, "float64").role, dtype="float64",
        source_concept=concept, analysis_window=window,
    )


def _status(name: str, concept: str, window: str = _BEFORE_T0) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name, role=classify_variable(name, "int64", [0, 1]).role, dtype="int64",
        source_concept=concept, analysis_window=window, observed_domain=_BINARY,
    )


def _compiled(*confounders: str, extra: tuple[ConceptDescriptor, ...]):
    context = target_trial_context()
    context = context.model_copy(update={"variables": [*context.variables, *extra]})
    spec = target_trial_spec(
        confounders=[
            {"name": "age", "source": "question",
             "clinical_rationale": "Older patients are started later and die more often."},
            *(
                {"name": name, "source": "conversation",
                 "clinical_rationale": f"{name} changes both the start and the risk of death."}
                for name in confounders
            ),
        ]
    )
    return compiled_target_trial(context, spec=spec)


#: The extraction's summaries of the confounders a probe's setup stated.
_SUMMARIZED = (
    _value("crea_mean", "crea"), _value("crea_max", "crea"),
    _value("bili_mean", "bili"), _value("alb_mean", "alb"),
    _value("fluid_balance_cumulative_mean", "fluid_balance_cumulative"),
    _status("vent_ind_max", "vent_ind"),
    _status("adv_resp_max", "adv_resp"), _status("sep3_sofa1_max", "sep3_sofa1"),
)


def test_each_concept_with_a_summary_before_time_zero_is_adjusted_for() -> None:
    names = ("crea", "bili", "alb", "fluid_balance_cumulative", "vent_ind", "adv_resp")

    trial = _compiled(*names, extra=_SUMMARIZED)

    read = {
        item.name: (item.disposition, item.column, item.summary)
        for item in trial.confounders
    }
    assert read == {
        "age": ("applied", "age", None),
        "crea": ("applied", "crea_mean", "mean"),
        "bili": ("applied", "bili_mean", "mean"),
        "alb": ("applied", "alb_mean", "mean"),
        "fluid_balance_cumulative": ("applied", "fluid_balance_cumulative_mean", "mean"),
        "vent_ind": ("applied", "vent_ind_max", "max"),
        "adv_resp": ("applied", "adv_resp_max", "max"),
    }
    assert trial.confounders_waiting == ()
    assert all(item.temporal_role for item in trial.confounders)
    # The record names the column the weights read, and the extraction asks for it.
    records = {item["name"]: item for item in trial.record()["confounders"]}
    assert (records["crea"]["column"], records["crea"]["summary"]) == ("crea_mean", "mean")
    assert "crea_mean" in trial.record()["materialization"]["columns"]
    (line,) = [item for item in trial.confirmations if item.kind == "confounder_set"]
    assert line.text.startswith(
        f"Adjusted for at time zero: age, crea (mean over [0, {TIME_ZERO}) h), "
    )
    assert f"vent_ind (whether it was recorded over [0, {TIME_ZERO}) h)" in line.text
    assert dict(CONFOUNDER_SUMMARIES) == {"event_status": "max", "value": "mean"}


def test_a_summary_whose_timing_the_owner_does_not_prove_is_not_adjusted_for() -> None:
    # The ICU rules give a sepsis status no role whose timing the host proves
    # (a severity score such as SOFA is the confounder that can be adjusted
    # for); the compile reads the column, and the card states why it is not.
    column = "sep3_sofa1_max"
    trial = _compiled("sep3_sofa1", extra=_SUMMARIZED)

    _, confounder = trial.confounders
    assert (confounder.disposition, confounder.reason, confounder.column) == (
        "not_applied", "tte_confounder_after_time_zero", None,
    )
    assert confounder.detail == (
        f"The host cannot prove {column!r} observed by time zero, so the weights "
        "do not read it."
    )


@pytest.mark.parametrize(
    ("extra", "reason", "named"),
    [
        pytest.param((), "tte_confounder_not_in_export", "column of 'crea'", id="no-column"),
        pytest.param(
            (_value("crea_max", "crea"),), "tte_confounder_not_in_export", "column of 'crea'",
            id="another-summary-only",
        ),
        pytest.param(
            (_value("crea_mean", "crea", "icu_admission[0,24]h"),),
            "tte_confounder_window_after_time_zero", "'crea_mean'",
            id="summarized-past-time-zero",
        ),
    ],
)
def test_a_concept_without_its_summary_before_time_zero_waits_for_an_extraction(
    extra, reason: str, named: str
) -> None:
    trial = _compiled("crea", extra=extra)

    _, crea = trial.confounders
    assert (crea.disposition, crea.reason, crea.column, crea.summary) == (
        "requires_extraction", reason, None, "mean",
    )
    assert named in crea.detail
    # An extraction is asked for the concept itself.
    assert "crea" in trial.record()["materialization"]["columns"]
    assert not trial.approvable


def test_a_confounder_stated_by_its_column_is_read_as_stated() -> None:
    trial = _compiled("crea_max", extra=(_value("crea_max", "crea"), _value("crea_mean", "crea")))

    _, crea = trial.confounders
    assert (crea.disposition, crea.column, crea.summary) == ("applied", "crea_max", None)
