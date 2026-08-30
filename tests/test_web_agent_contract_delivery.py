"""What Copilot collects has to reach the agent as the field it actually is.

``pipeline.run`` accepts 33 parameters and the Web caller filled 20. The gaps
were not all harmless: a value the researcher confirmed was dropped, and a
value they stated was delivered into the wrong slot, where it read as an
instruction to run analyses nobody asked for.

These tests lock the four channels audited on 2026-08-29, each of them the same
shape as the exclusion defect before it -- a contract that exists end-to-end
with one consumer that does not use it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from easyicu.research_agent.icu_rules import default_time_windows
from easyicu.research_agent.research_context.temporal_semantics import (
    TemporalAlignmentEngine,
)
from easyicu.research_agent.schema import TimeWindow
from easyicu.webserver import study_contexts
from easyicu.webserver.primary_cohort import MAX_DIAGNOSIS_TOKENS
from easyicu.webserver.agent_pipeline_runs import (
    _cohort_window,
    _declared_time_windows,
    _exclusion_criteria,
    _inclusion_criteria,
    _research_user_preferences,
)

_RUNS = Path("src/easyicu/webserver/agent_pipeline_runs.py")


# --------------------------------------------------------------------------
# A. the confirmed feature window must be declared, not re-inferred
# --------------------------------------------------------------------------


def test_the_materialized_window_is_declared_as_the_analysis_window() -> None:
    study = {
        "time_window": {
            "hours": 24,
            "anchor": "ICU admission",
            "label": "opening ICU feature window",
        }
    }

    windows = _declared_time_windows(_cohort_window(study), study)

    assert len(windows) == 1
    window = windows[0]
    assert isinstance(window, TimeWindow), "the engine takes typed TimeWindow"
    assert window.name == "opening ICU feature window"
    assert window.anchor == "icu_admission"
    assert (window.start_hours, window.end_hours) == (0.0, 24.0)
    assert window.rationale, "a declared window states where it came from"


def test_the_declared_window_is_the_one_materialization_used() -> None:
    """Recomputing it from the study would let the two drift apart."""

    source = _RUNS.read_text(encoding="utf-8")
    assert "time_windows=_declared_time_windows(window, study)," in source
    # `window` is the tuple already passed as `cohort_window=` for materialization
    assert "cohort_window=window," in source
    assert "window = _cohort_window(materialization_study)" in source


def test_a_study_without_a_label_still_names_its_window() -> None:
    window = _declared_time_windows((0.0, 48.0), {})[0]
    assert window.name == "icu_admission_0_48h"
    assert (window.start_hours, window.end_hours) == (0.0, 48.0)


def test_sending_no_window_would_have_offered_a_whole_stay_window() -> None:
    """The defect this fix prevents, stated as a fact about the fallback.

    With no explicit window the context builder falls back to
    ``inferred_windows or default_time_windows()``. That roster ends in a
    720-hour whole-stay window, so a cohort materialized to its opening hours
    was offering the Planner a window the data cannot support.
    """

    names = {window.name: window for window in default_time_windows()}
    assert "full_stay" in names
    assert names["full_stay"].end_hours == 720.0

    # and an explicit window is honoured instead of the inferred roster
    declared = _declared_time_windows((0.0, 24.0), {})
    windows, _constraints = TemporalAlignmentEngine().infer(
        research_question="does the exposure predict the outcome",
        explicit_windows=declared,
    )
    assert [w.name for w in windows] == [declared[0].name]


# --------------------------------------------------------------------------
# B. a diagnosis criterion may only be declared when it is actually applied
# --------------------------------------------------------------------------


def test_a_diagnosis_filter_that_cannot_run_is_never_declared() -> None:
    """The declaration follows the primary-cohort owner, not the conversation.

    The cohort ledger draws its exclusion stage from this declaration, so a
    filter the export would refuse must not arrive as a real criterion.
    """

    refused = {"cohort": {"preset": "icd"}}  # an ICD cohort carrying no tokens
    assert _inclusion_criteria(refused) == []
    assert _exclusion_criteria(refused) == []

    executable = {"cohort": {"include_diagnoses": ["a"], "exclude_diagnoses": ["b"]}}
    assert "include diagnoses: a" in _inclusion_criteria(executable)
    assert "exclude diagnoses: b" in _exclusion_criteria(executable)


def test_the_icd_fields_reach_their_own_side() -> None:
    """``icd_include``/``icd_exclude`` are what the export filters on."""

    study = {
        "cohort": {"preset": "icd", "icd_include": "A41", "icd_exclude": "T20-T32"}
    }
    assert "include diagnoses: A41" in _inclusion_criteria(study)
    assert "exclude diagnoses: T20-T32" in _exclusion_criteria(study)
    # the researcher's stated range is kept rather than the expanded roster
    assert "T21" not in " | ".join(_exclusion_criteria(study))


def test_every_stated_diagnosis_criterion_reaches_the_declaration() -> None:
    """The write-up must not state a narrower cohort than the one that ran.

    A private cap of 20 declared 20 of 35 stated criteria while 39 codes
    actually executed. The cohort ledger and the manuscript's inclusion
    criteria both read this declaration.
    """

    stated = "I21-I25, J44, J45, N17, A41, " + ", ".join(
        f"E{value:02d}" for value in range(10, 40)
    )
    study = {"cohort": {"preset": "icd", "icd_include": stated}}

    declared = _inclusion_criteria(study)[-1]
    items = declared.split(": ", 1)[1].split(", ")

    assert len(items) == 35
    assert items[-1] == "E39"
    assert "not declared" not in declared


def test_a_trimmed_declaration_says_that_it_was_trimmed() -> None:
    """Silence is the failure mode; an incomplete roster has to admit it.

    A roster of more than ``MAX_DIAGNOSIS_TOKENS`` distinct codes is refused
    upstream, so the only way to state more criteria than are declared is to
    repeat one -- which still leaves the researcher reading a shorter list than
    they wrote.
    """

    stated = [f"A{value:02d}" for value in range(10, 10 + MAX_DIAGNOSIS_TOKENS)]
    stated += ["A10"] * 5
    study = {
        "cohort": {"preset": "icd", "icd_enabled": True, "include_diagnoses": stated}
    }

    declared = _inclusion_criteria(study)[-1]

    assert "(+5 further stated criteria not declared)" in declared


def test_a_stated_removal_has_a_slot_that_is_not_the_inclusion_channel() -> None:
    """``review`` is the inclusion contract; removals need their own field."""

    study = {
        "cohort": {
            "review": "adult ICU stays",
            "exclusion_statement": "stays ending before the landmark",
        }
    }
    assert _inclusion_criteria(study) == ["adult ICU stays"]
    assert _exclusion_criteria(study) == ["stays ending before the landmark"]

    # the slot is persistable, and unknown cohort fields still fail closed
    accepted = study_contexts._sanitize_patch(  # noqa: SLF001
        {"cohort": {"exclusion_statement": "removed for x"}}
    )
    assert accepted["cohort"]["exclusion_statement"] == "removed for x"

    entrypoint = Path(
        "src/easyicu/webserver/pi_copilot/node_app/src/main.mjs"
    ).read_text(encoding="utf-8")
    assert "exclusion_statement: optionalText(" in entrypoint
    assert "belongs in cohort.exclusion_statement" in entrypoint

    # and the model can read back what it recorded, or it cannot tell an
    # unstated exclusion from one already written
    from easyicu.webserver.pi_copilot.projections import project_study_context

    projected = project_study_context(
        {"label": "s", "cohort": {"exclusion_statement": "removed for x"}}
    )
    assert projected["cohort"]["exclusion_statement"] == "removed for x"


# --------------------------------------------------------------------------
# C. a comparator is an estimand contrast, not a subgroup request
# --------------------------------------------------------------------------


def test_a_comparator_is_not_delivered_as_a_subgroup_request() -> None:
    preferences = _research_user_preferences(
        {"purpose": "quantify the association", "comparator": "unexposed stays"}
    )

    assert preferences.get("subgroup_sensitivity") is None
    notes = preferences["extra_notes"]
    assert "quantify the association" in notes, "the purpose is not overwritten"
    assert "Comparator stated by the researcher: unexposed stays" in notes


def test_the_slot_it_used_to_land_in_is_an_instruction() -> None:
    """Why the previous mapping was wrong, locked against the consumer."""

    from easyicu.research_agent import skills

    source = inspect.getsource(skills)
    assert "Include subgroup/sensitivity requests: {prefs.subgroup_sensitivity}" in source


def test_a_study_with_no_comparator_adds_nothing() -> None:
    preferences = _research_user_preferences({"purpose": "quantify the association"})
    assert preferences["extra_notes"] == "quantify the association"
    assert _research_user_preferences({}).get("extra_notes") is None


# --------------------------------------------------------------------------
# D. cross-database validation must stay a fact, never a conversational claim
# --------------------------------------------------------------------------


def test_cross_database_validation_stays_unclaimable_from_the_conversation() -> None:
    """``cross_database_validation`` is not inert, so it may not be asserted.

    Readiness reporting and novelty positioning put this list straight into the
    manuscript's own claims, so declaring it announces external validation. The
    study context has no such field and the conversation cannot write one.
    """

    assert "cross_database_validation" not in study_contexts._CONTEXT_FIELDS  # noqa: SLF001
    entrypoint = Path(
        "src/easyicu/webserver/pi_copilot/node_app/src/main.mjs"
    ).read_text(encoding="utf-8")
    assert "cross_database_validation" not in entrypoint
    assert "cross_database_validation=" not in _RUNS.read_text(encoding="utf-8")


def test_the_crossdb_selection_receipt_is_not_promoted_into_that_claim() -> None:
    """The one nearby receipt is deliberately weaker than the claim.

    ``crossdb_selection`` is a real digest-bound selection of two or more bound
    exports, so it looks like the missing input -- but its owner compares those
    exports on cohort-level aggregates only and holds formal cross-database
    claims fail-closed until the numeric evidence audit gate. Promoting the
    receipt into ``cross_database_validation`` would publish precisely the claim
    that owner withholds, so the two stay unconnected until a governed
    multi-export run can earn it.
    """

    assert "crossdb_selection" in study_contexts._CONTEXT_FIELDS  # noqa: SLF001
    owner = " ".join(
        Path("src/easyicu/webserver/crossdb_review.py")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "cohort-level aggregates only" in owner
    assert "formal cross-database claims remain fail-closed" in owner
    # and nothing turns the selection into the agent-facing declaration
    runs = _RUNS.read_text(encoding="utf-8")
    assert "crossdb_selection" not in runs
