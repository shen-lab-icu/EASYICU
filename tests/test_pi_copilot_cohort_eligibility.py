"""Copilot has to put cohort eligibility to the researcher, and only that.

The Research Agent is forbidden from inventing eligibility, and only explicit
structured filter fields authorize a filtered primary cohort. Nothing then
asked the researcher: on one development host 771 of 799 study contexts
carried no cohort preset at all, so conversation-configured studies reached
the Planner with no eligibility statement and were written up on every bound
input row.

These tests hold the two halves of the repair together: the options must stay
applicable and case-neutral, and the boundary that keeps the agent out of this
decision must stay intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import dataio
from easyicu.webserver.agent_pipeline_runs import primary_cohort_selection_mode
from easyicu.webserver.pi_copilot import cohort_eligibility
from easyicu.webserver.pi_copilot.projections import project_study_context


def test_the_option_roster_matches_the_fields_that_authorize_filtering() -> None:
    """A field Copilot offers that the pipeline ignores changes nothing."""

    source = Path(
        "src/easyicu/webserver/agent_pipeline_runs.py"
    ).read_text(encoding="utf-8")
    head = source.index("def _primary_cohort_selection_mode(")
    tail = source.index("primary_cohort_selection_mode = ", head)
    block = source[head:tail]
    for field in cohort_eligibility.ELIGIBILITY_FIELDS:
        assert field in block, field


def test_an_unstated_study_gets_the_question_and_a_way_to_decline() -> None:
    proposal = cohort_eligibility.eligibility_proposal({"cohort": {}})

    assert proposal["stated"] is False
    assert proposal["selection_mode"] == "all_input_rows"
    options = proposal["options"]
    assert len(options) >= 3
    declines = [option for option in options if option["declines_eligibility"]]
    assert len(declines) == 1, "declining must be exactly one first-class option"
    assert declines[0]["id"] == "no_eligibility_filter"
    # every option is bilingual and carries the patch it would apply
    for option in options:
        assert option["label"]["en"] and option["label"]["zh"]
        assert option["detail"]["en"] and option["detail"]["zh"]
        assert option["cohort"].get("preset")


def test_a_settled_study_is_not_asked_again() -> None:
    """Choosing "every bound stay" is an answer, not an unanswered question."""

    for cohort in (
        {"preset": "adult_first", "age_min": 18, "exclude_readmissions": True},
        {"preset": "all_icu"},
    ):
        proposal = cohort_eligibility.eligibility_proposal({"cohort": cohort})
        assert proposal["stated"] is True, cohort
        assert proposal["options"] == [], cohort


def test_every_option_is_applicable_and_lands_where_it_claims() -> None:
    """An option that does not survive the cohort owner is not an option."""

    for option in cohort_eligibility.ELIGIBILITY_OPTIONS:
        patch = option["cohort"]
        # the preset is one the Data Extraction cohort owner accepts
        assert dataio.normalize_export_cohort_preset(patch["preset"]) == patch["preset"]
        mode = primary_cohort_selection_mode({"cohort": patch})
        if option["declines_eligibility"]:
            assert mode == "all_input_rows", option["id"]
        else:
            assert mode == "predicate_filtered", option["id"]


def test_an_adult_option_states_its_age_floor_instead_of_trusting_the_preset() -> None:
    """The `adult_all` preset normalises `age_min` to 0 unless it is stated.

    An option that sent the preset alone would include children under a label
    that says adults, so the age floor is carried explicitly.
    """

    by_id = {option["id"]: option for option in cohort_eligibility.ELIGIBILITY_OPTIONS}
    for option_id in ("adults_first_admission", "adults_all_admissions"):
        assert by_id[option_id]["cohort"]["age_min"] == 18, option_id
    # the trap is real: the preset alone normalises the floor away
    without_floor = dataio._normalize_export_cohort({"preset": "adult_all"})  # noqa: SLF001
    assert without_floor["age_min"] == 0
    with_floor = dataio._normalize_export_cohort(by_id["adults_all_admissions"]["cohort"])  # noqa: SLF001
    assert with_floor["age_min"] == 18


def test_the_model_can_see_the_question_on_the_study_it_is_asked_about() -> None:
    projected = project_study_context({"label": "study", "cohort": {}})
    proposal = projected["cohort_eligibility"]

    assert proposal["schema_version"] == cohort_eligibility.SCHEMA_VERSION
    assert proposal["stated"] is False
    assert {option["id"] for option in proposal["options"]} == set(
        cohort_eligibility.option_ids()
    )
    settled = project_study_context(
        {"label": "study", "cohort": {"preset": "adult_first", "age_min": 18}}
    )
    assert settled["cohort_eligibility"]["stated"] is True


def test_the_entrypoint_makes_the_model_ask_before_setup_ends() -> None:
    entrypoint = Path(
        "src/easyicu/webserver/pi_copilot/node_app/src/main.mjs"
    ).read_text(encoding="utf-8")

    assert "study_context.cohort_eligibility.stated is false" in entrypoint
    assert "put its options to the user in one question" in entrypoint
    # the agent still may not decide this on the researcher's behalf
    assert "Never apply one silently" in entrypoint
    assert "never invent a criterion that is not in that list" in entrypoint
    # and the old unconditional stop no longer precedes the question. Assert
    # only the clause this module owns: the surrounding sentence names the
    # confirmed source in wording owned by the plan-handoff work, and pinning
    # that here would make this contract fail on an unrelated re-wording.
    assert (
        "and that eligibility answer are available, stop setup questioning"
        in entrypoint
    )


def test_the_update_tool_accepts_every_field_an_option_applies() -> None:
    """An offered answer the tool cannot carry is an answer that never lands."""

    entrypoint = Path(
        "src/easyicu/webserver/pi_copilot/node_app/src/main.mjs"
    ).read_text(encoding="utf-8")
    head = entrypoint.index("const studyCohort = Type.Object({")
    schema = entrypoint[head : head + 1500]
    applied = {
        key
        for option in cohort_eligibility.ELIGIBILITY_OPTIONS
        for key in option["cohort"]
    }
    for key in sorted(applied):
        assert f"{key}:" in schema, key


def test_choosing_the_all_admissions_option_counts_as_choosing_it() -> None:
    """The offered wording has to satisfy the gate the choice runs into.

    `adult_all` is confirmation-gated so the model cannot slide the whole
    universe in unasked. Once Copilot offers that cohort as a named option,
    picking it by its own label is an explicit selection; refusing it there
    would make the offered answer unusable.
    """

    from easyicu.webserver.pi_copilot.tools import (
        _message_explicitly_selects_all_stays as selects_all,
    )

    by_id = {option["id"]: option for option in cohort_eligibility.ELIGIBILITY_OPTIONS}
    all_admissions = by_id["adults_all_admissions"]["label"]
    assert selects_all(all_admissions["zh"]) is True
    assert selects_all(all_admissions["en"]) is True
    # and the gate still refuses everything that is not that choice
    first_admission = by_id["adults_first_admission"]["label"]
    assert selects_all(first_admission["zh"]) is False
    assert selects_all(first_admission["en"]) is False
    assert selects_all("请帮我看看这个研究") is False


def test_the_eligibility_owner_names_no_case_variable_or_database() -> None:
    """Prompt hygiene: these are shared, case-neutral study-design axes."""

    source = Path(
        "src/easyicu/webserver/pi_copilot/cohort_eligibility.py"
    ).read_text(encoding="utf-8").lower()
    for forbidden in (
        "lact",
        "sofa",
        "sepsis",
        "mimic",
        "eicu",
        "hirid",
        "amsterdam",
        "sicdb",
        "mortality",
        "death",
    ):
        assert forbidden not in source, forbidden


def test_the_patch_lookup_refuses_an_unknown_option() -> None:
    assert cohort_eligibility.cohort_patch_for_option("adults_first_admission")
    for unknown in ("", None, "not_an_option", "ADULTS_FIRST_ADMISSION"):
        assert cohort_eligibility.cohort_patch_for_option(unknown) == {}


@pytest.mark.parametrize("option_id", cohort_eligibility.option_ids())
def test_each_option_patch_is_returned_as_a_copy(option_id: str) -> None:
    patch = cohort_eligibility.cohort_patch_for_option(option_id)
    patch["preset"] = "tampered"
    assert cohort_eligibility.cohort_patch_for_option(option_id)["preset"] != "tampered"


def test_the_proposal_survives_a_study_shaped_like_anything() -> None:
    for study in (None, {}, {"cohort": None}, {"cohort": "not-a-mapping"}, []):
        proposal = cohort_eligibility.eligibility_proposal(study)
        assert proposal["schema_version"] == cohort_eligibility.SCHEMA_VERSION
        assert json.dumps(proposal)
