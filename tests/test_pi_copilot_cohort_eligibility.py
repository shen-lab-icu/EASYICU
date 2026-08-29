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
from easyicu.webserver import study_contexts
from easyicu.webserver.agent_pipeline_runs import primary_cohort_selection_mode
from easyicu.webserver.pi_copilot import cohort_eligibility
from easyicu.webserver.pi_copilot.projections import project_study_context


def _confirmed_study(option_id: str, *, revision: int = 3) -> dict:
    patch = cohort_eligibility.cohort_patch_for_option(option_id)
    return {
        "id": "study-confirmed",
        "revision": revision,
        "cohort": patch,
        "cohort_eligibility_authority": (
            cohort_eligibility.confirmation_authority_for_option(
                option_id,
                study_context_id="study-confirmed",
                study_context_revision=revision,
                confirmed_at="2026-08-29T12:00:00Z",
            )
        ),
    }


def test_the_option_roster_has_one_study_context_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copilot and the pipeline consume one field/mode contract, not mirrors."""

    assert (
        cohort_eligibility.ELIGIBILITY_FIELDS
        is study_contexts.COHORT_ELIGIBILITY_FIELDS
    )
    monkeypatch.setattr(
        study_contexts,
        "primary_cohort_selection_mode",
        lambda _study: "sentinel_mode",
    )
    assert primary_cohort_selection_mode({"cohort": {}}) == "sentinel_mode"


def test_an_unstated_study_gets_the_question_and_a_way_to_decline() -> None:
    proposal = cohort_eligibility.eligibility_proposal({"cohort": {}})

    assert proposal["stated"] is False
    assert proposal["selection_state"] == "unresolved"
    assert proposal["blocker_code"] == "cohort_eligibility_confirmation_required"
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


def test_legacy_values_are_not_upgraded_to_researcher_confirmation() -> None:
    """A preset or migrated filter is a value, not a confirmation receipt."""

    for cohort in (
        {"preset": "adult_first", "age_min": 18, "exclude_readmissions": True},
        {"preset": "all_icu"},
    ):
        proposal = cohort_eligibility.eligibility_proposal({"cohort": cohort})
        assert proposal["stated"] is False, cohort
        assert proposal["selection_state"] == "legacy_unconfirmed", cohort
        assert proposal["options"], cohort


@pytest.mark.parametrize("option_id", cohort_eligibility.option_ids())
def test_a_receipted_decision_is_not_asked_again(option_id: str) -> None:
    proposal = cohort_eligibility.eligibility_proposal(_confirmed_study(option_id))
    assert proposal["stated"] is True
    assert proposal["authority_status"] == "valid"
    assert proposal["options"] == []


@pytest.mark.parametrize(
    ("option_id", "repeated_admission_policy"),
    (
        ("adults_first_admission", "first_icu_admission_only"),
        ("adults_all_admissions", "all_icu_admissions"),
        ("adults_first_admission_min_stay", "first_icu_admission_only"),
        ("no_eligibility_filter", "all_bound_icu_stays"),
    ),
)
def test_receipt_states_the_repeated_admission_policy(
    option_id: str, repeated_admission_policy: str
) -> None:
    authority = _confirmed_study(option_id)["cohort_eligibility_authority"]
    assert authority["repeated_admission_policy"] == repeated_admission_policy


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
    settled = project_study_context(_confirmed_study("adults_first_admission"))
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
        with pytest.raises(cohort_eligibility.CohortEligibilityOptionError) as raised:
            cohort_eligibility.cohort_patch_for_option(unknown)
        assert raised.value.code == "cohort_eligibility_option_unknown"
        assert "adults_first_admission" in raised.value.allowed


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


def test_selection_mode_failure_is_not_downgraded_to_all_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_closed(_study) -> str:
        raise RuntimeError("selection owner unavailable")

    monkeypatch.setattr(
        study_contexts, "primary_cohort_selection_mode", fail_closed
    )
    with pytest.raises(RuntimeError, match="selection owner unavailable"):
        cohort_eligibility.eligibility_proposal({"cohort": {}})


def test_changed_cohort_makes_an_old_receipt_stale() -> None:
    study = _confirmed_study("adults_first_admission")
    study["cohort"] = {**study["cohort"], "min_icu_los_hours": 24}
    proposal = cohort_eligibility.eligibility_proposal(study)
    assert proposal["stated"] is False
    assert proposal["selection_state"] == "legacy_unconfirmed"
    assert proposal["authority_status"] == "stale"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("study_context_id", "another-study"),
        ("origin", "planner_inference"),
        ("confirmed_by", "system_default"),
        ("confirmed_at", "not-a-timestamp"),
        ("repeated_admission_policy", "unspecified"),
    ),
)
def test_tampered_confirmation_receipt_fails_closed(field: str, value: object) -> None:
    study = _confirmed_study("no_eligibility_filter")
    study["cohort_eligibility_authority"] = {
        **study["cohort_eligibility_authority"],
        field: value,
    }
    assert cohort_eligibility.validated_authority(study) is None
    assert cohort_eligibility.eligibility_stated(study) is False


def test_receipt_builder_rejects_unbound_or_invalid_provenance() -> None:
    with pytest.raises(
        ValueError, match="cohort_eligibility_authority_study_required"
    ):
        cohort_eligibility.confirmation_authority_for_option(
            "no_eligibility_filter",
            study_context_id="",
            study_context_revision=1,
        )
    with pytest.raises(
        ValueError, match="cohort_eligibility_authority_revision_invalid"
    ):
        cohort_eligibility.confirmation_authority_for_option(
            "no_eligibility_filter",
            study_context_id="study-invalid",
            study_context_revision=0,
        )
    with pytest.raises(
        ValueError, match="cohort_eligibility_authority_timestamp_invalid"
    ):
        cohort_eligibility.confirmation_authority_for_option(
            "no_eligibility_filter",
            study_context_id="study-invalid",
            study_context_revision=1,
            confirmed_at="yesterday",
        )


def test_unconfirmed_eligibility_blocks_candidate_planning() -> None:
    from easyicu.webserver.pi_copilot import workflow

    legacy = {
        "question": "Describe an ICU cohort.",
        "data_source": {"database": "miiv"},
        "cohort": {"preset": "all_icu"},
    }
    missing = workflow._setup_missing(legacy, active_export_present=False)
    assert "cohort_eligibility" in missing
    assert "cohort_eligibility" in workflow._planning_prerequisites_missing(missing)

    confirmed = {**legacy, **_confirmed_study("no_eligibility_filter")}
    confirmed_missing = workflow._setup_missing(
        confirmed, active_export_present=False
    )
    assert "cohort_eligibility" not in confirmed_missing
    assert "cohort_eligibility" not in workflow._planning_prerequisites_missing(
        confirmed_missing
    )


def test_study_context_accepts_only_server_bound_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        study_contexts, "_CONFIG_PATH", tmp_path / "study-contexts.json"
    )
    patch = cohort_eligibility.cohort_patch_for_option("adults_first_admission")
    initial = study_contexts.upsert_context(
        {"id": "study-authority", "cohort": patch}
    )
    authority = cohort_eligibility.confirmation_authority_for_option(
        "adults_first_admission",
        study_context_id=initial["id"],
        study_context_revision=initial["revision"] + 1,
        confirmed_at="2026-08-29T12:00:00Z",
    )
    with pytest.raises(study_contexts.StudyContextError) as forged:
        study_contexts.upsert_context(
            {
                "id": initial["id"],
                "cohort_eligibility_authority": authority,
            },
            expected_revision=initial["revision"],
            require_revision=True,
        )
    assert (
        forged.value.detail["error"]
        == "study_cohort_eligibility_authority_server_owned"
    )

    confirmed = study_contexts.upsert_context(
        {
            "id": initial["id"],
            "cohort": patch,
            "cohort_eligibility_authority": authority,
        },
        expected_revision=initial["revision"],
        require_revision=True,
        _server_cohort_eligibility_authority_write=True,
    )
    assert cohort_eligibility.eligibility_stated(confirmed) is True

    changed = study_contexts.upsert_context(
        {
            "id": initial["id"],
            "cohort": {**patch, "min_icu_los_hours": 24},
        },
        expected_revision=confirmed["revision"],
        require_revision=True,
    )
    assert changed["cohort_eligibility_authority"] == {}
    assert cohort_eligibility.eligibility_stated(changed) is False


def test_copilot_selection_persists_a_server_owned_revision_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.pi_copilot import tools as tool_module
    from easyicu.webserver.pi_copilot.contracts import (
        AuthorityBinding,
        PiSessionRecord,
        ToolExecutionContext,
    )

    current = {
        "id": "study-copilot-eligibility",
        "revision": 5,
        "active_job_id": None,
        "cohort": {},
    }
    writes: list[tuple[dict, dict]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: current)
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **kwargs: (
            writes.append((dict(raw), dict(kwargs)))
            or {**current, **raw, "revision": 6}
        ),
    )
    monkeypatch.setattr(
        tool_module,
        "_workflow_snapshot",
        lambda _context, **_kwargs: {"current_stage": "setup"},
    )
    session = PiSessionRecord(
        session_id="pi-cohort-eligibility",
        binding=AuthorityBinding(
            study_context_id=current["id"],
            study_revision=current["revision"],
        ),
    )

    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {
            "cohort": cohort_eligibility.cohort_patch_for_option(
                "adults_first_admission"
            )
        },
        ToolExecutionContext(
            session=session,
            user_message="选择成人，并且仅纳入首次 ICU 入住。",
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    written, kwargs = writes[0]
    authority = written["cohort_eligibility_authority"]
    assert authority["option_id"] == "adults_first_admission"
    assert authority["study_context_revision"] == 6
    assert len(authority["decision_sha256"]) == 64
    assert kwargs["_server_cohort_eligibility_authority_write"] is True
