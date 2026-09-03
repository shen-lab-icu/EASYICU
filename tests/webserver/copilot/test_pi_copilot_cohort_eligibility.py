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

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.webserver import dataio
from easyicu.webserver import study_contexts
from easyicu.webserver.agent_pipeline_runs import primary_cohort_selection_mode
from easyicu.webserver.pi_copilot import cohort_eligibility
from easyicu.webserver.pi_copilot.projections import project_study_context


def _selection_event(
    option_id: str,
    *,
    study_id: str,
    expected_revision: int,
    cohort: dict,
    session_id: str = "pi-host-selection",
):
    scope = study_contexts.normalize_primary_cohort_scope({"cohort": cohort})
    seed = f"{session_id}:{study_id}:{expected_revision}:{option_id}:{scope.sha256}"
    event_id = hashlib.sha256(f"event:{seed}".encode()).hexdigest()
    return cohort_eligibility.build_selection_event(
        option_id=option_id,
        study_context_id=study_id,
        expected_revision=expected_revision,
        session_id=session_id,
        user_turn_id=f"turn-{session_id}",
        event_id=event_id,
        one_use_grant_id=hashlib.sha256(f"grant:{seed}".encode()).hexdigest(),
        primary_cohort_contract_sha256=scope.sha256,
        actor_id_sha256=hashlib.sha256(f"actor:{session_id}".encode()).hexdigest(),
        selected_at="2026-08-29T12:00:00Z",
    )


def _confirmed_study(option_id: str, *, revision: int = 3) -> dict:
    current_cohort: dict = {}
    patch = cohort_eligibility.selection_cohort_for_option(
        {"cohort": current_cohort}, option_id
    )
    event = _selection_event(
        option_id,
        study_id="study-confirmed",
        expected_revision=revision - 1,
        cohort=patch,
    )
    return {
        "id": "study-confirmed",
        "revision": revision,
        "cohort": patch,
        "cohort_eligibility_authority": (
            cohort_eligibility.confirmation_authority_for_option(
                option_id,
                study_context_id="study-confirmed",
                study_context_revision=revision,
                current_cohort=current_cohort,
                selection_event=event,
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
        assert option["cohort"].get("preset") == "all_icu"
        assert len(option["primary_cohort_contract_sha256"]) == 64


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
        ("first_admission_only", "first_icu_admission_only"),
        ("adults_first_admission", "first_icu_admission_only"),
        ("adults_all_admissions", "all_icu_admissions"),
        ("adults_first_admission_min_stay", "first_icu_admission_only"),
        ("no_eligibility_filter", "all_icu_admissions"),
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
        patch = cohort_eligibility.apply_option_to_cohort({}, option["id"])
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


def test_first_admission_only_does_not_invent_an_adult_age_floor() -> None:
    patch = cohort_eligibility.selection_cohort_for_option(
        {"cohort": {}}, "first_admission_only"
    )
    scope = study_contexts.normalize_primary_cohort_scope({"cohort": patch})

    assert scope.admission_eligibility["minimum_age_years"] == 0
    assert (
        scope.admission_eligibility["repeated_admission_policy"]
        == "first_icu_admission_only"
    )


def test_plan_first_projection_does_not_offer_cohort_presets() -> None:
    projected = project_study_context({"label": "study", "cohort": {}})
    assert "cohort_eligibility" not in projected


def test_the_entrypoint_leaves_eligibility_for_plan_review() -> None:
    entrypoint = Path(
        "src/easyicu/webserver/pi_copilot/node_app/src/main.mjs"
    ).read_text(encoding="utf-8")

    assert "Do not ask the user to choose a cohort eligibility preset" in entrypoint
    assert "The candidate plan must explain its rationale and evidence" in entrypoint
    assert "only that later review may authorize the exact cohort contract" in entrypoint


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

    from easyicu.webserver.pi_copilot.study_context_update import (
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
    patch["tampered"] = True
    assert "tampered" not in cohort_eligibility.cohort_patch_for_option(option_id)


def test_the_proposal_survives_a_study_shaped_like_anything() -> None:
    for study in (None, {}, {"cohort": None}, {"cohort": "not-a-mapping"}, []):
        proposal = cohort_eligibility.eligibility_proposal(study)
        assert proposal["schema_version"] == cohort_eligibility.SCHEMA_VERSION
        assert json.dumps(proposal)


def test_selection_mode_failure_is_not_downgraded_to_all_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_closed(_cohort):
        raise RuntimeError("selection owner unavailable")

    monkeypatch.setattr(
        cohort_eligibility.primary_cohort,
        "normalize_primary_cohort_scope",
        fail_closed,
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


def test_diagnosis_aliases_compile_to_one_execution_contract() -> None:
    named = study_contexts.normalize_primary_cohort_scope(
        {
            "cohort": {
                "preset": "all_icu",
                "include_diagnoses": ["I50.0"],
                "exclude_diagnoses": ["Z99"],
            }
        }
    )
    extraction = study_contexts.normalize_primary_cohort_scope(
        {
            "cohort": {
                "preset": "all_icu",
                "icd_enabled": True,
                "icd_include": "I500",
                "icd_exclude": "Z99",
            }
        }
    )
    assert named.to_dict() == extraction.to_dict()
    assert named.sha256 == extraction.sha256
    assert dataio._normalize_export_cohort(  # noqa: SLF001
        {
            "preset": "all_icu",
            "include_diagnoses": ["I50.0"],
            "exclude_diagnoses": ["Z99"],
        }
    )["icd_enabled"] is True


def test_normalized_scope_is_immutable_and_metadata_is_not_a_filter() -> None:
    scope = study_contexts.normalize_primary_cohort_scope(
        {
            "cohort": {
                "label": "Descriptive label only",
                "review": "No structured filter was selected.",
            }
        }
    )

    assert scope.selection_mode == "all_input_rows"
    assert scope.population == {
        "kind": "all_icu",
        "definition": "all_bound_icu_stays",
    }
    with pytest.raises(TypeError):
        scope.population["kind"] = "tampered"  # type: ignore[index]
    normalized = dataio._normalize_export_cohort(  # noqa: SLF001
        {"label": "Descriptive label only"}
    )
    assert normalized["preset"] == "all_icu"
    assert normalized["age_min"] == 0
    assert normalized["exclude_readmissions"] is False


def test_icd_queue_change_invalidates_a_no_filter_receipt() -> None:
    study = _confirmed_study("no_eligibility_filter")
    study["cohort"] = {
        **study["cohort"],
        "icd_enabled": True,
        "icd_include": ["I50"],
    }
    assert cohort_eligibility.validated_authority(study) is None


def test_concept_population_window_is_in_primary_cohort_scope() -> None:
    at_24h = study_contexts.normalize_primary_cohort_scope(
        {"cohort": {"preset": "sepsis3", "observation_window_hours": 24}}
    )
    at_48h = study_contexts.normalize_primary_cohort_scope(
        {"cohort": {"preset": "sepsis3", "observation_window_hours": 48}}
    )
    assert at_24h.sha256 != at_48h.sha256


def test_concept_population_window_change_invalidates_receipt() -> None:
    current = {"preset": "sepsis3", "observation_window_hours": 24}
    target = cohort_eligibility.selection_cohort_for_option(
        {"cohort": current}, "adults_first_admission"
    )
    event = _selection_event(
        "adults_first_admission",
        study_id="study-concept-window",
        expected_revision=1,
        cohort=target,
    )
    study = {
        "id": "study-concept-window",
        "revision": 2,
        "cohort": target,
        "cohort_eligibility_authority": (
            cohort_eligibility.confirmation_authority_for_option(
                "adults_first_admission",
                study_context_id="study-concept-window",
                study_context_revision=2,
                current_cohort=current,
                selection_event=event,
            )
        ),
    }
    assert cohort_eligibility.validated_authority(study) is not None
    study["cohort"] = {**target, "observation_window_hours": 48}
    assert cohort_eligibility.validated_authority(study) is None


def test_admission_option_preserves_the_current_population_definition() -> None:
    combined = cohort_eligibility.apply_option_to_cohort(
        {"preset": "sepsis3", "observation_window_hours": 24},
        "adults_first_admission",
    )
    assert combined["preset"] == "sepsis3"
    assert combined["observation_window_hours"] == 24
    assert combined["age_min"] == 18
    assert combined["exclude_readmissions"] is True


def test_custom_cohort_gets_an_exact_confirmation_option() -> None:
    study = {
        "id": "study-custom",
        "revision": 4,
        "cohort": {
            "preset": "aki",
            "age_min": 21,
            "min_icu_los_hours": 12,
            "max_patients": 500,
        },
    }
    options = cohort_eligibility.selection_options_for_study(study)
    custom = next(
        option
        for option in options
        if option["id"] == cohort_eligibility.CUSTOM_OPTION_ID
    )
    assert custom["cohort"] == study["cohort"]
    assert custom["primary_cohort_contract"]["sampling"] == {
        "status": "capped",
        "max_patients": 500,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("study_context_id", "another-study"),
        ("study_context_id", ""),
        ("origin", "planner_inference"),
        ("confirmed_by", "system_default"),
        ("confirmed_actor_id_sha256", "0" * 63),
        ("user_turn_id", ""),
        ("stated_fields", ["forged"]),
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
        ValueError, match="cohort_eligibility_selection_event_required"
    ):
        cohort_eligibility.confirmation_authority_for_option(
            "no_eligibility_filter",
            study_context_id="study-invalid",
            study_context_revision=2,
            selection_event=None,
        )
    target = cohort_eligibility.selection_cohort_for_option(
        {"cohort": {}}, "no_eligibility_filter"
    )
    event = _selection_event(
        "no_eligibility_filter",
        study_id="study-invalid",
        expected_revision=1,
        cohort=target,
    )
    with pytest.raises(
        ValueError, match="cohort_eligibility_authority_study_required"
    ):
        cohort_eligibility.confirmation_authority_for_option(
            "no_eligibility_filter",
            study_context_id="",
            study_context_revision=2,
            current_cohort={},
            selection_event=event,
        )
    with pytest.raises(
        ValueError, match="cohort_eligibility_authority_revision_invalid"
    ):
        cohort_eligibility.confirmation_authority_for_option(
            "no_eligibility_filter",
            study_context_id="study-invalid",
            study_context_revision=1,
            current_cohort={},
            selection_event=event,
        )
    with pytest.raises(
        ValueError, match="cohort_eligibility_selection_timestamp_invalid"
    ):
        cohort_eligibility.build_selection_event(
            option_id="no_eligibility_filter",
            study_context_id="study-invalid",
            expected_revision=1,
            session_id="pi-invalid",
            user_turn_id="turn-pi-invalid",
            event_id="a" * 64,
            one_use_grant_id="b" * 64,
            primary_cohort_contract_sha256="c" * 64,
            actor_id_sha256="d" * 64,
            selected_at="yesterday",
        )


def test_unconfirmed_eligibility_remains_a_plan_review_requirement() -> None:
    from easyicu.webserver.pi_copilot import workflow

    legacy = {
        "question": "Describe an ICU cohort.",
        "data_source": {"database": "miiv"},
        "cohort": {"preset": "all_icu"},
    }
    snapshot = workflow.build_research_workflow_snapshot(
        study=legacy,
        active_export_present=False,
        active_job=None,
        latest_run=None,
    )
    missing = snapshot.missing_setup_fields
    assert "cohort_eligibility" in missing
    assert "cohort_eligibility" not in snapshot.planning_prerequisites_missing

    confirmed = {**legacy, **_confirmed_study("no_eligibility_filter")}
    confirmed_snapshot = workflow.build_research_workflow_snapshot(
        study=confirmed,
        active_export_present=False,
        active_job=None,
        latest_run=None,
    )
    confirmed_missing = confirmed_snapshot.missing_setup_fields
    assert "cohort_eligibility" not in confirmed_missing
    assert "cohort_eligibility" not in confirmed_snapshot.planning_prerequisites_missing


def test_study_context_accepts_only_server_bound_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        study_contexts, "_CONFIG_PATH", tmp_path / "study-contexts.json"
    )
    current_cohort = {"preset": "all_icu"}
    patch = cohort_eligibility.apply_option_to_cohort(
        current_cohort, "adults_first_admission"
    )
    initial = study_contexts.upsert_context(
        {"id": "study-authority", "cohort": current_cohort}
    )
    event = _selection_event(
        "adults_first_admission",
        study_id=initial["id"],
        expected_revision=initial["revision"],
        cohort=patch,
    )
    authority = cohort_eligibility.confirmation_authority_for_option(
        "adults_first_admission",
        study_context_id=initial["id"],
        study_context_revision=initial["revision"] + 1,
        current_cohort=current_cohort,
        selection_event=event,
        confirmed_at="2026-08-29T12:00:00Z",
    )
    incomplete = dict(authority)
    incomplete.pop("stated_fields")
    with pytest.raises(study_contexts.StudyContextError) as missing_fields:
        study_contexts.upsert_context(
            {
                "id": initial["id"],
                "cohort": patch,
                "cohort_eligibility_authority": incomplete,
            },
            expected_revision=initial["revision"],
            require_revision=True,
            _server_cohort_eligibility_authority_write=True,
        )
    assert (
        missing_fields.value.detail["error"]
        == "cohort_eligibility_authority_incomplete"
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


def test_host_selection_event_confirms_exact_custom_cohort_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord
    from easyicu.webserver.pi_copilot.service import PiCopilotError, PiCopilotService

    monkeypatch.setattr(
        study_contexts, "_CONFIG_PATH", tmp_path / "study-contexts.json"
    )
    initial = study_contexts.upsert_context(
        {
            "id": "study-host-custom",
            "cohort": {
                "preset": "aki",
                "age_min": 21,
                "min_icu_los_hours": 12,
                "max_patients": 500,
            },
        }
    )
    service = PiCopilotService(store_path=tmp_path / "sessions.json")
    service.project_store.bind("project-host-custom", initial["id"])
    record = PiSessionRecord(
        session_id="pi-host-custom",
        project_id="project-host-custom",
        binding=service._binding_for_context(initial),  # noqa: SLF001
    )
    service._save_record(record)  # noqa: SLF001

    selection = service._cohort_eligibility_selection_projection(record)  # noqa: SLF001
    option = next(
        row
        for row in selection["options"]
        if row["id"] == cohort_eligibility.CUSTOM_OPTION_ID
    )
    with pytest.raises(PiCopilotError) as forged:
        service.confirm_cohort_eligibility(
            record.session_id,
            project_id="project-host-custom",
            option_id=option["id"],
            expected_revision=option["expected_revision"],
            primary_cohort_contract_sha256=option[
                "primary_cohort_contract_sha256"
            ],
            selection_event_id="0" * 64,
        )
    assert forged.value.code == "cohort_eligibility_selection_event_invalid"

    result = service.confirm_cohort_eligibility(
        record.session_id,
        project_id="project-host-custom",
        option_id=option["id"],
        expected_revision=option["expected_revision"],
        primary_cohort_contract_sha256=option["primary_cohort_contract_sha256"],
        selection_event_id=option["selection_event_id"],
    )
    assert result["code"] == "cohort_eligibility_confirmed"
    updated = study_contexts.get_context(initial["id"])
    assert updated is not None
    assert updated["cohort"] == initial["cohort"]
    authority = cohort_eligibility.validate_authority_payload(updated)
    assert authority["option_id"] == cohort_eligibility.CUSTOM_OPTION_ID
    assert authority["origin"] == "host_selection_event"
    assert authority["stated_fields"]

    with pytest.raises(PiCopilotError) as replayed:
        service.confirm_cohort_eligibility(
            record.session_id,
            project_id="project-host-custom",
            option_id=option["id"],
            expected_revision=option["expected_revision"],
            primary_cohort_contract_sha256=option[
                "primary_cohort_contract_sha256"
            ],
            selection_event_id=option["selection_event_id"],
        )
    assert replayed.value.code == "cohort_eligibility_selection_revision_conflict"


@pytest.mark.parametrize(
    "message",
    (
        "选择成人，并且仅纳入首次 ICU 入住。",
        "不要选择成人首次 ICU 入住。",
        "请解释成人首次 ICU 入住，但不要应用。",
        "我还没决定首次还是全部入住。",
        "‘成人首次入住’是指什么？",
    ),
)
def test_conversation_text_cannot_mint_eligibility_authority(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
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
            user_message=message,
            allowed_actions={"configure"},
        ),
    )

    assert result["code"] == "study_context_updated"
    written, kwargs = writes[0]
    assert "cohort_eligibility_authority" not in written
    assert "_server_cohort_eligibility_authority_write" not in kwargs


def test_cohort_proposal_without_bound_context_returns_typed_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.pi_copilot import tools as tool_module
    from easyicu.webserver.pi_copilot.contracts import (
        PiSessionRecord,
        ToolExecutionContext,
    )

    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: None)
    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        {"cohort": {"preset": "all_icu"}},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-no-study-context"),
            user_message="Propose the all-ICU cohort.",
            allowed_actions={"configure"},
        ),
    )

    assert result["status"] == "blocked"
    assert result["code"] == "cohort_eligibility_study_context_required"
    assert result["owner"] == "easyicu.webserver.study_contexts"
