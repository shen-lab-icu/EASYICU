"""A causal question can be reviewed without supplying an internal trial schema."""

from pathlib import Path

import pytest

from easyicu.webserver import target_trial_card
from easyicu.webserver import target_trial_proposal as owner
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot


@pytest.fixture(
    params=[
        "In patients with the index condition, with time zero at hour 8 after ICU admission, "
        "start sample therapy within 10 hours versus not starting it within 10 hours "
        "for 90-day mortality, adjusted for age.",
        "示例病症患者中，以入 ICU 后第8小时为时间零点，10小时内开始示例治疗，"
        "与这10小时内不开始相比，比较90天死亡率，调整年龄。",
    ],
    ids=["en", "zh"],
)
def study(request, tmp_path: Path):
    return {
        "id": "synthetic",
        "revision": 3,
        "question": request.param,
        "data_source": {"database": "miiv", "path": str(tmp_path / "export")},
        "analysis_design": {
            "analysis_family": "causal_inference",
            "analysis_unit": "icu_stay",
            "variance_estimator": "bootstrap",
            "cluster_unit": "patient",
        },
        "stated_adjustments": [
            {
                "concept": "age",
                "variable": "age",
                "source_status": "available",
                "words": "调整年龄" if "示例" in request.param else "adjusted for age",
            }
        ],
    }


@pytest.fixture
def synthetic_treatment(monkeypatch):
    def exact_fixture_name(question, reading, database):
        if reading.treatment and question[slice(*reading.treatment)] in {"sample therapy", "示例治疗"}:
            return ("sample_event",), "sample_class"

    monkeypatch.setattr(owner, "_treatment", exact_fixture_name)


@pytest.fixture
def complete_trial(study, synthetic_treatment, monkeypatch):
    monkeypatch.setattr(owner, "_indication_concept", lambda words: "example_condition")
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "proposal"
    assert card["proposal"]["missing"] == []
    assert card["proposal"]["digest"] is not None
    return study


def test_question_has_an_inline_proposal_and_no_plan_offer_before_review(complete_trial):
    study = complete_trial
    snapshot = build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None, latest_run=None
    )
    assert snapshot.next_action_code == "target_trial_statement_needed"
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "proposal"
    assert card["approval"] is None
    draft = card["proposal"]
    assert not draft["missing"]
    assert draft["digest"]
    assert {row["key"] for row in draft["elements"]} >= {
        "treatment",
        "strategies",
        "time_zero",
        "grace_period",
        "outcome",
        "eligibility",
        "adjustment",
    }
    assert {row["source"] for row in draft["elements"]} == {"question", "design_choice"}
    statement = owner.target_trial_proposal(study).statement
    assert statement.spec.time_zero.hours_after_icu_admission == 8
    assert statement.spec.grace_period.hours == 10
    expected_defer = "这10小时内不开始" if "示例" in study["question"] else "not starting it within 10 hours"
    assert statement.spec.strategies.defer_label == expected_defer
    assert statement.spec.confounders[0].name == "age"


def test_missing_clinical_details_are_one_clarification(study):
    study["question"] = "What is the causal effect of an unspecified therapy?"
    study["stated_adjustments"] = []
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "clarification"
    assert "treatment" in card["proposal"]["missing"]
    assert card["proposal"]["digest"] is None
    assert card["approval"] is None


@pytest.mark.parametrize(
    ("removed_en", "removed_zh", "missing"),
    [
        (
            "start sample therapy within 10 hours versus not starting it within 10 hours ",
            "10小时内开始示例治疗，与这10小时内不开始相比，",
            ["treatment", "strategies", "grace_period"],
        ),
        (
            "with time zero at hour 8 after ICU admission, ",
            "以入 ICU 后第8小时为时间零点，",
            ["time_zero", "eligibility"],
        ),
    ],
)
def test_incomplete_causal_questions_show_a_clarification(
    complete_trial, removed_en, removed_zh, missing,
):
    study = complete_trial
    # Name the causal method so the reader still runs when a trial element is removed.
    zh = "示例" in study["question"]
    study["question"] += "采用目标试验模拟。" if zh else " Use target trial emulation."
    assert target_trial_card.target_trial_card(study, None)["state"] == "proposal"
    removed = removed_zh if zh else removed_en
    assert study["question"].count(removed) == 1
    study["question"] = study["question"].replace(removed, "", 1)
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "clarification"
    assert card["proposal"]["missing"] == missing
    assert card["proposal"]["digest"] is None


@pytest.mark.parametrize("extra", [False, True])
def test_trailing_method_sentence_covers_only_the_standalone_cue(complete_trial, monkeypatch, extra):
    study = complete_trial
    zh = "示例" in study["question"]
    if zh:
        study["question"] += "使用目标试验模拟并报告分层结果。" if extra else "使用目标试验模拟。"
    else:
        study["question"] += (
            " Apply target-trial emulation and report subgroup results." if extra
            else " Apply target-trial emulation."
        )
    proposal = owner.target_trial_proposal(study)
    if extra:
        assert proposal.statement is None
        assert "unread_words" in proposal.missing
        assert any(("分层结果" if zh else "subgroup results") in words for words in proposal.unread_words)
    else:
        from dataclasses import replace

        assert proposal.statement is not None
        assert proposal.unread_words == []
        reading = owner.causal_trial_reading(study["question"])
        without_method = replace(
            reading, elements=tuple(item for item in reading.elements if item.element != "causal_method"),
        )
        monkeypatch.setattr(owner, "causal_trial_reading", lambda _: without_method)
        assert owner.target_trial_proposal(study).missing == ["unread_words"]


@pytest.mark.parametrize(
    ("dropped_element", "missing"),
    [
        ("initiate_strategy", ["strategies", "grace_period"]),
        ("time_zero", ["time_zero", "eligibility"]),
    ],
)
def test_partial_reader_evidence_fails_closed_on_the_workflow_card(
    complete_trial, monkeypatch, dropped_element, missing
):
    from dataclasses import replace

    study = complete_trial
    reading = owner.causal_trial_reading(study["question"])
    partial = replace(
        reading,
        elements=tuple(item for item in reading.elements if item.element != dropped_element),
    )
    assert len(partial.elements) == len(reading.elements) - 1
    monkeypatch.setattr(owner, "causal_trial_reading", lambda _: partial)
    snapshot = build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None, latest_run=None
    )
    assert snapshot.next_action_code == "target_trial_statement_needed"
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "clarification"
    assert card["proposal"]["missing"] == [*missing, "unread_words"]
    assert card["proposal"]["digest"] is None
    assert card["approval"] is None


@pytest.mark.parametrize(
    ("population_en", "population_zh", "adult"),
    [
        ("Among patients with suspected infection", "在疑似感染患者中", False),
        ("Among adult patients with suspected infection", "在成人疑似感染患者中", True),
        ("In adults with suspected infection", "在成年疑似感染患者中", True),
        ("In non-adult patients with suspected infection", "在未成年疑似感染患者中", None),
        ("Among non-adult patients with suspected infection", "在非成人疑似感染患者中", None),
        ("In mechanically ventilated patients with suspected infection", "在机械通气疑似感染患者中", None),
        ("In patients with suspected infection, age >= 65", "在65岁以上疑似感染患者中", None),
        ("In patients with suspected infection, age 65", "在65岁疑似感染患者中", None),
        ("In patients with suspected infection, age ≥ 65", "在65岁及以上疑似感染患者中", None),
        ("In patients with suspected infection, age > 65", "在超过65岁的疑似感染患者中", None),
        ("In patients with suspected infection, at least 65 years old", "在年龄65岁的疑似感染患者中", None),
        ("Among patients with suspected infection, excluding children", "在除儿童外的疑似感染患者中", None),
        ("In patients with suspected infection, children excluded", "在不含儿童的疑似感染患者中", None),
        ("In patients with no suspected infection", "在非疑似感染患者中", None),
        ("Among mechanically ventilated adults aged at least 65 with sepsis", "在65岁以上机械通气脓毒症患者中", None),
        ("In patients with suspected infection and no prior ventilation", "在合并呼吸衰竭的疑似感染患者中", None),
        ("In patients with suspected infection at presentation", "在疑似感染的首次入住患者中", None),
    ],
)
def test_population_requires_a_bare_indication(study, synthetic_treatment, population_en, population_zh, adult):
    question = study["question"]
    zh = "示例" in question
    delimiter = "，" if zh else ","
    study["question"] = (population_zh if zh else population_en) + delimiter + question.split(delimiter, 1)[1]
    proposal = owner.target_trial_proposal(study)
    card = target_trial_card.target_trial_card(study, None)
    if adult is None:
        assert proposal.statement is None
        assert card["state"] == "clarification"
        missing = card["proposal"]["missing"]
        assert "unread_words" in missing
        assert ("eligibility" in missing) == (
            zh or not population_en.startswith(("In patients with suspected infection,", "Among patients with suspected infection,"))
        )
        assert card["proposal"]["digest"] is None
        return
    assert card["state"] == "proposal"
    assert proposal.missing == []
    criteria = proposal.statement.population_spec.criteria
    assert [(item.kind, item.role) for item in criteria] == (
        [("condition_present", "include")] + ([("age_years", "include")] if adult else [])
    )
    assert list(criteria[0].concepts_all_of) == ["susp_inf"]
    assert (criteria[0].window.start_hours, criteria[0].window.end_hours) == (0, 8)
    if adult:
        assert (criteria[1].min_years, criteria[1].max_years) == (18, None)
    assert proposal.statement.spec.confounders[0].name == "age"


@pytest.mark.parametrize(
    ("element", "en", "zh", "accepted"),
    [
        ("outcome", "the from-time-zero {term}", "自时间零点起{term}", False),
        ("outcome", "from time zero, {term}", "从开始治疗起{term}", False),
        ("outcome", "{term} from treatment start", "{term}自插管起算", False),
        ("outcome", "from intubation, {term}", "自插管起{term}", False),
        ("outcome", "{term} after discharge", "{term}从出院起算", False),
        ("outcome", "{term} counted starting at extubation", "拔管后{term}", False),
        ("outcome", "from ICU admission to extubation, {term}", "自入ICU后第2日起{term}", False),
        ("outcome", "from ICU admission, {term}", "自入ICU后起算，{term}", False),
        ("outcome", "from ICU admission {term}", "自入ICU起{term}", True),
        ("outcome", "{term} from ICU admission", "{term}自入ICU起算", True),
        ("outcome", "{term} after ICU admission", "入住ICU后{term}", True),
        ("grace_period", "from intubation, {term}", "自插管起{term}", False),
        ("grace_period", "after extubation, {term}", "拔管后{term}", False),
        ("grace_period", "{term} from intubation", "{term}自插管起算", False),
        ("grace_period", "from treatment start, {term}", "从开始治疗起{term}", False),
        ("grace_period", "{term} from treatment start", "{term}自时间零点起算", False),
        ("grace_period", "from time zero, {term}", "自时间零点起{term}", False),
        ("grace_period", "from ICU admission, {term}", "自入ICU起{term}", False),
        ("grace_period", "{term} from ICU admission", "{term}自入ICU起算", False),
    ],
)
def test_trial_origin_is_preserved_or_requested(complete_trial, element, en, zh, accepted):
    study = complete_trial
    question = study["question"]
    outcome = element == "outcome"
    if "示例" in question:
        term = "90天死亡率" if outcome else "10小时内开始示例治疗"
        wording = zh
    else:
        term = "90-day mortality" if outcome else "start sample therapy within 10 hours"
        wording = en
    study["question"] = question.replace(term, wording.format(term=term), 1)
    proposal = owner.target_trial_proposal(study)
    card = target_trial_card.target_trial_card(study, None)
    if not accepted:
        assert proposal.statement is None
        assert card["state"] == "clarification"
        assert "unread_words" in card["proposal"]["missing"]
        assert element in {row["key"] for row in card["proposal"]["elements"]}
        assert card["proposal"]["digest"] is None
        return
    assert card["state"] == "proposal"
    assert proposal.missing == []
    endpoint = owner.fixed_horizon_mortality_endpoint(proposal.statement.spec.outcome.endpoint)
    assert (endpoint.horizon_days, endpoint.time_origin) == (90, "icu_admission")


def test_rejected_treatment_is_absent_from_clarification_elements(complete_trial, monkeypatch):
    study = complete_trial
    assert owner.causal_trial_reading(study["question"]).treatment is not None
    monkeypatch.setattr(owner, "_treatment", lambda *args: None)
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "clarification"
    assert card["proposal"]["missing"] == ["treatment", "unread_words"]
    assert card["proposal"]["digest"] is None
    assert "treatment" not in {row["key"] for row in card["proposal"]["elements"]}


def test_noncausal_question_has_no_proposal(study):
    study["question"] = "Describe ICU admissions by shift."
    study.pop("analysis_design")
    assert target_trial_card.target_trial_card(study, None) is None


def test_clicked_draft_waits_for_approval_then_exposes_plan(
    study, tmp_path, monkeypatch
):
    import time

    from easyicu.webserver import (
        jobs,
        study_contexts,
        target_trial_proposal,
        target_trial_setup,
    )
    from tests.support.target_trial import kept_target_trial_record

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        study_contexts, "_CONFIG_PATH", tmp_path / "study-contexts.json"
    )
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    monkeypatch.setattr(
        target_trial_setup.dataio, "read_prepared_export_manifest", lambda _: {}
    )
    monkeypatch.setattr(
        target_trial_proposal,
        "_treatment",
        lambda *args: (("sample_event",), "sample_class"),
    )
    monkeypatch.setattr(
        target_trial_proposal, "_indication_concept", lambda _: "example_condition"
    )
    stored = study_contexts.upsert_context(
        {key: value for key, value in study.items() if key != "revision"},
        _server_question_wording_write=True,
    )
    kept = kept_target_trial_record()
    reviewed_digests = []

    def compile_fixture(job, *, study, statement, revision, **_):
        # The downstream scientific compiler has its own synthetic-export
        # integration tests. Here the worker checks the exact accepted input.
        reviewed_digests.append(statement.request()["request_sha256"])
        target_trial_card_record = kept.design()
        from easyicu.webserver.target_trial_records import keep_target_trial_record

        keep_target_trial_record(study["id"], kept)
        study_contexts.bind_target_trial_design(
            study["id"], target_trial_card_record, expected_revision=revision
        )
        return {
            "status": "compiled",
            "compile_sha256": kept.compile_sha256,
            "approvable": True,
        }

    monkeypatch.setattr(target_trial_setup, "_compile", compile_fixture)
    proposal = target_trial_card.target_trial_card(stored, None)["proposal"]
    assert study_contexts.get_context(stored["id"]).get("target_trial_design") in (
        None,
        {},
    )
    with pytest.raises(
        target_trial_setup.TargetTrialSetupError, match="current target trial proposal"
    ):
        target_trial_card.compile_target_trial_proposal(
            stored["id"],
            expected_revision=stored["revision"],
            proposal_sha256="4" * 64,
        )
    assert reviewed_digests == []

    started = target_trial_card.compile_target_trial_proposal(
        stored["id"],
        expected_revision=stored["revision"],
        proposal_sha256=proposal["digest"],
    )
    job = jobs.MANAGER.get(started["job_id"])
    deadline = time.monotonic() + 5
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status == "done"
    assert reviewed_digests == [proposal["digest"]]
    stated = study_contexts.get_context(stored["id"])
    latest = target_trial_setup.latest_target_trial_compile(stored["id"])
    card = target_trial_card.target_trial_card(stated, latest)
    assert card["state"] == "approvable"
    assert card["confirm_all_on_approval"] is True
    assert card["approval"] is None
    assert (
        target_trial_card.target_trial_next_action(stated, latest)
        == "target_trial_review"
    )

    approval = target_trial_card.approve_target_trial(
        stored["id"],
        expected_revision=stated["revision"],
        compile_sha256=card["compile_sha256"],
        n_lines_confirmed=card["confirmation_lines"],
    )
    assert approval["repeated"] is False
    assert (
        target_trial_card.target_trial_next_action(approval["study"], latest)
        == "target_trial_plan_ready"
    )


def test_proposal_post_compiles_a_prepared_export_and_only_approval_locks_the_trial(
    tmp_path, monkeypatch
):
    from hashlib import sha256
    import time

    import pandas as pd
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from easyicu.webserver import jobs, settings, study_contexts, target_trial_setup
    from easyicu.webserver.pi_copilot.service import PiCopilotService
    from easyicu.webserver.routes import pi_copilot as routes
    from tests.support.target_trial_export import target_trial_export
    from tests.webserver.copilot.pi_copilot_contract_fixtures import (
        FakeGateway,
        allow_unrelated_message_test,
    )

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True})
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())
    export = target_trial_export(tmp_path / "export", baseline_condition="susp_inf")
    manifest = export / "_manifest.json"
    mapping = tmp_path / "grouping.parquet"
    pd.read_parquet(export / "demographics.parquet", columns=["stay_id"]).assign(
        patient_key=lambda rows: rows["stay_id"] + 10_000
    ).to_parquet(mapping, index=False)
    for field, value in {
        "EXPORT_ROOT": str(export),
        "EXPORT_MANIFEST": manifest.name,
        "EXPORT_MANIFEST_SHA256": sha256(manifest.read_bytes()).hexdigest(),
        "DATABASE": "miiv",
        "MAPPING_PATH": str(mapping),
        "MAPPING_SHA256": sha256(mapping.read_bytes()).hexdigest(),
        "STAY_COLUMN": "stay_id",
        "PATIENT_COLUMN": "patient_key",
        "AUTHORITY_REF": "synthetic/grouping/v1",
    }.items():
        monkeypatch.setenv(f"EASYICU_PATIENT_GROUPING_{field}", value)
    study = study_contexts.upsert_context(
        {
            "id": "synthetic-trial",
            "question": (
                "In patients with suspected infection, set time zero at hour 6 after ICU admission "
                "and start vasoactive drugs within 6 hours versus deferring treatment. "
                "Compare 28-day mortality, adjusted for age."
            ),
            "outcome": "28-day mortality",
            "modules": ["demographics", "medications", "outcome"],
            "data_source": {"database": "miiv", "path": str(export)},
            "time_window": {"hours": 24, "anchor": "ICU admission"},
            "analysis_design": {
                "analysis_family": "causal_inference",
                "analysis_unit": "icu_stay",
                "variance_estimator": "model_based",
            },
            "stated_adjustments": [
                {"concept": "age", "variable": "age", "source_status": "available", "words": "adjusted for age"}
            ],
        },
        _server_question_wording_write=True,
    )
    gateway = FakeGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    project_id = "synthetic-project"
    study_id = study["id"]
    session_id = service.create_session(
        project_id=project_id, study_context_id=study_id, external_llm_opt_in=True
    )["session"]["session_id"]
    allow_unrelated_message_test(service, session_id)
    monkeypatch.setattr(routes, "get_pi_copilot_service", lambda: service)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    url = f"/api/copilot/pi/sessions/{session_id}"

    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "proposal", card
    assert not study_contexts.get_context(study_id).get("target_trial_design")
    body = {
        "project_id": project_id, "study_context_id": study_id,
        "expected_revision": study["revision"], "proposal_sha256": card["proposal"]["digest"],
    }
    wrong_digest = client.post(f"{url}/target-trial-proposal", json={**body, "proposal_sha256": "0" * 64})
    assert wrong_digest.status_code == 409, wrong_digest.json()
    assert target_trial_setup.latest_target_trial_compile(study_id) is None
    response = client.post(f"{url}/target-trial-proposal", json=body)
    assert response.status_code == 200, response.json()
    job = jobs.MANAGER.get(response.json()["job_id"])
    deadline = time.monotonic() + 15
    while job.status in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status == "done", job.error

    stated = study_contexts.get_context(study_id)
    latest = target_trial_setup.latest_target_trial_compile(study_id)
    review = target_trial_card.target_trial_card(stated, latest)
    assert review["state"] == "approvable", (job.result, review["blocking"])
    assert review["confirm_all_on_approval"] is True
    assert review["approval"] is None
    assert target_trial_card.target_trial_next_action(stated, latest) == "target_trial_review"
    assert not stated["target_trial_design"].get("approval")
    approval = client.post(
        f"{url}/target-trial-approval",
        json={
            "project_id": project_id, "study_context_id": study_id,
            "expected_revision": stated["revision"],
            "compile_sha256": review["compile_sha256"],
            "n_lines_confirmed": review["confirmation_lines"],
        },
    )
    assert approval.status_code == 200, approval.json()
    locked = study_contexts.get_context(study_id)
    assert locked["target_trial_design"]["approval"]["approval_event_id"] == approval.json()["approval_event_id"]
    assert target_trial_card.target_trial_next_action(locked, latest) == "target_trial_plan_ready"
    assert not any(method == "session.prompt" for method, *_ in gateway.calls)


@pytest.mark.parametrize("words", ["example class", "示例XRT治疗类", "EXAMPLE_xrt"])
def test_explicit_treatment_class_uses_the_sources_exact_registered_capture(
    monkeypatch, words,
):
    from types import SimpleNamespace

    from easyicu.webserver import target_trial_proposal as owner

    def capture(database, concept, *agents):
        return SimpleNamespace(
            database=database,
            concept=concept,
            agents=agents,
            absent_in_capture="absent_row_is_no_event",
        )

    entries = [
        capture("miiv", "example_drug_a", "a"),
        capture("miiv", "example_drug_b", "b"),
        capture("miiv", "broader_group", "a", "b", "c"),
        capture("eicu", "unrelated_source", "a", "b"),
    ]
    registry = SimpleNamespace(
        entries=entries,
        treatment_classes=("example_XRT",),
        class_agents=lambda _: frozenset({"a", "b"}),
    )
    monkeypatch.setattr(
        owner,
        "packaged_treatment_capture_registry",
        lambda: SimpleNamespace(registry=registry),
    )
    monkeypatch.setattr(
        owner,
        "supported_treatment_classes",
        lambda: [
            {
                "id": "example_XRT",
                "label_en": "example class",
                "label_zh": "示例XRT治疗类",
            },
        ],
    )
    monkeypatch.setattr(owner, "question_treatment_concepts", lambda *args: ())
    reading = owner.CausalTrialReading(
        "question_causal_method", (), treatment=(0, len(words))
    )
    assert owner._treatment(words, reading, "miiv") == (
        ("example_drug_a", "example_drug_b"),
        "example_XRT",
    )
    entries.pop(1)
    assert owner._treatment(words, reading, "miiv") is None


def test_only_missing_adjustment_is_requested_on_the_same_card(study, monkeypatch):
    from easyicu.webserver import target_trial_proposal as owner

    monkeypatch.setattr(
        owner, "_treatment", lambda *args: (("sample_event",), "sample_class")
    )
    monkeypatch.setattr(owner, "_indication_concept", lambda _: "example_condition")
    study["question"] = (
        study["question"].split(", adjusted", 1)[0].split("，调整", 1)[0] + "。"
    )
    study["stated_adjustments"] = []
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "clarification"
    assert card["proposal"]["missing"] == ["adjustment"]


@pytest.mark.parametrize(
    ("primary_en", "primary_zh"),
    [("365-day mortality", "365天死亡率"), ("mortality at 365 days", "365日全因死亡率")],
)
def test_confirmed_horizon_selects_the_primary_when_two_are_mentioned(complete_trial, primary_en, primary_zh):
    study = complete_trial
    zh = "示例" in study["question"]
    first, primary = ("90天死亡率", primary_zh) if zh else ("90-day mortality", primary_en)
    question = study["question"]
    join = "和" if zh else " and "
    study["question"] = question.replace(first, first + join + primary, 1)
    assert owner.target_trial_proposal(study).statement is None
    study["outcome"] = "In-hospital mortality"
    assert owner.target_trial_proposal(study).statement is None
    study["outcome"] = "365-day survival"
    assert owner.target_trial_proposal(study).statement is None
    study["outcome"] = primary
    proposal = owner.target_trial_proposal(study)
    assert proposal.statement is not None
    assert proposal.unread_words == []
    assert proposal.statement.spec.outcome.endpoint == "mort_365d"
    assert next(row["value"] for row in proposal.elements if row["key"] == "outcome") == primary


@pytest.mark.parametrize(
    ("secondary_en", "secondary_zh"),
    [
        ("90-day survival", "90天生存"),
        ("90-day in-hospital mortality", "90天院内死亡率"),
        ("90-day mortality from intubation", "90天死亡率自插管后起算"),
        ("90-day", "90天"),
    ],
)
def test_confirmed_horizon_leaves_inexact_secondary_endpoints_unread(
    complete_trial, secondary_en, secondary_zh,
):
    study = complete_trial
    zh = "示例" in study["question"]
    first, primary = ("90天死亡率", "365天死亡率") if zh else ("90-day mortality", "365-day mortality")
    secondary = secondary_zh if zh else secondary_en
    join = "和" if zh else " and "
    study["question"] = study["question"].replace(first, secondary + join + primary, 1)
    study["outcome"] = primary
    proposal = owner.target_trial_proposal(study)
    assert proposal.statement is None
    assert "unread_words" in proposal.missing


_REVIEW_EN = (
    "In patients with suspected infection, with time zero at hour 8 after ICU admission, "
    "compare starting sample therapy within 10 hours versus not starting it within 10 hours "
    "for 90-day mortality, adjusted for age."
)
_REVIEW_ZH = (
    "在疑似感染患者中，以入ICU后第8小时为时间零点，10小时内开始示例治疗，"
    "与这10小时内不开始相比，比较90天死亡率，调整年龄。"
)


_ORIGIN_THEFT = (
    (
        False, _REVIEW_EN.replace(" for 90-day mortality", ", 90-day mortality"),
        "not starting it within 10 hours", "after ICU admission",
    ),
    (
        False, _REVIEW_EN.replace(" versus", ", 90-day mortality, versus").replace(" for 90-day mortality", ""),
        "starting sample therapy within 10 hours", "after ICU admission",
    ),
    (
        False, _REVIEW_EN.replace(" for 90-day mortality, adjusted for age", ", adjusted for age, 90-day mortality"),
        "adjusted for age", "after ICU admission",
    ),
    (
        True, _REVIEW_ZH.replace("比较90天死亡率，调整年龄", "调整年龄，90天死亡率"),
        "调整年龄", "自入ICU后起算",
    ),
    (
        True, _REVIEW_ZH.replace("10小时内开始", "比较90天死亡率10小时内开始").replace("比较90天死亡率，", ""),
        "90天死亡率", "入ICU后",
    ),
    (
        False, _REVIEW_EN.replace(" for 90-day mortality", " 90-day mortality"),
        "not starting it within 10 hours", "after ICU admission",
    ),
)


def _review_study(question, zh):
    return {
        "question": question,
        "analysis_design": {"analysis_family": "causal_inference"},
        "data_source": {"database": "miiv", "path": "synthetic"},
        "stated_adjustments": [
            {"concept": "age", "variable": "年龄" if zh else "age",
             "source_status": "available", "words": "调整年龄" if zh else "adjusted for age"}
        ],
    }


@pytest.mark.parametrize(
    ("original", "replacement", "unread"),
    [
        ("90-day mortality", "90-day mortality 24 hours after ICU admission", "24 hours after ICU admission"),
        ("90-day mortality", "90-day mortality from ICU admission at hour 8", "at hour 8"),
        ("90天死亡率", "自入ICU后24小时起90天死亡率", "24小时起"),
        ("90天死亡率", "插管算起90天死亡率", "插管算起"),
        ("90天死亡率", "入院起90天死亡率", "入院起"),
        ("10小时内开始示例治疗", "插管算起10小时内开始示例治疗", "插管算起"),
        ("90-day mortality", "90-day mortality in patients with septic shock", "patients with septic shock"),
        ("age.", "age, age >= 65.", "age >= 65"),
        ("age.", "age, children excluded.", "children excluded"),
        ("90天死亡率", "90天死亡率，在脓毒症休克患者中", "脓毒症休克患者"),
        ("90-day mortality", "mortality after discharge, i.e. 90-day mortality", "mortality after discharge, i.e"),
        ("starting sample", "from intubation, e.g. starting sample", "from intubation, e.g"),
        ("90-day mortality", "90-day mortality from initiation of oxygen", "from initiation of oxygen"),
        ("90天死亡率", "转入病房后90天死亡率", "转入病房后"),
        ("90-day mortality", "90-day mortality in patients receiving dialysis", "patients receiving dialysis"),
        ("90天死亡率", "90天死亡率，肾功能正常者除外", "肾功能正常者除外"),
        ("90-day mortality", "shortly after transfer, namely 90-day mortality", "shortly after transfer"),
        ("90-day mortality", "90-day mortality from time zero", "from time zero"),
        ("90天死亡率", "自时间零点起90天死亡率", "自时间零点起"),
        ("starting sample", "from ICU admission, starting sample", "from ICU admission"),
        ("10小时内开始示例治疗", "拔管后10小时内开始示例治疗", "拔管后"),
        ("suspected infection,", "suspected infection, excluding children,", "excluding children"),
        ("疑似感染患者中，", "疑似感染患者中，年龄≥65岁，", "年龄≥65岁"),
    ],
)
def test_review_residual_words_stop_compilation(synthetic_treatment, original, replacement, unread):
    zh = any(ord(char) > 127 for char in original)
    question = (_REVIEW_ZH if zh else _REVIEW_EN).replace(original, replacement, 1)
    study = _review_study(question, zh)
    result = owner.target_trial_proposal(study)
    assert result is not None
    assert result.statement is None
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "clarification"
    assert "unread_words" in card["proposal"]["missing"]
    assert any(unread in fragment for fragment in card["proposal"]["unread_words"])
    assert {row["key"] for row in result.elements} >= {
        "eligibility", "treatment", "strategies", "time_zero", "grace_period", "outcome", "adjustment",
    }


@pytest.mark.parametrize(("zh", "question", "term", "origin"), _ORIGIN_THEFT)
def test_review_origin_between_elements_stays_unread(synthetic_treatment, zh, question, term, origin):
    start = question.index(term)
    altered = _insert_origin(question, start + len(term), origin, zh)
    proposal = owner.target_trial_proposal(_review_study(altered, zh))
    assert proposal is not None
    assert proposal.statement is None
    assert "unread_words" in proposal.missing
    assert any(origin in fragment for fragment in proposal.unread_words)


@pytest.mark.parametrize(
    ("zh", "suffix"),
    [
        (False, "adjusted for age in patients with septic shock"),
        (False, "adjusted for age from intubation"),
        (False, "adjusted for age among hospital survivors"),
        (True, "调整年龄（脓毒症休克患者中）"),
        (True, "调整插管后年龄"),
        (True, "校正入院前的年龄"),
    ],
)
def test_adjustment_quote_cannot_cover_a_population_or_origin(synthetic_treatment, zh, suffix):
    original = "调整年龄" if zh else "adjusted for age"
    question = (_REVIEW_ZH if zh else _REVIEW_EN).replace(original, suffix)
    study = _review_study(question, zh)
    study["stated_adjustments"][0]["words"] = suffix
    result = owner.target_trial_proposal(study)
    assert result.statement is None
    assert "unread_words" in result.missing


@pytest.mark.parametrize(
    ("zh", "before", "after"),
    [
        (False, "compare starting", "90-day mortality after ICU admission, compare starting"),
        (True, "10小时内开始", "比较入住ICU后90天死亡率，10小时内开始"),
        (True, "10小时内开始", "比较入住ICU后的90天死亡率，10小时内开始"),
        (True, "比较90天死亡率，调整年龄", "调整年龄后比较90天死亡率"),
        (True, "比较90天死亡率，调整年龄", "在调整年龄后，比较90天死亡率"),
        (True, "比较90天死亡率", "比较90天死亡率自入ICU后起算"),
        (True, "比较90天死亡率，调整年龄", "经校正年龄后比较90天死亡率"),
    ],
)
def test_outcome_and_adjustment_do_not_bleed_into_other_elements(
    synthetic_treatment, zh, before, after,
):
    question = _REVIEW_ZH if zh else _REVIEW_EN
    if "compare starting" == before or after.startswith("比较入住"):
        question = question.replace(" for 90-day mortality" if not zh else "比较90天死亡率，", "")
    question = question.replace(before, after, 1)
    study = _review_study(question, zh)
    if "校正" in question:
        study["stated_adjustments"][0]["words"] = "经校正年龄后"
    result = owner.target_trial_proposal(study)
    assert result is not None
    assert result.missing == []
    assert result.statement.spec.grace_period.hours == 10
    endpoint = owner.fixed_horizon_mortality_endpoint(result.statement.spec.outcome.endpoint)
    assert (endpoint.horizon_days, endpoint.time_origin) == (90, "icu_admission")


_BOUNDARY_TERMS = {
    False: (
        "In patients with suspected infection", "time zero at hour 8 after ICU admission",
        "starting sample therapy within 10 hours", "sample therapy",
        "not starting it within 10 hours", "90-day mortality", "adjusted for age",
    ),
    True: (
        "在疑似感染患者中", "入ICU后第8小时为时间零点", "10小时内开始示例治疗",
        "示例治疗", "这10小时内不开始", "90天死亡率", "调整年龄",
    ),
}


@pytest.mark.parametrize(
    ("zh", "question"), [(False, _REVIEW_EN), (True, _REVIEW_ZH)],
)
def test_non_connectives_cannot_compile_at_any_element_boundary(synthetic_treatment, zh, question):
    assert owner.target_trial_proposal(_review_study(question, zh)).statement
    for term in _BOUNDARY_TERMS[zh]:
        start = question.index(term)
        for boundary in (start, start + len(term)):
            for words in ("zzz", "5 days", "except", "除了", "转运"):
                altered = question[:boundary] + f" {words} " + question[boundary:]
                result = owner.target_trial_proposal(_review_study(altered, zh))
                assert result is None or result.statement is None, (term, boundary, words)


def _insert_origin(question, boundary, origin, zh):
    phrase = origin if zh and origin == "自入ICU后起算" else f" {origin}"
    following = question[boundary:]
    if not zh and following and following[0].isalnum():
        phrase += " "
    return question[:boundary] + phrase + following


def _trial_timing(statement):
    spec = statement.spec
    endpoint = owner.fixed_horizon_mortality_endpoint(spec.outcome.endpoint)
    return (
        spec.time_zero.hours_after_icu_admission, spec.grace_period.hours,
        endpoint.horizon_days, endpoint.time_origin,
    )


@pytest.mark.parametrize(
    ("zh", "question", "ambiguous_term", "ambiguous_origin"),
    [
        (False, _REVIEW_EN, None, None),
        (True, _REVIEW_ZH, None, None),
        *_ORIGIN_THEFT,
    ],
)
def test_origin_insertions_preserve_trial_timing_or_stop(
    synthetic_treatment, zh, question, ambiguous_term, ambiguous_origin,
):
    baseline = _trial_timing(owner.target_trial_proposal(_review_study(question, zh)).statement)
    origins = ("after ICU admission", "自入ICU后起算", "from intubation", "入ICU后")
    for term in _BOUNDARY_TERMS[zh]:
        start = question.index(term)
        end = start + len(term)
        for boundary in (start, end):
            for origin in origins:
                altered = _insert_origin(question, boundary, origin, zh)
                proposal = owner.target_trial_proposal(_review_study(altered, zh))
                statement = proposal.statement if proposal is not None else None
                case = (term, boundary, origin, altered)
                assert statement is None or _trial_timing(statement) == baseline, case
                if term == ambiguous_term and boundary == end and origin == ambiguous_origin:
                    assert statement is None, case
                    assert any(origin in fragment for fragment in proposal.unread_words), case


@pytest.mark.parametrize(
    ("zh", "extra"),
    [(False, "２４　 hours  after ICU admission"), (True, "插管　算起"), (False, "cafe\u0301 ﬃ")],
)
def test_normalized_coverage_displays_original_residual(synthetic_treatment, zh, extra):
    base = _REVIEW_ZH if zh else _REVIEW_EN
    study = _review_study("　" + base.replace("8", "８").replace("10", "１０"), zh)
    assert owner.target_trial_proposal(study).statement is not None
    term = "90天死亡率" if zh else "90-day mortality"
    study["question"] = study["question"].replace(term, f"{term} {extra}", 1)
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "clarification"
    assert any(extra in fragment for fragment in card["proposal"]["unread_words"])


def test_full_span_reader_preserves_the_resolvable_indication_names():
    from easyicu.webserver.study_intent import full_span_concepts

    names = {
        "suspected infection": "susp_inf",
        "疑似感染": "susp_inf",
        "Sepsis-3 (SOFA-1 based)": "sep3_sofa1",
        "Sepsis-3诊断 (基于传统SOFA)": "sep3_sofa1",
        "Circulatory Failure": "circ_failure",
        "循环衰竭": "circ_failure",
        "休克": "circ_failure",
        "RRT": "rrt",
        "Renal Replacement Therapy": "rrt",
        "肾脏替代治疗": "rrt",
        "透析": "rrt",
        "Supplemental Oxygen": "supp_o2",
        "辅助吸氧": "supp_o2",
        "Ventilation Duration Windows": "vent_ind",
        "机械通气时间窗": "vent_ind",
        "Vasopressor Indicator": "vaso_ind",
        "血管活性药物指示": "vaso_ind",
        "Advanced Respiratory Support": "adv_resp",
        "高级呼吸支持 (IMV/NIV/HFNC)": "adv_resp",
        "ECMO in Use": "ecmo",
        "ECMO使用中": "ecmo",
        "Positive Blood Culture": "bld_culture_positive",
        "血培养阳性": "bld_culture_positive",
        "Any Positive Culture": "culture_positive",
        "任意培养阳性": "culture_positive",
        "ICU Readmission (same hospitalisation)": "icu_readmission",
        "ICU 再入 (同次住院)": "icu_readmission",
        "Persistent Critical Illness": "persistent_critical_illness",
        "持续危重症": "persistent_critical_illness",
        "persistent_critical_illness": "persistent_critical_illness",
    }
    assert {words: owner._indication_concept(words) for words in names} == names
    assert all(full_span_concepts(words) == {concept} for words, concept in names.items())
    for words in ("possibly suspected infection", "复苏后的疑似感染", "adult patients with RRT"):
        assert not full_span_concepts(words)


@pytest.mark.parametrize(
    ("zh", "clause", "variables"),
    [
        (False, "after adjusting for age, sex, and BMI", ("age", "sex", "BMI")),
        (True, "在调整年龄、性别和体重指数后", ("年龄", "性别", "体重指数")),
    ],
)
def test_adjustment_accounts_for_a_closed_list_of_variables(synthetic_treatment, zh, clause, variables):
    original = "调整年龄" if zh else "adjusted for age"
    question = (_REVIEW_ZH if zh else _REVIEW_EN).replace(original, clause)
    study = _review_study(question, zh)
    study["stated_adjustments"] = [
        {"concept": concept, "variable": variable, "words": clause, "source_status": "available"}
        for variable, concept in zip(variables, ("age", "sex", "bmi"))
    ]
    proposal = owner.target_trial_proposal(study)
    assert proposal.statement is not None, proposal.unread_words
    assert [row.name for row in proposal.statement.spec.confounders] == ["age", "sex", "bmi"]
    study["stated_adjustments"].pop()
    assert owner.target_trial_proposal(study).missing == ["adjustment", "unread_words"]


def test_reversed_strategy_order_remains_an_unread_pair(synthetic_treatment):
    question = _REVIEW_EN.replace(
        "starting sample therapy within 10 hours versus not starting it within 10 hours",
        "not starting sample therapy within 10 hours versus starting it within 10 hours",
    ).replace("90-day mortality", "90-day mortality after ICU admission")
    assert owner.causal_trial_reading(question) is None
    assert owner.target_trial_proposal(_review_study(question, False)) is None
