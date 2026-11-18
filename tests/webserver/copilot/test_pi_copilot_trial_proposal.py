"""A causal question can be reviewed without supplying an internal trial schema."""

from pathlib import Path

import pytest

from easyicu.webserver import target_trial_card
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot


@pytest.fixture(
    params=[
        "In patients with the index condition, with time zero at hour 8 after ICU admission, "
        "start sample therapy within 10 hours versus not starting it within 10 hours "
        "for 90-day mortality, adjusted for age.",
        "示例病症患者中，以入 ICU 后第8小时为时间零点，10小时内开始示例治疗，"
        "与这10小时内不开始相比，比较90天死亡率，调整年龄。",
    ]
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


def test_question_has_an_inline_proposal_and_no_plan_offer_before_review(
    study, monkeypatch
):
    # Use a synthetic capture entry; the real owner tests exercise existing registry data.
    from easyicu.webserver import target_trial_proposal as owner

    monkeypatch.setattr(
        owner, "_treatment", lambda *args: (("sample_event",), "sample_class")
    )
    monkeypatch.setattr(owner, "_indication_concept", lambda words: "example_condition")
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
    assert statement.spec.confounders[0].name == "age"


def test_missing_clinical_details_are_one_clarification(study):
    study["question"] = "What is the causal effect of an unspecified therapy?"
    study["stated_adjustments"] = []
    card = target_trial_card.target_trial_card(study, None)
    assert card["state"] == "clarification"
    assert "treatment" in card["proposal"]["missing"]
    assert card["proposal"]["digest"] is None
    assert card["approval"] is None


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


def test_explicit_treatment_class_uses_the_sources_exact_registered_capture(
    monkeypatch,
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
        entries=entries, class_agents=lambda _: frozenset({"a", "b"})
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
                "id": "example_class",
                "label_en": "example class",
                "label_zh": "示例药物类",
            },
        ],
    )
    monkeypatch.setattr(owner, "question_treatment_concepts", lambda *args: ())
    words = "example class"
    reading = owner.CausalTrialReading(
        "question_causal_method", (), treatment=(0, len(words))
    )
    assert owner._treatment(words, reading, "miiv") == (
        ("example_drug_a", "example_drug_b"),
        "example_class",
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


def test_confirmed_horizon_selects_the_primary_when_two_are_mentioned():
    from easyicu.webserver import target_trial_proposal as owner

    question = "Assess 90-day mortality and 365-day mortality."
    assert owner._fixed_outcome(question, "", "miiv") is None
    assert owner._fixed_outcome(question, "In-hospital mortality", "miiv") is None
    endpoint, words = owner._fixed_outcome(question, "365-day mortality", "miiv")
    assert endpoint.horizon_days == 365
    assert words == "365-day mortality"
    assert (
        owner._fixed_outcome("Does 90-day mortality from time zero differ?", "", "miiv")
        is None
    )
