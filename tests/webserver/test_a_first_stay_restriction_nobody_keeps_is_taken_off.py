"""A first-stay restriction nobody keeps is taken off before the next plan.

UX rc_150 age leg (S1780/S1782): a plan quoted the conversation's note
「首次 ICU 住院与重复 ICU 住院的处理留待候选研究计划提出」 as the study's own
restriction, 应用这些人群条件 saved ``first_icu_stay_per_hospitalization``, the
export could not prove it, and every later plan -- including one the researcher
asked, through 提出修改, to drop it -- stopped at prepare on the same refusal.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.webserver import research_run_submission as submission
from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot.population_preparation import (
    UNSOURCED_FIRST_STAY_DROPPED,
    first_stay_planning_repair,
)
from easyicu.webserver.pi_copilot.study_context_update import (
    message_explicitly_selects_all_stays,
)
from easyicu.webserver.plan_change_request import PlanChangeRequest

UX_QUESTION = "成人 ICU 患者中，ICU 入科时年龄（每增加 10 岁）与院内死亡的关联，校正性别。请用官方 MIMIC-IV 演示数据。"
UX_REVIEW = "纳入成人 ICU 患者；首次 ICU 住院与重复 ICU 住院的处理留待候选研究计划提出。"


def _study(question: str = UX_QUESTION, **changes: Any) -> dict[str, Any]:
    study = {
        "id": "study-age", "revision": 28, "question": question,
        "cohort": {"label": "成人 ICU 患者", "review": UX_REVIEW, "age_min": 18,
                   "first_icu_stay_per_hospitalization": True},
        "confirmations": {"outcome": True, "plan_repeated_stays_clustered": True},
        "data_source": {"path": "/synthetic/export", "database": "miiv"},
    }
    return {**study, **changes}


def test_the_ux_studys_unsourced_unprovable_restriction_is_dropped_with_its_receipt() -> None:
    study = _study()

    patch = first_stay_planning_repair(study, source_proves_hospital_first=False)

    assert patch is not None
    assert "first_icu_stay_per_hospitalization" not in patch["cohort"]
    assert patch["cohort"]["age_min"] == 18 and patch["cohort"]["review"] == UX_REVIEW
    assert patch["confirmations"] == {**study["confirmations"], UNSOURCED_FIRST_STAY_DROPPED: True}


@pytest.mark.parametrize("keeps", ["source_proves", "question", "label", "chosen"])
def test_a_restriction_the_source_proves_or_the_researcher_states_is_kept(keeps) -> None:
    study = _study()
    if keeps == "question":
        study = _study("首次 ICU 住院的成人患者中，年龄与院内死亡的关联。")
    elif keeps == "label":
        study["cohort"] = {**study["cohort"], "label": "成人首次 ICU 住院患者"}
    elif keeps == "chosen":
        study["confirmations"] = {"plan_repeated_stays_first": True}

    assert first_stay_planning_repair(study, source_proves_hospital_first=keeps == "source_proves") is None


@pytest.mark.parametrize("message", [
    "请去掉首次 ICU 住院的限制，其余不变。",
    "取消“首次 ICU 住院”这个条件",
    "Please drop the first ICU stay restriction.",
])
def test_a_request_that_removes_the_restriction_keeps_every_stay(message) -> None:
    # Even when the question itself says first stay, the researcher's request wins.
    study = _study("首次 ICU 住院的成人患者中，年龄与院内死亡的关联。")

    assert message_explicitly_selects_all_stays(message)
    patch = first_stay_planning_repair(study, message=message, source_proves_hospital_first=True)

    assert patch == {"cohort": {**study["cohort"], "exclude_readmissions": False,
                                "first_icu_stay_per_hospitalization": False}}


def test_a_request_about_something_else_does_not_touch_a_kept_restriction() -> None:
    study = _study("首次 ICU 住院的成人患者中，年龄与院内死亡的关联。")

    assert not message_explicitly_selects_all_stays("请把年龄改为每增加 5 岁。")
    assert first_stay_planning_repair(
        study, message="请把年龄改为每增加 5 岁。", source_proves_hospital_first=False,
    ) is None


def _request(change: PlanChangeRequest | None) -> submission.ResearchRunSubmissionRequest:
    return submission.ResearchRunSubmissionRequest(
        study_context_id="study-age", provider="mock", credential_source="pi_verified",
        external_llm_opt_in=True, intent="candidate_plan", planner_start_mode="fresh",
        plan_change_request=change,
    )


@pytest.fixture
def launch(monkeypatch):
    from easyicu.webserver import research_launch_scientific, source_identity_authority

    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(source_identity_authority, "resolve_study_first_icu_stay_per_hospitalization",
                        lambda **_: None)
    monkeypatch.setattr(research_launch_scientific, "bound_source_authority_path", lambda path, study: path)

    def write(study, patch, *, label):
        if patch is None:
            return study
        writes.append(dict(patch))
        return {**study, **patch, "revision": study["revision"] + 1}

    monkeypatch.setattr(submission, "_with_planning_default", write)
    return writes


def test_a_fresh_plan_is_drafted_without_the_ux_restriction(launch) -> None:
    study, request = submission._with_first_stay_repair(_study(), _request(None), path="/synthetic/export")

    assert "first_icu_stay_per_hospitalization" not in study["cohort"]
    assert study["confirmations"][UNSOURCED_FIRST_STAY_DROPPED] is True
    assert request.plan_change_request is None and len(launch) == 1


def test_a_change_request_is_drafted_on_the_study_it_repaired(launch) -> None:
    study = _study()
    digest = study_contexts.scientific_configuration_sha256(study)
    change = PlanChangeRequest(
        source_run_id="run-age", user_message="请去掉首次 ICU 住院的限制。",
        source_scientific_configuration_sha256=digest, target_scientific_configuration_sha256=digest,
    )

    repaired, request = submission._with_first_stay_repair(study, _request(change), path="/synthetic/export")

    assert repaired["cohort"]["first_icu_stay_per_hospitalization"] is False
    assert repaired["cohort"]["exclude_readmissions"] is False
    revised = request.plan_change_request
    assert revised.source_scientific_configuration_sha256 == digest
    assert revised.target_scientific_configuration_sha256 == study_contexts.scientific_configuration_sha256(repaired)
    assert revised.target_scientific_configuration_sha256 != digest


def test_a_study_with_no_first_stay_restriction_is_not_read_or_written(launch, monkeypatch) -> None:
    from easyicu.webserver import source_identity_authority

    def unexpected(**_):
        raise AssertionError("a study without the restriction needs no source proof")

    monkeypatch.setattr(source_identity_authority, "resolve_study_first_icu_stay_per_hospitalization", unexpected)
    study = _study()
    study["cohort"] = {"label": "成人 ICU 患者", "age_min": 18}

    assert submission._with_first_stay_repair(study, _request(None), path="/synthetic/export")[0] is study
    assert launch == []
