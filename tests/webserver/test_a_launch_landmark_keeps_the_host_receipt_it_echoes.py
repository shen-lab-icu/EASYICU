"""A launch-staged landmark keeps the concept-selection receipt it echoes (B30, S1735).

UX leg 3 on rc_4cc: 生成候选计划 started no job ("study_concept_selection_
authority_server_owned").  The host's agent landmark patch merges the stored
confirmations, including a host-written ``concept_selection_*_user_turn_verified``
receipt, and the launch staged it through ``validate_context_update``, which
refused the receipt that ``upsert_context`` accepts as an unchanged echo.  The
host refused its own echo.  An unchanged receipt is now accepted on both paths;
a created or flipped one still fails closed.  Isolated stores only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.webserver import launch_study_updates
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.pi_copilot import plan_decisions

_RECEIPT = "concept_selection_lact_user_turn_verified"
_PEAK = "在脓毒症 ICU 患者中，ICU 入科后 0–24 小时的峰值乳酸与院内死亡有什么关联？"
_FIRST = "在脓毒症 ICU 患者中，ICU 入科时的首次乳酸与院内死亡有什么关联？"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )


def _verified_study(question: str = _PEAK) -> dict:
    return context_store.upsert_context(
        {
            "id": "study_landmark_receipt",
            "question": question,
            "analysis_design": {
                "analysis_family": "association_study",
                "analysis_unit": "icu_stay",
                "variance_estimator": "model_based",
            },
            "confirmations": {"feature_time_window": True, _RECEIPT: True},
        },
        lifecycle_write=False,
        _server_concept_selection_authority_write=True,
    )


def _staged(study: dict, patch: dict) -> dict:
    token = launch_study_updates._CURRENT.set(
        launch_study_updates._Updates(study["id"])
    )
    try:
        launch_study_updates.upsert_launch_context(study, {"id": study["id"], **patch})
        return launch_study_updates.handoff_launch_context(
            study["id"], expected_revision=study["revision"]
        )
    finally:
        launch_study_updates._CURRENT.reset(token)


@pytest.mark.parametrize("question", [_PEAK, _FIRST], ids=["24h-peak", "admission-first"])
def test_the_agent_landmark_stages_with_the_host_receipt(question: str) -> None:
    study = _verified_study(question)
    patch = plan_decisions.agent_post_baseline_landmark_patch(study)
    assert patch["confirmations"][_RECEIPT] is True  # the echo the host refused

    saved = _staged(study, patch)

    assert saved["confirmations"][_RECEIPT] is True
    assert saved["confirmations"]["plan_timing_landmark_24h"] is True
    assert context_store.get_context(study["id"])["confirmations"][_RECEIPT] is True


@pytest.mark.parametrize(
    "confirmations",
    [
        pytest.param({_RECEIPT: False}, id="flipped"),
        pytest.param({"concept_selection_sep3_sofa2_user_turn_verified": True}, id="created"),
    ],
)
def test_a_changed_receipt_is_still_refused_at_launch(confirmations: dict) -> None:
    study = _verified_study()
    patch = plan_decisions.agent_post_baseline_landmark_patch(study)
    patch["confirmations"] = {**patch["confirmations"], **confirmations}

    with pytest.raises(context_store.StudyContextError) as refused:
        _staged(study, patch)

    assert refused.value.detail["error"] == "study_concept_selection_authority_server_owned"
    assert context_store.get_context(study["id"])["confirmations"][_RECEIPT] is True
