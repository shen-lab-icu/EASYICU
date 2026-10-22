"""A causal study bound to a database v1 emulates no trial in is told so first.

v1 emulates target trials in MIMIC-IV only.  A causal question saved before
its source was bound recorded the causal design; binding eICU then asked the
researcher to state the trial, and only the compile of that statement would
have refused the database (hosted pilot, 10-22).  The card now stops with the
database's code before any statement, and the workflow does not ask for one.
"""

from __future__ import annotations

from typing import Any

from easyicu.webserver import target_trial_card as card_owner
from easyicu.webserver.target_trial_setup import TARGET_TRIAL_DATABASE_OUT_OF_SCOPE

_CAUSAL = {"analysis_family": "causal_inference", "analysis_unit": "icu_stay"}


def _study(database: str | None, **fields: Any) -> dict[str, Any]:
    return {
        "id": "study_trial_scope",
        "question": "早期使用去甲肾上腺素对院内死亡的因果效应有多大？请用目标试验模拟。",
        "analysis_design": dict(_CAUSAL),
        "data_source": {"database": database} if database else {},
        **fields,
    }


def test_an_eicu_causal_study_stops_before_a_trial_is_asked_for():
    study = _study("eicu")

    card = card_owner.target_trial_card(study, None)

    assert card is not None
    assert (card["state"], card["reason_code"], card["approvable"]) == (
        "stopped", TARGET_TRIAL_DATABASE_OUT_OF_SCOPE, False,
    )
    assert card_owner.target_trial_next_action(study, None) == "target_trial_review"


def test_a_mimic_iv_or_unbound_causal_study_is_still_asked_for_its_trial():
    for study in (_study("miiv"), _study(None)):
        assert card_owner.target_trial_card(study, None) is None
        assert card_owner.target_trial_next_action(study, None) == "target_trial_statement_needed"


def test_a_study_of_another_family_has_no_trial_card():
    study = _study("eicu", analysis_design={"analysis_family": "association_study"})

    assert card_owner.target_trial_card(study, None) is None
    assert card_owner.target_trial_next_action(study, None) is None
