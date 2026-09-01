"""The Copilot front door must read the user's question, never replace it.

These lock the defect this module was written for: a question about fluid
balance and AKI used to arrive at the backend as a question about lactate and
in-hospital mortality, because every slot the reader could not fill was filled
from a module default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import study_intent

STATIC = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static"


def _values(result):
    return {k: v["value"] for k, v in result["slots"].items() if v["value"] is not None}


# ---------------------------------------------------------------- reading ---
def test_aki_question_is_not_turned_into_a_mortality_question():
    result = study_intent.deterministic_intent(
        "在 ICU 患者中,早期液体正平衡与急性肾损伤(AKI)的发生风险是否相关?"
    )
    values = _values(result)
    assert values["outcome"] == "aki"
    assert values["exposure"] == "fluid_balance"
    # The specific substitution that shipped: never again.
    assert values["outcome"] != "death"
    assert values["exposure"] != "lact"


def test_sepsis3_is_the_cohort_not_the_outcome():
    result = study_intent.deterministic_intent(
        "Among Sepsis-3 patients, does early lactate predict in-hospital mortality?"
    )
    values = _values(result)
    assert values["population"] == "Sepsis-3 patients"
    assert values["outcome"] == "death"
    assert values["exposure"] == "lact"
    assert values["analysis_family"] == "prediction"
    readings = {concept for concept, _phrase in study_intent._match_concept(result["question"])}
    assert "sep3" in readings
    assert "sep3_sofa2" not in readings


def test_sofa2_sepsis_sensitivity_requires_an_explicit_sofa2_phrase():
    result = study_intent.deterministic_intent(
        "In an explicit SOFA-2 sepsis sensitivity analysis, describe mortality."
    )

    readings = {concept for concept, _phrase in study_intent._match_concept(result["question"])}
    assert "sep3_sofa2" in readings


def test_length_of_stay_is_an_outcome_not_a_population():
    result = study_intent.deterministic_intent(
        "Does first 24h KDIGO AKI stage associate with ICU length of stay?"
    )
    values = _values(result)
    assert values["outcome"] == "los_icu"
    assert values["outcome_type"] == "time_to_event"
    assert values["exposure"] == "aki_stage"
    assert values["time_window_hours"] == 24
    # "length of stay" must not be mistaken for a cohort statement.
    assert "population" not in values
    assert "population" in result["unread"]


# --------------------------------------------------- never fill a default ---
def test_unreadable_slots_stay_unread_and_are_named():
    result = study_intent.deterministic_intent("数据质量怎么样")
    values = _values(result)
    assert values == {"analysis_family": "data_quality"}
    for slot in ("population", "exposure", "outcome", "time_window_hours", "comparator"):
        assert slot in result["unread"], slot
    assert result["complete"] is False


def test_no_slot_is_ever_filled_from_a_default():
    """A sentence with no clinical content must read nothing at all."""
    result = study_intent.deterministic_intent("please help me write a paper")
    assert _values(result) == {}
    assert set(result["unread"]) == set(study_intent.SLOTS)


def test_sepsis_as_an_outcome_is_left_unread_rather_than_guessed():
    """Deliberate: "does X predict sepsis" needs the user, not a guess.

    Sepsis appears in this corpus overwhelmingly as the cohort. Rather than
    encode a rule that is right half the time, the reader declines and names
    the slot — the caller must ask. If a future change makes sepsis outcome-
    capable, it has to update this test on purpose.
    """
    result = study_intent.deterministic_intent(
        "Among ICU patients, does early lactate predict sepsis?"
    )
    assert "outcome" in result["unread"]
    assert _values(result).get("outcome") is None


@pytest.mark.parametrize(
    "question",
    [
        "我不是要研究死亡。我的结局是急性肾损伤 AKI(KDIGO 分期)。",
        "I am not studying mortality; my outcome is AKI stage.",
    ],
)
def test_a_negated_phrase_is_not_a_reading(question):
    """The user's correction must not become the thing they corrected."""
    result = study_intent.deterministic_intent(question)
    values = _values(result)
    assert values["outcome"] == "aki_stage"
    assert values.get("outcome") != "death"
    assert "death" not in values.values()


def test_amended_question_reads_exposure_from_the_first_sentence():
    """Guided appends a correction, so both sentences must be read together."""
    result = study_intent.deterministic_intent(
        "在 ICU 患者中,早期液体正平衡与急性肾损伤(AKI)的发生风险是否相关? "
        "我不是要研究死亡。我的结局是急性肾损伤 AKI(KDIGO 分期)。"
    )
    values = _values(result)
    assert values["exposure"] == "fluid_balance"
    assert values["outcome"] == "aki_stage"
    assert values["outcome_type"] == "ordinal"
    assert values["population"] == "ICU patients"


def test_one_disease_does_not_fill_two_slots():
    """"my outcome is AKI (KDIGO stage)" names one thing, not exposure+outcome."""
    result = study_intent.deterministic_intent(
        "I am not studying mortality; my outcome is AKI stage."
    )
    assert "exposure" in result["unread"]
    assert "population" in result["unread"]


def test_negation_scope_stops_at_a_sentence_boundary():
    """Chinese has no word spaces, so an unscoped negation window swallows the
    next clause: "不是要做质量审计。请问乳酸与院内死亡是否相关?" would lose both
    the exposure and the outcome."""
    result = study_intent.deterministic_intent(
        "不是要做质量审计。请问乳酸与院内死亡是否相关?"
    )
    values = _values(result)
    assert values["exposure"] == "lact"
    assert values["outcome"] == "death"


def test_a_negated_analysis_family_is_not_a_reading():
    result = study_intent.deterministic_intent(
        "Not a prediction study; is PEEP associated with 28-day mortality?"
    )
    assert _values(result)["analysis_family"] == "association"


def test_question_is_returned_verbatim():
    question = "Among adults, is early vasopressor exposure associated with 28-day mortality?"
    result = study_intent.deterministic_intent(question)
    assert result["question"] == question


# --------------------------------------------------------- the opt-in gate ---
def test_external_provider_without_opt_in_falls_back_and_says_so():
    result = study_intent.extract_study_intent(
        "Does lactate predict mortality?",
        llm_provider="openai",
        external_llm_opt_in=False,
        ai_enabled=False,
    )
    assert result["source"] == "deterministic"
    assert "llm_blocked_by_opt_in_gate" in result["notes"]
    assert result["provider_block"]["error"] == "external_llm_opt_in_required"


def test_offline_provider_never_calls_out():
    calls = []

    def transport(request, headers):  # pragma: no cover - must not run
        calls.append(request)
        return {}

    result = study_intent.extract_study_intent(
        "Does lactate predict mortality?",
        llm_provider="offline",
        transport=transport,
    )
    assert calls == []
    assert result["source"] == "deterministic"
    assert "llm_not_requested" in result["notes"]


# -------------------------------------------------- LLM output is validated ---
def _llm_response(payload):
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def test_llm_output_outside_the_closed_set_is_rejected(monkeypatch):
    monkeypatch.setattr(
        study_intent.provider_adapter,
        "_load_external_credentials",
        lambda *a, **k: {"model": "m", "api_key": "k", "base_url": "https://x/v1/chat/completions"},
    )
    result = study_intent.extract_study_intent(
        "Does lactate predict mortality?",
        llm_provider="openai",
        external_llm_opt_in=True,
        ai_enabled=True,
        transport=lambda req, hdr: _llm_response(
            {
                "population": None,
                "exposure": "lactate",
                "outcome": "mortality",
                "outcome_type": "binary",
                "time_window_hours": None,
                "comparator": None,
                "analysis_family": "vibes",  # not in the closed set
            }
        ),
    )
    assert result["source"] == "deterministic"
    assert any(n.startswith("llm_rejected:") for n in result["notes"])


def test_llm_may_not_invent_extra_fields(monkeypatch):
    monkeypatch.setattr(
        study_intent.provider_adapter,
        "_load_external_credentials",
        lambda *a, **k: {"model": "m", "api_key": "k", "base_url": "https://x/v1/chat/completions"},
    )
    result = study_intent.extract_study_intent(
        "Does lactate predict mortality?",
        llm_provider="openai",
        external_llm_opt_in=True,
        ai_enabled=True,
        transport=lambda req, hdr: _llm_response(
            {**{s: None for s in study_intent.SLOTS}, "recommended_model": "xgboost"}
        ),
    )
    assert result["source"] == "deterministic"
    assert "llm_rejected:study_intent_llm_unknown_fields" in result["notes"]


def test_valid_llm_output_is_used_and_never_loses_a_deterministic_read(monkeypatch):
    monkeypatch.setattr(
        study_intent.provider_adapter,
        "_load_external_credentials",
        lambda *a, **k: {"model": "m", "api_key": "k", "base_url": "https://x/v1/chat/completions"},
    )
    result = study_intent.extract_study_intent(
        "Among Sepsis-3 patients, does early lactate predict in-hospital mortality?",
        llm_provider="openai",
        external_llm_opt_in=True,
        ai_enabled=True,
        transport=lambda req, hdr: _llm_response(
            {
                "population": "Sepsis-3 adults",
                "exposure": "early lactate",
                "outcome": "in-hospital mortality",
                "outcome_type": "binary",
                "time_window_hours": None,
                "comparator": None,
                "analysis_family": "prediction",
            }
        ),
    )
    assert result["source"] == "llm"
    assert result["slots"]["population"]["value"] == "Sepsis-3 patients"
    # The model left the window unread; the deterministic reader had nothing
    # either, so it must stay unread rather than acquire a default.
    assert "time_window_hours" in result["unread"]


def test_llm_cannot_overwrite_a_deterministic_study_slot(monkeypatch):
    monkeypatch.setattr(
        study_intent.provider_adapter,
        "_load_external_credentials",
        lambda *a, **k: {"model": "m", "api_key": "k", "base_url": "https://x/v1/chat/completions"},
    )
    result = study_intent.extract_study_intent(
        "Does early lactate predict in-hospital mortality?",
        llm_provider="openai",
        external_llm_opt_in=True,
        ai_enabled=True,
        transport=lambda req, hdr: _llm_response(
            {
                "population": "all admissions",
                "exposure": "vasopressor dose",
                "outcome": "ICU length of stay",
                "outcome_type": "continuous",
                "time_window_hours": None,
                "comparator": None,
                "analysis_family": "prediction",
            }
        ),
    )

    values = _values(result)
    assert values["exposure"] == "lact"
    assert values["outcome"] == "death"


def test_llm_prompt_forbids_guessing(monkeypatch):
    """The instruction that stops substitution must actually be sent."""
    seen = {}

    monkeypatch.setattr(
        study_intent.provider_adapter,
        "_load_external_credentials",
        lambda *a, **k: {"model": "m", "api_key": "k", "base_url": "https://x/v1/chat/completions"},
    )

    def transport(request, headers):
        seen["request"] = request
        return _llm_response({s: None for s in study_intent.SLOTS})

    study_intent.extract_study_intent(
        "Does lactate predict mortality?",
        llm_provider="openai",
        external_llm_opt_in=True,
        ai_enabled=True,
        transport=transport,
    )
    system = seen["request"]["messages"][0]["content"]
    assert "never guess" in system
    assert "never substitute a more common study" in system


# ------------------------------------------------------------ input bounds ---
def test_empty_question_is_refused():
    with pytest.raises(study_intent.StudyIntentError):
        study_intent.deterministic_intent("   ")


def test_overlong_question_is_refused():
    with pytest.raises(study_intent.StudyIntentError):
        study_intent.deterministic_intent("x" * 5000)


# ------------------------------------------------------- frontend ownership ---
def test_intent_owner_file_is_wired_and_owns_the_contract_card():
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "js/screens-guided-intent.js" in index
    owner = (STATIC / "js/screens-guided-intent.js").read_text(encoding="utf-8")
    assert "EU_STUDY_INTENT" in owner
    assert "/api/copilot/study-intent" in owner
    # The contract card belongs to its owner file, not to the guided monolith.
    guided = (STATIC / "js/screens-guided.js").read_text(encoding="utf-8")
    assert "/api/copilot/study-intent" not in guided
