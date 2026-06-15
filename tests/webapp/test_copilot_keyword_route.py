"""Unit tests for the offline keyword router + session-title normalizer.

These cover the deterministic helpers added for the Research Copilot offline
path (no LLM needed): `_copilot_keyword_route` classifies a free-text study goal
into the route schema `_copilot_apply_llm_route` consumes, and
`_copilot_normalize_session_title` keeps the left-rail session list readable.

Both helpers are pure (no Streamlit session state), so they test standalone.
"""
from __future__ import annotations

from easyicu.webapp import llm_chat


# --------------------------------------------------------------------------- #
# _copilot_keyword_route
# --------------------------------------------------------------------------- #

def test_keyword_route_prediction_with_disease_and_concept():
    route = llm_chat._copilot_keyword_route(
        "Among Sepsis-3 patients, does early lactate predict in-hospital mortality",
        "en",
    )
    assert route is not None
    assert route["analysis_family"] == "prediction"
    # disease hint -> sepsis cohort label (same label the classic dropdown uses)
    sepsis_label = llm_chat._copilot_full_disease_options("en")["sepsis"]
    assert route.get("cohort", {}).get("label") == sepsis_label
    # concept hint -> lactate
    assert "lact" in route.get("suggested_concepts", [])
    assert route["study_frame"].startswith("Among Sepsis-3")


def test_keyword_route_cross_database_two_db_names():
    route = llm_chat._copilot_keyword_route(
        "Compare sepsis mortality across MIMIC-IV and eICU databases", "en"
    )
    assert route is not None
    assert route["analysis_family"] == "cross_database"


def test_keyword_route_cross_database_external_validation():
    route = llm_chat._copilot_keyword_route(
        "Externally validate a sepsis mortality model on eICU", "en"
    )
    assert route is not None
    assert route["analysis_family"] == "cross_database"


def test_keyword_route_quality_audit():
    route = llm_chat._copilot_keyword_route(
        "Audit data quality and missingness in the MIMIC-IV ICU cohort", "en"
    )
    assert route is not None
    assert route["analysis_family"] == "quality_audit"


def test_keyword_route_association():
    route = llm_chat._copilot_keyword_route(
        "Study the association between serum lactate and mortality in ICU patients",
        "en",
    )
    assert route is not None
    assert route["analysis_family"] == "association"


def test_keyword_route_single_db_prediction_not_crossdb():
    # one database name + "vs" between features must NOT trigger cross-database
    route = llm_chat._copilot_keyword_route(
        "predict mortality in MIMIC-IV using lactate vs sofa", "en"
    )
    assert route is not None
    assert route["analysis_family"] == "prediction"


def test_keyword_route_vague_returns_none():
    for vague in (
        "hello there how are you today",
        "tell me a joke about cats please",
        "what is the weather like tomorrow",
    ):
        assert llm_chat._copilot_keyword_route(vague, "en") is None


def test_keyword_route_too_short_returns_none():
    assert llm_chat._copilot_keyword_route("hi", "en") is None
    assert llm_chat._copilot_keyword_route("", "en") is None


def test_keyword_route_chinese_quality():
    route = llm_chat._copilot_keyword_route("审计 MIMIC-IV 数据质量与缺失情况", "zh")
    assert route is not None
    assert route["analysis_family"] == "quality_audit"


def test_copilot_usage_help_stays_deterministic():
    assert llm_chat._copilot_usage_help_requested("why Copilot?")
    assert llm_chat._copilot_usage_help_requested("how does this work?")
    assert not llm_chat._copilot_should_use_llm_route(
        "why Copilot?",
        usage_help_intent=True,
        step_by_step_intent=False,
        full_cohort_intent=False,
        cohort_step_intent=False,
        api_intent=False,
        path_help_intent=False,
        guided_choice_intent=False,
    )


# --------------------------------------------------------------------------- #
# _copilot_normalize_session_title
# --------------------------------------------------------------------------- #

def test_normalize_title_generic_label_falls_back_to_question():
    manifest = {
        "title": "New study / workspace",
        "study": {"question": "Sepsis mortality prediction"},
        "messages": [],
    }
    assert llm_chat._copilot_normalize_session_title(manifest, "en") == (
        "Sepsis mortality prediction"
    )


def test_normalize_title_empty_falls_back_to_untitled():
    manifest = {"title": "", "study": {}, "messages": []}
    assert llm_chat._copilot_normalize_session_title(manifest, "en") == (
        llm_chat._copilot_session_fallback_title("en")
    )


def test_normalize_title_real_title_unchanged():
    manifest = {"title": "AKI onset across MIMIC-IV", "study": {}, "messages": []}
    assert llm_chat._copilot_normalize_session_title(manifest, "en") == (
        "AKI onset across MIMIC-IV"
    )


def test_normalize_title_generic_uses_first_meaningful_user_message():
    manifest = {
        "title": "new study",
        "study": {},
        "messages": [
            {"role": "user", "content": "new study"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Lactate trajectory in septic shock"},
        ],
    }
    assert llm_chat._copilot_normalize_session_title(manifest, "en") == (
        "Lactate trajectory in septic shock"
    )
