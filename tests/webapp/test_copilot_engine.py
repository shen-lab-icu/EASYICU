"""Unit tests for the Copilot ↔ classic-engine wiring (copilot_engine).

These are import-light: they exercise the depth axis and the step dispatcher
with *fake* classic engines, so they run without Streamlit or real data. They
prove the routing contract; byte-for-byte parity against the real classic
functions is covered by ``test_copilot_classic_parity.py`` (real-data, opt-in).
"""

from __future__ import annotations

import pytest

from easyicu.webapp import copilot_engine as ce


# --------------------------------------------------------------------------- #
# Depth axis
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "depth, goal",
    [("extract", "extract"), ("review", "review"), ("full", "draft"),
     ("", "draft"), (None, "draft"), ("bogus", "draft")],
)
def test_goal_step(depth, goal):
    assert ce.copilot_goal_step(depth) == goal


def test_bump_depth_ladder():
    assert ce.copilot_bump_depth("extract") == "review"
    assert ce.copilot_bump_depth("review") == "full"
    assert ce.copilot_bump_depth("full") == "full"  # capped
    assert ce.copilot_bump_depth("bogus") == "full"  # normalized then capped


def test_beyond_goal_gating():
    # depth=extract: anything past 'extract' (idx 4) is beyond.
    assert ce.is_step_beyond_goal("extract", "review") is True
    assert ce.is_step_beyond_goal("extract", "draft") is True
    assert ce.is_step_beyond_goal("extract", "cohort") is False
    assert ce.is_step_beyond_goal("extract", "extract") is False
    # depth=full: draft is the goal, nothing is beyond it.
    assert ce.is_step_beyond_goal("full", "draft") is False  # draft == goal
    assert ce.is_step_beyond_goal("full", "analysis") is False


def test_next_step_capped_stops_at_goal():
    # extract depth stops at 'extract'.
    assert ce.next_step_capped("extract", "concepts") == "extract"
    assert ce.next_step_capped("extract", "extract") == "extract"  # never past goal
    # full depth advances normally.
    assert ce.next_step_capped("full", "concepts") == "extract"
    assert ce.next_step_capped("full", "analysis") == "draft"


def test_clamp_step_to_goal():
    assert ce.clamp_step_to_goal("extract", "draft") == "extract"
    assert ce.clamp_step_to_goal("review", "draft") == "review"
    assert ce.clamp_step_to_goal("full", "review") == "review"


# --------------------------------------------------------------------------- #
# Dispatcher routing with fake engines
# --------------------------------------------------------------------------- #

class _Spy:
    """Records calls so tests can assert the classic function was reached."""

    def __init__(self, return_value=None):
        self.calls: list[tuple] = []
        self.return_value = return_value

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value


def _engines(**overrides) -> ce.CopilotEngines:
    base = dict(
        apply_cohort_filter=_Spy(),
        post_filter_cohort_data=_Spy(),
        positive_patient_ids=_Spy(),
        check_data_status=_Spy({"ready": True}),
        load_data=_Spy({"vitals": "df"}),
        load_data_for_preview=_Spy({"vitals": "preview"}),
        execute_export=_Spy(),
        load_from_exported=_Spy(),
        cohort_feature_counts=_Spy({"modules": 3, "concepts": 9}),
    )
    base.update(overrides)
    return ce.CopilotEngines(**base)


def _real_state(**extra):
    state = {"data_path": "/data/miiv", "database": "miiv", "use_mock_data": False}
    state.update(extra)
    return state


def test_unknown_step_is_noop():
    res = ce.run_copilot_step("question", {}, {}, engines=_engines())
    assert res["status"] == "noop"


def test_cohort_runs_classic_apply_filter_and_stores_results():
    eng = _engines(apply_cohort_filter=_Spy({
        "id_col": "stay_id",
        "filtered_ids": [1, 2, 3],
        "total_before": 100,
        "total_after": 3,
    }))
    state = _real_state()
    res = ce.run_copilot_step("cohort", {}, state, app_context={"x": 1}, engines=eng)
    assert res["status"] == "ok"
    assert res["total_after"] == 3
    # The SAME classic engine was called...
    assert eng.apply_cohort_filter.calls, "apply_cohort_filter must be invoked"
    args, kwargs = eng.apply_cohort_filter.calls[0]
    assert args[0] == "/data/miiv" and args[1] == "miiv"
    assert kwargs["app_context"] == {"x": 1}
    # ...and its result is mirrored into the canonical state keys.
    assert state["_cohort_filtered_ids"] == [1, 2, 3]
    assert state["filtered_patient_count"] == 3
    assert state["_cohort_stats"]["total_before"] == 100


def test_cohort_no_active_filter_clears_state():
    eng = _engines(apply_cohort_filter=_Spy(None))
    state = _real_state(_cohort_filtered_ids=[9])
    res = ce.run_copilot_step("cohort", {}, state, engines=eng)
    assert res["status"] == "no_active_filter"
    assert state["_cohort_stats"] is None
    assert state["_cohort_filtered_ids"] is None


def test_cohort_skipped_without_real_data():
    eng = _engines()
    for state in ({"database": "mock"}, {"use_mock_data": True, "data_path": "/d", "database": "miiv"}, {}):
        res = ce.run_copilot_step("cohort", {}, state, engines=eng)
        assert res["status"] == "no_real_data"
    assert not eng.apply_cohort_filter.calls  # classic engine never touched on demo


def test_data_step_records_status():
    eng = _engines()
    state = _real_state()
    res = ce.run_copilot_step("data", {}, state, engines=eng)
    assert res["status"] == "ok"
    assert state["_data_status"] == {"ready": True}


def test_extract_preview_vs_full():
    eng = _engines()
    state = _real_state()
    ce.run_copilot_step("extract", {"patient_n": 25}, state, engines=eng, preview=True)
    assert eng.load_data_for_preview.calls and eng.load_data_for_preview.calls[0][0][0] == 25
    assert state["_extraction_done"] is True
    eng2 = _engines()
    state2 = _real_state()
    ce.run_copilot_step("extract", {}, state2, engines=eng2, preview=False)
    assert eng2.load_data.calls and not eng2.load_data_for_preview.calls


def test_export_runs_classic_export():
    eng = _engines()
    res = ce.run_copilot_step("export", {}, _real_state(), app_context={"c": 2}, engines=eng)
    assert res["status"] == "ok"
    assert eng.execute_export.calls[0][0][0] == {"c": 2}


def test_concepts_uses_shared_feature_counts():
    eng = _engines()
    state = _real_state()
    res = ce.run_copilot_step("concepts", {}, state, engines=eng)
    assert res["counts"] == {"modules": 3, "concepts": 9}
    assert state["_copilot_feature_counts"] == {"modules": 3, "concepts": 9}


def test_run_study_up_to_goal_respects_depth():
    # depth=extract -> runs data, cohort, concepts, extract; never review/export.
    eng = _engines()
    study = {"depth": "extract", "patient_n": 10}
    state = _real_state()
    results = ce.run_study_up_to_goal(study, state, engines=eng)
    steps = [r["step"] for r in results]
    assert steps == ["data", "cohort", "concepts", "extract"]
    assert "review" not in steps and "export" not in steps


def test_run_study_up_to_goal_full_depth_includes_review():
    eng = _engines()
    study = {"depth": "full"}
    results = ce.run_study_up_to_goal(study, _real_state(), engines=eng)
    steps = [r["step"] for r in results]
    assert "review" in steps  # full depth reaches review handler
