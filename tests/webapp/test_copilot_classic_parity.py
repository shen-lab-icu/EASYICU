"""Parity: a cohort filtered through the Copilot path must be byte-identical to
one filtered through the classic Data Extraction path.

This is the regression guard for the divergence described in
``easyicu美化/copilot_接线施工计划.md``. Both paths must end up calling the SAME
``data_workflows.apply_cohort_filter`` against the SAME canonical
``st.session_state['cohort_filter']`` and produce identical filtered IDs.

Real-data, opt-in: skipped unless ``--run-real`` is passed and
``EASYICU_DATA_PATH`` points to an existing local database directory. Pick the
database with ``EASYICU_DB`` (default ``miiv``). Example::

    EASYICU_DATA_PATH=/Volumes/外置硬盘/databases/miiv EASYICU_DB=miiv \
        pytest tests/webapp/test_copilot_classic_parity.py --run-real -q
"""

from __future__ import annotations

import os

import pytest

# Copilot path needs the real Streamlit session_state; skip cleanly if absent.
st = pytest.importorskip("streamlit")

import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402

from easyicu.webapp import cohort_config, cohort_filters  # noqa: E402
from easyicu.webapp import copilot_engine as ce  # noqa: E402
from easyicu.webapp import data_workflows  # noqa: E402

pytestmark = pytest.mark.needs_real_data


def _app_context() -> dict:
    """Mirror app.py globals() — the classic engines inject `st`/`pd`/helpers
    from here via ``_install_app_context``."""
    ctx: dict = {}
    ctx.update(vars(cohort_config))
    ctx.update(vars(cohort_filters))
    ctx["st"] = st
    ctx["pd"] = pd
    ctx["Path"] = Path
    return ctx


def _canonical_cohort_filter() -> dict:
    """A modest, deterministic filter exercising several criteria."""
    return {
        "age_min": 18,
        "age_max": None,
        "first_icu_stay": True,
        "los_min": None,
        "los_max": None,
        "gender": None,
        "survived": None,
        "has_sepsis": None,
        "disease_cohort": "none",
        "icd_query": "",
        "icd_include_query": "",
        "icd_exclude_query": "",
        "icd_mode": "include",
    }


def _reset_session_state(data_path: str, database: str, cohort_filter: dict) -> None:
    st.session_state.clear()
    st.session_state["data_path"] = data_path
    st.session_state["database"] = database
    st.session_state["use_mock_data"] = False
    st.session_state["cohort_enabled"] = True
    st.session_state["cohort_filter"] = dict(cohort_filter)


def test_copilot_cohort_matches_classic_cohort():
    data_path = os.environ["EASYICU_DATA_PATH"]
    database = os.environ.get("EASYICU_DB", "miiv")
    cohort_filter = _canonical_cohort_filter()

    # --- classic path: call the engine directly, as the classic view does ---
    _reset_session_state(data_path, database, cohort_filter)
    classic = data_workflows.apply_cohort_filter(data_path, database, app_context=_app_context())
    assert classic is not None, "classic cohort filter returned None (check data/filter)"

    # --- copilot path: same canonical state, routed through copilot_engine ---
    _reset_session_state(data_path, database, cohort_filter)
    study = {"depth": "extract"}
    result = ce.run_copilot_step("cohort", study, st.session_state, app_context=_app_context())
    assert result["status"] == "ok"

    # --- parity: identical filtered IDs and counts ---
    assert st.session_state["_cohort_filtered_ids"] == classic["filtered_ids"]
    assert st.session_state["filtered_patient_count"] == classic["total_after"]
    assert st.session_state["_cohort_stats"]["total_before"] == classic["total_before"]
    assert st.session_state["_cohort_stats"]["total_after"] == classic["total_after"]


def test_copilot_cohort_counterexample_nondefault_filter():
    """Anti-divergence: a non-default filter must still match the classic path."""
    data_path = os.environ["EASYICU_DATA_PATH"]
    database = os.environ.get("EASYICU_DB", "miiv")
    cohort_filter = _canonical_cohort_filter()
    cohort_filter["age_min"] = 65  # deliberately non-default

    _reset_session_state(data_path, database, cohort_filter)
    classic = data_workflows.apply_cohort_filter(data_path, database, app_context=_app_context())
    assert classic is not None

    _reset_session_state(data_path, database, cohort_filter)
    ce.run_copilot_step("cohort", {"depth": "extract"}, st.session_state, app_context=_app_context())

    assert st.session_state["_cohort_filtered_ids"] == classic["filtered_ids"]
    assert st.session_state["filtered_patient_count"] == classic["total_after"]
