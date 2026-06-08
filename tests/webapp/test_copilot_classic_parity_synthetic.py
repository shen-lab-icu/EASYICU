"""Self-contained Copilot↔classic cohort parity (no external data needed).

Builds a tiny synthetic MIMIC-IV-layout dataset in a tmp dir, then asserts the
Copilot cohort path produces byte-identical filtered IDs to the classic
``apply_cohort_filter`` path. Unlike ``test_copilot_classic_parity.py`` (which
points at a real local database via ``--run-real``), this runs in normal CI so
the divergence guard is always on. Skips cleanly if Streamlit/pyarrow aren't
installed.
"""

from __future__ import annotations

import pytest

st = pytest.importorskip("streamlit")
pytest.importorskip("pyarrow")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from easyicu.webapp import cohort_config, cohort_filters  # noqa: E402
from easyicu.webapp import copilot_engine as ce  # noqa: E402
from easyicu.webapp import data_workflows  # noqa: E402


def _app_context() -> dict:
    ctx: dict = {}
    ctx.update(vars(cohort_config))
    ctx.update(vars(cohort_filters))
    ctx.update({"st": st, "pd": pd, "Path": Path})
    return ctx


@pytest.fixture(scope="module")
def synth_miiv(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("synth_miiv")
    rng = np.random.default_rng(7)
    n = 300
    subject_ids = rng.integers(10000, 10180, size=n)  # repeats -> non-first stays
    hadm_ids = np.arange(50000, 50000 + n)
    base = pd.Timestamp("2150-01-01")
    intime = (base + pd.to_timedelta(rng.integers(0, 3650, size=n), unit="D")
              + pd.to_timedelta(rng.integers(0, 24, size=n), unit="h"))
    los_days = np.round(rng.uniform(0.2, 20.0, size=n), 3)
    outtime = intime + pd.to_timedelta(los_days, unit="D")
    admittime = intime - pd.to_timedelta(rng.integers(0, 48, size=n), unit="h")
    pd.DataFrame({
        "stay_id": np.arange(1, n + 1), "subject_id": subject_ids, "hadm_id": hadm_ids,
        "intime": intime, "outtime": outtime, "los": los_days,
    }).to_parquet(out / "icustays.parquet")
    usub = np.unique(subject_ids)
    pd.DataFrame({
        "subject_id": usub, "anchor_age": rng.integers(18, 95, size=len(usub)),
        "anchor_year": 2150, "gender": rng.choice(["M", "F"], size=len(usub)),
    }).to_parquet(out / "patients.parquet")
    expire = (rng.random(n) < 0.2).astype(int)
    deathtime = pd.Series(pd.NaT, index=range(n))
    deathtime[expire == 1] = outtime[expire == 1]
    pd.DataFrame({
        "hadm_id": hadm_ids, "subject_id": subject_ids, "admittime": admittime,
        "hospital_expire_flag": expire, "deathtime": deathtime,
    }).to_parquet(out / "admissions.parquet")
    return str(out)


def _reset(data_path: str, cohort_filter: dict) -> None:
    st.session_state.clear()
    st.session_state.update({
        "data_path": data_path, "database": "miiv", "use_mock_data": False,
        "cohort_enabled": True, "cohort_filter": dict(cohort_filter),
    })


_FILTERS = [
    pytest.param({"age_min": 18, "first_icu_stay": True}, id="age+first"),
    pytest.param({"age_min": 65, "gender": "M", "survived": True}, id="age+sex+survived"),
    pytest.param({"age_min": 65, "first_icu_stay": True, "los_min": 24,
                  "gender": "M", "survived": True}, id="five-stacked"),
    pytest.param({"los_min": 48}, id="los-only"),
]


@pytest.mark.parametrize("partial", _FILTERS)
def test_cohort_parity(synth_miiv, partial):
    cohort_filter = {
        "age_min": None, "age_max": None, "first_icu_stay": None, "los_min": None,
        "los_max": None, "gender": None, "survived": None, "has_sepsis": None,
        "disease_cohort": "none", "icd_query": "", "icd_include_query": "",
        "icd_exclude_query": "", "icd_mode": "include",
    }
    cohort_filter.update(partial)

    _reset(synth_miiv, cohort_filter)
    classic = data_workflows.apply_cohort_filter(synth_miiv, "miiv", app_context=_app_context())
    assert classic is not None

    _reset(synth_miiv, cohort_filter)
    result = ce.run_copilot_step("cohort", {"depth": "extract"}, st.session_state,
                                 app_context=_app_context())
    assert result["status"] == "ok"

    # Parity: identical filtered IDs and counts between the two paths.
    assert sorted(st.session_state["_cohort_filtered_ids"]) == sorted(classic["filtered_ids"])
    assert st.session_state["filtered_patient_count"] == classic["total_after"]
    assert st.session_state["_cohort_stats"]["total_before"] == classic["total_before"]
    # Sanity: the filter actually removed someone (not a vacuous pass).
    assert classic["total_after"] < classic["total_before"]
