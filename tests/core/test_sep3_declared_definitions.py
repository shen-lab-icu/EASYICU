"""Declared Sepsis-3 definition options (S3-default, S3-mimiccode, S3-yaib).

Each option is a validated keyword whose default reproduces the canonical
ricu result; the expected onsets below are computed by hand on small
synthetic stays (time = hours since ICU admission).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_sep3,
    _callback_sep3_sofa2,
    _callback_susp_inf,
)
from easyicu.scores.sepsis_declared import continuous_antibiotic_anchors
from easyicu.table import ICUTable

MIMICCODE = {"si_mode": "and", "baseline_sofa": "zero", "si_window": "any", "onset_scope": "icu"}


def _ctx(name, kwargs=None, database="miiv"):
    return ConceptCallbackContext(
        concept_name=name, target=None, interval=pd.Timedelta(hours=1), resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name=database)),
        patient_ids=None, kwargs=dict(kwargs or {}),
    )


def _ts(rows, value, index="charttime"):
    frame = pd.DataFrame(rows, columns=["stay_id", index, value])
    return ICUTable(frame, id_columns=["stay_id"], index_column=index, value_column=value)


def _sofa_rows(stay, start, scores):
    return [(stay, float(start + h), float(s)) for h, s in enumerate(scores)]


def _sep3_tables(sofa_rows, si_rows, los=None):
    tables = {
        "sofa": _ts(sofa_rows, "sofa"),
        "susp_inf": _ts([(s, float(t), True) for s, t in si_rows], "susp_inf"),
    }
    if los is not None:
        tables["los_icu"] = ICUTable(
            pd.DataFrame({"stay_id": list(los), "los_icu": list(los.values())}),
            id_columns=["stay_id"], value_column="los_icu",
        )
    return tables


def _onsets(table, value="sep3"):
    data = table.data
    return {int(s): float(t) for s, t in zip(data["stay_id"], data[table.index_column])} if len(data) else {}


# --------------------------------------------------------------------------- sep3

def _baseline_cohort():
    # 1: SOFA 3 throughout, SI at 10 -> no rise from the window minimum;
    #    absolute SOFA >= 2 from the first window hour (0).
    # 2: SOFA 0 for hours 0-9, then 2; SI at 12 -> both definitions: onset 10.
    # 3: SOFA 1 throughout, SI at 5 -> no onset under either definition.
    sofa = (
        _sofa_rows(1, 0, [3] * 31)
        + _sofa_rows(2, 0, [0] * 10 + [2] * 11)
        + _sofa_rows(3, 0, [1] * 20)
    )
    return _sep3_tables(sofa, [(1, 10), (2, 12), (3, 5)])


def test_sep3_defaults_are_unchanged_and_explicit_defaults_are_identical():
    tables = _baseline_cohort()
    default = _callback_sep3(tables, _ctx("sep3"))
    explicit = _callback_sep3(
        tables, _ctx("sep3", {"baseline_sofa": "window", "onset_scope": "all"})
    )
    pd.testing.assert_frame_equal(default.data, explicit.data)
    assert _onsets(default) == {2: 10.0}


def test_baseline_sofa_zero_counts_absolute_sofa_in_the_window():
    result = _callback_sep3(_baseline_cohort(), _ctx("sep3", {"baseline_sofa": "zero"}))
    assert _onsets(result) == {1: 0.0, 2: 10.0}


def _scope_cohort():
    # 4: SOFA 3 from -10 to 30, SI at -3 (before ICU) and 6, ICU stay 24 h.
    # 5: like 4 but no ICU length of stay.
    # 6: SOFA 3 from 0 to 40, SI only at 30, after the 24 h ICU stay.
    # 7: SOFA 0 at -5..-1 then 2 from 0; SI 2; stay 48 h. The baseline still
    #    uses the pre-admission hours inside the window: onset 0 either way.
    # 8: SOFA 0 at -6..-4, 2 at -3..-1, 3 from 0; SI 0; stay 48 h.
    #    All: onset -3 (first rise >= 2); ICU: onset 0 (delta 3).
    sofa = (
        _sofa_rows(4, -10, [3] * 41)
        + _sofa_rows(5, -10, [3] * 41)
        + _sofa_rows(6, 0, [3] * 41)
        + _sofa_rows(7, -5, [0] * 5 + [2] * 11)
        + _sofa_rows(8, -6, [0] * 3 + [2] * 3 + [3] * 10)
    )
    si = [(4, -3), (4, 6), (5, -3), (5, 6), (6, 30), (7, 2), (8, 0)]
    return _sep3_tables(sofa, si, los={4: 1.0, 6: 1.0, 7: 2.0, 8: 2.0})


def test_onset_scope_icu_counts_only_in_stay_suspicion_and_onsets():
    tables = _scope_cohort()
    everywhere = _callback_sep3(tables, _ctx("sep3", {"si_window": "any"}))
    assert _onsets(everywhere) == {7: 0.0, 8: -3.0}
    icu = _callback_sep3(tables, _ctx("sep3", {"si_window": "any", "onset_scope": "icu"}))
    assert _onsets(icu) == {7: 0.0, 8: 0.0}


def test_mimiccode_definition_absolute_sofa_any_window_icu_only():
    tables = _scope_cohort()
    everywhere = _callback_sep3(
        tables, _ctx("sep3", {**MIMICCODE, "onset_scope": "all"})
    )
    # SI -3 opens [-51, 21]: first SOFA >= 2 at -10; stay 6 SI 30 -> SOFA at 0.
    assert _onsets(everywhere) == {4: -10.0, 5: -10.0, 6: 0.0, 7: 0.0, 8: -3.0}
    mimiccode = _callback_sep3(tables, _ctx("sep3", MIMICCODE))
    # SI -3 is pre-ICU, SI 30 post-discharge; stay 5 has no ICU stay length.
    assert _onsets(mimiccode) == {4: 0.0, 7: 0.0, 8: 0.0}


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"baseline_sofa": "measured"}, "allowed: ['window', 'zero']"),
        ({"onset_scope": "hospital"}, "allowed: ['all', 'icu']"),
        ({"baseline_sofa": "zero", "delta_fun": "delta_min"}, "cannot be combined"),
    ],
)
def test_sep3_rejects_undeclared_values(kwargs, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        _callback_sep3(_baseline_cohort(), _ctx("sep3", kwargs))


def test_onset_scope_icu_fails_closed_without_icu_length_of_stay():
    with pytest.raises(ValueError, match="los_icu"):
        _callback_sep3(_baseline_cohort(), _ctx("sep3", {"onset_scope": "icu"}))


def test_onset_scope_icu_refuses_a_clock_time_axis():
    tables = _baseline_cohort()
    for name in ("sofa", "susp_inf"):
        frame = tables[name].data.copy()
        frame["charttime"] = pd.Timestamp("2100-01-01") + pd.to_timedelta(frame["charttime"], unit="h")
        tables[name] = ICUTable(frame, ["stay_id"], "charttime", name)
    tables["los_icu"] = ICUTable(
        pd.DataFrame({"stay_id": [1, 2, 3], "los_icu": [5.0] * 3}), ["stay_id"], value_column="los_icu"
    )
    with pytest.raises(ValueError, match="hours since ICU admission"):
        _callback_sep3(tables, _ctx("sep3", {"onset_scope": "icu"}))


def test_sep3_sofa2_refuses_options_declared_for_sep3_only():
    tables = {"sofa2": _ts([(1, 0.0, 3.0)], "sofa2"), "susp_inf": _ts([(1, 0.0, True)], "susp_inf")}
    with pytest.raises(ValueError, match="sep3 only"):
        _callback_sep3_sofa2(tables, _ctx("sep3_sofa2", {"baseline_sofa": "zero"}))
    default = _callback_sep3_sofa2(tables, _ctx("sep3_sofa2"))
    explicit = _callback_sep3_sofa2(
        tables, _ctx("sep3_sofa2", {"baseline_sofa": "window", "onset_scope": "all"})
    )
    pd.testing.assert_frame_equal(default.data, explicit.data)


# ------------------------------------------------------------------- susp_inf

def _susp_tables(abx_rows, samp_rows, death_rows=()):
    return {
        "abx": _ts([(s, float(t), True) for s, t in abx_rows], "abx"),
        "samp": _ts([(s, float(t), True) for s, t in samp_rows], "samp"),
        "death": _ts([(s, float(t), True) for s, t in death_rows], "death"),
    }


def _si_times(table):
    data = table.data
    out = {}
    for stay, time in zip(data["stay_id"], data[table.index_column]):
        out.setdefault(int(stay), set()).add(float(time))
    return out


def _yaib_cohort():
    # 1: abx 0, 24, 48 -> continuous from 0 only; culture at 1.
    # 2: abx 0, 24 alive -> last gap to 72 is 48 h: never continuous.
    # 3: abx 0, 24, death 40 -> window ends at death: anchors 0 and 24.
    # 4: abx 0, 30, 48 -> a 30 h gap: never continuous.
    return _susp_tables(
        abx_rows=[(1, 0), (1, 24), (1, 48), (2, 0), (2, 24), (3, 0), (3, 24), (4, 0), (4, 30), (4, 48)],
        samp_rows=[(1, 1), (2, 1), (3, 1), (4, 1)],
        death_rows=[(3, 40)],
    )


def test_susp_inf_defaults_are_unchanged_and_explicit_defaults_are_identical():
    tables = _yaib_cohort()
    default = _callback_susp_inf(tables, _ctx("susp_inf", {"si_mode": "and"}))
    explicit = _callback_susp_inf(
        tables, _ctx("susp_inf", {"si_mode": "and", "abx_min_days": None})
    )
    pd.testing.assert_frame_equal(default.data, explicit.data)
    # abx -> culture within 24 h stamps the abx time (0); culture -> abx within
    # 72 h stamps the culture time (1).
    assert _si_times(default) == {1: {0.0, 1.0}, 2: {0.0, 1.0}, 3: {0.0, 1.0}, 4: {0.0, 1.0}}


def test_abx_min_days_keeps_only_continuous_antibiotic_courses():
    result = _callback_susp_inf(
        _yaib_cohort(), _ctx("susp_inf", {"si_mode": "and", "abx_min_days": 3})
    )
    assert _si_times(result) == {1: {0.0}, 3: {0.0, 1.0}}


def test_abx_min_days_with_antibiotics_alone_uses_rolling_24h_blocks():
    # 5: abx every 12 h from 0 to 72 -> anchors 0, 12, 24 (36 would need abx
    #    within 24 h of 108). 6: death 30, abx at 50 is after death and ignored.
    tables = _susp_tables(
        abx_rows=[(5, t) for t in range(0, 73, 12)] + [(6, 0), (6, 24), (6, 50)],
        samp_rows=[],
        death_rows=[(6, 30)],
    )
    result = _callback_susp_inf(tables, _ctx("susp_inf", {"si_mode": "abx", "abx_min_days": 3}))
    assert _si_times(result) == {5: {0.0, 12.0, 24.0}, 6: {0.0, 24.0}}
    two_days = _callback_susp_inf(
        tables, _ctx("susp_inf", {"si_mode": "abx", "abx_min_days": 2, "abx_max_gap": 12})
    )
    # 6: anchor 0 fails (24 h gap > 12 h); anchor 24 reaches death 6 h later.
    assert _si_times(two_days) == {5: {0.0, 12.0, 24.0, 36.0}, 6: {24.0}}


def test_continuous_antibiotic_anchors_ignore_false_rows_and_accept_clock_time():
    base = pd.Timestamp("2100-01-01")
    abx = pd.DataFrame({
        "stay_id": [1, 1, 1, 1],
        "charttime": [base + pd.Timedelta(hours=h) for h in (0, 24, 48, 60)],
        "abx": [True, True, True, False],
    })
    kept = continuous_antibiotic_anchors(abx, ["stay_id"], "charttime", min_days=3)
    assert kept["charttime"].tolist() == [base]
    falsy_gap = abx.assign(abx=[True, False, True, True])
    assert continuous_antibiotic_anchors(falsy_gap, ["stay_id"], "charttime", min_days=3).empty
    assert np.array_equal(kept.index, [0])


@pytest.mark.parametrize(
    "kwargs,database,message",
    [
        ({"abx_min_days": 0}, "miiv", "positive integer"),
        ({"abx_min_days": 2.5}, "miiv", "positive integer"),
        ({"abx_min_days": "three"}, "miiv", "positive integer"),
        ({"abx_min_days": True}, "miiv", "positive integer"),
        ({"abx_min_days": 3, "si_mode": "samp"}, "miiv", "si_mode resolved to 'samp'"),
        ({"abx_min_days": 3}, "eicu", "si_mode resolved to 'icd_abx'"),
        ({"abx_max_gap": 12}, "miiv", "only together with abx_min_days"),
        ({"abx_min_days": 3, "abx_max_gap": "bogus"}, "miiv", "positive duration"),
    ],
)
def test_susp_inf_rejects_undeclared_continuity_options(kwargs, database, message):
    with pytest.raises(ValueError, match=message):
        _callback_susp_inf(_yaib_cohort(), _ctx("susp_inf", kwargs, database))


def test_abx_min_days_fails_closed_without_death_information():
    tables = _yaib_cohort()
    del tables["death"]
    with pytest.raises(ValueError, match="'death' concept"):
        _callback_susp_inf(tables, _ctx("susp_inf", {"si_mode": "and", "abx_min_days": 3}))


def test_abx_min_days_reads_the_death_time_companion_of_an_id_level_death_table():
    # The loaded ``death`` concept is one row per stay with a ``death_time``
    # companion (hours from ICU admission); survivors carry False and no time.
    tables = _yaib_cohort()
    tables["death"] = ICUTable(
        pd.DataFrame({"stay_id": [2, 3], "death": [False, True], "death_time": [np.nan, 40.0]}),
        id_columns=["stay_id"], value_column="death",
    )
    result = _callback_susp_inf(tables, _ctx("susp_inf", {"si_mode": "and", "abx_min_days": 3}))
    assert _si_times(result) == {1: {0.0}, 3: {0.0, 1.0}}
