"""Reference-profile integration must preserve HiRID's rate-source semantics."""
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.scores.aki_profiles import (
    apply_aki_profile,
    apply_reference_aki,
    build_renal_aki_bundle,
    load_renal_aki_bundle,
)
from easyicu.scores.kdigo_aki import kdigo_uo

REFERENCE = "MIT_LCP_KDIGO_REFERENCE_PORT_V1"


def _inputs(spacing=4, rate=80.0, bin_hours=1.0):
    times = list(range(0, 25, spacing))
    return {
        "urine_df": pd.DataFrame({
            "stay_id": [1] * len(times), "charttime": times,
            "urine": [rate * bin_hours] * len(times),
        }),
        "weight_df": pd.DataFrame({"stay_id": [1], "weight": [80.0]}),
        "id_col": "stay_id", "time_col": "charttime", "time_unit": "hours",
    }


@pytest.mark.parametrize("entrypoint", ["reference", "profile", "bundle"])
@pytest.mark.parametrize("spacing", [1, 2, 4, 6])
@pytest.mark.parametrize("bin_hours", [0.5, 1.0, 2.0])
def test_reference_constant_rate_does_not_become_aki_when_charting_is_sparse(
    entrypoint, spacing, bin_hours,
):
    kwargs = dict(_inputs(spacing, bin_hours=bin_hours),
                  urine_source_is_rate=True,
                  interval=pd.Timedelta(hours=bin_hours))
    if entrypoint == "reference":
        result = apply_reference_aki(**kwargs)
    elif entrypoint == "profile":
        result = apply_aki_profile(REFERENCE, **kwargs)
    else:
        result = build_renal_aki_bundle("hirid", **kwargs)
    observed = result.loc[result.charttime.isin([12, 24])]
    assert observed.aki_stage_uo_reference.tolist() == [0, 0]
    assert observed.aki_stage_reference.tolist() == [0, 0]
    assert observed.aki_severe_reference.tolist() == [False, False]


def test_hirid_bundle_infers_rate_source_when_flag_is_omitted():
    result = build_renal_aki_bundle("HiRID", **_inputs())
    assert result.loc[result.charttime.eq(24), "aki_stage_reference"].item() == 0
    assert result.aki_source_native_status.eq(
        "not_evaluable_required_source_missing"
    ).all()


@pytest.mark.parametrize("database", ["miiv", "mimic", "aumc", "eicu", "sic"])
def test_other_databases_keep_volume_event_semantics(database):
    # These values mean 80 mL per event, not 80 mL/h: the old result is valid.
    kwargs = _inputs()
    expected = kdigo_uo(**kwargs, source_is_rate=False)
    result = build_renal_aki_bundle(database, **kwargs)
    pd.testing.assert_series_equal(
        result.aki_stage_uo_reference.reset_index(drop=True),
        expected.aki_stage_uo.reset_index(drop=True), check_names=False,
    )
    assert result.loc[result.charttime.eq(12), "aki_stage_reference"].item() == 2
    assert result.loc[result.charttime.eq(24), "aki_stage_reference"].item() == 3


def test_reference_rate_path_preserves_real_oliguria_and_missing_components():
    result = build_renal_aki_bundle("hirid", **_inputs(rate=10.0),
                                    urine_source_is_rate=True)
    assert result.loc[result.charttime.eq(12), "aki_stage_reference"].item() == 2
    assert result.loc[result.charttime.eq(24), "aki_stage_reference"].item() == 3
    initial = result.loc[result.charttime.eq(0)].iloc[0]
    assert pd.isna(initial.aki_stage_uo_reference)
    # Keep public-reference combination policy, not a new strict phenotype.
    assert initial.aki_stage_reference == 0


def test_reference_rate_path_does_not_use_future_measurements():
    kwargs = _inputs()
    full = apply_reference_aki(**kwargs, urine_source_is_rate=True)
    earlier = dict(kwargs, urine_df=kwargs["urine_df"].query("charttime <= 12"))
    prefix = apply_reference_aki(**earlier, urine_source_is_rate=True)
    pd.testing.assert_frame_equal(full.loc[full.charttime.le(12)].reset_index(drop=True), prefix)


def test_reference_rate_path_preserves_creatinine_and_rrt_components():
    kwargs = dict(_inputs(),
        crea_df=pd.DataFrame({"stay_id": [1, 1], "charttime": [0, 12], "crea": [1.0, 2.1]}),
        rrt_df=pd.DataFrame({"stay_id": [1], "charttime": [24], "rrt": [True]}))
    result = build_renal_aki_bundle("hirid", **kwargs, urine_source_is_rate=True)
    assert result.loc[result.charttime.eq(12), "aki_stage_creat_reference"].item() == 2
    assert result.loc[result.charttime.eq(24), "aki_stage_rrt_reference"].item() == 3
    assert result.loc[result.charttime.eq(24), "aki_stage_reference"].item() == 3


def test_reference_rate_callback_propagates_interval():
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_kdigo_aki
    from easyicu.table import ICUTable
    kwargs = _inputs(bin_hours=0.5)
    tables = {
        "kdigo_urine_input": ICUTable(
            data=kwargs["urine_df"].rename(columns={"urine": "kdigo_urine_input"}),
            id_columns=["stay_id"], index_column="charttime", value_column="kdigo_urine_input",
        ),
        "weight": ICUTable(data=kwargs["weight_df"], id_columns=["stay_id"],
                           index_column=None, value_column="weight"),
    }
    ctx = ConceptCallbackContext(
        concept_name="kdigo_aki", target=None, interval=pd.Timedelta(minutes=30),
        resolver=object(), data_source=SimpleNamespace(config=SimpleNamespace(name="hirid")),
        patient_ids=None,
    )
    result = _callback_kdigo_aki(tables, ctx).data
    assert result.loc[result.charttime.eq(24), "aki_stage_reference"].item() == 0
    assert result.loc[result.charttime.eq(24), "uo_rt_12hr"].item() == pytest.approx(1.0)


def test_reference_rate_loader_preserves_hirid_semantics(monkeypatch):
    inputs = _inputs()
    components = {
        "kdigo_urine_input": inputs["urine_df"].rename(columns={"urine": "kdigo_urine_input"}),
        "weight": inputs["weight_df"],
        "kdigo_creatinine_input": pd.DataFrame(columns=["stay_id", "charttime", "kdigo_creatinine_input"]),
        "acute_rrt_input": pd.DataFrame(columns=["stay_id", "charttime", "acute_rrt_input"]),
        "crrt_mode_input": pd.DataFrame(columns=["stay_id", "charttime", "crrt_mode_input"]),
    }
    monkeypatch.setattr("easyicu.api.load_concepts", lambda concepts, **kw: components[concepts[0]])
    result = load_renal_aki_bundle("hirid", verbose=False)
    assert result.loc[result.charttime.eq(24), "aki_stage_reference"].item() == 0
