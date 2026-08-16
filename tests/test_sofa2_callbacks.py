import numpy as np
import pandas as pd
import pytest

from easyicu.callbacks import sofa2_cardio, sofa2_cns, sofa2_renal, sofa2_resp
from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_sofa2_score,
    _callback_sofa_component,
)
from easyicu.scores.sofa2 import sofa2_cardio as standalone_sofa2_cardio
from easyicu.scores.sofa2 import sofa2_cns as standalone_sofa2_cns
from easyicu.scores.sofa2 import sofa2_cns_ascertainment
from easyicu.scores.sofa2 import sofa2_cns_delirium_tx_ascertainment
from easyicu.scores.sofa2 import sofa2_cns_proxy_sensitivity
from easyicu.scores.sofa2 import sofa2_renal as standalone_sofa2_renal
from easyicu.scores.sofa2 import sofa2_resp as standalone_sofa2_resp
from easyicu.table import ICUTable


def test_sofa2_resp_uses_fractional_fio2_for_safi_fallback():
    score = sofa2_resp(
        pd.Series([None, None]),
        spo2=pd.Series([95, 97]),
        fio2=pd.Series([50, 40]),
        adv_resp=pd.Series([True, True]),
        oxygenation_sustained_1h=pd.Series([True, True]),
    )

    assert score.tolist() == [3, 2]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("scorer", [sofa2_resp, standalone_sofa2_resp])
def test_sofa2_resp_unknown_persistence_is_not_treated_as_transient(scorer):
    unknown = scorer(pd.Series([180.0]))
    explicitly_transient = scorer(
        pd.Series([180.0]),
        oxygenation_sustained_1h=pd.Series([False]),
    )

    assert unknown.tolist() == [2]
    assert explicitly_transient.tolist() == [0]


def test_sofa2_cns_uses_motor_response_when_gcs_missing():
    score = sofa2_cns(
        pd.Series([None, None, None, None, None, None]),
        motor_response=pd.Series([6, 5, 4, 3, 2, 1]),
    )

    assert score.tolist() == [0, 1, 2, 3, 4, 4]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("scorer", [sofa2_cns, standalone_sofa2_cns])
def test_sofa2_brain_score_one_requires_gcs_13_14_or_confirmed_delirium_treatment(scorer):
    """SOFA-2 Table 2 does not score a positive CAM assessment by itself."""
    score = scorer(
        pd.Series([15, 15, 14]),
        delirium_tx_evidence=pd.Series(["unavailable", "confirmed", "unavailable"]),
        delirium_positive=pd.Series([True, False, False]),
    )

    assert score.tolist() == [0, 1, 1]


def test_standalone_sofa2_cardio_mechanical_support_is_not_downgraded():
    score = standalone_sofa2_cardio(
        pd.Series([65, 75, 75]),
        norepi60=pd.Series([0.0, 0.1, 0.0]),
        mech_circ_support=pd.Series([True, True, True]),
    )

    assert score.tolist() == [4, 4, 4]


def test_sofa2_resp_any_ecmo_scores_four_regardless_of_indication():
    """SOFA-2 footnote (i): ANY ECMO scores 4 on the respiratory component,
    even with a normal PaO2:FiO2 and a non-respiratory (or unknown) indication.
    Pins the 2026-06 fix that replaced the respiratory-indication-only gate."""
    # pafi=400 alone would score 0; ECMO must floor it to 4.
    for scorer in (standalone_sofa2_resp, sofa2_resp):
        cardiovascular = scorer(
            pd.Series([400.0]),
            ecmo=pd.Series([True]),
            ecmo_indication=pd.Series(["cardiovascular"]),
        )
        assert cardiovascular.tolist() == [4]

        unknown_indication = scorer(
            pd.Series([400.0]),
            ecmo=pd.Series([True]),
        )
        assert unknown_indication.tolist() == [4]


def test_sofa2_cardio_va_ecmo_scores_four_but_vv_ecmo_does_not():
    """SOFA-2 footnotes (i)+(n): cardiovascular-indication (VA) ECMO is
    mechanical circulatory support and floors the cardiovascular component to 4;
    respiratory-indication (VV) ECMO is scored only on the respiratory component."""
    va = standalone_sofa2_cardio(
        pd.Series([75.0]),
        norepi60=pd.Series([0.0]),
        ecmo=pd.Series([True]),
        ecmo_indication=pd.Series(["cardiovascular"]),
    )
    assert va.tolist() == [4]

    vv = standalone_sofa2_cardio(
        pd.Series([75.0]),
        norepi60=pd.Series([0.0]),
        ecmo=pd.Series([True]),
        ecmo_indication=pd.Series(["respiratory"]),
    )
    assert vv.tolist() == [0]


def test_sofa2_resp_component_preserves_ecmo_indication_strings():
    callback = _callback_sofa_component(sofa2_resp)
    tables = {
        "pafi": ICUTable(
            pd.DataFrame({"stay_id": [1], "charttime": [0], "pafi": [400]}),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="pafi",
        ),
        "ecmo": ICUTable(
            pd.DataFrame({"stay_id": [1], "charttime": [0], "ecmo": [True]}),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="ecmo",
        ),
        "ecmo_indication": ICUTable(
            pd.DataFrame(
                {"stay_id": [1], "charttime": [0], "ecmo_indication": ["respiratory"]}
            ),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="ecmo_indication",
        ),
    }
    ctx = ConceptCallbackContext(
        concept_name="sofa2_resp",
        target=None,
        interval=None,
        resolver=None,
        data_source=None,
        patient_ids=None,
    )

    result = callback(tables, ctx)

    assert result.data["sofa2_resp"].tolist() == [4]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("scorer", [sofa2_cardio, standalone_sofa2_cardio])
def test_sofa2_cardio_counts_dopamine_and_dobutamine_as_adjuncts(scorer):
    score = scorer(
        pd.Series([75.0, 75.0, 75.0]),
        norepi60=pd.Series([0.10, 0.30, 0.0]),
        dopa60=pd.Series([5.0, 0.0, 50.0]),
        dobu60=pd.Series([0.0, 1.0, 1.0]),
    )

    # Low NE + dopamine -> 3; medium NE + dobutamine -> 4. Dopamine tiers
    # apply only when dopamine is the single vasoactive drug, so dopamine plus
    # dobutamine is the generic "other vaso/inotrope" 2-point rule.
    assert score.tolist() == [3, 4, 2]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("scorer", [sofa2_cns, standalone_sofa2_cns])
def test_sofa2_cns_uses_pre_sedation_gcs_and_confirmed_treatment_is_independent(scorer):
    score = scorer(
        pd.Series([6.0, None, 6.0]),
        sedated_gcs=pd.Series([15.0, None, None]),
        sedated=pd.Series([True, False, True]),
        motor_response=pd.Series([2.0, 6.0, 2.0]),
        delirium_tx_evidence=pd.Series(["unavailable", "confirmed", "unavailable"]),
    )

    assert score.tolist() == [0, 1, 0]


@pytest.mark.clinical_conformance
def test_delirium_treatment_proxy_never_confirms_main_cns_score():
    gcs = pd.Series([15, 15, 15, 15])
    evidence = pd.Series(
        ["confirmed", "proxy_only", "not_detected", "unavailable"]
    )
    proxy = pd.Series([True, True, False, False])

    assert standalone_sofa2_cns(
        gcs,
        delirium_tx_proxy=proxy,
        delirium_tx_evidence=evidence,
    ).tolist() == [1, 0, 0, 0]
    assert sofa2_cns_proxy_sensitivity(
        gcs,
        delirium_tx_proxy=proxy,
        delirium_tx_evidence=evidence,
    ).tolist() == [1, 1, 0, 0]
    assert sofa2_cns_ascertainment(
        gcs,
        delirium_tx_proxy=proxy,
        delirium_tx_evidence=evidence,
    ).tolist() == [
        "complete",
        "proxy_only",
        "complete_for_proxy_source",
        "unavailable",
    ]


@pytest.mark.clinical_conformance
def test_deprecated_delirium_tx_alias_is_proxy_only():
    legacy = pd.Series([True])

    assert standalone_sofa2_cns(
        pd.Series([15]), delirium_tx=legacy
    ).tolist() == [0]
    assert sofa2_cns_proxy_sensitivity(
        pd.Series([15]), delirium_tx=legacy
    ).tolist() == [1]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("scorer", [sofa2_resp, standalone_sofa2_resp])
def test_sofa2_resp_ceiling_exception_and_one_hour_persistence_gate(scorer):
    score = scorer(
        pd.Series([70.0, 70.0, 140.0]),
        adv_resp=pd.Series([False, False, True]),
        support_unavailable_or_ceiling=pd.Series([True, True, False]),
        oxygenation_sustained_1h=pd.Series([True, False, True]),
    )

    assert score.tolist() == [4, 0, 3]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("scorer", [sofa2_renal, standalone_sofa2_renal])
def test_sofa2_renal_accepts_episode_state_and_excludes_nonrenal_only_rrt(scorer):
    score = scorer(
        pd.Series([1.0, 1.0, 1.0, 1.0]),
        rrt=pd.Series([False, True, True, True]),
        rrt_criteria=pd.Series([False, False, False, True]),
        rrt_episode_active=pd.Series([True, False, False, False]),
        rrt_nonrenal_only=pd.Series([False, True, False, True]),
    )

    assert score.tolist() == [4, 0, 4, 4]


def _component_table(name: str, values) -> ICUTable:
    frame = pd.DataFrame(
        {
            "stay_id": [1] * len(values),
            "charttime": list(range(len(values))),
            name: values,
        }
    )
    if name in {
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    }:
        present = frame[name].notna().astype(int)
        frame[f"{name}_observed"] = present
        frame[f"{name}_available"] = present
    return ICUTable(
        frame,
        id_columns=["stay_id"],
        index_column="charttime",
        value_column=name,
    )


def _component_context(name: str, **kwargs) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name=name,
        target=None,
        interval=None,
        resolver=None,
        data_source=None,
        patient_ids=None,
        kwargs=kwargs,
    )


@pytest.mark.clinical_conformance
def test_sofa2_cardio_production_callback_preserves_va_ecmo_indication():
    callback = _callback_sofa_component(standalone_sofa2_cardio)
    common = {
        "map": _component_table("map", [75.0, 75.0]),
        "ecmo": _component_table("ecmo", [True, True]),
        "ecmo_indication": _component_table(
            "ecmo_indication", ["cardiovascular", "respiratory"]
        ),
    }

    result = callback(common, _component_context("sofa2_cardio"))

    assert result.data["sofa2_cardio"].tolist() == [4, 0]


@pytest.mark.clinical_conformance
def test_sofa2_production_callbacks_cover_pre_sedation_and_aggregate_completeness():
    cns = _callback_sofa_component(standalone_sofa2_cns)(
        {
            "gcs": _component_table("gcs", [6.0]),
            "sedated_gcs": _component_table("sedated_gcs", [15.0]),
        },
        _component_context("sofa2_cns"),
    )
    assert cns.data["sofa2_cns"].tolist() == [0]

    components = {
        name: _component_table(name, [score])
        for name, score in {
            "sofa2_resp": 1,
            "sofa2_coag": 2,
            "sofa2_liver": 3,
            "sofa2_cardio": 4,
            "sofa2_cns": 0,
            "sofa2_renal": 2,
        }.items()
    }
    aggregate = _callback_sofa2_score(
        components,
        _component_context("sofa2", keep_components=True),
    )

    assert aggregate.data["sofa2"].tolist() == [12]
    assert aggregate.data["sofa2_n_components"].tolist() == [6]


@pytest.mark.clinical_conformance
def test_sofa2_component_receipts_do_not_treat_synthetic_zero_as_evidence():
    respiratory = _callback_sofa_component(standalone_sofa2_resp)(
        {
            "pafi": _component_table("pafi", [np.nan]),
            "adv_resp": _component_table("adv_resp", [True]),
        },
        _component_context("sofa2_resp"),
    ).data
    cns = _callback_sofa_component(standalone_sofa2_cns)(
        {
            "gcs": _component_table("gcs", [np.nan]),
            "delirium_tx_evidence": _component_table(
                "delirium_tx_evidence", ["proxy_only"]
            ),
        },
        _component_context("sofa2_cns"),
    ).data

    assert respiratory["sofa2_resp"].tolist() == [0]
    assert respiratory["sofa2_resp_observed"].tolist() == [0]
    assert respiratory["sofa2_resp_available"].tolist() == [0]
    assert cns["sofa2_cns"].tolist() == [0]
    assert cns["sofa2_cns_observed"].tolist() == [0]
    assert cns["sofa2_cns_available"].tolist() == [0]


@pytest.mark.clinical_conformance
def test_sofa2_component_receipts_fail_closed_on_nullable_evidence():
    cns = _callback_sofa_component(standalone_sofa2_cns)(
        {
            "gcs": _component_table("gcs", [np.nan, np.nan]),
            "delirium_tx_evidence": _component_table(
                "delirium_tx_evidence",
                pd.Series([pd.NA, "proxy_only"], dtype="string"),
            ),
        },
        _component_context("sofa2_cns"),
    ).data

    assert cns["sofa2_cns_observed"].tolist() == [0, 0]
    assert cns["sofa2_cns_available"].tolist() == [0, 0]


@pytest.mark.clinical_conformance
def test_sofa2_aggregate_counts_component_receipts_not_score_non_nullness():
    components = {
        name: _component_table(name, [0.0])
        for name in (
            "sofa2_resp",
            "sofa2_coag",
            "sofa2_liver",
            "sofa2_cardio",
            "sofa2_cns",
            "sofa2_renal",
        )
    }
    components["sofa2_resp"].data["sofa2_resp_observed"] = 0
    components["sofa2_resp"].data["sofa2_resp_available"] = 0

    result = _callback_sofa2_score(
        components,
        _component_context("sofa2", keep_components=True),
    ).data

    assert pd.isna(result["sofa2"].tolist()[0])
    assert result["sofa2_n_observed_components"].tolist() == [5]
    assert result["sofa2_n_available_components"].tolist() == [5]
    assert result["sofa2_n_components"].tolist() == [5]


@pytest.mark.clinical_conformance
def test_sofa2_owner_receipt_prevents_synthetic_zero_from_truncating_locf():
    timeline = list(range(31))
    cns = _callback_sofa_component(standalone_sofa2_cns)(
        {
            "gcs": _component_table("gcs", [10.0] + [np.nan] * 30),
            "delirium_tx_evidence": _component_table(
                "delirium_tx_evidence",
                ["unavailable"] * 12
                + ["proxy_only"]
                + ["unavailable"] * 18,
            ),
        },
        _component_context("sofa2_cns"),
    )
    assert cns.data["charttime"].tolist() == timeline
    assert cns.data.loc[12, "sofa2_cns"] == 0
    assert cns.data.loc[12, "sofa2_cns_available"] == 0

    components = {
        name: _component_table(name, [0.0] * 31)
        for name in (
            "sofa2_resp",
            "sofa2_coag",
            "sofa2_liver",
            "sofa2_cardio",
            "sofa2_renal",
        )
    }
    components["sofa2_cns"] = cns

    at_30h = _callback_sofa2_score(
        components,
        _component_context("sofa2", keep_components=True),
    ).data.set_index("charttime").loc[30]

    assert at_30h["sofa2_cns"] == 2
    assert at_30h["sofa2"] == 2
    assert at_30h["sofa2_cns_observed"] == 0
    assert at_30h["sofa2_cns_available"] == 1
    assert at_30h["sofa2_n_observed_components"] == 5
    assert at_30h["sofa2_n_available_components"] == 6


def test_sofa2_aggregate_treats_legacy_score_only_inputs_as_asserted():
    components = {
        name: ICUTable(
            pd.DataFrame(
                {
                    "stay_id": [1],
                    "charttime": [0],
                    name: [0.0],
                }
            ),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column=name,
        )
        for name in (
            "sofa2_resp",
            "sofa2_coag",
            "sofa2_liver",
            "sofa2_cardio",
            "sofa2_cns",
            "sofa2_renal",
        )
    }

    result = _callback_sofa2_score(
        components,
        _component_context("sofa2"),
    ).data

    assert result["sofa2"].tolist() == [0]
    assert result["sofa2_n_observed_components"].tolist() == [6]
    assert result["sofa2_n_available_components"].tolist() == [6]


@pytest.mark.clinical_conformance
def test_sofa2_production_aggregate_keeps_observation_count_separate_from_zero_imputation():
    complete = {
        name: _component_table(name, [0.0])
        for name in (
            "sofa2_resp",
            "sofa2_coag",
            "sofa2_liver",
            "sofa2_cardio",
            "sofa2_cns",
            "sofa2_renal",
        )
    }
    five = dict(complete)
    five.pop("sofa2_renal")
    none_observed = {
        name: _component_table(name, [np.nan]) for name in complete
    }

    assert _callback_sofa2_score(
        complete, _component_context("sofa2")
    ).data["sofa2_n_components"].tolist() == [6]
    assert _callback_sofa2_score(
        five, _component_context("sofa2")
    ).data["sofa2_n_components"].tolist() == [5]
    all_missing = _callback_sofa2_score(
        none_observed, _component_context("sofa2")
    ).data
    assert all_missing["sofa2_n_components"].tolist() == [0]
    # No observed component: the total must stay unknown, not masquerade as a
    # true SOFA-2 score of zero.
    assert pd.isna(all_missing["sofa2"].tolist()[0])


@pytest.mark.clinical_conformance
def test_sofa2_production_aggregate_carries_last_component_beyond_24_hours():
    components = {
        "sofa2_liver": ICUTable(
            pd.DataFrame(
                {
                    "stay_id": [1],
                    "charttime": [0.0],
                    "sofa2_liver": [3.0],
                    "sofa2_liver_observed": [1],
                    "sofa2_liver_available": [1],
                }
            ),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="sofa2_liver",
        )
    }
    for name in (
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ):
        components[name] = ICUTable(
            pd.DataFrame(
                {
                    "stay_id": [1],
                    "charttime": [25.0],
                    name: [0.0],
                    f"{name}_observed": [1],
                    f"{name}_available": [1],
                }
            ),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column=name,
        )

    result = _callback_sofa2_score(
        components,
        _component_context("sofa2", keep_components=True),
    ).data.set_index("charttime")

    assert result.loc[25.0, "sofa2_liver"] == 3
    assert result.loc[25.0, "sofa2"] == 3
    assert result.loc[25.0, "sofa2_n_observed_components"] == 5
    assert result.loc[25.0, "sofa2_n_available_components"] == 6
    assert result.loc[25.0, "sofa2_n_components"] == 6


def test_sofa2_empty_aggregate_keeps_a_stable_schema():
    components = {
        name: _component_table(name, [])
        for name in (
            "sofa2_resp",
            "sofa2_coag",
            "sofa2_liver",
            "sofa2_cardio",
            "sofa2_cns",
            "sofa2_renal",
        )
    }

    compact = _callback_sofa2_score(
        components,
        _component_context("sofa2"),
    ).data
    detailed = _callback_sofa2_score(
        components,
        _component_context("sofa2", keep_components=True),
    ).data

    aggregate_columns = {
        "sofa2",
        "sofa2_n_observed_components",
        "sofa2_n_available_components",
        "sofa2_n_components",
    }
    component_columns = set(components)
    receipt_columns = {
        f"{name}_{kind}"
        for name in components
        for kind in ("observed", "available")
    }
    assert aggregate_columns <= set(compact)
    assert aggregate_columns | component_columns | receipt_columns <= set(detailed)


def test_legacy_cns_ascertainment_alias_matches_precise_public_name():
    inputs = {
        "gcs": pd.Series([15, 10]),
        "delirium_tx_evidence": pd.Series(["proxy_only", "unavailable"]),
    }

    precise = sofa2_cns_delirium_tx_ascertainment(**inputs)

    assert precise.tolist() == ["proxy_only", "not_score_relevant"]
    assert sofa2_cns_ascertainment(**inputs).equals(precise)
