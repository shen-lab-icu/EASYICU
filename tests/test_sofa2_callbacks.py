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


def test_sofa2_cns_uses_motor_response_when_gcs_missing():
    score = sofa2_cns(
        pd.Series([None, None, None, None, None, None]),
        motor_response=pd.Series([6, 5, 4, 3, 2, 1]),
    )

    assert score.tolist() == [0, 1, 2, 3, 4, 4]


@pytest.mark.clinical_conformance
@pytest.mark.parametrize("scorer", [sofa2_cns, standalone_sofa2_cns])
def test_sofa2_brain_score_one_requires_gcs_13_14_or_delirium_treatment(scorer):
    """SOFA-2 Table 2 does not score a positive CAM assessment by itself."""
    score = scorer(
        pd.Series([15, 15, 14]),
        delirium_tx=pd.Series([False, True, False]),
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
def test_sofa2_cns_uses_pre_sedation_gcs_and_delirium_treatment_is_independent(scorer):
    score = scorer(
        pd.Series([6.0, None, 6.0]),
        sedated_gcs=pd.Series([15.0, None, None]),
        sedated=pd.Series([True, False, True]),
        motor_response=pd.Series([2.0, 6.0, 2.0]),
        delirium_tx=pd.Series([False, True, False]),
    )

    assert score.tolist() == [0, 1, 0]


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
    return ICUTable(
        pd.DataFrame({"stay_id": [1] * len(values), "charttime": list(range(len(values))), name: values}),
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
