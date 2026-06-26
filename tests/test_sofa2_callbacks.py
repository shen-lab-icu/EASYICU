import pandas as pd

from easyicu.callbacks import sofa2_cns, sofa2_resp
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_sofa_component
from easyicu.scores.sofa2 import sofa2_cardio as standalone_sofa2_cardio
from easyicu.scores.sofa2 import sofa2_resp as standalone_sofa2_resp
from easyicu.table import ICUTable


def test_sofa2_resp_uses_fractional_fio2_for_safi_fallback():
    score = sofa2_resp(
        pd.Series([None, None]),
        spo2=pd.Series([95, 97]),
        fio2=pd.Series([50, 40]),
        adv_resp=pd.Series([True, True]),
    )

    assert score.tolist() == [3, 2]


def test_sofa2_cns_uses_motor_response_when_gcs_missing():
    score = sofa2_cns(
        pd.Series([None, None, None, None, None, None]),
        motor_response=pd.Series([6, 5, 4, 3, 2, 1]),
    )

    assert score.tolist() == [0, 1, 2, 3, 4, 4]


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
