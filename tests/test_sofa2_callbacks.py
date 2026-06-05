import pandas as pd

from easyicu.callbacks import sofa2_cns, sofa2_resp
from easyicu.concept_callbacks import ConceptCallbackContext, _callback_sofa_component
from easyicu.sofa2 import sofa2_cardio as standalone_sofa2_cardio
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
