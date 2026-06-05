import pandas as pd

from easyicu.circ_failure import load_circ_failure


def _concept_frame(name: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0, 5, 10],
            name: values,
        }
    )


def test_load_circ_failure_uses_vasopressin_and_level1_drugs():
    preloaded = {
        "lact": _concept_frame("lact", [3.0, 3.0, 3.0]),
        "map": _concept_frame("map", [80.0, 80.0, 80.0]),
        "norepi_rate": _concept_frame("norepi_rate", [0.0, 0.0, 0.0]),
        "epi_rate": _concept_frame("epi_rate", [0.0, 0.0, 0.0]),
        "adh_rate": _concept_frame("adh_rate", [0.03, 0.0, 0.0]),
        "dobu_rate": _concept_frame("dobu_rate", [0.0, 0.0, 0.0]),
        "dopa_rate": _concept_frame("dopa_rate", [0.0, 0.0, 0.0]),
        "phn_rate": _concept_frame("phn_rate", [0.0, 0.2, 0.0]),
        "milrinone": _concept_frame("milrinone", [0.0, 0.0, 1.0]),
    }

    result = load_circ_failure(
        "miiv",
        preloaded_data=preloaded,
        use_rolling_window=False,
        verbose=False,
    ).sort_values("charttime")

    assert result["circ_event"].tolist() == [3, 1, 1]
    assert result["level3_drugs"].tolist() == [True, False, False]
    assert result["level1_drugs"].tolist() == [False, True, True]
