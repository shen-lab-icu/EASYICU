import pandas as pd

from easyicu.scores.circ_failure import calculate_circ_failure_status, load_circ_failure


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


def test_load_circ_failure_uses_levosimendan_and_theophylline() -> None:
    preloaded = {
        "lact": _concept_frame("lact", [3.0, 3.0, 3.0]),
        "map": _concept_frame("map", [80.0, 80.0, 80.0]),
        "levo_rate": _concept_frame("levo_rate", [1.0, 0.0, 0.0]),
        "theo_rate": _concept_frame("theo_rate", [0.0, 1.0, 0.0]),
    }

    result = load_circ_failure(
        "miiv",
        preloaded_data=preloaded,
        use_rolling_window=False,
        verbose=False,
    ).sort_values("charttime")

    assert result["circ_event"].tolist() == [1, 1, 0]
    assert result["level1_drugs"].tolist() == [True, True, False]


def test_rolling_window_does_not_promote_event_from_single_drugged_point():
    # MAP is low throughout (sustained Event 1), lactate elevated throughout, but
    # a Level-3 vasopressor appears at a single timepoint only. The faithful
    # rolling rule labels Event k only when level >= k is sustained over >= 2/3
    # of the window, so this stays Event 1 — it must NOT be promoted to Event 3
    # by one drugged point (the previous `.any()` logic did exactly that).
    n = 6
    df = pd.DataFrame({
        "stay_id": [1] * n,
        "charttime": [i * 5 for i in range(n)],          # 5-min grid
        "lact": [3.0] * n,                               # elevated throughout
        "map": [60.0] * n,                               # low throughout -> Event 1
        "norepi_rate": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],   # Level 3 at one point
        "epi_rate": [0.0] * n,
        "vaso_rate": [0.0] * n,
    })

    out = calculate_circ_failure_status(
        df,
        window_size_minutes=15,   # window_steps = 3
        grid_size_minutes=5,
        use_rolling_window=True,
    ).sort_values("charttime")

    events = out["circ_event"].tolist()
    # Every center sees sustained lactate+low-MAP -> Event 1; the lone Level-3
    # point is never sustained over 2/3 of any window, so no Event 3 anywhere.
    assert max(events) == 1
    assert 3 not in events
    assert events == [1, 1, 1, 1, 1, 1]
