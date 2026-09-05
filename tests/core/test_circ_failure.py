import pandas as pd

from easyicu.scores.circ_failure import calculate_circ_failure_status, load_circ_failure
import pytest


def _concept_frame(name: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0, 5, 10],
            name: values,
        }
    )


def _complete_preloaded_data(**drug_values):
    """Give the drug-recognition fixtures explicit, complete zero-rate evidence."""
    preloaded = {
        "lact": _concept_frame("lact", [3.0, 3.0, 3.0]),
        "map": _concept_frame("map", [80.0, 80.0, 80.0]),
    }
    for name in (
        "norepi_rate", "epi_rate", "adh_rate", "dobu_rate", "dopa_rate",
        "phn_rate", "milrinone", "levo_rate", "theo_rate",
    ):
        preloaded[name] = _concept_frame(name, drug_values.get(name, [0.0] * 3))
    return preloaded


@pytest.fixture(autouse=True)
def forbid_live_concept_loading(monkeypatch):
    """These are unit tests; an accidental supplementary raw load is a failure."""
    def forbidden(*args, **kwargs):
        pytest.fail("circulatory unit tests must not load real ICU data")

    monkeypatch.setattr("easyicu.api.load_concepts", forbidden)


def test_load_circ_failure_uses_vasopressin_and_level1_drugs():
    preloaded = _complete_preloaded_data(
        adh_rate=[0.03, 0.0, 0.0],
        phn_rate=[0.0, 0.2, 0.0],
        milrinone=[0.0, 0.0, 1.0],
    )

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
    preloaded = _complete_preloaded_data(
        levo_rate=[1.0, 0.0, 0.0], theo_rate=[0.0, 1.0, 0.0],
    )

    result = load_circ_failure(
        "miiv",
        preloaded_data=preloaded,
        use_rolling_window=False,
        verbose=False,
    ).sort_values("charttime")

    assert result["circ_event"].tolist() == [1, 1, 0]
    assert result["level1_drugs"].tolist() == [True, True, False]


def test_preloaded_missing_rate_remains_unknown_without_being_reloaded():
    preloaded = _complete_preloaded_data(
        levo_rate=[1.0, 1.0, 0.0], norepi_rate=[0.0, pd.NA, 0.0],
    )
    result = load_circ_failure("miiv", preloaded_data=preloaded, verbose=False)

    pd.testing.assert_series_equal(
        result["circ_event"].reset_index(drop=True),
        pd.Series([1, pd.NA, 0], dtype="Int64", name="circ_event"),
    )
    assert pd.isna(result["circ_failure"].iloc[1])


def test_partially_supplied_optional_stream_preserves_unmatched_unknowns(monkeypatch):
    preloaded = _complete_preloaded_data(
        levo_rate=[1.0, 0.0, 0.0], theo_rate=[0.0, 1.0, 0.0],
    )
    del preloaded["norepi_rate"]
    calls = []

    def load_missing(**kwargs):
        calls.append(kwargs)
        return {"norepi_rate": _concept_frame("norepi_rate", [0.0] * 3).iloc[:1]}

    monkeypatch.setattr("easyicu.api.load_concepts", load_missing)
    result = load_circ_failure(
        "miiv", preloaded_data=preloaded, patient_ids=[1], verbose=False,
    )

    assert len(calls) == 1
    assert calls[0]["concepts"] == ["norepi_rate"]
    assert calls[0]["patient_ids"] == [1]
    pd.testing.assert_series_equal(
        result["circ_event"].reset_index(drop=True),
        pd.Series([1, pd.NA, pd.NA], dtype="Int64", name="circ_event"),
    )
    assert result["circ_failure"].isna().tolist() == [False, True, True]


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


# --- fail-closed and row-level unknowns (2026-08-16 data review) ---

def test_circ_failure_missing_map_fails_closed() -> None:
    from easyicu.scores.circ_failure import calculate_circ_failure_status

    df = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 5],
            "lact": [3.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="MAP"):
        calculate_circ_failure_status(df)


@pytest.mark.parametrize(
    ("missing_column", "expected_known_flag", "expected_unknown_flag"),
    [
        ("map", "lactate_elevated", "map_low"),
        ("lact", "map_low", "lactate_elevated"),
    ],
)

def test_circ_failure_row_level_core_missing_stays_unknown(
    missing_column: str,
    expected_known_flag: str,
    expected_unknown_flag: str,
) -> None:
    from easyicu.scores.circ_failure import calculate_circ_failure_status

    df = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0],
            "lact": [3.0],
            "map": [60.0],
            missing_column: [pd.NA],
        }
    )

    out = calculate_circ_failure_status(df, use_rolling_window=False)

    assert bool(out.loc[0, expected_known_flag]) is True
    assert pd.isna(out.loc[0, expected_unknown_flag])
    assert pd.isna(out.loc[0, "circ_event"])
    assert pd.isna(out.loc[0, "circ_failure"])

def test_circ_failure_row_level_drug_missing_stays_unknown() -> None:
    from easyicu.scores.circ_failure import calculate_circ_failure_status

    df = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0],
            "lact": [3.0],
            "map": [80.0],
            "norepi_rate": [pd.NA],
        }
    )

    out = calculate_circ_failure_status(df, use_rolling_window=False)

    assert pd.isna(out.loc[0, "level2_drugs"])
    assert pd.isna(out.loc[0, "level3_drugs"])
    assert pd.isna(out.loc[0, "circ_event"])
    assert pd.isna(out.loc[0, "circ_failure"])

def test_circ_failure_does_not_use_dataframe_index_as_row_identity() -> None:
    from easyicu.scores.circ_failure import calculate_circ_failure_status

    df = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 5],
            "lact": [3.0, 1.0],
            "map": [60.0, 80.0],
        },
        index=[0, 0],
    )

    out = calculate_circ_failure_status(df, use_rolling_window=False)

    assert out["circ_event"].tolist() == [1, 0]

def test_circ_failure_first_event_level_matches_first_time() -> None:
    from easyicu.scores.circ_failure import get_circ_failure_incidence

    df = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [300, 100],
            "circ_event": [3, 1],
        }
    )

    out = get_circ_failure_incidence(df).set_index("stay_id")

    assert out.loc[1, "first_circ_failure_time"] == 100
    assert out.loc[1, "first_event_level"] == 1
