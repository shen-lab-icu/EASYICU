"""Hospital status belongs to the admission, not to one timed ICU event."""

from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def raw():
    stays = pd.DataFrame(
        {
            "stay_id": [11, 12, 13, 14, 15, 16],
            "hadm_id": [101, 101, 102, 103, 104, 105],
            "intime": pd.to_datetime(["2026-01-01", "2026-01-02"] + ["2026-01-01"] * 4),
        }
    )
    admissions = pd.DataFrame(
        {
            "hadm_id": [101, 102, 103, 104],
            "hospital_expire_flag": [1, 1, 0, 2],
            "deathtime": pd.to_datetime(["2026-01-03", None, None, "2026-01-03"]),
        }
    )
    return stays, admissions


def test_status_covers_sibling_stays_and_does_not_require_a_clock(raw):
    from easyicu.hospital_mortality import derive_mimic_hospital_mortality_status

    stays, admissions = raw
    result = derive_mimic_hospital_mortality_status(stays, admissions)
    assert result.frame.stay_id.tolist() == stays.stay_id.tolist()
    pd.testing.assert_series_equal(
        result.frame.hospital_death,
        pd.Series(
            [True, True, True, False, pd.NA, pd.NA],
            dtype="boolean",
            name="hospital_death",
        ),
    )
    assert result.receipt["unknown_status_stays"] == 2
    assert result.receipt["event_stays"] == 3
    assert result.receipt["clock_required"] is False
    assert "death_time_hours" not in result.frame
    no_clocks = derive_mimic_hospital_mortality_status(
        stays.drop(columns="intime"), admissions.drop(columns="deathtime")
    )
    pd.testing.assert_frame_equal(no_clocks.frame, result.frame)


@pytest.mark.parametrize("table,key", [(0, "stay_id"), (1, "hadm_id")])
def test_ambiguous_keys_fail_closed(raw, table, key):
    from easyicu.hospital_mortality import derive_mimic_hospital_mortality_status

    tables = list(raw)
    tables[table] = pd.concat(
        [tables[table], tables[table].iloc[:1]], ignore_index=True
    )
    with pytest.raises(ValueError, match="key_nonunique"):
        derive_mimic_hospital_mortality_status(*tables)


@pytest.mark.parametrize(
    "database,id_column",
    [("miiv", "stay_id"), ("mimic", "icustay_id"), ("mimic_demo", "icustay_id")],
)
def test_default_concept_routes_to_status_owner_and_preserves_untimed_death(
    raw, database, id_column
):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary

    stays, admissions = raw
    config = SimpleNamespace(name=database, class_prefix=[])
    tables = {
        "icustays": stays.rename(columns={"stay_id": id_column}),
        "admissions": admissions,
    }
    source = SimpleNamespace(
        config=config,
        load_table=lambda name, **kwargs: SimpleNamespace(data=tables[name].copy()),
    )
    resolver = object.__new__(ConceptResolver)
    resolver.dictionary = load_dictionary()
    result = resolver._load_single_concept(
        "death",
        source,
        aggregator=None,
        patient_ids={id_column: [11, 13, 14]},
        verbose=False,
    )
    assert result.index_column is None
    assert result.data[id_column].tolist() == [11, 13, 14]
    assert result.data.death.tolist() == [True, True, False]
    assert result.data.death_time.iloc[0] == 48.0
    assert result.data.death_time.iloc[1:].isna().all()


def test_dense_status_cohort_projection_preserves_unknown_and_clock_companion():
    from easyicu.research_agent.cohort.materializer import (
        _binary_event_column,
        _event_time_column,
    )

    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "death": [1, 0, None, 1],
            "charttime": [0, 0, 0, 0],
            "death_time": [48.0, None, None, None],
        }
    )
    status = _binary_event_column(frame, "death", preserve_unknown=True)
    assert status.death.iloc[:2].tolist() == [1, 0]
    assert pd.isna(status.death.iloc[2])
    assert status.death.iloc[3] == 1
    timing = _event_time_column(frame, "death")
    assert timing.death_time.iloc[0] == 48.0
    assert timing.death_time.iloc[1:].isna().all()
