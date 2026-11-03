"""MIMIC-III prescriptions: stay assignment and day-to-noon placement.

Audit 2026-11-03 (#26). MIMIC-III ``prescriptions`` is keyed by ``hadm_id``
(data-sources.json, as in ricu) but also ships its own ``icustay_id``, which is
empty for 35% of antibiotic orders (those outside MIMIC's own date rule, e.g.
the ward/ED day before ICU admission). The loader used that partial column as
the stay identifier and skipped the hadm rolling join, so every such order was
dropped. ``mimic_abx_presc`` then added the bare number 720 to the integer form
of a datetime, i.e. 720 microseconds instead of ricu's ``mins(720L)``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.concept.mimic_prescription_time import shift_index_minutes
from easyicu.config import DataSourceRegistry
from easyicu.datasource import ICUDataSource

REPO = Path(__file__).resolve().parents[2]


def _mimic_config():
    return DataSourceRegistry.from_json(REPO / "src/easyicu/data/data-sources.json").get("mimic")


def _write_tables(root: Path) -> None:
    ts = pd.Timestamp
    pd.DataFrame(
        {
            "row_id": [1, 2],
            "subject_id": [9, 9],
            "hadm_id": [100, 100],
            "icustay_id": [1, 2],
            "dbsource": ["carevue", "carevue"],
            "first_careunit": ["MICU", "MICU"],
            "last_careunit": ["MICU", "MICU"],
            "first_wardid": [1, 1],
            "last_wardid": [1, 1],
            "intime": [ts("2101-01-01 15:00"), ts("2101-01-05 10:00")],
            "outtime": [ts("2101-01-03 09:00"), ts("2101-01-07 09:00")],
            "los": [1.75, 1.96],
        }
    ).to_parquet(root / "icustays.parquet", index=False)
    pd.DataFrame(
        {
            "row_id": [11, 12, 13, 14],
            "subject_id": [9, 9, 9, 9],
            "hadm_id": [100, 100, 100, 100],
            # MIMIC leaves the stay empty for orders outside its own rule.
            "icustay_id": [None, 1, None, 2],
            "startdate": [ts("2101-01-01"), ts("2101-01-02"), ts("2101-01-04"), ts("2101-01-06")],
            "enddate": [ts("2101-01-02"), ts("2101-01-03"), ts("2101-01-05"), ts("2101-01-07")],
            "drug_type": ["MAIN"] * 4,
            "drug": ["Vancomycin", "Cefepime", "Ciprofloxacin", "Metronidazole"],
            "dose_val_rx": ["1"] * 4,
            "dose_unit_rx": ["g"] * 4,
            "route": ["IV"] * 4,
        }
    ).astype({"icustay_id": "Int64"}).to_parquet(root / "prescriptions.parquet", index=False)


def test_hadm_keyed_prescriptions_are_assigned_by_rolling_join_not_the_partial_stay_column(tmp_path):
    _write_tables(tmp_path)
    source = ICUDataSource(config=_mimic_config(), base_path=tmp_path)

    frame = source.load_table("prescriptions", verbose=False).data
    assigned = dict(zip(frame["drug"], frame["icustay_id"]))

    # Every order keeps exactly one row and one stay: the stay with the
    # smallest outtime at or after the order (ricu roll = -Inf on outtime).
    assert len(frame) == 4
    assert frame["icustay_id"].notna().all()
    assert assigned == {
        "Vancomycin": 1,  # the ward day before ICU admission
        "Cefepime": 1,
        "Ciprofloxacin": 2,  # between the stays -> the next stay
        "Metronidazole": 2,
    }


def test_abx_prescription_is_placed_at_noon_of_its_date():
    source = ConceptSource.from_mapping(
        {
            "regex": "vanco",
            "table": "prescriptions",
            "sub_var": "drug",
            "index_var": "startdate",
            "callback": "mimic_abx_presc",
            "class_name": "rgx_itm",
        }
    )
    frame = pd.DataFrame(
        {
            "icustay_id": [1],
            "startdate": pd.to_datetime(["2101-01-01"]).astype("datetime64[us]"),
            "drug": ["Vancomycin"],
        }
    )

    result = _apply_callback(frame, source, concept_name="abx")

    assert pd.api.types.is_datetime64_any_dtype(result["startdate"])
    assert result["startdate"].tolist() == [pd.Timestamp("2101-01-01 12:00")]
    assert result["abx"].tolist() == [True]


def test_relative_hour_index_moves_twelve_hours_and_garbage_is_not_reinterpreted():
    assert shift_index_minutes(pd.Series([-3.0, 10.5])).tolist() == [9.0, 22.5]
    with pytest.raises(ValueError, match="neither datetimes nor"):
        shift_index_minutes(pd.Series(["not a time"], dtype=object))
