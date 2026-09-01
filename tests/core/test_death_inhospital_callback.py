"""Regression tests for AUMC/SICdb in-hospital ``death`` callbacks."""

from pathlib import Path
import types

import pandas as pd
import pytest

from easyicu.config import DataSourceConfig
from easyicu.concept import ConceptResolver
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptDictionary
from easyicu.table import ICUTable


def _src(callback, index_var=None, value_var=None):
    return types.SimpleNamespace(
        callback=callback,
        index_var=index_var,
        value_var=value_var,
        sub_var=None,
        unit_var=None,
        table=None,
        ids=None,
    )


def test_aumc_death_matches_ricu_72h_window():
    n = 20
    frame = pd.DataFrame(
        {
            "admissionid": list(range(1, n + 1)),
            # The loader renames dischargedat to the concept name; values are minutes.
            "death": [1000] * n,
            "dateofdeath": [
                900,
                1000 + 1440,
                1000 + 200 * 24 * 60,
                None,
            ]
            + [None] * 16,
        }
    )

    out = _apply_callback(
        frame,
        _src("aumc_death", "dateofdeath", "dischargedat"),
        "death",
    )

    died = out["death"] == True  # noqa: E712
    assert int(died.sum()) == 2
    assert pd.isna(out.loc[out["admissionid"] == 3, "death"].iloc[0])


def test_aumc_death_is_row_local_for_high_mortality_cohort():
    # Legitimate high mortality must not switch the endpoint definition for all rows.
    frame = pd.DataFrame(
        {
            "admissionid": [1, 2, 3, 4],
            "death": [1000, 1000, 1000, 1000],
            "dateofdeath": [900, 1100, 1200, None],
        }
    )

    out = _apply_callback(
        frame,
        _src("aumc_death", "dateofdeath", "dischargedat"),
        "death",
    )

    assert int((out["death"] == True).sum()) == 3  # noqa: E712


def test_sic_death_uses_hospital_discharge_type_not_offset():
    frame = pd.DataFrame(
        {
            "CaseID": [1, 2, 3, 4, 5, 6],
            "death": [3600, 7200, None, 3_700_000, None, 1800],
            "HospitalDischargeType": [2028, 2028, 2026, 2026, None, 9999],
        }
    )

    out = _apply_callback(
        frame,
        _src("sic_death", "OffsetOfDeath", "OffsetOfDeath"),
        "death",
    )

    assert int((out["death"] == True).sum()) == 2  # noqa: E712
    assert pd.isna(out.loc[out["CaseID"] == 4, "death"].iloc[0])
    assert out.loc[out["CaseID"] == 1, "charttime"].iloc[0] == 1.0
    assert pd.isna(out.loc[out["CaseID"] == 4, "charttime"].iloc[0])
    assert pd.isna(out.loc[out["CaseID"] == 6, "death"].iloc[0])


def test_sic_death_missing_disposition_fails_closed():
    frame = pd.DataFrame({"CaseID": [1, 2], "death": [3600, 7200]})

    with pytest.raises(ValueError, match="HospitalDischargeType"):
        _apply_callback(
            frame,
            _src("sic_death", "OffsetOfDeath", "OffsetOfDeath"),
            "death",
        )


def test_sic_death_loader_requests_authoritative_disposition_column():
    class SicDataSource:
        base_path = None

        def __init__(self) -> None:
            self.requested_columns: list[str] = []
            self.config = DataSourceConfig(
                name="sic",
                tables={
                    "cases": {
                        "defaults": {
                            "id_var": "CaseID",
                            "index_var": "OffsetOfDeath",
                            "val_var": "OffsetOfDeath",
                        }
                    }
                },
            )

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            del table_name, filters, verbose
            self.requested_columns = list(columns or [])
            frame = pd.DataFrame(
                {
                    "CaseID": [1, 2, 3],
                    "OffsetOfDeath": [3600, 3_700_000, None],
                    "HospitalDischargeType": [2028, 2026, 2026],
                }
            )
            keep = list(
                dict.fromkeys(
                    [
                        "CaseID",
                        *[
                            column
                            for column in self.requested_columns
                            if column in frame.columns
                        ],
                    ]
                )
            )
            return ICUTable(
                data=frame[keep],
                id_columns=["CaseID"],
                index_column="OffsetOfDeath",
                value_column="OffsetOfDeath",
            )

    dictionary = ConceptDictionary.from_json(
        Path(__file__).parents[1] / "src/easyicu/data/concept-dict.json"
    )
    source = SicDataSource()

    loaded = ConceptResolver(dictionary).load_concepts(
        ["death"],
        source,
        merge=False,
        r_compatible=False,
        verbose=False,
        concept_workers=1,
    )
    result = loaded["death"].data

    assert "HospitalDischargeType" in source.requested_columns
    assert result.loc[result["CaseID"] == 1, "death"].eq(True).all()  # noqa: E712
    assert not result.loc[result["CaseID"] == 2, "death"].eq(True).any()  # noqa: E712
