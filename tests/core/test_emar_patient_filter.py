"""MIMIC-IV eMAR reads are scoped to the requested patients.

2026-11-03 extraction audit 2: miiv ``emar`` declares no ``id_var`` default, so
an eMAR-backed concept read all 26.8M rows (about 9 GB) before the stay join.
The loader routes eMAR through ``subject_id`` like ``prescriptions``.
"""

from __future__ import annotations

import pandas as pd

import easyicu
import easyicu.datasource as datasource


def test_emar_load_is_filtered_by_subject(tmp_path, monkeypatch):
    t = pd.Timestamp("2150-01-01 00:00")
    pd.DataFrame(
        {
            "subject_id": [10, 20],
            "hadm_id": [100, 200],
            "stay_id": [1000, 2000],
            "intime": [t, t],
            "outtime": [t + pd.Timedelta(days=3)] * 2,
            "first_careunit": ["MICU", "MICU"],
            "last_careunit": ["MICU", "MICU"],
            "los": [3.0, 3.0],
        }
    ).to_parquet(tmp_path / "icustays.parquet")
    pd.DataFrame(
        {
            "subject_id": [10, 20],
            "hadm_id": [100, 200],
            "emar_id": ["10-1", "20-1"],
            "emar_seq": [1, 1],
            "poe_id": ["p1", "p2"],
            "pharmacy_id": [1, 2],
            "enter_provider_id": [None, None],
            "charttime": [t + pd.Timedelta(hours=5)] * 2,
            "medication": ["Apixaban", "Apixaban"],
            "event_txt": ["Administered", "Administered"],
            "scheduletime": [t] * 2,
            "storetime": [t] * 2,
        }
    ).to_parquet(tmp_path / "emar.parquet")

    seen = []
    original = datasource.ICUDataSource.load_table

    def spy(self, table_name, *args, **kwargs):
        if table_name == "emar":
            seen.append([(f.column, list(f.value)) for f in kwargs.get("filters") or []])
        return original(self, table_name, *args, **kwargs)

    monkeypatch.setattr(datasource.ICUDataSource, "load_table", spy)
    out = easyicu.load_concepts(
        ["apixaban"], database="miiv", data_path=str(tmp_path),
        patient_ids=[1000], merge=False, verbose=False,
    )
    out = out["apixaban"] if isinstance(out, dict) else out

    assert seen and all(("subject_id", [10]) in filters for filters in seen)
    assert set(out["stay_id"]) == {1000}
