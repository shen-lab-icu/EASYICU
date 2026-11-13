"""S2RECR: opt-in ``vent_native`` module — respiratory mechanics at the one-minute grid."""

import json

import pandas as pd
import pytest

from easyicu.api import extraction as api
from easyicu.concept.catalog import VENT_NATIVE_CONCEPTS
from easyicu.concept.minute_resolution import SUBHOUR_MODULE_INTERVALS


def test_vent_native_is_an_opt_in_minute_module():
    assert VENT_NATIVE_CONCEPTS == (
        "plateau_pres_raw", "tidal_vol", "peep_set", "peep_total", "compliance", "po2", "fio2_chart",
    )
    assert api.EXTRACT_MODULES["vent_native"] == list(VENT_NATIVE_CONCEPTS)
    assert "vent_native" not in api.EXTRACT_MODULE_ORDER
    assert SUBHOUR_MODULE_INTERVALS["vent_native"] == pd.Timedelta(minutes=1)


@pytest.fixture
def hirid_mechanics_source(tmp_path):
    """Synthetic HiRID with plateau pressure (3200) and tidal volume (2410) samples
    two and four minutes after one charted record of the same stay."""
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    source = tmp_path / "hirid"
    generate_source(source, "hirid", stays=3)
    path = source / "observations" / "1.parquet"
    table = pd.read_parquet(path)
    anchor = table.loc[table["variableid"].eq(110)].sort_values(["patientid", "datetime"]).iloc[[0]]
    extra = pd.concat([
        anchor.assign(variableid=item, value=value, datetime=anchor["datetime"] + pd.Timedelta(minutes=minute))
        for item, minute, value in ((3200, 2, 21.0), (3200, 4, 23.0), (2410, 2, 410.0), (2410, 4, 450.0))
    ])
    pd.concat([table, extra], ignore_index=True).astype(table.dtypes.to_dict()).to_parquet(path, index=False)
    yield source, int(anchor["patientid"].iloc[0])
    clear_global_loader()


def test_minute_rows_keep_samples_that_the_hourly_grid_merges(hirid_mechanics_source, tmp_path):
    source, patient = hirid_mechanics_source
    api._run_module_extraction(
        module_name="vent_native", concepts=api.EXTRACT_MODULES["vent_native"], database="hirid",
        data_path=str(source), patient_ids_filter={"patientid": [1, 2, 3]}, batch_size=3,
        output_dir=str(tmp_path),
    )
    assert json.loads((tmp_path / "_manifest.json").read_text())["errors"] == []
    frame = pd.read_parquet(tmp_path / "vent_native.parquet")
    stay = frame.loc[frame.iloc[:, 0].eq(patient)]
    minutes = (pd.to_numeric(stay.iloc[:, 1]) * 60).round(6)
    plateau = dict(zip(minutes[stay["plateau_pres_raw"].notna()], stay["plateau_pres_raw"].dropna()))
    volume = dict(zip(minutes[stay["tidal_vol"].notna()], stay["tidal_vol"].dropna()))
    assert {k: plateau.get(k) for k in (2.0, 4.0)} == {2.0: 21.0, 4.0: 23.0}
    assert {k: volume.get(k) for k in (2.0, 4.0)} == {2.0: 410.0, 4.0: 450.0}


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_native_v2_export_publishes_the_minute_receipt(tmp_path, database):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    source = tmp_path / "src"
    generate_source(source, database, stays=3)
    out = tmp_path / "out"
    try:
        result = api.extract_database(
            database, data_path=str(source), output_dir=str(out), modules=["outcome", "vent_native"],
            native_export_v2=True, resource_budget_mb=2048, verbose=False,
        )
    finally:
        clear_global_loader()
    assert not {k: v["errors"] for k, v in result["modules"].items() if v.get("errors")}
    entry = next(row for row in json.loads((out / "_manifest.json").read_text())["files"]
                 if row["module"] == "vent_native")
    assert entry["physical_concept_ids"] == list(VENT_NATIVE_CONCEPTS)
    assert entry["time_resolution"]["signals"]
