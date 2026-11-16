"""S2HN: opt-in ``haemo_native`` module — haemodynamics at the one-minute grid."""

import json

import pandas as pd
import pytest

from easyicu.api import extraction as api
from easyicu.concept.catalog import BP_CHANNEL_CONCEPTS, HAEMO_NATIVE_CONCEPTS, HIDDEN_DICTIONARY_CONCEPTS
from easyicu.concept.minute_resolution import SUBHOUR_MODULE_INTERVALS
from easyicu.concept.source_channel import validate_sources
from easyicu.resources import load_dictionary


def test_haemo_native_is_an_opt_in_minute_module():
    assert api.EXTRACT_MODULES["haemo_native"] == list(HAEMO_NATIVE_CONCEPTS)
    assert "haemo_native" not in api.EXTRACT_MODULE_ORDER
    assert SUBHOUR_MODULE_INTERVALS["haemo_native"] == pd.Timedelta(minutes=1)
    assert set(BP_CHANNEL_CONCEPTS) <= HIDDEN_DICTIONARY_CONCEPTS
    defaults = {c for m in api.EXTRACT_MODULE_ORDER for c in api.EXTRACT_MODULES[m]}
    assert not defaults & set(BP_CHANNEL_CONCEPTS)


@pytest.mark.parametrize("base", ["sbp", "dbp"])
def test_pressure_channels_partition_the_combined_sources(base):
    """Each channel concept lists exactly the combined concept's sources of that channel."""
    dictionary = load_dictionary()
    definition = dictionary.get(base)
    for database, sources in definition.sources.items():
        if not validate_sources(definition, sources):
            # Untagged sources (community databases) cannot be split: no channel concept.
            assert all(database not in dictionary.get(f"{base}_{c}").sources for c in ("invasive", "nibp"))
            continue
        for channel, channels in (("invasive", {"invasive", "invasive_secondary"}), ("nibp", {"nibp"})):
            mine = [s for s in sources if s.params.get("source_channel") in channels]
            theirs = dictionary.get(f"{base}_{channel}").sources.get(database, [])
            key = lambda s: (str(s.table), str(s.value_var), str(s.sub_var), tuple(sorted(map(str, s.ids or ()))),  # noqa: E731
                             str(s.params.get("callback")))
            assert sorted(map(key, mine)) == sorted(map(key, theirs)), (database, channel)
        for channel in ("invasive", "nibp"):
            other = dictionary.get(f"{base}_{channel}")
            assert (other.units, other.minimum, other.maximum) == (
                definition.units, definition.minimum, definition.maximum)


@pytest.fixture
def hirid_pressure_source(tmp_path):
    """Synthetic HiRID with arterial (100) and cuff (600) systolic samples two and
    four minutes after one charted record of the same stay."""
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    source = tmp_path / "hirid"
    generate_source(source, "hirid", stays=3)
    path = source / "observations" / "1.parquet"
    table = pd.read_parquet(path)
    anchor = table.loc[table["variableid"].eq(110)].sort_values(["patientid", "datetime"]).iloc[[0]]
    extra = pd.concat([
        anchor.assign(variableid=item, value=value, datetime=anchor["datetime"] + pd.Timedelta(minutes=minute))
        for item, minute, value in ((100, 2, 121.0), (100, 4, 123.0), (600, 3, 131.0))
    ])
    pd.concat([table, extra], ignore_index=True).astype(table.dtypes.to_dict()).to_parquet(path, index=False)
    yield source, int(anchor["patientid"].iloc[0])
    clear_global_loader()


def test_minute_rows_keep_arterial_and_cuff_samples_apart(hirid_pressure_source, tmp_path):
    source, patient = hirid_pressure_source
    concepts = ["sbp", "sbp_invasive", "sbp_nibp"]
    api._run_module_extraction(
        module_name="haemo_native", concepts=concepts, database="hirid",
        data_path=str(source), patient_ids_filter={"patientid": [1, 2, 3]}, batch_size=3,
        output_dir=str(tmp_path),
    )
    assert json.loads((tmp_path / "_manifest.json").read_text())["errors"] == []
    frame = pd.read_parquet(tmp_path / "haemo_native.parquet")
    stay = frame.loc[frame.iloc[:, 0].eq(patient)]
    minutes = (pd.to_numeric(stay.iloc[:, 1]) * 60).round(6)

    def at(name):
        return dict(zip(minutes[stay[name].notna()], stay[name].dropna()))

    art, cuff = at("sbp_invasive"), at("sbp_nibp")
    assert {k: art.get(k) for k in (2.0, 4.0)} == {2.0: 121.0, 4.0: 123.0}
    assert cuff.get(3.0) == 131.0 and 3.0 not in art and 2.0 not in cuff


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_native_v2_export_publishes_the_minute_receipt(tmp_path, database):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    source = tmp_path / "src"
    generate_source(source, database, stays=3)
    out = tmp_path / "out"
    try:
        result = api.extract_database(
            database, data_path=str(source), output_dir=str(out), modules=["outcome", "haemo_native"],
            native_export_v2=True, resource_budget_mb=2048, verbose=False,
        )
    finally:
        clear_global_loader()
    assert not {k: v["errors"] for k, v in result["modules"].items() if v.get("errors")}
    entry = next(row for row in json.loads((out / "_manifest.json").read_text())["files"]
                 if row["module"] == "haemo_native")
    assert entry["physical_concept_ids"] == list(HAEMO_NATIVE_CONCEPTS)
    assert entry["time_resolution"]["signals"]
