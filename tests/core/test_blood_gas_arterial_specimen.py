"""Generic blood-gas concepts keep only arterial samples.

Synthetic raw sources go through the shipped dictionary, the real loader and
hourly aggregation.  Each stay mixes arterial gases with venous, mixed-venous,
unrecorded or conflicting specimens; only the arterial values may surface.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.blood_gas_specimen import (
    ARTERIAL_SPECIMEN_CALLBACK,
    SPECIMEN_RULES,
    ArterialSpecimenContractError,
    is_arterial_specimen,
    parse_arterial_specimen_callback,
    restrict_to_arterial_specimens,
)
from easyicu.resources import load_dictionary
from tests.support.blood_gas_sources import (
    aumc_source,
    hourly,
    miiv_source,
    mimic_source,
    sic_source,
)

READERS = pytest.mark.parametrize("scan", [True, False], ids=["parquet_scan", "table_reader"])


@READERS
def test_miiv_uses_the_specimen_type_of_the_same_specimen_id(tmp_path, scan):
    source, id_column = miiv_source(tmp_path)
    # VEN., CENTRAL VENOUS., unrecorded and ART.+VEN. conflicts are excluded.
    assert hourly(source, id_column, "po2", parquet_scan=scan) == {0.0: 90.0}


@READERS
def test_miiv_ph_drops_venous_lab_and_venous_chart_channels(tmp_path, scan):
    source, id_column = miiv_source(tmp_path)
    # Lab 7.30 (VEN.) and chart 220274 "PH (Venous)" 7.31 must not appear.
    assert hourly(source, id_column, "ph", parquet_scan=scan) == {0.0: 7.45, 2.0: 7.40}


@READERS
@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
def test_mimic_iii_matches_specimen_type_by_subject_and_charttime(tmp_path, database, scan):
    source, id_column = mimic_source(tmp_path, database)
    # MIX and ART+VEN are excluded, an unrecorded gas is kept; 490 (alveolar
    # PAO2, 300 mmHg) is no longer a PaO2 channel while 779 "Arterial PaO2" is.
    assert hourly(source, id_column, "po2", parquet_scan=scan) == {0.0: 90.0, 2.0: 80.0, 5.0: 110.0}


@READERS
def test_mimic_iii_ph_keeps_arterial_chart_channels_only(tmp_path, scan):
    source, id_column = mimic_source(tmp_path)
    # 860 "Venous pH" and 4202 "pH (cap)" are removed; 780 "Arterial pH" stays.
    assert hourly(source, id_column, "ph", parquet_scan=scan) == {0.0: 7.45, 2.0: 7.40}


@READERS
def test_aumc_drops_recorded_non_arterial_specimens_and_keeps_unrecorded(tmp_path, scan):
    source, id_column = aumc_source(tmp_path)
    # Ven. (35) and the "Vervalt" status code (80) are excluded; the kPa gas
    # without a specimen record is kept and still converted (12 kPa).
    assert hourly(source, id_column, "po2", parquet_scan=scan) == {0.0: 90.0, 2.0: 90.007, 4.0: 95.0}


@READERS
def test_sic_keeps_laboratory_type_arterial_and_drops_temperature_corrected_po2(tmp_path, scan):
    source, id_column = sic_source(tmp_path)
    # Empty LaboratoryType (35) and the 664 "PO2 (T)" duplicate are excluded.
    assert hourly(source, id_column, "po2", parquet_scan=scan) == {2.0: 90.0}
    assert hourly(source, id_column, "ph", parquet_scan=scan) == {3.0: 7.40}


def test_specimen_token_rule():
    miiv = SPECIMEN_RULES["miiv"].arterial_tokens
    mimic = SPECIMEN_RULES["mimic"].arterial_tokens
    assert is_arterial_specimen(["ART."], miiv) is True
    assert is_arterial_specimen([" art. ", "ART."], miiv) is True
    assert is_arterial_specimen(["ART"], miiv) is False  # unknown spelling
    assert is_arterial_specimen(["ART"], mimic) is True
    assert is_arterial_specimen(["Art."], SPECIMEN_RULES["aumc"].arterial_tokens) is True
    assert is_arterial_specimen(["ART.", "VEN."], miiv) is False
    assert is_arterial_specimen(["CENTRAL VENOUS."], miiv) is False
    assert is_arterial_specimen(["Gestold"], SPECIMEN_RULES["aumc"].arterial_tokens) is False
    assert is_arterial_specimen(["", None], miiv) is None
    assert parse_arterial_specimen_callback("arterial_specimen_only") == (True, None)
    assert parse_arterial_specimen_callback(
        "arterial_specimen_only(transform_fun(binary_op(`*`, 100)))"
    ) == (True, "transform_fun(binary_op(`*`, 100))")
    assert parse_arterial_specimen_callback("convert_unit(x)") == (False, None)


def test_missing_specimen_key_fails_instead_of_passing_rows_through():
    class Source:
        class config:
            name = "miiv"

    frame = pd.DataFrame({"subject_id": [1], "po2": [90.0]})
    with pytest.raises(ArterialSpecimenContractError) as error:
        restrict_to_arterial_specimens(frame, data_source=Source(), concept_name="po2")
    assert error.value.reason == "arterial_specimen_keys_missing"


@pytest.mark.parametrize("concept", ["po2", "pco2", "ph", "sao2"])
@pytest.mark.parametrize("database", ["miiv", "mimic", "mimic_demo", "aumc", "sic"])
def test_every_specimen_mixed_blood_gas_source_is_restricted(concept, database):
    """Lab/numeric gas sources opt in; arterial-labelled chart channels do not."""

    sources = load_dictionary()[concept].sources.get(database, [])
    gas_tables = {"labevents", "numericitems", "laboratory"}
    gas_sources = [s for s in sources if s.table in gas_tables]
    assert gas_sources
    for source in gas_sources:
        applies, _ = parse_arterial_specimen_callback(source.callback or "")
        assert applies, (concept, database, source.ids)
    if database == "miiv":
        assert all("specimen_id" in s.params["extra_vars"] for s in gas_sources)
    if database == "sic":
        assert all("LaboratoryType" in s.params["extra_vars"] for s in gas_sources)
    assert ARTERIAL_SPECIMEN_CALLBACK == "arterial_specimen_only"
