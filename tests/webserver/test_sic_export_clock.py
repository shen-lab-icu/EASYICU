"""The SIC clock declaration describes new producer metadata, not a source grant."""
from types import SimpleNamespace

import pytest

from easyicu.concept.provenance import capture_data_provenance, metadata_digest
from easyicu.concept.sic_landmark import SIC_CASE_LANDMARK_COLUMNS
from easyicu.config import load_src_cfg
from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
from easyicu.resources import load_dictionary
from easyicu.webserver.sic_export_clock import sic_export_clock


def _inputs():
    config = load_src_cfg("sic")
    snapshot = capture_data_provenance(
        load_dictionary(include_sofa2=True), concepts=sorted(SIC_CASE_LANDMARK_COLUMNS),
        source_database="sic", source_config=config,
        source_config_sha256=metadata_digest(config.model_dump(mode="json")),
    )
    return {"snapshots": [snapshot.model_dump(mode="json")]}


def _binding(*, origin=SIC_CARE_UNIT_ORIGIN, unit="h", identity="CaseID"):
    coordinate = SimpleNamespace(column="charttime", origin=origin, unit=unit)
    return SimpleNamespace(identity_column=identity, time_coordinates=[coordinate])


def test_current_sic_clock_requires_matching_provenance_bounds_and_coordinates():
    base = dict(database="sic", file_bindings=[_binding()], data_provenance=_inputs(),
                time_window_authority={"required": True, "source": "outcome.los_icu", "bounded_stays": 2})
    assert sic_export_clock(**base)["time_origin"] == SIC_CARE_UNIT_ORIGIN
    with pytest.raises(ValueError, match="time_metadata_mismatch"):
        sic_export_clock(**{**base, "file_bindings": [_binding(origin="icu_admission")]})
    with pytest.raises(ValueError, match="time_metadata_mismatch"):
        sic_export_clock(**{**base, "file_bindings": [_binding(unit="s")]})
    with pytest.raises(ValueError, match="identity_metadata_mismatch"):
        sic_export_clock(**{**base, "file_bindings": [_binding(identity="hadm_id")]})
    with pytest.raises(ValueError, match="time_window_authority_missing"):
        sic_export_clock(**{**base, "time_window_authority": {"required": True, "bounded_stays": 0}})
    assert sic_export_clock(**{**base, "data_provenance": {"snapshots": []}}) is None
    assert sic_export_clock(**{**base, "database": "miiv"}) is None
