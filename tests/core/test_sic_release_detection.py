"""SICdb release from the prepared schema; 1.0.8 is the default, 1.0.6 stays readable."""

import pandas as pd

from easyicu.databases.detection import detect_sic_release
from easyicu.io.data_paths import find_database_path
from easyicu.research_agent.planning.literature_contract import (
    CURRENT_OFFICIAL_SOURCE_REGISTRY_VERSION,
    OFFICIAL_RESOURCE_REGISTRIES,
)
from tests.support.sic_references import sic_references


def _prepared(root, *, release):
    root.mkdir(parents=True)
    cases = {"CaseID": [1], "HospitalDischargeType": [3129 if release == "1.0.8" else 2026]}
    references = sic_references(release)
    if release == "1.0.8":
        cases.update(HospitalDischargeDay=[3], HospitalStayDays=[4])
        references = references.assign(LOINC_code=None)
    pd.DataFrame(cases).to_parquet(root / "cases.parquet", index=False)
    references.to_parquet(root / "d_references.parquet", index=False)
    return root


def test_the_release_is_read_from_the_schema(tmp_path):
    assert detect_sic_release(_prepared(tmp_path / "new", release="1.0.8")) == "1.0.8"
    assert detect_sic_release(_prepared(tmp_path / "old", release="1.0.6")) == "1.0.6"
    mixed = _prepared(tmp_path / "mixed", release="1.0.6")
    pd.DataFrame({"CaseID": [1], "HospitalDischargeDay": [3], "HospitalStayDays": [4]}).to_parquet(
        mixed / "cases.parquet", index=False)
    assert detect_sic_release(mixed) is None
    assert detect_sic_release(tmp_path / "absent") is None


def test_the_newest_release_directory_is_the_default_and_the_old_one_stays_found(tmp_path):
    _prepared(tmp_path / "sicdb" / "1.0.6", release="1.0.6")
    assert find_database_path(str(tmp_path), "sic").endswith("1.0.6")
    _prepared(tmp_path / "sicdb" / "1.0.8", release="1.0.8")
    assert find_database_path(str(tmp_path), "sic").endswith("1.0.8")


def test_both_releases_are_registered_and_the_old_registry_replays():
    current = OFFICIAL_RESOURCE_REGISTRIES[CURRENT_OFFICIAL_SOURCE_REGISTRY_VERSION]
    releases = {entry.dataset_version: entry.doi for entry in current
                if entry.database == "sic" and entry.kind == "dataset_release"}
    assert releases == {"1.0.6": "10.13026/dm9f-dm56", "1.0.8": "10.13026/8m72-6j83"}
    old = OFFICIAL_RESOURCE_REGISTRIES["easyicu.official_source_identity/2"]
    assert not any(entry.dataset_version == "1.0.8" for entry in old)
