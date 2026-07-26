"""Regression tests for the canonical database-profile registry."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu import api
from easyicu.api import concepts as concept_api
from easyicu.config import (
    DATABASE_ID_CONFIG,
    DataSourceConfig,
    DataSourceRegistry,
)
from easyicu.databases.profiles import (
    DatabaseIdConfigView,
    DatabaseProfileMetadata,
    get_database_profile,
    get_packaged_registry,
    iter_database_profiles,
    normalize_database_key,
    public_database_keys,
)
from easyicu.io.data_paths import DATABASE_ALIASES, find_database_path
from easyicu.webserver import crossdb_review


EXPECTED_PUBLIC = ("miiv", "eicu", "aumc", "hirid", "mimic", "sic")


def test_packaged_registry_has_six_typed_public_profiles_and_two_demo_profiles() -> (
    None
):
    registry = get_packaged_registry()
    assert isinstance(registry.get("miiv").profile, DatabaseProfileMetadata)
    assert registry.get("mimic_demo").profile is None

    profiles = iter_database_profiles(registry=registry)
    assert (
        tuple(profile.key for profile in profiles if profile.is_public)
        == EXPECTED_PUBLIC
    )
    assert {profile.key for profile in profiles if not profile.is_public} == {
        "eicu_demo",
        "mimic_demo",
    }

    mimic_demo = get_database_profile("mimic-demo", registry=registry)
    assert mimic_demo.parent_key == "mimic"
    assert mimic_demo.is_public is False
    assert mimic_demo.stay_table == "icustays"
    assert mimic_demo.stay_id_col == "icustay_id"


@pytest.mark.parametrize(
    ("key", "display_name", "display_order", "stay_table", "stay_id_col"),
    [
        ("miiv", "MIMIC-IV", 10, "icustays", "stay_id"),
        ("eicu", "eICU", 20, "patient", "patientunitstayid"),
        ("aumc", "AmsterdamUMCdb", 30, "admissions", "admissionid"),
        ("hirid", "HiRID", 40, "general", "patientid"),
        ("mimic", "MIMIC-III", 50, "icustays", "icustay_id"),
        ("sic", "SICdb", 60, "cases", "CaseID"),
    ],
)
def test_public_profile_metadata_and_icustay_contract(
    key: str,
    display_name: str,
    display_order: int,
    stay_table: str,
    stay_id_col: str,
) -> None:
    profile = get_database_profile(key)
    assert profile.display_name == display_name
    assert profile.display_order == display_order
    assert profile.stay_table == stay_table
    assert profile.stay_id_col == stay_id_col
    assert normalize_database_key(display_name) == key


def test_stay_identifier_uses_icustay_config_not_default_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = DataSourceConfig(
        name="custom",
        profile={
            "display_name": "Custom ICU",
            "aliases": ["custom-icu"],
            "display_order": 1,
        },
        id_cfg={
            "patient": {
                "id": "wrong_patient_id",
                "position": 1,
                "table": "patients",
            },
            "icustay": {
                "id": "exact_stay_id",
                "position": 99,
                "table": "exact_stays",
            },
        },
        tables={},
    )
    registry = DataSourceRegistry([source])

    def reject_default_id(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("get_default_id must not define the ICU-stay contract")

    monkeypatch.setattr(DataSourceConfig, "get_default_id", reject_default_id)
    profile = get_database_profile("custom-icu", registry=registry)
    assert profile.stay_table == "exact_stays"
    assert profile.stay_id_col == "exact_stay_id"


def test_packaged_registry_and_compatibility_view_are_lazy_and_read_only() -> None:
    get_packaged_registry.cache_clear()
    view = DatabaseIdConfigView()
    assert get_packaged_registry.cache_info().currsize == 0

    assert view["miiv"] == {"table": "icustays", "id_col": "stay_id"}
    assert get_packaged_registry.cache_info().currsize == 1
    assert isinstance(DATABASE_ID_CONFIG, Mapping)
    assert DATABASE_ID_CONFIG.get("mimic_demo") == {
        "table": "icustays",
        "id_col": "icustay_id",
    }
    assert dict(DATABASE_ID_CONFIG)["eicu"]["id_col"] == "patientunitstayid"
    assert dict(DATABASE_ID_CONFIG.items())["sic"]["table"] == "cases"

    with pytest.raises(TypeError):
        DATABASE_ID_CONFIG["miiv"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        DATABASE_ID_CONFIG["miiv"]["id_col"] = "changed"  # type: ignore[index]


def test_api_mimic_demo_normalize_sample_batch_count_and_public_scan(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = SimpleNamespace(database="mimic_demo")
    assert api._normalize_patient_ids_for_db("mimic_demo", [2, 1]) == {
        "icustay_id": [2, 1]
    }
    assert api._get_patient_id_source(loader) == ("icustays", "icustay_id")

    sample_calls = []

    def sample_fast(_loader, table, id_col, **kwargs):
        sample_calls.append((table, id_col, kwargs))
        return [9, 3]

    monkeypatch.setattr(concept_api, "_query_patient_ids_fast", sample_fast)
    assert api._sample_patient_ids(loader, 2, sample_strategy="sorted") == [9, 3]
    assert sample_calls[0][:2] == ("icustays", "icustay_id")

    monkeypatch.setattr(
        concept_api,
        "_query_patient_ids_fast",
        lambda _loader, table, id_col, **_kwargs: (
            [5, 6] if (table, id_col) == ("icustays", "icustay_id") else []
        ),
    )
    assert list(api._iter_patient_id_batches(loader, 2, total_patients=2)) == [
        {"icustay_id": [5, 6]}
    ]

    count_calls = []

    def count_fast(_loader, table, id_col):
        count_calls.append((table, id_col))
        return 123

    monkeypatch.setattr(concept_api, "_count_patient_ids_fast", count_fast)
    assert api._get_total_patient_count(loader) == 123
    assert count_calls == [("icustays", "icustay_id")]

    (tmp_path / "icustays.parquet").touch()
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: pd.DataFrame({"icustay_id": [7, 8]}),
    )
    assert api.get_all_patient_ids(tmp_path, database="mimic_demo") == (
        [7, 8],
        "icustay_id",
    )


def test_load_concepts_explicit_batch_size_uses_mimic_demo_profile(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.runtime import memory_manager

    loader = SimpleNamespace(database="mimic_demo", data_path=tmp_path)
    captured = {}
    monkeypatch.setattr(concept_api, "_get_global_loader", lambda **_kwargs: loader)
    monkeypatch.setattr(concept_api, "_get_total_patient_count", lambda _loader: 3)
    monkeypatch.setattr(
        concept_api,
        "_sample_patient_ids",
        lambda *_args, **_kwargs: [101, 102, 103],
    )
    monkeypatch.setenv("EASYICU_FORCE_INPROCESS_BATCH", "1")

    def fake_batch_load(*, patient_ids, **_kwargs):
        captured["patient_ids"] = patient_ids
        return pd.DataFrame({"icustay_id": [101, 102, 103], "hr": [70, 71, 72]})

    monkeypatch.setattr(memory_manager, "inprocess_batch_load", fake_batch_load)
    result = api.load_concepts(
        "hr",
        database="mimic_demo",
        batch_size=2,
        concept_workers=1,
        parallel_workers=1,
    )
    assert captured["patient_ids"] == {"icustay_id": [101, 102, 103]}
    assert result["icustay_id"].tolist() == [101, 102, 103]


@pytest.mark.parametrize("sampled_patient_ids", [None, []])
def test_load_concepts_required_bounded_sample_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sampled_patient_ids,
) -> None:
    loader = SimpleNamespace(database="mimic_demo", data_path=tmp_path)
    monkeypatch.setattr(concept_api, "_get_global_loader", lambda **_kwargs: loader)
    monkeypatch.setattr(
        concept_api,
        "_sample_patient_ids",
        lambda *_args, **_kwargs: sampled_patient_ids,
    )

    with pytest.raises(RuntimeError, match="refusing to fall back"):
        api.load_concepts(
            "hr",
            database="mimic_demo",
            max_patients=2,
            require_bounded_sample=True,
        )


def test_load_concepts_bounded_flag_preserves_legacy_positional_tail() -> None:
    parameters = list(inspect.signature(api.load_concepts).parameters)

    # New flags may only be APPENDED here — every existing name must keep its
    # index so legacy positional callers stay correct.
    assert parameters[parameters.index("max_patients") : parameters.index("kwargs")] == [
        "max_patients",
        "limit",
        "sample_strategy",
        "batch_size",
        "memory_efficient",
        "require_bounded_sample",
        "allow_unbounded_fallback",
    ]


def test_load_concepts_positional_database_guard_resolves_alias_to_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    loader = SimpleNamespace(database="miiv")

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return loader

    monkeypatch.setattr(concept_api, "_get_global_loader", fake_loader)
    result = api.load_concepts("hr", "mimic-iv", max_patients=0)
    assert result.empty
    assert captured["database"] == "miiv"


def test_data_paths_and_crossdb_views_share_profile_alias_label_and_order(
    tmp_path,
) -> None:
    version = tmp_path / "mimic-iv" / "3.1"
    version.mkdir(parents=True)

    assert DATABASE_ALIASES["miiv"][:2] == ("mimiciv", "mimic-iv")
    assert find_database_path(str(tmp_path), "MIMIC-IV") == str(version)
    # The canonical MIMIC-III key must not become a broad ``mimic`` substring
    # alias that accidentally selects a MIMIC-IV directory.
    assert find_database_path(str(tmp_path), "mimic") == str(tmp_path)
    mimic_iii_version = tmp_path / "mimiciii" / "1.4"
    mimic_iii_version.mkdir(parents=True)
    assert find_database_path(str(tmp_path), "mimic") == str(mimic_iii_version)
    assert public_database_keys() == EXPECTED_PUBLIC
    assert crossdb_review._DEMO_MULTIDB_DATABASES == EXPECTED_PUBLIC
    assert list(crossdb_review._RAW_DB_LABELS) == list(EXPECTED_PUBLIC)
    assert crossdb_review._RAW_DB_LABELS["aumc"] == "AmsterdamUMCdb"
    assert crossdb_review._normalize_database_key("AmsterdamUMCdb") == "aumc"
    assert crossdb_review._raw_database_aliases_payload()["miiv"]["aliases"][:2] == [
        "mimiciv",
        "mimic-iv",
    ]
